"""Decoded CLT feature-delta interventions for MCQA PLOT.

This backend treats CLTs as a source of feature directions only. For a source
feature ``(k, j)`` and write layer ``l``, it adds
``(a_j^k(source) - a_j^k(base)) d_j^{k->l}`` to the base MLP output at layer
``l`` and then runs the model forward normally.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from time import perf_counter
from typing import Any

import torch

from ..logging import log_progress
from .clt_backend import (
    CLTActivationCache,
    CLTSite,
    _bank_cache_key,
    _last_token_logits,
    _should_log_progress,
    _site_feature_ids,
    _site_weights_cache_key,
)
from .data import MCQAPairBank


def _hook_name_for_layer(model: Any, layer: int) -> str:
    feature_output_hook = getattr(model, "feature_output_hook", None)
    if not feature_output_hook:
        original = getattr(model, "original_feature_output_hook", None)
        if not original:
            raise RuntimeError(
                "Decoded CLT interventions require a ReplacementModel-like object with "
                "`feature_output_hook` or `original_feature_output_hook`."
            )
        feature_output_hook = f"{original}.hook_out_grad"
    return f"blocks.{int(layer)}.{feature_output_hook}"


def _model_num_layers(model: Any) -> int:
    cfg = getattr(model, "cfg", None)
    if cfg is not None and getattr(cfg, "n_layers", None) is not None:
        return int(cfg.n_layers)
    if hasattr(model, "blocks"):
        return len(model.blocks)
    raise RuntimeError("Could not infer model layer count for decoded CLT intervention.")


def _decoder_vector(
    *,
    model: Any,
    source_layer: int,
    write_layer: int,
    feature_idx: int,
) -> torch.Tensor:
    """Return the decoder vector d_j^{source_layer -> write_layer}."""
    if int(write_layer) < int(source_layer):
        raise ValueError(
            "CLT decoded interventions can only write to the source layer or a later layer: "
            f"source_layer={source_layer}, write_layer={write_layer}"
        )
    transcoders = getattr(model, "transcoders", None)
    if transcoders is None:
        raise RuntimeError("Decoded CLT interventions require `model.transcoders`.")

    feature_ids = torch.tensor([int(feature_idx)], dtype=torch.long)
    if hasattr(transcoders, "_get_decoder_vectors"):
        decoder = transcoders._get_decoder_vectors(int(source_layer), feature_ids)
        if decoder.ndim == 2:
            if int(write_layer) != int(source_layer):
                raise ValueError(
                    "Single-layer decoder vectors can only be written to their own layer: "
                    f"source_layer={source_layer}, write_layer={write_layer}"
                )
            return decoder[0].detach()
        if decoder.ndim == 3:
            output_offset = int(write_layer) - int(source_layer)
            if output_offset >= decoder.shape[1]:
                raise ValueError(
                    "Requested write layer is outside this CLT decoder's output range: "
                    f"source_layer={source_layer}, write_layer={write_layer}, "
                    f"decoder_remaining_layers={decoder.shape[1]}"
                )
            return decoder[0, output_offset].detach()
        raise ValueError(f"Unexpected decoder shape {tuple(decoder.shape)}")

    try:
        transcoder = transcoders[int(source_layer)]
    except (TypeError, IndexError, KeyError) as exc:
        raise RuntimeError(
            "Could not index `model.transcoders`; decoded interventions need either a "
            "CrossLayerTranscoder with `_get_decoder_vectors` or per-layer transcoders."
        ) from exc
    if hasattr(transcoder, "_get_decoder_vectors"):
        decoder = transcoder._get_decoder_vectors(int(source_layer), feature_ids)
        if decoder.ndim == 2:
            return decoder[0].detach()
        output_offset = int(write_layer) - int(source_layer)
        return decoder[0, output_offset].detach()
    if hasattr(transcoder, "W_dec"):
        if int(write_layer) != int(source_layer):
            raise ValueError(
                "Per-layer W_dec vectors can only be written to their own layer: "
                f"source_layer={source_layer}, write_layer={write_layer}"
            )
        return transcoder.W_dec[int(feature_idx)].detach()
    raise RuntimeError("Could not locate decoder vectors on the installed transcoder object.")


def _add_deltas_hook(
    activations: torch.Tensor,
    _hook,
    *,
    deltas_by_position: dict[int, torch.Tensor],
) -> torch.Tensor:
    if not deltas_by_position:
        return activations
    updated = activations.clone()
    for position, delta in deltas_by_position.items():
        if activations.ndim == 3:
            updated[0, int(position)] = updated[0, int(position)] + delta.to(
                device=activations.device,
                dtype=activations.dtype,
            )
        elif activations.ndim == 2:
            updated[int(position)] = updated[int(position)] + delta.to(
                device=activations.device,
                dtype=activations.dtype,
            )
        else:
            raise ValueError(f"Unexpected activation shape for MLP output hook: {tuple(activations.shape)}")
    return updated


def _run_with_hooks(model: Any, prompt: str, hooks: list[tuple[str, Callable]]) -> torch.Tensor:
    if hasattr(model, "run_with_hooks"):
        return model.run_with_hooks(prompt, fwd_hooks=hooks)
    if hasattr(model, "hooks"):
        with model.hooks(hooks):
            return model(prompt)
    raise RuntimeError("Decoded CLT interventions require `run_with_hooks` or `hooks` on the model.")


def run_clt_decoded_site_intervention(
    *,
    model,
    bank: MCQAPairBank,
    site_weights: dict[CLTSite, float],
    strength: float,
    cache: CLTActivationCache,
    log_context: str | None = None,
) -> torch.Tensor:
    """Run decoded feature-delta interventions and return last-token logits."""
    cache_key = (
        "run_clt_decoded_site_intervention",
        _bank_cache_key(bank),
        _site_weights_cache_key(site_weights),
        float(strength),
    )
    cached_logits = cache.get_intervention_logits(cache_key)
    if cached_logits is not None:
        context = str(log_context) if log_context is not None else f"target_var={bank.target_var}"
        log_progress(
            "decoded CLT intervention cache hit "
            f"split={bank.split} {context} examples={bank.size} sites={len(site_weights)} "
            f"strength={float(strength):g}"
        )
        return cached_logits

    outputs = []
    start = perf_counter()
    context = str(log_context) if log_context is not None else f"target_var={bank.target_var}"
    decoder_cache: dict[tuple[int, int, int], torch.Tensor] = {}
    log_progress(
        "decoded CLT intervention start "
        f"split={bank.split} {context} examples={bank.size} sites={len(site_weights)} "
        f"strength={float(strength):g} first_sites={[site.label for site in site_weights][:3]}"
    )
    with torch.inference_mode():
        for row_index, base_input in enumerate(bank.base_inputs):
            if _should_log_progress(row_index, bank.size):
                log_progress(
                    "decoded CLT intervention "
                    f"row={row_index + 1}/{bank.size} {context} "
                    f"sites={len(site_weights)} elapsed={perf_counter() - start:.1f}s"
                )
            base_prompt = str(base_input["raw_input"])
            source_prompt = str(bank.source_inputs[row_index]["raw_input"])
            deltas_by_write_site: dict[tuple[int, int], torch.Tensor] = {}
            for site, weight in site_weights.items():
                source_layer = int(site.layer)
                write_layer = int(site.resolved_write_layer)
                if write_layer >= _model_num_layers(model):
                    raise ValueError(f"write_layer={write_layer} is outside the model layer range.")
                base_position = int(bank.base_position_by_id[site.token_position_id][row_index].item())
                source_position = int(bank.source_position_by_id[site.token_position_id][row_index].item())
                read_top_k = None if site.top_features is None else int(site.top_features)
                source_values = cache.value_map(
                    prompt=source_prompt,
                    layer=source_layer,
                    position=source_position,
                    top_k=read_top_k,
                )
                base_values = cache.value_map(
                    prompt=base_prompt,
                    layer=source_layer,
                    position=base_position,
                    top_k=read_top_k,
                )
                if site.feature_idx is None and site.top_features is None:
                    feature_ids = sorted(set(source_values) | set(base_values))
                else:
                    feature_ids = _site_feature_ids(site=site, source_values=source_values)
                for feature_idx in feature_ids:
                    activation_delta = float(source_values.get(int(feature_idx), 0.0)) - float(
                        base_values.get(int(feature_idx), 0.0)
                    )
                    scale = float(strength) * float(weight) * activation_delta
                    if scale == 0.0:
                        continue
                    decoder_key = (source_layer, write_layer, int(feature_idx))
                    if decoder_key not in decoder_cache:
                        decoder_cache[decoder_key] = _decoder_vector(
                            model=model,
                            source_layer=source_layer,
                            write_layer=write_layer,
                            feature_idx=int(feature_idx),
                        )
                    delta = decoder_cache[decoder_key] * scale
                    write_key = (write_layer, base_position)
                    if write_key in deltas_by_write_site:
                        deltas_by_write_site[write_key] = deltas_by_write_site[write_key] + delta
                    else:
                        deltas_by_write_site[write_key] = delta

            hooks = [
                (
                    _hook_name_for_layer(model, write_layer),
                    partial(_add_deltas_hook, deltas_by_position=deltas_by_position),
                )
                for write_layer, deltas_by_position in _group_deltas_by_layer(
                    deltas_by_write_site
                ).items()
            ]
            logits = _run_with_hooks(model, base_prompt, hooks) if hooks else model(base_prompt)
            outputs.append(_last_token_logits(logits))

    logits = torch.stack(outputs, dim=0)
    cache.set_intervention_logits(cache_key, logits)
    log_progress(
        "decoded CLT intervention complete "
        f"{context} examples={len(outputs)} sites={len(site_weights)} "
        f"elapsed={perf_counter() - start:.1f}s"
    )
    return logits


def _group_deltas_by_layer(
    deltas_by_write_site: dict[tuple[int, int], torch.Tensor],
) -> dict[int, dict[int, torch.Tensor]]:
    grouped: dict[int, dict[int, torch.Tensor]] = {}
    for (write_layer, position), delta in deltas_by_write_site.items():
        grouped.setdefault(int(write_layer), {})[int(position)] = delta
    return grouped
