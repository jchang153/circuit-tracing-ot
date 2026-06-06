"""CLT feature sites and intervention backend for the MCQA PLOT protocol."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch

from ..clt_features import CLTFeatureValue, extract_clt_feature_values
from ..interventions import FeatureIntervention
from ..logging import log_progress
from .data import MCQAPairBank
from .metrics import signature_from_logits


def _should_log_progress(index: int, total: int, *, interval: int = 10) -> bool:
    return index == 0 or index == total - 1 or (index + 1) % max(1, int(interval)) == 0


@dataclass(frozen=True)
class CLTSite:
    """One candidate CLT site.

    ``feature_idx=None`` means a layer-level site: copy the full extracted CLT feature layer from
    source to base. A concrete ``feature_idx`` means copy only that feature.
    """

    layer: int
    token_position_id: str
    feature_idx: int | None = None
    top_features: int | None = None

    @property
    def label(self) -> str:
        if self.feature_idx is None:
            if self.top_features is None:
                return f"CLT-L{int(self.layer)}:{self.token_position_id}:all"
            return f"CLT-L{int(self.layer)}:{self.token_position_id}:top{int(self.top_features)}"
        return f"CLT-L{int(self.layer)}:{self.token_position_id}:f{int(self.feature_idx)}"

    @property
    def dim_start(self) -> int:
        return 0 if self.feature_idx is None else int(self.feature_idx)

    @property
    def dim_end(self) -> int:
        if self.feature_idx is not None:
            return int(self.feature_idx) + 1
        return -1 if self.top_features is None else int(self.top_features)


class CLTActivationCache:
    """Memoized no-op CLT activations keyed by prompt text, layer, and position."""

    def __init__(self, model: Any):
        self.model = model
        self._payload_by_prompt: dict[str, Any] = {}
        self._values_by_key: dict[tuple[str, int, int, int | None], list[CLTFeatureValue]] = {}
        self._intervention_logits_by_key: dict[tuple[object, ...], torch.Tensor] = {}

    def payload(self, prompt: str) -> Any:
        if prompt not in self._payload_by_prompt:
            with torch.inference_mode():
                _, payload = self.model.feature_intervention(prompt, [])
            self._payload_by_prompt[prompt] = payload
        return self._payload_by_prompt[prompt]

    def values(
        self,
        *,
        prompt: str,
        layer: int,
        position: int,
        top_k: int | None,
    ) -> list[CLTFeatureValue]:
        key = (prompt, int(layer), int(position), None if top_k is None else int(top_k))
        if key not in self._values_by_key:
            self._values_by_key[key] = extract_clt_feature_values(
                self.payload(prompt),
                layer=int(layer),
                position=int(position),
                top_k=top_k,
            )
        return self._values_by_key[key]

    def value_map(
        self,
        *,
        prompt: str,
        layer: int,
        position: int,
        top_k: int | None,
    ) -> dict[int, float]:
        return {
            int(value.feature_idx): float(value.value)
            for value in self.values(prompt=prompt, layer=layer, position=position, top_k=top_k)
        }

    def get_intervention_logits(self, key: tuple[object, ...]) -> torch.Tensor | None:
        return self._intervention_logits_by_key.get(key)

    def set_intervention_logits(self, key: tuple[object, ...], logits: torch.Tensor) -> None:
        self._intervention_logits_by_key[key] = logits


def _bank_cache_key(bank: MCQAPairBank) -> tuple[object, ...]:
    return (
        id(bank),
        bank.split,
        bank.target_var,
        bank.size,
        tuple(bank.dataset_names),
    )


def _site_weights_cache_key(site_weights: dict[CLTSite, float]) -> tuple[tuple[str, float], ...]:
    return tuple(
        sorted(
            ((site.label, float(weight)) for site, weight in site_weights.items()),
            key=lambda item: item[0],
        )
    )


def enumerate_clt_layer_sites(
    *,
    num_layers: int,
    token_position_id: str,
    layers: tuple[int, ...] | None,
    top_features: int | None,
) -> list[CLTSite]:
    layer_ids = tuple(range(int(num_layers))) if layers is None else tuple(int(layer) for layer in layers)
    return [
        CLTSite(
            layer=int(layer),
            token_position_id=str(token_position_id),
            feature_idx=None,
            top_features=None if top_features is None else int(top_features),
        )
        for layer in layer_ids
    ]


def enumerate_top_clt_feature_sites(
    *,
    model,
    bank: MCQAPairBank,
    layers: tuple[int, ...],
    token_position_id: str,
    top_features_per_layer: int,
    activation_read_top_k: int,
    cache: CLTActivationCache | None = None,
) -> list[CLTSite]:
    """Rank feature candidates by mean absolute source activation on D_train."""
    cache = cache or CLTActivationCache(model)
    scores: dict[tuple[int, int], float] = {}
    start = perf_counter()
    log_progress(
        "Stage B feature candidate scan start "
        f"target_var={bank.target_var} split={bank.split} examples={bank.size} "
        f"layers={list(int(layer) for layer in layers)} read_top_k={int(activation_read_top_k)}"
    )
    for row_index, source_input in enumerate(bank.source_inputs):
        if _should_log_progress(row_index, bank.size):
            log_progress(
                "Stage B feature candidate scan "
                f"row={row_index + 1}/{bank.size} elapsed={perf_counter() - start:.1f}s"
            )
        prompt = str(source_input["raw_input"])
        position = int(bank.source_position_by_id[token_position_id][row_index].item())
        for layer in layers:
            for value in cache.values(
                prompt=prompt,
                layer=int(layer),
                position=position,
                top_k=int(activation_read_top_k),
            ):
                key = (int(layer), int(value.feature_idx))
                scores[key] = scores.get(key, 0.0) + abs(float(value.value))
    log_progress(
        "Stage B feature candidate scan complete "
        f"unique_features={len(scores)} elapsed={perf_counter() - start:.1f}s"
    )
    sites: list[CLTSite] = []
    for layer in layers:
        layer_items = [
            (feature_idx, score)
            for (item_layer, feature_idx), score in scores.items()
            if int(item_layer) == int(layer)
        ]
        layer_items.sort(key=lambda item: (-float(item[1]), int(item[0])))
        for feature_idx, _score in layer_items[: int(top_features_per_layer)]:
            sites.append(
                CLTSite(
                    layer=int(layer),
                    token_position_id=str(token_position_id),
                    feature_idx=int(feature_idx),
                    top_features=int(activation_read_top_k),
                )
            )
    return sites


def _last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3:
        return logits.squeeze(0)[-1].detach().cpu()
    if logits.ndim == 2:
        return logits[-1].detach().cpu()
    raise ValueError(f"Unexpected logits shape from feature_intervention: {tuple(logits.shape)}")


def collect_base_logits_clt(*, model, bank: MCQAPairBank) -> torch.Tensor:
    outputs = []
    start = perf_counter()
    log_progress(f"collecting base logits split={bank.split} target_var={bank.target_var} examples={bank.size}")
    with torch.inference_mode():
        for row_index, base_input in enumerate(bank.base_inputs):
            if _should_log_progress(row_index, bank.size):
                log_progress(
                    "collecting base logits "
                    f"row={row_index + 1}/{bank.size} elapsed={perf_counter() - start:.1f}s"
                )
            logits, _ = model.feature_intervention(str(base_input["raw_input"]), [])
            outputs.append(_last_token_logits(logits))
    log_progress(f"collected base logits examples={len(outputs)} elapsed={perf_counter() - start:.1f}s")
    return torch.stack(outputs, dim=0)


def _site_feature_ids(
    *,
    site: CLTSite,
    source_values: dict[int, float],
) -> list[int]:
    if site.feature_idx is not None:
        return [int(site.feature_idx)]
    ranked = sorted(source_values.items(), key=lambda item: (-abs(float(item[1])), int(item[0])))
    if site.top_features is None:
        return [int(feature_idx) for feature_idx, _value in ranked]
    return [int(feature_idx) for feature_idx, _value in ranked[: int(site.top_features)]]


def run_clt_site_intervention(
    *,
    model,
    bank: MCQAPairBank,
    site_weights: dict[CLTSite, float],
    strength: float,
    cache: CLTActivationCache,
    log_context: str | None = None,
) -> torch.Tensor:
    """Run CLT feature-value copying interventions and return last-token logits."""
    cache_key = (
        "run_clt_site_intervention",
        _bank_cache_key(bank),
        _site_weights_cache_key(site_weights),
        float(strength),
    )
    cached_logits = cache.get_intervention_logits(cache_key)
    if cached_logits is not None:
        context = str(log_context) if log_context is not None else f"target_var={bank.target_var}"
        log_progress(
            "CLT intervention cache hit "
            f"split={bank.split} {context} examples={bank.size} sites={len(site_weights)} "
            f"strength={float(strength):g}"
        )
        return cached_logits
    outputs = []
    start = perf_counter()
    site_labels = [site.label for site in site_weights]
    context = str(log_context) if log_context is not None else f"target_var={bank.target_var}"
    log_progress(
        "CLT intervention start "
        f"split={bank.split} {context} examples={bank.size} "
        f"sites={len(site_weights)} strength={float(strength):g} "
        f"first_sites={site_labels[:3]}"
    )
    with torch.inference_mode():
        for row_index, base_input in enumerate(bank.base_inputs):
            if _should_log_progress(row_index, bank.size):
                log_progress(
                    "CLT intervention "
                    f"row={row_index + 1}/{bank.size} {context} "
                    f"sites={len(site_weights)} elapsed={perf_counter() - start:.1f}s"
                )
            base_prompt = str(base_input["raw_input"])
            source_prompt = str(bank.source_inputs[row_index]["raw_input"])
            interventions: dict[tuple[int, int, int], float] = {}
            for site, weight in site_weights.items():
                base_position = int(bank.base_position_by_id[site.token_position_id][row_index].item())
                source_position = int(bank.source_position_by_id[site.token_position_id][row_index].item())
                read_top_k = None if site.top_features is None else int(site.top_features)
                source_values = cache.value_map(
                    prompt=source_prompt,
                    layer=int(site.layer),
                    position=source_position,
                    top_k=read_top_k,
                )
                base_values = cache.value_map(
                    prompt=base_prompt,
                    layer=int(site.layer),
                    position=base_position,
                    top_k=read_top_k,
                )
                if site.feature_idx is None and site.top_features is None:
                    feature_ids = sorted(set(source_values) | set(base_values))
                else:
                    feature_ids = _site_feature_ids(site=site, source_values=source_values)
                for feature_idx in feature_ids:
                    base_value = float(base_values.get(int(feature_idx), 0.0))
                    source_value = float(source_values.get(int(feature_idx), 0.0))
                    new_value = base_value + float(strength) * float(weight) * (source_value - base_value)
                    interventions[(int(site.layer), base_position, int(feature_idx))] = float(new_value)
            intervention_tuples = [
                FeatureIntervention(
                    layer=layer,
                    position=position,
                    feature_idx=feature_idx,
                    value=value,
                ).as_tuple()
                for (layer, position, feature_idx), value in sorted(interventions.items())
            ]
            logits, _ = model.feature_intervention(base_prompt, intervention_tuples)
            outputs.append(_last_token_logits(logits))
    log_progress(
        "CLT intervention complete "
        f"{context} examples={len(outputs)} sites={len(site_weights)} "
        f"elapsed={perf_counter() - start:.1f}s"
    )
    logits = torch.stack(outputs, dim=0)
    cache.set_intervention_logits(cache_key, logits)
    return logits


def collect_clt_site_signatures(
    *,
    model,
    bank: MCQAPairBank,
    sites: list[CLTSite],
    base_logits: torch.Tensor,
    signature_mode: str,
    cache: CLTActivationCache,
) -> torch.Tensor:
    signatures = []
    start = perf_counter()
    log_progress(
        "collecting CLT site signatures start "
        f"split={bank.split} reference_bank={bank.target_var} sites={len(sites)} examples={bank.size} "
        f"signature_mode={signature_mode}"
    )
    for site_index, site in enumerate(sites):
        if _should_log_progress(site_index, len(sites), interval=1 if len(sites) <= 10 else 5):
            log_progress(
                "collecting CLT site signature "
                f"site={site_index + 1}/{len(sites)} label={site.label} "
                f"elapsed={perf_counter() - start:.1f}s"
            )
        site_logits = run_clt_site_intervention(
            model=model,
            bank=bank,
            site_weights={site: 1.0},
            strength=1.0,
            cache=cache,
            log_context="site_signature",
        )
        signatures.append(
            signature_from_logits(
                counterfactual_logits=site_logits,
                base_logits=base_logits,
                bank=bank,
                signature_mode=signature_mode,
            )
        )
    log_progress(
        "collecting CLT site signatures complete "
        f"sites={len(signatures)} elapsed={perf_counter() - start:.1f}s"
    )
    return torch.stack(signatures, dim=0)
