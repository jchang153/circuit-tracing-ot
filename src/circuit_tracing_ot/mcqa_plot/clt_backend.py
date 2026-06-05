"""CLT feature sites and intervention backend for the MCQA PLOT protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..clt_features import CLTFeatureValue, extract_clt_feature_values
from ..interventions import FeatureIntervention
from .data import MCQAPairBank
from .metrics import signature_from_logits


@dataclass(frozen=True)
class CLTSite:
    """One candidate CLT site.

    ``feature_idx=None`` means a layer-level site: copy the top active CLT features in the source
    prompt for that layer. A concrete ``feature_idx`` means copy only that feature.
    """

    layer: int
    token_position_id: str
    feature_idx: int | None = None
    top_features: int = 256

    @property
    def label(self) -> str:
        if self.feature_idx is None:
            return f"CLT-L{int(self.layer)}:{self.token_position_id}:top{int(self.top_features)}"
        return f"CLT-L{int(self.layer)}:{self.token_position_id}:f{int(self.feature_idx)}"

    @property
    def dim_start(self) -> int:
        return 0 if self.feature_idx is None else int(self.feature_idx)

    @property
    def dim_end(self) -> int:
        return int(self.top_features) if self.feature_idx is None else int(self.feature_idx) + 1


class CLTActivationCache:
    """Memoized no-op CLT activations keyed by prompt text, layer, and position."""

    def __init__(self, model: Any):
        self.model = model
        self._payload_by_prompt: dict[str, Any] = {}
        self._values_by_key: dict[tuple[str, int, int, int | None], list[CLTFeatureValue]] = {}

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


def enumerate_clt_layer_sites(
    *,
    num_layers: int,
    token_position_id: str,
    layers: tuple[int, ...] | None,
    top_features: int,
) -> list[CLTSite]:
    layer_ids = tuple(range(int(num_layers))) if layers is None else tuple(int(layer) for layer in layers)
    return [
        CLTSite(
            layer=int(layer),
            token_position_id=str(token_position_id),
            feature_idx=None,
            top_features=int(top_features),
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
    for row_index, source_input in enumerate(bank.source_inputs):
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
    with torch.inference_mode():
        for base_input in bank.base_inputs:
            logits, _ = model.feature_intervention(str(base_input["raw_input"]), [])
            outputs.append(_last_token_logits(logits))
    return torch.stack(outputs, dim=0)


def _site_feature_ids(
    *,
    site: CLTSite,
    source_values: dict[int, float],
) -> list[int]:
    if site.feature_idx is not None:
        return [int(site.feature_idx)]
    ranked = sorted(source_values.items(), key=lambda item: (-abs(float(item[1])), int(item[0])))
    return [int(feature_idx) for feature_idx, _value in ranked[: int(site.top_features)]]


def run_clt_site_intervention(
    *,
    model,
    bank: MCQAPairBank,
    site_weights: dict[CLTSite, float],
    strength: float,
    cache: CLTActivationCache,
) -> torch.Tensor:
    """Run CLT feature-value copying interventions and return last-token logits."""
    outputs = []
    with torch.inference_mode():
        for row_index, base_input in enumerate(bank.base_inputs):
            base_prompt = str(base_input["raw_input"])
            source_prompt = str(bank.source_inputs[row_index]["raw_input"])
            interventions: dict[tuple[int, int, int], float] = {}
            for site, weight in site_weights.items():
                base_position = int(bank.base_position_by_id[site.token_position_id][row_index].item())
                source_position = int(bank.source_position_by_id[site.token_position_id][row_index].item())
                read_top_k = int(site.top_features)
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
                for feature_idx in _site_feature_ids(site=site, source_values=source_values):
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
    return torch.stack(outputs, dim=0)


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
    for site in sites:
        site_logits = run_clt_site_intervention(
            model=model,
            bank=bank,
            site_weights={site: 1.0},
            strength=1.0,
            cache=cache,
        )
        signatures.append(
            signature_from_logits(
                counterfactual_logits=site_logits,
                base_logits=base_logits,
                bank=bank,
                signature_mode=signature_mode,
            )
        )
    return torch.stack(signatures, dim=0)
