"""CLT activation helpers for progressive PLOT experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .interventions import FeatureIntervention


@dataclass(frozen=True)
class CLTFeatureValue:
    """Activation value for one CLT feature at one token position."""

    layer: int
    position: int
    feature_idx: int
    value: float

    @property
    def key(self) -> tuple[int, int, int]:
        return (int(self.layer), int(self.position), int(self.feature_idx))

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def _last_position_from_prompt(model: Any, prompt: str) -> int:
    encoded = model.tokenizer.encode(prompt, add_special_tokens=True)
    return len(encoded) - 1


def normalize_position(model: Any, prompt: str, position: int) -> int:
    """Resolve negative token positions against the tokenized prompt length."""
    position = int(position)
    if position >= 0:
        return position
    return _last_position_from_prompt(model, prompt) + 1 + position


def _iter_feature_values_from_mapping(data: Mapping[Any, Any]) -> list[CLTFeatureValue]:
    values: list[CLTFeatureValue] = []
    for raw_key, raw_value in data.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 3:
            continue
        layer, position, feature_idx = raw_key
        if isinstance(raw_value, torch.Tensor):
            if raw_value.numel() != 1:
                continue
            raw_value = raw_value.detach().cpu().item()
        values.append(
            CLTFeatureValue(
                layer=int(layer),
                position=int(position),
                feature_idx=int(feature_idx),
                value=float(raw_value),
            )
        )
    return values


def _iter_feature_values_from_tensor(
    tensor: torch.Tensor,
    *,
    layer: int | None,
    position: int,
    top_k: int | None,
) -> list[CLTFeatureValue]:
    if layer is None:
        raise ValueError("Tensor activation extraction requires a layer id.")
    if tensor.ndim == 3:
        vector = tensor[0, position]
    elif tensor.ndim == 2:
        vector = tensor[position]
    elif tensor.ndim == 1:
        vector = tensor
    else:
        raise ValueError(f"Unsupported activation tensor shape: {tuple(tensor.shape)}")
    vector = vector.detach().float().cpu()
    if top_k is not None:
        count = min(int(top_k), int(vector.numel()))
        activation_values, feature_indices = vector.topk(count)
    else:
        feature_indices = torch.arange(vector.numel())
        activation_values = vector
    return [
        CLTFeatureValue(
            layer=int(layer),
            position=int(position),
            feature_idx=int(feature_idx),
            value=float(value),
        )
        for feature_idx, value in zip(
            feature_indices.tolist(),
            activation_values.tolist(),
            strict=True,
        )
    ]


def extract_clt_feature_values(
    activation_payload: Any,
    *,
    layer: int | None = None,
    position: int,
    top_k: int | None = None,
) -> list[CLTFeatureValue]:
    """Extract feature values from common circuit-tracer activation payload shapes.

    This intentionally accepts several shapes because `ReplacementModel.feature_intervention`
    versions differ: some return a mapping keyed by ``(layer, position, feature)``, while others
    expose per-layer tensors through a cache-like object.
    """
    if isinstance(activation_payload, Mapping):
        values = _iter_feature_values_from_mapping(activation_payload)
        if values:
            filtered = [value for value in values if value.position == position]
            if layer is not None:
                filtered = [value for value in filtered if value.layer == int(layer)]
            filtered.sort(key=lambda value: abs(value.value), reverse=True)
            return filtered[:top_k] if top_k is not None else filtered
        for key in ("clt_features", "features", "feature_activations", "activations"):
            if key in activation_payload:
                return extract_clt_feature_values(
                    activation_payload[key],
                    layer=layer,
                    position=position,
                    top_k=top_k,
                )
    if isinstance(activation_payload, torch.Tensor):
        return _iter_feature_values_from_tensor(
            activation_payload,
            layer=layer,
            position=position,
            top_k=top_k,
        )
    for attr in ("clt_features", "features", "feature_activations", "activations"):
        if hasattr(activation_payload, attr):
            return extract_clt_feature_values(
                getattr(activation_payload, attr),
                layer=layer,
                position=position,
                top_k=top_k,
            )
    raise TypeError(
        "Could not extract CLT feature activations from circuit-tracer payload. "
        "Expected a mapping keyed by (layer, position, feature), a tensor, or an object with "
        "clt_features/features/feature_activations/activations."
    )


def get_prompt_feature_values(
    *,
    model: Any,
    prompt: str,
    layer: int | None = None,
    position: int = -1,
    top_k: int | None = None,
) -> list[CLTFeatureValue]:
    """Run a no-op feature intervention and return active CLT feature values."""
    resolved_position = normalize_position(model, prompt, position)
    with torch.inference_mode():
        _, payload = model.feature_intervention(prompt, [])
    return extract_clt_feature_values(
        payload,
        layer=layer,
        position=resolved_position,
        top_k=top_k,
    )


def paired_feature_interventions(
    *,
    factual_values: list[CLTFeatureValue],
    counterfactual_values: list[CLTFeatureValue],
    layer: int | None = None,
    feature_idx: int | None = None,
) -> list[FeatureIntervention]:
    """Create interventions that set factual features to counterfactual activations."""
    factual_by_feature = {
        (value.layer, value.feature_idx): value
        for value in factual_values
    }
    interventions: list[FeatureIntervention] = []
    for value in counterfactual_values:
        if layer is not None and value.layer != int(layer):
            continue
        if feature_idx is not None and value.feature_idx != int(feature_idx):
            continue
        factual_value = factual_by_feature.get((value.layer, value.feature_idx))
        if factual_value is None:
            continue
        interventions.append(
            FeatureIntervention(
                layer=value.layer,
                position=factual_value.position,
                feature_idx=value.feature_idx,
                value=value.value,
            )
        )
    return interventions
