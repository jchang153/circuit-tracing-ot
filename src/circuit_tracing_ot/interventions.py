"""Feature intervention helpers mirroring circuit-tracer's intervention demo."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FeatureIntervention:
    """One circuit-tracer feature intervention tuple."""

    layer: int
    position: int
    feature_idx: int
    value: float

    def as_tuple(self) -> tuple[int, int, int, float]:
        return (int(self.layer), int(self.position), int(self.feature_idx), float(self.value))


def parse_feature_intervention(text: str) -> FeatureIntervention:
    """Parse layer:position:feature_idx:value from the CLI."""
    parts = str(text).split(":")
    if len(parts) != 4:
        raise ValueError("Feature interventions must look like layer:position:feature_idx:value")
    return FeatureIntervention(
        layer=int(parts[0]),
        position=int(parts[1]),
        feature_idx=int(parts[2]),
        value=float(parts[3]),
    )


def top_token_predictions(model, logits: torch.Tensor, *, k: int = 5) -> list[dict[str, float | str]]:
    """Return top-k next-token predictions from circuit-tracer logits."""
    next_token_logits = logits.squeeze(0)[-1]
    probs, token_ids = next_token_logits.softmax(dim=-1).topk(int(k))
    return [
        {
            "token": model.tokenizer.decode([int(token_id)]),
            "probability": float(prob),
            "token_id": int(token_id),
        }
        for prob, token_id in zip(probs.detach().cpu(), token_ids.detach().cpu(), strict=True)
    ]


def run_feature_intervention(
    *,
    model,
    prompt: str,
    interventions: list[FeatureIntervention],
    top_k: int = 5,
) -> dict[str, object]:
    """Run a no-op baseline and a feature intervention, returning top-token summaries."""
    intervention_tuples = [intervention.as_tuple() for intervention in interventions]
    with torch.inference_mode():
        original_logits, _ = model.feature_intervention(prompt, [])
        new_logits, _ = model.feature_intervention(prompt, intervention_tuples)
    return {
        "prompt": prompt,
        "interventions": [intervention.as_tuple() for intervention in interventions],
        "original_top_tokens": top_token_predictions(model, original_logits, k=top_k),
        "intervened_top_tokens": top_token_predictions(model, new_logits, k=top_k),
    }
