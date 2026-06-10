#!/usr/bin/env python
"""Inspect circuit-tracer feature_intervention payload structure."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from circuit_tracing_ot.clt_features import extract_clt_feature_values
from circuit_tracing_ot.config import MODEL_NAME, resolve_transcoder_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="A is red. B is blue. C is green. D is yellow. red?")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--transcoder-size", default="426k", choices=("426k", "2.5m"))
    parser.add_argument("--transcoder-set", default=None)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--offload", default=None, choices=(None, "cpu", "disk"))
    parser.add_argument("--backend", default=None, choices=("nnsight", "transformerlens"))
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--layers", default="0,1,2,10,20,25")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    return parser


def _parse_layers(text: str) -> list[int]:
    layers = []
    for raw_part in str(text).split(","):
        part = raw_part.strip()
        if not part:
            continue
        layers.append(int(part))
    return layers


def _tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "type": "Tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
    }


def _summarize_payload(
    payload: Any,
    *,
    depth: int,
    max_depth: int,
    max_items: int,
) -> dict[str, object]:
    if isinstance(payload, torch.Tensor):
        return _tensor_summary(payload)
    if depth >= max_depth:
        return {"type": type(payload).__name__, "truncated": True}
    if isinstance(payload, Mapping):
        items = list(payload.items())
        return {
            "type": type(payload).__name__,
            "len": len(items),
            "items": [
                {
                    "key": repr(key),
                    "value": _summarize_payload(
                        value,
                        depth=depth + 1,
                        max_depth=max_depth,
                        max_items=max_items,
                    ),
                }
                for key, value in items[:max_items]
            ],
        }
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return {
            "type": type(payload).__name__,
            "len": len(payload),
            "items": [
                _summarize_payload(
                    value,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                )
                for value in list(payload)[:max_items]
            ],
        }
    attrs = {}
    for attr in ("clt_features", "features", "feature_activations", "activations"):
        if hasattr(payload, attr):
            attrs[attr] = _summarize_payload(
                getattr(payload, attr),
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
            )
    if attrs:
        return {"type": type(payload).__name__, "attrs": attrs}
    return {"type": type(payload).__name__, "repr": repr(payload)[:500]}


def _resolve_position(model: Any, prompt: str, position: int) -> int:
    if int(position) >= 0:
        return int(position)
    encoded = model.tokenizer.encode(prompt, add_special_tokens=True)
    return len(encoded) + int(position)


def main() -> None:
    args = build_parser().parse_args()
    from circuit_tracing_ot.model import load_replacement_model

    transcoder_set = resolve_transcoder_set(args.transcoder_set, args.transcoder_size)
    model = load_replacement_model(
        model_name=args.model_name,
        transcoder_set=transcoder_set,
        dtype_name=args.dtype,
        offload=args.offload,
        backend=args.backend,
    )
    with torch.inference_mode():
        logits, payload = model.feature_intervention(str(args.prompt), [])

    position = _resolve_position(model, str(args.prompt), int(args.position))
    layer_values = {}
    for layer in _parse_layers(args.layers):
        try:
            values = extract_clt_feature_values(
                payload,
                layer=layer,
                position=position,
                top_k=int(args.top_k),
            )
            layer_values[str(layer)] = [
                {
                    "layer": int(value.layer),
                    "position": int(value.position),
                    "feature_idx": int(value.feature_idx),
                    "value": float(value.value),
                }
                for value in values
            ]
        except Exception as exc:
            layer_values[str(layer)] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    print(
        json.dumps(
            {
                "prompt": str(args.prompt),
                "position": position,
                "logits_shape": list(logits.shape),
                "payload_summary": _summarize_payload(
                    payload,
                    depth=0,
                    max_depth=int(args.max_depth),
                    max_items=int(args.max_items),
                ),
                "top_values_by_layer": layer_values,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
