"""Model loading helpers following the upstream circuit-tracer demos."""

from __future__ import annotations

import torch

from circuit_tracer import ReplacementModel

from .config import MODEL_NAME


def parse_dtype(dtype_name: str):
    normalized = str(dtype_name).strip().lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {dtype_name!r}; use bf16, fp16, or fp32.")


def load_replacement_model(
    *,
    model_name: str = MODEL_NAME,
    transcoder_set: str,
    dtype_name: str = "bf16",
    offload: str | None = None,
    backend: str | None = None,
):
    """Load Gemma-2-2B plus CLTs as a circuit-tracer ReplacementModel."""
    kwargs: dict[str, object] = {
        "dtype": parse_dtype(dtype_name),
    }
    if offload:
        kwargs["offload"] = offload
    if backend:
        kwargs["backend"] = backend
    return ReplacementModel.from_pretrained(model_name, transcoder_set, **kwargs)
