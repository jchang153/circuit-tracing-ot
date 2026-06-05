"""Model loading helpers following the upstream circuit-tracer demos."""

from __future__ import annotations

import torch

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


def check_cuda_usable() -> None:
    """Fail early when a visible GPU cannot be used by the installed PyTorch build."""
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    if cuda_device_count > 0 and not cuda_available:
        raise RuntimeError(
            "A CUDA GPU is visible, but PyTorch cannot initialize CUDA. This usually means the "
            "installed torch wheel is incompatible with the cluster NVIDIA driver. On Delta, "
            "reinstall PyTorch inside the virtualenv with:\n\n"
            "  pip uninstall -y torch torchvision torchaudio\n"
            "  pip install --index-url https://download.pytorch.org/whl/cu126 "
            "torch torchvision torchaudio\n\n"
            "Then verify with:\n\n"
            "  python - <<'PY'\n"
            "  import torch\n"
            "  print(torch.__version__)\n"
            "  print(torch.cuda.is_available())\n"
            "  print(torch.cuda.get_device_name(0))\n"
            "  PY\n"
        )


def _iter_module_parameters(value, *, max_depth: int = 3, _seen: set[int] | None = None):
    """Yield parameters from common wrapped-model attributes without depending on one backend."""
    if _seen is None:
        _seen = set()
    if max_depth < 0 or id(value) in _seen:
        return
    _seen.add(id(value))
    if hasattr(value, "parameters"):
        try:
            yield from value.parameters()
            return
        except TypeError:
            pass
    for attr in ("model", "base_model", "hf_model", "wrapped_model", "tl_model"):
        if hasattr(value, attr):
            yield from _iter_module_parameters(
                getattr(value, attr),
                max_depth=max_depth - 1,
                _seen=_seen,
            )


def infer_model_device(model) -> torch.device | None:
    """Infer the device of a ReplacementModel-like object when possible."""
    for parameter in _iter_module_parameters(model):
        return parameter.device
    for attr in ("device", "model_device"):
        if hasattr(model, attr):
            try:
                return torch.device(getattr(model, attr))
            except (TypeError, RuntimeError):
                pass
    return None


def ensure_model_on_cuda(model):
    """Move a model wrapper to CUDA when supported and verify device placement."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch in this environment.")
    initial_device = infer_model_device(model)
    if initial_device is not None and initial_device.type == "cuda":
        return model
    if hasattr(model, "to"):
        moved = model.to("cuda")
        if moved is not None:
            model = moved
    resolved_device = infer_model_device(model)
    if resolved_device is None:
        print(
            "[model] warning: could not infer ReplacementModel parameter device; "
            "continuing because PyTorch CUDA is available."
        )
        return model
    if resolved_device.type != "cuda":
        raise RuntimeError(
            f"ReplacementModel appears to be on {resolved_device}, not CUDA. "
            "Check the circuit-tracer backend/offload settings and PyTorch CUDA wheel."
        )
    return model


def load_replacement_model(
    *,
    model_name: str = MODEL_NAME,
    transcoder_set: str,
    dtype_name: str = "bf16",
    offload: str | None = None,
    backend: str | None = None,
    require_cuda: bool = True,
):
    """Load Gemma-2-2B plus CLTs as a circuit-tracer ReplacementModel."""
    if backend == "cuda":
        raise ValueError(
            "--backend cuda is not valid for circuit-tracer. The backend must be 'nnsight' or "
            "'transformerlens'; CUDA device use is controlled by the installed PyTorch build. "
            "Omit --backend for the default behavior."
        )
    check_cuda_usable()
    from circuit_tracer import ReplacementModel

    kwargs: dict[str, object] = {
        "dtype": parse_dtype(dtype_name),
    }
    if offload:
        kwargs["offload"] = offload
    if backend:
        kwargs["backend"] = backend
    model = ReplacementModel.from_pretrained(model_name, transcoder_set, **kwargs)
    if require_cuda:
        model = ensure_model_on_cuda(model)
    resolved_device = infer_model_device(model)
    if resolved_device is not None:
        print(f"[model] ReplacementModel device={resolved_device}")
    print(f"[model] torch.cuda.is_available()={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[model] CUDA device={torch.cuda.get_device_name(0)}")
    return model
