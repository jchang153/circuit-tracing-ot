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


def check_cuda_usable(*, backend: str | None) -> None:
    """Fail early when a visible GPU cannot be used by the installed PyTorch build."""
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    should_use_cuda = backend == "cuda" or (backend is None and cuda_device_count > 0)
    if should_use_cuda and not cuda_available:
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


def load_replacement_model(
    *,
    model_name: str = MODEL_NAME,
    transcoder_set: str,
    dtype_name: str = "bf16",
    offload: str | None = None,
    backend: str | None = None,
):
    """Load Gemma-2-2B plus CLTs as a circuit-tracer ReplacementModel."""
    check_cuda_usable(backend=backend)
    from circuit_tracer import ReplacementModel

    kwargs: dict[str, object] = {
        "dtype": parse_dtype(dtype_name),
    }
    if offload:
        kwargs["offload"] = offload
    if backend:
        kwargs["backend"] = backend
    return ReplacementModel.from_pretrained(model_name, transcoder_set, **kwargs)
