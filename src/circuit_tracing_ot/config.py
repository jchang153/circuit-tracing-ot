"""Shared defaults for Gemma-2-2B circuit tracing."""

from __future__ import annotations

from pathlib import Path

MODEL_NAME = "google/gemma-2-2b"
GEMMA_2_2B_PLT_TRANSCODER_SET = "mntss/gemma-scope-transcoders"
SMALL_CLT_TRANSCODER_SET = "mntss/clt-gemma-2-2b-426k"
LARGE_CLT_TRANSCODER_SET = "mntss/clt-gemma-2-2b-2.5M"

DEFAULT_GRAPH_DIR = Path("graphs")
DEFAULT_GRAPH_FILE_DIR = Path("graph_files")
DEFAULT_RUN_DIR = Path("runs")
DEFAULT_RESULT_DIR = Path("results/delta_default_train_0")

DEFAULT_MAX_FEATURE_NODES = 1000
DEFAULT_NODE_THRESHOLD = 1.0
DEFAULT_EDGE_THRESHOLD = 1.0
DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_N_LOGITS = 10
DEFAULT_DESIRED_LOGIT_PROB = 0.95


def resolve_transcoder_set(transcoder_set: str | None, transcoder_size: str | None) -> str:
    """Resolve a named CLT size or explicit Hugging Face transcoder repo."""
    if transcoder_set:
        return str(transcoder_set)
    normalized_size = (transcoder_size or "426k").strip().lower()
    if normalized_size in {"426k", "small", "default"}:
        return SMALL_CLT_TRANSCODER_SET
    if normalized_size in {"2.5m", "2500k", "large"}:
        return LARGE_CLT_TRANSCODER_SET
    raise ValueError(
        f"Unsupported transcoder size {transcoder_size!r}; use 426k, 2.5m, or --transcoder-set."
    )
