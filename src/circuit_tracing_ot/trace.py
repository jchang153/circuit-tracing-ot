"""Attribution graph creation and visualization export."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DESIRED_LOGIT_PROB,
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_GRAPH_DIR,
    DEFAULT_GRAPH_FILE_DIR,
    DEFAULT_MAX_N_LOGITS,
    DEFAULT_NODE_THRESHOLD,
    DEFAULT_RUN_DIR,
    MODEL_NAME,
)
from .logging import log_progress
from .mcqa_prompts import MCQAPrompt


def export_graph_files(
    *,
    graph_path: Path,
    slug: str,
    graph_file_dir: Path,
    node_threshold: float,
    edge_threshold: float,
) -> None:
    """Export graph files across circuit-tracer versions with different output_path semantics."""
    from circuit_tracer.utils import create_graph_files

    try:
        create_graph_files(
            graph_or_path=graph_path,
            slug=slug,
            output_path=graph_file_dir,
            node_threshold=float(node_threshold),
            edge_threshold=float(edge_threshold),
        )
    except AssertionError as exc:
        if "Could not find" not in str(exc):
            raise
        log_progress(
            "retrying graph export with JSON output path for this circuit-tracer version"
        )
        create_graph_files(
            graph_or_path=graph_path,
            slug=slug,
            output_path=graph_file_dir / f"{slug}.json",
            node_threshold=float(node_threshold),
            edge_threshold=float(edge_threshold),
        )


@dataclass(frozen=True)
class TraceConfig:
    """Parameters for one circuit-tracer attribution run."""

    model_name: str = MODEL_NAME
    transcoder_set: str = ""
    dtype: str = "bf16"
    batch_size: int = DEFAULT_BATCH_SIZE
    max_n_logits: int = DEFAULT_MAX_N_LOGITS
    desired_logit_prob: float = DEFAULT_DESIRED_LOGIT_PROB
    max_feature_nodes: int | None = None
    node_threshold: float = DEFAULT_NODE_THRESHOLD
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD


@dataclass(frozen=True)
class TraceResult:
    """Paths and metadata from one traced MCQA prompt."""

    prompt_id: str
    slug: str
    graph_path: str
    graph_file_dir: str
    elapsed_seconds: float


def trace_prompt(
    *,
    model,
    prompt: MCQAPrompt,
    config: TraceConfig,
    graph_dir: Path = DEFAULT_GRAPH_DIR,
    graph_file_dir: Path = DEFAULT_GRAPH_FILE_DIR,
    run_dir: Path = DEFAULT_RUN_DIR,
) -> TraceResult:
    """Run attribution, save the raw graph, and export pruned visualization files."""
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_file_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    graph_path = graph_dir / f"{prompt.slug}.pt"
    start = perf_counter()
    log_progress(f"running attribution for {prompt.prompt_id}")
    from circuit_tracer import attribute

    graph = attribute(
        prompt.prompt,
        model,
        batch_size=int(config.batch_size),
        max_n_logits=int(config.max_n_logits),
        desired_logit_prob=float(config.desired_logit_prob),
        max_feature_nodes=config.max_feature_nodes,
    )
    log_progress(f"saving raw graph to {graph_path}")
    graph.to_pt(graph_path)

    log_progress(
        f"exporting/pruning graph files to {graph_file_dir} "
        f"(node_threshold={config.node_threshold}, edge_threshold={config.edge_threshold})"
    )
    export_graph_files(
        graph_path=graph_path,
        slug=prompt.slug,
        graph_file_dir=graph_file_dir,
        node_threshold=config.node_threshold,
        edge_threshold=config.edge_threshold,
    )
    elapsed = perf_counter() - start

    result = TraceResult(
        prompt_id=prompt.prompt_id,
        slug=prompt.slug,
        graph_path=str(graph_path),
        graph_file_dir=str(graph_file_dir),
        elapsed_seconds=float(elapsed),
    )
    manifest_path = run_dir / f"{prompt.slug}.json"
    log_progress(f"writing run manifest to {manifest_path}")
    manifest_path.write_text(
        json.dumps(
            {
                "prompt": asdict(prompt),
                "config": asdict(config),
                "result": asdict(result),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result
