"""Attribution graph creation and visualization export."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DESIRED_LOGIT_PROB,
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_GRAPH_DIR,
    DEFAULT_GRAPH_FILE_DIR,
    DEFAULT_MAX_FEATURE_NODES,
    DEFAULT_MAX_N_LOGITS,
    DEFAULT_NODE_THRESHOLD,
    DEFAULT_RESULT_DIR,
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

    output_path = str(graph_file_dir)
    if not output_path.endswith(os.sep):
        output_path += os.sep
    create_graph_files(
        graph_or_path=graph_path,
        slug=slug,
        output_path=output_path,
        node_threshold=float(node_threshold),
        edge_threshold=float(edge_threshold),
    )
    enrich_graph_metadata(graph_path=graph_path, slug=slug, graph_file_dir=graph_file_dir)


def _raw_graph_stats(graph_path: Path) -> dict[str, object]:
    import torch

    raw_graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    stats: dict[str, object] = {}
    if not isinstance(raw_graph, dict):
        return {"type": type(raw_graph).__name__}

    adjacency_matrix = raw_graph.get("adjacency_matrix")
    selected_features = raw_graph.get("selected_features")
    active_features = raw_graph.get("active_features")
    input_tokens = raw_graph.get("input_tokens")
    logit_targets = raw_graph.get("logit_targets")

    if hasattr(adjacency_matrix, "shape"):
        shape = tuple(int(dim) for dim in adjacency_matrix.shape)
        stats["adjacency_matrix_shape"] = list(shape)
        if len(shape) == 2 and shape[0] == shape[1]:
            stats["total_raw_nodes"] = shape[0]
    if hasattr(selected_features, "shape"):
        stats["selected_feature_nodes"] = int(selected_features.shape[0])
    if hasattr(active_features, "shape"):
        stats["active_features"] = int(active_features.shape[0])
    if hasattr(input_tokens, "shape"):
        stats["input_token_nodes"] = int(input_tokens.shape[0])
    if isinstance(logit_targets, list):
        stats["logit_target_nodes"] = len(logit_targets)
    if "total_raw_nodes" in stats and "selected_feature_nodes" in stats:
        stats["non_feature_nodes"] = int(stats["total_raw_nodes"]) - int(
            stats["selected_feature_nodes"]
        )
    return stats


def _sort_count_items(counter: Counter[str]) -> dict[str, int]:
    def sort_key(item: tuple[str, int]) -> tuple[int, int | str]:
        key = item[0]
        return (0, int(key)) if key.isdigit() else (1, key)

    return dict(sorted(counter.items(), key=sort_key))


def _viewer_graph_stats(graph_data: dict[str, object]) -> dict[str, object]:
    nodes = graph_data.get("nodes", [])
    links = graph_data.get("links", [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(links, list):
        links = []

    node_ids_by_type: dict[str, list[str]] = defaultdict(list)
    layers_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    ctx_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    target_logit_ids: list[str] = []

    for node in nodes:
        if not isinstance(node, dict):
            continue
        feature_type = str(node.get("feature_type", "unknown"))
        node_id = str(node.get("node_id", ""))
        if node_id:
            node_ids_by_type[feature_type].append(node_id)
        if node.get("layer") is not None:
            layers_by_type[feature_type][str(node.get("layer"))] += 1
        if node.get("ctx_idx") is not None:
            ctx_by_type[feature_type][str(node.get("ctx_idx"))] += 1
        if node.get("is_target_logit"):
            target_logit_ids.append(node_id)

    feature_type_counts = {
        feature_type: len(node_ids) for feature_type, node_ids in sorted(node_ids_by_type.items())
    }
    return {
        "total_viewer_nodes": len(nodes),
        "total_viewer_links": len(links),
        "feature_type_counts": feature_type_counts,
        "layer_counts_by_feature_type": {
            feature_type: _sort_count_items(counter)
            for feature_type, counter in sorted(layers_by_type.items())
        },
        "context_counts_by_feature_type": {
            feature_type: _sort_count_items(counter)
            for feature_type, counter in sorted(ctx_by_type.items())
        },
        "node_ids_by_feature_type": {
            feature_type: sorted(node_ids)
            for feature_type, node_ids in sorted(node_ids_by_type.items())
        },
        "target_logit_node_ids": sorted(target_logit_ids),
    }


def enrich_graph_metadata(*, graph_path: Path, slug: str, graph_file_dir: Path) -> None:
    """Add node and edge statistics to graph-metadata.json after viewer export."""
    graph_json_path = graph_file_dir / f"{slug}.json"
    metadata_path = graph_file_dir / "graph-metadata.json"
    if not graph_json_path.exists() or not metadata_path.exists():
        return

    graph_data = json.loads(graph_json_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stats = {
        "viewer_graph": _viewer_graph_stats(graph_data),
        "raw_graph": _raw_graph_stats(graph_path),
    }

    graphs = metadata.get("graphs", [])
    if not isinstance(graphs, list):
        return
    for graph in graphs:
        if isinstance(graph, dict) and graph.get("slug") == slug:
            graph["statistics"] = stats
            break
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class ResultPaths:
    """Output locations for a flat result directory."""

    graph_dir: Path
    graph_file_dir: Path
    run_dir: Path


def result_paths(result_dir: Path = DEFAULT_RESULT_DIR) -> ResultPaths:
    """Return the standard flat result layout."""
    return ResultPaths(
        graph_dir=result_dir,
        graph_file_dir=result_dir,
        run_dir=result_dir,
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
    max_feature_nodes: int | None = DEFAULT_MAX_FEATURE_NODES
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
    manifest_path = run_dir / f"{prompt.slug}.manifest.json"
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
