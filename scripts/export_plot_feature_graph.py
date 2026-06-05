#!/usr/bin/env python
"""Export a circuit-tracer viewer graph centered on PLOT-selected CLT features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-results", type=Path, required=True)
    parser.add_argument("--graph-json", type=Path, required=True)
    parser.add_argument("--graph-metadata", type=Path, default=None)
    parser.add_argument("--slug", default="plot-selected-features")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-non-feature-nodes",
        action="store_true",
        help=(
            "Keep embeddings, logits, and reconstruction-error nodes connected to "
            "selected features."
        ),
    )
    return parser.parse_args()


def _selected_feature_keys(plot_results: dict[str, object]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    selected_items = list(plot_results.get("selected_features", []))
    stage_b = plot_results.get("stage_b", {})
    if isinstance(stage_b, dict):
        selected_items.extend(stage_b.get("selected_features", []))
    for item in selected_items:
        if not isinstance(item, dict):
            continue
        direct_layer = item.get("layer")
        direct_feature_idx = item.get("feature_idx")
        if direct_layer is not None and direct_feature_idx is not None:
            keys.add((str(direct_layer), int(direct_feature_idx)))
        for ranking_item in item.get("ranking", []):
            if not isinstance(ranking_item, dict):
                continue
            layer = ranking_item.get("layer", direct_layer)
            feature_idx = ranking_item.get("feature_idx")
            if layer is not None and feature_idx is not None:
                keys.add((str(layer), int(feature_idx)))
    return keys


def _node_feature_key(node: dict[str, object]) -> tuple[str, int] | None:
    if node.get("feature_type") != "cross layer transcoder":
        return None
    if node.get("layer") is None or node.get("feature") is None:
        return None
    return (str(node["layer"]), int(node["feature"]))


def export_plot_graph(
    *,
    plot_results_path: Path,
    graph_json_path: Path,
    graph_metadata_path: Path | None,
    slug: str,
    output_dir: Path,
    include_non_feature_nodes: bool,
) -> None:
    plot_results = json.loads(plot_results_path.read_text(encoding="utf-8"))
    graph = json.loads(graph_json_path.read_text(encoding="utf-8"))
    selected_keys = _selected_feature_keys(plot_results)
    if not selected_keys:
        raise RuntimeError(f"No selected CLT features found in {plot_results_path}")

    nodes = graph.get("nodes", [])
    links = graph.get("links", [])
    selected_node_ids = {
        str(node.get("node_id"))
        for node in nodes
        if isinstance(node, dict) and _node_feature_key(node) in selected_keys
    }
    if not selected_node_ids:
        raise RuntimeError(
            "None of the PLOT-selected features were present in the graph JSON. "
            "Rerun tracing with a larger --max-feature-nodes value, lower thresholds, "
            "or a representative prompt where these features activate."
        )
    if include_non_feature_nodes:
        connected_ids = set(selected_node_ids)
        for link in links:
            if not isinstance(link, dict):
                continue
            source = str(link.get("source"))
            target = str(link.get("target"))
            if source in selected_node_ids or target in selected_node_ids:
                connected_ids.add(source)
                connected_ids.add(target)
        keep_ids = connected_ids
    else:
        keep_ids = selected_node_ids

    exported = dict(graph)
    exported["nodes"] = [
        node for node in nodes if isinstance(node, dict) and str(node.get("node_id")) in keep_ids
    ]
    exported["links"] = [
        link
        for link in links
        if isinstance(link, dict)
        and str(link.get("source")) in keep_ids
        and str(link.get("target")) in keep_ids
    ]
    exported["metadata"] = dict(exported.get("metadata", {}))
    exported["metadata"]["plot_selected_feature_count"] = len(selected_keys)
    exported["metadata"]["plot_results"] = str(plot_results_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{slug}.json").write_text(
        json.dumps(exported, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metadata = {"graphs": []}
    if graph_metadata_path and graph_metadata_path.exists():
        metadata = json.loads(graph_metadata_path.read_text(encoding="utf-8"))
    graph_entry = {
        "slug": slug,
        "node_threshold": 0.0,
        "edge_threshold": 0.0,
        "schema_version": 1,
        "plot_selected_features": [
            {"layer": layer, "feature_idx": feature_idx}
            for layer, feature_idx in sorted(
                selected_keys,
                key=lambda item: (int(item[0]), item[1]),
            )
        ],
        "statistics": {
            "viewer_graph": {
                "total_viewer_nodes": len(exported["nodes"]),
                "total_viewer_links": len(exported["links"]),
            }
        },
    }
    metadata["graphs"] = [graph_entry]
    (output_dir / "graph-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    export_plot_graph(
        plot_results_path=args.plot_results,
        graph_json_path=args.graph_json,
        graph_metadata_path=args.graph_metadata,
        slug=args.slug,
        output_dir=args.output_dir,
        include_non_feature_nodes=args.include_non_feature_nodes,
    )


if __name__ == "__main__":
    main()
