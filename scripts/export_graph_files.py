#!/usr/bin/env python
"""Export/prune saved circuit-tracer attribution graphs for the HTML viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

from circuit_tracing_ot.config import (
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_GRAPH_FILE_DIR,
    DEFAULT_NODE_THRESHOLD,
)
from circuit_tracing_ot.logging import log_progress
from circuit_tracing_ot.trace import export_graph_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_path", type=Path)
    parser.add_argument("--slug", default=None)
    parser.add_argument("--graph-file-dir", type=Path, default=DEFAULT_GRAPH_FILE_DIR)
    parser.add_argument("--node-threshold", type=float, default=DEFAULT_NODE_THRESHOLD)
    parser.add_argument("--edge-threshold", type=float, default=DEFAULT_EDGE_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slug = args.slug or args.graph_path.stem
    args.graph_file_dir.mkdir(parents=True, exist_ok=True)
    log_progress(f"exporting/pruning {args.graph_path} as slug={slug}")
    export_graph_files(
        graph_path=args.graph_path,
        slug=slug,
        graph_file_dir=args.graph_file_dir,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
    )
    log_progress(f"exported graph files to {args.graph_file_dir}")


if __name__ == "__main__":
    main()
