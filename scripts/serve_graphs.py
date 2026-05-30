#!/usr/bin/env python
"""Serve exported circuit-tracer graph files."""

from __future__ import annotations

import argparse
from pathlib import Path

from circuit_tracing_ot.server import serve_graph_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-file-dir", type=Path, default=Path("graph_files"))
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--port", type=int, default=8046)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = serve_graph_files(
        graph_file_dir=args.graph_file_dir,
        port=args.port,
        features_dir=args.features_dir,
    )
    print(f"serving {args.graph_file_dir} at http://localhost:{args.port}/index.html")
    try:
        input("Press Enter to stop the server...\n")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
