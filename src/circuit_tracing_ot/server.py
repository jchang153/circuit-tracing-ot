"""Local graph visualization server wrapper."""

from __future__ import annotations

from pathlib import Path

from circuit_tracer.frontend.local_server import serve


def serve_graph_files(*, graph_file_dir: str | Path, port: int = 8046, features_dir: str | None = None):
    """Serve exported circuit-tracer graph files in the interactive HTML UI."""
    kwargs = {"data_dir": str(graph_file_dir), "port": int(port)}
    if features_dir:
        kwargs["features_dir"] = str(features_dir)
    return serve(**kwargs)
