"""Small progress logging helper for long cluster runs."""

from __future__ import annotations

from datetime import datetime


def log_progress(message: str) -> None:
    """Print a timestamped progress message immediately."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)
