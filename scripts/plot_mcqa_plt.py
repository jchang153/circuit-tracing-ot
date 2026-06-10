#!/usr/bin/env python
"""MCQA PLOT protocol with Gemma-2-2B per-layer transcoder interventions."""

from __future__ import annotations

import sys

from circuit_tracing_ot.config import GEMMA_2_2B_PLT_TRANSCODER_SET
from plot_mcqa_clt import main as run_clt_plot


def _option_value(args: list[str], option: str) -> str | None:
    for index, arg in enumerate(args):
        if arg == option and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(f"{option}="):
            return arg.split("=", maxsplit=1)[1]
    return None


def _append_default(args: list[str], option: str, value: str) -> None:
    if _option_value(args, option) is None:
        args.extend([option, value])


def main() -> None:
    args = sys.argv[1:]
    write_layer_mode = _option_value(args, "--clt-write-layer-mode")
    if write_layer_mode is not None and write_layer_mode != "same":
        raise ValueError(
            "PLT experiments must use same-layer writes. "
            "Remove --clt-write-layer-mode or set it to 'same'."
        )
    transcoder_set = _option_value(args, "--transcoder-set")
    if transcoder_set is not None and transcoder_set != GEMMA_2_2B_PLT_TRANSCODER_SET:
        raise ValueError(
            "scripts/plot_mcqa_plt.py is pinned to the Gemma-2-2B PLT transcoders "
            f"({GEMMA_2_2B_PLT_TRANSCODER_SET}). Use scripts/plot_mcqa_clt.py for custom sets."
        )

    _append_default(args, "--transcoder-set", GEMMA_2_2B_PLT_TRANSCODER_SET)
    _append_default(args, "--clt-intervention-mode", "decoded_mlp")
    _append_default(args, "--clt-write-layer-mode", "same")
    _append_default(args, "--results-timestamp", "mcqa_plt")

    sys.argv = [sys.argv[0], *args]
    run_clt_plot()


if __name__ == "__main__":
    main()
