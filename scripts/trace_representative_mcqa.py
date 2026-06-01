#!/usr/bin/env python
"""Trace representative MCQA prompts with Gemma-2-2B CLTs."""

from __future__ import annotations

import argparse
from pathlib import Path

from circuit_tracing_ot.config import resolve_transcoder_set
from circuit_tracing_ot.logging import log_progress
from circuit_tracing_ot.mcqa_prompts import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    get_prompt,
    load_prompts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-id", default=None, help="Dataset row id or formatted prompt id.")
    parser.add_argument("--all", action="store_true", help="Trace all loaded prompts.")
    parser.add_argument("--limit", type=int, default=None, help="Limit loaded dataset rows.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--model-name", default="google/gemma-2-2b")
    parser.add_argument("--transcoder-size", default="426k", choices=("426k", "2.5m"))
    parser.add_argument("--transcoder-set", default=None)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--offload", default=None, choices=(None, "cpu", "disk"))
    parser.add_argument("--backend", default=None, choices=("nnsight", "transformerlens"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-n-logits", type=int, default=10)
    parser.add_argument("--desired-logit-prob", type=float, default=0.95)
    parser.add_argument("--max-feature-nodes", type=int, default=None)
    parser.add_argument("--node-threshold", type=float, default=0.8)
    parser.add_argument("--edge-threshold", type=float, default=0.98)
    parser.add_argument("--graph-dir", type=Path, default=Path("graphs"))
    parser.add_argument("--graph-file-dir", type=Path, default=Path("graph_files"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from circuit_tracing_ot.model import load_replacement_model
    from circuit_tracing_ot.trace import TraceConfig, trace_prompt

    transcoder_set = resolve_transcoder_set(args.transcoder_set, args.transcoder_size)
    log_progress(
        f"loading prompts from {args.dataset_name} "
        f"config={args.dataset_config or 'default'} split={args.dataset_split}"
    )
    loaded_prompts = load_prompts(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        limit=args.limit,
    )
    if args.all:
        prompts = loaded_prompts
    elif args.prompt_id:
        prompts = [get_prompt(args.prompt_id, loaded_prompts)]
    else:
        prompts = [loaded_prompts[0]]
    log_progress(
        f"selected {len(prompts)} prompt(s): "
        + ", ".join(prompt.prompt_id for prompt in prompts[:5])
        + (", ..." if len(prompts) > 5 else "")
    )

    log_progress(
        f"loading model {args.model_name} with transcoder_set={transcoder_set} dtype={args.dtype}"
    )
    model = load_replacement_model(
        model_name=args.model_name,
        transcoder_set=transcoder_set,
        dtype_name=args.dtype,
        offload=args.offload,
        backend=args.backend,
    )
    log_progress("model and transcoders loaded")
    config = TraceConfig(
        model_name=args.model_name,
        transcoder_set=transcoder_set,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_n_logits=args.max_n_logits,
        desired_logit_prob=args.desired_logit_prob,
        max_feature_nodes=args.max_feature_nodes,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
    )
    for index, prompt in enumerate(prompts, start=1):
        log_progress(f"starting prompt {index}/{len(prompts)}: {prompt.prompt_id}")
        result = trace_prompt(
            model=model,
            prompt=prompt,
            config=config,
            graph_dir=args.graph_dir,
            graph_file_dir=args.graph_file_dir,
            run_dir=args.run_dir,
        )
        print(
            f"traced {result.prompt_id}: graph={result.graph_path} "
            f"graph_files={result.graph_file_dir} elapsed={result.elapsed_seconds:.1f}s"
        )


if __name__ == "__main__":
    main()
