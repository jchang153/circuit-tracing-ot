#!/usr/bin/env python
"""Run feature interventions on representative MCQA prompts."""

from __future__ import annotations

import argparse
import json

from circuit_tracing_ot.config import resolve_transcoder_set
from circuit_tracing_ot.interventions import parse_feature_intervention, run_feature_intervention
from circuit_tracing_ot.mcqa_prompts import get_prompt
from circuit_tracing_ot.model import load_replacement_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-id", default="copycolors-a")
    parser.add_argument("--model-name", default="google/gemma-2-2b")
    parser.add_argument("--transcoder-size", default="426k", choices=("426k", "2.5m"))
    parser.add_argument("--transcoder-set", default=None)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--offload", default=None, choices=(None, "cpu", "disk"))
    parser.add_argument("--backend", default=None)
    parser.add_argument(
        "--feature",
        action="append",
        required=True,
        help="Feature intervention as layer:position:feature_idx:value. Can be repeated.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = get_prompt(args.prompt_id)
    transcoder_set = resolve_transcoder_set(args.transcoder_set, args.transcoder_size)
    model = load_replacement_model(
        model_name=args.model_name,
        transcoder_set=transcoder_set,
        dtype_name=args.dtype,
        offload=args.offload,
        backend=args.backend,
    )
    interventions = [parse_feature_intervention(value) for value in args.feature]
    result = run_feature_intervention(
        model=model,
        prompt=prompt.prompt,
        interventions=interventions,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
