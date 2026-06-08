#!/usr/bin/env python
"""MCQA PLOT over last-token CLT features from a pruned attribution graph."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

import torch

from circuit_tracing_ot.config import MODEL_NAME, resolve_transcoder_set
from circuit_tracing_ot.logging import log_progress
from circuit_tracing_ot.mcqa_plot.clt_backend import CLTActivationCache
from circuit_tracing_ot.mcqa_plot.data import (
    COUNTERFACTUAL_FAMILIES,
    MCQACausalModel,
    build_pair_banks,
    canonicalize_target_var,
    get_token_positions,
    load_public_mcqa_datasets,
)
from circuit_tracing_ot.mcqa_plot.ot import OTConfig, prepare_alignment_artifacts_clt
from circuit_tracing_ot.mcqa_plot.pruned_graph_sites import load_pruned_last_token_clt_sites
from plot_mcqa_clt import (
    DEFAULT_CALIBRATION_FAMILY_WEIGHTS,
    DEFAULT_COUNTERFACTUAL_NAMES,
    DEFAULT_OT_EPSILONS,
    DEFAULT_SIGNATURE_MODE,
    DEFAULT_TARGET_VARS,
    DEFAULT_UOT_BETA_NEURALS,
    evaluate_stage_a_holdout,
    filter_correct_examples_with_hf_model,
    load_filter_model_and_tokenizer,
    load_stage_a_config_checkpoint,
    parse_csv_floats,
    parse_csv_strings,
    run_stage_a_config,
    save_stage_a_config_checkpoint,
    stage_a_config_key,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-json", type=Path, required=True)
    parser.add_argument("--candidate-top-k", type=int, default=300)
    parser.add_argument("--candidate-rank-by", default="influence", choices=("influence", "activation"))
    parser.add_argument("--dataset-path", default="jchang153/copycolors_mcqa")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-size", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-pool-size", type=int, default=200)
    parser.add_argument("--calibration-pool-size", type=int, default=100)
    parser.add_argument("--test-pool-size", type=int, default=100)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--transcoder-size", default="426k", choices=("426k", "2.5m"))
    parser.add_argument("--transcoder-set", default=None)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--offload", default=None, choices=(None, "cpu", "disk"))
    parser.add_argument("--backend", default=None, choices=("nnsight", "transformerlens"))
    parser.add_argument("--target-vars", default=",".join(DEFAULT_TARGET_VARS))
    parser.add_argument("--counterfactual-names", default=",".join(DEFAULT_COUNTERFACTUAL_NAMES))
    parser.add_argument("--signature-mode", default=DEFAULT_SIGNATURE_MODE)
    parser.add_argument("--stage-a-transport-methods", default="uot")
    parser.add_argument("--ot-epsilons", default=",".join(str(value) for value in DEFAULT_OT_EPSILONS))
    parser.add_argument(
        "--uot-beta-neurals",
        default=",".join(str(value) for value in DEFAULT_UOT_BETA_NEURALS),
    )
    parser.add_argument("--stage-a-row-top-k", type=int, default=20)
    parser.add_argument(
        "--calibration-family-weights",
        default=",".join(str(value) for value in DEFAULT_CALIBRATION_FAMILY_WEIGHTS),
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--results-timestamp")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    parser.add_argument("--filter-batch-size", type=int, default=32)
    parser.add_argument("--skip-stage-a-holdout", action="store_true")
    return parser


def build_stage_a_site_rankings(
    *,
    stage_a_payloads: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    rankings_by_var: dict[str, list[dict[str, object]]] = {}
    for payload in stage_a_payloads:
        method = str(payload.get("method"))
        epsilon = float(payload.get("epsilon", 0.0))
        beta_neural = payload.get("uot_beta_neural")
        per_var_records = payload.get("per_var_records", {})
        if not isinstance(per_var_records, dict):
            continue
        for target_var, record in per_var_records.items():
            if not isinstance(record, dict):
                continue
            rankings_by_var.setdefault(str(target_var), []).append(
                {
                    "target_var": str(target_var),
                    "layer": int(record["layer"]),
                    "feature_idx": record.get("feature_idx"),
                    "site_index": int(record["site_index"]),
                    "site_label": str(record["site_label"]),
                    "method": method,
                    "epsilon": epsilon,
                    "uot_beta_neural": None if beta_neural is None else float(beta_neural),
                    "selection_score": float(record["selection_score"]),
                    "calibration_exact_acc": float(record["calibration_exact_acc"]),
                    "selection_exact_acc": float(record["selection_exact_acc"]),
                    "candidate_records": record.get("candidate_records", []),
                    "target_row_ranking": record.get("target_row_ranking", []),
                }
            )
    for target_var, rankings in rankings_by_var.items():
        rankings.sort(
            key=lambda record: (
                float(record["selection_score"]),
                float(record["calibration_exact_acc"]),
                -int(record["site_index"]),
            ),
            reverse=True,
        )
        best_by_site: dict[int, dict[str, object]] = {}
        for record in rankings:
            site_index = int(record["site_index"])
            if site_index not in best_by_site:
                best_by_site[site_index] = record
        rankings_by_var[target_var] = list(best_by_site.values())
    return rankings_by_var


def main() -> None:
    stage_start = perf_counter()
    args = build_parser().parse_args()
    results_timestamp = (
        args.results_timestamp
        or os.environ.get("RESULTS_TIMESTAMP")
        or "pruned_last_token_clt"
    )
    output_dir = Path(args.results_root) / f"{results_timestamp}_mcqa_plot_pruned_last_token_clt"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_vars = tuple(canonicalize_target_var(target_var) for target_var in parse_csv_strings(args.target_vars))
    counterfactual_names = tuple(parse_csv_strings(args.counterfactual_names) or DEFAULT_COUNTERFACTUAL_NAMES)
    ot_epsilons = tuple(parse_csv_floats(args.ot_epsilons) or DEFAULT_OT_EPSILONS)
    beta_neurals = tuple(parse_csv_floats(args.uot_beta_neurals) or DEFAULT_UOT_BETA_NEURALS)
    calibration_family_weights = tuple(
        parse_csv_floats(args.calibration_family_weights) or DEFAULT_CALIBRATION_FAMILY_WEIGHTS
    )
    stage_a_methods = tuple(parse_csv_strings(args.stage_a_transport_methods) or ("uot",))
    transcoder_set = resolve_transcoder_set(args.transcoder_set, args.transcoder_size)

    graph_sites, graph_site_records = load_pruned_last_token_clt_sites(
        args.graph_json,
        top_k=int(args.candidate_top_k),
        rank_by=str(args.candidate_rank_by),
    )
    if not graph_sites:
        raise RuntimeError(f"No last-token cross layer transcoder nodes found in {args.graph_json}")
    log_progress(
        "loaded pruned last-token graph CLT sites "
        f"graph={args.graph_json} sites={len(graph_sites)} "
        f"rank_by={args.candidate_rank_by} top={graph_site_records[:3]}"
    )

    log_progress(f"loading HF filter model model={args.model_name}")
    filter_model, tokenizer = load_filter_model_and_tokenizer(
        model_name=args.model_name,
        dtype_name=args.dtype,
        hf_token=args.hf_token,
    )
    causal_model = MCQACausalModel()
    token_positions = get_token_positions(tokenizer, causal_model)

    log_progress(f"loading public MCQA counterfactual datasets from {args.dataset_path}")
    public_datasets = load_public_mcqa_datasets(
        size=int(args.dataset_size),
        dataset_path=str(args.dataset_path),
        dataset_config=args.dataset_config or None,
        hf_token=args.hf_token,
    )
    filtered_datasets = filter_correct_examples_with_hf_model(
        model=filter_model,
        tokenizer=tokenizer,
        causal_model=causal_model,
        datasets_by_name=public_datasets,
        batch_size=int(args.filter_batch_size),
    )
    del filter_model
    torch.cuda.empty_cache()
    log_progress(
        "filtered dataset counts "
        + ", ".join(f"{name}={len(rows)}" for name, rows in sorted(filtered_datasets.items()))
    )
    banks_by_split, data_metadata = build_pair_banks(
        tokenizer=tokenizer,
        causal_model=causal_model,
        token_positions=token_positions,
        datasets_by_name=filtered_datasets,
        counterfactual_names=counterfactual_names,
        target_vars=target_vars,
        split_seed=int(args.split_seed),
        train_pool_size=int(args.train_pool_size),
        calibration_pool_size=int(args.calibration_pool_size),
        test_pool_size=int(args.test_pool_size),
    )
    log_progress(
        "built MCQA pair banks "
        + ", ".join(
            f"{split}/{target_var}={bank.size}"
            for split, banks_by_var in banks_by_split.items()
            for target_var, bank in banks_by_var.items()
        )
    )

    from circuit_tracing_ot.model import load_replacement_model

    log_progress(f"loading ReplacementModel model={args.model_name} transcoder_set={transcoder_set}")
    model = load_replacement_model(
        model_name=args.model_name,
        transcoder_set=transcoder_set,
        dtype_name=args.dtype,
        offload=args.offload,
        backend=args.backend,
    )
    if getattr(model.tokenizer, "pad_token", None) is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"

    cache = CLTActivationCache(model)
    stage_a_config = OTConfig(
        method="ot",
        epsilon=1.0,
        signature_mode=str(args.signature_mode),
        source_target_vars=target_vars,
        calibration_family_weights=calibration_family_weights,
    )
    reference_train_bank = banks_by_split["train"][target_vars[0]]
    signature_checkpoint_metadata = {
        "kind": "mcqa_plot_pruned_last_token_clt_signatures",
        "graph_json": str(args.graph_json),
        "candidate_top_k": int(args.candidate_top_k),
        "candidate_rank_by": str(args.candidate_rank_by),
        "model_name": str(args.model_name),
        "transcoder_set": str(transcoder_set),
        "transcoder_size": str(args.transcoder_size),
        "dtype": str(args.dtype),
        "dataset_path": str(args.dataset_path),
        "dataset_config": None if args.dataset_config is None else str(args.dataset_config),
        "dataset_size": int(args.dataset_size),
        "split_seed": int(args.split_seed),
        "train_pool_size": int(args.train_pool_size),
        "target_vars": list(target_vars),
        "counterfactual_names": list(counterfactual_names),
        "train_base_prompts": [str(item["raw_input"]) for item in reference_train_bank.base_inputs],
        "train_source_prompts": [str(item["raw_input"]) for item in reference_train_bank.source_inputs],
        "train_counterfactual_families": list(reference_train_bank.counterfactual_family_names),
        "stage_a_sites": [site.label for site in graph_sites],
    }
    signature_checkpoint_path = output_dir / "stage_a_signature_checkpoint.pt"
    prepared_stage_a = prepare_alignment_artifacts_clt(
        model=model,
        fit_banks_by_var={target_var: banks_by_split["train"][target_var] for target_var in target_vars},
        sites=graph_sites,
        config=stage_a_config,
        cache=cache,
        checkpoint_path=signature_checkpoint_path,
        checkpoint_metadata=signature_checkpoint_metadata,
    )
    log_progress(
        "Stage A prepared pruned-feature signatures "
        f"sites={len(graph_sites)} runtime={float(prepared_stage_a.get('prepare_runtime_seconds', 0.0)):.1f}s"
    )

    stage_a_config_plan = []
    for method in stage_a_methods:
        if method not in {"ot", "uot"}:
            raise ValueError(f"Unsupported Stage A method {method}")
        for epsilon in ot_epsilons:
            betas = (None,) if method == "ot" else beta_neurals
            for beta in betas:
                stage_a_config_plan.append((str(method), float(epsilon), None if beta is None else float(beta)))

    config_checkpoint_path = output_dir / "stage_a_config_payloads.json"
    config_checkpoint_metadata = {
        "kind": "mcqa_plot_pruned_last_token_clt_stage_a_config_payloads",
        "graph_json": str(args.graph_json),
        "candidate_top_k": int(args.candidate_top_k),
        "candidate_rank_by": str(args.candidate_rank_by),
        "model_name": str(args.model_name),
        "transcoder_set": str(transcoder_set),
        "dataset_path": str(args.dataset_path),
        "dataset_config": None if args.dataset_config is None else str(args.dataset_config),
        "dataset_size": int(args.dataset_size),
        "split_seed": int(args.split_seed),
        "train_pool_size": int(args.train_pool_size),
        "calibration_pool_size": int(args.calibration_pool_size),
        "target_vars": list(target_vars),
        "counterfactual_names": list(counterfactual_names),
        "signature_mode": str(args.signature_mode),
        "calibration_family_weights": [float(weight) for weight in calibration_family_weights],
        "stage_a_row_top_k": int(args.stage_a_row_top_k),
        "stage_a_sites": [site.label for site in graph_sites],
        "config_keys": [
            stage_a_config_key(method, epsilon, beta)
            for method, epsilon, beta in stage_a_config_plan
        ],
        "calibration_base_prompts": {
            target_var: [
                str(item["raw_input"])
                for item in banks_by_split["calibration"][target_var].base_inputs
            ]
            for target_var in target_vars
        },
        "calibration_source_prompts": {
            target_var: [
                str(item["raw_input"])
                for item in banks_by_split["calibration"][target_var].source_inputs
            ]
            for target_var in target_vars
        },
    }
    stage_a_payloads_by_key = load_stage_a_config_checkpoint(
        path=config_checkpoint_path,
        metadata=config_checkpoint_metadata,
    )
    stage_a_payloads = []
    total_configs = len(stage_a_config_plan)
    for config_index, (method, epsilon, beta) in enumerate(stage_a_config_plan, start=1):
        config_key = stage_a_config_key(method, epsilon, beta)
        if config_key in stage_a_payloads_by_key:
            payload = stage_a_payloads_by_key[config_key]
            stage_a_payloads.append(payload)
            log_progress(
                f"Stage A config {config_index}/{total_configs} cache hit "
                f"method={method} epsilon={epsilon} beta={beta}"
            )
            continue
        log_progress(
            f"Stage A config {config_index}/{total_configs} "
            f"method={method} epsilon={epsilon} beta={beta} sites={len(graph_sites)}"
        )
        payload = run_stage_a_config(
            model=model,
            tokenizer=tokenizer,
            banks_by_split=banks_by_split,
            sites=graph_sites,
            prepared_artifacts=prepared_stage_a,
            method=method,
            epsilon=float(epsilon),
            beta_neural=beta,
            signature_mode=str(args.signature_mode),
            row_top_k=int(args.stage_a_row_top_k),
            calibration_family_weights=calibration_family_weights,
            cache=cache,
        )
        stage_a_payloads.append(payload)
        stage_a_payloads_by_key[config_key] = payload
        save_stage_a_config_checkpoint(
            path=config_checkpoint_path,
            metadata=config_checkpoint_metadata,
            payloads_by_key=stage_a_payloads_by_key,
        )
        log_progress(
            "Stage A config complete "
            f"method={method} epsilon={epsilon} beta={beta} "
            f"mean_score={float(payload['mean_calibration_score']):.4f} "
            f"mean_exact={float(payload['mean_calibration_exact_acc']):.4f}"
        )

    selected_stage_a = max(
        stage_a_payloads,
        key=lambda payload: (
            float(payload["mean_calibration_score"]),
            float(payload["mean_calibration_exact_acc"]),
        ),
    )
    stage_a_rankings_by_var = build_stage_a_site_rankings(stage_a_payloads=stage_a_payloads)
    if args.skip_stage_a_holdout:
        stage_a_holdout = {}
        log_progress("Stage A holdout evaluation skipped")
    else:
        log_progress("Stage A holdout evaluation start")
        stage_a_holdout = evaluate_stage_a_holdout(
            model=model,
            tokenizer=tokenizer,
            banks_by_split=banks_by_split,
            sites=graph_sites,
            selected_config=selected_stage_a,
            cache=cache,
        )
        log_progress("Stage A holdout evaluation complete")

    final_payload = {
        "kind": "mcqa_plot_pruned_last_token_clt",
        "config": {
            **vars(args),
            "graph_json": str(args.graph_json),
            "transcoder_set": transcoder_set,
            "target_vars": list(target_vars),
            "counterfactual_names": list(counterfactual_names),
            "ot_epsilons": [float(value) for value in ot_epsilons],
            "uot_beta_neurals": [float(value) for value in beta_neurals],
            "calibration_family_weights": [float(weight) for weight in calibration_family_weights],
        },
        "data": data_metadata,
        "counterfactual_families": list(COUNTERFACTUAL_FAMILIES),
        "candidate_graph_sites": graph_site_records,
        "stage_a": {
            "sites": [site.label for site in graph_sites],
            "payloads": stage_a_payloads,
            "selected_config": selected_stage_a,
            "rankings_by_var": stage_a_rankings_by_var,
            "holdout": stage_a_holdout,
        },
        "runtime_seconds": float(perf_counter() - stage_start),
    }
    output_path = output_dir / "mcqa_pruned_last_token_clt_results.json"
    write_json(output_path, final_payload)
    log_progress(f"wrote {output_path}")


if __name__ == "__main__":
    main()
