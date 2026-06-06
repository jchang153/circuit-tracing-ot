#!/usr/bin/env python
"""MCQA PLOT protocol with CLT feature interventions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import torch

from circuit_tracing_ot.config import MODEL_NAME, resolve_transcoder_set
from circuit_tracing_ot.logging import log_progress
from circuit_tracing_ot.mcqa_plot.clt_backend import (
    CLTActivationCache,
    enumerate_clt_layer_sites,
    enumerate_top_clt_feature_sites,
)
from circuit_tracing_ot.mcqa_plot.data import (
    COUNTERFACTUAL_FAMILIES,
    MCQACausalModel,
    build_pair_banks,
    canonicalize_target_var,
    get_token_positions,
    load_public_mcqa_datasets,
)
from circuit_tracing_ot.mcqa_plot.ot import (
    OTConfig,
    evaluate_single_site_intervention_clt,
    normalize_transport_rows,
    prepare_alignment_artifacts_clt,
    run_alignment_pipeline_clt,
    solve_ot_transport,
    solve_uot_transport,
    stage_a_calibration_score,
    target_row_ranking,
)
from circuit_tracing_ot.mcqa_plot.metrics import build_variable_signature


DEFAULT_TARGET_VARS = ("answer_pointer", "answer_token")
DEFAULT_COUNTERFACTUAL_NAMES = ("answerPosition", "randomLetter", "answerPosition_randomLetter")
DEFAULT_TOKEN_POSITION_ID = "last_token"
DEFAULT_SIGNATURE_MODE = "family_label_delta_norm"
DEFAULT_CALIBRATION_METRIC = "family_weighted_macro_exact_acc"
DEFAULT_CALIBRATION_FAMILY_WEIGHTS = (1.0, 1.0, 1.0)
DEFAULT_OT_EPSILONS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_UOT_BETA_NEURALS = (0.1, 0.3, 1.0, 3.0)


def parse_csv_ints(value: str | None) -> tuple[int, ...] | None:
    if value is None or not str(value).strip():
        return None
    parsed: list[int] = []
    for item in str(value).split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            step = 1 if end >= start else -1
            parsed.extend(range(start, end + step, step))
        else:
            parsed.append(int(token))
    return tuple(parsed)


def parse_csv_floats(value: str | None) -> tuple[float, ...] | None:
    if value is None or not str(value).strip():
        return None
    return tuple(float(item.strip()) for item in str(value).split(",") if item.strip())


def parse_csv_strings(value: str | None) -> tuple[str, ...] | None:
    if value is None or not str(value).strip():
        return None
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def parse_stage_a_layer_features(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "all", "full", "none"}:
        return None
    parsed = int(normalized)
    if parsed <= 0:
        raise ValueError("--stage-a-layer-features must be 'all' or a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--layers", help="Comma-separated layer indices/ranges, e.g. 0-25 or 0,3,7. Default: all CLT layers.")
    parser.add_argument(
        "--token-position-id",
        default=DEFAULT_TOKEN_POSITION_ID,
        choices=(DEFAULT_TOKEN_POSITION_ID,),
        help="MCQA PLOT CLT runs are restricted to the last token position.",
    )
    parser.add_argument("--target-vars", default=",".join(DEFAULT_TARGET_VARS))
    parser.add_argument("--counterfactual-names", default=",".join(DEFAULT_COUNTERFACTUAL_NAMES))
    parser.add_argument("--signature-mode", default=DEFAULT_SIGNATURE_MODE)
    parser.add_argument("--stage-a-transport-methods", default="uot")
    parser.add_argument("--ot-epsilons", default=",".join(str(value) for value in DEFAULT_OT_EPSILONS))
    parser.add_argument(
        "--uot-beta-neurals",
        default=",".join(str(value) for value in DEFAULT_UOT_BETA_NEURALS),
    )
    parser.add_argument("--stage-a-row-top-k", type=int, default=6)
    parser.add_argument(
        "--stage-a-layer-features",
        default="all",
        help="Number of CLT features copied for each Stage A layer site, or 'all'. Default: all.",
    )
    parser.add_argument("--top-layers", type=int, default=4)
    parser.add_argument("--stage-b-feature-candidates-per-layer", type=int, default=128)
    parser.add_argument("--stage-b-activation-read-top-k", type=int, default=512)
    parser.add_argument("--stage-b-top-k-values", default="1,2,4")
    parser.add_argument("--stage-b-lambdas", default="0.5,1.0,2.0,4.0")
    parser.add_argument(
        "--calibration-family-weights",
        default=",".join(str(value) for value in DEFAULT_CALIBRATION_FAMILY_WEIGHTS),
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--results-timestamp")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    parser.add_argument(
        "--filter-batch-size",
        type=int,
        default=32,
        help="Prompt batch size for plain-HF factual filtering.",
    )
    parser.add_argument("--skip-stage-b", action="store_true")
    parser.add_argument(
        "--skip-stage-a-holdout",
        action="store_true",
        help="Skip Stage A evaluation on the test split after calibration selection.",
    )
    return parser


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def stage_a_config_key(method: str, epsilon: float, beta_neural: float | None) -> str:
    beta_text = "none" if beta_neural is None else f"{float(beta_neural):.12g}"
    return f"method={method}|epsilon={float(epsilon):.12g}|beta={beta_text}"


def load_stage_a_config_checkpoint(
    *,
    path: Path,
    metadata: dict[str, object],
) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log_progress(
            "failed to load Stage A config checkpoint; recomputing configs "
            f"path={path} error={type(exc).__name__}: {exc}"
        )
        return {}
    if not isinstance(payload, dict) or payload.get("metadata") != metadata:
        log_progress(f"ignoring stale Stage A config checkpoint path={path}")
        return {}
    payloads_by_key = payload.get("payloads_by_key", {})
    if not isinstance(payloads_by_key, dict):
        return {}
    log_progress(
        "loaded Stage A config checkpoint "
        f"path={path} completed_configs={len(payloads_by_key)}"
    )
    return {
        str(key): value
        for key, value in payloads_by_key.items()
        if isinstance(value, dict)
    }


def save_stage_a_config_checkpoint(
    *,
    path: Path,
    metadata: dict[str, object],
    payloads_by_key: dict[str, dict[str, object]],
) -> None:
    write_json_atomic(
        path,
        {
            "metadata": metadata,
            "payloads_by_key": payloads_by_key,
        },
    )


def load_filter_model_and_tokenizer(
    *,
    model_name: str,
    dtype_name: str,
    hf_token: str | None,
):
    """Load a plain HF causal LM for original-style batched factual filtering."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from circuit_tracing_ot.model import check_cuda_usable, parse_dtype

    check_cuda_usable()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MCQA filtering.")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=parse_dtype(dtype_name),
        token=hf_token,
        attn_implementation="eager",
    )
    model.to("cuda")
    model.eval()
    log_progress(
        "loaded HF filter model "
        f"device={next(model.parameters()).device} cuda={torch.cuda.get_device_name(0)}"
    )
    return model, tokenizer


def build_position_ids_from_left_padded_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


def infer_next_token_ids_hf(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = build_position_ids_from_left_padded_attention_mask(attention_mask)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
    logits = outputs.logits
    reversed_mask = torch.flip(attention_mask.long(), dims=(1,))
    trailing_pad = torch.argmax(reversed_mask, dim=1)
    last_indices = logits.shape[1] - 1 - trailing_pad
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, last_indices].argmax(dim=-1)


def predicted_token_ids_hf(
    *,
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
) -> list[int]:
    batch_size = max(1, int(batch_size))
    predictions: list[int] = []
    device = next(model.parameters()).device
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        batch_predictions = infer_next_token_ids_hf(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        predictions.extend(int(token_id) for token_id in batch_predictions.detach().cpu().tolist())
    return predictions


def encode_symbol_variants(symbol: str, tokenizer) -> set[int]:
    variants = set()
    for candidate in (" " + str(symbol).strip(), str(symbol).strip()):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            variants.add(int(ids[0]))
    return variants


def filter_correct_examples_with_hf_model(
    *,
    model,
    tokenizer,
    causal_model: MCQACausalModel,
    datasets_by_name: dict[str, list[dict[str, object]]],
    batch_size: int,
) -> dict[str, list[dict[str, object]]]:
    """Original factual filtering, using batched plain-HF next-token logits."""
    filtered: dict[str, list[dict[str, object]]] = {}
    for dataset_name, rows in datasets_by_name.items():
        kept = []
        log_progress(f"filtering {dataset_name} rows={len(rows)}")
        base_inputs = [row["input"] for row in rows]
        source_inputs = [row["counterfactual_inputs"][0] for row in rows]
        base_expected_answers = [
            str(causal_model.run_forward(base_input)["raw_output"]).strip()
            for base_input in base_inputs
        ]
        source_expected_answers = [
            str(causal_model.run_forward(source_input)["raw_output"]).strip()
            for source_input in source_inputs
        ]
        base_expected_variants = [
            encode_symbol_variants(expected, tokenizer) for expected in base_expected_answers
        ]
        source_expected_variants = [
            encode_symbol_variants(expected, tokenizer) for expected in source_expected_answers
        ]
        base_predictions = predicted_token_ids_hf(
            model=model,
            tokenizer=tokenizer,
            prompts=[str(base_input["raw_input"]) for base_input in base_inputs],
            batch_size=int(batch_size),
        )
        source_predictions = predicted_token_ids_hf(
            model=model,
            tokenizer=tokenizer,
            prompts=[str(source_input["raw_input"]) for source_input in source_inputs],
            batch_size=int(batch_size),
        )
        for row, base_predicted, source_predicted, base_variants, source_variants in zip(
            rows,
            base_predictions,
            source_predictions,
            base_expected_variants,
            source_expected_variants,
            strict=True,
        ):
            if (
                int(base_predicted) in base_variants
                and int(source_predicted) in source_variants
            ):
                kept.append(row)
        filtered[dataset_name] = kept
        log_progress(f"filtered {dataset_name}: kept={len(kept)}/{len(rows)}")
    return filtered


def solve_transport_only(
    *,
    fit_banks_by_var,
    site_signatures,
    config: OTConfig,
) -> tuple[dict[str, torch.Tensor], object, dict[str, object]]:
    variable_signatures_by_var = {
        target_var: build_variable_signature(fit_banks_by_var[target_var], config.signature_mode)
        for target_var in config.source_target_vars
    }
    if config.method == "ot":
        transport, transport_meta = solve_ot_transport(variable_signatures_by_var, site_signatures, config)
    else:
        transport, transport_meta = solve_uot_transport(variable_signatures_by_var, site_signatures, config)
    log_progress(
        "Stage A transport solved "
        f"method={config.method} epsilon={float(config.epsilon):g} "
        f"beta={None if config.method == 'ot' else float(config.uot_beta_neural)} "
        f"transport_shape={tuple(transport.shape)}"
    )
    return variable_signatures_by_var, transport, transport_meta


def run_stage_a_config(
    *,
    model,
    tokenizer,
    banks_by_split,
    sites,
    prepared_artifacts,
    method: str,
    epsilon: float,
    beta_neural: float | None,
    signature_mode: str,
    row_top_k: int,
    calibration_family_weights: tuple[float, ...],
    cache: CLTActivationCache,
) -> dict[str, object]:
    target_vars = tuple(canonicalize_target_var(target_var) for target_var in DEFAULT_TARGET_VARS)
    config = OTConfig(
        method=method,
        epsilon=float(epsilon),
        uot_beta_neural=1.0 if beta_neural is None else float(beta_neural),
        signature_mode=signature_mode,
        source_target_vars=target_vars,
        calibration_family_weights=calibration_family_weights,
        top_k_values=(1,),
        lambda_values=(1.0,),
        store_prediction_details=False,
    )
    _variable_signatures, transport, transport_meta = solve_transport_only(
        fit_banks_by_var={target_var: banks_by_split["train"][target_var] for target_var in target_vars},
        site_signatures=prepared_artifacts["site_signatures"],
        config=config,
    )
    normalized_transport = normalize_transport_rows(transport)
    per_var_records = {}
    for target_row_index, target_var in enumerate(target_vars):
        row_payload = {
            "target_normalized_transport": normalized_transport[target_row_index : target_row_index + 1].tolist(),
            "target_transport": transport[target_row_index : target_row_index + 1].tolist(),
        }
        row_ranking = target_row_ranking(row_payload, sites=sites)
        log_progress(
            "Stage A target row ranking "
            f"target_var={target_var} top="
            + ", ".join(
                f"{entry['site_label']}:{float(entry['transport_mass']):.4g}"
                for entry in row_ranking[: min(5, len(row_ranking))]
            )
        )
        candidate_records = []
        for rank_index, entry in enumerate(row_ranking[: max(1, int(row_top_k))]):
            site_index = int(entry["site_index"])
            site = sites[site_index]
            log_progress(
                "Stage A calibration candidate "
                f"target_var={target_var} rank={rank_index + 1}/{max(1, int(row_top_k))} "
                f"site={site.label} transport_mass={float(entry['transport_mass']):.4g}"
            )
            calibration_result, calibration_ranking = evaluate_single_site_intervention_clt(
                model=model,
                bank=banks_by_split["calibration"][target_var],
                site=site,
                site_index=site_index,
                strength=1.0,
                tokenizer=tokenizer,
                cache=cache,
                include_details=True,
            )
            calibration_score = stage_a_calibration_score(
                result=calibration_result,
                calibration_family_weights=calibration_family_weights,
            )
            log_progress(
                "Stage A calibration candidate result "
                f"target_var={target_var} site={site.label} "
                f"exact_acc={float(calibration_result['exact_acc']):.4f} "
                f"selection_score={float(calibration_score):.4f}"
            )
            candidate_records.append(
                {
                    "rank_index": int(rank_index),
                    "site_index": int(site_index),
                    "site": site,
                    "site_label": site.label,
                    "layer": int(site.layer),
                    "feature_idx": site.feature_idx,
                    "transport_mass": float(entry["transport_mass"]),
                    "calibration_score": float(calibration_score),
                    "calibration_exact_acc": float(calibration_result["exact_acc"]),
                    "calibration_result": calibration_result,
                    "calibration_ranking": calibration_ranking,
                }
            )
        best = max(
            candidate_records,
            key=lambda record: (
                float(record["calibration_score"]),
                float(record["calibration_exact_acc"]),
                float(record["transport_mass"]),
                -int(record["rank_index"]),
            ),
        )
        log_progress(
            "Stage A selected candidate "
            f"target_var={target_var} site={best['site_label']} "
            f"score={float(best['calibration_score']):.4f} "
            f"exact_acc={float(best['calibration_exact_acc']):.4f}"
        )
        per_var_records[target_var] = {
            "method": method,
            "variable": target_var,
            "selection_score": float(best["calibration_score"]),
            "selection_exact_acc": float(best["calibration_exact_acc"]),
            "calibration_exact_acc": float(best["calibration_exact_acc"]),
            "site_index": int(best["site_index"]),
            "site_label": str(best["site_label"]),
            "layer": int(best["layer"]),
            "feature_idx": best["feature_idx"],
            "epsilon": float(epsilon),
            "uot_beta_neural": None if beta_neural is None else float(beta_neural),
            "candidate_records": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"site", "calibration_result", "calibration_ranking"}
                }
                for candidate in candidate_records
            ],
            "target_row_ranking": row_ranking,
            "calibration_payload": best["calibration_result"],
        }
    scores = [float(record["selection_score"]) for record in per_var_records.values()]
    exacts = [float(record["selection_exact_acc"]) for record in per_var_records.values()]
    return {
        "kind": "mcqa_plot_clt_stage_a_config",
        "method": method,
        "epsilon": float(epsilon),
        "uot_beta_neural": None if beta_neural is None else float(beta_neural),
        "transport": transport.tolist(),
        "normalized_transport": normalized_transport.tolist(),
        "transport_meta": transport_meta,
        "per_var_records": per_var_records,
        "mean_calibration_score": float(sum(scores) / len(scores)),
        "mean_calibration_exact_acc": float(sum(exacts) / len(exacts)),
    }


def evaluate_stage_a_holdout(
    *,
    model,
    tokenizer,
    banks_by_split,
    sites,
    selected_config: dict[str, object],
    cache: CLTActivationCache,
) -> dict[str, object]:
    holdout_records = {}
    for target_var, record in selected_config["per_var_records"].items():
        site = sites[int(record["site_index"])]
        holdout_result, holdout_ranking = evaluate_single_site_intervention_clt(
            model=model,
            bank=banks_by_split["test"][target_var],
            site=site,
            site_index=int(record["site_index"]),
            strength=1.0,
            tokenizer=tokenizer,
            cache=cache,
            include_details=True,
        )
        holdout_result["method"] = "single_layer_full_swap"
        holdout_result["selection_score"] = float(record["selection_score"])
        holdout_result["selection_exact_acc"] = float(record["selection_exact_acc"])
        holdout_result["calibration_exact_acc"] = float(record["calibration_exact_acc"])
        holdout_records[target_var] = {
            "selected_site_label": site.label,
            "selected_layer": int(site.layer),
            "selected_feature_idx": site.feature_idx,
            "ranking": holdout_ranking,
            "results": [holdout_result],
        }
    return holdout_records


def build_stage_a_layer_rankings(
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
            candidate = {
                "target_var": str(target_var),
                "layer": int(record["layer"]),
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
            rankings_by_var.setdefault(str(target_var), []).append(candidate)
    for target_var, rankings in rankings_by_var.items():
        rankings.sort(
            key=lambda record: (
                float(record["selection_score"]),
                float(record["calibration_exact_acc"]),
                -int(record["site_index"]),
            ),
            reverse=True,
        )
        best_by_layer: dict[int, dict[str, object]] = {}
        for record in rankings:
            layer = int(record["layer"])
            if layer not in best_by_layer:
                best_by_layer[layer] = record
        rankings_by_var[target_var] = list(best_by_layer.values())
    return rankings_by_var


def main() -> None:
    stage_start = perf_counter()
    args = build_parser().parse_args()
    results_timestamp = args.results_timestamp or os.environ.get("RESULTS_TIMESTAMP") or "mcqa_clt"
    output_dir = Path(args.results_root) / f"{results_timestamp}_mcqa_plot_clt"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_vars = tuple(canonicalize_target_var(target_var) for target_var in parse_csv_strings(args.target_vars))
    counterfactual_names = tuple(parse_csv_strings(args.counterfactual_names) or DEFAULT_COUNTERFACTUAL_NAMES)
    ot_epsilons = tuple(parse_csv_floats(args.ot_epsilons) or DEFAULT_OT_EPSILONS)
    beta_neurals = tuple(parse_csv_floats(args.uot_beta_neurals) or DEFAULT_UOT_BETA_NEURALS)
    calibration_family_weights = tuple(
        parse_csv_floats(args.calibration_family_weights) or DEFAULT_CALIBRATION_FAMILY_WEIGHTS
    )
    stage_b_top_k_values = tuple(parse_csv_ints(args.stage_b_top_k_values) or (1, 2, 4))
    stage_b_lambdas = tuple(parse_csv_floats(args.stage_b_lambdas) or (0.5, 1.0, 2.0, 4.0))
    stage_a_methods = tuple(parse_csv_strings(args.stage_a_transport_methods) or ("uot",))
    stage_a_layer_features = parse_stage_a_layer_features(args.stage_a_layer_features)

    log_progress(f"loading HF filter model model={args.model_name}")
    filter_model, tokenizer = load_filter_model_and_tokenizer(
        model_name=args.model_name,
        dtype_name=args.dtype,
        hf_token=args.hf_token,
    )
    causal_model = MCQACausalModel()
    token_positions = get_token_positions(tokenizer, causal_model)
    token_position_ids = tuple(token_position.id for token_position in token_positions)
    if args.token_position_id not in token_position_ids:
        raise ValueError(f"Unknown token position {args.token_position_id}; available={token_position_ids}")

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

    transcoder_set = resolve_transcoder_set(args.transcoder_set, args.transcoder_size)
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

    num_layers = 26
    layers = parse_csv_ints(args.layers) or tuple(range(num_layers))
    cache = CLTActivationCache(model)
    layer_sites = enumerate_clt_layer_sites(
        num_layers=num_layers,
        token_position_id=str(args.token_position_id),
        layers=layers,
        top_features=stage_a_layer_features,
    )
    log_progress(
        "Stage A layer sites "
        f"count={len(layer_sites)} layers={[int(site.layer) for site in layer_sites]} "
        f"copy_features={'all' if stage_a_layer_features is None else int(stage_a_layer_features)}"
    )
    stage_a_config = OTConfig(
        method="ot",
        epsilon=1.0,
        signature_mode=str(args.signature_mode),
        source_target_vars=target_vars,
        calibration_family_weights=calibration_family_weights,
    )
    reference_train_bank = banks_by_split["train"][target_vars[0]]
    stage_a_checkpoint_metadata = {
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
        "token_position_id": str(args.token_position_id),
        "stage_a_layer_features": "all"
        if stage_a_layer_features is None
        else int(stage_a_layer_features),
        "train_base_prompts": [str(item["raw_input"]) for item in reference_train_bank.base_inputs],
        "train_source_prompts": [str(item["raw_input"]) for item in reference_train_bank.source_inputs],
        "train_counterfactual_families": list(reference_train_bank.counterfactual_family_names),
    }
    stage_a_checkpoint_path = output_dir / "stage_a_signature_checkpoint.pt"
    prepared_stage_a = prepare_alignment_artifacts_clt(
        model=model,
        fit_banks_by_var={target_var: banks_by_split["train"][target_var] for target_var in target_vars},
        sites=layer_sites,
        config=stage_a_config,
        cache=cache,
        checkpoint_path=stage_a_checkpoint_path,
        checkpoint_metadata=stage_a_checkpoint_metadata,
    )
    log_progress(
        "Stage A prepared signatures "
        f"sites={len(layer_sites)} runtime={float(prepared_stage_a.get('prepare_runtime_seconds', 0.0)):.1f}s"
    )

    stage_a_config_plan = []
    for method in stage_a_methods:
        if method not in {"ot", "uot"}:
            raise ValueError(f"Unsupported Stage A method {method}")
        for epsilon in ot_epsilons:
            betas = (None,) if method == "ot" else beta_neurals
            for beta in betas:
                stage_a_config_plan.append((str(method), float(epsilon), None if beta is None else float(beta)))
    stage_a_config_checkpoint_path = output_dir / "stage_a_config_payloads.json"
    stage_a_config_checkpoint_metadata = {
        "kind": "mcqa_plot_clt_stage_a_config_payloads",
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
        "stage_a_layer_features": "all"
        if stage_a_layer_features is None
        else int(stage_a_layer_features),
        "stage_a_sites": [site.label for site in layer_sites],
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
        path=stage_a_config_checkpoint_path,
        metadata=stage_a_config_checkpoint_metadata,
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
            f"method={method} epsilon={epsilon} beta={beta} sites={len(layer_sites)}"
        )
        payload = run_stage_a_config(
            model=model,
            tokenizer=tokenizer,
            banks_by_split=banks_by_split,
            sites=layer_sites,
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
            path=stage_a_config_checkpoint_path,
            metadata=stage_a_config_checkpoint_metadata,
            payloads_by_key=stage_a_payloads_by_key,
        )
        log_progress(
            "Stage A config checkpoint saved "
            f"path={stage_a_config_checkpoint_path} completed_configs={len(stage_a_payloads_by_key)}/{total_configs}"
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
    stage_a_layer_rankings_by_var = build_stage_a_layer_rankings(stage_a_payloads=stage_a_payloads)
    for target_var, ranking in stage_a_layer_rankings_by_var.items():
        log_progress(
            "Stage A layer ranking "
            f"target_var={target_var} top="
            + ", ".join(
                f"L{int(record['layer'])}:score={float(record['selection_score']):.4f}"
                for record in ranking[: min(5, len(ranking))]
            )
        )
    if args.skip_stage_a_holdout:
        stage_a_holdout = {}
        log_progress("Stage A holdout evaluation skipped")
    else:
        log_progress("Stage A holdout evaluation start")
        stage_a_holdout = evaluate_stage_a_holdout(
            model=model,
            tokenizer=tokenizer,
            banks_by_split=banks_by_split,
            sites=layer_sites,
            selected_config=selected_stage_a,
            cache=cache,
        )
        log_progress("Stage A holdout evaluation complete")
    selected_layers = sorted(
        {
            int(record["layer"])
            for record in selected_stage_a["per_var_records"].values()
        }
    )[: int(args.top_layers)]

    stage_b_payloads = []
    if not args.skip_stage_b:
        for layer in selected_layers:
            log_progress(f"Stage B enumerating CLT feature candidates for layer={layer}")
            feature_sites = enumerate_top_clt_feature_sites(
                model=model,
                bank=banks_by_split["train"][target_vars[0]],
                layers=(int(layer),),
                token_position_id=str(args.token_position_id),
                top_features_per_layer=int(args.stage_b_feature_candidates_per_layer),
                activation_read_top_k=int(args.stage_b_activation_read_top_k),
                cache=cache,
            )
            if not feature_sites:
                continue
            log_progress(
                "Stage B feature sites "
                f"layer={layer} count={len(feature_sites)} "
                f"first_sites={[site.label for site in feature_sites[:5]]}"
            )
            prepared_stage_b = prepare_alignment_artifacts_clt(
                model=model,
                fit_banks_by_var={target_var: banks_by_split["train"][target_var] for target_var in target_vars},
                sites=feature_sites,
                config=stage_a_config,
                cache=cache,
            )
            for target_var in target_vars:
                best_payload = None
                for epsilon in ot_epsilons:
                    config = OTConfig(
                        method="ot",
                        epsilon=float(epsilon),
                        signature_mode=str(args.signature_mode),
                        top_k_values=stage_b_top_k_values,
                        lambda_values=stage_b_lambdas,
                        source_target_vars=target_vars,
                        calibration_metric=DEFAULT_CALIBRATION_METRIC,
                        calibration_family_weights=calibration_family_weights,
                        store_prediction_details=True,
                    )
                    payload = run_alignment_pipeline_clt(
                        model=model,
                        fit_banks_by_var={source_var: banks_by_split["train"][source_var] for source_var in target_vars},
                        calibration_bank=banks_by_split["calibration"][target_var],
                        holdout_bank=banks_by_split["test"][target_var],
                        sites=feature_sites,
                        tokenizer=tokenizer,
                        config=config,
                        cache=cache,
                        prepared_artifacts=prepared_stage_b,
                    )
                    payload["layer"] = int(layer)
                    payload["candidate_sites"] = [site.label for site in feature_sites]
                    payload["candidate_features"] = [site.feature_idx for site in feature_sites]
                    payload["ot_epsilon"] = float(epsilon)
                    if best_payload is None or (
                        float(payload["selected_calibration_result"]["exact_acc"]),
                        float(payload["results"][0]["selection_score"]),
                    ) > (
                        float(best_payload["selected_calibration_result"]["exact_acc"]),
                        float(best_payload["results"][0]["selection_score"]),
                    ):
                        best_payload = payload
                if best_payload is not None:
                    stage_b_payloads.append(best_payload)

    final_payload = {
        "kind": "mcqa_plot_clt",
        "config": {
            **vars(args),
            "transcoder_set": transcoder_set,
            "target_vars": list(target_vars),
            "counterfactual_names": list(counterfactual_names),
            "ot_epsilons": [float(value) for value in ot_epsilons],
            "uot_beta_neurals": [float(value) for value in beta_neurals],
            "calibration_family_weights": [float(value) for value in calibration_family_weights],
            "stage_a_layer_features": "all"
            if stage_a_layer_features is None
            else int(stage_a_layer_features),
        },
        "data": data_metadata,
        "counterfactual_families": list(COUNTERFACTUAL_FAMILIES),
        "stage_a": {
            "sites": [site.label for site in layer_sites],
            "payloads": stage_a_payloads,
            "selected_config": selected_stage_a,
            "layer_rankings_by_var": stage_a_layer_rankings_by_var,
            "holdout": stage_a_holdout,
            "selected_layers": selected_layers,
        },
        "stage_b": {
            "payloads": stage_b_payloads,
            "selected_features": [
                {
                    "target_var": payload["target_var"],
                    "layer": int(payload["layer"]),
                    "ranking": payload.get("ranking", [])[:10],
                    "result": payload["results"][0],
                    "selected_hyperparameters": payload["selected_hyperparameters"],
                }
                for payload in stage_b_payloads
            ],
        },
        "runtime_seconds": float(perf_counter() - stage_start),
    }
    output_path = output_dir / "mcqa_plot_clt_results.json"
    write_json(output_path, final_payload)
    log_progress(f"wrote {output_path}")


if __name__ == "__main__":
    main()
