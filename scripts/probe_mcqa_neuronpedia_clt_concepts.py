#!/usr/bin/env python
"""Train MCQA concept probes on Neuronpedia clt-hp features without ReplacementModel."""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from circuit_tracing_ot.config import MODEL_NAME
from circuit_tracing_ot.logging import log_progress
from circuit_tracing_ot.mcqa_plot.data import (
    COUNTERFACTUAL_FAMILIES,
    MCQACausalModel,
    load_public_mcqa_datasets,
)
from circuit_tracing_ot.model import check_cuda_usable, parse_dtype
from probe_mcqa_clt_concepts import (
    SplitIndices,
    aggregate_bootstrap_ranking,
    build_prompt_records,
    canonicalize_target_var,
    deep_update,
    layer_probe_grid,
    load_config,
    num_classes_for_target,
    parse_csv_ints,
    parse_csv_strings,
    run_bootstraps,
    split_records_by_group,
    target_labels,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mcqa_neuronpedia_clt_probe.yaml"))
    parser.add_argument("--dataset-size", type=int)
    parser.add_argument("--layers")
    parser.add_argument("--targets")
    parser.add_argument("--neuronpedia-clt-repo")
    parser.add_argument("--activation-top-k", type=int)
    parser.add_argument("--activation-batch-size", type=int)
    parser.add_argument("--feature-cap-per-layer", type=int)
    parser.add_argument("--num-bootstraps", type=int)
    parser.add_argument("--results-timestamp")
    return parser


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if args.dataset_size is not None:
        updates.setdefault("dataset", {})["dataset_size"] = int(args.dataset_size)
    if args.layers is not None:
        updates.setdefault("features", {})["layers"] = str(args.layers)
    if args.targets is not None:
        updates.setdefault("labels", {})["targets"] = list(parse_csv_strings(args.targets) or ())
    if args.neuronpedia_clt_repo is not None:
        updates.setdefault("model", {})["neuronpedia_clt_repo"] = str(args.neuronpedia_clt_repo)
    if args.activation_top_k is not None:
        updates.setdefault("features", {})["activation_top_k"] = int(args.activation_top_k)
    if args.activation_batch_size is not None:
        updates.setdefault("features", {})["activation_batch_size"] = int(args.activation_batch_size)
    if args.feature_cap_per_layer is not None:
        updates.setdefault("features", {})["feature_cap_per_layer"] = int(args.feature_cap_per_layer)
    if args.num_bootstraps is not None:
        updates.setdefault("stability", {})["num_bootstraps"] = int(args.num_bootstraps)
    if args.results_timestamp is not None:
        updates.setdefault("outputs", {})["results_timestamp"] = str(args.results_timestamp)
    return deep_update(config, updates)


def load_hooked_transformer(*, model_name: str, dtype_name: str, device: torch.device):
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        model_name,
        device=str(device),
        dtype=parse_dtype(dtype_name),
    )
    if getattr(model.tokenizer, "pad_token", None) is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"
    model.eval()
    return model


def load_neuronpedia_encoder(
    *,
    repo_id: str,
    layer: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, Path]:
    path = Path(hf_hub_download(repo_id=repo_id, filename=f"layer_{int(layer)}.safetensors"))
    with safe_open(str(path), framework="pt", device=str(device)) as f:
        w_enc = f.get_tensor("W_enc").to(device=device, dtype=dtype)
        b_enc = f.get_tensor("b_enc").to(device=device, dtype=dtype)
    return w_enc, b_enc, path


def tokenizer_batch(
    model,
    prompts: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.tokenizer.padding_side = "right"
    encoded = model.tokenizer(
        prompts,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    last_positions = attention_mask.sum(dim=-1).to(torch.long) - 1
    return input_ids, attention_mask, last_positions


def collect_layer_topk_rows(
    *,
    model,
    records,
    layer: int,
    repo_id: str,
    activation_top_k: int,
    activation_batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[int], dict[str, object]]:
    w_enc, b_enc, encoder_path = load_neuronpedia_encoder(
        repo_id=repo_id,
        layer=int(layer),
        device=device,
        dtype=dtype,
    )
    hook_name = f"blocks.{int(layer)}.hook_resid_mid"
    rows: list[tuple[torch.Tensor, torch.Tensor]] = []
    counts: Counter[int] = Counter()
    start = perf_counter()
    with torch.inference_mode():
        for start_index in range(0, len(records), int(activation_batch_size)):
            end_index = min(len(records), start_index + int(activation_batch_size))
            if start_index == 0 or end_index == len(records) or end_index % 100 == 0:
                log_progress(
                    "collecting Neuronpedia CLT top-k "
                    f"layer={layer} row={end_index}/{len(records)} "
                    f"elapsed={perf_counter() - start:.1f}s"
                )
            prompts = [record.prompt for record in records[start_index:end_index]]
            tokens, attention_mask, last_positions = tokenizer_batch(model, prompts, device)
            _logits, cache = model.run_with_cache(
                tokens,
                attention_mask=attention_mask,
                names_filter=[hook_name],
                remove_batch_dim=False,
            )
            batch_rows = torch.arange(tokens.shape[0], device=device)
            resid_mid = cache[hook_name][batch_rows, last_positions, :]
            pre_acts = F.linear(resid_mid.to(dtype), w_enc, b_enc)
            activations = F.relu(pre_acts).float()
            k = min(int(activation_top_k), int(activations.shape[-1]))
            values, indices = torch.topk(activations, k=k, dim=-1)
            for row_values, row_indices in zip(values.cpu(), indices.cpu()):
                counts.update(int(feature_idx) for feature_idx in row_indices.tolist())
                rows.append((row_indices, row_values))
            del cache, resid_mid, pre_acts, activations, values, indices, tokens, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), int(item[0])))
    metadata = {
        "repo_id": str(repo_id),
        "layer": int(layer),
        "hook_name": hook_name,
        "encoder_path": str(encoder_path),
        "d_features": int(b_enc.numel()),
    }
    del w_enc, b_enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, [int(feature_idx) for feature_idx, _count in ranked], metadata


def build_layer_feature_payload(
    *,
    model,
    records,
    layer: int,
    splits: SplitIndices,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, object]:
    feature_config = config["features"]
    model_config = config["model"]
    rows_topk, ranked_features, metadata = collect_layer_topk_rows(
        model=model,
        records=records,
        layer=int(layer),
        repo_id=str(model_config["neuronpedia_clt_repo"]),
        activation_top_k=int(feature_config["activation_top_k"]),
        activation_batch_size=int(feature_config.get("activation_batch_size", 8)),
        dtype=parse_dtype(str(model_config.get("dtype", "bf16"))),
        device=device,
    )
    feature_ids = ranked_features[: int(feature_config["feature_cap_per_layer"])]
    if not feature_ids:
        raise RuntimeError(f"No screened features for layer {layer}")
    feature_to_col = {int(feature_idx): col for col, feature_idx in enumerate(feature_ids)}
    matrix = torch.zeros((len(records), len(feature_ids)), dtype=torch.float32)
    for row_index, (indices, values) in enumerate(rows_topk):
        for feature_idx, value in zip(indices.tolist(), values.tolist()):
            col = feature_to_col.get(int(feature_idx))
            if col is not None:
                matrix[row_index, int(col)] = float(value)
    mean = std = None
    if bool(feature_config.get("standardize_features", True)):
        train_index_tensor = torch.tensor(splits.train, dtype=torch.long)
        train_matrix = matrix.index_select(0, train_index_tensor)
        mean = train_matrix.mean(dim=0)
        variance = torch.clamp(train_matrix.square().mean(dim=0) - mean.square(), min=0.0)
        std = torch.sqrt(variance + float(feature_config.get("standardization_eps", 1.0e-6)))
        matrix.sub_(mean.view(1, -1)).div_(std.view(1, -1))
    log_progress(
        "built Neuronpedia CLT probe matrix "
        f"layer={layer} shape={tuple(matrix.shape)}"
    )
    return {
        "layer": int(layer),
        "feature_ids": feature_ids,
        "features": matrix,
        "mean": mean,
        "std": std,
        "metadata": metadata,
    }


def neuronpedia_feature_url(config: dict[str, Any], layer: int, feature_idx: int) -> str:
    model_id = str(config["model"].get("neuronpedia_model_id", "gemma-2-2b"))
    suffix = str(config["model"].get("neuronpedia_source_suffix", "clt-hp"))
    return f"https://www.neuronpedia.org/{model_id}/{int(layer)}-{suffix}/{int(feature_idx)}"


def add_neuronpedia_urls(
    *,
    rows: list[dict[str, object]],
    config: dict[str, Any],
    layer: int,
) -> list[dict[str, object]]:
    enriched = []
    for row in rows:
        item = dict(row)
        item["neuronpedia_url"] = neuronpedia_feature_url(
            config,
            layer=int(layer),
            feature_idx=int(item["feature_idx"]),
        )
        enriched.append(item)
    return enriched


def run_targets(
    *,
    records,
    targets: tuple[str, ...],
    splits: SplitIndices,
    layers: tuple[int, ...],
    model,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, object]:
    labels_by_target = {target_var: target_labels(records, target_var) for target_var in targets}
    states: dict[str, dict[str, object]] = {
        target_var: {
            "layer_results": [],
            "selected_layer_summary": None,
            "selected_payload": None,
        }
        for target_var in targets
    }
    for layer in layers:
        layer_feature_payload = build_layer_feature_payload(
            model=model,
            records=records,
            layer=int(layer),
            splits=splits,
            config=config,
            device=device,
        )
        selected_by_any_target = False
        for target_var in targets:
            layer_payload = layer_probe_grid(
                target_var=target_var,
                labels=labels_by_target[target_var],
                splits=splits,
                layer_payload=layer_feature_payload,
                config=config,
                device=device,
            )
            top_features = add_neuronpedia_urls(
                rows=layer_payload["top_features_by_main_probe"],
                config=config,
                layer=int(layer),
            )
            layer_summary = {
                "layer": int(layer),
                "best_record": layer_payload["best_record"],
                "top_features_by_main_probe": top_features,
                "screened_feature_count": len(layer_payload["feature_ids"]),
                "feature_metadata": layer_feature_payload["metadata"],
            }
            states[target_var]["layer_results"].append(layer_summary)
            write_json(output_dir / f"{target_var}_layer_{int(layer)}_summary.json", layer_summary)
            candidate_key = (
                float(layer_summary["best_record"]["validation_macro_accuracy"]),
                float(layer_summary["best_record"]["validation_accuracy"]),
                -float(layer_summary["best_record"]["validation_loss"]),
                -int(layer_summary["layer"]),
            )
            selected_layer_summary = states[target_var]["selected_layer_summary"]
            best_key = (-1.0, -1.0, -math.inf, -10**9)
            if selected_layer_summary is not None:
                best_key = (
                    float(selected_layer_summary["best_record"]["validation_macro_accuracy"]),
                    float(selected_layer_summary["best_record"]["validation_accuracy"]),
                    -float(selected_layer_summary["best_record"]["validation_loss"]),
                    -int(selected_layer_summary["layer"]),
                )
            if selected_layer_summary is None or candidate_key > best_key:
                states[target_var]["selected_layer_summary"] = layer_summary
                states[target_var]["selected_payload"] = layer_payload
                selected_by_any_target = True
                log_progress(
                    "selected layer candidate updated "
                    f"target={target_var} layer={int(layer)} "
                    f"val_macro={float(layer_summary['best_record']['validation_macro_accuracy']):.4f}"
                )
            else:
                del layer_payload
        if not selected_by_any_target:
            del layer_feature_payload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    target_payloads: dict[str, object] = {}
    for target_var in targets:
        selected_layer_summary = states[target_var]["selected_layer_summary"]
        selected_payload = states[target_var]["selected_payload"]
        if selected_layer_summary is None or selected_payload is None:
            raise RuntimeError(f"No selected layer payload for target={target_var}")
        selected_layer = int(selected_layer_summary["layer"])
        selected_fit = selected_payload["best_fit"]
        bootstrap_records = run_bootstraps(
            features=selected_payload["features"],
            labels=labels_by_target[target_var],
            train_indices=splits.train,
            validation_indices=splits.validation,
            feature_ids=selected_payload["feature_ids"],
            num_classes=num_classes_for_target(target_var),
            selected_fit=selected_payload["best_record"],
            config=config,
            device=device,
            seed=int(config["dataset"]["split_seed"]) + selected_layer * 9973,
        )
        feature_ranking = aggregate_bootstrap_ranking(
            bootstrap_records=bootstrap_records,
            feature_ids=selected_payload["feature_ids"],
            main_importance=selected_payload["main_importance"],
        )
        feature_ranking = add_neuronpedia_urls(
            rows=feature_ranking[: int(config["stability"]["max_report_features"])],
            config=config,
            layer=selected_layer,
        )
        if bool(config["outputs"].get("save_probe_weights", True)):
            torch.save(
                {
                    "target_var": target_var,
                    "selected_layer": selected_layer,
                    "feature_ids": selected_payload["feature_ids"],
                    "weights": selected_fit.weights,
                    "bias": selected_fit.bias,
                    "best_record": selected_payload["best_record"],
                    "feature_metadata": selected_layer_summary.get("feature_metadata"),
                },
                output_dir / f"{target_var}_selected_probe.pt",
            )
        target_payloads[target_var] = {
            "target_var": target_var,
            "layer_results": states[target_var]["layer_results"],
            "selected_layer": selected_layer,
            "selected_probe": selected_payload["best_record"],
            "bootstrap_records": [
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"importance", "selected_columns"}
                }
                for record in bootstrap_records
            ],
            "feature_ranking": feature_ranking,
            "causal_validation": {
                "enabled": False,
                "reason": "Neuronpedia CLT probe runner trains probes only; no decoded interventions.",
            },
        }
        write_json(output_dir / f"{target_var}_results.json", target_payloads[target_var])
    return target_payloads


def main() -> None:
    start = perf_counter()
    args = build_parser().parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    config["causal_validation"] = {"enabled": False}
    results_timestamp = (
        config["outputs"].get("results_timestamp")
        or os.environ.get("RESULTS_TIMESTAMP")
        or "mcqa_neuronpedia_clt_probe"
    )
    output_dir = (
        Path(config["outputs"]["results_root"])
        / f"{results_timestamp}_mcqa_neuronpedia_clt_probe"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved_config.json", config)

    if config["features"]["token_position_id"] != "last_token":
        raise ValueError("Neuronpedia CLT probe runner only supports token_position_id=last_token.")
    if config["probe"]["modes"] != ["per_layer"]:
        raise ValueError("Neuronpedia CLT probe runner only supports probe.modes=[per_layer].")

    device_name = str(config["model"].get("device", "cuda")).lower()
    if device_name == "cuda":
        check_cuda_usable()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dataset_config = config["dataset"].get("dataset_config")
    causal_model = MCQACausalModel()
    log_progress(f"loading public MCQA datasets path={config['dataset']['dataset_path']}")
    datasets_by_name = load_public_mcqa_datasets(
        size=int(config["dataset"]["dataset_size"]),
        dataset_path=str(config["dataset"]["dataset_path"]),
        dataset_config=None if dataset_config is None else str(dataset_config),
        hf_token=hf_token,
    )
    if bool(config["dataset"].get("filter_model_correct", True)):
        from plot_mcqa_clt import filter_correct_examples_with_hf_model, load_filter_model_and_tokenizer

        log_progress("loading HF filter model for factual filtering")
        filter_model, filter_tokenizer = load_filter_model_and_tokenizer(
            model_name=str(config["model"].get("model_name", MODEL_NAME)),
            dtype_name=str(config["model"].get("dtype", "bf16")),
            hf_token=hf_token,
        )
        datasets_by_name = filter_correct_examples_with_hf_model(
            model=filter_model,
            tokenizer=filter_tokenizer,
            causal_model=causal_model,
            datasets_by_name=datasets_by_name,
            batch_size=int(config["dataset"].get("filter_batch_size", 32)),
        )
        del filter_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    records = build_prompt_records(
        datasets_by_name=datasets_by_name,
        include_base_prompts=bool(config["dataset"].get("include_base_prompts", True)),
        include_counterfactual_prompts=bool(config["dataset"].get("include_counterfactual_prompts", True)),
        counterfactual_families=tuple(config["dataset"].get("counterfactual_families", COUNTERFACTUAL_FAMILIES)),
        deduplicate_prompts=bool(config["dataset"].get("deduplicate_prompts", True)),
        causal_model=causal_model,
    )
    if not records:
        raise RuntimeError("No prompt records constructed for probe training.")
    splits = split_records_by_group(
        records=records,
        train_fraction=float(config["dataset"]["train_fraction"]),
        validation_fraction=float(config["dataset"]["validation_fraction"]),
        split_seed=int(config["dataset"]["split_seed"]),
    )
    log_progress(
        "probe records prepared "
        f"records={len(records)} train={len(splits.train)} val={len(splits.validation)} test={len(splits.test)}"
    )

    log_progress(
        "loading HookedTransformer "
        f"model={config['model'].get('model_name', MODEL_NAME)} device={device}"
    )
    model = load_hooked_transformer(
        model_name=str(config["model"].get("model_name", MODEL_NAME)),
        dtype_name=str(config["model"].get("dtype", "bf16")),
        device=device,
    )
    log_progress(f"probe training device={device}")
    layers = parse_csv_ints(str(config["features"]["layers"])) or tuple(range(26))
    targets = tuple(canonicalize_target_var(target) for target in config["labels"]["targets"])
    target_payloads = run_targets(
        records=records,
        targets=targets,
        splits=splits,
        layers=layers,
        model=model,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    final_payload = {
        "kind": "mcqa_neuronpedia_clt_probe",
        "config": config,
        "data": {
            "prompt_records": len(records),
            "splits": asdict(splits),
            "targets": list(targets),
            "layers": [int(layer) for layer in layers],
        },
        "targets": target_payloads,
        "runtime_seconds": float(perf_counter() - start),
    }
    write_json(output_dir / "mcqa_neuronpedia_clt_probe_results.json", final_payload)
    log_progress(f"wrote {output_dir / 'mcqa_neuronpedia_clt_probe_results.json'}")


if __name__ == "__main__":
    main()
