#!/usr/bin/env python
"""Train per-layer CLT linear probes for MCQA answer-pointer/token concepts."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from circuit_tracing_ot.config import MODEL_NAME, resolve_transcoder_set
from circuit_tracing_ot.logging import log_progress
from circuit_tracing_ot.mcqa_plot.clt_backend import CLTActivationCache, CLTSite
from circuit_tracing_ot.mcqa_plot.data import (
    ALPHABET_LABELS,
    COUNTERFACTUAL_FAMILIES,
    MCQACausalModel,
    build_pair_banks,
    canonicalize_target_var,
    get_token_positions,
    load_public_mcqa_datasets,
)
from circuit_tracing_ot.mcqa_plot.decoded_interventions import run_clt_decoded_site_intervention
from circuit_tracing_ot.mcqa_plot.metrics import metrics_from_logits, prediction_details_from_logits


@dataclass(frozen=True)
class PromptRecord:
    prompt: str
    group_key: str
    source_name: str
    answer_pointer: int
    answer_token_index: int
    answer_token: str
    input_dict: dict[str, object]


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    validation: list[int]
    test: list[int]


@dataclass(frozen=True)
class ProbeFit:
    weights: torch.Tensor
    bias: torch.Tensor
    validation_macro_accuracy: float
    validation_accuracy: float
    validation_loss: float
    epochs_trained: int


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


def parse_csv_strings(value: str | None) -> tuple[str, ...] | None:
    if value is None or not str(value).strip():
        return None
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mcqa_clt_probe.yaml"))
    parser.add_argument("--dataset-size", type=int)
    parser.add_argument("--layers")
    parser.add_argument("--targets")
    parser.add_argument("--activation-top-k", type=int)
    parser.add_argument("--feature-cap-per-layer", type=int)
    parser.add_argument("--num-bootstraps", type=int)
    parser.add_argument("--results-timestamp")
    parser.add_argument("--skip-causal-validation", action="store_true")
    return parser


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if args.dataset_size is not None:
        updates.setdefault("dataset", {})["dataset_size"] = int(args.dataset_size)
    if args.layers is not None:
        updates.setdefault("features", {})["layers"] = str(args.layers)
    if args.targets is not None:
        updates.setdefault("labels", {})["targets"] = list(parse_csv_strings(args.targets) or ())
    if args.activation_top_k is not None:
        updates.setdefault("features", {})["activation_top_k"] = int(args.activation_top_k)
    if args.feature_cap_per_layer is not None:
        updates.setdefault("features", {})["feature_cap_per_layer"] = int(args.feature_cap_per_layer)
    if args.num_bootstraps is not None:
        updates.setdefault("stability", {})["num_bootstraps"] = int(args.num_bootstraps)
    if args.results_timestamp is not None:
        updates.setdefault("outputs", {})["results_timestamp"] = str(args.results_timestamp)
    if args.skip_causal_validation:
        updates.setdefault("causal_validation", {})["enabled"] = False
    return deep_update(config, updates)


def _row_group_key(row: dict[str, object]) -> str:
    return str(row["input"]["raw_input"])


def build_prompt_records(
    *,
    datasets_by_name: dict[str, list[dict[str, object]]],
    include_base_prompts: bool,
    include_counterfactual_prompts: bool,
    counterfactual_families: tuple[str, ...],
    deduplicate_prompts: bool,
    causal_model: MCQACausalModel,
) -> list[PromptRecord]:
    records_by_prompt: dict[str, PromptRecord] = {}
    records: list[PromptRecord] = []

    def add_record(*, input_dict: dict[str, object], group_key: str, source_name: str) -> None:
        output = causal_model.run_forward(input_dict)
        answer_token = str(output["answer"]).strip()
        record = PromptRecord(
            prompt=str(input_dict["raw_input"]),
            group_key=str(group_key),
            source_name=str(source_name),
            answer_pointer=int(output["answer_pointer"]),
            answer_token_index=int(ALPHABET_LABELS.index(answer_token)),
            answer_token=answer_token,
            input_dict=input_dict,
        )
        if deduplicate_prompts:
            records_by_prompt.setdefault(record.prompt, record)
        else:
            records.append(record)

    for dataset_name, rows in sorted(datasets_by_name.items()):
        counterfactual_name, _, _split_name = dataset_name.rpartition("_")
        if counterfactual_name not in counterfactual_families:
            continue
        for row in rows:
            group_key = _row_group_key(row)
            if include_base_prompts:
                add_record(
                    input_dict=row["input"],
                    group_key=group_key,
                    source_name=f"{dataset_name}:base",
                )
            if include_counterfactual_prompts:
                add_record(
                    input_dict=row["counterfactual_inputs"][0],
                    group_key=group_key,
                    source_name=f"{dataset_name}:counterfactual",
                )
    if deduplicate_prompts:
        records = list(records_by_prompt.values())
    records.sort(key=lambda record: (record.group_key, record.prompt))
    return records


def split_records_by_group(
    *,
    records: list[PromptRecord],
    train_fraction: float,
    validation_fraction: float,
    split_seed: int,
) -> SplitIndices:
    groups = sorted({record.group_key for record in records})
    rng = random.Random(int(split_seed))
    rng.shuffle(groups)
    n_groups = len(groups)
    n_train = int(round(float(train_fraction) * n_groups))
    n_validation = int(round(float(validation_fraction) * n_groups))
    n_train = min(n_train, n_groups)
    n_validation = min(n_validation, max(0, n_groups - n_train))
    train_groups = set(groups[:n_train])
    validation_groups = set(groups[n_train : n_train + n_validation])
    test_groups = set(groups[n_train + n_validation :])
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for index, record in enumerate(records):
        if record.group_key in train_groups:
            train.append(index)
        elif record.group_key in validation_groups:
            validation.append(index)
        elif record.group_key in test_groups:
            test.append(index)
    return SplitIndices(train=train, validation=validation, test=test)


def target_labels(records: list[PromptRecord], target_var: str) -> torch.Tensor:
    canonical = canonicalize_target_var(target_var)
    if canonical == "answer_pointer":
        return torch.tensor([record.answer_pointer for record in records], dtype=torch.long)
    if canonical == "answer_token":
        return torch.tensor([record.answer_token_index for record in records], dtype=torch.long)
    raise ValueError(f"Unsupported target var {target_var}")


def num_classes_for_target(target_var: str) -> int:
    canonical = canonicalize_target_var(target_var)
    return 4 if canonical == "answer_pointer" else len(ALPHABET_LABELS)


def last_token_positions(records: list[PromptRecord], tokenizer) -> list[int]:
    positions = []
    for record in records:
        encoded = tokenizer(
            record.prompt,
            add_special_tokens=True,
            return_attention_mask=False,
        )["input_ids"]
        positions.append(len(encoded) - 1)
    return positions


def drop_cached_activation(
    *,
    cache: CLTActivationCache,
    prompt: str,
    layer: int,
    position: int,
    top_k: int | None,
) -> None:
    """Drop per-prompt CLT caches after extraction to avoid GPU memory growth."""
    payload_cache = getattr(cache, "_payload_by_prompt", None)
    if isinstance(payload_cache, dict):
        payload_cache.pop(prompt, None)
    values_cache = getattr(cache, "_values_by_key", None)
    if isinstance(values_cache, dict):
        values_cache.pop((prompt, int(layer), int(position), None if top_k is None else int(top_k)), None)


def screen_layer_features(
    *,
    records: list[PromptRecord],
    positions: list[int],
    layer: int,
    activation_top_k: int,
    feature_cap: int,
    cache: CLTActivationCache,
) -> list[int]:
    counts: Counter[int] = Counter()
    start = perf_counter()
    for row_index, record in enumerate(records):
        if row_index == 0 or row_index == len(records) - 1 or (row_index + 1) % 100 == 0:
            log_progress(
                "screening CLT features "
                f"layer={layer} row={row_index + 1}/{len(records)} elapsed={perf_counter() - start:.1f}s"
            )
        values = cache.values(
            prompt=record.prompt,
            layer=int(layer),
            position=int(positions[row_index]),
            top_k=int(activation_top_k),
        )
        counts.update(int(value.feature_idx) for value in values)
        drop_cached_activation(
            cache=cache,
            prompt=record.prompt,
            layer=int(layer),
            position=int(positions[row_index]),
            top_k=int(activation_top_k),
        )
        if torch.cuda.is_available() and (row_index + 1) % 100 == 0:
            torch.cuda.empty_cache()
    ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), int(item[0])))
    return [int(feature_idx) for feature_idx, _count in ranked[: int(feature_cap)]]


def collect_layer_rows(
    *,
    records: list[PromptRecord],
    positions: list[int],
    layer: int,
    feature_ids: list[int],
    activation_top_k: int,
    cache: CLTActivationCache,
) -> list[dict[int, float]]:
    feature_to_col = {int(feature_idx): col for col, feature_idx in enumerate(feature_ids)}
    rows: list[dict[int, float]] = []
    start = perf_counter()
    for row_index, record in enumerate(records):
        if row_index == 0 or row_index == len(records) - 1 or (row_index + 1) % 100 == 0:
            log_progress(
                "collecting layer matrix rows "
                f"layer={layer} row={row_index + 1}/{len(records)} elapsed={perf_counter() - start:.1f}s"
            )
        values = cache.values(
            prompt=record.prompt,
            layer=int(layer),
            position=int(positions[row_index]),
            top_k=int(activation_top_k),
        )
        row: dict[int, float] = {}
        for value in values:
            col = feature_to_col.get(int(value.feature_idx))
            if col is not None:
                row[int(col)] = float(value.value)
        rows.append(row)
        drop_cached_activation(
            cache=cache,
            prompt=record.prompt,
            layer=int(layer),
            position=int(positions[row_index]),
            top_k=int(activation_top_k),
        )
        if torch.cuda.is_available() and (row_index + 1) % 100 == 0:
            torch.cuda.empty_cache()
    return rows


def standardization_stats(
    rows: list[dict[int, float]],
    train_indices: list[int],
    num_features: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.zeros(num_features, dtype=torch.float32)
    sums_sq = torch.zeros(num_features, dtype=torch.float32)
    for index in train_indices:
        for col, value in rows[index].items():
            value_float = float(value)
            sums[int(col)] += value_float
            sums_sq[int(col)] += value_float * value_float
    count = max(1, len(train_indices))
    mean = sums / count
    variance = torch.clamp(sums_sq / count - mean.square(), min=0.0)
    std = torch.sqrt(variance + float(eps))
    return mean, std


def make_dense_batch(
    rows: list[dict[int, float]],
    indices: list[int],
    num_features: int,
    *,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    if mean is not None and std is not None:
        batch = (-mean / std).repeat(len(indices), 1)
        for row_offset, index in enumerate(indices):
            for col, value in rows[index].items():
                batch[row_offset, int(col)] = (float(value) - mean[int(col)]) / std[int(col)]
    else:
        batch = torch.zeros((len(indices), num_features), dtype=torch.float32)
        for row_offset, index in enumerate(indices):
            for col, value in rows[index].items():
                batch[row_offset, int(col)] = float(value)
    return batch.to(device)


def macro_accuracy(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    accuracies = []
    for class_index in range(int(num_classes)):
        mask = labels == int(class_index)
        if bool(mask.any()):
            accuracies.append((predictions[mask] == labels[mask]).float().mean())
    if not accuracies:
        return 0.0
    return float(torch.stack(accuracies).mean().item())


def class_weights(labels: torch.Tensor, indices: list[int], num_classes: int) -> torch.Tensor:
    selected = labels[torch.tensor(indices, dtype=torch.long)]
    counts = torch.bincount(selected, minlength=int(num_classes)).float()
    weights = torch.zeros(int(num_classes), dtype=torch.float32)
    nonzero = counts > 0
    weights[nonzero] = selected.numel() / (float(num_classes) * counts[nonzero])
    weights[~nonzero] = 0.0
    return weights


def evaluate_probe(
    *,
    weights: torch.Tensor,
    bias: torch.Tensor,
    rows: list[dict[int, float]],
    labels: torch.Tensor,
    indices: list[int],
    num_features: int,
    num_classes: int,
    batch_size: int,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    device: torch.device,
) -> dict[str, float]:
    weights = weights.to(device)
    bias = bias.to(device)
    all_predictions = []
    total_loss = 0.0
    total_count = 0
    with torch.inference_mode():
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            x = make_dense_batch(
                rows,
                batch_indices,
                num_features,
                mean=mean,
                std=std,
                device=device,
            )
            y = labels[torch.tensor(batch_indices, dtype=torch.long)].to(device)
            logits = x @ weights.T + bias
            total_loss += float(F.cross_entropy(logits, y, reduction="sum").item())
            total_count += int(y.numel())
            all_predictions.append(logits.argmax(dim=-1).detach().cpu())
    predictions = torch.cat(all_predictions, dim=0) if all_predictions else torch.empty(0, dtype=torch.long)
    y_cpu = labels[torch.tensor(indices, dtype=torch.long)] if indices else torch.empty(0, dtype=torch.long)
    accuracy = float((predictions == y_cpu).float().mean().item()) if len(indices) else 0.0
    return {
        "accuracy": accuracy,
        "macro_accuracy": macro_accuracy(predictions, y_cpu, num_classes),
        "loss": total_loss / max(1, total_count),
    }


def train_linear_probe(
    *,
    rows: list[dict[int, float]],
    labels: torch.Tensor,
    train_indices: list[int],
    validation_indices: list[int],
    num_features: int,
    num_classes: int,
    l1_lambda: float,
    l2_lambda: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    early_stopping_patience: int,
    balanced_classes: bool,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    seed: int,
    device: torch.device,
) -> ProbeFit:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    linear = torch.nn.Linear(num_features, num_classes)
    torch.nn.init.zeros_(linear.weight)
    torch.nn.init.zeros_(linear.bias)
    linear.to(device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=float(learning_rate), weight_decay=0.0)
    ce_weight = (
        class_weights(labels, train_indices, num_classes).to(device)
        if balanced_classes
        else None
    )
    best_state = copy.deepcopy(linear.state_dict())
    best_metrics = {"macro_accuracy": -1.0, "accuracy": 0.0, "loss": math.inf}
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, int(epochs) + 1):
        order = torch.randperm(len(train_indices), generator=generator).tolist()
        linear.train()
        for start in range(0, len(order), int(batch_size)):
            batch_positions = order[start : start + int(batch_size)]
            batch_indices = [train_indices[position] for position in batch_positions]
            x = make_dense_batch(
                rows,
                batch_indices,
                num_features,
                mean=mean,
                std=std,
                device=device,
            )
            y = labels[torch.tensor(batch_indices, dtype=torch.long)].to(device)
            logits = linear(x)
            loss = F.cross_entropy(logits, y, weight=ce_weight)
            if float(l2_lambda) != 0.0:
                loss = loss + float(l2_lambda) * linear.weight.square().sum()
            if float(l1_lambda) != 0.0:
                loss = loss + float(l1_lambda) * linear.weight.abs().sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        linear.eval()
        metrics = evaluate_probe(
            weights=linear.weight.detach().cpu(),
            bias=linear.bias.detach().cpu(),
            rows=rows,
            labels=labels,
            indices=validation_indices,
            num_features=num_features,
            num_classes=num_classes,
            batch_size=batch_size,
            mean=mean,
            std=std,
            device=device,
        )
        if (
            metrics["macro_accuracy"],
            metrics["accuracy"],
            -metrics["loss"],
        ) > (
            best_metrics["macro_accuracy"],
            best_metrics["accuracy"],
            -best_metrics["loss"],
        ):
            best_metrics = metrics
            best_state = copy.deepcopy(linear.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(early_stopping_patience):
                break
    linear.load_state_dict(best_state)
    return ProbeFit(
        weights=linear.weight.detach().cpu().clone(),
        bias=linear.bias.detach().cpu().clone(),
        validation_macro_accuracy=float(best_metrics["macro_accuracy"]),
        validation_accuracy=float(best_metrics["accuracy"]),
        validation_loss=float(best_metrics["loss"]),
        epochs_trained=int(best_epoch),
    )


def centered_column_l2(weights: torch.Tensor) -> torch.Tensor:
    centered = weights - weights.mean(dim=0, keepdim=True)
    return torch.linalg.vector_norm(centered, ord=2, dim=0)


def feature_records_from_importance(
    *,
    feature_ids: list[int],
    importance: torch.Tensor,
    limit: int,
) -> list[dict[str, object]]:
    ranking = sorted(
        range(len(feature_ids)),
        key=lambda col: (-float(importance[int(col)].item()), int(feature_ids[int(col)])),
    )
    return [
        {
            "rank": rank + 1,
            "feature_idx": int(feature_ids[col]),
            "column": int(col),
            "importance": float(importance[col].item()),
        }
        for rank, col in enumerate(ranking[: int(limit)])
    ]


def run_bootstraps(
    *,
    rows: list[dict[int, float]],
    labels: torch.Tensor,
    train_indices: list[int],
    validation_indices: list[int],
    feature_ids: list[int],
    num_classes: int,
    selected_fit: dict[str, object],
    config: dict[str, Any],
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    device: torch.device,
    seed: int,
) -> list[dict[str, object]]:
    stability = config["stability"]
    probe_config = config["probe"]
    num_bootstraps = int(stability["num_bootstraps"])
    train_fraction = float(stability["bootstrap_train_fraction"])
    top_q = int(stability["selected_feature_top_q"])
    bootstrap_records: list[dict[str, object]] = []
    rng = random.Random(int(seed))
    for bootstrap_index in range(num_bootstraps):
        sample_size = max(1, int(round(train_fraction * len(train_indices))))
        bootstrap_train = [rng.choice(train_indices) for _ in range(sample_size)]
        log_progress(
            "bootstrap probe fit "
            f"{bootstrap_index + 1}/{num_bootstraps} sample_size={sample_size}"
        )
        fit = train_linear_probe(
            rows=rows,
            labels=labels,
            train_indices=bootstrap_train,
            validation_indices=validation_indices,
            num_features=len(feature_ids),
            num_classes=num_classes,
            l1_lambda=float(selected_fit["l1_lambda"]),
            l2_lambda=float(selected_fit["l2_lambda"]),
            learning_rate=float(probe_config["learning_rate"]),
            batch_size=int(probe_config["batch_size"]),
            epochs=int(probe_config["epochs"]),
            early_stopping_patience=int(probe_config["early_stopping_patience"]),
            balanced_classes=str(probe_config.get("class_weighting", "balanced")) == "balanced",
            mean=mean,
            std=std,
            seed=int(seed) + bootstrap_index + 1,
            device=device,
        )
        importance = centered_column_l2(fit.weights)
        top_count = min(top_q, len(feature_ids))
        top_cols = torch.topk(importance, k=top_count).indices.tolist() if top_count else []
        bootstrap_records.append(
            {
                "bootstrap_index": int(bootstrap_index),
                "validation_macro_accuracy": float(fit.validation_macro_accuracy),
                "validation_accuracy": float(fit.validation_accuracy),
                "validation_loss": float(fit.validation_loss),
                "epochs_trained": int(fit.epochs_trained),
                "selected_columns": [int(col) for col in top_cols],
                "importance": importance.tolist(),
            }
        )
    return bootstrap_records


def aggregate_bootstrap_ranking(
    *,
    bootstrap_records: list[dict[str, object]],
    feature_ids: list[int],
    main_importance: torch.Tensor,
) -> list[dict[str, object]]:
    selected_counts = [0 for _ in feature_ids]
    importance_sums = [0.0 for _ in feature_ids]
    for record in bootstrap_records:
        for col in record["selected_columns"]:
            selected_counts[int(col)] += 1
        for col, value in enumerate(record["importance"]):
            importance_sums[int(col)] += float(value)
    denominator = max(1, len(bootstrap_records))
    rows = []
    for col, feature_idx in enumerate(feature_ids):
        rows.append(
            {
                "feature_idx": int(feature_idx),
                "column": int(col),
                "selection_frequency": float(selected_counts[col] / denominator),
                "mean_centered_column_l2": float(importance_sums[col] / denominator),
                "main_centered_column_l2": float(main_importance[col].item()),
            }
        )
    rows.sort(
        key=lambda item: (
            -float(item["selection_frequency"]),
            -float(item["mean_centered_column_l2"]),
            -float(item["main_centered_column_l2"]),
            int(item["feature_idx"]),
        )
    )
    for rank, item in enumerate(rows, start=1):
        item["rank"] = int(rank)
    return rows


def evaluate_feature_set_intervention(
    *,
    model,
    bank,
    layer: int,
    feature_indices: list[int],
    token_position_id: str,
    activation_top_k: int,
    strength: float,
    cache: CLTActivationCache,
    tokenizer,
    include_details: bool,
) -> dict[str, object]:
    sites = [
        CLTSite(
            layer=int(layer),
            write_layer=int(layer),
            token_position_id=str(token_position_id),
            feature_idx=int(feature_idx),
            top_features=int(activation_top_k),
        )
        for feature_idx in feature_indices
    ]
    logits = run_clt_decoded_site_intervention(
        model=model,
        bank=bank,
        site_weights={site: 1.0 for site in sites},
        strength=float(strength),
        cache=cache,
        log_context=f"probe_causal_validation:k{len(feature_indices)}",
    )
    record = {
        "layer": int(layer),
        "write_layer": int(layer),
        "feature_indices": [int(feature_idx) for feature_idx in feature_indices],
        "feature_set_size": int(len(feature_indices)),
        "intervention_strength": float(strength),
        "selected_site_labels": [site.label for site in sites],
        **metrics_from_logits(logits, bank, tokenizer=tokenizer),
    }
    if include_details:
        record["prediction_details"] = prediction_details_from_logits(logits, bank, tokenizer=tokenizer)
    return record


def run_causal_validation(
    *,
    model,
    tokenizer,
    datasets_by_name: dict[str, list[dict[str, object]]],
    target_var: str,
    selected_layer: int,
    feature_ranking: list[dict[str, object]],
    config: dict[str, Any],
    cache: CLTActivationCache,
) -> dict[str, object]:
    causal_config = config["causal_validation"]
    if not bool(causal_config.get("enabled", True)):
        return {"enabled": False}
    causal_model = MCQACausalModel()
    token_positions = get_token_positions(tokenizer, causal_model)
    calibration_pool_size = int(causal_config["calibration_pool_size"])
    test_pool_size = int(causal_config["test_pool_size"])
    banks_by_split, data_metadata = build_pair_banks(
        tokenizer=tokenizer,
        causal_model=causal_model,
        token_positions=token_positions,
        datasets_by_name=datasets_by_name,
        counterfactual_names=tuple(config["dataset"]["counterfactual_families"]),
        target_vars=(target_var,),
        split_seed=int(config["dataset"]["split_seed"]),
        train_pool_size=int(causal_config.get("train_pool_size", 100)),
        calibration_pool_size=calibration_pool_size,
        test_pool_size=test_pool_size,
    )
    calibration_bank = banks_by_split["calibration"][target_var]
    test_bank = banks_by_split["test"][target_var]
    sweep = []
    feature_set_sizes = [int(value) for value in causal_config["feature_set_sizes"]]
    strengths = [float(value) for value in causal_config["intervention_strengths"]]
    ranked_feature_indices = [int(item["feature_idx"]) for item in feature_ranking]
    for feature_set_size in feature_set_sizes:
        feature_indices = ranked_feature_indices[: int(feature_set_size)]
        if not feature_indices:
            continue
        for strength in strengths:
            log_progress(
                "causal validation calibration "
                f"target={target_var} layer={selected_layer} k={feature_set_size} strength={strength:g}"
            )
            result = evaluate_feature_set_intervention(
                model=model,
                bank=calibration_bank,
                layer=int(selected_layer),
                feature_indices=feature_indices,
                token_position_id=str(config["features"]["token_position_id"]),
                activation_top_k=int(config["features"]["activation_top_k"]),
                strength=float(strength),
                cache=cache,
                tokenizer=tokenizer,
                include_details=False,
            )
            sweep.append(result)
    selected = max(
        sweep,
        key=lambda item: (
            float(item["exact_acc"]),
            -int(item["feature_set_size"]),
            -abs(float(item["intervention_strength"]) - 1.0),
        ),
    )
    log_progress(
        "causal validation selected "
        f"target={target_var} k={selected['feature_set_size']} "
        f"strength={float(selected['intervention_strength']):g} "
        f"calibration_exact_acc={float(selected['exact_acc']):.4f}"
    )
    test_result = evaluate_feature_set_intervention(
        model=model,
        bank=test_bank,
        layer=int(selected_layer),
        feature_indices=[int(value) for value in selected["feature_indices"]],
        token_position_id=str(config["features"]["token_position_id"]),
        activation_top_k=int(config["features"]["activation_top_k"]),
        strength=float(selected["intervention_strength"]),
        cache=cache,
        tokenizer=tokenizer,
        include_details=bool(config["outputs"].get("save_prediction_details", False)),
    )
    return {
        "enabled": True,
        "data": data_metadata,
        "calibration_sweep": sweep,
        "selected_hyperparameters": {
            "feature_set_size": int(selected["feature_set_size"]),
            "intervention_strength": float(selected["intervention_strength"]),
            "write_layer_mode": str(causal_config.get("write_layer_mode", "same")),
        },
        "selected_calibration_result": selected,
        "test_result": test_result,
    }


def layer_probe_grid(
    *,
    records: list[PromptRecord],
    positions: list[int],
    target_var: str,
    layer: int,
    labels: torch.Tensor,
    splits: SplitIndices,
    cache: CLTActivationCache,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, object]:
    feature_config = config["features"]
    probe_config = config["probe"]
    feature_ids = screen_layer_features(
        records=records,
        positions=positions,
        layer=int(layer),
        activation_top_k=int(feature_config["activation_top_k"]),
        feature_cap=int(feature_config["feature_cap_per_layer"]),
        cache=cache,
    )
    if not feature_ids:
        raise RuntimeError(f"No screened features for layer {layer}")
    rows = collect_layer_rows(
        records=records,
        positions=positions,
        layer=int(layer),
        feature_ids=feature_ids,
        activation_top_k=int(feature_config["activation_top_k"]),
        cache=cache,
    )
    mean = std = None
    if bool(feature_config.get("standardize_features", True)):
        mean, std = standardization_stats(
            rows,
            splits.train,
            len(feature_ids),
            eps=float(feature_config.get("standardization_eps", 1.0e-6)),
        )
    num_classes = num_classes_for_target(target_var)
    grid_records = []
    best = None
    for l1_lambda in probe_config["l1_lambdas"]:
        for l2_lambda in probe_config["l2_lambdas"]:
            log_progress(
                "training probe "
                f"target={target_var} layer={layer} l1={float(l1_lambda):g} l2={float(l2_lambda):g}"
            )
            fit = train_linear_probe(
                rows=rows,
                labels=labels,
                train_indices=splits.train,
                validation_indices=splits.validation,
                num_features=len(feature_ids),
                num_classes=num_classes,
                l1_lambda=float(l1_lambda),
                l2_lambda=float(l2_lambda),
                learning_rate=float(probe_config["learning_rate"]),
                batch_size=int(probe_config["batch_size"]),
                epochs=int(probe_config["epochs"]),
                early_stopping_patience=int(probe_config["early_stopping_patience"]),
                balanced_classes=str(probe_config.get("class_weighting", "balanced")) == "balanced",
                mean=mean,
                std=std,
                seed=int(config["dataset"]["split_seed"]) + int(layer) * 1009,
                device=device,
            )
            test_metrics = evaluate_probe(
                weights=fit.weights,
                bias=fit.bias,
                rows=rows,
                labels=labels,
                indices=splits.test,
                num_features=len(feature_ids),
                num_classes=num_classes,
                batch_size=int(probe_config["batch_size"]),
                mean=mean,
                std=std,
                device=device,
            )
            record = {
                "layer": int(layer),
                "target_var": target_var,
                "l1_lambda": float(l1_lambda),
                "l2_lambda": float(l2_lambda),
                "validation_macro_accuracy": float(fit.validation_macro_accuracy),
                "validation_accuracy": float(fit.validation_accuracy),
                "validation_loss": float(fit.validation_loss),
                "test_macro_accuracy": float(test_metrics["macro_accuracy"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_loss": float(test_metrics["loss"]),
                "epochs_trained": int(fit.epochs_trained),
            }
            grid_records.append(record)
            if best is None or (
                record["validation_macro_accuracy"],
                record["validation_accuracy"],
                -record["validation_loss"],
            ) > (
                best["record"]["validation_macro_accuracy"],
                best["record"]["validation_accuracy"],
                -best["record"]["validation_loss"],
            ):
                best = {"record": record, "fit": fit}
    assert best is not None
    importance = centered_column_l2(best["fit"].weights)
    top_features = feature_records_from_importance(
        feature_ids=feature_ids,
        importance=importance,
        limit=int(config["stability"]["max_report_features"]),
    )
    return {
        "layer": int(layer),
        "target_var": target_var,
        "feature_ids": feature_ids,
        "rows": rows,
        "mean": mean,
        "std": std,
        "grid_records": grid_records,
        "best_record": best["record"],
        "best_fit": best["fit"],
        "main_importance": importance,
        "top_features_by_main_probe": top_features,
    }


def run_target(
    *,
    records: list[PromptRecord],
    positions: list[int],
    target_var: str,
    splits: SplitIndices,
    layers: tuple[int, ...],
    model,
    tokenizer,
    datasets_by_name: dict[str, list[dict[str, object]]],
    cache: CLTActivationCache,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, object]:
    labels = target_labels(records, target_var)
    layer_results = []
    selected_layer_summary = None
    selected_payload = None
    for layer in layers:
        log_progress(f"target={target_var} layer={layer} probe grid start")
        layer_payload = layer_probe_grid(
            records=records,
            positions=positions,
            target_var=target_var,
            layer=int(layer),
            labels=labels,
            splits=splits,
            cache=cache,
            config=config,
            device=device,
        )
        layer_results.append(
            {
                "layer": int(layer),
                "best_record": layer_payload["best_record"],
                "top_features_by_main_probe": layer_payload["top_features_by_main_probe"],
                "screened_feature_count": len(layer_payload["feature_ids"]),
            }
        )
        write_json(
            output_dir / f"{target_var}_layer_{int(layer)}_summary.json",
            layer_results[-1],
        )
        candidate_key = (
            float(layer_results[-1]["best_record"]["validation_macro_accuracy"]),
            float(layer_results[-1]["best_record"]["validation_accuracy"]),
            -float(layer_results[-1]["best_record"]["validation_loss"]),
            -int(layer_results[-1]["layer"]),
        )
        best_key = (
            -1.0,
            -1.0,
            -math.inf,
            -10**9,
        )
        if selected_layer_summary is not None:
            best_key = (
                float(selected_layer_summary["best_record"]["validation_macro_accuracy"]),
                float(selected_layer_summary["best_record"]["validation_accuracy"]),
                -float(selected_layer_summary["best_record"]["validation_loss"]),
                -int(selected_layer_summary["layer"]),
            )
        if selected_layer_summary is None or candidate_key > best_key:
            selected_layer_summary = layer_results[-1]
            selected_payload = layer_payload
            log_progress(
                "selected layer candidate updated "
                f"target={target_var} layer={int(layer)} "
                f"val_macro={float(selected_layer_summary['best_record']['validation_macro_accuracy']):.4f}"
            )
        else:
            del layer_payload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if selected_layer_summary is None or selected_payload is None:
        raise RuntimeError(f"No selected layer payload for target={target_var}")
    selected_layer = int(selected_layer_summary["layer"])
    selected_fit = selected_payload["best_fit"]
    bootstrap_records = run_bootstraps(
        rows=selected_payload["rows"],
        labels=labels,
        train_indices=splits.train,
        validation_indices=splits.validation,
        feature_ids=selected_payload["feature_ids"],
        num_classes=num_classes_for_target(target_var),
        selected_fit=selected_payload["best_record"],
        config=config,
        mean=selected_payload["mean"],
        std=selected_payload["std"],
        device=device,
        seed=int(config["dataset"]["split_seed"]) + selected_layer * 9973,
    )
    feature_ranking = aggregate_bootstrap_ranking(
        bootstrap_records=bootstrap_records,
        feature_ids=selected_payload["feature_ids"],
        main_importance=selected_payload["main_importance"],
    )
    causal_validation = run_causal_validation(
        model=model,
        tokenizer=tokenizer,
        datasets_by_name=datasets_by_name,
        target_var=target_var,
        selected_layer=selected_layer,
        feature_ranking=feature_ranking,
        config=config,
        cache=cache,
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
            },
            output_dir / f"{target_var}_selected_probe.pt",
        )
    return {
        "target_var": target_var,
        "layer_results": layer_results,
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
        "feature_ranking": feature_ranking[: int(config["stability"]["max_report_features"])],
        "causal_validation": causal_validation,
    }


def main() -> None:
    start = perf_counter()
    args = build_parser().parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    results_timestamp = (
        config["outputs"].get("results_timestamp")
        or os.environ.get("RESULTS_TIMESTAMP")
        or "mcqa_clt_probe"
    )
    output_dir = Path(config["outputs"]["results_root"]) / f"{results_timestamp}_mcqa_clt_probe"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved_config.json", config)

    if config["features"]["token_position_id"] != "last_token":
        raise ValueError("This first probe runner only supports token_position_id=last_token.")
    if config["probe"]["modes"] != ["per_layer"]:
        raise ValueError("This first probe runner only supports probe.modes=[per_layer].")
    if config["causal_validation"].get("write_layer_mode", "same") != "same":
        raise ValueError("This first probe runner only supports causal_validation.write_layer_mode=same.")

    from circuit_tracing_ot.model import load_replacement_model

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

    transcoder_set = resolve_transcoder_set(
        config["model"].get("transcoder_set"),
        config["model"].get("transcoder_size"),
    )
    log_progress(f"loading ReplacementModel transcoder_set={transcoder_set}")
    model = load_replacement_model(
        model_name=str(config["model"].get("model_name", MODEL_NAME)),
        transcoder_set=transcoder_set,
        dtype_name=str(config["model"].get("dtype", "bf16")),
        offload=config["model"].get("offload"),
        backend=config["model"].get("backend"),
    )
    if getattr(model.tokenizer, "pad_token", None) is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"
    positions = last_token_positions(records, model.tokenizer)
    cache = CLTActivationCache(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = parse_csv_ints(str(config["features"]["layers"])) or tuple(range(26))
    targets = tuple(canonicalize_target_var(target) for target in config["labels"]["targets"])

    target_payloads = {}
    for target_var in targets:
        target_payloads[target_var] = run_target(
            records=records,
            positions=positions,
            target_var=target_var,
            splits=splits,
            layers=layers,
            model=model,
            tokenizer=model.tokenizer,
            datasets_by_name=datasets_by_name,
            cache=cache,
            config=config,
            device=device,
            output_dir=output_dir,
        )
        write_json(output_dir / f"{target_var}_results.json", target_payloads[target_var])

    final_payload = {
        "kind": "mcqa_clt_probe",
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
    write_json(output_dir / "mcqa_clt_probe_results.json", final_payload)
    log_progress(f"wrote {output_dir / 'mcqa_clt_probe_results.json'}")


if __name__ == "__main__":
    main()
