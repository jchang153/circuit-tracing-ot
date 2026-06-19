#!/usr/bin/env python
"""Screen CLT features by source-base activation deltas on MCQA counterfactuals."""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from circuit_tracing_ot.config import MODEL_NAME, resolve_transcoder_set
from circuit_tracing_ot.logging import log_progress
from circuit_tracing_ot.mcqa_plot.clt_backend import (
    CLTActivationCache,
    CLTSite,
    _should_log_progress,
)
from circuit_tracing_ot.mcqa_plot.data import (
    ALPHABET_LABELS,
    COUNTERFACTUAL_FAMILIES,
    MCQACausalModel,
    MCQAPairBank,
    TokenPosition,
    _alphabet_index,
    _encode_symbol_token,
    _encode_symbol_token_variants,
    _validate_answer_tokenization,
    get_token_positions,
    load_public_mcqa_datasets,
)
from circuit_tracing_ot.mcqa_plot.decoded_interventions import run_clt_decoded_site_intervention
from circuit_tracing_ot.mcqa_plot.metrics import metrics_from_logits, prediction_details_from_logits
from plot_mcqa_clt import (
    filter_correct_examples_with_hf_model,
    load_filter_model_and_tokenizer,
    parse_csv_ints,
    parse_csv_strings,
    write_json,
)


DEFAULT_COUNTERFACTUAL_NAMES = COUNTERFACTUAL_FAMILIES
DEFAULT_TOKEN_POSITION_ID = "last_token"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="jchang153/copycolors_mcqa")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-size", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--examples-per-family", type=int, default=100)
    parser.add_argument("--counterfactual-names", default=",".join(DEFAULT_COUNTERFACTUAL_NAMES))
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--transcoder-size", default="426k", choices=("426k", "2.5m"))
    parser.add_argument("--transcoder-set", default=None)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--offload", default=None, choices=(None, "cpu", "disk"))
    parser.add_argument("--backend", default=None, choices=("nnsight", "transformerlens"))
    parser.add_argument(
        "--layers",
        help="Comma-separated layer indices/ranges. Default: all model layers.",
    )
    parser.add_argument(
        "--token-position-id",
        default=DEFAULT_TOKEN_POSITION_ID,
        choices=(DEFAULT_TOKEN_POSITION_ID,),
    )
    parser.add_argument("--screen-top-k-per-layer", type=int, default=2048)
    parser.add_argument("--num-candidate-features", type=int, default=128)
    parser.add_argument("--filter-batch-size", type=int, default=32)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--results-timestamp")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    parser.add_argument(
        "--clt-intervention-mode",
        default="decoded_mlp",
        choices=("decoded_mlp",),
        help=(
            "Only decoded_mlp same-layer interventions are supported by this "
            "screening experiment."
        ),
    )
    parser.add_argument(
        "--clt-write-layer-mode",
        default="same",
        choices=("same",),
        help="Feature-screening interventions are same-layer only.",
    )
    parser.add_argument(
        "--store-prediction-details",
        action="store_true",
        help=(
            "Store per-example prediction details for every evaluated feature. "
            "This can make JSON large."
        ),
    )
    return parser


def _model_num_layers(model: Any) -> int:
    cfg = getattr(model, "cfg", None)
    if cfg is not None and getattr(cfg, "n_layers", None) is not None:
        return int(cfg.n_layers)
    if hasattr(model, "blocks"):
        return len(model.blocks)
    return 26


def _row_family(row: dict[str, object]) -> str:
    return str(row["counterfactual_family"])


def _source_answer_changed(row: dict[str, object], causal_model: MCQACausalModel) -> bool:
    base_output = causal_model.run_forward(row["input"])
    source_output = causal_model.run_forward(row["counterfactual_inputs"][0])
    return str(base_output["answer"]) != str(source_output["answer"])


def sample_changed_rows_by_family(
    *,
    datasets_by_name: dict[str, list[dict[str, object]]],
    counterfactual_names: tuple[str, ...],
    examples_per_family: int,
    split_seed: int,
    causal_model: MCQACausalModel,
) -> list[dict[str, object]]:
    rows_by_family: dict[str, list[dict[str, object]]] = {
        family: [] for family in counterfactual_names
    }
    for dataset_name in sorted(datasets_by_name):
        counterfactual_name, _, _split_name = dataset_name.rpartition("_")
        if counterfactual_name not in rows_by_family:
            continue
        for row in datasets_by_name[dataset_name]:
            family = _row_family(row)
            if family in rows_by_family and _source_answer_changed(row, causal_model):
                rows_by_family[family].append(row)

    sampled_rows: list[dict[str, object]] = []
    for family in counterfactual_names:
        family_rows = list(rows_by_family.get(family, []))
        if len(family_rows) < int(examples_per_family):
            raise ValueError(
                f"Requested examples_per_family={examples_per_family} for family={family}, "
                f"but only {len(family_rows)} filtered rows have changed source answer tokens."
            )
        rng = random.Random(f"{int(split_seed)}:feature_screen:{family}")
        rng.shuffle(family_rows)
        sampled_rows.extend(family_rows[: int(examples_per_family)])
    output_rng = random.Random(f"{int(split_seed)}:feature_screen:combined")
    output_rng.shuffle(sampled_rows)
    return sampled_rows


def _position_by_id(
    *,
    rows: list[dict[str, object]],
    token_positions: list[TokenPosition],
    tokenizer,
    input_key: str,
) -> dict[str, torch.Tensor]:
    return {
        token_position.id: torch.tensor(
            [
                token_position.resolve(
                    row["input"] if input_key == "base" else row["counterfactual_inputs"][0],
                    tokenizer,
                )
                for row in rows
            ],
            dtype=torch.long,
        )
        for token_position in token_positions
    }


def build_answer_token_bank(
    *,
    rows: list[dict[str, object]],
    tokenizer,
    causal_model: MCQACausalModel,
    token_positions: list[TokenPosition],
) -> MCQAPairBank:
    base_inputs = [row["input"] for row in rows]
    source_inputs = [row["counterfactual_inputs"][0] for row in rows]
    base_outputs = [causal_model.run_forward(base_input) for base_input in base_inputs]
    source_outputs = [causal_model.run_forward(source_input) for source_input in source_inputs]
    labels = torch.tensor(
        [_alphabet_index(str(output["answer"])) for output in source_outputs],
        dtype=torch.long,
    )
    base_position_by_id = _position_by_id(
        rows=rows,
        token_positions=token_positions,
        tokenizer=tokenizer,
        input_key="base",
    )
    source_position_by_id = _position_by_id(
        rows=rows,
        token_positions=token_positions,
        tokenizer=tokenizer,
        input_key="source",
    )
    symbol_variant_token_ids = torch.tensor(
        [
            [_encode_symbol_token_variants(str(base[f"symbol{i}"]), tokenizer) for i in range(4)]
            for base in base_inputs
        ],
        dtype=torch.long,
    )
    source_symbol_variant_token_ids = torch.tensor(
        [
            [_encode_symbol_token_variants(str(source[f"symbol{i}"]), tokenizer) for i in range(4)]
            for source in source_inputs
        ],
        dtype=torch.long,
    )
    alphabet_variant_token_ids = torch.tensor(
        [
            [_encode_symbol_token_variants(letter, tokenizer) for letter in ALPHABET_LABELS]
            for _ in base_inputs
        ],
        dtype=torch.long,
    )
    base_answer_token_ids = torch.tensor(
        [
            _encode_symbol_token(str(output["raw_output"]).strip(), tokenizer)
            for output in base_outputs
        ],
        dtype=torch.long,
    )
    answer_token_ids = torch.tensor(
        [
            _encode_symbol_token(str(output["raw_output"]).strip(), tokenizer)
            for output in source_outputs
        ],
        dtype=torch.long,
    )
    changed_mask = torch.tensor(
        [
            str(base["answer"]) != str(source["answer"])
            for base, source in zip(base_outputs, source_outputs)
        ],
        dtype=torch.bool,
    )
    return MCQAPairBank(
        split="feature_screen",
        target_var="answer_token",
        dataset_names=tuple(sorted({f"{_row_family(row)}_filtered" for row in rows})),
        labels=labels,
        base_inputs=base_inputs,
        source_inputs=source_inputs,
        base_outputs=base_outputs,
        source_outputs=source_outputs,
        base_position_by_id=base_position_by_id,
        source_position_by_id=source_position_by_id,
        symbol_token_ids=symbol_variant_token_ids[:, :, 0],
        symbol_variant_token_ids=symbol_variant_token_ids,
        source_symbol_token_ids=source_symbol_variant_token_ids[:, :, 0],
        source_symbol_variant_token_ids=source_symbol_variant_token_ids,
        alphabet_token_ids=alphabet_variant_token_ids[:, :, 0],
        alphabet_variant_token_ids=alphabet_variant_token_ids,
        canonical_answer_token_ids=_validate_answer_tokenization(tokenizer),
        answer_token_ids=answer_token_ids,
        base_answer_token_ids=base_answer_token_ids,
        changed_mask=changed_mask,
        counterfactual_family_names=[_row_family(row) for row in rows],
        expected_answer_texts=[str(output["raw_output"]).strip() for output in source_outputs],
    )


def _activation_map(
    cache: CLTActivationCache,
    *,
    prompt: str,
    layer: int,
    position: int,
    top_k: int,
) -> dict[int, float]:
    return cache.value_map(
        prompt=prompt,
        layer=int(layer),
        position=int(position),
        top_k=int(top_k),
    )


def accumulate_sparse_abs_deltas(
    *,
    total_scores: defaultdict[tuple[int, int], float],
    family_scores: dict[str, defaultdict[tuple[int, int], float]],
    layer: int,
    family: str,
    base_values: dict[int, float],
    source_values: dict[int, float],
) -> None:
    for feature_idx in set(base_values) | set(source_values):
        delta = abs(
            float(source_values.get(feature_idx, 0.0))
            - float(base_values.get(feature_idx, 0.0))
        )
        key = (int(layer), int(feature_idx))
        total_scores[key] += delta
        family_scores[str(family)][key] += delta


def screening_records_from_scores(
    *,
    total_scores: dict[tuple[int, int], float],
    family_scores: dict[str, dict[tuple[int, int], float] | defaultdict[tuple[int, int], float]],
    total_count: int,
    family_counts: dict[str, int],
    families: tuple[str, ...] = COUNTERFACTUAL_FAMILIES,
) -> list[dict[str, object]]:
    records = []
    family_denominators = {family: max(1, int(family_counts.get(family, 0))) for family in families}
    for (layer, feature_idx), total in total_scores.items():
        mean_delta_by_family = {
            family: float(
                family_scores.get(family, {}).get((layer, feature_idx), 0.0)
                / family_denominators[family]
            )
            for family in families
        }
        records.append(
            {
                "layer": int(layer),
                "feature_idx": int(feature_idx),
                "screen_score": float(total / max(1, int(total_count))),
                "mean_abs_delta_by_family": mean_delta_by_family,
            }
        )
    records.sort(
        key=lambda item: (
            -float(item["screen_score"]),
            int(item["layer"]),
            int(item["feature_idx"]),
        )
    )
    return records


def family_accuracy_summary(
    *,
    predictions: list[int],
    labels: list[int],
    families: list[str],
) -> dict[str, object]:
    if not (len(predictions) == len(labels) == len(families)):
        raise ValueError("predictions, labels, and families must have the same length")
    correct = [
        int(prediction) == int(label)
        for prediction, label in zip(predictions, labels)
    ]
    family_correct: dict[str, list[bool]] = defaultdict(list)
    for is_correct, family in zip(correct, families):
        family_correct[str(family)].append(bool(is_correct))
    return {
        "exact_acc": float(sum(correct) / max(1, len(correct))),
        "family_exact_accs": {
            family: float(sum(values) / max(1, len(values)))
            for family, values in sorted(family_correct.items())
        },
    }


def screen_features(
    *,
    model,
    bank: MCQAPairBank,
    layers: tuple[int, ...],
    token_position_id: str,
    top_k_per_layer: int,
    cache: CLTActivationCache,
) -> tuple[list[dict[str, object]], dict[tuple[int, int], dict[str, float]]]:
    total_scores: defaultdict[tuple[int, int], float] = defaultdict(float)
    family_scores: dict[str, defaultdict[tuple[int, int], float]] = {
        family: defaultdict(float) for family in COUNTERFACTUAL_FAMILIES
    }
    family_counts = {family: 0 for family in COUNTERFACTUAL_FAMILIES}
    start = perf_counter()
    for row_index, (base_input, source_input, family) in enumerate(
        zip(bank.base_inputs, bank.source_inputs, bank.counterfactual_family_names)
    ):
        if _should_log_progress(row_index, bank.size):
            log_progress(
                "feature screening "
                f"row={row_index + 1}/{bank.size} family={family} "
                f"elapsed={perf_counter() - start:.1f}s"
            )
        family_counts[str(family)] = family_counts.get(str(family), 0) + 1
        base_prompt = str(base_input["raw_input"])
        source_prompt = str(source_input["raw_input"])
        base_position = int(bank.base_position_by_id[token_position_id][row_index].item())
        source_position = int(bank.source_position_by_id[token_position_id][row_index].item())
        for layer in layers:
            base_values = _activation_map(
                cache,
                prompt=base_prompt,
                layer=int(layer),
                position=base_position,
                top_k=int(top_k_per_layer),
            )
            source_values = _activation_map(
                cache,
                prompt=source_prompt,
                layer=int(layer),
                position=source_position,
                top_k=int(top_k_per_layer),
            )
            accumulate_sparse_abs_deltas(
                total_scores=total_scores,
                family_scores=family_scores,
                layer=int(layer),
                family=str(family),
                base_values=base_values,
                source_values=source_values,
            )

    records = screening_records_from_scores(
        total_scores=total_scores,
        family_scores=family_scores,
        total_count=bank.size,
        family_counts=family_counts,
    )
    score_lookup = {
        (int(record["layer"]), int(record["feature_idx"])): {
            "screen_score": float(record["screen_score"]),
            **{
                f"screen_score_{family}": float(record["mean_abs_delta_by_family"][family])
                for family in COUNTERFACTUAL_FAMILIES
            },
        }
        for record in records
    }
    log_progress(
        "feature screening complete "
        f"unique_features={len(records)} elapsed={perf_counter() - start:.1f}s"
    )
    return records, score_lookup


def evaluate_feature(
    *,
    model,
    bank: MCQAPairBank,
    layer: int,
    feature_idx: int,
    token_position_id: str,
    activation_read_top_k: int,
    cache: CLTActivationCache,
    tokenizer,
    store_prediction_details: bool,
) -> dict[str, object]:
    site = CLTSite(
        layer=int(layer),
        write_layer=int(layer),
        token_position_id=str(token_position_id),
        feature_idx=int(feature_idx),
        top_features=int(activation_read_top_k),
    )
    logits = run_clt_decoded_site_intervention(
        model=model,
        bank=bank,
        site_weights={site: 1.0},
        strength=1.0,
        cache=cache,
        log_context="feature_screen_eval",
    )
    record = {
        "site_label": site.label,
        "layer": int(layer),
        "write_layer": int(layer),
        "feature_idx": int(feature_idx),
        **metrics_from_logits(logits, bank, tokenizer=tokenizer),
    }
    if store_prediction_details:
        record["prediction_details"] = prediction_details_from_logits(
            logits,
            bank,
            tokenizer=tokenizer,
        )
    return record


def main() -> None:
    stage_start = perf_counter()
    args = build_parser().parse_args()
    results_timestamp = (
        args.results_timestamp
        or os.environ.get("RESULTS_TIMESTAMP")
        or "mcqa_clt_feature_screen_top128"
    )
    output_dir = Path(args.results_root) / f"{results_timestamp}_mcqa_clt_feature_screen"
    output_dir.mkdir(parents=True, exist_ok=True)

    counterfactual_names = tuple(
        parse_csv_strings(args.counterfactual_names) or DEFAULT_COUNTERFACTUAL_NAMES
    )
    transcoder_set = resolve_transcoder_set(args.transcoder_set, args.transcoder_size)

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    selected_rows = sample_changed_rows_by_family(
        datasets_by_name=filtered_datasets,
        counterfactual_names=counterfactual_names,
        examples_per_family=int(args.examples_per_family),
        split_seed=int(args.split_seed),
        causal_model=causal_model,
    )
    log_progress(
        "sampled feature-screen rows "
        + ", ".join(
            f"{family}={sum(1 for row in selected_rows if _row_family(row) == family)}"
            for family in counterfactual_names
        )
    )

    from circuit_tracing_ot.model import load_replacement_model

    log_progress(
        f"loading ReplacementModel model={args.model_name} "
        f"transcoder_set={transcoder_set}"
    )
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

    bank = build_answer_token_bank(
        rows=selected_rows,
        tokenizer=model.tokenizer,
        causal_model=causal_model,
        token_positions=token_positions,
    )
    num_layers = _model_num_layers(model)
    layers = parse_csv_ints(args.layers) or tuple(range(num_layers))
    cache = CLTActivationCache(model)
    screen_records, score_lookup = screen_features(
        model=model,
        bank=bank,
        layers=tuple(int(layer) for layer in layers),
        token_position_id=str(args.token_position_id),
        top_k_per_layer=int(args.screen_top_k_per_layer),
        cache=cache,
    )
    candidates = screen_records[: int(args.num_candidate_features)]
    log_progress(
        "selected feature candidates "
        f"count={len(candidates)} first={candidates[:5]}"
    )

    feature_results = []
    eval_start = perf_counter()
    for candidate_index, candidate in enumerate(candidates):
        log_progress(
            "evaluating feature candidate "
            f"{candidate_index + 1}/{len(candidates)} "
            f"layer={candidate['layer']} feature={candidate['feature_idx']} "
            f"screen_score={float(candidate['screen_score']):.6g}"
        )
        result = evaluate_feature(
            model=model,
            bank=bank,
            layer=int(candidate["layer"]),
            feature_idx=int(candidate["feature_idx"]),
            token_position_id=str(args.token_position_id),
            activation_read_top_k=int(args.screen_top_k_per_layer),
            cache=cache,
            tokenizer=model.tokenizer,
            store_prediction_details=bool(args.store_prediction_details),
        )
        key = (int(candidate["layer"]), int(candidate["feature_idx"]))
        result.update(score_lookup[key])
        result["mean_abs_delta_by_family"] = candidate["mean_abs_delta_by_family"]
        feature_results.append(result)
    feature_results.sort(
        key=lambda item: (
            -float(item["exact_acc"]),
            -max(float(value) for value in item.get("family_exact_accs", {}).values()),
            -float(item["screen_score"]),
            int(item["layer"]),
            int(item["feature_idx"]),
        )
    )
    log_progress(f"feature evaluation complete elapsed={perf_counter() - eval_start:.1f}s")

    payload = {
        "kind": "mcqa_clt_feature_screen",
        "config": {
            **vars(args),
            "transcoder_set": str(transcoder_set),
            "counterfactual_names": list(counterfactual_names),
            "layers": [int(layer) for layer in layers],
        },
        "data": bank.metadata(),
        "screening": {
            "top_k_per_layer": int(args.screen_top_k_per_layer),
            "unique_feature_count": len(screen_records),
            "num_candidate_features": int(args.num_candidate_features),
            "top_candidates": candidates,
        },
        "results": feature_results,
        "best_overall": feature_results[0] if feature_results else None,
        "best_by_family": {
            family: max(
                feature_results,
                key=lambda item, current_family=family: (
                    float(item.get("family_exact_accs", {}).get(current_family, 0.0)),
                    float(item["exact_acc"]),
                    float(item["screen_score"]),
                ),
            )
            if feature_results
            else None
            for family in COUNTERFACTUAL_FAMILIES
        },
        "runtime_seconds": float(perf_counter() - stage_start),
    }
    output_path = output_dir / "mcqa_clt_feature_screen_results.json"
    write_json(output_path, payload)
    log_progress(f"wrote {output_path}")


if __name__ == "__main__":
    main()
