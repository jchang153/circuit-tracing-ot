"""MCQA counterfactual data and split logic ported from causal-abstractions-ot."""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
from pathlib import Path
from typing import Callable

from datasets import get_dataset_split_names, load_dataset
import torch


CANONICAL_ANSWER_STRINGS = (" A", " B", " C", " D")
ALPHABET_LABELS = tuple(chr(codepoint) for codepoint in range(ord("A"), ord("Z") + 1))
COUNTERFACTUAL_FAMILIES = ("answerPosition", "randomLetter", "answerPosition_randomLetter")
TARGET_VAR_ALIASES = {
    "answer_pointer": "answer_pointer",
    "ans_index": "answer_pointer",
    "answer": "answer_token",
    "answer_token": "answer_token",
    "ans_value": "answer_token",
}


def canonicalize_target_var(target_var: str) -> str:
    canonical = TARGET_VAR_ALIASES.get(str(target_var))
    if canonical is None:
        raise ValueError(f"Unsupported MCQA target variable {target_var}")
    return canonical


class MCQACausalModel:
    """Small copy of the CopyColors MCQA causal model."""

    def run_forward(self, input_dict: dict[str, object]) -> dict[str, object]:
        output = dict(input_dict)
        question = tuple(output["question"])
        choices = [str(output[f"choice{index}"]) for index in range(4)]
        symbols = [str(output[f"symbol{index}"]) for index in range(4)]
        pointer = None
        for index, choice in enumerate(choices):
            if choice == question[0]:
                pointer = index
                break
        if pointer is None:
            raise ValueError(f"Could not resolve answer_pointer from question={question}")
        answer = " " + symbols[pointer]
        output["answer_pointer"] = int(pointer)
        output["answer"] = answer
        output["raw_output"] = answer
        return output

    def run_interchange(
        self,
        base_input: dict[str, object],
        source_input: dict[str, object],
        target_variables: tuple[str, ...] | list[str],
    ) -> dict[str, object]:
        base_setting = self.run_forward(base_input)
        source_setting = self.run_forward(source_input)
        canonical_targets = {canonicalize_target_var(str(variable)) for variable in target_variables}
        setting = dict(base_setting)
        if "answer_pointer" in canonical_targets:
            setting["answer_pointer"] = int(source_setting["answer_pointer"])
        if "answer_token" in canonical_targets:
            setting["answer"] = str(source_setting["answer"])
        if "answer_pointer" in canonical_targets and "answer_token" not in canonical_targets:
            pointer = int(setting["answer_pointer"])
            setting["answer"] = " " + str(setting[f"symbol{pointer}"])
        if canonical_targets:
            setting["raw_output"] = str(setting["answer"])
        return setting


@dataclass(frozen=True)
class TokenPosition:
    resolver: Callable[[dict[str, object], object], list[int]]
    id: str

    def resolve(self, input_dict: dict[str, object], tokenizer) -> int:
        positions = self.resolver(input_dict, tokenizer)
        if not positions:
            raise ValueError(f"Token position {self.id} returned no indices")
        return int(positions[0])


@dataclass(frozen=True)
class MCQAPairBank:
    split: str
    target_var: str
    dataset_names: tuple[str, ...]
    labels: torch.Tensor
    base_inputs: list[dict[str, object]]
    source_inputs: list[dict[str, object]]
    base_outputs: list[dict[str, object]]
    source_outputs: list[dict[str, object]]
    base_position_by_id: dict[str, torch.Tensor]
    source_position_by_id: dict[str, torch.Tensor]
    symbol_token_ids: torch.Tensor
    symbol_variant_token_ids: torch.Tensor
    source_symbol_token_ids: torch.Tensor
    source_symbol_variant_token_ids: torch.Tensor
    alphabet_token_ids: torch.Tensor
    alphabet_variant_token_ids: torch.Tensor
    canonical_answer_token_ids: torch.Tensor
    answer_token_ids: torch.Tensor
    base_answer_token_ids: torch.Tensor
    changed_mask: torch.Tensor
    counterfactual_family_names: list[str]
    expected_answer_texts: list[str]

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])

    def metadata(self) -> dict[str, object]:
        family_counts: dict[str, int] = {}
        for family_name in self.counterfactual_family_names:
            family_counts[str(family_name)] = family_counts.get(str(family_name), 0) + 1
        return {
            "split": self.split,
            "target_var": self.target_var,
            "size": self.size,
            "dataset_names": list(self.dataset_names),
            "changed_count": int(self.changed_mask.sum().item()),
            "changed_rate": float(self.changed_mask.float().mean().item()) if self.size else 0.0,
            "family_counts": family_counts,
        }


def parse_mcqa_example(row: dict[str, object]) -> dict[str, object]:
    prompt_str = str(row.get("prompt", ""))
    if " is " in prompt_str:
        noun, color = prompt_str.split(" is ", 1)
    elif " are " in prompt_str:
        noun, color = prompt_str.split(" are ", 1)
    else:
        raise ValueError(f"Could not parse MCQA question text from prompt: {prompt_str}")
    noun = noun.strip().lower()
    color = color.split(".", 1)[0].strip().lower()
    variables_dict: dict[str, object] = {"question": (color, noun), "raw_input": prompt_str}
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    for index, label in enumerate(labels):
        variables_dict[f"symbol{index}"] = str(label)
        variables_dict[f"choice{index}"] = str(texts[index])
    return variables_dict


def _find_correct_symbol_index(
    input_dict: dict[str, object],
    tokenizer,
    causal_model: MCQACausalModel,
) -> list[int]:
    output = causal_model.run_forward(input_dict)
    pointer = int(output["answer_pointer"])
    correct_symbol = str(output[f"symbol{pointer}"])
    prompt = str(input_dict["raw_input"])
    matches = list(re.finditer(r"\b[A-Z]\b", prompt))
    symbol_match = None
    for match in matches:
        if prompt[match.start() : match.end()] == correct_symbol:
            symbol_match = match
            break
    if symbol_match is None:
        raise ValueError(f"Could not find correct symbol {correct_symbol} in prompt: {prompt}")
    substring = prompt[: symbol_match.end()]
    tokenized = tokenizer(substring, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    return [len(tokenized) - 1]


def get_token_positions(tokenizer, causal_model: MCQACausalModel) -> list[TokenPosition]:
    def correct_symbol(input_dict: dict[str, object], current_tokenizer) -> list[int]:
        return _find_correct_symbol_index(input_dict, current_tokenizer, causal_model)

    def correct_symbol_period(input_dict: dict[str, object], current_tokenizer) -> list[int]:
        return [correct_symbol(input_dict, current_tokenizer)[0] + 1]

    def last_token(input_dict: dict[str, object], current_tokenizer) -> list[int]:
        prompt = str(input_dict["raw_input"])
        tokenized = current_tokenizer(prompt, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        return [len(tokenized) - 1]

    return [
        TokenPosition(correct_symbol, "correct_symbol"),
        TokenPosition(correct_symbol_period, "correct_symbol_period"),
        TokenPosition(last_token, "last_token"),
    ]


def load_counterfactual_rows(
    *,
    split: str,
    size: int | None,
    dataset_path: str,
    dataset_config: str | None,
    hf_token: str | None,
) -> dict[str, list[dict[str, object]]]:
    dataset_path_obj = Path(dataset_path)
    if dataset_path_obj.exists():
        split_file = dataset_path_obj / f"{split}.jsonl"
        dataset = load_dataset("json", data_files={split: str(split_file)}, split=split)
    elif dataset_config:
        dataset = load_dataset(dataset_path, dataset_config, split=split, token=hf_token)
    else:
        dataset = load_dataset(dataset_path, split=split, token=hf_token)
    if size is not None:
        dataset = dataset.select(range(min(int(size), len(dataset))))
    sample = dataset[0]
    counterfactual_names = [
        key
        for key in sample.keys()
        if key.endswith("_counterfactual")
        and "noun" not in key
        and "color" not in key
        and "symbol" not in key
    ]
    datasets: dict[str, list[dict[str, object]]] = {}
    for counterfactual_name in counterfactual_names:
        dataset_name = counterfactual_name.replace("_counterfactual", f"_{split}")
        counterfactual_family = counterfactual_name.replace("_counterfactual", "")
        rows: list[dict[str, object]] = []
        for row in dataset:
            rows.append(
                {
                    "input": parse_mcqa_example(row),
                    "counterfactual_inputs": [parse_mcqa_example(row[counterfactual_name])],
                    "counterfactual_family": counterfactual_family,
                }
            )
        datasets[dataset_name] = rows
    return datasets


def load_public_mcqa_datasets(
    *,
    size: int | None,
    dataset_path: str,
    dataset_config: str | None,
    hf_token: str | None,
) -> dict[str, list[dict[str, object]]]:
    dataset_path_obj = Path(dataset_path)
    if dataset_path_obj.exists():
        candidate_splits = tuple(
            split_file.stem for split_file in sorted(dataset_path_obj.glob("*.jsonl")) if split_file.stem
        )
    elif dataset_config:
        candidate_splits = tuple(get_dataset_split_names(dataset_path, dataset_config, token=hf_token))
    else:
        candidate_splits = tuple(get_dataset_split_names(dataset_path, token=hf_token))
    datasets: dict[str, list[dict[str, object]]] = {}
    for split in candidate_splits:
        datasets.update(
            load_counterfactual_rows(
                split=split,
                size=size,
                dataset_path=dataset_path,
                dataset_config=dataset_config,
                hf_token=hf_token,
            )
        )
    return datasets


def _validate_answer_tokenization(tokenizer) -> torch.Tensor:
    token_ids = []
    for token_text in CANONICAL_ANSWER_STRINGS:
        ids = tokenizer.encode(token_text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Expected {token_text!r} to map to one token, got {ids}")
        token_ids.append(int(ids[0]))
    return torch.tensor(token_ids, dtype=torch.long)


def _encode_symbol_token(symbol: str, tokenizer) -> int:
    ids = tokenizer.encode(" " + str(symbol), add_special_tokens=False)
    if len(ids) != 1:
        ids = tokenizer.encode(str(symbol), add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected symbol {symbol!r} to map to one token, got {ids}")
    return int(ids[0])


def _encode_symbol_token_variants(symbol: str, tokenizer) -> tuple[int, int]:
    symbol = str(symbol).strip()
    variant_ids = []
    for candidate in (" " + symbol, symbol):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            variant_ids.append(int(ids[0]))
    if not variant_ids:
        raise ValueError(f"Expected symbol {symbol!r} to have one-token encoding")
    if len(variant_ids) == 1:
        variant_ids.append(variant_ids[0])
    return (variant_ids[0], variant_ids[1])


def _alphabet_index(symbol: str) -> int:
    normalized = str(symbol).strip()
    if len(normalized) != 1 or normalized not in ALPHABET_LABELS:
        raise ValueError(f"Expected uppercase alphabet symbol, got {symbol!r}")
    return ALPHABET_LABELS.index(normalized)


def _compute_row_change_masks(
    rows: list[dict[str, object]],
    causal_model: MCQACausalModel,
) -> dict[str, list[bool]]:
    base_inputs = [row["input"] for row in rows]
    source_inputs = [row["counterfactual_inputs"][0] for row in rows]
    base_outputs = [causal_model.run_forward(base_input) for base_input in base_inputs]
    source_outputs = [causal_model.run_forward(source_input) for source_input in source_inputs]
    changed_pointer = [
        int(base_output["answer_pointer"]) != int(source_output["answer_pointer"])
        for base_output, source_output in zip(base_outputs, source_outputs)
    ]
    changed_answer = [
        str(base_output["answer"]) != str(source_output["answer"])
        for base_output, source_output in zip(base_outputs, source_outputs)
    ]
    return {
        "answer_pointer": changed_pointer,
        "answer_token": changed_answer,
        "answer": changed_answer,
    }


def build_pair_banks(
    *,
    tokenizer,
    causal_model: MCQACausalModel,
    token_positions: list[TokenPosition],
    datasets_by_name: dict[str, list[dict[str, object]]],
    counterfactual_names: tuple[str, ...],
    target_vars: tuple[str, ...],
    split_seed: int,
    train_pool_size: int,
    calibration_pool_size: int,
    test_pool_size: int,
) -> tuple[dict[str, dict[str, MCQAPairBank]], dict[str, object]]:
    canonical_target_vars = tuple(canonicalize_target_var(target_var) for target_var in target_vars)
    canonical_answer_token_ids = _validate_answer_tokenization(tokenizer)

    def make_bank(
        output_split: str,
        split_dataset_names: list[str],
        combined_rows: list[dict[str, object]],
    ) -> dict[str, MCQAPairBank]:
        base_inputs = [row["input"] for row in combined_rows]
        source_inputs = [row["counterfactual_inputs"][0] for row in combined_rows]
        family_names = [str(row["counterfactual_family"]) for row in combined_rows]
        base_outputs = [causal_model.run_forward(base_input) for base_input in base_inputs]
        source_outputs = [causal_model.run_forward(source_input) for source_input in source_inputs]
        base_position_by_id = {
            token_position.id: torch.tensor(
                [token_position.resolve(base_input, tokenizer) for base_input in base_inputs],
                dtype=torch.long,
            )
            for token_position in token_positions
        }
        source_position_by_id = {
            token_position.id: torch.tensor(
                [token_position.resolve(source_input, tokenizer) for source_input in source_inputs],
                dtype=torch.long,
            )
            for token_position in token_positions
        }
        symbol_token_ids = torch.tensor(
            [[_encode_symbol_token(str(base[f"symbol{i}"]), tokenizer) for i in range(4)] for base in base_inputs],
            dtype=torch.long,
        )
        symbol_variant_token_ids = torch.tensor(
            [
                [_encode_symbol_token_variants(str(base[f"symbol{i}"]), tokenizer) for i in range(4)]
                for base in base_inputs
            ],
            dtype=torch.long,
        )
        source_symbol_token_ids = torch.tensor(
            [
                [_encode_symbol_token(str(source[f"symbol{i}"]), tokenizer) for i in range(4)]
                for source in source_inputs
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
            [[_encode_symbol_token_variants(letter, tokenizer) for letter in ALPHABET_LABELS] for _ in base_inputs],
            dtype=torch.long,
        )
        alphabet_token_ids = alphabet_variant_token_ids[:, :, 0]
        base_answer_token_ids = torch.tensor(
            [_encode_symbol_token(str(base_output["raw_output"]).strip(), tokenizer) for base_output in base_outputs],
            dtype=torch.long,
        )
        answer_label_indices = torch.tensor(
            [_alphabet_index(str(source_output["answer"])) for source_output in source_outputs],
            dtype=torch.long,
        )
        pointer_label_indices = torch.tensor(
            [int(source_output["answer_pointer"]) for source_output in source_outputs],
            dtype=torch.long,
        )
        changed_pointer = torch.tensor(
            [
                int(base_output["answer_pointer"]) != int(source_output["answer_pointer"])
                for base_output, source_output in zip(base_outputs, source_outputs)
            ],
            dtype=torch.bool,
        )
        changed_answer = torch.tensor(
            [
                str(base_output["answer"]) != str(source_output["answer"])
                for base_output, source_output in zip(base_outputs, source_outputs)
            ],
            dtype=torch.bool,
        )
        banks: dict[str, MCQAPairBank] = {}
        for target_var in canonical_target_vars:
            labels = pointer_label_indices if target_var == "answer_pointer" else answer_label_indices
            changed_mask = changed_pointer if target_var == "answer_pointer" else changed_answer
            interchange_outputs = [
                causal_model.run_interchange(base_input, source_input, (target_var,))
                for base_input, source_input in zip(base_inputs, source_inputs)
            ]
            answer_token_ids = torch.tensor(
                [_encode_symbol_token(str(setting["raw_output"]).strip(), tokenizer) for setting in interchange_outputs],
                dtype=torch.long,
            )
            banks[target_var] = MCQAPairBank(
                split=output_split,
                target_var=target_var,
                dataset_names=tuple(split_dataset_names),
                labels=labels,
                base_inputs=base_inputs,
                source_inputs=source_inputs,
                base_outputs=base_outputs,
                source_outputs=source_outputs,
                base_position_by_id=base_position_by_id,
                source_position_by_id=source_position_by_id,
                symbol_token_ids=symbol_token_ids,
                symbol_variant_token_ids=symbol_variant_token_ids,
                source_symbol_token_ids=source_symbol_token_ids,
                source_symbol_variant_token_ids=source_symbol_variant_token_ids,
                alphabet_token_ids=alphabet_token_ids,
                alphabet_variant_token_ids=alphabet_variant_token_ids,
                canonical_answer_token_ids=canonical_answer_token_ids,
                answer_token_ids=answer_token_ids,
                base_answer_token_ids=base_answer_token_ids,
                changed_mask=changed_mask,
                counterfactual_family_names=family_names,
                expected_answer_texts=[str(setting["raw_output"]).strip() for setting in interchange_outputs],
            )
        return banks

    pooled_dataset_names = []
    pooled_rows: list[dict[str, object]] = []
    for dataset_name in sorted(datasets_by_name):
        counterfactual_name, _, _split_name = dataset_name.rpartition("_")
        if counterfactual_name in counterfactual_names:
            pooled_dataset_names.append(dataset_name)
            pooled_rows.extend(datasets_by_name[dataset_name])
    if not pooled_rows:
        raise ValueError("No MCQA rows found for pooled bank construction")
    rng = random.Random(int(split_seed))
    shuffled_rows = list(pooled_rows)
    rng.shuffle(shuffled_rows)
    total = len(shuffled_rows)
    if int(train_pool_size) > total:
        raise ValueError(f"Requested train_pool_size={train_pool_size}, only {total} rows available")
    train_rows = shuffled_rows[: int(train_pool_size)]
    holdout_candidate_rows = shuffled_rows[int(train_pool_size) :]
    train_rng = random.Random(f"{int(split_seed)}:train:shared")
    shared_train_rows = list(train_rows)
    train_rng.shuffle(shared_train_rows)
    banks_by_split: dict[str, dict[str, MCQAPairBank]] = {
        "train": make_bank("train", pooled_dataset_names, shared_train_rows),
        "calibration": {},
        "test": {},
    }
    holdout_changed_masks = _compute_row_change_masks(holdout_candidate_rows, causal_model)
    for target_var in canonical_target_vars:
        changed_mask = holdout_changed_masks[target_var]
        positive_rows = [row for row, changed in zip(holdout_candidate_rows, changed_mask) if changed]
        local_rng = random.Random(f"{int(split_seed)}:holdout:{target_var}")
        local_rng.shuffle(positive_rows)
        required = int(calibration_pool_size) + int(test_pool_size)
        if len(positive_rows) < required:
            raise ValueError(
                f"Requested calibration_pool_size={calibration_pool_size} and "
                f"test_pool_size={test_pool_size} for target_var={target_var}, "
                f"but only {len(positive_rows)} sensitive rows are available"
            )
        calibration_rows = positive_rows[: int(calibration_pool_size)]
        test_rows = positive_rows[int(calibration_pool_size) : required]
        banks_by_split["calibration"][target_var] = make_bank(
            "calibration",
            pooled_dataset_names,
            calibration_rows,
        )[target_var]
        banks_by_split["test"][target_var] = make_bank("test", pooled_dataset_names, test_rows)[target_var]
    metadata = {
        split: {target_var: bank.metadata() for target_var, bank in banks.items()}
        for split, banks in banks_by_split.items()
    }
    return banks_by_split, metadata
