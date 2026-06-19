"""CopyColors MCQA prompt loading and formatting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


DEFAULT_DATASET_NAME = "jchang153/copycolors_mcqa"
DEFAULT_DATASET_CONFIG = None
DEFAULT_DATASET_SPLIT = "train"


@dataclass(frozen=True)
class MCQAPrompt:
    """One CopyColors MCQA row formatted for next-token attribution."""

    prompt_id: str
    prompt: str
    expected_answer: str
    question: str
    choices: tuple[tuple[str, str], ...]
    dataset_id: int | str
    dataset_name: str
    dataset_config: str | None
    dataset_split: str

    @property
    def slug(self) -> str:
        return self.prompt_id.replace("_", "-").replace("/", "-")


def _format_prompt(question: str, choices: tuple[tuple[str, str], ...]) -> str:
    choice_lines = "\n".join(f"{label}. {text}" for label, text in choices)
    return f"{question}\n{choice_lines}\nAnswer:"


def _expected_answer_from_prompt(prompt: str, choices: tuple[tuple[str, str], ...]) -> str:
    match = re.search(r"Question:\s+.*?\b(?:is|are)\s+([^.]+)\.\s+What color\b", prompt)
    if not match:
        raise ValueError(f"Could not infer CopyColors answer from prompt: {prompt!r}")
    answer_text = match.group(1).strip().lower()
    for label, choice_text in choices:
        if choice_text.strip().lower() == answer_text:
            return f" {label}"
    raise ValueError(f"Answer text {answer_text!r} was not found in choices: {choices!r}")


def _prompt_from_row(
    row: dict[str, Any],
    *,
    row_index: int,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
) -> MCQAPrompt:
    labels = [str(label) for label in row["choices"]["label"]]
    texts = [str(text) for text in row["choices"]["text"]]
    choices = tuple(zip(labels, texts))
    if "answerKey" in row:
        expected_answer = f" {labels[int(row['answerKey'])]}"
    else:
        expected_answer = _expected_answer_from_prompt(str(row["prompt"]), choices)
    dataset_id = row.get("id", row_index)
    config_label = dataset_config or "default"
    prompt_id = f"{config_label}-{dataset_split}-{dataset_id}"
    prompt_text = str(row["prompt"]) if "prompt" in row else _format_prompt(str(row["question"]), choices)
    question = str(row["question"]) if "question" in row else prompt_text.splitlines()[0]
    return MCQAPrompt(
        prompt_id=prompt_id,
        prompt=prompt_text,
        expected_answer=expected_answer,
        question=question,
        choices=choices,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
    )


def load_prompts(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str | None = DEFAULT_DATASET_CONFIG,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    limit: int | None = None,
) -> list[MCQAPrompt]:
    """Load CopyColors MCQA rows from Hugging Face and format them for circuit-tracer."""
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)
    rows = dataset if limit is None else dataset.select(range(min(int(limit), len(dataset))))
    return [
        _prompt_from_row(
            dict(row),
            row_index=row_index,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            dataset_split=dataset_split,
        )
        for row_index, row in enumerate(rows)
    ]


def get_prompt(prompt_id: str, prompts: list[MCQAPrompt]) -> MCQAPrompt:
    for prompt in prompts:
        if prompt.prompt_id == prompt_id or str(prompt.dataset_id) == str(prompt_id):
            return prompt
    available = ", ".join(prompt.prompt_id for prompt in prompts[:10])
    suffix = "" if len(prompts) <= 10 else ", ..."
    raise ValueError(f"Unknown prompt_id {prompt_id!r}; available prompt IDs: {available}{suffix}")
