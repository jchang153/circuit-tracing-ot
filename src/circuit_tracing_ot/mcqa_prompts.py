"""Representative MCQA prompts for initial circuit-tracer experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MCQAPrompt:
    """One representative CopyColors-style MCQA prompt."""

    prompt_id: str
    prompt: str
    expected_answer: str
    target_variable: str
    notes: str

    @property
    def slug(self) -> str:
        return self.prompt_id.replace("_", "-")


PROMPTS: tuple[MCQAPrompt, ...] = (
    MCQAPrompt(
        prompt_id="copycolors-a",
        prompt=(
            "red is apple. blue is car. green is frog. yellow is banana.\n"
            "Question: what color is apple?\n"
            "A. red\n"
            "B. blue\n"
            "C. green\n"
            "D. yellow\n"
            "Answer:"
        ),
        expected_answer=" A",
        target_variable="answer_pointer",
        notes="Correct answer appears in option A.",
    ),
    MCQAPrompt(
        prompt_id="copycolors-d",
        prompt=(
            "purple is grape. orange is carrot. black is tire. white is snow.\n"
            "Question: what color is snow?\n"
            "A. purple\n"
            "B. orange\n"
            "C. black\n"
            "D. white\n"
            "Answer:"
        ),
        expected_answer=" D",
        target_variable="answer_pointer",
        notes="Correct answer appears in option D.",
    ),
    MCQAPrompt(
        prompt_id="copycolors-token-blue",
        prompt=(
            "green is mint. blue is ocean. brown is wood. pink is flower.\n"
            "Question: what color is ocean?\n"
            "A. green\n"
            "B. blue\n"
            "C. brown\n"
            "D. pink\n"
            "Answer:"
        ),
        expected_answer=" B",
        target_variable="answer_token",
        notes="Answer token is the symbol for the blue option.",
    ),
    MCQAPrompt(
        prompt_id="copycolors-token-yellow",
        prompt=(
            "black is ink. yellow is lemon. red is rose. gray is stone.\n"
            "Question: what color is lemon?\n"
            "A. black\n"
            "B. yellow\n"
            "C. red\n"
            "D. gray\n"
            "Answer:"
        ),
        expected_answer=" B",
        target_variable="answer_token",
        notes="Different object/color mapping with the same answer position as blue.",
    ),
)

PROMPTS_BY_ID = {prompt.prompt_id: prompt for prompt in PROMPTS}


def get_prompt(prompt_id: str) -> MCQAPrompt:
    try:
        return PROMPTS_BY_ID[prompt_id]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPTS_BY_ID))
        raise ValueError(f"Unknown prompt_id {prompt_id!r}; available prompt IDs: {available}") from exc
