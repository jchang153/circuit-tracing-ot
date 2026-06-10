from collections import defaultdict
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from circuit_tracing_ot.clt_features import extract_clt_feature_values
from scripts.screen_mcqa_clt_features import (
    accumulate_sparse_abs_deltas,
    family_accuracy_summary,
    screening_records_from_scores,
)


def test_sparse_accumulation_treats_missing_top_k_activations_as_zero():
    total_scores = defaultdict(float)
    family_scores = {"answerPosition": defaultdict(float)}

    accumulate_sparse_abs_deltas(
        total_scores=total_scores,
        family_scores=family_scores,
        layer=3,
        family="answerPosition",
        base_values={10: 2.0, 11: -1.0},
        source_values={10: 5.5, 12: -4.0},
    )

    assert total_scores[(3, 10)] == 3.5
    assert total_scores[(3, 11)] == 1.0
    assert total_scores[(3, 12)] == 4.0
    assert family_scores["answerPosition"][(3, 12)] == 4.0


def test_screening_records_sort_by_mean_abs_delta():
    records = screening_records_from_scores(
        total_scores={(1, 7): 2.0, (0, 4): 8.0, (2, 5): 8.0},
        family_scores={
            "answerPosition": defaultdict(float, {(1, 7): 2.0, (0, 4): 8.0, (2, 5): 8.0}),
            "randomLetter": defaultdict(float),
            "answerPosition_randomLetter": defaultdict(float),
        },
        total_count=4,
        family_counts={"answerPosition": 4, "randomLetter": 0, "answerPosition_randomLetter": 0},
    )

    assert [(record["layer"], record["feature_idx"]) for record in records] == [
        (0, 4),
        (2, 5),
        (1, 7),
    ]
    assert records[0]["screen_score"] == 2.0
    assert records[0]["mean_abs_delta_by_family"]["answerPosition"] == 2.0


def test_family_accuracy_summary():
    summary = family_accuracy_summary(
        predictions=[1, 2, 3, 4],
        labels=[1, 0, 3, 0],
        families=["answerPosition", "answerPosition", "randomLetter", "randomLetter"],
    )

    assert summary["exact_acc"] == 0.5
    assert summary["family_exact_accs"] == {"answerPosition": 0.5, "randomLetter": 0.5}


def test_extract_clt_feature_values_reads_layer_axis_in_3d_tensor():
    tensor = torch.zeros(3, 2, 5)
    tensor[0, 1, 0] = 10.0
    tensor[2, 1, 4] = 9.0

    values = extract_clt_feature_values(tensor, layer=2, position=1, top_k=1)

    assert values[0].layer == 2
    assert values[0].feature_idx == 4
    assert values[0].value == 9.0


def test_extract_clt_feature_values_rejects_nonzero_layer_from_2d_tensor():
    tensor = torch.zeros(2, 5)

    with pytest.raises(ValueError, match="nonzero layer-specific"):
        extract_clt_feature_values(tensor, layer=2, position=1, top_k=1)
