"""MCQA signatures and metrics ported from causal-abstractions-ot."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .checking import checker_accuracy
from .data import (
    ALPHABET_LABELS,
    COUNTERFACTUAL_FAMILIES,
    MCQAPairBank,
    canonicalize_target_var,
)


STRUCTURED_SLOT_DIM = 4
STRUCTURED_LABEL_DIM = len(ALPHABET_LABELS)
STRUCTURED_FEATURE_DIM = STRUCTURED_SLOT_DIM + STRUCTURED_LABEL_DIM


def _gather_variant_logits(logits: torch.Tensor, variant_token_ids: torch.Tensor) -> torch.Tensor:
    batch_size, num_classes, num_variants = variant_token_ids.shape
    gathered = torch.gather(
        logits,
        dim=1,
        index=variant_token_ids.to(logits.device).reshape(batch_size, num_classes * num_variants),
    )
    gathered = gathered.reshape(batch_size, num_classes, num_variants)
    return gathered.max(dim=-1).values


def _gather_slot_logits(logits: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    return _gather_variant_logits(logits, bank.symbol_variant_token_ids)


def _gather_label_logits(logits: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    return _gather_variant_logits(logits, bank.alphabet_variant_token_ids)


def structured_output_features(logits: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    return torch.cat((_gather_slot_logits(logits, bank), _gather_label_logits(logits, bank)), dim=1)


def aggregate_family_features(per_example_features: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    blocks = []
    feature_dim = int(per_example_features.shape[1])
    for family_name in COUNTERFACTUAL_FAMILIES:
        mask = torch.tensor(
            [str(current_family) == str(family_name) for current_family in bank.counterfactual_family_names],
            device=per_example_features.device,
            dtype=torch.bool,
        )
        block = (
            per_example_features[mask].mean(dim=0)
            if bool(mask.any())
            else torch.zeros(feature_dim, dtype=per_example_features.dtype, device=per_example_features.device)
        )
        blocks.append(block)
    return torch.cat(blocks, dim=0)


def normalize_family_feature_blocks(aggregated_features: torch.Tensor) -> torch.Tensor:
    normalized_blocks = []
    offset = 0
    for _family_name in COUNTERFACTUAL_FAMILIES:
        family_block = aggregated_features[offset : offset + STRUCTURED_FEATURE_DIM]
        slot_block = family_block[:STRUCTURED_SLOT_DIM] - family_block[:STRUCTURED_SLOT_DIM].mean()
        label_block = family_block[STRUCTURED_SLOT_DIM:] - family_block[STRUCTURED_SLOT_DIM:].mean()
        slot_norm = torch.linalg.vector_norm(slot_block, ord=2)
        label_norm = torch.linalg.vector_norm(label_block, ord=2)
        if float(slot_norm.item()) > 0.0:
            slot_block = slot_block / slot_norm
        if float(label_norm.item()) > 0.0:
            label_block = label_block / label_norm
        normalized_blocks.append(torch.cat((slot_block, label_block), dim=0))
        offset += STRUCTURED_FEATURE_DIM
    return torch.cat(normalized_blocks, dim=0)


def normalize_family_label_feature_blocks(aggregated_features: torch.Tensor) -> torch.Tensor:
    normalized_blocks = []
    offset = 0
    for _family_name in COUNTERFACTUAL_FAMILIES:
        family_block = aggregated_features[offset : offset + STRUCTURED_LABEL_DIM]
        family_block = family_block - family_block.mean()
        label_norm = torch.linalg.vector_norm(family_block, ord=2)
        if float(label_norm.item()) > 0.0:
            family_block = family_block / label_norm
        normalized_blocks.append(family_block)
        offset += STRUCTURED_LABEL_DIM
    return torch.cat(normalized_blocks, dim=0)


def build_family_signature(
    per_example_features: torch.Tensor,
    bank: MCQAPairBank,
    *,
    normalize_blocks: bool = False,
) -> torch.Tensor:
    aggregated = aggregate_family_features(per_example_features, bank)
    return normalize_family_feature_blocks(aggregated) if normalize_blocks else aggregated


def build_family_label_signature(
    per_example_label_features: torch.Tensor,
    bank: MCQAPairBank,
    *,
    normalize_blocks: bool = False,
) -> torch.Tensor:
    aggregated = aggregate_family_features(per_example_label_features, bank)
    return normalize_family_label_feature_blocks(aggregated) if normalize_blocks else aggregated


def _base_answer_indices(bank: MCQAPairBank) -> torch.Tensor:
    return torch.tensor(
        [ALPHABET_LABELS.index(str(output["answer"]).strip()) for output in bank.base_outputs],
        dtype=torch.long,
    )


def _target_answer_indices(bank: MCQAPairBank) -> torch.Tensor:
    target_var = canonicalize_target_var(bank.target_var)
    if target_var == "answer_pointer":
        source_pointer_indices = torch.tensor(
            [int(output["answer_pointer"]) for output in bank.source_outputs],
            dtype=torch.long,
        )
        return torch.tensor(
            [
                ALPHABET_LABELS.index(str(bank.base_inputs[index][f"symbol{int(pointer)}"]).strip())
                for index, pointer in enumerate(source_pointer_indices.tolist())
            ],
            dtype=torch.long,
        )
    if target_var == "answer_token":
        return torch.tensor(
            [ALPHABET_LABELS.index(str(output["answer"]).strip()) for output in bank.source_outputs],
            dtype=torch.long,
        )
    raise ValueError(f"Unsupported MCQA target variable {bank.target_var}")


def _example_label_delta_signature(bank: MCQAPairBank) -> torch.Tensor:
    target_onehot = F.one_hot(_target_answer_indices(bank), num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
    base_onehot = F.one_hot(_base_answer_indices(bank), num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
    return (target_onehot - base_onehot).reshape(-1)


def _example_label_logit_delta_signature(
    *,
    counterfactual_logits: torch.Tensor,
    base_logits: torch.Tensor,
    bank: MCQAPairBank,
) -> torch.Tensor:
    delta = _gather_label_logits(counterfactual_logits, bank) - _gather_label_logits(base_logits, bank)
    return delta.reshape(-1)


def gather_variable_logits(logits: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    target_var = canonicalize_target_var(bank.target_var)
    if target_var == "answer_pointer":
        return _gather_variant_logits(logits, bank.symbol_variant_token_ids)
    if target_var == "answer_token":
        return _gather_variant_logits(logits, bank.alphabet_variant_token_ids)
    raise ValueError(f"Unsupported MCQA target variable {bank.target_var}")


def _family_exact_accs(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    bank: MCQAPairBank,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for family_name in COUNTERFACTUAL_FAMILIES:
        mask = torch.tensor(
            [str(current_family) == str(family_name) for current_family in bank.counterfactual_family_names],
            device=predictions.device,
            dtype=torch.bool,
        )
        if bool(mask.any()):
            metrics[str(family_name)] = float((predictions[mask] == labels[mask]).float().mean().item())
    return metrics


def metrics_from_logits(logits: torch.Tensor, bank: MCQAPairBank, tokenizer=None) -> dict[str, object]:
    """Compute exact accuracy, family-wise accuracy, and optional decoded answer accuracy."""
    target_logits = gather_variable_logits(logits, bank)
    predictions = target_logits.argmax(dim=-1)
    labels = bank.labels.to(predictions.device)
    exact_acc = float((predictions == labels).float().mean().item())
    metrics: dict[str, object] = {
        "exact_acc": exact_acc,
        "family_exact_accs": _family_exact_accs(predictions, labels, bank),
    }
    if tokenizer is not None:
        target_var = canonicalize_target_var(bank.target_var)
        token_bank = bank.symbol_token_ids if target_var == "answer_pointer" else bank.alphabet_token_ids
        token_predictions = torch.gather(
            token_bank.to(logits.device),
            dim=1,
            index=predictions.view(-1, 1),
        ).view(-1)
        decoded_predictions = [
            tokenizer.decode([int(token_id)]) for token_id in token_predictions.detach().cpu().tolist()
        ]
        if target_var == "answer_pointer":
            target_token_ids = torch.gather(
                bank.symbol_token_ids.to(logits.device),
                dim=1,
                index=labels.view(-1, 1),
            ).view(-1)
        else:
            target_token_ids = bank.answer_token_ids.to(logits.device)
        decoded_targets = [
            tokenizer.decode([int(token_id)]) for token_id in target_token_ids.detach().cpu().tolist()
        ]
        decoded_acc = sum(
            int(str(expected).strip() == str(decoded).strip())
            for expected, decoded in zip(decoded_targets, decoded_predictions)
        ) / max(1, len(decoded_predictions))
        metrics["decoded_answer_acc"] = float(decoded_acc)
        metrics["checker_acc"] = float(checker_accuracy(decoded_predictions, bank.expected_answer_texts))
    return metrics


def prediction_details_from_logits(logits: torch.Tensor, bank: MCQAPairBank, tokenizer=None) -> dict[str, object]:
    target_logits = gather_variable_logits(logits, bank)
    predictions = target_logits.argmax(dim=-1)
    labels = bank.labels.to(predictions.device)
    details: dict[str, object] = {
        "labels": labels.detach().cpu().tolist(),
        "predictions": predictions.detach().cpu().tolist(),
        "correct": (predictions == labels).detach().cpu().to(torch.int64).tolist(),
        "target_logits": target_logits.detach().cpu().tolist(),
        "base_raw_inputs": [str(item["raw_input"]) for item in bank.base_inputs],
        "source_raw_inputs": [str(item["raw_input"]) for item in bank.source_inputs],
        "expected_answer_texts": list(bank.expected_answer_texts),
    }
    if tokenizer is not None:
        target_var = canonicalize_target_var(bank.target_var)
        token_bank = bank.symbol_token_ids if target_var == "answer_pointer" else bank.alphabet_token_ids
        predicted_token_ids = torch.gather(
            token_bank.to(logits.device),
            dim=1,
            index=predictions.view(-1, 1),
        ).view(-1)
        details["predicted_token_ids"] = predicted_token_ids.detach().cpu().tolist()
        details["predicted_text"] = [
            tokenizer.decode([int(token_id)]) for token_id in predicted_token_ids.detach().cpu().tolist()
        ]
    return details


def build_variable_signature(bank: MCQAPairBank, signature_mode: str) -> torch.Tensor:
    if signature_mode == "example_label_delta":
        return _example_label_delta_signature(bank)
    raise ValueError(
        f"Unsupported signature_mode={signature_mode!r}. "
        "Use 'example_label_delta' so abstract variables and neural sites share "
        "the same 26 x N alphabet-label signature space."
    )


def signature_from_logits(
    *,
    counterfactual_logits: torch.Tensor,
    base_logits: torch.Tensor,
    bank: MCQAPairBank,
    signature_mode: str,
) -> torch.Tensor:
    if signature_mode == "example_label_delta":
        return _example_label_logit_delta_signature(
            counterfactual_logits=counterfactual_logits,
            base_logits=base_logits,
            bank=bank,
        )
    raise ValueError(
        f"Unsupported signature_mode={signature_mode!r}. "
        "Use 'example_label_delta' so abstract variables and neural sites share "
        "the same 26 x N alphabet-label signature space."
    )
