"""MCQA signatures and metrics ported from causal-abstractions-ot."""

from __future__ import annotations

import torch
import torch.nn.functional as F

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
    target_var = canonicalize_target_var(bank.target_var)
    if signature_mode == "whole_vocab_kl_t1":
        return bank.changed_mask.to(torch.float32)
    if signature_mode == "answer_logit_delta":
        if target_var == "answer_pointer":
            source_onehot = F.one_hot(bank.labels.to(torch.long), num_classes=4).to(torch.float32)
            base_indices = torch.tensor(
                [int(output["answer_pointer"]) for output in bank.base_outputs],
                dtype=torch.long,
            )
            return (source_onehot - F.one_hot(base_indices, num_classes=4).to(torch.float32)).reshape(-1)
        source_onehot = F.one_hot(bank.labels.to(torch.long), num_classes=26).to(torch.float32)
        base_indices = torch.tensor(
            [ALPHABET_LABELS.index(str(output["answer"]).strip()) for output in bank.base_outputs],
            dtype=torch.long,
        )
        return (source_onehot - F.one_hot(base_indices, num_classes=26).to(torch.float32)).reshape(-1)
    base_pointer_indices = torch.tensor(
        [int(output["answer_pointer"]) for output in bank.base_outputs],
        dtype=torch.long,
    )
    base_answer_indices = torch.tensor(
        [ALPHABET_LABELS.index(str(output["answer"]).strip()) for output in bank.base_outputs],
        dtype=torch.long,
    )
    if signature_mode in {"family_slot_label_delta", "family_slot_label_delta_norm"}:
        slot_delta = torch.zeros((bank.size, STRUCTURED_SLOT_DIM), dtype=torch.float32)
        label_delta = torch.zeros((bank.size, STRUCTURED_LABEL_DIM), dtype=torch.float32)
        if target_var == "answer_pointer":
            source_pointer_indices = torch.tensor(
                [int(output["answer_pointer"]) for output in bank.source_outputs],
                dtype=torch.long,
            )
            slot_delta = (
                F.one_hot(source_pointer_indices, num_classes=STRUCTURED_SLOT_DIM).to(torch.float32)
                - F.one_hot(base_pointer_indices, num_classes=STRUCTURED_SLOT_DIM).to(torch.float32)
            )
            target_label_indices = torch.tensor(
                [
                    ALPHABET_LABELS.index(str(bank.base_inputs[index][f"symbol{int(pointer)}"]).strip())
                    for index, pointer in enumerate(source_pointer_indices.tolist())
                ],
                dtype=torch.long,
            )
            label_delta = (
                F.one_hot(target_label_indices, num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
                - F.one_hot(base_answer_indices, num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
            )
        else:
            source_answer_indices = torch.tensor(
                [ALPHABET_LABELS.index(str(output["answer"]).strip()) for output in bank.source_outputs],
                dtype=torch.long,
            )
            label_delta = (
                F.one_hot(source_answer_indices, num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
                - F.one_hot(base_answer_indices, num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
            )
        return build_family_signature(
            torch.cat((slot_delta, label_delta), dim=1),
            bank,
            normalize_blocks=(signature_mode == "family_slot_label_delta_norm"),
        )
    if signature_mode in {
        "family_label_delta",
        "family_label_delta_norm",
        "family_label_logit_delta",
        "family_label_logit_delta_norm",
    }:
        if target_var == "answer_pointer":
            source_pointer_indices = torch.tensor(
                [int(output["answer_pointer"]) for output in bank.source_outputs],
                dtype=torch.long,
            )
            target_label_indices = torch.tensor(
                [
                    ALPHABET_LABELS.index(str(bank.base_inputs[index][f"symbol{int(pointer)}"]).strip())
                    for index, pointer in enumerate(source_pointer_indices.tolist())
                ],
                dtype=torch.long,
            )
        else:
            target_label_indices = torch.tensor(
                [ALPHABET_LABELS.index(str(output["answer"]).strip()) for output in bank.source_outputs],
                dtype=torch.long,
            )
        label_delta = (
            F.one_hot(target_label_indices, num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
            - F.one_hot(base_answer_indices, num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
        )
        return build_family_label_signature(
            label_delta,
            bank,
            normalize_blocks=signature_mode in {"family_label_delta_norm", "family_label_logit_delta_norm"},
        )
    raise ValueError(f"Unsupported signature_mode={signature_mode}")


def signature_from_logits(
    *,
    counterfactual_logits: torch.Tensor,
    base_logits: torch.Tensor,
    bank: MCQAPairBank,
    signature_mode: str,
) -> torch.Tensor:
    if signature_mode == "whole_vocab_kl_t1":
        base_log_probs = torch.log_softmax(base_logits, dim=-1)
        counterfactual_log_probs = torch.log_softmax(counterfactual_logits, dim=-1)
        counterfactual_probs = counterfactual_log_probs.exp()
        return torch.sum(
            counterfactual_probs * (counterfactual_log_probs - base_log_probs),
            dim=-1,
        ).reshape(-1)
    if signature_mode == "answer_logit_delta":
        return (gather_variable_logits(counterfactual_logits, bank) - gather_variable_logits(base_logits, bank)).reshape(-1)
    if signature_mode in {"family_slot_label_delta", "family_slot_label_delta_norm"}:
        delta = structured_output_features(counterfactual_logits, bank) - structured_output_features(base_logits, bank)
        return build_family_signature(
            delta,
            bank,
            normalize_blocks=(signature_mode == "family_slot_label_delta_norm"),
        )
    if signature_mode in {
        "family_label_delta",
        "family_label_delta_norm",
        "family_label_logit_delta",
        "family_label_logit_delta_norm",
    }:
        delta = _gather_label_logits(counterfactual_logits, bank) - _gather_label_logits(base_logits, bank)
        return build_family_label_signature(
            delta,
            bank,
            normalize_blocks=signature_mode in {"family_label_delta_norm", "family_label_logit_delta_norm"},
        )
    raise ValueError(f"Unsupported signature_mode={signature_mode}")
