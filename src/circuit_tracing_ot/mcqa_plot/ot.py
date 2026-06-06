"""PLOT OT/UOT selection logic ported from causal-abstractions-ot for CLT sites."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .clt_backend import (
    CLTActivationCache,
    CLTSite,
    collect_base_logits_clt,
    collect_clt_site_signatures,
    run_clt_site_intervention,
)
from .data import COUNTERFACTUAL_FAMILIES, MCQAPairBank, canonicalize_target_var
from ..logging import log_progress
from .metrics import (
    build_variable_signature,
    metrics_from_logits,
    prediction_details_from_logits,
)


@dataclass(frozen=True)
class OTConfig:
    method: str = "ot"
    epsilon: float = 1.0
    uot_beta_neural: float = 1.0
    max_iter: int = 500
    tol: float = 1e-9
    signature_mode: str = "family_label_delta_norm"
    top_k_values: tuple[int, ...] = (1,)
    lambda_values: tuple[float, ...] = (1.0,)
    source_target_vars: tuple[str, ...] = ("answer_pointer", "answer_token")
    calibration_metric: str = "family_weighted_macro_exact_acc"
    calibration_family_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    store_prediction_details: bool = True


def _squared_euclidean_cost(u_points: torch.Tensor, v_points: torch.Tensor) -> torch.Tensor:
    return torch.cdist(u_points.to(dtype=torch.float32), v_points.to(dtype=torch.float32), p=2).pow(2)


def _balanced_marginal_error(pi: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> float:
    row_error = torch.max(torch.abs(pi.sum(dim=1) - p))
    col_error = torch.max(torch.abs(pi.sum(dim=0) - q))
    return float(torch.maximum(row_error, col_error).item())


def _stack_cost_matrix(
    variable_signatures_by_var: dict[str, torch.Tensor],
    site_signatures: torch.Tensor,
    source_target_vars: tuple[str, ...],
) -> torch.Tensor:
    rows = []
    reshaped_site_signatures = site_signatures.reshape(site_signatures.shape[0], -1)
    for target_var in source_target_vars:
        variable_signature = variable_signatures_by_var[target_var].reshape(1, -1)
        rows.append(_squared_euclidean_cost(variable_signature, reshaped_site_signatures))
    return torch.cat(rows, dim=0)


def sinkhorn_from_cost_matrix(
    cost: torch.Tensor,
    *,
    p: torch.Tensor,
    q: torch.Tensor,
    epsilon: float,
    n_iter: int,
    tol: float = 1e-9,
) -> tuple[torch.Tensor, float]:
    kernel = torch.exp(-cost.to(torch.float32) / float(epsilon)).clamp_min(1e-30)
    r = torch.ones_like(p)
    c = torch.ones_like(q)
    for _ in range(int(n_iter)):
        r = p / (kernel @ c).clamp_min(1e-30)
        c = q / (kernel.transpose(0, 1) @ r).clamp_min(1e-30)
        pi = r[:, None] * kernel * c[None, :]
        if _balanced_marginal_error(pi, p, q) <= float(tol):
            break
    pi = r[:, None] * kernel * c[None, :]
    return pi, float((pi * cost).sum().item())


def sinkhorn_unbalanced_from_cost_matrix(
    cost: torch.Tensor,
    *,
    p: torch.Tensor,
    q: torch.Tensor,
    epsilon: float,
    n_iter: int,
    tau_neural: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    kernel = torch.exp(-cost.to(torch.float32) / float(epsilon)).clamp_min(1e-30)
    rho_b = float(tau_neural / (tau_neural + epsilon))
    r = torch.ones_like(p)
    c = torch.ones_like(q)
    for _ in range(int(n_iter)):
        r = p / (kernel @ c).clamp_min(1e-30)
        c = (q / (kernel.transpose(0, 1) @ r).clamp_min(1e-30)).pow(rho_b)
    pi = r[:, None] * kernel * c[None, :]
    pi_row = pi.sum(dim=1)
    pi_col = pi.sum(dim=0)
    transport_cost = float((pi * cost).sum().item())
    kl_row = float(
        (pi_row * torch.log(pi_row.clamp_min(1e-30) / p.clamp_min(1e-30)) - pi_row + p)
        .sum()
        .item()
    )
    kl_col = float(
        (pi_col * torch.log(pi_col.clamp_min(1e-30) / q.clamp_min(1e-30)) - pi_col + q)
        .sum()
        .item()
    )
    return pi, {
        "transport_cost": transport_cost,
        "kl_abstract": kl_row,
        "kl_neural": kl_col,
        "estimated_cost": transport_cost + float(tau_neural) * kl_col,
        "matched_mass": float(pi.sum().item()),
    }


def solve_ot_transport(
    variable_signatures_by_var: dict[str, torch.Tensor],
    site_signatures: torch.Tensor,
    config: OTConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    cost = _stack_cost_matrix(variable_signatures_by_var, site_signatures, config.source_target_vars)
    m, n = cost.shape
    log_progress(
        "solving balanced OT "
        f"cost_shape={tuple(cost.shape)} epsilon={float(config.epsilon):g} "
        f"max_iter={int(config.max_iter)}"
    )
    start = perf_counter()
    p = torch.full((m,), 1.0 / m, dtype=torch.float32, device=cost.device)
    q = torch.full((n,), 1.0 / n, dtype=torch.float32, device=cost.device)
    transport_tensor, transport_cost = sinkhorn_from_cost_matrix(
        cost,
        p=p,
        q=q,
        epsilon=float(config.epsilon),
        n_iter=int(config.max_iter),
        tol=float(config.tol),
    )
    transport = transport_tensor.detach().cpu().numpy()
    log_progress(
        "solved balanced OT "
        f"transport_cost={float(transport_cost):.6g} matched_mass={float(transport.sum()):.6g} "
        f"elapsed={perf_counter() - start:.2f}s"
    )
    return transport, {
        "method": "ot",
        "regularization_used": float(config.epsilon),
        "epsilon_config": float(config.epsilon),
        "transport_cost": float(transport_cost),
        "matched_mass": float(transport.sum()),
        "max_row_residual": float(np.max(np.abs(transport.sum(axis=1) - p.detach().cpu().numpy()))),
        "max_col_residual": float(np.max(np.abs(transport.sum(axis=0) - q.detach().cpu().numpy()))),
    }


def solve_uot_transport(
    variable_signatures_by_var: dict[str, torch.Tensor],
    site_signatures: torch.Tensor,
    config: OTConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    cost = _stack_cost_matrix(variable_signatures_by_var, site_signatures, config.source_target_vars)
    m, n = cost.shape
    log_progress(
        "solving unbalanced OT "
        f"cost_shape={tuple(cost.shape)} epsilon={float(config.epsilon):g} "
        f"beta_neural={float(config.uot_beta_neural):g} max_iter={int(config.max_iter)}"
    )
    start = perf_counter()
    p = torch.full((m,), 1.0 / m, dtype=torch.float32, device=cost.device)
    q = torch.full((n,), 1.0 / n, dtype=torch.float32, device=cost.device)
    transport_tensor, info = sinkhorn_unbalanced_from_cost_matrix(
        cost,
        p=p,
        q=q,
        epsilon=float(config.epsilon),
        n_iter=int(config.max_iter),
        tau_neural=float(config.uot_beta_neural),
    )
    transport = transport_tensor.detach().cpu().numpy()
    log_progress(
        "solved unbalanced OT "
        f"transport_cost={float(info['transport_cost']):.6g} "
        f"matched_mass={float(info['matched_mass']):.6g} elapsed={perf_counter() - start:.2f}s"
    )
    return transport, {
        "method": "uot",
        "regularization_used": float(config.epsilon),
        "uot_beta_neural": float(config.uot_beta_neural),
        "epsilon_config": float(config.epsilon),
        **info,
    }


def normalize_transport_rows(transport: np.ndarray) -> np.ndarray:
    row_sums = transport.sum(axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    return transport / safe_row_sums


def truncate_transport_rows(
    normalized_transport: np.ndarray,
    top_k: int,
    *,
    renormalize: bool = False,
) -> np.ndarray:
    truncated = np.zeros_like(normalized_transport)
    limit = max(1, min(int(top_k), normalized_transport.shape[1]))
    site_scores = normalized_transport.max(axis=0)
    dominant_source = normalized_transport.argmax(axis=0)
    order = np.argsort(-site_scores, kind="stable")[:limit]
    for site_index in order:
        row_index = int(dominant_source[site_index])
        truncated[row_index, site_index] = normalized_transport[row_index, site_index]
    if renormalize:
        row_sums = truncated.sum(axis=1, keepdims=True)
        safe_row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
        truncated = truncated / safe_row_sums
    return truncated


def _site_weights_from_transport(selected_transport: np.ndarray, sites: list[CLTSite]) -> dict[CLTSite, float]:
    column_mass = selected_transport.sum(axis=0)
    return {
        sites[index]: float(column_mass[index])
        for index in range(selected_transport.shape[1])
        if float(column_mass[index]) > 0.0
    }


def build_rankings(
    transport: np.ndarray,
    sites: list[CLTSite],
    ranking_k: int,
    source_target_vars: tuple[str, ...],
) -> list[dict[str, object]]:
    site_scores = transport.max(axis=0)
    dominant_source = transport.argmax(axis=0)
    order = np.argsort(-site_scores, kind="stable")[: int(ranking_k)]
    return [
        {
            "site_index": int(site_index),
            "site_label": sites[int(site_index)].label,
            "layer": int(sites[int(site_index)].layer),
            "token_position_id": str(sites[int(site_index)].token_position_id),
            "dim_start": int(sites[int(site_index)].dim_start),
            "dim_end": int(sites[int(site_index)].dim_end),
            "feature_idx": sites[int(site_index)].feature_idx,
            "transport_mass": float(site_scores[int(site_index)]),
            "dominant_source_index": int(dominant_source[int(site_index)]),
            "dominant_source_var": str(source_target_vars[int(dominant_source[int(site_index)])]),
        }
        for site_index in order
    ]


def _site_ranking_record(site: CLTSite, *, site_index: int, target_var: str) -> dict[str, object]:
    return {
        "site_index": int(site_index),
        "site_label": site.label,
        "layer": int(site.layer),
        "token_position_id": str(site.token_position_id),
        "dim_start": int(site.dim_start),
        "dim_end": int(site.dim_end),
        "feature_idx": site.feature_idx,
        "transport_mass": 1.0,
        "dominant_source_index": 0,
        "dominant_source_var": str(target_var),
    }


def prepare_alignment_artifacts_clt(
    *,
    model,
    fit_banks_by_var: dict[str, MCQAPairBank],
    sites: list[CLTSite],
    config: OTConfig,
    cache: CLTActivationCache,
    checkpoint_path: Path | None = None,
    checkpoint_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    start = perf_counter()
    reference_target_var = str(next(iter(fit_banks_by_var)))
    reference_bank = fit_banks_by_var[reference_target_var]
    site_labels = [site.label for site in sites]
    checkpoint_signatures: dict[str, torch.Tensor] = {}
    shared_base_logits = None
    metadata = {
        "kind": "mcqa_plot_clt_stage_a_signatures",
        "reference_target_var": reference_target_var,
        "signature_mode": str(config.signature_mode),
        "site_labels": site_labels,
        **(checkpoint_metadata or {}),
    }
    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(loaded, dict) and loaded.get("metadata") == metadata:
                raw_signatures = loaded.get("site_signatures_by_label", {})
                if isinstance(raw_signatures, dict):
                    checkpoint_signatures = {
                        str(label): tensor
                        for label, tensor in raw_signatures.items()
                        if isinstance(tensor, torch.Tensor)
                    }
                if isinstance(loaded.get("base_logits"), torch.Tensor):
                    shared_base_logits = loaded["base_logits"]
                log_progress(
                    "loaded Stage A signature checkpoint "
                    f"path={checkpoint_path} completed_sites={len(checkpoint_signatures)}/{len(sites)} "
                    f"has_base_logits={shared_base_logits is not None}"
                )
            else:
                log_progress(f"ignoring stale Stage A signature checkpoint path={checkpoint_path}")
        except Exception as exc:
            log_progress(
                "failed to load Stage A signature checkpoint; recomputing "
                f"path={checkpoint_path} error={type(exc).__name__}: {exc}"
            )

    def save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(
            {
                "metadata": metadata,
                "base_logits": shared_base_logits,
                "site_signatures_by_label": checkpoint_signatures,
            },
            tmp_path,
        )
        tmp_path.replace(checkpoint_path)

    log_progress(
        "CLT OT prep shared site signatures "
        f"reference_bank={reference_target_var} "
        f"examples={reference_bank.size} sites={len(sites)}"
    )
    if shared_base_logits is None:
        shared_base_logits = collect_base_logits_clt(model=model, bank=reference_bank)
        save_checkpoint()
        if checkpoint_path is not None:
            log_progress(f"saved Stage A base-logit checkpoint path={checkpoint_path}")
    else:
        log_progress(f"reusing cached Stage A base logits examples={shared_base_logits.shape[0]}")

    def on_site_signature(site: CLTSite, signature: torch.Tensor) -> None:
        checkpoint_signatures[site.label] = signature.detach().cpu()
        save_checkpoint()
        if checkpoint_path is not None:
            log_progress(
                "saved Stage A site-signature checkpoint "
                f"path={checkpoint_path} completed_sites={len(checkpoint_signatures)}/{len(sites)}"
            )

    shared_site_signatures = collect_clt_site_signatures(
        model=model,
        bank=reference_bank,
        sites=sites,
        base_logits=shared_base_logits,
        signature_mode=config.signature_mode,
        cache=cache,
        existing_signatures=checkpoint_signatures,
        on_site_signature=on_site_signature,
    )
    if checkpoint_path is not None:
        checkpoint_signatures = {
            site.label: shared_site_signatures[index].detach().cpu()
            for index, site in enumerate(sites)
        }
        save_checkpoint()
        log_progress(f"saved complete Stage A signature checkpoint path={checkpoint_path}")
    return {
        "base_logits_by_var": {str(target_var): shared_base_logits for target_var in fit_banks_by_var},
        "site_signatures": shared_site_signatures,
        "prepare_runtime_seconds": float(perf_counter() - start),
        "signature_checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
    }


def _calibration_score_from_result(result: dict[str, object], config: OTConfig) -> float:
    exact_acc = float(result.get("exact_acc", 0.0))
    if config.calibration_metric == "exact_acc":
        return exact_acc
    if config.calibration_metric == "family_weighted_macro_exact_acc":
        family_exact_accs = result.get("family_exact_accs", {})
        if not isinstance(family_exact_accs, dict):
            return exact_acc
        weighted_sum = 0.0
        total_weight = 0.0
        for family_name, weight in zip(COUNTERFACTUAL_FAMILIES, config.calibration_family_weights):
            if family_name not in family_exact_accs:
                continue
            weighted_sum += float(weight) * float(family_exact_accs[family_name])
            total_weight += float(weight)
        return exact_acc if total_weight <= 0.0 else float(weighted_sum / total_weight)
    raise ValueError(f"Unsupported calibration_metric={config.calibration_metric}")


def evaluate_single_site_intervention_clt(
    *,
    model,
    bank: MCQAPairBank,
    site: CLTSite,
    site_index: int,
    strength: float,
    tokenizer,
    cache: CLTActivationCache,
    include_details: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    log_progress(
        "evaluating single CLT site "
        f"split={bank.split} target_var={bank.target_var} site_index={site_index} "
        f"site={site.label} examples={bank.size}"
    )
    start = perf_counter()
    logits = run_clt_site_intervention(
        model=model,
        bank=bank,
        site_weights={site: 1.0},
        strength=float(strength),
        cache=cache,
    )
    ranking = [_site_ranking_record(site, site_index=site_index, target_var=bank.target_var)]
    record = {
        "method": "bruteforce",
        "variable": bank.target_var,
        "split": bank.split,
        "site_label": site.label,
        "layer": int(site.layer),
        "token_position_id": str(site.token_position_id),
        "feature_idx": site.feature_idx,
        "top_k": 1,
        "lambda": float(strength),
        "top_site_label": site.label,
        "selected_site_labels": [site.label],
        **metrics_from_logits(logits, bank, tokenizer=tokenizer),
    }
    if include_details:
        record["prediction_details"] = prediction_details_from_logits(logits, bank, tokenizer=tokenizer)
    log_progress(
        "evaluated single CLT site "
        f"target_var={bank.target_var} site={site.label} exact_acc={float(record['exact_acc']):.4f} "
        f"elapsed={perf_counter() - start:.1f}s"
    )
    return record, ranking


def _evaluate_soft_intervention_clt(
    *,
    model,
    bank: MCQAPairBank,
    sites: list[CLTSite],
    selected_transport: np.ndarray,
    top_k: int,
    strength: float,
    tokenizer,
    source_target_vars: tuple[str, ...],
    cache: CLTActivationCache,
    include_details: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    site_weights = _site_weights_from_transport(selected_transport, sites)
    logits = run_clt_site_intervention(
        model=model,
        bank=bank,
        site_weights=site_weights,
        strength=float(strength),
        cache=cache,
    )
    ranking = build_rankings(selected_transport, sites, ranking_k=max(1, top_k), source_target_vars=source_target_vars)
    record = {
        "method": "soft_transport",
        "variable": bank.target_var,
        "split": bank.split,
        "site_label": f"soft:k{int(top_k)},l{float(strength):g}",
        "top_k": int(top_k),
        "lambda": float(strength),
        "top_site_label": ranking[0]["site_label"] if ranking else None,
        "selected_site_labels": [site.label for site in site_weights],
        **metrics_from_logits(logits, bank, tokenizer=tokenizer),
    }
    if include_details:
        record["prediction_details"] = prediction_details_from_logits(logits, bank, tokenizer=tokenizer)
    return record, ranking


def select_transport_row_for_target(
    transport: np.ndarray,
    source_target_vars: tuple[str, ...],
    target_var: str,
) -> tuple[np.ndarray, tuple[str, ...], int]:
    row_index = tuple(str(variable) for variable in source_target_vars).index(str(target_var))
    return transport[row_index : row_index + 1], (str(source_target_vars[row_index]),), int(row_index)


def selection_transport_for_target(
    *,
    method: str,
    target_transport: np.ndarray,
    target_normalized_transport: np.ndarray,
) -> tuple[np.ndarray, bool]:
    if method == "uot":
        return target_transport, False
    return target_normalized_transport, True


def select_hyperparameters_clt(
    *,
    model,
    calibration_bank: MCQAPairBank,
    sites: list[CLTSite],
    selection_transport: np.ndarray,
    renormalize_selected_transport: bool,
    tokenizer,
    config: OTConfig,
    source_target_vars: tuple[str, ...],
    cache: CLTActivationCache,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    best = None
    sweep_records: list[dict[str, object]] = []
    candidates = [
        (int(top_k), float(strength))
        for top_k in config.top_k_values
        for strength in config.lambda_values
    ]
    log_progress(
        "calibration hyperparameter sweep start "
        f"target_var={calibration_bank.target_var} candidates={len(candidates)} "
        f"examples={calibration_bank.size}"
    )
    for candidate_index, (top_k, strength) in enumerate(candidates):
        log_progress(
            "calibration hyperparameter candidate "
            f"{candidate_index + 1}/{len(candidates)} target_var={calibration_bank.target_var} "
            f"top_k={top_k} lambda={strength:g}"
        )
        truncated = truncate_transport_rows(
            selection_transport,
            top_k,
            renormalize=renormalize_selected_transport,
        )
        result, ranking = _evaluate_soft_intervention_clt(
            model=model,
            bank=calibration_bank,
            sites=sites,
            selected_transport=truncated,
            top_k=top_k,
            strength=strength,
            tokenizer=tokenizer,
            source_target_vars=source_target_vars,
            cache=cache,
            include_details=False,
        )
        calibration_score = _calibration_score_from_result(result, config)
        candidate = {
            "top_k": int(top_k),
            "lambda": float(strength),
            "result": result,
            "ranking": ranking,
            "exact_acc": float(result["exact_acc"]),
            "calibration_score": float(calibration_score),
            "calibration_metric": str(config.calibration_metric),
        }
        sweep_records.append(candidate)
        log_progress(
            "calibration hyperparameter result "
            f"{candidate_index + 1}/{len(candidates)} exact_acc={float(candidate['exact_acc']):.4f} "
            f"selection_score={float(candidate['calibration_score']):.4f}"
        )
        if best is None or (
            float(candidate["calibration_score"]),
            float(candidate["exact_acc"]),
        ) > (
            float(best["calibration_score"]),
            float(best["exact_acc"]),
        ):
            best = candidate
    if best is None:
        raise RuntimeError(f"Failed to select OT/UOT hyperparameters for {calibration_bank.target_var}")
    log_progress(
        "calibration hyperparameter sweep selected "
        f"target_var={calibration_bank.target_var} top_k={int(best['top_k'])} "
        f"lambda={float(best['lambda']):g} exact_acc={float(best['exact_acc']):.4f} "
        f"selection_score={float(best['calibration_score']):.4f}"
    )
    return best, sweep_records


def run_alignment_pipeline_clt(
    *,
    model,
    fit_banks_by_var: dict[str, MCQAPairBank],
    calibration_bank: MCQAPairBank,
    holdout_bank: MCQAPairBank,
    sites: list[CLTSite],
    tokenizer,
    config: OTConfig,
    cache: CLTActivationCache,
    prepared_artifacts: dict[str, object] | None = None,
) -> dict[str, object]:
    total_start = perf_counter()
    if prepared_artifacts is None:
        prepared_artifacts = prepare_alignment_artifacts_clt(
            model=model,
            fit_banks_by_var=fit_banks_by_var,
            sites=sites,
            config=config,
            cache=cache,
        )
    variable_signatures_by_var = {
        target_var: build_variable_signature(fit_banks_by_var[target_var], config.signature_mode)
        for target_var in config.source_target_vars
    }
    site_signatures = prepared_artifacts["site_signatures"]
    transport_start = perf_counter()
    if config.method == "ot":
        transport, transport_meta = solve_ot_transport(variable_signatures_by_var, site_signatures, config)
    elif config.method == "uot":
        transport, transport_meta = solve_uot_transport(variable_signatures_by_var, site_signatures, config)
    else:
        raise ValueError(f"Unsupported method {config.method}")
    transport_solve_seconds = float(perf_counter() - transport_start)
    normalized_transport = normalize_transport_rows(transport)
    target_transport, target_source_target_vars, target_row_index = select_transport_row_for_target(
        transport,
        config.source_target_vars,
        holdout_bank.target_var,
    )
    target_normalized_transport = normalized_transport[target_row_index : target_row_index + 1]
    selection_transport, renormalize_selected_transport = selection_transport_for_target(
        method=config.method,
        target_transport=target_transport,
        target_normalized_transport=target_normalized_transport,
    )
    calibration_start = perf_counter()
    selected, calibration_sweep = select_hyperparameters_clt(
        model=model,
        calibration_bank=calibration_bank,
        sites=sites,
        selection_transport=selection_transport,
        renormalize_selected_transport=renormalize_selected_transport,
        tokenizer=tokenizer,
        config=config,
        source_target_vars=target_source_target_vars,
        cache=cache,
    )
    calibration_seconds = float(perf_counter() - calibration_start)
    top_k = int(selected["top_k"])
    strength = float(selected["lambda"])
    selected_transport = truncate_transport_rows(
        selection_transport,
        top_k,
        renormalize=renormalize_selected_transport,
    )
    holdout_start = perf_counter()
    holdout_result, holdout_ranking = _evaluate_soft_intervention_clt(
        model=model,
        bank=holdout_bank,
        sites=sites,
        selected_transport=selected_transport,
        top_k=top_k,
        strength=strength,
        tokenizer=tokenizer,
        source_target_vars=target_source_target_vars,
        cache=cache,
        include_details=bool(config.store_prediction_details),
    )
    holdout_seconds = float(perf_counter() - holdout_start)
    selected_calibration_result = dict(selected["result"])
    holdout_result["method"] = config.method
    holdout_result["selection_exact_acc"] = float(selected_calibration_result["exact_acc"])
    holdout_result["calibration_exact_acc"] = float(selected_calibration_result["exact_acc"])
    holdout_result["selection_score"] = float(selected["calibration_score"])
    holdout_result["calibration_metric"] = str(config.calibration_metric)
    holdout_result["signature_mode"] = str(config.signature_mode)
    return {
        "target_var": holdout_bank.target_var,
        "source_target_vars": list(config.source_target_vars),
        "target_var_row_index": int(target_row_index),
        "signature_mode": config.signature_mode,
        "calibration_metric": config.calibration_metric,
        "calibration_family_weights": [float(weight) for weight in config.calibration_family_weights],
        "transport": transport.tolist(),
        "normalized_transport": normalized_transport.tolist(),
        "target_transport": target_transport.tolist(),
        "target_normalized_transport": target_normalized_transport.tolist(),
        "selection_transport": selection_transport.tolist(),
        "selection_transport_renormalized": bool(renormalize_selected_transport),
        "selected_transport": selected_transport.tolist(),
        "transport_meta": transport_meta,
        "signature_prepare_runtime_seconds": float(prepared_artifacts.get("prepare_runtime_seconds", 0.0)),
        "wall_runtime_seconds": float(perf_counter() - total_start),
        "runtime_seconds": float(perf_counter() - total_start),
        "timing_seconds": {
            "t_signature_prepare": float(prepared_artifacts.get("prepare_runtime_seconds", 0.0)),
            "t_transport_solve": float(transport_solve_seconds),
            "t_calibration_select": float(calibration_seconds),
            "t_final_holdout_eval": float(holdout_seconds),
        },
        "selected_hyperparameters": {
            "top_k": top_k,
            "lambda": strength,
            "signature_mode": config.signature_mode,
            "calibration_metric": config.calibration_metric,
        },
        "selected_calibration_result": selected_calibration_result,
        "selected_calibration_ranking": selected.get("ranking", []),
        "ranking": holdout_ranking,
        "calibration_sweep": calibration_sweep,
        "results": [holdout_result],
    }


def target_row_ranking(payload: dict[str, object], *, sites: list[CLTSite]) -> list[dict[str, object]]:
    target_row_transport = payload.get("target_normalized_transport", payload.get("target_transport", []))
    if not isinstance(target_row_transport, list) or not target_row_transport:
        return []
    row_transport = target_row_transport[0] if isinstance(target_row_transport[0], list) else target_row_transport
    ranking = []
    for site_index in range(min(len(sites), len(row_transport))):
        site = sites[int(site_index)]
        ranking.append(
            {
                "site_index": int(site_index),
                "site_label": site.label,
                "layer": int(site.layer),
                "token_position_id": str(site.token_position_id),
                "feature_idx": site.feature_idx,
                "transport_mass": float(row_transport[site_index]),
            }
        )
    return sorted(
        ranking,
        key=lambda entry: (float(entry["transport_mass"]), -int(entry["site_index"])),
        reverse=True,
    )


def stage_a_calibration_score(
    *,
    result: dict[str, object],
    calibration_family_weights: tuple[float, ...],
) -> float:
    exact_acc = float(result.get("exact_acc", 0.0))
    family_exact_accs = result.get("family_exact_accs", {})
    if not isinstance(family_exact_accs, dict):
        return exact_acc
    weighted_sum = 0.0
    total_weight = 0.0
    for family_name, weight in zip(COUNTERFACTUAL_FAMILIES, calibration_family_weights):
        if family_name not in family_exact_accs:
            continue
        weighted_sum += float(weight) * float(family_exact_accs[family_name])
        total_weight += float(weight)
    return exact_acc if total_weight <= 0.0 else float(weighted_sum / total_weight)


def canonical_source_target_vars(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(canonicalize_target_var(value) for value in values)
