"""Candidate CLT sites from pruned circuit-tracer viewer graphs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .clt_backend import CLTSite


RankBy = Literal["influence", "activation"]


@dataclass(frozen=True)
class PrunedGraphCLTSite:
    """One last-token CLT feature node that survived attribution-graph pruning."""

    node_id: str
    layer: int
    feature_idx: int
    graph_feature_id: int
    ctx_idx: int
    reverse_ctx_idx: int
    influence: float
    activation: float

    @property
    def ranking_score(self) -> dict[str, float]:
        return {
            "influence": abs(float(self.influence)),
            "activation": abs(float(self.activation)),
        }

    def to_clt_site(self) -> CLTSite:
        return CLTSite(
            layer=int(self.layer),
            token_position_id="last_token",
            feature_idx=int(self.feature_idx),
        )

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def _node_float(node: dict[str, object], key: str) -> float:
    value = node.get(key, 0.0)
    return 0.0 if value is None else float(value)


def _node_int(node: dict[str, object], key: str) -> int:
    value = node.get(key)
    if value is None:
        raise ValueError(f"Pruned graph node is missing {key}: {node}")
    return int(value)


def _parse_local_site_from_node_id(node: dict[str, object]) -> tuple[int, int, int]:
    node_id = str(node.get("node_id", ""))
    parts = node_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"Expected CLT node_id to look like layer_feature_ctx, got {node_id!r}")
    layer, feature_idx, ctx_idx = (int(part) for part in parts)
    return layer, feature_idx, ctx_idx


def load_pruned_last_token_clt_sites(
    graph_json: Path,
    *,
    top_k: int | None = None,
    rank_by: RankBy = "influence",
) -> tuple[list[CLTSite], list[dict[str, object]]]:
    """Load last-token CLT feature sites from a pruned viewer graph JSON.

    The returned ``CLTSite`` objects are deduplicated by ``(layer, feature_idx)`` because all
    selected nodes are constrained to ``reverse_ctx_idx == 0`` and therefore map to the existing
    ``last_token`` token-position id used by the MCQA PLOT backend.
    """
    if rank_by not in {"influence", "activation"}:
        raise ValueError(f"Unsupported rank_by={rank_by}; expected influence or activation")
    payload = json.loads(Path(graph_json).read_text(encoding="utf-8"))
    nodes = payload.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError(f"Graph JSON {graph_json} has no node list")

    best_by_site: dict[tuple[int, int], PrunedGraphCLTSite] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if str(node.get("feature_type")) != "cross layer transcoder":
            continue
        if bool(node.get("is_target_logit", False)):
            continue
        reverse_ctx_idx = _node_int(node, "reverse_ctx_idx")
        if reverse_ctx_idx != 0:
            continue
        layer, feature_idx, ctx_idx = _parse_local_site_from_node_id(node)
        graph_layer = _node_int(node, "layer")
        graph_ctx_idx = _node_int(node, "ctx_idx")
        if graph_layer != layer or graph_ctx_idx != ctx_idx:
            raise ValueError(
                "Pruned graph node has inconsistent layer/ctx fields: "
                f"node_id={node.get('node_id')} layer={node.get('layer')} ctx_idx={node.get('ctx_idx')}"
            )
        site = PrunedGraphCLTSite(
            node_id=str(node.get("node_id", "")),
            layer=layer,
            feature_idx=feature_idx,
            graph_feature_id=_node_int(node, "feature"),
            ctx_idx=ctx_idx,
            reverse_ctx_idx=reverse_ctx_idx,
            influence=_node_float(node, "influence"),
            activation=_node_float(node, "activation"),
        )
        key = (int(site.layer), int(site.feature_idx))
        current = best_by_site.get(key)
        if current is None or site.ranking_score[rank_by] > current.ranking_score[rank_by]:
            best_by_site[key] = site

    records = sorted(
        best_by_site.values(),
        key=lambda site: (
            -site.ranking_score[rank_by],
            int(site.layer),
            int(site.feature_idx),
            str(site.node_id),
        ),
    )
    if top_k is not None:
        records = records[: max(0, int(top_k))]
    sites = [record.to_clt_site() for record in records]
    return sites, [record.to_json() for record in records]
