from __future__ import annotations

"""Transitive fit-lineage validator for nested RACER-C predictions."""

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class FitNode:
    node_id: str
    stage: str
    direct_row_ids: frozenset[str]
    parent_fit_node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PredictionNode:
    prediction_id: str
    row_id: str
    outer_role: str
    root_fit_node_ids: tuple[str, ...]


def resolve_fit_rows(nodes: Iterable[FitNode]) -> dict[str, frozenset[str]]:
    materialized = list(nodes)
    by_id = {node.node_id: node for node in materialized}
    if len(by_id) != len(materialized):
        raise ValueError("duplicate fit node ID")
    resolved: dict[str, frozenset[str]] = {}
    visiting: set[str] = set()

    def visit(node_id: str) -> frozenset[str]:
        if node_id in resolved:
            return resolved[node_id]
        if node_id in visiting:
            raise ValueError(f"fit-lineage cycle at {node_id}")
        if node_id not in by_id:
            raise ValueError(f"unknown parent fit node: {node_id}")
        visiting.add(node_id)
        node = by_id[node_id]
        rows = set(node.direct_row_ids)
        for parent in node.parent_fit_node_ids:
            rows.update(visit(parent))
        visiting.remove(node_id)
        resolved[node_id] = frozenset(rows)
        return resolved[node_id]

    for node_id in by_id:
        visit(node_id)
    return resolved


def validate_prediction_lineage(
    fit_nodes: Iterable[FitNode],
    predictions: Iterable[PredictionNode],
    row_outer_roles: Mapping[str, str],
) -> dict[str, frozenset[str]]:
    fit_nodes = list(fit_nodes)
    predictions = list(predictions)
    resolved = resolve_fit_rows(fit_nodes)
    if len({item.prediction_id for item in predictions}) != len(predictions):
        raise ValueError("duplicate prediction ID")
    nondev_rows = {
        row_id for row_id, role in row_outer_roles.items() if role != "dev"
    }
    for node in fit_nodes:
        used = resolved[node.node_id]
        leaked_outer = used & nondev_rows
        if leaked_outer:
            raise ValueError(
                f"outer-role leakage in {node.node_id}: {sorted(leaked_outer)[:5]}"
            )
        unknown = used - set(row_outer_roles)
        if unknown:
            raise ValueError(f"unknown fit rows in {node.node_id}: {sorted(unknown)[:5]}")
    prediction_lineage: dict[str, frozenset[str]] = {}
    for prediction in predictions:
        if prediction.row_id not in row_outer_roles:
            raise ValueError(f"unknown prediction row: {prediction.row_id}")
        if row_outer_roles[prediction.row_id] != prediction.outer_role:
            raise ValueError(f"outer-role mismatch for {prediction.prediction_id}")
        fit_rows: set[str] = set()
        for root in prediction.root_fit_node_ids:
            if root not in resolved:
                raise ValueError(f"unknown root fit node: {root}")
            fit_rows.update(resolved[root])
        if prediction.row_id in fit_rows:
            raise ValueError(
                f"self/OOF leakage for {prediction.prediction_id}: {prediction.row_id}"
            )
        prediction_lineage[prediction.prediction_id] = frozenset(fit_rows)
    return prediction_lineage
