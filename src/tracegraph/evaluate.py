from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from .graph_builder import is_placeholder_graph
from .schema import AccomplishmentGraph, NodeStatus, NodeType

STATUS_CREDIT = {
    NodeStatus.CLAIMED: 0.0,
    NodeStatus.EVIDENCED: 0.0,
    NodeStatus.VERIFIED: 1.0,
    NodeStatus.SUPERSEDED: 0.0,
    NodeStatus.INVALIDATED: -0.5,
}


def graph_metrics(graph: AccomplishmentGraph) -> dict[str, float | int]:
    canonical: dict[str, Any] = {}
    for node in graph.nodes:
        prior = canonical.get(node.semantic_key)
        if prior is None or STATUS_CREDIT[node.status] > STATUS_CREDIT[prior.status]:
            canonical[node.semantic_key] = node
    positive_weight = sum(node.weight for node in canonical.values())
    earned = sum(node.weight * STATUS_CREDIT[node.status] for node in canonical.values())
    evidenced = sum(node.status in {NodeStatus.EVIDENCED, NodeStatus.VERIFIED} for node in canonical.values())
    claimed = sum(node.status == NodeStatus.CLAIMED for node in canonical.values())
    verified = sum(node.status == NodeStatus.VERIFIED for node in canonical.values())
    invalidated = sum(node.status == NodeStatus.INVALIDATED for node in canonical.values())
    return {
        "semantic_nodes": len(canonical),
        "verified_nodes": verified,
        "evidenced_nodes": evidenced,
        "claimed_nodes": claimed,
        "invalidated_nodes": invalidated,
        "verified_precision": verified / len(canonical) if canonical else 0.0,
        "evidence_precision": evidenced / len(canonical) if canonical else 0.0,
        "invalidation_rate": invalidated / len(canonical) if canonical else 0.0,
        "normalized_progress": max(-1.0, min(1.0, earned / positive_weight)) if positive_weight else 0.0,
        "progress_auc": progress_auc(graph),
    }


def progress_auc(graph: AccomplishmentGraph) -> float:
    if not graph.nodes:
        return 0.0
    points: list[tuple[int, float]] = []
    running: dict[str, float] = {}
    total_weight = sum(node.weight for node in graph.nodes) or 1.0
    for node in sorted(graph.nodes, key=lambda n: _event_number(n.event_span[1])):
        running[node.semantic_key] = node.weight * STATUS_CREDIT[node.status]
        points.append((_event_number(node.event_span[1]), sum(running.values()) / total_weight))
    if len(points) == 1:
        return points[0][1]
    horizon = max(1, points[-1][0])
    area = 0.0
    previous_x, previous_y = 0, 0.0
    for x, y in points:
        area += (x - previous_x) * (previous_y + y) / 2
        previous_x, previous_y = x, y
    return area / horizon


def usable_graphs(graphs: list[AccomplishmentGraph]) -> list[AccomplishmentGraph]:
    """Drop dummy 1-node 'test'/'x' graphs from scoring."""
    return [graph for graph in graphs if not is_placeholder_graph(graph)]


def evaluate_graphs(graphs: list[AccomplishmentGraph]) -> dict[str, Any]:
    graphs = usable_graphs(graphs)
    rows = [
        {
            "instance_id": graph.instance_id,
            "model": graph.model,
            "resolved": graph.resolved,
            **graph_metrics(graph),
            "event_count": int(graph.metadata.get("event_count", 0)),
            "search_count": int(graph.metadata.get("search_count", 0)),
            "read_count": int(graph.metadata.get("read_count", 0)),
            "edit_count": int(graph.metadata.get("edit_count", 0)),
            "test_count": int(graph.metadata.get("test_count", 0)),
        }
        for graph in graphs
    ]
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    return {
        "n_trajectories": len(rows),
        "n_tasks": len({row["instance_id"] for row in rows}),
        "aggregate": _aggregate(rows),
        "by_model": {model: _aggregate(group) for model, group in sorted(by_model.items())},
        "predictive_validity": predictive_validity(rows),
        "records": rows,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {}
    metric_names = (
        "normalized_progress",
        "progress_auc",
        "verified_precision",
        "evidence_precision",
        "invalidation_rate",
    )
    labeled = [row for row in rows if row["resolved"] is not None]
    return {
        "n": len(rows),
        "n_labeled": len(labeled),
        "resolved_rate": (
            sum(bool(row["resolved"]) for row in labeled) / len(labeled) if labeled else math.nan
        ),
        **{f"mean_{name}": float(np.mean([row[name] for row in rows])) for name in metric_names},
    }


def predictive_validity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in rows if row["resolved"] is not None]
    labels = np.asarray([int(row["resolved"]) for row in rows])
    if len(rows) < 12 or len(np.unique(labels)) < 2:
        return {"available": False, "reason": "Need at least 12 rows from both outcome classes."}
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    groups = np.asarray([row["instance_id"] for row in rows])
    unique_groups = len(np.unique(groups))
    if unique_groups < 3:
        return {"available": False, "reason": "Need at least three task groups."}
    folds = min(5, unique_groups)
    baseline = np.asarray(
        [
            [
                row["event_count"],
                row["search_count"],
                row["read_count"],
                row["edit_count"],
                row["test_count"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    graph_features = np.asarray(
        [
            [
                row["event_count"],
                row["search_count"],
                row["read_count"],
                row["edit_count"],
                row["test_count"],
                row["normalized_progress"],
                row["progress_auc"],
                row["invalidation_rate"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    splitter = GroupKFold(n_splits=folds)

    def score(features: np.ndarray) -> float:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        try:
            predictions = cross_val_predict(
                model, features, labels, groups=groups, cv=splitter, method="predict_proba"
            )[:, 1]
            return float(roc_auc_score(labels, predictions))
        except ValueError:
            return math.nan

    baseline_auc = score(baseline)
    graph_auc = score(graph_features)
    return {
        "available": True,
        "grouped_folds": folds,
        "baseline_roc_auc": baseline_auc,
        "graph_roc_auc": graph_auc,
        "incremental_roc_auc": graph_auc - baseline_auc,
    }


def annotation_agreement(
    annotations_a: list[dict[str, Any]], annotations_b: list[dict[str, Any]]
) -> dict[str, float]:
    by_trace_a = _annotation_sets(annotations_a)
    by_trace_b = _annotation_sets(annotations_b)
    traces = set(by_trace_a) | set(by_trace_b)
    intersections = unions = exact_status = shared = 0
    for trace in traces:
        keys_a = set(by_trace_a.get(trace, {}))
        keys_b = set(by_trace_b.get(trace, {}))
        intersections += len(keys_a & keys_b)
        unions += len(keys_a | keys_b)
        for key in keys_a & keys_b:
            shared += 1
            exact_status += by_trace_a[trace][key] == by_trace_b[trace][key]
    return {
        "semantic_node_jaccard": intersections / unions if unions else 1.0,
        "shared_node_status_agreement": exact_status / shared if shared else 1.0,
    }


def handoff_lift(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize paired continuation trials keyed by instance_id and arm."""
    arms: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        arms[str(record["arm"])].append(record)
    return {
        arm: {
            "n": len(group),
            "success_rate": sum(bool(row["resolved"]) for row in group) / len(group),
            "mean_cost": float(np.mean([float(row["cost"]) for row in group])),
        }
        for arm, group in sorted(arms.items())
        if group
    }


def _annotation_sets(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        result[str(row["instance_id"])][str(row["semantic_key"])] = str(row["status"])
    return result


def _event_number(event_id: str) -> int:
    digits = "".join(character for character in event_id if character.isdigit())
    return int(digits or 0)
