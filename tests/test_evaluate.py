from copy import deepcopy

from tracegraph.graph_builder import graph_from_draft
from tracegraph.evaluate import annotation_agreement, evaluate_graphs, graph_metrics, handoff_lift
from tracegraph.schema import NodeStatus

from test_graph import fixture_graph_draft, fixture_trace


def test_graph_metrics_penalize_invalidation_and_deduplicate() -> None:
    graph = graph_from_draft(fixture_trace(), fixture_graph_draft(), "agent:test")
    for node in graph.nodes:
        node.status = NodeStatus.VERIFIED
    duplicate = deepcopy(graph.nodes[0])
    duplicate.node_id = "duplicate"
    duplicate.status = NodeStatus.INVALIDATED
    graph.nodes.append(duplicate)
    metrics = graph_metrics(graph)
    assert metrics["semantic_nodes"] == len(graph.nodes) - 1
    assert metrics["normalized_progress"] > 0


def test_annotation_agreement() -> None:
    a = [{"instance_id": "x", "semantic_key": "diagnosis:a", "status": "evidenced"}]
    b = [
        {"instance_id": "x", "semantic_key": "diagnosis:a", "status": "verified"},
        {"instance_id": "x", "semantic_key": "edit:b", "status": "evidenced"},
    ]
    result = annotation_agreement(a, b)
    assert result["semantic_node_jaccard"] == 0.5
    assert result["shared_node_status_agreement"] == 0.0


def test_handoff_lift_summary() -> None:
    records = [
        {"instance_id": "x", "arm": "prompt", "resolved": False, "cost": 2},
        {"instance_id": "x", "arm": "graph", "resolved": True, "cost": 1},
    ]
    result = handoff_lift(records)
    assert result["graph"]["success_rate"] == 1
    assert result["prompt"]["mean_cost"] == 2


def test_missing_outcome_is_not_counted_as_failure() -> None:
    graph = graph_from_draft(fixture_trace(), fixture_graph_draft(), "agent:test")
    graph.resolved = None
    result = evaluate_graphs([graph])
    assert result["aggregate"]["n"] == 1
    assert result["aggregate"]["n_labeled"] == 0
