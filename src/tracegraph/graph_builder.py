from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from .graph_spec import DEFAULT_GRAPH_SPEC, spec_metadata
from .schema import (
    AccomplishmentGraph,
    AccomplishmentNode,
    Evidence,
    GraphEdge,
    GraphSpec,
    NodeStatus,
    Trace,
)


NODE_TYPES = DEFAULT_GRAPH_SPEC.node_name_set()
EDGE_TYPES = DEFAULT_GRAPH_SPEC.edge_name_set()
PLACEHOLDER_TEXT = {"c", "n", "node", "test", "todo", "x"}
MIN_CLAIM_CHARS = 8
MIN_EXCERPT_CHARS = 3


def is_placeholder_text(text: str, *, min_chars: int, require_space: bool = False) -> bool:
    """True for dummy claims/excerpts such as 'test' or 'x'."""
    value = " ".join(str(text or "").split()).strip().lower()
    if not value:
        return False
    if value in PLACEHOLDER_TEXT or len(value) < min_chars:
        return True
    return require_space and " " not in value and len(value) < 12


def is_placeholder_node(claim: str, excerpts: list[str] | None = None) -> bool:
    if is_placeholder_text(claim, min_chars=MIN_CLAIM_CHARS, require_space=True):
        return True
    return any(
        is_placeholder_text(excerpt, min_chars=MIN_EXCERPT_CHARS) for excerpt in excerpts or []
    )


def is_placeholder_graph(graph: AccomplishmentGraph) -> bool:
    """True when every node is a dummy 'test'/'x' graph, including 1-node junk."""
    return bool(graph.nodes) and all(
        is_placeholder_node(node.claim, [item.excerpt for item in node.evidence])
        for node in graph.nodes
    )


def coerce_json_value(value: Any) -> Any:
    """Parse a JSON string once; leave non-strings and invalid JSON unchanged."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def coerce_json_array(value: Any) -> Any:
    """Turn a JSON-encoded array (or array of JSON-encoded objects) into a list."""
    value = coerce_json_value(value)
    if isinstance(value, list):
        return [coerce_json_value(item) for item in value]
    return value


def normalize_graph_draft(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Copy a submit_graph payload and coerce nodes/edges into real arrays."""
    payload = dict(raw or {})
    payload["nodes"] = coerce_json_array(payload.get("nodes"))
    payload["edges"] = coerce_json_array(payload.get("edges"))
    return payload


def semantic_key(claim: str, node_type: str) -> str:
    terms = re.findall(r"[a-z0-9_./]+", claim.lower())
    stop = {"the", "a", "an", "to", "in", "and", "of", "for", "with", "agent"}
    return f"{node_type}:" + " ".join(term for term in terms if term not in stop)[:160]


def default_weight(node_type: str, spec: GraphSpec | None = None) -> float:
    return (spec or DEFAULT_GRAPH_SPEC).weight_for(str(node_type))


def trace_metadata(trace: Trace, decomposer: str, **extra: Any) -> dict[str, Any]:
    counts = {event_type: 0 for event_type in ("search", "read", "edit", "test", "submission")}
    for event in trace.events:
        if event.type.value in counts:
            counts[event.type.value] += 1
    return {
        "decomposer": decomposer,
        "event_count": len(trace.events),
        "api_calls": trace.api_calls,
        **{f"{name}_count": count for name, count in counts.items()},
        **extra,
    }


def _has_cycle(node_ids: set[str], edges: list[dict[str, Any]]) -> bool:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        if source in node_ids and target in node_ids:
            adjacency[str(source)].append(str(target))
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(dfs(child) for child in adjacency.get(node, [])):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(dfs(node_id) for node_id in node_ids if node_id not in visited)


def validate_graph_draft(
    trace: Trace,
    raw: dict[str, Any],
    spec: GraphSpec | None = None,
) -> list[str]:
    spec = spec or DEFAULT_GRAPH_SPEC
    event_ids = {event.event_id for event in trace.events}
    errors: list[str] = []
    raw = normalize_graph_draft(raw)
    raw_nodes = raw.get("nodes")
    raw_edges = raw.get("edges")
    if not isinstance(raw_nodes, list):
        errors.append(
            f"nodes must be a JSON array of objects, not {type(raw_nodes).__name__}."
        )
    if not isinstance(raw_edges, list):
        errors.append(
            f"edges must be a JSON array of objects, not {type(raw_edges).__name__}."
        )
    if errors:
        return errors

    nodes = raw_nodes
    edges = raw_edges
    allowed_nodes = spec.node_name_set()
    allowed_edges = spec.edge_name_set()
    if not nodes:
        errors.append("Graph must contain at least one node.")
    if spec.require_edges_if_multiple_nodes and len(nodes) > 1 and not edges:
        errors.append(
            "A graph with more than one node must include causal edges; "
            "independent roots are allowed, but do not submit edges: []."
        )
    seen_ids: set[str] = set()
    node_types: dict[str, str] = {}
    for index, item in enumerate(nodes):
        if not isinstance(item, dict):
            errors.append(f"Node {index} must be an object, not {type(item).__name__}.")
            continue
        node_id = str(item.get("node_id") or f"n{index:03d}")
        if node_id in seen_ids:
            errors.append(f"Duplicate node_id: {node_id}")
        seen_ids.add(node_id)
        node_type = item.get("type")
        if node_type not in allowed_nodes:
            errors.append(
                f"Node {node_id} has invalid type: {node_type}. "
                f"Allowed: {sorted(allowed_nodes)}."
            )
        else:
            node_types[node_id] = str(node_type)
        span = item.get("event_span") or []
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            errors.append(f"Node {node_id} must have event_span [start, end].")
            continue
        if not all(event_id in event_ids for event_id in span):
            errors.append(f"Node {node_id} cites unknown event ids in event_span.")
        evidence = item.get("evidence") or []
        if not evidence:
            errors.append(f"Node {node_id} must cite at least one evidence event.")
        for evidence_item in evidence:
            if not isinstance(evidence_item, dict):
                errors.append(f"Node {node_id} has a non-object evidence item.")
                continue
            event_id = evidence_item.get("event_id")
            if event_id not in event_ids:
                errors.append(f"Node {node_id} cites unknown evidence event_id {event_id}.")
        claim = str(item.get("claim") or "").strip()
        if not claim:
            errors.append(f"Node {node_id} must include a non-empty claim.")
        excerpts = [
            str(evidence_item.get("excerpt") or "")
            for evidence_item in evidence
            if isinstance(evidence_item, dict)
        ]
        if is_placeholder_node(claim, excerpts):
            errors.append(
                f"Node {node_id} looks like a placeholder (claim/excerpt such as 'test' or 'x'). "
                "Write a specific accomplishment sentence and quote real event text."
            )

    valid_edges: list[dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            errors.append(f"Edge must be an object, not {type(edge).__name__}.")
            continue
        edge_type = edge.get("type")
        if edge_type not in allowed_edges:
            errors.append(f"Invalid edge type: {edge_type}. Allowed: {sorted(allowed_edges)}.")
        if edge.get("source") not in seen_ids or edge.get("target") not in seen_ids:
            errors.append(
                f"Edge {edge.get('source')} -> {edge.get('target')} references unknown nodes."
            )
            continue
        edge_spec = spec.edge_spec(str(edge_type)) if edge_type in allowed_edges else None
        if edge_spec is not None:
            source_type = node_types.get(str(edge.get("source")))
            target_type = node_types.get(str(edge.get("target")))
            if edge_spec.source_types and source_type not in edge_spec.source_types:
                errors.append(
                    f"Edge {edge.get('source')} -> {edge.get('target')} type {edge_type} "
                    f"requires source types {edge_spec.source_types}, got {source_type}."
                )
            if edge_spec.target_types and target_type not in edge_spec.target_types:
                errors.append(
                    f"Edge {edge.get('source')} -> {edge.get('target')} type {edge_type} "
                    f"requires target types {edge_spec.target_types}, got {target_type}."
                )
        valid_edges.append(edge)

    if spec.require_dag and seen_ids and _has_cycle(seen_ids, valid_edges):
        errors.append("Graph contains a cycle; submit a DAG.")
    return errors


def graph_from_draft(
    trace: Trace,
    raw: dict[str, Any],
    decomposer: str,
    spec: GraphSpec | None = None,
    **metadata: Any,
) -> AccomplishmentGraph:
    spec = spec or DEFAULT_GRAPH_SPEC
    raw = normalize_graph_draft(raw)
    errors = validate_graph_draft(trace, raw, spec)
    if errors:
        raise ValueError("; ".join(errors))
    event_ids = {event.event_id for event in trace.events}
    nodes: list[AccomplishmentNode] = []
    for index, item in enumerate(raw.get("nodes", [])):
        span = item["event_span"]
        node_type = str(item["type"])
        claim = str(item["claim"]).strip()
        evidence = [
            Evidence(
                event_id=e["event_id"],
                kind=e.get("kind", "semantic"),
                excerpt=str(e.get("excerpt", ""))[:400],
            )
            for e in item.get("evidence", [])
            if e.get("event_id") in event_ids
        ]
        nodes.append(
            AccomplishmentNode(
                node_id=str(item.get("node_id") or f"n{index:03d}"),
                claim=claim,
                type=node_type,
                event_span=(span[0], span[1]),
                evidence=evidence,
                artifact_delta=item.get("artifact_delta", {}),
                status=NodeStatus.CLAIMED,
                confidence=min(1.0, max(0.0, float(item.get("confidence", 0.5)))),
                remaining_work=str(item.get("remaining_work", "")),
                semantic_key=semantic_key(claim, node_type),
                weight=default_weight(node_type, spec),
                segment_ids=[str(value) for value in item.get("segment_ids") or []],
            )
        )
    valid_nodes = {node.node_id for node in nodes}
    edges = [
        GraphEdge(e["source"], e["target"], str(e["type"]))
        for e in raw.get("edges", [])
        if e.get("source") in valid_nodes and e.get("target") in valid_nodes
    ]
    extra = {**spec_metadata(spec), **metadata}
    return AccomplishmentGraph(
        instance_id=trace.instance_id,
        model=trace.model,
        nodes=nodes,
        edges=edges,
        resolved=trace.resolved,
        metadata=trace_metadata(trace, decomposer, **extra),
    )
