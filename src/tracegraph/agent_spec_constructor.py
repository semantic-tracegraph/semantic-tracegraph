from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .agent_segmenter import WorkSegment, segment_trace, trace_key
from .agent_trace import agent_runtime, assistant_message, complete_chat, parse_tool_arguments
from .graph_builder import coerce_json_array
from .graph_spec import validate_graph_spec
from .sandbox import TraceSandbox
from .schema import GraphSpec, Trace, coerce_string_list, graph_spec_from_dict, to_dict


@dataclass(slots=True)
class SegmentCluster:
    """A semantic class of similar work segments; one cluster becomes one node type."""

    node_type: str
    member_segment_ids: list[str]
    description: str
    similarity_basis: list[str]
    distinguishing_criteria: list[str]
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    evidence_requirements: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    singleton_justification: str = ""
    default_weight: float = 1.0


CLUSTERING_SYSTEM_PROMPT = """You construct a reusable graph spec by clustering normalized work
segments from several coding-agent trajectories.

This is the second stage. Every input segment was produced independently without candidate node
types. Compare segments semantically, partition them into classes, and make each class one node
type.

Compare segments using:
1. Intent: what local sub-goal was pursued?
2. State transition: what changed between the start and end of the segment?
3. Evidence pattern: what observations establish that work (for example, locating a site,
   observing a failure, changing source, checking behavior, or producing an artifact)?

Ignore superficial similarity:
- repository, file, and symbol names;
- exact shell commands or tool syntax;
- chronology by itself;
- instance-specific bug details.

Clustering rules:
- Every segment must belong to exactly one cluster.
- A cluster must be semantically coherent and distinguishable from nearby clusters.
- Node type names describe reusable accomplishment kinds, not commands or specific bugs.
- similarity_basis must explain what members share.
- distinguishing_criteria must explain why the cluster is separate from its nearest alternatives.
- A singleton cluster is allowed only with a concrete singleton_justification explaining why the
  class should recur in unseen trajectories.
- Prefer a compact ontology; merge clusters whose only differences are command/file details.
- Do not force together segments with materially different goals or state transitions.

After clustering, infer edge types from recurring causal relationships and transitions between
the segment classes. Prefer reusable semantic relations over a separate edge name for every
pair of node types. Mark prerequisite/artifact/refinement relations is_dependency=true and
contradiction/invalidation relations is_dependency=false.

Call `compare_segments` when raw span context is useful. Call `submit_clustering` with the
partition, edge types, usual order, and graph rules. The node type definitions in the final
GraphSpec will be constructed mechanically from your clusters."""


CLUSTERING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_segments",
            "description": "List all normalized work segments that must be clustered.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_segments",
            "description": (
                "Inspect two or more normalized segments side by side, including their raw "
                "trajectory spans. Use this to test semantic similarity or resolve boundaries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                    }
                },
                "required": ["segment_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_clustering",
            "description": (
                "Submit a complete partition of segments into node-type classes, plus reusable "
                "edge definitions and graph rules."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "spec_id": {"type": "string"},
                    "version": {"type": "string"},
                    "notes": {"type": "string"},
                    "typical_order": {"type": "array", "items": {"type": "string"}},
                    "require_dag": {"type": "boolean"},
                    "allow_disconnected_roots": {"type": "boolean"},
                    "require_edges_if_multiple_nodes": {"type": "boolean"},
                    "clusters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_type": {"type": "string"},
                                "member_segment_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "description": {"type": "string"},
                                "similarity_basis": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "distinguishing_criteria": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "inclusion_criteria": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "exclusion_criteria": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "evidence_requirements": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "examples": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "singleton_justification": {"type": "string"},
                                "default_weight": {"type": "number"},
                            },
                            "required": [
                                "node_type",
                                "member_segment_ids",
                                "description",
                                "similarity_basis",
                                "distinguishing_criteria",
                            ],
                        },
                    },
                    "edge_types": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "is_dependency": {"type": "boolean"},
                                "source_types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "target_types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["name", "description", "is_dependency"],
                        },
                    },
                },
                "required": [
                    "spec_id",
                    "version",
                    "clusters",
                    "edge_types",
                    "typical_order",
                ],
            },
        },
    },
]


def _parse_cluster(raw: dict[str, Any]) -> SegmentCluster:
    return SegmentCluster(
        node_type=str(raw.get("node_type") or "").strip(),
        member_segment_ids=coerce_string_list(raw.get("member_segment_ids")),
        description=str(raw.get("description") or "").strip(),
        similarity_basis=coerce_string_list(raw.get("similarity_basis")),
        distinguishing_criteria=coerce_string_list(raw.get("distinguishing_criteria")),
        inclusion_criteria=coerce_string_list(raw.get("inclusion_criteria")),
        exclusion_criteria=coerce_string_list(raw.get("exclusion_criteria")),
        evidence_requirements=coerce_string_list(raw.get("evidence_requirements")),
        examples=coerce_string_list(raw.get("examples")),
        singleton_justification=str(raw.get("singleton_justification") or "").strip(),
        default_weight=float(raw.get("default_weight", 1.0)),
    )


def graph_spec_from_clustering(
    payload: dict[str, Any],
    clusters: list[SegmentCluster],
) -> GraphSpec:
    """Build the runtime GraphSpec from clusters so classes and node types cannot diverge."""
    raw_spec = {
        "spec_id": payload.get("spec_id"),
        "version": payload.get("version"),
        "notes": payload.get("notes", ""),
        "typical_order": payload.get("typical_order") or [],
        "require_dag": payload.get("require_dag", True),
        "allow_disconnected_roots": payload.get("allow_disconnected_roots", True),
        "require_edges_if_multiple_nodes": payload.get(
            "require_edges_if_multiple_nodes", True
        ),
        "node_types": [
            {
                "name": cluster.node_type,
                "description": cluster.description,
                "inclusion_criteria": cluster.inclusion_criteria,
                "exclusion_criteria": cluster.exclusion_criteria,
                "evidence_requirements": cluster.evidence_requirements,
                "examples": cluster.examples,
                "default_weight": cluster.default_weight,
            }
            for cluster in clusters
        ],
        "edge_types": payload.get("edge_types") or [],
    }
    return graph_spec_from_dict(raw_spec)


def validate_clustering(
    segments: list[WorkSegment],
    payload: dict[str, Any],
) -> tuple[list[str], list[SegmentCluster], GraphSpec | None]:
    """Require clusters to form an auditable partition before producing a GraphSpec."""
    raw_clusters = coerce_json_array(payload.get("clusters"))
    raw_edges = coerce_json_array(payload.get("edge_types"))
    typical_order = coerce_json_array(payload.get("typical_order") or [])
    normalized = {
        **payload,
        "clusters": raw_clusters,
        "edge_types": raw_edges,
        "typical_order": typical_order,
    }
    if not isinstance(raw_clusters, list):
        return ["clusters must be a JSON array."], [], None
    if not isinstance(raw_edges, list):
        return ["edge_types must be a JSON array."], [], None
    if not isinstance(typical_order, list):
        return ["typical_order must be a JSON array."], [], None

    errors: list[str] = []
    clusters: list[SegmentCluster] = []
    for index, raw in enumerate(raw_clusters):
        if not isinstance(raw, dict):
            errors.append(f"Cluster {index} must be an object.")
            continue
        try:
            cluster = _parse_cluster(raw)
        except (TypeError, ValueError) as exc:
            errors.append(f"Cluster {index} could not be parsed: {exc}")
            continue
        clusters.append(cluster)
        if not cluster.member_segment_ids:
            errors.append(f"Cluster {cluster.node_type or index} has no members.")
        if not cluster.description:
            errors.append(f"Cluster {cluster.node_type or index} needs a description.")
        if not any(cluster.similarity_basis):
            errors.append(f"Cluster {cluster.node_type or index} needs similarity_basis.")
        if not any(cluster.distinguishing_criteria):
            errors.append(
                f"Cluster {cluster.node_type or index} needs distinguishing_criteria."
            )
        if (
            len(cluster.member_segment_ids) == 1
            and not cluster.singleton_justification
        ):
            errors.append(
                f"Singleton cluster {cluster.node_type or index} needs "
                "singleton_justification."
            )

    expected_ids = {segment.segment_id for segment in segments}
    assignments: dict[str, list[str]] = {}
    for cluster in clusters:
        for segment_id in cluster.member_segment_ids:
            assignments.setdefault(segment_id, []).append(cluster.node_type)
    unknown = sorted(set(assignments) - expected_ids)
    missing = sorted(expected_ids - set(assignments))
    duplicates = {
        segment_id: names
        for segment_id, names in assignments.items()
        if len(names) > 1
    }
    if unknown:
        errors.append(f"Clusters reference unknown segment IDs: {unknown}.")
    if missing:
        errors.append(f"Segments are not assigned to a cluster: {missing}.")
    if duplicates:
        errors.append(f"Segments are assigned to multiple clusters: {duplicates}.")

    spec: GraphSpec | None = None
    try:
        spec = graph_spec_from_clustering(normalized, clusters)
    except (TypeError, KeyError, ValueError) as exc:
        errors.append(f"Graph spec could not be built from clusters: {exc}")
    if spec is not None:
        errors.extend(validate_graph_spec(spec))
    return errors, clusters, spec


class AgentSpecConstructor:
    """Construct a GraphSpec by first segmenting traces, then clustering segments."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        max_steps: int = 24,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        session_id: str | None = None,
        user_agent: str | None = None,
        segmentation_max_steps: int = 8,
    ) -> None:
        runtime = agent_runtime(
            model=model,
            api_key=api_key,
            api_base=api_base,
            session_id=session_id,
            user_agent=user_agent,
        )
        self.model = runtime["model"]
        self.api_key = runtime["api_key"]
        self.api_base = runtime["api_base"]
        self.max_steps = max_steps
        self.segmentation_max_steps = segmentation_max_steps
        self.transport = transport
        self.session_id = runtime["session_id"]
        self.user_agent = runtime["user_agent"]

    def construct(self, traces: list[Trace]) -> tuple[GraphSpec, dict[str, Any]]:
        if not traces:
            raise ValueError("AgentSpecConstructor requires at least one calibration trace.")

        segments: list[WorkSegment] = []
        segmentation_traces: dict[str, Any] = {}
        sandboxes: dict[str, TraceSandbox] = {}
        for trace_index, trace in enumerate(traces):
            key = trace_key(trace)
            sandbox = TraceSandbox(trace)
            sandboxes[key] = sandbox
            trace_segments, phase_trace = segment_trace(
                trace,
                sandbox=sandbox,
                model=self.model,
                api_key=self.api_key,
                api_base=self.api_base,
                session_id=f"{self.session_id}-segment-{trace_index}",
                user_agent=self.user_agent,
                max_steps=self.segmentation_max_steps,
                transport=self.transport,
                id_prefix=f"t{trace_index:03d}_s",
            )
            segments.extend(trace_segments)
            segmentation_traces[key] = phase_trace

        if not segments:
            raise RuntimeError(
                "Segmentation found no task-relevant work in the calibration traces."
            )

        spec, clusters, clustering_trace = self._cluster_segments(
            segments,
            sandboxes,
        )
        total_steps = sum(
            int(trace["steps"]) for trace in segmentation_traces.values()
        ) + int(clustering_trace["steps"])
        construction_trace = {
            "agent": "spec_constructor",
            "model": self.model,
            "session_id": self.session_id,
            "steps": total_steps,
            "max_steps": {
                "segmentation_per_trace": self.segmentation_max_steps,
                "clustering": self.max_steps,
            },
            "calibration_keys": [trace_key(trace) for trace in traces],
            "segments": [to_dict(segment) for segment in segments],
            "clusters": [to_dict(cluster) for cluster in clusters],
            "segmentation": segmentation_traces,
            "clustering": clustering_trace,
            "spec": to_dict(spec),
        }
        return spec, construction_trace

    def _cluster_segments(
        self,
        segments: list[WorkSegment],
        sandboxes: dict[str, TraceSandbox],
    ) -> tuple[GraphSpec, list[SegmentCluster], dict[str, Any]]:
        segment_map = {segment.segment_id: segment for segment in segments}
        segment_payloads = [to_dict(segment) for segment in segments]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": CLUSTERING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Cluster every normalized work segment into a reusable node-type class. "
                    "Base similarity on intent, state transition, and evidence pattern, then "
                    "derive reusable edge relationships.\n\n"
                    f"{json.dumps({'segments': segment_payloads}, ensure_ascii=False)}"
                ),
            },
        ]
        submitted_spec: GraphSpec | None = None
        submitted_clusters: list[SegmentCluster] = []
        steps = 0
        for steps in range(1, self.max_steps + 1):
            response = self._chat(
                messages,
                CLUSTERING_TOOLS,
                f"{self.session_id}-cluster",
            )
            message = assistant_message(response)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                messages.append(message)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Call submit_clustering with a complete partition of all segment IDs."
                        ),
                    }
                )
                continue

            messages.append(message)
            for tool_call in tool_calls:
                name, arguments, parse_error = parse_tool_arguments(tool_call)
                if parse_error:
                    content = parse_error
                elif name == "list_segments":
                    content = json.dumps(
                        {"segments": segment_payloads},
                        ensure_ascii=False,
                    )
                elif name == "compare_segments":
                    requested = [
                        str(value) for value in arguments.get("segment_ids") or []
                    ]
                    unknown = [value for value in requested if value not in segment_map]
                    if len(requested) < 2:
                        content = json.dumps(
                            {
                                "ok": False,
                                "errors": ["Compare at least two segment IDs."],
                            }
                        )
                    elif unknown:
                        content = json.dumps(
                            {
                                "ok": False,
                                "errors": [f"Unknown segment IDs: {unknown}."],
                            }
                        )
                    else:
                        comparisons = []
                        for segment_id in requested:
                            segment = segment_map[segment_id]
                            comparisons.append(
                                {
                                    "segment": to_dict(segment),
                                    "events": sandboxes[segment.trace_key].span(
                                        *segment.event_span
                                    ),
                                }
                            )
                        content = json.dumps(
                            {"segments": comparisons},
                            ensure_ascii=False,
                        )
                elif name == "submit_clustering":
                    errors, clusters, spec = validate_clustering(segments, arguments)
                    if errors:
                        content = json.dumps({"ok": False, "errors": errors})
                    else:
                        submitted_clusters = clusters
                        submitted_spec = spec
                        content = json.dumps(
                            {
                                "ok": True,
                                "node_types": submitted_spec.node_names(),
                                "edge_types": submitted_spec.edge_names(),
                            }
                        )
                else:
                    content = f"Unknown tool: {name}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": content,
                    }
                )
            if submitted_spec is not None:
                break

        if submitted_spec is None:
            raise RuntimeError(
                f"Clusterer did not submit a valid graph spec within {self.max_steps} steps."
            )
        return submitted_spec, submitted_clusters, {
            "agent": "segment_clusterer",
            "model": self.model,
            "session_id": f"{self.session_id}-cluster",
            "steps": steps,
            "max_steps": self.max_steps,
            "messages": messages,
            "clusters": [to_dict(cluster) for cluster in submitted_clusters],
        }

    def _chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        session_id: str,
    ) -> dict[str, Any]:
        return complete_chat(
            model=self.model,
            messages=messages,
            tools=tools,
            api_key=self.api_key,
            api_base=self.api_base,
            session_id=session_id,
            user_agent=self.user_agent,
            transport=self.transport,
            timeout=600,
        )
