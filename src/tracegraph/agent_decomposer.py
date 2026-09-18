from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .agent_segmenter import WorkSegment, segment_trace
from .agent_trace import (
    agent_runtime,
    assistant_message,
    complete_chat,
    make_agent_trace,
    parse_tool_arguments,
)
from .graph_builder import graph_from_draft, normalize_graph_draft, validate_graph_draft
from .graph_spec import DEFAULT_GRAPH_SPEC, render_graph_spec
from .sandbox import TraceSandbox
from .schema import AccomplishmentGraph, GraphSpec, Trace, to_dict


def decomposer_system_prompt(spec: GraphSpec) -> str:
    return f"""You assign type-blind work segments to an evidence-grounded accomplishment graph.

A previous agent already segmented this trajectory without using a node-type ontology. Those
segments are the only allowed units of work. Classify them against the graph spec, merge
consecutive same-type segments when they are one accomplishment, and skip segments that have
no matching type.

Process:
1. Read the provided segments. Use `list_segments` if you need the full list again.
2. Inspect the original events with `run_python` so claims and evidence excerpts quote real
   event text.
3. Call `submit_graph` with nodes and a causal edge DAG.

Assignment rules:
- Every node must cite one or more `segment_ids` from the segmented trace.
- Do not invent new event spans. `event_span` is filled mechanically from the union of each
  node's assigned segments.
- Merge only consecutive segments of the same type when they are one accomplishment.
- Leave unmatched segments unassigned. Assignment is not one node per segment.
- Never assign the same segment to two nodes.
- Never emit duplicate or overlapping nodes for the same accomplishment.
- Cover the important work described by the graph spec when matching segments are present.
- When an edit is involved, record the changed file paths in artifact_delta as {{"paths": [...]}}.
- Set confidence in [0, 1] to how strongly the cited events support the claim. Do not inflate it.
- Do not invent evidence, and never rely on gold patches or hidden tests.
- Claims must be specific sentences (not "test", "c", or other placeholders). Evidence
  excerpts must quote real event text, not "x".
- Use only the node types defined in the graph spec. Do not invent extra type names.

Graph structure rules:
- A graph with more than one node MUST include edges. Never submit disconnected nodes with
  `edges: []`.
- Build a causal DAG using the spec's edge types and typical order. Branch when the trace
  contains independent fixes or tests.
- Every non-root node must have at least one incoming dependency edge from a node that
  genuinely supports it. A final verification or handoff should depend on each implementation
  that it checks or packages.
- Use source -> target direction: the source accomplishment comes first and supports, supplies,
  or is corrected by the target.
- Independent roots and branches may remain disconnected from one another when there is no
  shared downstream accomplishment. Connect each branch internally, and connect branches only
  to nodes that genuinely consume results from those branches.
- Before submitting, check that every important node is connected to the causal structure and
  that the graph contains no cycles.

{render_graph_spec(spec)}

If submit_graph returns validation errors, fix them and submit again.

Each node needs: node_id, claim, type, segment_ids, evidence [{{event_id,kind,excerpt}}],
artifact_delta, confidence, remaining_work.

Tool argument formatting: `nodes` and `edges` must be actual JSON arrays. Never encode either
array as a quoted JSON string."""


def decomposer_tools(spec: GraphSpec) -> list[dict[str, Any]]:
    node_enum = spec.node_names()
    edge_enum = spec.edge_names()
    return [
        {
            "type": "function",
            "function": {
                "name": "list_segments",
                "description": "List the type-blind work segments that may be assigned to nodes.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_python",
                "description": "Run Python against the loaded trace sandbox. Available: trace.problem(), trace.summary(), trace.events(), trace.event(id), trace.turn(n), trace.grep(pattern), trace.span(start,end), trace.uncovered(spans). Print with print(...) or set `result = ...` to return a value.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_graph",
                "description": (
                    "Submit the final accomplishment graph as JSON nodes and a causal edge DAG. "
                    "Each node must cite segment_ids from the segmented trace. Consecutive "
                    "same-type segments may be merged into one node; unmatched segments may be "
                    "skipped. "
                    f"Node types must be one of: {', '.join(node_enum)}. "
                    f"Edge types must be one of: {', '.join(edge_enum)}. "
                    "If there is more than one node, edges must not be empty and every non-root "
                    "node must have an incoming dependency edge. Pass nodes and edges as actual "
                    "JSON arrays, never as JSON-encoded strings."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "node_id": {"type": "string"},
                                    "claim": {"type": "string"},
                                    "type": {"type": "string", "enum": node_enum},
                                    "segment_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "minItems": 1,
                                    },
                                    "event_span": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "minItems": 2,
                                        "maxItems": 2,
                                    },
                                    "evidence": {"type": "array", "items": {"type": "object"}},
                                    "artifact_delta": {"type": "object"},
                                    "confidence": {"type": "number"},
                                    "remaining_work": {"type": "string"},
                                },
                                "required": [
                                    "node_id",
                                    "claim",
                                    "type",
                                    "segment_ids",
                                    "evidence",
                                ],
                            },
                        },
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "target": {"type": "string"},
                                    "type": {"type": "string", "enum": edge_enum},
                                },
                                "required": ["source", "target", "type"],
                            },
                        },
                    },
                    "required": ["nodes", "edges"],
                },
            },
        },
    ]


SYSTEM_PROMPT = decomposer_system_prompt(DEFAULT_GRAPH_SPEC)
TOOLS = decomposer_tools(DEFAULT_GRAPH_SPEC)


def apply_segment_assignments(
    draft: dict[str, Any],
    segments: list[WorkSegment],
) -> tuple[dict[str, Any], list[str]]:
    """Fill node event_span from contiguous assigned segments; report assignment errors."""
    payload = normalize_graph_draft(draft)
    nodes = payload.get("nodes")
    errors: list[str] = []
    if not isinstance(nodes, list):
        return payload, [f"nodes must be a JSON array of objects, not {type(nodes).__name__}."]

    order = {segment.segment_id: index for index, segment in enumerate(segments)}
    by_id = {segment.segment_id: segment for segment in segments}
    assigned: dict[str, str] = {}
    for index, item in enumerate(nodes):
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id") or f"n{index:03d}")
        raw_ids = item.get("segment_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            errors.append(f"Node {node_id} must cite segment_ids from the segmented trace.")
            continue
        ids = [str(value) for value in raw_ids]
        unknown = [value for value in ids if value not in by_id]
        if unknown:
            errors.append(f"Node {node_id} cites unknown segment IDs: {unknown}.")
            continue
        for segment_id in ids:
            previous = assigned.get(segment_id)
            if previous is not None:
                errors.append(
                    f"Segment {segment_id} is assigned to both {previous} and {node_id}."
                )
            assigned[segment_id] = node_id
        indices = [order[segment_id] for segment_id in ids]
        start, end = min(indices), max(indices)
        if sorted(indices) != list(range(start, end + 1)):
            errors.append(
                f"Node {node_id} segment_ids must be consecutive in segment order; "
                "merge only adjacent same-type work, or emit separate nodes."
            )
            continue
        ordered = segments[start : end + 1]
        item["segment_ids"] = [segment.segment_id for segment in ordered]
        item["event_span"] = [ordered[0].event_span[0], ordered[-1].event_span[1]]
    return payload, errors


class AgentDecomposer:
    """Segment a trace, then assign those segments to a validated graph."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        max_steps: int = 16,
        segmentation_max_steps: int = 8,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        session_id: str | None = None,
        user_agent: str | None = None,
        graph_spec: GraphSpec | None = None,
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
        self.graph_spec = graph_spec or DEFAULT_GRAPH_SPEC
        self._system_prompt = decomposer_system_prompt(self.graph_spec)
        self._tools = decomposer_tools(self.graph_spec)

    def decompose(self, trace: Trace) -> AccomplishmentGraph:
        sandbox = TraceSandbox(trace)
        segments, segmentation_trace = segment_trace(
            trace,
            sandbox=sandbox,
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            session_id=f"{self.session_id}-segment",
            user_agent=self.user_agent,
            max_steps=self.segmentation_max_steps,
            transport=self.transport,
        )
        if not segments:
            raise RuntimeError("Segmenter found no task-relevant work to classify.")

        submitted, steps, messages = self._assign_segments(trace, sandbox, segments)
        graph = graph_from_draft(
            trace,
            submitted,
            decomposer=f"agent:{self.model}",
            spec=self.graph_spec,
            agent_steps=steps,
            segmentation_steps=segmentation_trace["steps"],
        )
        traces = dict(graph.metadata.get("agent_traces") or {})
        traces["segmenter"] = segmentation_trace
        traces["decomposer"] = make_agent_trace(
            agent="decomposer",
            model=self.model,
            steps=steps,
            max_steps=self.max_steps,
            session_id=self.session_id,
            messages=messages,
        )
        graph.metadata["agent_traces"] = traces
        graph.metadata["segments"] = [to_dict(segment) for segment in segments]
        return graph

    def _assign_segments(
        self,
        trace: Trace,
        sandbox: TraceSandbox,
        segments: list[WorkSegment],
    ) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
        segment_payloads = [to_dict(segment) for segment in segments]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": (
                    "Assign these type-blind work segments to an accomplishment graph using the "
                    "provided graph spec. Merge consecutive same-type segments when they are one "
                    "accomplishment, and skip unmatched segments. Use run_python to quote real "
                    "event text before you submit.\n\n"
                    f"{json.dumps({'summary': sandbox.summary(), 'segments': segment_payloads}, ensure_ascii=False)}"
                ),
            },
        ]
        submitted: dict[str, Any] | None = None
        steps = 0
        for steps in range(1, self.max_steps + 1):
            response = self._chat(messages)
            message = assistant_message(response)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                messages.append(message)
                if submitted is not None:
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": "Call submit_graph with your final nodes and edges.",
                    }
                )
                continue

            messages.append(message)
            for tool_call in tool_calls:
                name, arguments, parse_error = parse_tool_arguments(tool_call)
                if parse_error:
                    content = parse_error
                elif name == "list_segments":
                    content = json.dumps({"segments": segment_payloads}, ensure_ascii=False)
                elif name == "run_python":
                    content = sandbox.run_python(str(arguments.get("code", "")))
                elif name == "submit_graph":
                    draft, assignment_errors = apply_segment_assignments(
                        {
                            "nodes": arguments.get("nodes") or [],
                            "edges": arguments.get("edges") or [],
                        },
                        segments,
                    )
                    errors = assignment_errors + validate_graph_draft(
                        trace, draft, self.graph_spec
                    )
                    if errors:
                        content = json.dumps({"ok": False, "errors": errors})
                    else:
                        submitted = draft
                        content = json.dumps({"ok": True, "node_count": len(draft["nodes"])})
                else:
                    content = f"Unknown tool: {name}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": content,
                    }
                )
            if submitted is not None:
                break

        if submitted is None:
            raise RuntimeError(
                f"Agent did not submit a valid graph within {self.max_steps} steps."
            )
        return submitted, steps, messages

    def _chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return complete_chat(
            model=self.model,
            messages=messages,
            tools=self._tools,
            api_key=self.api_key,
            api_base=self.api_base,
            session_id=self.session_id,
            user_agent=self.user_agent,
            transport=self.transport,
        )
