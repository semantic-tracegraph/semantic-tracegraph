from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .agent_trace import agent_runtime, assistant_message, complete_chat, make_agent_trace
from .graph_spec import DEFAULT_GRAPH_SPEC, spec_from_graph
from .sandbox import TraceSandbox
from .schema import AccomplishmentGraph, AccomplishmentNode, GraphSpec, NodeStatus, Trace, to_dict


@dataclass(slots=True)
class VerificationResult:
    status: NodeStatus
    confidence: float
    reason: str

SYSTEM_PROMPT = """You verify one accomplishment node against a coding-agent trajectory.

The trajectory is loaded in a Python sandbox as `trace`, and its summary is shown below. The
node cites event IDs and excerpts, but those excerpts were written by another agent and may be
incomplete or misleading, so always confirm them against the real events.

Steps:
1. Use `inspect_graph` to understand this node's parents, children, and role in the
   decomposed graph. Follow incoming dependencies when the claim relies on earlier work.
2. Use `run_python` to read the cited events (trace.event(id)), the node span
   (trace.span(start, end)), and enough surrounding context to judge the claim.
3. Check whether later events undo, revert, or contradict the accomplishment (for example a
   git checkout/reset/restore, an rm of the changed file, or a subsequent failing test).
4. Call `submit_verdict`.

Graph context is supporting context, not evidence by itself. Confirm parent and child claims
against real trace events. If this node semantically depends on a contradicted or unsupported
parent, account for that in the verdict; do not mark it verified merely because a downstream
action ran. Use the provided node-type definition when judging whether the claim matches the
intended accomplishment kind.

Status guide:
- `claimed`: cited evidence is missing, weak, misread, or the work was later undone.
- `evidenced`: the trace supports the semantic accomplishment (e.g. a search located the code,
  an edit succeeded) but there is no mechanical proof it is correct.
- `verified`: mechanical proof is present (tests passed, a patch/diff was emitted, or a failing
  reproduction was observed).

Set confidence in [0, 1] for how strongly the evidence supports the chosen status, and give a
concise reason citing specific event IDs. If submit_verdict returns validation errors, fix them
and submit again."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_graph",
            "description": (
                "Inspect the decomposed accomplishment graph. Omit node_id for all nodes, "
                "edges, and roots; provide node_id for that node's incoming parents and "
                "outgoing children. Graph claims are context and must still be checked "
                "against trajectory events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Optional node whose graph neighborhood should be returned.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run Python against the loaded trace sandbox. Available: trace.problem(), trace.summary(), trace.events(), trace.event(id), trace.turn(n), trace.grep(pattern), trace.span(start,end). Print with print(...) or set `result = ...` to return a value.",
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
            "name": "submit_verdict",
            "description": "Submit the final semantic judgment for the node under review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["claimed", "evidenced", "verified"],
                        "description": "Whether cited evidence supports the claim.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence in [0, 1].",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short justification citing event IDs.",
                    },
                },
                "required": ["status", "confidence", "reason"],
            },
        },
    },
]


VALID_VERDICT_STATUSES = {
    NodeStatus.CLAIMED.value,
    NodeStatus.EVIDENCED.value,
    NodeStatus.VERIFIED.value,
}


def validate_verdict(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    status = payload.get("status")
    if status not in VALID_VERDICT_STATUSES:
        errors.append("status must be claimed, evidenced, or verified")
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)):
        errors.append("confidence must be a number")
    elif not 0.0 <= float(confidence) <= 1.0:
        errors.append("confidence must be in [0, 1]")
    reason = payload.get("reason")
    if not reason or not str(reason).strip():
        errors.append("reason is required")
    return errors


def verdict_from_payload(payload: dict[str, Any]) -> VerificationResult:
    status = NodeStatus(payload["status"])
    return VerificationResult(
        status=status,
        confidence=min(1.0, max(0.0, float(payload["confidence"]))),
        reason=str(payload["reason"]),
    )


def inspect_graph(
    graph: AccomplishmentGraph,
    node_id: str | None = None,
    spec: GraphSpec | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe structural view of a decomposed graph."""
    spec = spec or spec_from_graph(graph)

    def node_payload(node: AccomplishmentNode) -> dict[str, Any]:
        return {
            "node_id": node.node_id,
            "claim": node.claim,
            "type": str(node.type),
            "event_span": list(node.event_span),
            "evidence": [
                {"event_id": item.event_id, "kind": item.kind, "excerpt": item.excerpt}
                for item in node.evidence
            ],
            "artifact_delta": {
                key: value
                for key, value in node.artifact_delta.items()
                if key != "verification_reason"
            },
        }

    nodes = {node.node_id: node for node in graph.nodes}
    edges = [
        {"source": edge.source, "target": edge.target, "type": str(edge.type)}
        for edge in graph.edges
    ]
    dependency_types = spec.dependency_edge_names()
    dependency_targets = {
        edge.target for edge in graph.edges if str(edge.type) in dependency_types
    }
    roots = [node.node_id for node in graph.nodes if node.node_id not in dependency_targets]

    if node_id is None:
        return {
            "nodes": [node_payload(node) for node in graph.nodes],
            "edges": edges,
            "root_node_ids": roots,
        }
    if node_id not in nodes:
        return {
            "error": f"Unknown node_id {node_id}",
            "available_node_ids": list(nodes),
        }

    incoming = [edge for edge in edges if edge["target"] == node_id]
    outgoing = [edge for edge in edges if edge["source"] == node_id]
    return {
        "node": node_payload(nodes[node_id]),
        "incoming_edges": incoming,
        "parents": [node_payload(nodes[edge["source"]]) for edge in incoming],
        "outgoing_edges": outgoing,
        "children": [node_payload(nodes[edge["target"]]) for edge in outgoing],
        "is_root": node_id in roots,
    }


class AgentJudge:
    """Multi-turn LLM agent that inspects trace evidence and submits a verdict."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        max_steps: int = 12,
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
        self.transport = transport
        self.session_id = runtime["session_id"]
        self.user_agent = runtime["user_agent"]
        self.graph_spec = graph_spec
        self._sandboxes: dict[tuple[str, str], TraceSandbox] = {}

    def _resolve_spec(self, graph: AccomplishmentGraph | None) -> GraphSpec:
        if self.graph_spec is not None:
            return self.graph_spec
        if graph is not None:
            return spec_from_graph(graph)
        return DEFAULT_GRAPH_SPEC

    def __call__(
        self,
        node: AccomplishmentNode,
        trace: Trace,
        graph: AccomplishmentGraph | None = None,
    ) -> tuple[VerificationResult, dict[str, Any]]:
        # Key on (instance_id, model): multiple models share instance_ids in this dataset,
        # so caching on instance_id alone could return the wrong trajectory's sandbox.
        cache_key = (trace.instance_id, trace.model)
        sandbox = self._sandboxes.get(cache_key)
        if sandbox is None:
            sandbox = TraceSandbox(trace)
            self._sandboxes[cache_key] = sandbox
        spec = self._resolve_spec(graph)
        node_type = str(node.type)
        type_spec = spec.node_spec(node_type)
        node_payload = {
            "node_id": node.node_id,
            "claim": node.claim,
            "type": node_type,
            "event_span": list(node.event_span),
            "evidence": [
                {"event_id": item.event_id, "kind": item.kind, "excerpt": item.excerpt}
                for item in node.evidence
            ],
            "artifact_delta": node.artifact_delta,
            "type_definition": to_dict(type_spec) if type_spec is not None else {
                "name": node_type,
                "description": "Unknown type in the current graph spec.",
            },
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Verify this accomplishment node against the trajectory. Confirm the cited "
                    "evidence in the real events, and check for later reverts, before you decide."
                    "\n\n"
                    f"{json.dumps({'trace': sandbox.summary(), 'node': node_payload}, ensure_ascii=False)}"
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
                        "content": "Call submit_verdict with your final status, confidence, and reason.",
                    }
                )
                continue

            messages.append(message)
            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                try:
                    arguments = json.loads(tool_call["function"].get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(
                                {
                                    "ok": False,
                                    "errors": [f"arguments were not valid JSON ({exc}); resend a valid JSON object"],
                                }
                            ),
                        }
                    )
                    continue
                if name == "run_python":
                    content = sandbox.run_python(str(arguments.get("code", "")))
                elif name == "inspect_graph":
                    if graph is None:
                        content = json.dumps(
                            {"error": "Graph context is unavailable for this direct judge call."}
                        )
                    else:
                        content = json.dumps(
                            inspect_graph(
                                graph,
                                arguments.get("node_id"),
                                spec=self._resolve_spec(graph),
                            ),
                            ensure_ascii=False,
                        )
                elif name == "submit_verdict":
                    errors = validate_verdict(arguments)
                    if errors:
                        content = json.dumps({"ok": False, "errors": errors})
                    else:
                        submitted = arguments
                        content = json.dumps({"ok": True})
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
                f"Agent judge did not submit a valid verdict for {node.node_id} "
                f"within {self.max_steps} steps."
            )
        return verdict_from_payload(submitted), make_agent_trace(
            agent="judge",
            model=self.model,
            steps=steps,
            max_steps=self.max_steps,
            session_id=self.session_id,
            messages=messages,
            node_id=node.node_id,
        )

    def verify(self, graph: AccomplishmentGraph, trace: Trace) -> AccomplishmentGraph:
        event_map = {event.event_id: event for event in trace.events}
        node_traces: dict[str, Any] = {}
        total_steps = 0
        for node in graph.nodes:
            result, node_trace = self(node, trace, graph)
            node.status = result.status
            node.confidence = result.confidence
            node.artifact_delta["verification_reason"] = result.reason
            for evidence in node.evidence:
                evidence.mechanically_checked = evidence.event_id in event_map
            node_traces[node.node_id] = node_trace
            total_steps += int(node_trace["steps"])

        traces = dict(graph.metadata.get("agent_traces") or {})
        traces["judge"] = {
            "agent": "judge",
            "model": self.model,
            "session_id": self.session_id,
            "total_steps": total_steps,
            "node_count": len(graph.nodes),
            "nodes": node_traces,
        }
        graph.metadata["agent_traces"] = traces
        graph.metadata["verifier"] = f"agent-judge:{self.model}"
        graph.metadata["judge_steps"] = total_steps
        graph.metadata["verified_nodes"] = sum(n.status == NodeStatus.VERIFIED for n in graph.nodes)
        graph.metadata["invalidated_nodes"] = sum(
            n.status == NodeStatus.INVALIDATED for n in graph.nodes
        )
        return graph

    def _chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return complete_chat(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            api_key=self.api_key,
            api_base=self.api_base,
            session_id=self.session_id,
            user_agent=self.user_agent,
            transport=self.transport,
        )
