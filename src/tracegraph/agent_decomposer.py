from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from typing import Any

from .agent_trace import assistant_message, make_agent_trace
from .graph_builder import graph_from_draft, normalize_graph_draft, validate_graph_draft
from .sandbox import TraceSandbox
from .schema import AccomplishmentGraph, Trace

SYSTEM_PROMPT = """You decompose coding-agent trajectories into evidence-grounded accomplishment graphs.

The trajectory is loaded in a Python sandbox as `trace`, and its summary is shown below.
Work in two phases:
1. Investigate with `run_python`: read the goal with trace.problem(), inspect key events,
   commands, and outputs, and check coverage with trace.uncovered(spans) so you do not miss
   major work.
2. Call `submit_graph` with a small set of coarse, non-overlapping accomplishments.

Node rules:
- Emit a few meaningful accomplishments, not one node per command. Merge steps that serve the
  same goal into a single node with a wider event_span.
- Never emit duplicate or overlapping nodes for the same accomplishment.
- Every node must cite real event IDs, both in event_span and in each evidence item.
- Cover the important work: localization, the key edits, reproductions/tests, and the final
  patch or handoff when present.
- When an edit is involved, record the changed file paths in artifact_delta as {"paths": [...]}.
- Set confidence in [0, 1] to how strongly the cited events support the claim. Do not inflate it.
- Do not invent evidence, and never rely on gold patches or hidden tests.
- Claims must be specific sentences (not "test", "c", or other placeholders). Evidence
  excerpts must quote real event text, not "x".

Node types: localization, diagnosis, reproduction, implementation, verification, handoff_artifact.

Graph structure rules:
- A graph with more than one node MUST include edges. Never submit disconnected nodes with
  `edges: []`.
- Build a causal DAG, usually following localization -> diagnosis/reproduction -> implementation
  -> verification -> handoff. Branch when the trace contains independent fixes or tests.
- A graph may have multiple root nodes when the trajectory contains genuinely independent
  starting accomplishments. Do not invent dependencies merely to force a single root.
- Every non-root node must have at least one incoming `requires`, `produces`, or `refines` edge
  from a node that genuinely supports it. A final verification or handoff should depend on each
  implementation that it checks or packages.
- Use source -> target direction: the source accomplishment comes first and supports, supplies,
  or is corrected by the target.
- Do not add an edge merely because two events happened in sequence. Independent roots and
  branches may remain disconnected from one another when there is no shared downstream
  accomplishment. Connect each branch internally, and connect branches only to verification or
  handoff nodes that genuinely consume results from those branches.
- Before submitting, check that every important node is connected to the causal structure and
  that the graph contains no cycles.

Edge meanings:
- requires: the target could not start until the source was done.
- produces: the source yields an artifact the target consumes.
- refines: the target improves or corrects the source.
- contradicts / invalidates: the target undoes or disproves the source.

If submit_graph returns validation errors, fix them and submit again.

Each node needs: node_id, claim, type, event_span [start,end], evidence [{event_id,kind,excerpt}],
artifact_delta, confidence, remaining_work.

Tool argument formatting: `nodes` and `edges` must be actual JSON arrays. Never encode either
array as a quoted JSON string."""

TOOLS = [
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
                "If there is more than one node, edges must not be empty and every non-root "
                "node must have an incoming dependency edge. Pass nodes and edges as actual "
                "JSON arrays, never as JSON-encoded strings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {"type": "array", "items": {"type": "object"}},
                    "edges": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["nodes", "edges"],
            },
        },
    },
]


class AgentDecomposer:
    """Multi-turn LLM agent that explores a trace and submits a validated graph."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        max_steps: int = 16,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        session_id: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("TRACEGRAPH_MODEL", "gpt-4.1-mini")
        self.api_key = api_key or os.environ.get("TRACEGRAPH_API_KEY", "")
        self.api_base = (
            api_base or os.environ.get("TRACEGRAPH_API_BASE") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.max_steps = max_steps
        self.transport = transport
        # OpenCode Go requires a stable per-conversation session id and a self-identifying
        # user agent; these headers are harmless for the OpenAI API.
        self.session_id = session_id or os.environ.get("TRACEGRAPH_SESSION") or uuid.uuid4().hex
        self.user_agent = user_agent or os.environ.get("TRACEGRAPH_USER_AGENT", "tracegraph-agent/0.1")

    def decompose(self, trace: Trace) -> AccomplishmentGraph:
        sandbox = TraceSandbox(trace)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Decompose this trajectory into an accomplishment graph. The summary is "
                    "below; use run_python (trace.problem(), key events, trace.uncovered(...)) "
                    "to investigate before you submit.\n\n"
                    f"{json.dumps(sandbox.summary(), ensure_ascii=False)}"
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
                elif name == "submit_graph":
                    draft = normalize_graph_draft(
                        {
                            "nodes": arguments.get("nodes") or [],
                            "edges": arguments.get("edges") or [],
                        }
                    )
                    errors = validate_graph_draft(trace, draft)
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
        graph = graph_from_draft(
            trace,
            submitted,
            decomposer=f"agent:{self.model}",
            agent_steps=steps,
        )
        traces = dict(graph.metadata.get("agent_traces") or {})
        traces["decomposer"] = make_agent_trace(
            agent="decomposer",
            model=self.model,
            steps=steps,
            max_steps=self.max_steps,
            session_id=self.session_id,
            messages=messages,
        )
        graph.metadata["agent_traces"] = traces
        return graph

    def _chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": messages,
            "tools": TOOLS,
        }
        if self.transport is not None:
            return self.transport(payload)
        if not self.api_key:
            raise RuntimeError("Set TRACEGRAPH_API_KEY before running the agent decomposer.")
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
                "x-opencode-session": self.session_id,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"LLM request failed (HTTP {exc.code}): {body}") from exc
