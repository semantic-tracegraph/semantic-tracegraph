from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .agent_trace import assistant_message, complete_chat, parse_tool_arguments
from .graph_builder import coerce_json_array
from .sandbox import TraceSandbox
from .schema import Trace, to_dict


@dataclass(slots=True)
class WorkSegment:
    """One coherent, task-relevant unit of work within a trajectory."""

    segment_id: str
    trace_key: str
    event_span: tuple[str, str]
    objective: str
    action_summary: str
    outcome_summary: str
    evidence_event_ids: list[str] = field(default_factory=list)
    artifact_paths: list[str] = field(default_factory=list)


SEGMENTATION_SYSTEM_PROMPT = """You segment one coding-agent trajectory into coherent units of work.

Do not invent or assign node types. Your only job is to identify bottom-up sub-task
boundaries in this trajectory. A later stage will classify or cluster the segments.

Process:
1. Inspect the problem and trajectory with `run_python`.
2. Divide the task-relevant work into individual segments.
3. Call `submit_segments`.

A segment is one coherent sub-task with a shared objective and state transition. Some
examples of boundary signals include:
- the agent changes from understanding/searching to editing;
- the target file, symbol, or sub-problem changes;
- an implementation ends and a test or other check begins;

Segmentation rules:
- Use real event IDs for event_span [start, end] and evidence_event_ids.
- Segments must be ordered and non-overlapping. Gaps for irrelevant chatter are allowed.
- Include all task-relevant edits, tests, and submissions.
- Merge consecutive commands that serve the same immediate sub-task.
- Split independent fixes even when they use the same command.
- Objective describes the local goal; action_summary describes what the agent did;
  outcome_summary describes the resulting state or observed result.
- Do not label segments with ontology classes such as localization or implementation.
- Do not infer hidden-test or gold-patch results.

If submit_segments returns errors, correct the boundaries or fields and submit again."""


SEGMENTATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python against the loaded trace sandbox as `trace`. Available: "
                "trace.problem(), trace.summary(), trace.events(), trace.event(id), "
                "trace.turn(n), trace.grep(pattern), trace.span(start,end), "
                "trace.uncovered(spans). Print with print(...) or set `result = ...`."
            ),
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
            "name": "submit_segments",
            "description": (
                "Submit the ordered, non-overlapping units of work for this trajectory. "
                "Return an empty array only when the trajectory contains no task-relevant work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "segment_id": {"type": "string"},
                                "event_span": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                                "objective": {"type": "string"},
                                "action_summary": {"type": "string"},
                                "outcome_summary": {"type": "string"},
                                "evidence_event_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "artifact_paths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "segment_id",
                                "event_span",
                                "objective",
                                "action_summary",
                                "outcome_summary",
                                "evidence_event_ids",
                            ],
                        },
                    },
                    "coverage_note": {"type": "string"},
                },
                "required": ["segments"],
            },
        },
    },
]


def trace_key(trace: Trace) -> str:
    return f"{trace.instance_id}::{trace.model}"


def parse_segment(raw: dict[str, Any], key: str, canonical_id: str) -> WorkSegment:
    span = raw.get("event_span") or []
    return WorkSegment(
        segment_id=canonical_id,
        trace_key=key,
        event_span=(str(span[0]), str(span[1])),
        objective=str(raw.get("objective") or "").strip(),
        action_summary=str(raw.get("action_summary") or "").strip(),
        outcome_summary=str(raw.get("outcome_summary") or "").strip(),
        evidence_event_ids=[str(event_id) for event_id in raw.get("evidence_event_ids") or []],
        artifact_paths=[str(path) for path in raw.get("artifact_paths") or []],
    )


def validate_segments(trace: Trace, raw_segments: Any) -> list[str]:
    """Validate per-trace segment boundaries before classification or clustering."""
    if not isinstance(raw_segments, list):
        return [f"segments must be a JSON array, not {type(raw_segments).__name__}."]

    event_order = {event.event_id: index for index, event in enumerate(trace.events)}
    seen_local_ids: set[str] = set()
    spans: list[tuple[int, int, str]] = []
    covered_indices: set[int] = set()
    errors: list[str] = []

    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            errors.append(f"Segment {index} must be an object.")
            continue
        segment_id = str(raw.get("segment_id") or "").strip()
        if not segment_id:
            errors.append(f"Segment {index} needs a segment_id.")
        elif segment_id in seen_local_ids:
            errors.append(f"Duplicate segment_id: {segment_id}")
        seen_local_ids.add(segment_id)

        span = raw.get("event_span") or []
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            errors.append(f"Segment {segment_id or index} needs event_span [start, end].")
            continue
        start_id, end_id = span
        if start_id not in event_order or end_id not in event_order:
            errors.append(f"Segment {segment_id or index} cites unknown event IDs in event_span.")
            continue
        start, end = event_order[start_id], event_order[end_id]
        if start > end:
            errors.append(f"Segment {segment_id or index} has a reversed event_span.")
            continue
        spans.append((start, end, segment_id))
        covered_indices.update(range(start, end + 1))

        for field_name in ("objective", "action_summary", "outcome_summary"):
            if not str(raw.get(field_name) or "").strip():
                errors.append(f"Segment {segment_id or index} needs {field_name}.")

        evidence = raw.get("evidence_event_ids")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"Segment {segment_id or index} needs evidence_event_ids.")
        else:
            for event_id in evidence:
                if event_id not in event_order:
                    errors.append(
                        f"Segment {segment_id or index} cites unknown evidence event {event_id}."
                    )
                elif not start <= event_order[event_id] <= end:
                    errors.append(
                        f"Segment {segment_id or index} evidence {event_id} is outside its span."
                    )

    spans.sort()
    for previous, current in zip(spans, spans[1:]):
        if current[0] <= previous[1]:
            errors.append(
                f"Segments {previous[2]} and {current[2]} overlap; segment spans must be disjoint."
            )

    required_types = {"edit", "test", "submission"}
    uncovered_required = [
        event.event_id
        for index, event in enumerate(trace.events)
        if event.type.value in required_types and index not in covered_indices
    ]
    if uncovered_required:
        errors.append(
            "Segments leave task-relevant edit/test/submission events uncovered: "
            + ", ".join(uncovered_required)
        )
    return errors


def segment_trace(
    trace: Trace,
    *,
    model: str,
    api_key: str,
    api_base: str,
    session_id: str,
    user_agent: str,
    max_steps: int = 8,
    transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    sandbox: TraceSandbox | None = None,
    id_prefix: str = "s",
) -> tuple[list[WorkSegment], dict[str, Any]]:
    """Run the type-blind segmentation loop for one trajectory."""
    sandbox = sandbox or TraceSandbox(trace)
    key = trace_key(trace)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SEGMENTATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Segment this trajectory into coherent task-relevant units of work. "
                "Investigate the actual events before submitting boundaries.\n\n"
                f"{json.dumps(sandbox.summary(), ensure_ascii=False)}"
            ),
        },
    ]
    submitted: list[dict[str, Any]] | None = None
    coverage_note = ""
    steps = 0
    for steps in range(1, max_steps + 1):
        response = complete_chat(
            model=model,
            messages=messages,
            tools=SEGMENTATION_TOOLS,
            api_key=api_key,
            api_base=api_base,
            session_id=session_id,
            user_agent=user_agent,
            transport=transport,
        )
        message = assistant_message(response)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            messages.append(message)
            messages.append(
                {
                    "role": "user",
                    "content": "Call submit_segments with the final segment boundaries.",
                }
            )
            continue

        messages.append(message)
        for tool_call in tool_calls:
            name, arguments, parse_error = parse_tool_arguments(tool_call)
            if parse_error:
                content = parse_error
            elif name == "run_python":
                content = sandbox.run_python(str(arguments.get("code", "")))
            elif name == "submit_segments":
                raw_segments = coerce_json_array(arguments.get("segments"))
                errors = validate_segments(trace, raw_segments)
                if errors:
                    content = json.dumps({"ok": False, "errors": errors})
                else:
                    submitted = raw_segments
                    coverage_note = str(arguments.get("coverage_note") or "")
                    content = json.dumps({"ok": True, "segment_count": len(raw_segments)})
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
            f"Segmenter did not submit valid segments for {key} within {max_steps} steps."
        )
    segments = [
        parse_segment(raw, key, f"{id_prefix}{index:03d}")
        for index, raw in enumerate(submitted)
    ]
    return segments, {
        "agent": "segmenter",
        "model": model,
        "session_id": session_id,
        "steps": steps,
        "max_steps": max_steps,
        "trace_key": key,
        "coverage_note": coverage_note,
        "messages": messages,
        "segments": [to_dict(segment) for segment in segments],
    }
