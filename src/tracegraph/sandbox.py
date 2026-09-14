from __future__ import annotations

import io
import json
import re
from contextlib import redirect_stdout
from typing import Any

from .schema import Trace, TraceEvent


class TraceSandbox:
    """Read-only trajectory view exposed to the decomposer agent."""

    def __init__(self, trace: Trace, content_limit: int = 4000) -> None:
        self._trace = trace
        self._content_limit = content_limit
        self._by_id = {event.event_id: event for event in trace.events}
        self._order = [event.event_id for event in trace.events]

    @property
    def instance_id(self) -> str:
        return self._trace.instance_id

    @property
    def model(self) -> str:
        return self._trace.model

    def problem(self) -> str:
        """Return the full problem statement (summary only shows a preview)."""
        return self._trace.problem_statement

    def summary(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        turns = set()
        for event in self._trace.events:
            by_type[event.type.value] = by_type.get(event.type.value, 0) + 1
            turns.add(event.turn)
        return {
            "instance_id": self._trace.instance_id,
            "model": self._trace.model,
            "resolved": self._trace.resolved,
            "event_count": len(self._trace.events),
            "turn_count": len(turns),
            "event_ids": self._order,
            "events_by_type": by_type,
            "problem_statement_preview": self._trace.problem_statement[:800],
        }

    def events(
        self,
        start_id: str | None = None,
        end_id: str | None = None,
        types: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        selected = self._trace.events
        if start_id or end_id:
            start = self._order.index(start_id) if start_id else 0
            end = self._order.index(end_id) if end_id else len(self._order) - 1
            selected = self._trace.events[min(start, end) : max(start, end) + 1]
        if types:
            allowed = {value.lower() for value in types}
            selected = [event for event in selected if event.type.value in allowed]
        return [self._event_dict(event) for event in selected[:limit]]

    def event(self, event_id: str) -> dict[str, Any]:
        event = self._by_id.get(event_id)
        if event is None:
            raise KeyError(f"Unknown event_id: {event_id}")
        return self._event_dict(event, full=True)

    def turn(self, turn_number: int) -> list[dict[str, Any]]:
        return [
            self._event_dict(event)
            for event in self._trace.events
            if event.turn == turn_number
        ]

    def grep(self, pattern: str, limit: int = 20) -> list[dict[str, Any]]:
        regex = re.compile(pattern, re.IGNORECASE)
        hits: list[dict[str, Any]] = []
        for event in self._trace.events:
            if regex.search(event.content):
                hits.append(
                    {
                        "event_id": event.event_id,
                        "turn": event.turn,
                        "type": event.type.value,
                        "snippet": event.content[:300],
                    }
                )
            if len(hits) >= limit:
                break
        return hits

    def span(self, start_id: str, end_id: str) -> list[dict[str, Any]]:
        return self.events(start_id=start_id, end_id=end_id)

    def uncovered(self, covered_spans: list[tuple[str, str]]) -> list[str]:
        covered: set[str] = set()
        for start_id, end_id in covered_spans:
            if start_id not in self._order or end_id not in self._order:
                continue
            start = self._order.index(start_id)
            end = self._order.index(end_id)
            covered.update(self._order[min(start, end) : max(start, end) + 1])
        return [event_id for event_id in self._order if event_id not in covered]

    def run_python(self, code: str, max_output: int = 8000) -> str:
        namespace: dict[str, Any] = {
            "trace": self,
            "json": json,
            "re": re,
            "result": None,
        }
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                exec(compile(code, "<agent>", "exec"), namespace, namespace)
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
        parts = []
        stdout = buffer.getvalue().strip()
        if stdout:
            parts.append(stdout)
        if namespace.get("result") is not None:
            parts.append(repr(namespace["result"]))
        if not parts:
            return "OK (no stdout and no result variable set)"
        output = "\n".join(parts)
        if len(output) > max_output:
            return output[: max_output - 20] + "\n...[truncated]"
        return output

    def _event_dict(self, event: TraceEvent, full: bool = False) -> dict[str, Any]:
        limit = None if full else self._content_limit
        content = event.content if limit is None else event.content[:limit]
        return {
            "event_id": event.event_id,
            "turn": event.turn,
            "type": event.type.value,
            "content": content,
            "content_truncated": limit is not None and len(event.content) > limit,
            "returncode": event.returncode,
            "paths": event.paths,
            "symbols": event.symbols,
        }
