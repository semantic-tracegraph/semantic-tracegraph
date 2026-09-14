from __future__ import annotations

import json
from typing import Any


def assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    """Normalize a chat-completions assistant message into a dict with optional tool_calls."""
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if isinstance(message, str):
        return {"role": "assistant", "content": message}
    if not isinstance(message, dict):
        return {"role": "assistant", "content": str(message or "")}
    normalized = dict(message)
    if "role" not in normalized:
        normalized["role"] = "assistant"
    tool_calls = normalized.get("tool_calls") or []
    if isinstance(tool_calls, str):
        try:
            tool_calls = json.loads(tool_calls)
        except json.JSONDecodeError:
            tool_calls = []
    cleaned: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, str):
            continue
        item = dict(tool_call)
        function = item.get("function")
        if isinstance(function, str):
            item["function"] = {"name": function, "arguments": item.get("arguments") or "{}"}
        elif isinstance(function, dict):
            item["function"] = dict(function)
        cleaned.append(item)
    if cleaned:
        normalized["tool_calls"] = cleaned
    else:
        normalized.pop("tool_calls", None)
    return normalized


def snapshot_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy a chat history into JSON-safe dicts for later analysis."""
    return json.loads(json.dumps(messages, ensure_ascii=False, default=str))


def make_agent_trace(
    *,
    agent: str,
    model: str,
    steps: int,
    max_steps: int,
    session_id: str,
    messages: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "agent": agent,
        "model": model,
        "steps": steps,
        "max_steps": max_steps,
        "session_id": session_id,
        "messages": snapshot_messages(messages),
        **extra,
    }
