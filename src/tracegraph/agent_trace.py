from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
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


def complete_chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    api_key: str,
    api_base: str,
    session_id: str,
    user_agent: str,
    transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    timeout: int = 180,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0,
        "messages": messages,
        "tools": tools,
    }
    if transport is not None:
        return transport(payload)
    if not api_key:
        raise RuntimeError("Set TRACEGRAPH_API_KEY before running a tracegraph agent.")
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": user_agent,
            "x-opencode-session": session_id,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"LLM request failed (HTTP {exc.code}): {body}") from exc


def agent_runtime(
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    session_id: str | None = None,
    user_agent: str | None = None,
) -> dict[str, str]:
    return {
        "model": model or os.environ.get("TRACEGRAPH_MODEL", "glm-3.5-flash"),
        "api_key": api_key or os.environ.get("TRACEGRAPH_API_KEY", ""),
        "api_base": (
            api_base
            or os.environ.get("TRACEGRAPH_API_BASE")
            or "https://opencode.ai/zen/go/v1"
        ).rstrip("/"),
        "session_id": session_id or os.environ.get("TRACEGRAPH_SESSION") or uuid.uuid4().hex,
        "user_agent": user_agent or os.environ.get("TRACEGRAPH_USER_AGENT", "tracegraph-agent/0.1"),
    }


def parse_tool_arguments(
    tool_call: dict[str, Any],
) -> tuple[str, dict[str, Any], str | None]:
    """Return (name, arguments, error_content). error_content is set when JSON is invalid."""
    name = str(tool_call.get("function", {}).get("name") or "")
    try:
        arguments = json.loads(tool_call.get("function", {}).get("arguments") or "{}")
    except json.JSONDecodeError as exc:
        return (
            name,
            {},
            json.dumps(
                {
                    "ok": False,
                    "errors": [
                        f"arguments were not valid JSON ({exc}); resend a valid JSON object"
                    ],
                }
            ),
        )
    if not isinstance(arguments, dict):
        return (
            name,
            {},
            json.dumps(
                {
                    "ok": False,
                    "errors": ["arguments must be a JSON object"],
                }
            ),
        )
    return name, arguments, None
