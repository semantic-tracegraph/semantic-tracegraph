from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from .data import _compact_record, write_jsonl
from .ingest import classify_command, extract_paths
from .schema import EventType, Trace, TraceEvent

USER_QUERY = re.compile(r"<user_query>\s*(?P<query>.*?)\s*</user_query>", re.DOTALL)
TIMESTAMP = re.compile(r"<timestamp>.*?</timestamp>", re.DOTALL)
DEFAULT_CURSOR_ROOT = Path.home() / ".cursor" / "projects"

SKIP_TOOLS = {
    "AskQuestion",
    "GetDynamicTools",
    "SwitchMode",
    "TodoWrite",
    "cursor_dialog",
    "open_resource",
    "open_automation",
    "rename_chat",
}

READ_TOOLS = {"Read", "ReadFile", "ReadLints"}
SEARCH_TOOLS = {"Grep", "Glob", "rg", "WebSearch", "WebFetch", "SemanticSearch"}
EDIT_TOOLS = {
    "ApplyPatch",
    "Delete",
    "StrReplace",
    "Write",
    "EditNotebook",
}


def looks_like_cursor_messages(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("tool_calls"):
            return True
        content = message.get("content")
        if isinstance(content, list):
            return True
        if isinstance(message.get("message"), dict):
            return True
    return False


def default_cursor_root() -> Path:
    override = os.environ.get("TRACEGRAPH_CURSOR_ROOT")
    return Path(override) if override else DEFAULT_CURSOR_ROOT


def discover_cursor_transcripts(
    root: str | Path | None = None,
    *,
    include_subagents: bool = False,
    include_empty_window: bool = False,
    projects: list[str] | None = None,
) -> list[Path]:
    """Find Cursor agent JSONL transcripts under `~/.cursor/projects`."""
    base = Path(root) if root else default_cursor_root()
    if not base.exists():
        return []
    found: list[Path] = []
    for path in sorted(base.glob("*/agent-transcripts/**/*.jsonl")):
        if path.name.endswith(".failures.jsonl"):
            continue
        if not include_subagents and "subagents" in path.parts:
            continue
        project = _project_name(path, base)
        if not include_empty_window and project == "empty-window":
            continue
        if projects and not any(token in project for token in projects):
            continue
        found.append(path)
    return found


def ingest_cursor_transcripts(
    root: str | Path | None = None,
    output: str | Path | None = None,
    *,
    include_subagents: bool = False,
    include_empty_window: bool = False,
    include_chat: bool = False,
    split_user_turns: bool = True,
    projects: list[str] | None = None,
    min_tools: int = 1,
    limit: int | None = None,
    max_message_chars: int = 8000,
) -> list[dict[str, Any]]:
    """Convert local Cursor transcripts into trajectory JSONL rows."""
    rows: list[dict[str, Any]] = []
    for path in discover_cursor_transcripts(
        root,
        include_subagents=include_subagents,
        include_empty_window=include_empty_window,
        projects=projects,
    ):
        for row in transcript_to_rows(
            path, root=root, split_user_turns=split_user_turns
        ):
            tool_count = int(row.get("tool_calls") or 0)
            if not include_chat and tool_count < min_tools:
                continue
            rows.append(_compact_record(row, max_message_chars=max_message_chars))
            if limit is not None and len(rows) >= limit:
                if output is not None:
                    write_jsonl(rows, output)
                return rows
    if output is not None:
        write_jsonl(rows, output)
    return rows


def transcript_to_row(path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    """Load one Cursor transcript as a single multi-turn trajectory."""
    return transcript_to_rows(path, root=root, split_user_turns=False)[0]


def transcript_to_rows(
    path: str | Path,
    root: str | Path | None = None,
    *,
    split_user_turns: bool = True,
) -> list[dict[str, Any]]:
    path = Path(path)
    base = Path(root) if root else _infer_projects_root(path)
    project = _project_name(path, base)
    conversation_id = path.stem
    records = list(_read_jsonl(path))
    messages = normalize_cursor_records(records)
    session = _row_from_messages(
        messages,
        instance_id=f"{project}/{conversation_id}",
        project=project,
        conversation_id=conversation_id,
        source_path=str(path),
    )
    if not split_user_turns:
        return [session]
    rows: list[dict[str, Any]] = []
    prior_queries: list[str] = []
    slices = split_messages_by_user_turn(messages)
    for index, slice_messages in enumerate(slices):
        query = _first_user_query(slice_messages)
        row = _row_from_messages(
            slice_messages,
            instance_id=f"{project}/{conversation_id}/u{index:03d}",
            project=project,
            conversation_id=conversation_id,
            source_path=str(path),
        )
        row["problem_statement"] = query
        row["user_turn_index"] = index
        row["session_user_turns"] = len(slices)
        row["prior_user_queries"] = list(prior_queries)
        rows.append(row)
        if query:
            prior_queries.append(query)
    return rows or [session]


def split_messages_by_user_turn(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Cut a conversation so each slice starts at a user request and ends before the next."""
    slices: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "user" and current:
            slices.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        slices.append(current)
    return [item for item in slices if any(message.get("role") == "user" for message in item)]


def _first_user_query(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return extract_user_query(str(message.get("content") or ""))
    return ""


def _row_from_messages(
    messages: list[dict[str, Any]],
    *,
    instance_id: str,
    project: str,
    conversation_id: str,
    source_path: str,
) -> dict[str, Any]:
    queries = [
        extract_user_query(str(message.get("content") or ""))
        for message in messages
        if message.get("role") == "user"
    ]
    queries = [query for query in queries if query]
    return {
        "instance_id": instance_id,
        "model": "cursor",
        "problem_statement": queries[0] if queries else "",
        "messages": messages,
        "resolved": None,
        "source": "cursor",
        "source_path": source_path,
        "project": project,
        "conversation_id": conversation_id,
        "user_turns": len(queries),
        "tool_calls": sum(len(message.get("tool_calls") or []) for message in messages),
    }


def normalize_cursor_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten Cursor JSONL bubbles into `{role, content, tool_calls?}` messages."""
    messages: list[dict[str, Any]] = []
    for record in records:
        role = record.get("role")
        if role not in {"user", "assistant"}:
            continue
        payload = record.get("message") if isinstance(record.get("message"), dict) else record
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        content = payload.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                kind = block.get("type")
                if kind == "text":
                    text_parts.append(str(block.get("text") or ""))
                elif kind == "tool_use":
                    tool_calls.append(
                        {
                            "name": str(block.get("name") or "unknown"),
                            "arguments": block.get("input") or block.get("arguments") or {},
                        }
                    )
        text = "\n".join(part for part in text_parts if part.strip()).strip()
        if role == "assistant" and text in {"[REDACTED]", ""}:
            text = ""
        message: dict[str, Any] = {"role": role, "content": text}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if text or tool_calls:
            messages.append(message)
    return messages


def extract_user_query(text: str) -> str:
    match = USER_QUERY.search(text)
    if match:
        return match.group("query").strip()
    cleaned = TIMESTAMP.sub("", text).strip()
    return cleaned


def parse_cursor_messages(row: dict[str, Any]) -> Trace:
    events: list[TraceEvent] = []
    turn = 0
    messages = row.get("messages") or []
    if messages and isinstance(messages[0].get("message"), dict):
        messages = normalize_cursor_records(messages)

    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "")
        tool_calls = message.get("tool_calls") or []
        if role == "user":
            query = extract_user_query(content)
            if turn == 0:
                continue
            events.append(
                TraceEvent(
                    event_id=f"e{len(events):04d}",
                    turn=turn,
                    type=EventType.OBSERVATION,
                    content=query or content,
                    metadata={"source": "user"},
                )
            )
            continue
        if role != "assistant":
            continue
        turn += 1
        if content.strip():
            events.append(
                TraceEvent(
                    event_id=f"e{len(events):04d}",
                    turn=turn,
                    type=EventType.REASONING,
                    content=content.strip(),
                )
            )
        for call in tool_calls:
            name = str(call.get("name") or "unknown")
            if name in SKIP_TOOLS:
                continue
            arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            command = format_tool_call(name, arguments)
            event_type = classify_cursor_tool(name, arguments)
            events.append(
                TraceEvent(
                    event_id=f"e{len(events):04d}",
                    turn=turn,
                    type=event_type,
                    content=command,
                    command=command,
                    paths=tool_paths(name, arguments),
                    metadata={"tool": name, "arguments": _small_arguments(arguments)},
                )
            )

    problem = str(row.get("problem_statement") or "")
    if not problem:
        for message in messages:
            if message.get("role") == "user":
                problem = extract_user_query(str(message.get("content") or ""))
                break

    return Trace(
        instance_id=str(row.get("instance_id") or "cursor-unknown"),
        model=str(row.get("model") or "cursor"),
        problem_statement=problem,
        events=events,
        resolved=row.get("resolved"),
        instance_cost=row.get("instance_cost"),
        api_calls=row.get("api_calls") or row.get("tool_calls"),
        split=row.get("_split") or row.get("project"),
        source_split=row.get("source_split") or "cursor",
    )


def classify_cursor_tool(name: str, arguments: dict[str, Any]) -> EventType:
    if name in EDIT_TOOLS:
        return EventType.EDIT
    if name in READ_TOOLS:
        return EventType.READ
    if name in SEARCH_TOOLS:
        return EventType.SEARCH
    if name == "Shell":
        return classify_command(str(arguments.get("command") or ""))
    return EventType.COMMAND


def format_tool_call(name: str, arguments: dict[str, Any]) -> str:
    if name == "Shell":
        return str(arguments.get("command") or "").strip() or "Shell"
    if not arguments:
        return name
    parts = []
    for key, value in arguments.items():
        rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        if len(rendered) > 400:
            rendered = rendered[:200] + "…" + rendered[-80:]
        parts.append(f"{key}={rendered}")
    return f"{name} " + " ".join(parts)


def tool_paths(name: str, arguments: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in (
        "path",
        "target_directory",
        "target_notebook",
        "working_directory",
        "glob_pattern",
        "file",
    ):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value.strip())
    for extra in extract_paths(" ".join(str(value) for value in arguments.values() if isinstance(value, str))):
        if extra not in paths:
            paths.append(extra)
    if name == "Shell":
        for extra in extract_paths(str(arguments.get("command") or "")):
            if extra not in paths:
                paths.append(extra)
    return paths


def _small_arguments(arguments: dict[str, Any], limit: int = 800) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str) and len(value) > limit:
            compact[key] = value[: limit // 2] + "\n...[TRACEGRAPH TRUNCATED]...\n" + value[-limit // 2 :]
        else:
            compact[key] = value
    return compact


def _project_name(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(Path(root).resolve())
        return relative.parts[0]
    except ValueError:
        return path.parent.parent.name


def _infer_projects_root(path: Path) -> Path:
    parts = list(path.resolve().parts)
    if "projects" in parts:
        index = parts.index("projects")
        return Path(*parts[: index + 1])
    return path.parent.parent.parent


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows
