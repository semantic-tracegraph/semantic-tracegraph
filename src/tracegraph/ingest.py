from __future__ import annotations

import re
import shlex
from pathlib import PurePosixPath
from typing import Any

from .schema import EventType, Trace, TraceEvent

BASH_BLOCK = re.compile(r"```(?:bash|sh)?\s*\n(?P<command>.*?)\n```", re.DOTALL)
THOUGHT = re.compile(r"(?:^|\n)THOUGHT:\s*(?P<thought>.*?)(?=\n```)", re.DOTALL | re.IGNORECASE)
RETURNCODE = re.compile(r"<returncode>(?P<code>-?\d+)</returncode>")
OUTPUT = re.compile(r"<output>\n?(?P<output>.*?)\n?</output>", re.DOTALL)
PATH = re.compile(r"(?<![\w.-])(/(?:testbed/)?[\w./-]+|(?:src|test|tests|lib|app)/[\w./-]+)")
SYMBOL = re.compile(r"\b(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)")


def classify_command(command: str) -> EventType:
    lower = command.lower()
    if "complete_task_and_submit_final_output" in lower or "git diff --cached" in lower:
        return EventType.SUBMISSION
    if re.search(r"\b(pytest|unittest|npm test|cargo test|go test|rspec|tox)\b", lower):
        return EventType.TEST
    if re.search(r"\b(sed\s+-i|apply_patch|cat\s+.*>|python\s+-c|perl\s+-[pi])\b", lower):
        return EventType.EDIT
    if re.search(r"\b(grep|rg|find)\b", lower):
        return EventType.SEARCH
    if re.search(r"\b(cat|sed\s+-n|head|tail|less|nl\s+)\b", lower):
        return EventType.READ
    return EventType.COMMAND


def parse_messages(row: dict[str, Any]) -> Trace:
    from .cursor_ingest import looks_like_cursor_messages, parse_cursor_messages

    messages = row.get("messages") or []
    if row.get("source") == "cursor" or looks_like_cursor_messages(messages):
        return parse_cursor_messages(row)

    events: list[TraceEvent] = []
    turn = 0
    pending_command_id: str | None = None
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "assistant":
            turn += 1
            thought_match = THOUGHT.search(content)
            if thought_match and thought_match.group("thought").strip():
                events.append(
                    TraceEvent(
                        event_id=f"e{len(events):04d}",
                        turn=turn,
                        type=EventType.REASONING,
                        content=thought_match.group("thought").strip(),
                    )
                )
            command_match = BASH_BLOCK.search(content)
            if command_match:
                command = command_match.group("command").strip()
                event = TraceEvent(
                    event_id=f"e{len(events):04d}",
                    turn=turn,
                    type=classify_command(command),
                    content=command,
                    command=command,
                    paths=extract_paths(command),
                )
                events.append(event)
                pending_command_id = event.event_id
        elif role == "user" and ("<returncode>" in content or "<output>" in content):
            code_match = RETURNCODE.search(content)
            output_match = OUTPUT.search(content)
            output = output_match.group("output").strip() if output_match else content.strip()
            event = TraceEvent(
                event_id=f"e{len(events):04d}",
                turn=turn,
                type=EventType.OBSERVATION,
                content=output,
                returncode=int(code_match.group("code")) if code_match else None,
                paths=extract_paths(output),
                symbols=SYMBOL.findall(output),
                metadata={"command_event_id": pending_command_id},
            )
            events.append(event)
            if pending_command_id:
                for previous in reversed(events[:-1]):
                    if previous.event_id == pending_command_id:
                        previous.returncode = event.returncode
                        break
            pending_command_id = None

    return Trace(
        instance_id=str(row["instance_id"]),
        model=str(row.get("model") or row.get("_model_key") or "unknown"),
        problem_statement=str(row.get("problem_statement") or ""),
        events=events,
        resolved=row.get("resolved"),
        instance_cost=row.get("instance_cost"),
        api_calls=row.get("api_calls"),
        split=row.get("_split"),
        source_split=row.get("source_split"),
    )


def extract_paths(text: str) -> list[str]:
    paths: list[str] = []
    for raw in PATH.findall(text):
        clean = raw.rstrip(".,:;)'\"")
        if clean.startswith("/testbed/"):
            clean = clean.removeprefix("/testbed/")
        try:
            normalized = str(PurePosixPath(clean))
        except ValueError:
            continue
        if normalized not in paths:
            paths.append(normalized)
    return paths


def safe_command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()
