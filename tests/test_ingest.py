import json
from pathlib import Path

from tracegraph.cursor_ingest import ingest_cursor_transcripts, transcript_to_row, transcript_to_rows
from tracegraph.ingest import classify_command, extract_paths, parse_messages
from tracegraph.schema import EventType


def test_parse_shell_turn_and_observation() -> None:
    row = {
        "instance_id": "task-1",
        "model": "test-model",
        "resolved": False,
        "messages": [
            {
                "role": "assistant",
                "content": "THOUGHT: Inspect the implementation.\n\n```bash\nsed -n '1,20p' /testbed/src/mod.py\n```",
            },
            {
                "role": "user",
                "content": "<returncode>0</returncode>\n<output>\ndef target():\n    pass\n</output>",
            },
        ],
    }
    trace = parse_messages(row)
    assert [event.type for event in trace.events] == [
        EventType.REASONING,
        EventType.READ,
        EventType.OBSERVATION,
    ]
    assert trace.events[1].returncode == 0
    assert trace.events[1].paths == ["src/mod.py"]
    assert trace.events[2].symbols == ["target"]


def test_command_classification_and_paths() -> None:
    assert classify_command("pytest -q") == EventType.TEST
    assert classify_command("sed -i 's/a/b/' /testbed/app/a.py") == EventType.EDIT
    assert classify_command("rg needle src") == EventType.SEARCH
    assert extract_paths("cat /testbed/src/a.py and tests/test_a.py") == [
        "src/a.py",
        "tests/test_a.py",
    ]


def _write_cursor_transcript(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "role": "user",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": "<timestamp>now</timestamp>\n<user_query>Fix the login bug</user_query>",
                    }
                ]
            },
        },
        {
            "role": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "I'll search for the login handler."},
                    {
                        "type": "tool_use",
                        "name": "Grep",
                        "input": {"pattern": "login", "path": "src/auth.py"},
                    },
                    {
                        "type": "tool_use",
                        "name": "SwitchMode",
                        "input": {"target_mode_id": "agent"},
                    },
                ]
            },
        },
        {
            "role": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "StrReplace",
                        "input": {"path": "src/auth.py", "old_string": "old", "new_string": "new"},
                    }
                ]
            },
        },
        {
            "role": "user",
            "message": {
                "content": [{"type": "text", "text": "<user_query>also add a test</user_query>"}]
            },
        },
        {
            "role": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Shell",
                        "input": {"command": "pytest -q tests/test_auth.py"},
                    }
                ]
            },
        },
        {"type": "turn_ended", "status": "success"},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_parse_cursor_transcript_events(tmp_path: Path) -> None:
    transcript = (
        tmp_path
        / "projects"
        / "demo-repo"
        / "agent-transcripts"
        / "abc123"
        / "abc123.jsonl"
    )
    _write_cursor_transcript(transcript)
    row = transcript_to_row(transcript, root=tmp_path / "projects")
    assert row["instance_id"] == "demo-repo/abc123"
    assert row["problem_statement"] == "Fix the login bug"
    assert row["model"] == "cursor"
    trace = parse_messages(row)
    assert [event.type for event in trace.events] == [
        EventType.REASONING,
        EventType.SEARCH,
        EventType.EDIT,
        EventType.OBSERVATION,
        EventType.TEST,
    ]
    assert trace.events[1].paths == ["src/auth.py"]
    assert trace.events[2].metadata["tool"] == "StrReplace"
    assert trace.events[3].content == "also add a test"
    assert "SwitchMode" not in {event.metadata.get("tool") for event in trace.events}


def test_cursor_session_splits_at_each_user_request(tmp_path: Path) -> None:
    transcript = (
        tmp_path
        / "projects"
        / "demo-repo"
        / "agent-transcripts"
        / "abc123"
        / "abc123.jsonl"
    )
    _write_cursor_transcript(transcript)
    rows = transcript_to_rows(transcript, root=tmp_path / "projects")
    assert [row["instance_id"] for row in rows] == [
        "demo-repo/abc123/u000",
        "demo-repo/abc123/u001",
    ]
    assert rows[0]["problem_statement"] == "Fix the login bug"
    assert rows[1]["problem_statement"] == "also add a test"
    assert rows[1]["prior_user_queries"] == ["Fix the login bug"]
    assert rows[1]["user_turn_index"] == 1
    first = parse_messages(rows[0])
    second = parse_messages(rows[1])
    assert [event.type for event in first.events] == [
        EventType.REASONING,
        EventType.SEARCH,
        EventType.EDIT,
    ]
    assert [event.type for event in second.events] == [EventType.TEST]
    assert "also add a test" not in first.problem_statement


def test_ingest_cursor_cli_writes_jsonl(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    transcript = root / "demo-repo" / "agent-transcripts" / "abc123" / "abc123.jsonl"
    _write_cursor_transcript(transcript)
    chat_only = root / "demo-repo" / "agent-transcripts" / "chat1" / "chat1.jsonl"
    chat_only.parent.mkdir(parents=True, exist_ok=True)
    chat_only.write_text(
        json.dumps(
            {
                "role": "user",
                "message": {"content": [{"type": "text", "text": "<user_query>hi</user_query>"}]},
            }
        )
        + "\n"
    )
    output = tmp_path / "cursor.jsonl"
    from tracegraph.cli import main

    main(
        [
            "ingest-cursor",
            "--root",
            str(root),
            "--output",
            str(output),
            "--project",
            "demo-repo",
        ]
    )
    written = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    assert [row["instance_id"] for row in written] == [
        "demo-repo/abc123/u000",
        "demo-repo/abc123/u001",
    ]
    rows = ingest_cursor_transcripts(root, include_chat=True)
    assert len(rows) == 3
