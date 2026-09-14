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
