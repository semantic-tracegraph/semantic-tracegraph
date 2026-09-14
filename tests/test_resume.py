import json
from pathlib import Path
from unittest.mock import patch

from tracegraph.agent_decomposer import AgentDecomposer
from tracegraph.cli import main
from tracegraph.data import completed_keys, matches_trace_models, trajectory_key
from tracegraph.graph_builder import graph_from_draft
from tracegraph.ingest import parse_messages
from tracegraph.schema import to_dict

from test_graph import _tool_response

def test_assistant_message_accepts_string_payload() -> None:
    from tracegraph.agent_trace import assistant_message

    message = assistant_message({"choices": [{"message": "just text"}]})
    assert message["role"] == "assistant"
    assert message["content"] == "just text"
    assert not message.get("tool_calls")



def test_trajectory_key_from_raw_and_graph_records() -> None:
    raw = {"instance_id": "a", "model": "claude-opus-4.7"}
    assert trajectory_key(raw) == "a::claude-opus-4.7"
    wrapped = {"trace": {"instance_id": "a", "model": "gpt-5-mini"}}
    assert trajectory_key(wrapped) == "a::gpt-5-mini"


def test_matches_trace_models() -> None:
    row = {"instance_id": "a", "model": "gpt-5-mini"}
    assert matches_trace_models(row, set())
    assert matches_trace_models(row, {"gpt-5-mini"})
    assert not matches_trace_models(row, {"claude-opus-4.7"})


def _small_draft() -> dict:
    return {
        "nodes": [
            {
                "node_id": "n000",
                "claim": "Localized should_close in message.py",
                "type": "localization",
                "event_span": ["e0001", "e0002"],
                "evidence": [
                    {"event_id": "e0001", "kind": "action", "excerpt": "rg should_close"},
                    {"event_id": "e0002", "kind": "observation", "excerpt": "message.py"},
                ],
                "artifact_delta": {"paths": ["src/message.py"]},
                "confidence": 0.8,
                "remaining_work": "",
            }
        ],
        "edges": [],
    }


def test_decompose_resumes_from_partial_output(tmp_path: Path) -> None:
    input_path = tmp_path / "pilot.jsonl"
    output_path = tmp_path / "graphs.jsonl"
    first_row = {
        "instance_id": "task",
        "model": "agent",
        "resolved": True,
        "messages": [
            {
                "role": "assistant",
                "content": "THOUGHT: Find it.\n```bash\nrg should_close /testbed/src/message.py\n```",
            },
            {
                "role": "user",
                "content": "<returncode>0</returncode><output>/testbed/src/message.py:10:def should_close</output>",
            },
        ],
    }
    second_row = {**first_row, "instance_id": "task-2"}
    input_path.write_text(json.dumps(first_row) + "\n" + json.dumps(second_row) + "\n")

    seed_trace = parse_messages(first_row)
    seed_graph = graph_from_draft(seed_trace, _small_draft(), "agent:fake")
    output_path.write_text(
        json.dumps({"trace": to_dict(seed_trace), "graph": to_dict(seed_graph)}) + "\n"
    )

    calls = {"count": 0}

    def transport(_payload: dict) -> dict:
        calls["count"] += 1
        return _tool_response(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "submit_graph",
                        "arguments": json.dumps(_small_draft()),
                    },
                }
            ]
        )

    def factory(**kwargs):
        return AgentDecomposer(
            model=kwargs.get("model"),
            max_steps=kwargs.get("max_steps", 4),
            transport=transport,
        )

    with patch("tracegraph.cli.AgentDecomposer", factory):
        main(
            [
                "decompose",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--model",
                "fake",
            ]
        )

    lines = [line for line in output_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    assert calls["count"] == 1
    assert completed_keys(output_path) == {"task::agent", "task-2::agent"}
