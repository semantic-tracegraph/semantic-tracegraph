from copy import deepcopy
import json

from tracegraph.agent_decomposer import AgentDecomposer, apply_segment_assignments
from tracegraph.agent_judge import AgentJudge, inspect_graph, validate_verdict
from tracegraph.agent_segmenter import WorkSegment
from tracegraph.graph_builder import (
    coerce_json_array,
    graph_from_draft,
    is_placeholder_graph,
    validate_graph_draft,
)
from tracegraph.ingest import parse_messages
from tracegraph.sandbox import TraceSandbox
from tracegraph.schema import NodeStatus, NodeType


def fixture_trace():
    return parse_messages(
        {
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
                {
                    "role": "assistant",
                    "content": "THOUGHT: Fix it.\n```bash\nsed -i 's/old/new/' /testbed/src/message.py\n```",
                },
                {"role": "user", "content": "<returncode>0</returncode><output></output>"},
                {
                    "role": "assistant",
                    "content": "THOUGHT: Test.\n```bash\npytest -q tests/test_message.py\n```",
                },
                {"role": "user", "content": "<returncode>0</returncode><output>1 passed</output>"},
                {
                    "role": "assistant",
                    "content": "THOUGHT: Submit.\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff --cached\n```",
                },
                {
                    "role": "user",
                    "content": "<returncode>0</returncode><output>diff --git a/src/message.py b/src/message.py</output>",
                },
            ],
        }
    )


def fixture_graph_draft() -> dict:
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
                "remaining_work": "Inspect implementation",
            },
            {
                "node_id": "n001",
                "claim": "Patched message.py",
                "type": "implementation",
                "event_span": ["e0004", "e0005"],
                "evidence": [
                    {"event_id": "e0004", "kind": "action", "excerpt": "sed -i"},
                    {"event_id": "e0005", "kind": "observation", "excerpt": ""},
                ],
                "artifact_delta": {"paths": ["src/message.py"]},
                "confidence": 0.8,
                "remaining_work": "Run tests",
            },
            {
                "node_id": "n002",
                "claim": "Ran pytest successfully",
                "type": "verification",
                "event_span": ["e0007", "e0008"],
                "evidence": [
                    {"event_id": "e0007", "kind": "action", "excerpt": "pytest"},
                    {"event_id": "e0008", "kind": "observation", "excerpt": "1 passed"},
                ],
                "artifact_delta": {},
                "confidence": 0.9,
                "remaining_work": "Submit patch",
            },
            {
                "node_id": "n003",
                "claim": "Submitted patch",
                "type": "handoff_artifact",
                "event_span": ["e0010", "e0011"],
                "evidence": [
                    {"event_id": "e0010", "kind": "action", "excerpt": "git diff --cached"},
                    {"event_id": "e0011", "kind": "observation", "excerpt": "diff --git"},
                ],
                "artifact_delta": {},
                "confidence": 0.9,
                "remaining_work": "",
            },
        ],
        "edges": [
            {"source": "n000", "target": "n001", "type": "requires"},
            {"source": "n001", "target": "n002", "type": "requires"},
            {"source": "n002", "target": "n003", "type": "requires"},
        ],
    }


def fixture_segment_submission() -> dict:
    return {
        "segments": [
            {
                "segment_id": "local-find",
                "event_span": ["e0000", "e0002"],
                "objective": "Find the relevant implementation.",
                "action_summary": "Searched for should_close and found message.py.",
                "outcome_summary": "The relevant source location was identified.",
                "evidence_event_ids": ["e0001", "e0002"],
                "artifact_paths": ["src/message.py"],
            },
            {
                "segment_id": "local-impl",
                "event_span": ["e0003", "e0005"],
                "objective": "Patch the relevant implementation.",
                "action_summary": "Edited message.py.",
                "outcome_summary": "The source change was written.",
                "evidence_event_ids": ["e0004", "e0005"],
                "artifact_paths": ["src/message.py"],
            },
            {
                "segment_id": "local-test",
                "event_span": ["e0006", "e0008"],
                "objective": "Check the patched behavior.",
                "action_summary": "Ran pytest on the targeted tests.",
                "outcome_summary": "The targeted test passed.",
                "evidence_event_ids": ["e0007", "e0008"],
            },
            {
                "segment_id": "local-submit",
                "event_span": ["e0009", "e0011"],
                "objective": "Package the final patch.",
                "action_summary": "Submitted the git diff.",
                "outcome_summary": "The patch artifact was emitted.",
                "evidence_event_ids": ["e0010", "e0011"],
            },
        ],
        "coverage_note": "All task-relevant edit, test, and submission events are covered.",
    }


def short_segment_submission() -> dict:
    return {
        "segments": [
            {
                "segment_id": "local-find",
                "event_span": ["e0000", "e0002"],
                "objective": "Find the relevant implementation.",
                "action_summary": "Searched for should_close and found message.py.",
                "outcome_summary": "The relevant source location was identified.",
                "evidence_event_ids": ["e0001", "e0002"],
                "artifact_paths": ["src/message.py"],
            }
        ],
        "coverage_note": "Short trajectory contains only localization work.",
    }


def with_segment_ids(draft: dict, ids: list[list[str]] | None = None) -> dict:
    payload = deepcopy(draft)
    for index, node in enumerate(payload["nodes"]):
        node["segment_ids"] = ids[index] if ids is not None else [f"s{index:03d}"]
    return payload


def _tool_response(tool_calls: list[dict]) -> dict:
    return {"choices": [{"message": {"role": "assistant", "tool_calls": tool_calls}}]}


def two_phase_decomposer_transport(
    graph_draft: dict,
    *,
    segments: dict | None = None,
    run_python_first: bool = False,
    bad_graph_first: dict | None = None,
    malformed_first: bool = False,
    capture: dict | None = None,
):
    calls = {"count": 0, "assignment": 0}
    segment_payload = segments or fixture_segment_submission()

    def transport(payload: dict) -> dict:
        calls["count"] += 1
        names = {tool["function"]["name"] for tool in payload.get("tools") or []}
        if capture is not None and "submit_graph" in names:
            capture["payload"] = payload
        if "submit_segments" in names:
            if run_python_first and calls["count"] == 1:
                return _tool_response(
                    [
                        {
                            "id": "call_py",
                            "type": "function",
                            "function": {
                                "name": "run_python",
                                "arguments": json.dumps({"code": "result = trace.summary()"}),
                            },
                        }
                    ]
                )
            return _tool_response(
                [
                    {
                        "id": f"call_seg_{calls['count']}",
                        "type": "function",
                        "function": {
                            "name": "submit_segments",
                            "arguments": json.dumps(segment_payload),
                        },
                    }
                ]
            )
        calls["assignment"] += 1
        if malformed_first and calls["assignment"] == 1:
            return _tool_response(
                [
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {"name": "submit_graph", "arguments": "{not valid json"},
                    }
                ]
            )
        draft = graph_draft
        if bad_graph_first is not None and calls["assignment"] == 1:
            draft = bad_graph_first
        return _tool_response(
            [
                {
                    "id": f"call_g_{calls['count']}",
                    "type": "function",
                    "function": {
                        "name": "submit_graph",
                        "arguments": json.dumps(draft),
                    },
                }
            ]
        )

    transport.calls = calls  # type: ignore[attr-defined]
    return transport


def test_sandbox_summary_and_grep() -> None:
    sandbox = TraceSandbox(fixture_trace())
    summary = sandbox.summary()
    assert summary["event_count"] == 12
    assert sandbox.grep("should_close")[0]["event_id"] == "e0001"


def test_validate_graph_draft_rejects_unknown_event() -> None:
    trace = fixture_trace()
    draft = fixture_graph_draft()
    draft["nodes"][0]["event_span"] = ["missing", "e0002"]
    errors = validate_graph_draft(trace, draft)
    assert any("unknown event" in error for error in errors)


def test_validate_graph_draft_rejects_string_nodes() -> None:
    trace = fixture_trace()
    errors = validate_graph_draft(trace, {"nodes": ["not a node"], "edges": []})
    assert any("must be an object" in error for error in errors)


def test_coerce_json_array_parses_encoded_lists_and_objects() -> None:
    nodes = [{"node_id": "n1"}]
    assert coerce_json_array(json.dumps(nodes)) == nodes
    assert coerce_json_array([json.dumps(nodes[0])]) == nodes
    assert coerce_json_array(nodes) == nodes
    assert coerce_json_array("not json") == "not json"


def test_validate_graph_draft_accepts_json_encoded_array_strings() -> None:
    trace = fixture_trace()
    draft = fixture_graph_draft()
    errors = validate_graph_draft(
        trace,
        {
            "nodes": json.dumps(draft["nodes"]),
            "edges": json.dumps(draft["edges"]),
        },
    )
    assert errors == []
    graph = graph_from_draft(
        trace,
        {
            "nodes": json.dumps(draft["nodes"]),
            "edges": [json.dumps(edge) for edge in draft["edges"]],
        },
        "agent:test",
    )
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3


def test_validate_graph_draft_rejects_placeholder_one_node_graph() -> None:
    trace = fixture_trace()
    errors = validate_graph_draft(
        trace,
        {
            "nodes": [
                {
                    "node_id": "n1",
                    "claim": "test",
                    "type": "localization",
                    "event_span": ["e0000", "e0001"],
                    "evidence": [{"event_id": "e0001", "kind": "search", "excerpt": "x"}],
                    "artifact_delta": {},
                    "confidence": 0.5,
                    "remaining_work": "",
                }
            ],
            "edges": [],
        },
    )
    assert any("placeholder" in error for error in errors)


def test_placeholder_graph_filter_excludes_dummy_nodes() -> None:
    trace = fixture_trace()
    graph = graph_from_draft(trace, fixture_graph_draft(), "agent:test")
    assert is_placeholder_graph(graph) is False
    graph.nodes = graph.nodes[:1]
    graph.nodes[0].claim = "test"
    graph.nodes[0].evidence[0].excerpt = "x"
    assert is_placeholder_graph(graph) is True


def test_validate_graph_draft_requires_edges_for_multiple_nodes() -> None:
    trace = fixture_trace()
    draft = fixture_graph_draft()
    draft["edges"] = []
    errors = validate_graph_draft(trace, draft)
    assert any("must include causal edges" in error for error in errors)


def _fixture_judge_transport() -> object:
    verdicts = {
        "localization": ("evidenced", 0.72, "e0002 shows message.py match for should_close."),
        "implementation": ("evidenced", 0.75, "e0004-e0005 show a successful edit."),
        "verification": ("verified", 0.9, "e0008 shows pytest passed."),
        "handoff_artifact": ("verified", 0.88, "e0011 contains a git diff."),
    }

    def transport(_payload: dict) -> dict:
        content = _payload["messages"][-1]["content"]
        node_type = json.loads(content.split("\n\n", 1)[1])["node"]["type"]
        status, confidence, reason = verdicts[node_type]
        return _tool_response(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "submit_verdict",
                        "arguments": json.dumps(
                            {
                                "status": status,
                                "confidence": confidence,
                                "reason": reason,
                            }
                        ),
                    },
                }
            ]
        )

    return transport


def test_graph_from_draft_and_verification() -> None:
    trace = fixture_trace()
    graph = graph_from_draft(trace, fixture_graph_draft(), "agent:test")
    judge = AgentJudge("fake", transport=_fixture_judge_transport(), max_steps=4)
    verified = judge.verify(graph, trace)
    by_type = {node.type: node for node in verified.nodes}
    assert by_type[NodeType.LOCALIZATION].status == NodeStatus.EVIDENCED
    assert by_type[NodeType.IMPLEMENTATION].status == NodeStatus.EVIDENCED
    assert by_type[NodeType.VERIFICATION].status == NodeStatus.VERIFIED
    assert by_type[NodeType.HANDOFF_ARTIFACT].status == NodeStatus.VERIFIED
    assert verified.metadata["verifier"] == "agent-judge:fake"
    judge_trace = verified.metadata["agent_traces"]["judge"]
    assert judge_trace["total_steps"] == 4
    assert set(judge_trace["nodes"]) == {node.node_id for node in verified.nodes}


def test_judge_can_inspect_graph_structure() -> None:
    trace = fixture_trace()
    graph = graph_from_draft(trace, fixture_graph_draft(), "agent:test")

    whole = inspect_graph(graph)
    assert whole["root_node_ids"] == ["n000"]
    assert len(whole["nodes"]) == 4
    assert len(whole["edges"]) == 3

    neighborhood = inspect_graph(graph, "n001")
    assert [node["node_id"] for node in neighborhood["parents"]] == ["n000"]
    assert [node["node_id"] for node in neighborhood["children"]] == ["n002"]
    assert neighborhood["is_root"] is False


def test_agent_decomposer_submits_valid_graph() -> None:
    trace = fixture_trace()
    transport = two_phase_decomposer_transport(
        with_segment_ids(fixture_graph_draft()),
        run_python_first=True,
    )
    graph = AgentDecomposer("fake", transport=transport, max_steps=4).decompose(trace)
    assert len(graph.nodes) == 4
    assert graph.metadata["decomposer"] == "agent:fake"
    assert graph.metadata["agent_steps"] == 1
    assert graph.nodes[0].segment_ids == ["s000"]
    assert graph.nodes[0].event_span == ("e0000", "e0002")
    assert graph.metadata["agent_traces"]["segmenter"]["steps"] == 2
    decomposer_trace = graph.metadata["agent_traces"]["decomposer"]
    assert decomposer_trace["steps"] == 1
    assert decomposer_trace["model"] == "fake"
    roles = [message["role"] for message in decomposer_trace["messages"]]
    assert "system" in roles
    assert "tool" in roles


def test_agent_decomposer_accepts_json_encoded_node_array() -> None:
    trace = fixture_trace()
    draft = with_segment_ids(fixture_graph_draft())
    encoded = {
        "nodes": json.dumps(draft["nodes"]),
        "edges": json.dumps(draft["edges"]),
    }
    transport = two_phase_decomposer_transport(encoded)
    graph = AgentDecomposer("fake", transport=transport, max_steps=2).decompose(trace)
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3


def test_agent_decomposer_retries_after_validation_error() -> None:
    trace = fixture_trace()
    draft = with_segment_ids(fixture_graph_draft())
    bad_draft = deepcopy(draft)
    bad_draft["nodes"][0]["claim"] = "test"
    transport = two_phase_decomposer_transport(draft, bad_graph_first=bad_draft)
    graph = AgentDecomposer("fake", transport=transport, max_steps=4).decompose(trace)
    assert len(graph.nodes) == 4
    assert transport.calls["assignment"] == 2


def test_agent_decomposer_recovers_from_malformed_arguments() -> None:
    trace = fixture_trace()
    transport = two_phase_decomposer_transport(
        with_segment_ids(fixture_graph_draft()),
        malformed_first=True,
    )
    graph = AgentDecomposer("fake", transport=transport, max_steps=4).decompose(trace)
    assert len(graph.nodes) == 4
    assert transport.calls["assignment"] == 2


def test_apply_segment_assignments_merges_consecutive_and_skips_unmatched() -> None:
    segments = [
        WorkSegment(
            segment_id="s000",
            trace_key="task::agent",
            event_span=("e0000", "e0002"),
            objective="Find code",
            action_summary="Searched",
            outcome_summary="Found",
            evidence_event_ids=["e0001"],
        ),
        WorkSegment(
            segment_id="s001",
            trace_key="task::agent",
            event_span=("e0003", "e0005"),
            objective="Keep looking",
            action_summary="Read more",
            outcome_summary="Confirmed site",
            evidence_event_ids=["e0004"],
        ),
        WorkSegment(
            segment_id="s002",
            trace_key="task::agent",
            event_span=("e0006", "e0008"),
            objective="Unmatched chatter",
            action_summary="Noted a side issue",
            outcome_summary="Left it alone",
            evidence_event_ids=["e0007"],
        ),
        WorkSegment(
            segment_id="s003",
            trace_key="task::agent",
            event_span=("e0009", "e0011"),
            objective="Patch",
            action_summary="Edited",
            outcome_summary="Changed source",
            evidence_event_ids=["e0010"],
        ),
    ]
    draft = {
        "nodes": [
            {
                "node_id": "n000",
                "claim": "Located the relevant site",
                "type": "localization",
                "segment_ids": ["s001", "s000"],
                "evidence": [{"event_id": "e0001", "kind": "action", "excerpt": "rg"}],
            },
            {
                "node_id": "n001",
                "claim": "Patched the relevant site",
                "type": "implementation",
                "segment_ids": ["s003"],
                "evidence": [{"event_id": "e0010", "kind": "action", "excerpt": "sed"}],
            },
        ],
        "edges": [{"source": "n000", "target": "n001", "type": "requires"}],
    }
    filled, errors = apply_segment_assignments(draft, segments)
    assert errors == []
    assert filled["nodes"][0]["segment_ids"] == ["s000", "s001"]
    assert filled["nodes"][0]["event_span"] == ["e0000", "e0005"]
    assert filled["nodes"][1]["event_span"] == ["e0009", "e0011"]


def test_apply_segment_assignments_rejects_nonconsecutive_and_duplicate_ids() -> None:
    segments = [
        WorkSegment(
            segment_id="s000",
            trace_key="task::agent",
            event_span=("e0000", "e0002"),
            objective="Find code",
            action_summary="Searched",
            outcome_summary="Found",
            evidence_event_ids=["e0001"],
        ),
        WorkSegment(
            segment_id="s001",
            trace_key="task::agent",
            event_span=("e0003", "e0005"),
            objective="Patch",
            action_summary="Edited",
            outcome_summary="Changed",
            evidence_event_ids=["e0004"],
        ),
        WorkSegment(
            segment_id="s002",
            trace_key="task::agent",
            event_span=("e0006", "e0008"),
            objective="Test",
            action_summary="Ran tests",
            outcome_summary="Passed",
            evidence_event_ids=["e0007"],
        ),
    ]
    skipped, skip_errors = apply_segment_assignments(
        {
            "nodes": [
                {
                    "node_id": "n000",
                    "segment_ids": ["s000", "s002"],
                }
            ],
            "edges": [],
        },
        segments,
    )
    assert any("consecutive" in error for error in skip_errors)
    assert "event_span" not in skipped["nodes"][0]

    _, dup_errors = apply_segment_assignments(
        {
            "nodes": [
                {"node_id": "n000", "segment_ids": ["s000"]},
                {"node_id": "n001", "segment_ids": ["s000"]},
            ],
            "edges": [],
        },
        segments,
    )
    assert any("assigned to both" in error for error in dup_errors)


def test_agent_judge_does_not_share_sandbox_across_models() -> None:
    trace_a = fixture_trace()
    trace_b = parse_messages(
        {
            "instance_id": trace_a.instance_id,
            "model": "other-model",
            "resolved": True,
            "messages": [
                {"role": "assistant", "content": "THOUGHT: noop.\n```bash\nls\n```"},
                {"role": "user", "content": "<returncode>0</returncode><output>x</output>"},
            ],
        }
    )
    node = graph_from_draft(trace_a, fixture_graph_draft(), "agent:test").nodes[0]

    def transport(_payload: dict) -> dict:
        return _tool_response(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "submit_verdict",
                        "arguments": json.dumps(
                            {"status": "evidenced", "confidence": 0.6, "reason": "e0002 ok."}
                        ),
                    },
                }
            ]
        )

    judge = AgentJudge("fake", transport=transport, max_steps=4)
    judge(node, trace_a)
    judge(node, trace_b)
    assert (trace_a.instance_id, trace_a.model) in judge._sandboxes
    assert (trace_b.instance_id, trace_b.model) in judge._sandboxes
    assert judge._sandboxes[(trace_a.instance_id, trace_a.model)].model == trace_a.model
    assert judge._sandboxes[(trace_b.instance_id, trace_b.model)].model == "other-model"


def test_validate_verdict_rejects_invalid_status() -> None:
    errors = validate_verdict(
        {"status": "superseded", "confidence": 0.9, "reason": "Looks good."}
    )
    assert any("claimed, evidenced, or verified" in error for error in errors)


def test_agent_judge_submits_valid_verdict() -> None:
    trace = fixture_trace()
    graph = graph_from_draft(trace, fixture_graph_draft(), "agent:test")
    node = graph.nodes[0]
    calls = {"count": 0}

    def transport(_payload: dict) -> dict:
        calls["count"] += 1
        if calls["count"] == 1:
            return _tool_response(
                [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "run_python",
                            "arguments": json.dumps(
                                {"code": "result = trace.event('e0002')"}
                            ),
                        },
                    }
                ]
            )
        return _tool_response(
            [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "submit_verdict",
                        "arguments": json.dumps(
                            {
                                "status": "evidenced",
                                "confidence": 0.81,
                                "reason": "e0002 shows message.py match for should_close.",
                            }
                        ),
                    },
                }
            ]
        )

    verdict, agent_trace = AgentJudge("fake", transport=transport, max_steps=4)(node, trace)
    assert verdict.status == NodeStatus.EVIDENCED
    assert verdict.confidence == 0.81
    assert "e0002" in verdict.reason
    assert calls["count"] == 2
    assert agent_trace["steps"] == 2
    assert agent_trace["node_id"] == node.node_id


def test_agent_judge_retries_after_validation_error() -> None:
    trace = fixture_trace()
    graph = graph_from_draft(trace, fixture_graph_draft(), "agent:test")
    node = graph.nodes[0]
    calls = {"count": 0}

    def transport(_payload: dict) -> dict:
        calls["count"] += 1
        verdict = (
            {"status": "superseded", "confidence": 0.9, "reason": "Strong evidence."}
            if calls["count"] == 1
            else {
                "status": "claimed",
                "confidence": 0.4,
                "reason": "Excerpt overstates what e0002 shows.",
            }
        )
        return _tool_response(
            [
                {
                    "id": f"call_{calls['count']}",
                    "type": "function",
                    "function": {
                        "name": "submit_verdict",
                        "arguments": json.dumps(verdict),
                    },
                }
            ]
        )

    verdict, agent_trace = AgentJudge("fake", transport=transport, max_steps=4)(node, trace)
    assert verdict.status == NodeStatus.CLAIMED
    assert calls["count"] == 2
    assert agent_trace["steps"] == 2
