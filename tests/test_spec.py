from pathlib import Path
from unittest.mock import patch
import json

import pytest

from tracegraph.agent_decomposer import AgentDecomposer, decomposer_tools
from tracegraph.agent_segmenter import WorkSegment, validate_segments
from tracegraph.agent_spec_constructor import (
    AgentSpecConstructor,
    validate_clustering,
)
from tracegraph.cli import main
from tracegraph.graph_builder import graph_from_draft, validate_graph_draft
from tracegraph.graph_spec import (
    DEFAULT_GRAPH_SPEC,
    load_graph_spec,
    spec_hash,
    validate_graph_spec,
)
from tracegraph.schema import (
    EdgeType,
    EdgeTypeSpec,
    GraphSpec,
    NodeType,
    NodeTypeSpec,
    graph_spec_from_dict,
    to_dict,
)

from test_graph import (
    _tool_response,
    fixture_graph_draft,
    fixture_trace,
    two_phase_decomposer_transport,
    with_segment_ids,
)


def tiny_spec() -> GraphSpec:
    return GraphSpec(
        spec_id="tiny-coding",
        version="1",
        notes="Minimal find/fix ontology for tests.",
        typical_order=["find", "fix"],
        node_types=[
            NodeTypeSpec(
                name="find",
                description="Located the relevant code or failure site.",
                inclusion_criteria=["Search or read identified a specific file or symbol."],
                default_weight=0.6,
            ),
            NodeTypeSpec(
                name="fix",
                description="Implemented the change that addresses the task.",
                evidence_requirements=["Cite the edit events and changed paths."],
                default_weight=1.2,
            ),
        ],
        edge_types=[
            EdgeTypeSpec(
                name="leads_to",
                description="The source enables the target accomplishment.",
                is_dependency=True,
            )
        ],
    )


def tiny_draft() -> dict:
    return {
        "nodes": [
            {
                "node_id": "n000",
                "claim": "Located should_close in message.py",
                "type": "find",
                "event_span": ["e0001", "e0002"],
                "evidence": [
                    {"event_id": "e0001", "kind": "action", "excerpt": "rg should_close"},
                    {"event_id": "e0002", "kind": "observation", "excerpt": "message.py"},
                ],
                "artifact_delta": {"paths": ["src/message.py"]},
                "confidence": 0.8,
                "remaining_work": "Patch the function",
            },
            {
                "node_id": "n001",
                "claim": "Patched message.py to fix should_close",
                "type": "fix",
                "event_span": ["e0004", "e0005"],
                "evidence": [
                    {"event_id": "e0004", "kind": "action", "excerpt": "sed -i"},
                    {"event_id": "e0005", "kind": "observation", "excerpt": "returncode 0"},
                ],
                "artifact_delta": {"paths": ["src/message.py"]},
                "confidence": 0.8,
                "remaining_work": "",
            },
        ],
        "edges": [{"source": "n000", "target": "n001", "type": "leads_to"}],
    }


def segment_submission() -> dict:
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
                "segment_id": "local-fix",
                "event_span": ["e0003", "e0011"],
                "objective": "Patch and check the relevant implementation.",
                "action_summary": "Edited message.py, ran the test, and submitted the diff.",
                "outcome_summary": "The patch was produced and its targeted test passed.",
                "evidence_event_ids": ["e0004", "e0005", "e0007", "e0008", "e0010", "e0011"],
                "artifact_paths": ["src/message.py"],
            },
        ],
        "coverage_note": "All task-relevant edit, test, and submission events are covered.",
    }


def clustering_submission() -> dict:
    spec = tiny_spec()
    return {
        "spec_id": spec.spec_id,
        "version": spec.version,
        "notes": spec.notes,
        "typical_order": spec.typical_order,
        "require_dag": True,
        "allow_disconnected_roots": True,
        "require_edges_if_multiple_nodes": True,
        "clusters": [
            {
                "node_type": "find",
                "member_segment_ids": ["t000_s000"],
                "description": spec.node_types[0].description,
                "similarity_basis": [
                    "The intent and state transition identify a relevant code location."
                ],
                "distinguishing_criteria": [
                    "Unlike fix work, this class does not change repository state."
                ],
                "inclusion_criteria": spec.node_types[0].inclusion_criteria,
                "singleton_justification": (
                    "Code-location work is a reusable class seen in coding trajectories."
                ),
                "default_weight": spec.node_types[0].default_weight,
            },
            {
                "node_type": "fix",
                "member_segment_ids": ["t000_s001"],
                "description": spec.node_types[1].description,
                "similarity_basis": [
                    "The intent is to change task-relevant repository state."
                ],
                "distinguishing_criteria": [
                    "Unlike find work, this class produces the implementation artifact."
                ],
                "evidence_requirements": spec.node_types[1].evidence_requirements,
                "singleton_justification": (
                    "Implementation work is a reusable class seen in coding trajectories."
                ),
                "default_weight": spec.node_types[1].default_weight,
            },
        ],
        "edge_types": [to_dict(edge_type) for edge_type in spec.edge_types],
    }


def test_default_spec_matches_legacy_enums() -> None:
    assert DEFAULT_GRAPH_SPEC.node_name_set() == {item.value for item in NodeType}
    assert DEFAULT_GRAPH_SPEC.edge_name_set() == {item.value for item in EdgeType}
    assert validate_graph_spec(DEFAULT_GRAPH_SPEC) == []


def test_validate_graph_spec_rejects_bad_names_and_missing_dependency() -> None:
    spec = tiny_spec()
    spec.node_types[0].name = "Find Site"
    spec.edge_types[0].is_dependency = False
    errors = validate_graph_spec(spec)
    assert any("Invalid node type name" in error for error in errors)
    assert any("is_dependency=true" in error for error in errors)


def test_custom_spec_accepts_new_types_and_rejects_default_labels() -> None:
    trace = fixture_trace()
    spec = tiny_spec()
    assert validate_graph_draft(trace, tiny_draft(), spec) == []
    graph = graph_from_draft(trace, tiny_draft(), "agent:test", spec=spec)
    assert [node.type for node in graph.nodes] == ["find", "fix"]
    assert graph.nodes[1].weight == 1.2
    assert graph.metadata["graph_spec_id"] == "tiny-coding"
    assert graph.metadata["graph_spec_hash"] == spec_hash(spec)

    default_draft = fixture_graph_draft()
    errors = validate_graph_draft(trace, default_draft, spec)
    assert any("invalid type" in error for error in errors)


def test_validate_graph_draft_rejects_unknown_default_type() -> None:
    trace = fixture_trace()
    draft = fixture_graph_draft()
    draft["nodes"][0]["type"] = "planning"
    errors = validate_graph_draft(trace, draft)
    assert any("invalid type: planning" in error for error in errors)


def test_validate_graph_draft_rejects_cycles() -> None:
    trace = fixture_trace()
    draft = fixture_graph_draft()
    draft["edges"].append({"source": "n003", "target": "n000", "type": "requires"})
    errors = validate_graph_draft(trace, draft)
    assert any("cycle" in error for error in errors)


def test_decomposer_prompt_and_tool_schema_use_spec() -> None:
    spec = tiny_spec()
    captured: dict = {}
    transport = two_phase_decomposer_transport(
        with_segment_ids(tiny_draft()),
        segments=segment_submission(),
        capture=captured,
    )

    graph = AgentDecomposer("fake", transport=transport, max_steps=2, graph_spec=spec).decompose(
        fixture_trace()
    )
    system = captured["payload"]["messages"][0]["content"]
    assert "`find`" in system
    assert "Located the relevant code or failure site." in system
    assert "`leads_to`" in system
    assert "localization" not in system
    submit = next(
        tool["function"] for tool in captured["payload"]["tools"] if tool["function"]["name"] == "submit_graph"
    )
    assert submit["parameters"]["properties"]["nodes"]["items"]["properties"]["type"]["enum"] == [
        "find",
        "fix",
    ]
    assert graph.nodes[0].type == "find"
    assert graph.nodes[0].segment_ids == ["s000"]
    assert graph.nodes[0].event_span == ("e0000", "e0002")


def test_decomposer_tools_default_enum_matches_schema() -> None:
    tools = decomposer_tools(DEFAULT_GRAPH_SPEC)
    submit = next(tool["function"] for tool in tools if tool["function"]["name"] == "submit_graph")
    assert submit["parameters"]["properties"]["nodes"]["items"]["properties"]["type"]["enum"] == (
        DEFAULT_GRAPH_SPEC.node_names()
    )
    assert "segment_ids" in submit["parameters"]["properties"]["nodes"]["items"]["required"]


def test_validate_segments_rejects_overlap_and_out_of_span_evidence() -> None:
    payload = segment_submission()
    payload["segments"][1]["event_span"] = ["e0002", "e0011"]
    payload["segments"][0]["evidence_event_ids"].append("e0004")
    errors = validate_segments(fixture_trace(), payload["segments"])
    assert any("overlap" in error for error in errors)
    assert any("outside its span" in error for error in errors)


def test_validate_segments_requires_edit_test_and_submission_coverage() -> None:
    payload = segment_submission()
    payload["segments"] = payload["segments"][:1]
    errors = validate_segments(fixture_trace(), payload["segments"])
    assert any("edit/test/submission events uncovered" in error for error in errors)


def test_validate_clustering_requires_exact_partition() -> None:
    segments = [
        WorkSegment(
            segment_id="s1",
            trace_key="task::agent",
            event_span=("e0000", "e0002"),
            objective="Find code",
            action_summary="Searched source",
            outcome_summary="Found source",
            evidence_event_ids=["e0001"],
        ),
        WorkSegment(
            segment_id="s2",
            trace_key="task::agent",
            event_span=("e0003", "e0005"),
            objective="Change code",
            action_summary="Edited source",
            outcome_summary="Produced patch",
            evidence_event_ids=["e0004"],
        ),
    ]
    payload = clustering_submission()
    payload["clusters"][0]["member_segment_ids"] = ["s1"]
    payload["clusters"][1]["member_segment_ids"] = ["s1"]
    errors, _, _ = validate_clustering(segments, payload)
    assert any("not assigned" in error and "s2" in error for error in errors)
    assert any("multiple clusters" in error and "s1" in error for error in errors)


def test_spec_constructor_submits_valid_spec() -> None:
    calls = {"count": 0}

    def transport(payload: dict) -> dict:
        calls["count"] += 1
        tool_names = {tool["function"]["name"] for tool in payload["tools"]}
        name = (
            "submit_segments"
            if "submit_segments" in tool_names
            else "submit_clustering"
        )
        arguments = (
            segment_submission() if name == "submit_segments" else clustering_submission()
        )
        return _tool_response(
            [
                {
                    "id": f"call_{calls['count']}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ]
        )

    built, agent_trace = AgentSpecConstructor("fake", transport=transport, max_steps=4).construct(
        [fixture_trace()]
    )
    assert built.node_names() == ["find", "fix"]
    assert built.edge_names() == ["leads_to"]
    assert agent_trace["agent"] == "spec_constructor"
    assert len(agent_trace["segments"]) == 2
    assert [cluster["node_type"] for cluster in agent_trace["clusters"]] == ["find", "fix"]
    assert set(agent_trace["segmentation"]) == {"task::agent"}
    assert agent_trace["clustering"]["agent"] == "segment_clusterer"
    assert calls["count"] == 2


def test_spec_constructor_retries_after_validation_error() -> None:
    calls = {"count": 0}

    def transport(payload: dict) -> dict:
        calls["count"] += 1
        tool_names = {tool["function"]["name"] for tool in payload["tools"]}
        if "submit_segments" in tool_names:
            arguments = segment_submission()
            if calls["count"] == 1:
                arguments["segments"][1]["event_span"] = ["missing", "e0011"]
            name = "submit_segments"
        else:
            arguments = clustering_submission()
            if calls["count"] == 3:
                arguments["clusters"][1]["member_segment_ids"] = []
            name = "submit_clustering"
        return _tool_response(
            [
                {
                    "id": f"call_{calls['count']}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ]
        )

    built, _ = AgentSpecConstructor("fake", transport=transport, max_steps=4).construct(
        [fixture_trace()]
    )
    assert built.spec_id == "tiny-coding"
    assert calls["count"] == 4


def test_load_graph_spec_reads_wrapped_construct_spec_file(tmp_path: Path) -> None:
    spec = tiny_spec()
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"spec": to_dict(spec), "provenance": {"seed": 7}}))
    loaded = load_graph_spec(path)
    roundtrip = graph_spec_from_dict(to_dict(spec))
    assert loaded.spec_id == roundtrip.spec_id
    assert spec_hash(loaded) == spec_hash(roundtrip)


def test_construct_spec_cli_writes_wrapped_json(tmp_path: Path) -> None:
    input_path = tmp_path / "pilot.jsonl"
    output_path = tmp_path / "spec.json"
    row = {
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
    input_path.write_text(json.dumps(row) + "\n")

    def factory(**kwargs):
        def transport(payload: dict) -> dict:
            tool_names = {tool["function"]["name"] for tool in payload["tools"]}
            if "submit_segments" in tool_names:
                name = "submit_segments"
                arguments = {
                    "segments": [
                        {
                            "segment_id": "local-find",
                            "event_span": ["e0000", "e0002"],
                            "objective": "Find the relevant implementation.",
                            "action_summary": "Searched for should_close.",
                            "outcome_summary": "Found message.py.",
                            "evidence_event_ids": ["e0001", "e0002"],
                            "artifact_paths": ["src/message.py"],
                        }
                    ]
                }
            else:
                name = "submit_clustering"
                arguments = clustering_submission()
                arguments["clusters"] = [arguments["clusters"][0]]
                arguments["typical_order"] = ["find"]
            return _tool_response(
                [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                ]
            )

        return AgentSpecConstructor(
            model=kwargs.get("model"),
            max_steps=kwargs.get("max_steps", 4),
            transport=transport,
        )

    with patch("tracegraph.cli.AgentSpecConstructor", factory):
        main(
            [
                "construct-spec",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--model",
                "fake",
                "--sample-size",
                "1",
            ]
        )

    payload = json.loads(output_path.read_text())
    assert payload["spec"]["spec_id"] == "tiny-coding"
    assert payload["provenance"]["calibration_keys"] == ["task::agent"]
    constructor_trace = payload["provenance"]["agent_traces"]["spec_constructor"]
    assert len(constructor_trace["segments"]) == 1
    assert len(constructor_trace["clusters"]) == 1
    assert load_graph_spec(output_path).node_names() == ["find"]


def test_decompose_rejects_mixed_spec_resume(tmp_path: Path) -> None:
    from tracegraph.graph_builder import graph_from_draft
    from tracegraph.ingest import parse_messages
    from tracegraph.schema import to_dict as schema_to_dict

    input_path = tmp_path / "pilot.jsonl"
    output_path = tmp_path / "graphs.jsonl"
    row = {
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
    input_path.write_text(json.dumps(row) + "\n")
    one_node = {
        "nodes": [tiny_draft()["nodes"][0]],
        "edges": [],
    }
    seeded = graph_from_draft(parse_messages(row), one_node, "agent:fake", spec=tiny_spec())
    output_path.write_text(
        json.dumps({"trace": schema_to_dict(parse_messages(row)), "graph": schema_to_dict(seeded)})
        + "\n"
    )

    with pytest.raises(SystemExit, match="spec hash"):
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


def test_edge_source_type_constraints() -> None:
    spec = tiny_spec()
    spec.edge_types[0].source_types = ["fix"]
    spec.edge_types[0].target_types = ["find"]
    errors = validate_graph_draft(fixture_trace(), tiny_draft(), spec)
    assert any("requires source types" in error for error in errors)


def test_coerce_string_list_rejoins_character_split_criteria() -> None:
    from tracegraph.schema import coerce_string_list, graph_spec_from_dict

    split = list("Skip edits; keep analysis.")
    split = [ch if ch != " " else "" for ch in split]
    assert coerce_string_list(split) == ["Skip edits", "keep analysis."]
    assert coerce_string_list("Do not count plans") == ["Do not count plans"]
    spec = graph_spec_from_dict(
        {
            "spec_id": "tiny-coding",
            "version": "1",
            "node_types": [
                {
                    "name": "find",
                    "description": "Located the site.",
                    "exclusion_criteria": split,
                }
            ],
            "edge_types": [{"name": "leads_to", "description": "enables", "is_dependency": True}],
        }
    )
    assert spec.node_types[0].exclusion_criteria == ["Skip edits", "keep analysis."]

