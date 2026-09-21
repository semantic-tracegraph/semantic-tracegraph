import json
import math
from pathlib import Path

from tracegraph.cli import main
from tracegraph.graph_builder import graph_from_draft
from tracegraph.schema import GraphEdge, to_dict
from tracegraph.visualize import (
    derive_primary_parents,
    graph_visualization_data,
    render_visualization_html,
)

from test_graph import fixture_graph_draft, fixture_trace


def test_primary_parent_prefers_edge_type_then_nearest_source() -> None:
    trace = fixture_trace()
    graph = graph_from_draft(trace, fixture_graph_draft(), "agent:test")
    graph.edges.extend(
        [
            GraphEdge(source="n000", target="n002", type="requires"),
            GraphEdge(source="n000", target="n003", type="refines"),
            GraphEdge(source="n001", target="n003", type="produces"),
        ]
    )

    parents = derive_primary_parents(graph, trace)

    assert parents["n000"] is None
    assert parents["n002"] == "n001"
    assert parents["n003"] == "n002"


def test_visualization_data_keeps_unselected_edges_as_secondary() -> None:
    trace = fixture_trace()
    graph = graph_from_draft(trace, fixture_graph_draft(), "agent:test")
    graph.edges.append(GraphEdge(source="n000", target="n002", type="requires"))

    payload = graph_visualization_data(graph, trace)

    node = next(item for item in payload["nodes"] if item["node_id"] == "n002")
    assert node["primary_parent_id"] == "n001"
    assert payload["secondary_edges"] == [
        {"source": "n000", "target": "n002", "type": "requires"}
    ]
    assert payload["problem_statement"] == trace.problem_statement
    assert payload["weight_unit"] == "estimated_tokens"
    expected = sum(math.ceil(len(event.content or "") / 4) for event in trace.events[7:9])
    assert node["own_weight"] == expected


def test_html_is_self_contained_and_escapes_script_end() -> None:
    trace = fixture_trace()
    graph = graph_from_draft(trace, fixture_graph_draft(), "agent:test")
    payload = graph_visualization_data(graph, trace)
    payload["problem_statement"] = "Do not allow </script> to end the payload"

    html = render_visualization_html([payload])

    assert "__TRACEGRAPH_DATA__" not in html
    assert "<\\/script>" in html
    assert "<script src=" not in html
    assert "<link rel=\"stylesheet\"" not in html
    assert "TraceGraph · Task decomposition" in html


def test_visualize_cli_writes_selected_graph(tmp_path: Path) -> None:
    trace = fixture_trace()
    graph = graph_from_draft(trace, fixture_graph_draft(), "agent:test")
    input_path = tmp_path / "graphs.jsonl"
    output_path = tmp_path / "flamegraph.html"
    input_path.write_text(
        json.dumps({"trace": to_dict(trace), "graph": to_dict(graph)}) + "\n",
        encoding="utf-8",
    )

    main(
        [
            "visualize",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--instance-id",
            trace.instance_id,
        ]
    )

    html = output_path.read_text(encoding="utf-8")
    assert trace.instance_id in html
    assert "Patched message.py" in html
