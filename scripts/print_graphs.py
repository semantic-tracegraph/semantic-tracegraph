#!/usr/bin/env python3
"""Pretty-print decomposed accomplishment graphs from a JSONL file.

Usage:
  python scripts/print_graphs.py artifacts/verified.jsonl
  python scripts/print_graphs.py artifacts/graphs.jsonl --limit 3
  python scripts/print_graphs.py artifacts/verified.jsonl --instance-id jd__tenacity.0d40e76f.combine_file__8pa1fxvj
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_records(path: Path) -> list[dict]:
    records = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            graph = record.get("graph", record)
            if "nodes" not in graph:
                raise SystemExit(f"{path}:{line_no} is missing a graph.nodes field")
            records.append(record)
    return records


def _truncate(text: str, width: int = 140) -> str:
    text = " ".join((text or "").split())
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def _print_graph(graph: dict, *, show_evidence: bool) -> None:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    metadata = graph.get("metadata") or {}
    print("=" * 88)
    print(f"{graph.get('instance_id')}  |  {graph.get('model')}  |  resolved={graph.get('resolved')}")
    extras = []
    if metadata.get("decomposer"):
        extras.append(f"decomposer={metadata['decomposer']}")
    if metadata.get("verifier"):
        extras.append(f"verifier={metadata['verifier']}")
    extras.append(f"nodes={len(nodes)}")
    extras.append(f"edges={len(edges)}")
    if metadata.get("event_count") is not None:
        extras.append(f"events={metadata['event_count']}")
    traces = metadata.get("agent_traces") or {}
    decomposer_trace = traces.get("decomposer") or {}
    if decomposer_trace.get("steps") is not None:
        extras.append(f"decompose_steps={decomposer_trace['steps']}")
    elif metadata.get("agent_steps") is not None:
        extras.append(f"decompose_steps={metadata['agent_steps']}")
    judge_trace = traces.get("judge") or {}
    if judge_trace.get("total_steps") is not None:
        extras.append(f"judge_steps={judge_trace['total_steps']}")
    elif metadata.get("judge_steps") is not None:
        extras.append(f"judge_steps={metadata['judge_steps']}")
    print("  " + "  ".join(extras))
    print()
    print(f"{'id':<6}{'type':<18}{'status':<14}claim")
    print("-" * 88)
    for node in nodes:
        print(
            f"{node.get('node_id', ''):<6}"
            f"{node.get('type', ''):<18}"
            f"{node.get('status', ''):<14}"
            f"{_truncate(node.get('claim', ''))}"
        )
        print(f"      span={tuple(node.get('event_span') or ())}  key={node.get('semantic_key', '')}")
        if show_evidence:
            for evidence in node.get("evidence") or []:
                print(
                    f"      evidence {evidence.get('event_id')} [{evidence.get('kind')}]: "
                    f"{_truncate(evidence.get('excerpt', ''), 110)}"
                )
        reason = (node.get("artifact_delta") or {}).get("verification_reason")
        if reason:
            print(f"      verify: {reason}")
        print()
    if edges:
        print("edges")
        print("-" * 88)
        for edge in edges:
            print(f"  {edge.get('source')} --{edge.get('type')}--> {edge.get('target')}")
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pretty-print decomposed accomplishment graphs")
    parser.add_argument("input", help="JSONL from `tracegraph decompose` or `tracegraph verify`")
    parser.add_argument("--instance-id", help="Print only this instance_id")
    parser.add_argument("--model", help="Print only this model")
    parser.add_argument("--limit", type=int, default=0, help="Max graphs to print (0 = all matches)")
    parser.add_argument("--no-evidence", action="store_true", help="Hide evidence excerpts")
    args = parser.parse_args(argv)

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    printed = 0
    for record in _load_records(path):
        graph = record.get("graph", record)
        if args.instance_id and graph.get("instance_id") != args.instance_id:
            continue
        if args.model and graph.get("model") != args.model:
            continue
        _print_graph(graph, show_evidence=not args.no_evidence)
        printed += 1
        if args.limit and printed >= args.limit:
            break
    if printed == 0:
        print("No matching graphs.", file=sys.stderr)
        raise SystemExit(1)
    print(f"Printed {printed} graph(s).")


if __name__ == "__main__":
    main()
