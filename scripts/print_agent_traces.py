#!/usr/bin/env python3
"""Pretty-print decomposer / judge agent chat traces from a verified JSONL file.

Usage:
  python scripts/print_agent_traces.py artifacts/verified_test_claude-opus-4.7.jsonl
  python scripts/print_agent_traces.py artifacts/verified_test_claude-opus-4.7.jsonl \\
      --instance-id pallets__click.fde47b4b.lm_rewrite__zjr9zw0w --judge-node n1
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any


def _truncate(text: str, width: int = 500) -> str:
    text = (text or "").replace("\r\n", "\n")
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def _format_tool_args(arguments: str, width: int = 600) -> str:
    try:
        parsed = json.loads(arguments or "{}")
        formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        formatted = arguments or ""
    return textwrap.indent(_truncate(formatted, width), "      ")


def _print_message(msg: dict[str, Any], *, step: int | None = None) -> None:
    role = msg.get("role", "?")
    prefix = f"[step {step}] " if step is not None else ""
    print(f"{prefix}{role.upper()}")
    content = msg.get("content")
    if content:
        print(textwrap.indent(_truncate(str(content), 800), "  "))
    for tool_call in msg.get("tool_calls") or []:
        fn = tool_call.get("function") or {}
        name = fn.get("name", "?")
        print(f"  -> tool_call {name}")
        print(_format_tool_args(str(fn.get("arguments") or "")))
    print()


def _print_decomposer_trace(trace: dict[str, Any]) -> None:
    print("=" * 88)
    print("DECOMPOSER AGENT TRACE")
    print(
        f"model={trace.get('model')}  steps={trace.get('steps')}/{trace.get('max_steps')}  "
        f"session={trace.get('session_id')}"
    )
    print("=" * 88)
    messages = trace.get("messages") or []
    step = 0
    for msg in messages:
        if msg.get("role") == "assistant":
            step += 1
            _print_message(msg, step=step)
        elif msg.get("role") == "tool":
            print(f"[step {step}] TOOL RESULT ({msg.get('tool_call_id', '')})")
            print(textwrap.indent(_truncate(str(msg.get("content") or ""), 800), "  "))
            print()
        elif msg.get("role") in {"system", "user"} and step == 0:
            _print_message(msg)


def _print_judge_node_trace(node_id: str, trace: dict[str, Any]) -> None:
    print("=" * 88)
    print(f"JUDGE AGENT TRACE — node {node_id}")
    print(
        f"model={trace.get('model')}  steps={trace.get('steps')}/{trace.get('max_steps')}  "
        f"session={trace.get('session_id')}"
    )
    print("=" * 88)
    messages = trace.get("messages") or []
    step = 0
    for msg in messages:
        if msg.get("role") == "assistant":
            step += 1
            _print_message(msg, step=step)
        elif msg.get("role") == "tool":
            print(f"[step {step}] TOOL RESULT ({msg.get('tool_call_id', '')})")
            print(textwrap.indent(_truncate(str(msg.get("content") or ""), 800), "  "))
            print()
        elif msg.get("role") in {"system", "user"} and step == 0:
            _print_message(msg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print decomposer and judge agent traces")
    parser.add_argument("input", help="Verified JSONL from tracegraph verify")
    parser.add_argument("--instance-id", help="Pick this trajectory")
    parser.add_argument("--model", help="Pick this coding-agent model")
    parser.add_argument("--judge-node", help="Print only this judge node trace (e.g. n1)")
    parser.add_argument("--all-judge-nodes", action="store_true", help="Print every judge node trace")
    args = parser.parse_args(argv)

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    for line in path.open():
        if not line.strip():
            continue
        record = json.loads(line)
        graph = record.get("graph", record)
        if args.instance_id and graph.get("instance_id") != args.instance_id:
            continue
        if args.model and graph.get("model") != args.model:
            continue

        traces = (graph.get("metadata") or {}).get("agent_traces") or {}
        decomposer = traces.get("decomposer")
        judge = traces.get("judge")
        if not decomposer and not judge:
            raise SystemExit("No agent_traces found in graph metadata.")

        print()
        print("#" * 88)
        print(f"INSTANCE {graph.get('instance_id')}  model={graph.get('model')}  resolved={graph.get('resolved')}")
        print("#" * 88)
        print()

        if decomposer:
            _print_decomposer_trace(decomposer)
        else:
            print("(no decomposer trace in metadata)")

        if judge:
            nodes = judge.get("nodes") or {}
            if args.judge_node:
                if args.judge_node not in nodes:
                    raise SystemExit(f"Judge trace for node {args.judge_node} not found.")
                _print_judge_node_trace(args.judge_node, nodes[args.judge_node])
            elif args.all_judge_nodes:
                for node_id in sorted(nodes):
                    _print_judge_node_trace(node_id, nodes[node_id])
            else:
                first = sorted(nodes)[0]
                _print_judge_node_trace(first, nodes[first])
                if len(nodes) > 1:
                    print(f"(judge also ran for {len(nodes) - 1} more nodes; use --all-judge-nodes to print all)")
        else:
            print("(no judge trace in metadata)")
        return

    raise SystemExit("No matching record found.")


if __name__ == "__main__":
    main()
