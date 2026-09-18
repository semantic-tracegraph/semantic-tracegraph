from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from .data import (
    append_jsonl,
    audit_datasets,
    completed_keys,
    matches_trace_models,
    output_spec_hashes,
    read_jsonl,
    sample_rows,
    select_per_model_pilot,
    select_stratified_pilot,
    trajectory_key,
    export_all_by_model,
    write_json,
    write_jsonl,
)
from .agent_decomposer import AgentDecomposer
from .agent_spec_constructor import AgentSpecConstructor
from .evaluate import annotation_agreement, evaluate_graphs, handoff_lift
from .graph_spec import DEFAULT_GRAPH_SPEC, load_graph_spec, spec_hash, spec_metadata
from .ingest import parse_messages
from .schema import graph_from_dict, to_dict, trace_from_dict
from .agent_judge import AgentJudge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tracegraph")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="Audit remote trajectory datasets")
    audit.add_argument("--output", default="artifacts/audit.json")

    sample = commands.add_parser("sample", help="Create a paired stratified pilot")
    sample.add_argument("--output", default="artifacts/pilot.jsonl")
    sample.add_argument("--total-rows", type=int, default=60)
    sample.add_argument("--per-model", type=int, help="Independent sample size per coding-agent model")
    sample.add_argument("--models", help="Comma-separated coding-agent models to sample")
    sample.add_argument("--seed", type=int, default=7)
    sample.add_argument(
        "--all",
        action="store_true",
        help="Export every trajectory in --splits for --models (no sampling)",
    )
    sample.add_argument(
        "--splits",
        default="valid,test",
        help="Comma-separated Hugging Face splits to export with --all (default: valid,test)",
    )

    construct = commands.add_parser(
        "construct-spec",
        help="Inspect a calibration set of trajectories and write a graph spec",
    )
    construct.add_argument("--input", required=True)
    construct.add_argument("--output", required=True)
    construct.add_argument("--model", help="LLM model for the spec constructor")
    construct.add_argument("--max-steps", type=int, default=24, help="Clustering step budget")
    construct.add_argument(
        "--segmentation-max-steps",
        type=int,
        default=8,
        help="Step budget for segmenting each calibration trajectory",
    )
    construct.add_argument("--sample-size", type=int, default=8)
    construct.add_argument("--seed", type=int, default=7)
    construct.add_argument(
        "--trace-model",
        action="append",
        dest="trace_models",
        help="Only calibrate on trajectories from this coding-agent model; repeatable",
    )

    decompose = commands.add_parser("decompose", help="Build accomplishment graphs")
    decompose.add_argument("--input", required=True)
    decompose.add_argument("--output", required=True)
    decompose.add_argument("--model", help="LLM model for the agent decomposer")
    decompose.add_argument("--max-steps", type=int, default=16)
    decompose.add_argument(
        "--segmentation-max-steps",
        type=int,
        default=8,
        help="Step budget for type-blind segmentation before graph assignment",
    )
    decompose.add_argument("--limit", type=int, help="Process at most this many new trajectories")
    decompose.add_argument(
        "--trace-model",
        action="append",
        dest="trace_models",
        help="Only decompose trajectories from this coding-agent model; repeatable",
    )
    decompose.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip trajectories already present in --output (default: resume)",
    )
    decompose.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log failures and keep going (default: true)",
    )
    decompose.add_argument(
        "--spec",
        help="Path to a graph spec JSON from construct-spec (default: built-in coding-agent spec)",
    )
    verify = commands.add_parser("verify", help="Verify graph evidence")
    verify.add_argument("--input", required=True)
    verify.add_argument("--output", required=True)
    verify.add_argument("--judge-model")
    verify.add_argument("--judge-max-steps", type=int, default=12)
    verify.add_argument("--limit", type=int, help="Process at most this many new graphs")
    verify.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip graphs already present in --output (default: resume)",
    )
    verify.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log failures and keep going (default: true)",
    )

    evaluate = commands.add_parser("evaluate", help="Evaluate verified graphs")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--output", required=True)

    agreement = commands.add_parser("agreement", help="Compare two annotation JSONL files")
    agreement.add_argument("--annotator-a", required=True)
    agreement.add_argument("--annotator-b", required=True)
    agreement.add_argument("--output", required=True)

    template = commands.add_parser("annotation-template", help="Create a blinded annotation packet")
    template.add_argument("--input", required=True)
    template.add_argument("--output", required=True)
    template.add_argument("--limit", type=int, default=30)

    handoff = commands.add_parser("handoff", help="Summarize handoff trial records")
    handoff.add_argument("--input", required=True)
    handoff.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        write_json(audit_datasets(), args.output)
    elif args.command == "sample":
        models = tuple(item.strip() for item in args.models.split(",")) if args.models else None
        if args.all:
            split_names = tuple(item.strip() for item in args.splits.split(",") if item.strip())
            grouped = export_all_by_model(
                models or ("claude-opus-4.7", "gemini-3-pro"),
                splits=split_names,
            )
            out = Path(args.output)
            for model, rows in grouped.items():
                path = out.with_name(f"{out.stem}_{model}{out.suffix}")
                write_jsonl(rows, path)
                print(f"wrote {len(rows)} rows -> {path}", file=sys.stderr)
        elif args.per_model:
            rows = select_per_model_pilot(
                args.per_model,
                args.seed,
                models or ("claude-opus-4.7", "gpt-5-mini"),
            )
            write_jsonl(rows, args.output)
        elif models:
            rows = select_stratified_pilot(args.total_rows, args.seed, models)
            write_jsonl(rows, args.output)
        else:
            rows = select_stratified_pilot(args.total_rows, args.seed)
            write_jsonl(rows, args.output)
    elif args.command == "construct-spec":
        _construct_spec(
            args.input,
            args.output,
            args.model,
            args.max_steps,
            segmentation_max_steps=args.segmentation_max_steps,
            sample_size=args.sample_size,
            seed=args.seed,
            trace_models=set(args.trace_models or []),
        )
    elif args.command == "decompose":
        _decompose(
            args.input,
            args.output,
            args.model,
            args.max_steps,
            segmentation_max_steps=args.segmentation_max_steps,
            limit=args.limit,
            trace_models=set(args.trace_models or []),
            resume=args.resume,
            continue_on_error=args.continue_on_error,
            spec_path=args.spec,
        )
    elif args.command == "verify":
        _verify(
            args.input,
            args.output,
            args.judge_model,
            args.judge_max_steps,
            limit=args.limit,
            resume=args.resume,
            continue_on_error=args.continue_on_error,
        )
    elif args.command == "evaluate":
        records = read_jsonl(args.input)
        graphs = [graph_from_dict(record.get("graph", record)) for record in records]
        write_json(evaluate_graphs(graphs), args.output)
    elif args.command == "agreement":
        write_json(
            annotation_agreement(read_jsonl(args.annotator_a), read_jsonl(args.annotator_b)),
            args.output,
        )
    elif args.command == "annotation-template":
        _annotation_template(args.input, args.output, args.limit)
    elif args.command == "handoff":
        write_json(handoff_lift(read_jsonl(args.input)), args.output)


def _construct_spec(
    input_path: str,
    output_path: str,
    model: str | None,
    max_steps: int,
    segmentation_max_steps: int = 8,
    sample_size: int = 8,
    seed: int = 7,
    trace_models: set[str] | None = None,
) -> None:
    rows = [row for row in read_jsonl(input_path) if matches_trace_models(row, trace_models)]
    if not rows:
        raise SystemExit("construct-spec found no matching trajectories in --input.")
    selected = sample_rows(rows, sample_size, seed)
    traces = [parse_messages(row) for row in selected]
    spec, agent_trace = AgentSpecConstructor(
        model=model,
        max_steps=max_steps,
        segmentation_max_steps=segmentation_max_steps,
    ).construct(traces)
    payload = {
        **spec_metadata(spec),
        "spec": to_dict(spec),
        "provenance": {
            "input": input_path,
            "sample_size": len(traces),
            "seed": seed,
            "calibration_keys": [
                f"{trace.instance_id}::{trace.model}" for trace in traces
            ],
            "constructor": f"agent:{agent_trace['model']}",
            "agent_traces": {"spec_constructor": agent_trace},
        },
    }
    write_json(payload, output_path)
    print(
        f"wrote spec {spec.spec_id} v{spec.version} "
        f"({len(spec.node_types)} node types, {len(spec.edge_types)} edge types) -> {output_path}",
        file=sys.stderr,
    )


def _decompose(
    input_path: str,
    output_path: str,
    model: str | None,
    max_steps: int,
    segmentation_max_steps: int = 8,
    limit: int | None = None,
    trace_models: set[str] | None = None,
    resume: bool = True,
    continue_on_error: bool = True,
    spec_path: str | None = None,
) -> None:
    spec = load_graph_spec(spec_path) if spec_path else DEFAULT_GRAPH_SPEC
    current_hash = spec_hash(spec)
    if resume:
        existing = output_spec_hashes(output_path)
        if existing and existing != {current_hash}:
            raise SystemExit(
                f"Output {output_path} contains graphs built with spec hash "
                f"{sorted(existing)}, but this run uses {current_hash}. "
                "Use a new --output or --no-resume."
            )
    decomposer = AgentDecomposer(
        model=model,
        max_steps=max_steps,
        segmentation_max_steps=segmentation_max_steps,
        graph_spec=spec,
    )
    done = completed_keys(output_path) if resume else set()
    if not resume:
        Path(output_path).unlink(missing_ok=True)
    processed = 0
    skipped = 0
    failures_path = f"{output_path}.failures.jsonl"
    for row in read_jsonl(input_path):
        if not matches_trace_models(row, trace_models):
            continue
        key = trajectory_key(row)
        if key in done:
            skipped += 1
            continue
        if limit is not None and processed >= limit:
            break
        try:
            trace = parse_messages(row)
            graph = decomposer.decompose(trace)
            append_jsonl({"trace": to_dict(trace), "graph": to_dict(graph)}, output_path)
            done.add(key)
            processed += 1
            print(f"decomposed {key} ({processed} new, {skipped} skipped)", flush=True)
        except Exception as exc:
            print(f"FAILED {key}: {exc}", flush=True)
            append_jsonl(
                {
                    "key": key,
                    "error": str(exc),
                    "instance_id": row.get("instance_id"),
                    "traceback": traceback.format_exc(),
                },
                failures_path,
            )
            if not continue_on_error:
                raise
    print(f"decompose done: {processed} new, {skipped} already in {output_path}", flush=True)


def _verify(
    input_path: str,
    output_path: str,
    judge_model: str | None = None,
    judge_max_steps: int = 12,
    limit: int | None = None,
    resume: bool = True,
    continue_on_error: bool = True,
) -> None:
    judge = AgentJudge(model=judge_model, max_steps=judge_max_steps)
    done = completed_keys(output_path) if resume else set()
    if not resume:
        Path(output_path).unlink(missing_ok=True)
    processed = 0
    skipped = 0
    failures_path = f"{output_path}.failures.jsonl"
    for record in read_jsonl(input_path):
        key = trajectory_key(record)
        if key in done:
            skipped += 1
            continue
        if limit is not None and processed >= limit:
            break
        try:
            trace = trace_from_dict(record["trace"])
            graph = graph_from_dict(record["graph"])
            graph.metadata["event_count"] = len(trace.events)
            append_jsonl(
                {"trace": to_dict(trace), "graph": to_dict(judge.verify(graph, trace))},
                output_path,
            )
            done.add(key)
            processed += 1
            print(f"verified {key} ({processed} new, {skipped} skipped)", flush=True)
        except Exception as exc:
            print(f"FAILED {key}: {exc}", flush=True)
            append_jsonl(
                {"key": key, "error": str(exc), "traceback": traceback.format_exc()},
                failures_path,
            )
            if not continue_on_error:
                raise
    print(f"verify done: {processed} new, {skipped} already in {output_path}", flush=True)


def _annotation_template(input_path: str, output_path: str, limit: int) -> None:
    packet: list[dict[str, Any]] = []
    for record in read_jsonl(input_path)[:limit]:
        trace = trace_from_dict(record["trace"]) if "trace" in record else parse_messages(record)
        packet.append(
            {
                "instance_id": trace.instance_id,
                "problem_statement": trace.problem_statement,
                "events": [
                    {
                        "event_id": event.event_id,
                        "turn": event.turn,
                        "type": event.type.value,
                        "content": event.content[:4000],
                        "content_truncated": len(event.content) > 4000,
                        "returncode": event.returncode,
                        "paths": event.paths,
                    }
                    for event in trace.events
                ],
                "annotations": [],
            }
        )
    write_jsonl(packet, output_path)


if __name__ == "__main__":
    main()
