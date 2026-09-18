from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

DATASETS = {
    "claude-opus-4.7": "SWE-Router/v4-4k-traj-claude-opus-4.7",
    "gemini-3-pro": "SWE-Router/v4-4k-traj-gemini-3-pro",
    "deepseek-v3.2": "SWE-Router/v4-4k-traj-deepseek-v3.2",
    "gpt-5-mini": "SWE-Router/v4-4k-traj-gpt-5-mini",
}

EXPECTED_SPLITS = {
    "claude-opus-4.7": ("valid", "test"),
    "gemini-3-pro": ("valid", "test"),
    "deepseek-v3.2": ("train", "valid", "test"),
    "gpt-5-mini": ("train", "train_1", "train_2", "valid", "valid_1", "valid_2", "test"),
}


def _iter_split(dataset_name: str, split: str) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    return load_dataset(dataset_name, split=split, streaming=True)


def audit_datasets(
    datasets: dict[str, str] | None = None,
    splits: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    datasets = datasets or DATASETS
    splits = splits or EXPECTED_SPLITS
    report: dict[str, Any] = {"datasets": {}, "overlap": {}}
    ids_by_model_split: dict[tuple[str, str], set[str]] = {}

    for model, dataset_name in datasets.items():
        model_report: dict[str, Any] = {"dataset": dataset_name, "splits": {}}
        for split in splits[model]:
            ids: set[str] = set()
            resolved: Counter[bool] = Counter()
            message_lengths: list[int] = []
            api_calls: list[int] = []
            costs: list[float] = []
            fields: set[str] = set()
            for row in _iter_split(dataset_name, split):
                fields.update(row)
                ids.add(str(row["instance_id"]))
                if row.get("resolved") is not None:
                    resolved[bool(row["resolved"])] += 1
                message_lengths.append(len(row.get("messages") or []))
                if row.get("api_calls") is not None:
                    api_calls.append(int(row["api_calls"]))
                if row.get("instance_cost") is not None:
                    costs.append(float(row["instance_cost"]))
            ids_by_model_split[(model, split)] = ids
            model_report["splits"][split] = {
                "rows": len(message_lengths),
                "unique_instance_ids": len(ids),
                "duplicate_instance_ids": len(message_lengths) - len(ids),
                "resolved": dict(resolved),
                "missing_resolved": len(message_lengths) - sum(resolved.values()),
                "resolved_rate": resolved[True] / len(message_lengths) if message_lengths else None,
                "messages": _summary(message_lengths),
                "api_calls": _summary(api_calls),
                "instance_cost": _summary(costs),
                "fields": sorted(fields),
            }
        report["datasets"][model] = model_report

    shared_models = ("claude-opus-4.7", "gemini-3-pro", "deepseek-v3.2")
    for split in ("valid", "test"):
        sets = [ids_by_model_split.get((model, split), set()) for model in shared_models]
        common = set.intersection(*sets) if all(sets) else set()
        report["overlap"][split] = {
            "models": list(shared_models),
            "intersection": len(common),
            "union": len(set.union(*sets)) if sets else 0,
            "pairwise": {
                f"{a}::{b}": len(
                    ids_by_model_split.get((a, split), set())
                    & ids_by_model_split.get((b, split), set())
                )
                for index, a in enumerate(shared_models)
                for b in shared_models[index + 1 :]
            },
        }
    return report


def _summary(values: list[int] | list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
    }


def select_stratified_pilot(
    total_rows: int = 60,
    seed: int = 7,
    models: tuple[str, ...] = ("claude-opus-4.7", "gemini-3-pro", "deepseek-v3.2"),
    splits: tuple[str, ...] = ("valid", "test"),
) -> list[dict[str, Any]]:
    """Select solved/unsolved and short/long traces, retaining cross-model pairs."""
    randomizer = random.Random(seed)
    rows_by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model in models:
        for split in splits:
            for row in _iter_split(DATASETS[model], split):
                record = dict(row)
                record["_model_key"] = model
                record["_split"] = split
                rows_by_instance[str(record["instance_id"])].append(record)

    paired = [rows for rows in rows_by_instance.values() if len({r["_model_key"] for r in rows}) == len(models)]
    buckets: dict[tuple[bool, str], list[list[dict[str, Any]]]] = defaultdict(list)
    for rows in paired:
        median_calls = statistics.median(int(r.get("api_calls") or 0) for r in rows)
        any_resolved = any(bool(r.get("resolved")) for r in rows)
        buckets[(any_resolved, "long" if median_calls >= 12 else "short")].append(rows)

    target_groups = max(1, total_rows // len(models))
    selected_groups: list[list[dict[str, Any]]] = []
    for key in ((True, "short"), (True, "long"), (False, "short"), (False, "long")):
        candidates = buckets[key]
        randomizer.shuffle(candidates)
        selected_groups.extend(candidates[: max(1, target_groups // 4)])

    if len(selected_groups) < target_groups:
        used = {rows[0]["instance_id"] for rows in selected_groups}
        remainder = [rows for rows in paired if rows[0]["instance_id"] not in used]
        randomizer.shuffle(remainder)
        selected_groups.extend(remainder[: target_groups - len(selected_groups)])
    return [_compact_record(row) for rows in selected_groups[:target_groups] for row in rows]


def select_per_model_pilot(
    per_model: int = 20,
    seed: int = 7,
    models: tuple[str, ...] = ("claude-opus-4.7", "gpt-5-mini"),
    splits: tuple[str, ...] = ("valid", "test"),
) -> list[dict[str, Any]]:
    """Sample independently per coding-agent model (no cross-model pairing)."""
    randomizer = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for model in models:
        if model not in DATASETS:
            raise ValueError(f"Unknown model {model}. Expected one of {sorted(DATASETS)}")
        available_splits = [split for split in splits if split in EXPECTED_SPLITS[model]]
        rows: list[dict[str, Any]] = []
        for split in available_splits:
            for row in _iter_split(DATASETS[model], split):
                record = dict(row)
                record["_model_key"] = model
                record["_split"] = split
                rows.append(record)
        buckets: dict[tuple[bool | None, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            resolved = row.get("resolved")
            length = "long" if int(row.get("api_calls") or 0) >= 12 else "short"
            buckets[(None if resolved is None else bool(resolved), length)].append(row)
        take: list[dict[str, Any]] = []
        per_bucket = max(1, per_model // 4)
        for key in (
            (True, "short"),
            (True, "long"),
            (False, "short"),
            (False, "long"),
            (None, "short"),
            (None, "long"),
        ):
            candidates = list(buckets[key])
            randomizer.shuffle(candidates)
            take.extend(candidates[:per_bucket])
        if len(take) < per_model:
            used = {(row["instance_id"], row["_model_key"]) for row in take}
            remainder = [
                row for row in rows if (row["instance_id"], row["_model_key"]) not in used
            ]
            randomizer.shuffle(remainder)
            take.extend(remainder[: per_model - len(take)])
        selected.extend(_compact_record(row) for row in take[:per_model])
    return selected


def export_all_rows(
    models: tuple[str, ...],
    splits: tuple[str, ...] = ("valid", "test"),
) -> list[dict[str, Any]]:
    """Write every trajectory in the requested splits, with no sampling."""
    selected: list[dict[str, Any]] = []
    for rows in export_all_by_model(models, splits).values():
        selected.extend(rows)
    return selected


def export_all_by_model(
    models: tuple[str, ...],
    splits: tuple[str, ...] = ("valid", "test"),
) -> dict[str, list[dict[str, Any]]]:
    """Same as export_all_rows, grouped so each coding agent can be written to its own file."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        if model not in DATASETS:
            raise ValueError(f"Unknown model {model}. Expected one of {sorted(DATASETS)}")
        rows: list[dict[str, Any]] = []
        available_splits = [split for split in splits if split in EXPECTED_SPLITS[model]]
        for split in available_splits:
            for row in _iter_split(DATASETS[model], split):
                record = dict(row)
                record["_model_key"] = model
                record["_split"] = split
                rows.append(_compact_record(record))
        grouped[model] = rows
    return grouped


def trajectory_key(record: dict[str, Any]) -> str:
    """Stable id for a coding-agent trajectory, used to resume batch jobs."""
    if isinstance(record.get("trace"), dict):
        trace = record["trace"]
        return f"{trace.get('instance_id')}::{trace.get('model')}"
    return f"{record.get('instance_id')}::{record.get('model') or record.get('_model_key')}"


def matches_trace_models(record: dict[str, Any], models: set[str] | None) -> bool:
    if not models:
        return True
    if isinstance(record.get("trace"), dict):
        value = record["trace"].get("model")
    else:
        value = record.get("model") or record.get("_model_key")
    return str(value) in models


def completed_keys(path: str | Path) -> set[str]:
    target = Path(path)
    if not target.exists():
        return set()
    keys: set[str] = set()
    with target.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            keys.add(trajectory_key(json.loads(line)))
    return keys


def sample_rows(
    rows: list[dict[str, Any]],
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Take a seeded subset, or all rows when the requested size is larger."""
    if size < 1:
        raise ValueError("sample size must be at least 1")
    if size >= len(rows):
        return list(rows)
    return random.Random(seed).sample(rows, size)


def output_spec_hashes(path: str | Path) -> set[str]:
    """Spec hashes already written to a graph JSONL file; missing hashes count as default."""
    from .graph_spec import DEFAULT_GRAPH_SPEC, spec_hash

    target = Path(path)
    if not target.exists():
        return set()
    default = spec_hash(DEFAULT_GRAPH_SPEC)
    hashes: set[str] = set()
    with target.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            graph = record.get("graph") if isinstance(record.get("graph"), dict) else record
            metadata = (graph or {}).get("metadata") or {}
            hashes.add(str(metadata.get("graph_spec_hash") or default))
    return hashes


def append_jsonl(row: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _compact_record(row: dict[str, Any], max_message_chars: int = 8000) -> dict[str, Any]:
    """Bound cached pilot size while preserving the beginning and end of long tool output."""
    record = dict(row)
    compact_messages = []
    for message in row.get("messages") or []:
        content = str(message.get("content") or "")
        if len(content) > max_message_chars:
            half = max_message_chars // 2
            content = content[:half] + "\n...[TRACEGRAPH TRUNCATED]...\n" + content[-half:]
        compact_messages.append({**message, "content": content})
    record["messages"] = compact_messages
    return record


def write_json(value: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]
