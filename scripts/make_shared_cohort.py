#!/usr/bin/env python3
"""Build per-model JSONL files for a shared instance_id cohort.

The ID order is prefix-stable: the first 20 of a 50-ID list stay the first 20
when you later raise --n to 50. Resume decompose/verify on the same output
files to process only the new rows.

Example:
  python scripts/make_shared_cohort.py --n 20 --pool 50 \\
      --input-dir artifacts/old --output-dir artifacts/compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = ("claude-opus-4.7", "gemini-3-pro", "gpt-5-mini", "deepseek-v3.2")


def _load_by_id(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["instance_id"]] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="How many shared IDs to write now")
    parser.add_argument("--pool", type=int, default=50, help="Prefix-stable ID list length")
    parser.add_argument("--input-dir", default="artifacts/old")
    parser.add_argument("--output-dir", default="artifacts/compare")
    parser.add_argument("--prefix", default="test")
    args = parser.parse_args()
    if args.n < 1 or args.pool < args.n:
        raise SystemExit("--pool must be >= --n >= 1")

    input_dir = Path(args.input_dir)
    by_model = {model: _load_by_id(input_dir / f"{args.prefix}_{model}.jsonl") for model in MODELS}
    common = set.intersection(*(set(rows) for rows in by_model.values()))
    common = {
        iid
        for iid in common
        if len({by_model[model][iid].get("problem_statement") for model in MODELS}) == 1
    }
    order = [row["instance_id"] for row in _load_by_id(input_dir / f"{args.prefix}_{MODELS[0]}.jsonl").values()]
    ranked = [iid for iid in order if iid in common][: args.pool]
    if len(ranked) < args.n:
        raise SystemExit(f"Only {len(ranked)} shared IDs available, need {args.n}")

    selected = ranked[: args.n]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids_path = out / "ids.jsonl"
    with ids_path.open("w") as handle:
        for rank, iid in enumerate(ranked, start=1):
            handle.write(
                json.dumps(
                    {
                        "rank": rank,
                        "instance_id": iid,
                        "in_current_run": rank <= args.n,
                        "problem_statement": by_model[MODELS[0]][iid]["problem_statement"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    for model in MODELS:
        path = out / f"test_{model}.jsonl"
        with path.open("w") as handle:
            for iid in selected:
                handle.write(json.dumps(by_model[model][iid], ensure_ascii=False) + "\n")
        print(f"wrote {len(selected)} -> {path}")
    print(f"wrote {len(ranked)} ranked IDs ({args.n} selected) -> {ids_path}")


if __name__ == "__main__":
    main()
