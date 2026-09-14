# Semantic TraceGraph

**Authors:** [Xinyi Wang](https://github.com/cindyxinyiwang) and [Neil Xu](https://github.com/neilzxu)  
**Project page:** https://semantic-tracegraph.github.io/semantic-tracegraph/

A research prototype for turning coding-agent transcripts into evidence-grounded
semantic accomplishment graphs. It separates three questions that are often
collapsed into a single score:

1. What did the agent claim to accomplish?
2. What does the trajectory actually evidence?
3. Which claims have mechanical proof (tests, a patch, a reproduction)?

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
tracegraph audit --output artifacts/audit.json
tracegraph sample --total-rows 60 --output artifacts/pilot.jsonl
tracegraph sample --per-model 20 --models claude-opus-4.7,gpt-5-mini --output artifacts/pilot_claude_gpt20.jsonl
tracegraph decompose --input artifacts/pilot_claude_gpt20.jsonl --output artifacts/graphs_claude_gpt.jsonl --model glm-5.3-flash
# Later, continue the same output file (already-finished trajectories are skipped):
tracegraph decompose --input artifacts/pilot_claude_gpt.jsonl --output artifacts/graphs_claude_gpt.jsonl --model glm-5.3-flash
tracegraph verify --input artifacts/graphs.jsonl --output artifacts/verified.jsonl --judge-model gpt-4.1-mini
tracegraph evaluate --input artifacts/verified.jsonl --output artifacts/evaluation.json
tracegraph annotation-template --input artifacts/graphs.jsonl --output annotations/pilot.jsonl
python scripts/print_graphs.py artifacts/verified.jsonl --limit 1
pytest
```

Set `TRACEGRAPH_API_KEY` and optionally `TRACEGRAPH_MODEL` (default `gpt-4.1-mini`).

## Decomposer and judge agents

Both stages are tool-using LLM loops, not a single completion. The model only sees a
trace summary plus tool results. Invalid submits are returned as errors so the agent
can retry inside a step budget (16 for decompose, 12 per node for judge).

**Decomposer** (`AgentDecomposer`) builds the graph. It is prompted to emit a few coarse,
non-overlapping accomplishments (localization, diagnosis, reproduction, implementation,
verification, handoff), each citing real event IDs and excerpts, then a causal DAG of
`requires` / `produces` / `refines` edges. Independent roots are allowed; placeholder
claims and string-encoded `nodes`/`edges` are rejected.

Tools:

- `run_python` — execute read-only Python against the loaded `trace` sandbox:
  `trace.problem()`, `trace.summary()`, `trace.events()`, `trace.event(id)`,
  `trace.turn(n)`, `trace.grep(pattern)`, `trace.span(start, end)`,
  `trace.uncovered(spans)` (coverage of uncited events).
- `submit_graph` — submit `nodes` and `edges` as JSON arrays. Accepted only after
  `validate_graph_draft` (real event IDs, evidence excerpts, no placeholders, causal
  edges when there is more than one node).

**Judge** (`AgentJudge`) does not rewrite the graph. It scores one node at a time
against the same sandbox. Graph neighborhood is supporting context, not evidence.
It must re-check cited events and look for later reverts.

Tools:

- `inspect_graph` — omit `node_id` for the full node/edge/root list; pass `node_id`
  for that node's parents and children.
- `run_python` — same sandbox as the decomposer (`trace.problem()`, `trace.event`,
  `trace.span`, `trace.grep`, …). The judge uses it to re-read cited events and
  later context, not to cover the whole trace.
- `submit_verdict` — `status` (`claimed` / `evidenced` / `verified`), `confidence`
  in `[0, 1]`, and a reason citing event IDs. `claimed` = weak, missing, or undone;
  `evidenced` = the work happened; `verified` = mechanical proof. Graph progress
  uses verified = 1, claimed/evidenced = 0.

## Research guardrails

- Gold patches and hidden tests are evaluation-only inputs.
- A successful command is evidence of execution, not evidence that its semantic
claim is correct.
- Raw node counts are not a progress metric. The evaluator deduplicates semantic
nodes and penalizes invalidation and regressions.
- Report results grouped by task ID so multiple model trajectories for one task
are not treated as independent samples.

