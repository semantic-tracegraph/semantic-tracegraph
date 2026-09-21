# Semantic TraceGraph

**Authors:** [Xinyi Wang](https://github.com/cindyxinyiwang) and [Neil Xu](https://github.com/neilzxu)  
**Project page:** https://semantic-tracegraph.github.io/semantic-tracegraph/

A research prototype for turning coding-agent transcripts into evidence-grounded
semantic accomplishment graphs. It separates three questions that are often
collapsed into a single score:

1. What did the agent claim to accomplish?
2. What does the trajectory actually evidence?
3. Which claims have mechanical proof (tests, a patch, a reproduction)?

## Architecture

The pipeline separates boundary discovery from semantic classification:

```text
Calibration trajectories
  -> shared type-blind segmenter
  -> cluster segments
  -> GraphSpec

New trajectory
  -> shared type-blind segmenter
  -> assign/merge segments using GraphSpec
  -> accomplishment graph
  -> node-by-node evidence judge
  -> evaluation
```

All agent stages are multi-turn tool loops. A stage can inspect only the trace
summary and data returned by its tools. Structured submissions are validated and
returned to the agent with errors when correction is needed.

### 1. Shared type-blind segmentation

`agent_segmenter.py` defines the common `WorkSegment` representation and
`segment_trace` loop used by both spec construction and decomposition. The
segmenter sees no candidate node types. It finds bottom-up subtask boundaries and
submits ordered, non-overlapping spans with:

- an objective, action summary, and outcome summary;
- real start/end event IDs and evidence event IDs;
- affected artifact paths when present.

Validation rejects unknown or reversed event IDs, overlapping segments,
out-of-span evidence, and uncovered edit, test, or submission events. Keeping
segmentation ontology-independent prevents the active spec from changing where
the trajectory is cut.

### 2. Graph spec construction

`AgentSpecConstructor` runs the shared segmenter independently over every
calibration trajectory, then clusters all normalized segments by intent, state
transition, and evidence pattern. Clusters must form an exact partition: every
calibration segment belongs to exactly one cluster. The constructor mechanically
turns each cluster into a `NodeTypeSpec` and adds reusable `EdgeTypeSpec`
relationships and graph constraints.

The output `GraphSpec` is the runtime contract for downstream decomposition. It
contains node definitions, evidence criteria, weights, edge definitions, typical
ordering, and DAG/connectivity rules. The saved provenance includes the source
segments, cluster membership, model messages, and step counts. Use
`--segmentation-max-steps` for each calibration trace and `--max-steps` for
clustering.

### 3. Segment assignment and graph construction

`AgentDecomposer` first invokes the same type-blind segmenter on a new
trajectory. A second loop receives those segments plus the active `GraphSpec`,
then classifies and connects the work:

- a node must cite one or more source `segment_ids`;
- consecutive same-type segments may merge into one accomplishment;
- unmatched segments may remain outside the graph;
- one segment cannot be assigned to multiple nodes;
- node `event_span` values are derived mechanically from assigned segments.

The spec's node and edge labels are injected into both the assignment prompt and
the `submit_graph` JSON Schema. `validate_graph_draft` then checks event IDs,
evidence, claims, allowed types, edge constraints, required edges, and acyclicity.
Without `--spec`, decomposition uses the built-in SWE-style coding ontology.
Resume mode refuses to mix graphs built under different spec hashes. Use
`--segmentation-max-steps` for segmentation and `--max-steps` for assignment.

### 4. Evidence verification and evaluation

`AgentJudge` does not rewrite the graph. It evaluates one node at a time against
the original trace, using graph parents and children only as context. It re-reads
cited events, inspects later events for reversions or contradictions, and assigns
one fixed `NodeStatus`:

- `claimed`: evidence is missing, weak, misread, or the work was undone;
- `evidenced`: the trace supports that the accomplishment occurred;
- `verified`: the claim has mechanical support such as a passing test, observed
  reproduction, or emitted patch.

The evaluator consumes these verified graphs, deduplicates semantic
accomplishments, and computes graph-level progress and evidence metrics.

## Agent tools

**Segmenter**

- `run_python` — execute read-only Python against the loaded `trace` sandbox:
  `trace.problem()`, `trace.summary()`, `trace.events()`, `trace.event(id)`,
  `trace.turn(n)`, `trace.grep(pattern)`, `trace.span(start, end)`,
  `trace.uncovered(spans)` (coverage of uncited events).
- `submit_segments` — submit ordered, non-overlapping work units grounded in real
  event IDs.

**Spec constructor**

- `list_segments` — inspect all normalized calibration segments.
- `compare_segments` — compare selected segments with their original event spans.
- `submit_clustering` — submit the exact segment partition, edge types, ordering,
  and graph rules used to build a `GraphSpec`.

**Decomposer**

- `list_segments` — inspect the type-blind work segments produced for this trajectory.
- `run_python` — inspect original events before writing claims and evidence.
- `submit_graph` — submit `nodes` (with `segment_ids`) and `edges` as JSON arrays. Accepted
  only after assignment checks and `validate_graph_draft` (real event IDs, evidence excerpts,
  no placeholders, causal edges when there is more than one node).

**Judge**

- `inspect_graph` — omit `node_id` for the full node/edge/root list; pass `node_id`
  for that node's parents and children.
- `run_python` — same sandbox as the decomposer (`trace.problem()`, `trace.event`,
  `trace.span`, `trace.grep`, …). The judge uses it to re-read cited events and
  later context, not to cover the whole trace.
- `submit_verdict` — `status` (`claimed` / `evidenced` / `verified`), `confidence`
  in `[0, 1]`, and a reason citing event IDs. `claimed` = weak, missing, or undone;
  `evidenced` = the work happened; `verified` = mechanical proof. Graph progress
  uses verified = 1, claimed/evidenced = 0.

## Core data contracts

- `WorkSegment` is the shared, ontology-free unit of trajectory work.
- `SegmentCluster` records the calibration provenance behind one learned node type.
- `GraphSpec` is the serialized contract consumed by the decomposer and judge.
- `AccomplishmentGraph` contains typed nodes, causal edges, spec metadata, source
  segments, and agent traces.
- `NodeStatus` remains fixed across custom specs so verification and evaluation
  are comparable.

## Usage

### Install and configure

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

export TRACEGRAPH_API_KEY="..."
# Optional:
export TRACEGRAPH_MODEL="glm-3.5-flash"
export TRACEGRAPH_API_BASE="https://opencode.ai/zen/go/v1"
```

The defaults use OpenCode Go: `TRACEGRAPH_MODEL` defaults to
`glm-3.5-flash`, and `TRACEGRAPH_API_BASE` defaults to
`https://opencode.ai/zen/go/v1`. Each agent command also accepts an explicit
model flag.

### 1. Prepare trajectories

Audit the configured remote datasets and create a calibration/evaluation sample:

```bash
tracegraph audit --output artifacts/audit.json

tracegraph sample \
  --per-model 20 \
  --models claude-opus-4.7,gpt-5-mini \
  --output artifacts/pilot.jsonl
```

You can instead use an existing trajectory JSONL file as the input to
`construct-spec` and `decompose`.

To run the same pipeline on your own Cursor agent sessions, convert the local
transcript cache (`~/.cursor/projects/*/agent-transcripts`) first:

```bash
tracegraph ingest-cursor \
  --output artifacts/cursor_trajectories.jsonl \
  --project agent-goal-decomp

tracegraph construct-spec \
  --input artifacts/cursor_trajectories.jsonl \
  --output artifacts/cursor_graph_spec.json \
  --sample-size 8 \
  --seed 7

tracegraph decompose \
  --input artifacts/cursor_trajectories.jsonl \
  --output artifacts/cursor_graphs.jsonl \
  --spec artifacts/cursor_graph_spec.json
```

`ingest-cursor` writes one JSONL row per user request by default. A long
Cursor session is split at each `<user_query>`: `problem_statement` is that
request, and `messages` are only the following assistant turns until the next
user message. Earlier queries are stored on `prior_user_queries`. Use
`--no-split-user-turns` to keep each conversation as a single trajectory.
Tool *results* are usually absent from the cache, so observations are sparse;
spec construction and decomposition still run over tool invocations and
assistant reasoning. Skip `--include-empty-window` and subagent transcripts
unless you want those duplicates. Use `--include-chat` to keep turns with no
tools.

### 2. Construct a graph spec

Select a seeded calibration sample, segment each trace, and cluster its work
units into a reusable ontology:

```bash
tracegraph construct-spec \
  --input artifacts/pilot.jsonl \
  --output artifacts/graph_spec.json \
  --sample-size 10 \
  --seed 7 \
  --model glm-3.5-flash \
  --segmentation-max-steps 8 \
  --max-steps 24
```

`--segmentation-max-steps` is the per-trajectory segmentation budget;
`--max-steps` is the clustering budget. Use repeatable `--trace-model` flags to
restrict calibration to particular coding-agent models.

### 3. Decompose trajectories

Apply the learned spec using type-blind segmentation followed by segment
assignment:

```bash
tracegraph decompose \
  --input artifacts/pilot.jsonl \
  --output artifacts/graphs.jsonl \
  --spec artifacts/graph_spec.json \
  --model glm-3.5-flash \
  --segmentation-max-steps 8 \
  --max-steps 16
```

Omit `--spec` to use the built-in coding-agent ontology. Decomposition resumes
by default: rerunning the same command skips trajectory/model pairs already in
the output. Use `--no-resume` to rebuild the output, `--limit` for a smoke test,
and `--no-continue-on-error` to stop on the first failure.

### 4. Verify and evaluate

Verify each graph node against its original trajectory, then compute aggregate
metrics:

```bash
tracegraph verify \
  --input artifacts/graphs.jsonl \
  --output artifacts/verified.jsonl \
  --judge-model glm-3.5-flash \
  --judge-max-steps 12

tracegraph evaluate \
  --input artifacts/verified.jsonl \
  --output artifacts/evaluation.json
```

Verification also resumes by default and supports `--limit`,
`--no-resume`, and `--no-continue-on-error`.

### Inspect and annotate results

Pretty-print decomposed or verified graphs:

```bash
python scripts/print_graphs.py artifacts/graphs.jsonl --limit 1
python scripts/print_graphs.py artifacts/verified.jsonl --instance-id TASK_ID
```

Create a blinded annotation packet:

```bash
tracegraph annotation-template \
  --input artifacts/graphs.jsonl \
  --output annotations/pilot.jsonl \
  --limit 30
```

Run the test suite:

```bash
pytest
```

## Research guardrails

- Gold patches and hidden tests are evaluation-only inputs.
- A successful command is evidence of execution, not evidence that its semantic
claim is correct.
- Raw node counts are not a progress metric. The evaluator deduplicates semantic
nodes and penalizes invalidation and regressions.
- Report results grouped by task ID so multiple model trajectories for one task
are not treated as independent samples.

