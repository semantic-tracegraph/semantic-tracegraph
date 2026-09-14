# Pilot findings and decision framework

## What the prototype can establish now

The parser reconstructs typed events from SWE-Router's bash-only message format.
The baseline decomposer emits only action-grounded nodes, and the verifier keeps
semantic correctness distinct from command execution. Graph scoring deduplicates
semantic keys and applies negative credit to invalidated work.

The reproducible heuristic baseline was run on 60 trajectories covering 20
paired task IDs across Claude, Gemini, and DeepSeek. Six trajectories had no
terminal outcome and were excluded from predictive scoring. In this deliberately
small pilot, graph features achieved grouped ROC-AUC 0.563 versus 0.561 for
simple search/read/edit/test and trace-length features, an incremental gain of
0.002. This is below the
preregistered 0.05 go threshold and has no uncertainty estimate yet. It should
be treated as a pipeline smoke test, not evidence for the research claim.

Across the pilot, mean normalized progress was 0.502, mean evidence precision
was 0.830, mean verified precision was only 0.184, and mean invalidation rate
was 0.010. The low verified fraction is expected from transcript-only evidence,
but the low detected invalidation rate likely means the baseline invalidation
rules are too conservative.

The strongest observed qualitative example is a short Gunicorn trace. It:

1. found `should_close`,
2. exposed an early return that made later connection-header logic unreachable,
3. moved the version fallback after the header loop, and
4. submitted a focused patch.

The task's terminal label is resolved, but the trajectory did not run a test.
Accordingly, the prototype marks localization, diagnosis, and implementation as
evidenced rather than mechanically verified. This is the intended distinction:
terminal success can retrospectively validate the complete patch without
pretending every intermediate semantic claim was independently established.

## Claims not yet supported by data alone

- Human reliability requires two independent annotators on the blinded pilot.
- Execution-level intermediate verification requires reconstructable repository
  snapshots and sandbox images.
- Handoff utility requires new controlled continuation runs.
- Generalization to real human issues requires a second dataset such as
  SWE-bench Verified; the selected SWE-Router split appears synthetic.

These are evaluation runs, not missing software features. The repository includes
the annotation protocol, agreement computation, execution-evidence interfaces,
grouped predictive evaluation, and handoff record summarizer needed to run them.

## Go / no-go thresholds

Treat these as preregistration defaults, not discovered cutoffs:

- semantic-node Jaccard between annotators at least 0.60;
- shared-node status agreement at least 0.75;
- grouped graph ROC-AUC exceeds the trace-length/tool-count baseline by at least
  0.05 with a paired bootstrap interval reported;
- verified-graph handoff improves resolution by at least 5 percentage points or
  reduces successful-run cost by at least 10% versus a token-matched raw summary.

If agreement fails, simplify the ontology. If prediction improves but handoff
does not, frame the result as diagnosis. If handoff improves only when gold
information enters the graph, reject the reference-free accomplishment claim.

## Later 50-cohort analysis (unresolved partial credit)

The agent decompose/judge pipeline was run on 50 shared test IDs across the four
coding models (`artifacts/compare/`). Mean graph progress on unresolved traces
is high (especially Gemini) because `verified` means in-trace mechanical proof,
not hidden-test success. A better failure metric is whether localization and
reproduction nodes are verified. Counts, loc/repro rates, examples, and
recommended next experiments are in
[`unresolved-partial-credit.md`](unresolved-partial-credit.md).
