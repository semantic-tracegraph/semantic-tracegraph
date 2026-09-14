# Handoff utility experiment

This experiment is the main construct-validity test. A graph is useful partial
progress only if it improves a continuation agent, not merely because a judge
finds it plausible.

## Design

Sample at least 20 unresolved trajectory prefixes, blocked by task ID. Start each
continuation from the same repository state and randomize three arms:

1. `prompt`: original issue only.
2. `raw_summary`: original issue plus a token-matched summary of the raw prefix.
3. `verified_graph`: original issue plus verified accomplishment nodes,
   evidence, invalidations, changed artifacts, and remaining work.

Use the same continuation model, harness, token budget, time limit, and
temperature in every arm. Run at least two seeds per task and arm. The evaluator
must not reveal the gold patch or hidden-test results in any prompt.

## Record format

Write one JSON object per run:

```json
{"instance_id":"...","prefix_turn":8,"arm":"verified_graph","seed":1,"model":"...","resolved":true,"cost":0.42,"api_calls":11,"wall_seconds":230}
```

Summarize with:

```bash
tracegraph handoff --input artifacts/handoff-runs.jsonl --output artifacts/handoff-summary.json
```

Report paired success lift, cost conditional on success, and time-to-resolution.
Bootstrap task IDs rather than individual runs. A graph that predicts final
success but does not beat a token-matched raw summary should be described as a
diagnostic representation, not as measured completed work.
