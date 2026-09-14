# Blinded annotation protocol

Annotators receive the issue and trajectory events, but not the model name,
terminal `resolved` label, gold patch, peer trajectories, or verifier output.

For each trajectory:

1. Mark the smallest non-overlapping spans that produce a durable task-relevant
   state change.
2. Assign one type: `localization`, `diagnosis`, `reproduction`,
   `implementation`, `verification`, or `handoff_artifact`.
3. Cite at least one environment event. Reasoning text alone is insufficient.
4. Write a concise claim and a normalized `semantic_key` of the form
   `<type>:<object>:<change>`.
5. Mark status:
   - `claimed`: the trace only asserts it.
   - `evidenced`: observable evidence supports activity, but not correctness.
   - `verified`: execution or an artifact establishes correctness.
   - `superseded`: a later node replaces an equivalent earlier node.
   - `invalidated`: later evidence reverses or disproves it.
6. Record remaining work without guessing whether the final benchmark passed.

Use one JSON object per node:

```json
{"instance_id":"...","annotator":"A","node_id":"a001","type":"diagnosis","claim":"Identified the early return that bypasses header handling","semantic_key":"diagnosis:should_close:early-return","event_span":["e0002","e0003"],"evidence_event_ids":["e0003"],"status":"evidenced","remaining_work":"Move the fallback after header handling."}
```

Agreement is computed on semantic keys and status, not exact prose. Disagreements
about boundaries are adjudicated only after independent annotation is complete.
Target at least 30 traces balanced by model, outcome, and trace length.
