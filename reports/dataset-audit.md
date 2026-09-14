# Dataset audit

Audit date: 2026-09-04. Test-set resolved rates re-counted from Hugging Face on
2026-09-10.

## Public metadata

The SWE-Router datasets expose raw alternating chat messages, terminal outcome,
cost, and call count:

- `instance_id`, `problem_statement`, `messages[{role, content}]`, `model`,
  `resolved`, `instance_cost`, and `api_calls`.
- Gemini and DeepSeek additionally publish `step_cost_list` and `source_split`.
  The Claude Opus 4.7 dataset does not expose those fields.
- Claude Opus 4.7: 210 validation + 346 test trajectories (556 total), 243 MB
  compressed and approximately 828 MB after conversion.
- Gemini 3 Pro: 210 validation + 346 test trajectories.
- DeepSeek v3.2: 1,755 train + 210 validation + 346 test trajectories.
- GPT-5-mini uses several replicated train/validation splits and has 339 test
  rows, so it should not be assumed to align one-to-one with the other models.

The live audit confirmed exact three-model alignment: all 210 validation IDs and
all 346 test IDs are shared by Claude, Gemini, and DeepSeek v3.2, with no
duplicate IDs in any audited split. GPT-5-mini's test split is a slightly
different 339-row set.

Validation `resolved` rates from the 2026-09-04 audit (missing outcomes counted
in the denominator): Claude 56.7%, Gemini 30.5%, DeepSeek v3.2 26.7%, GPT-5-mini
31.0% on its primary `valid` split.

Several Claude, Gemini, and DeepSeek validation/test rows have null `resolved`
values. Analyses should retain a separate missing-outcome category rather than
silently treating null as failure.

## Test-set resolved rates (2026-09-10)

Counted from the Hugging Face `resolved` column on the `test` split (or the
aligned `val` split noted below). **Rate (missing = fail)** uses all rows in the
denominator. **Rate among labeled** drops null `resolved` values.

| Model | Dataset | split | n | resolved | failed | missing | rate (missing = fail) | rate among labeled |
|---|---|---|---|---|---|---|---|---|
| claude-opus-4.7 | `SWE-Router/v4-4k-traj-claude-opus-4.7` | test | 346 | 268 | 76 | 2 | 77.5% | 77.9% |
| gemini-3-pro | `SWE-Router/v4-4k-traj-gemini-3-pro` | test | 346 | 185 | 154 | 7 | 53.5% | 54.6% |
| gpt-5-mini | `SWE-Router/v4-4k-traj-gpt-5-mini` | test | 339 | 165 | 174 | 0 | 48.7% | 48.7% |
| deepseek-v3.2 | `SWE-Router/v4-4k-traj-deepseek-v3.2` | test | 346 | 146 | 196 | 4 | 42.2% | 42.7% |
| deepseek-v4-flash | `SWE-Router/v3-2k-traj-deepseek-v4-flash` | val | 346 | 185 | 159 | 2 | 53.5% | 53.8% |

There is no `SWE-Router/v4-4k-traj-deepseek-v4-flash` dataset. The v3-2k dump has
no `test` split; its `val` split contains the same 346 `instance_id`s as the
v4-4k test pack, so it is the comparable SWE-Smith holdout.

A separate SWE-bench Verified dump exists at
`SWE-Router/swebench-verified-deepseek-v4-flash` (`test`, n=500) with 316/500
resolved (63.2%). That is not the SWE-Smith v4-4k test pack.

The dataset cards are empty. The repository names and mutation-style suffixes in
instance IDs, together with the split sizes reported by SWE-Router, indicate that
these validation/test records are SWE-Smith synthetic mutations rather than the
500-task SWE-bench Verified set. This makes them suitable for rapid controlled
comparison, but not sufficient for an ecological-validity claim about human
software issues.

## Trace properties

The messages are shell-harness transcripts, not structured tool calls. Commands,
return codes, output, edits, tests, and final diffs can still be recovered with a
deterministic parser. A sampled successful Gunicorn trajectory had five API calls
and cleanly exposed localization, source inspection, implementation, and patch
submission—but did not run a test. That example illustrates why “command
succeeded” and “task accomplishment was verified” must remain separate.

## Limitations affecting the study

1. `resolved` is the only supplied correctness label; there are no intermediate
   repository snapshots or per-step test labels.
2. Replaying intermediate states requires the corresponding repository snapshot
   and sandbox, which are not embedded in these parquet files.
3. Gold-patch alignment is useful for offline evaluation but must be hidden from
   the decomposer to avoid answer leakage.
4. The same task across models is a paired observation. Statistical intervals
   and splits must group by `instance_id`.
5. Thought text may rationalize actions after the fact. Environment observations
   and artifact deltas should receive higher evidentiary weight.

Run `tracegraph audit` to produce an exact live audit, including field sets,
resolution rates, trace-length distributions, duplicates, and cross-model ID
intersections.
