# Unresolved trajectories: partial credit and next directions

Analysis date: 2026-09-13. Graphs are the glm-5.3-flash decompose/judge run on the
prefix-stable **50-ID shared cohort** in `artifacts/compare/` (first 20 plus
ranks 21–50). Progress is weighted verified-node credit (verified = 1,
claimed/evidenced = 0). Hidden-test `resolved` is never used in that formula.

## Why look at failures

Binary `resolved` treats every failed trace as zero. The graphs split failures
into at least three kinds:

1. Never found or never edited the bug (low progress; implementation `claimed`).
2. Found and reproduced the issue, but the patch is wrong or incomplete.
3. Full SWE ritual (edit, local script/pytest, submit) that still fails hidden tests.

Overall graph progress is a poor separator of (2) vs (3). Verified
**localization** and **reproduction** are the better partial-credit axis: useful
leftover work on a failed attempt, not a second resolve score.

## Unresolved counts

### Current 50 shared tasks (test JSONL labels)

| Model | Unresolved / 50 | Resolve rate | Verified graph | No graph among unresolved |
|---|---:|---:|---:|---:|
| Claude Opus 4.7 | 5 | 90% | 5 / 5 | 0 |
| Gemini 3 Pro | 22 | 56% | 14 / 22 | 4 decompose (+ 4 judge) |
| GPT-5-mini | 20 | 40% | 19 / 20 | 1 |
| DeepSeek v3.2 | 28 | 56% | 17 / 28 | 9 decompose (+ 2 judge) |

The framework can currently score **55** unresolved graphs. Missing graphs are
biased toward incomplete traces (DeepSeek/Gemini decompose timeouts), so the
scored failure set over-represents “looks like an attempt.”

Among the 50 tasks, 18 are resolved by all four models, 4 by none, and 12 fail
for three models. Pairing must stay grouped by `instance_id`.

### Shared-26 (verified graph in all four models)

Claude **1** unresolved, Gemini **9**, GPT-5-mini **10**, DeepSeek **14**.
Claude’s n=1 is not a class comparison.

### Full SWE-Router test dumps (scale, not yet graphed)

From `reports/dataset-audit.md` (2026-09-10, missing = fail): Claude **76**/346,
Gemini **154**/346, GPT-5-mini **174**/339, DeepSeek **196**/346.

## What overall progress does on unresolved graphs

On the shared 26, mean progress on unresolved traces is still high: Gemini
**83%**, GPT **74%**, DeepSeek **67%** (Claude n=1). Gemini failures often have
verified implementation **and** verification nodes because the judge treats a
local reproduce script, pytest in the sandbox, or an observed `git diff` as
mechanical proof. Hidden tests are evaluation-only and are not in the judge
loop.

Setting handoff weight to 0 raises GPT more than Gemini (shared-26 overall
+5.0 vs +1.3 points) because GPT handoffs are often `evidenced` (submit issued,
no observed diff). It does **not** explain Gemini’s unresolved progress.

**Do not use mean unresolved progress as partial credit.** It scores
complete-but-wrong attempts near 1.0.

## Partial credit: verified localization and reproduction

Restricted to **unresolved** verified graphs.

### All 55 unresolved graphs

| Model | n | Verified loc | Verified repro | Loc and repro | ≥1 of loc/diag/repro |
|---|---:|---:|---:|---:|---:|
| Claude | 5 | 80% | 40% | 40% | 80% |
| Gemini | 14 | 50% | 79% | 36% | 93% |
| GPT-5-mini | 19 | 58% | 26% | 16% | 79% |
| DeepSeek | 17 | 53% | 65% | 29% | 88% |
| Pooled | 55 | **56%** | **53%** | **27%** | **86%** |

Diagnosis looks weak (pooled 26% verified) mostly because only **42%** of
unresolved graphs emit a diagnosis node. Conditional on the node existing:
loc 59%, diag 61%, repro 81%. GPT’s low verified-repro rate is largely
**omitted repro nodes** (only 37% of GPT failures include one); when present,
5/7 are verified.

“≥1 verified early node” is too easy (86%). Prefer:

- verified localization (~56%) — found the file/code;
- **verified loc and repro (~27%)** — found it and showed the bug;
- every present loc/diag/repro node verified (~44%).

### Shared-26 unresolved

| Model | n | Verified loc | Verified repro | Both |
|---|---:|---:|---:|---:|
| Claude | 1 | 1/1 | 0/1 (no repro node) | 0/1 |
| Gemini | 9 | 5/9 (56%) | 7/9 (78%) | 4/9 (44%) |
| GPT-5-mini | 10 | 5/10 (50%) | 4/10 (40%) | 2/10 (20%) |
| DeepSeek | 14 | 8/14 (57%) | 8/14 (57%) | 4/14 (29%) |

Gemini unresolved traces most often have verified reproduction. GPT unresolved
traces more often locate the file without a verified repro node.

## When the graph explains the failure

Of 55 unresolved graphs, **11** have progress ≤ 0.5. Those usually state a
missing edit, empty diff, unrun test, or still-failing repro. Examples:

- GPT `combine_file__xhwp2gow`: implementation `claimed`, empty `git diff`,
  pytest passed on unchanged code (progress 0.28).
- DeepSeek `func_pm_remove_cond__t9rkf3hs`: no edit event; submit after grep
  (progress 0.36).
- GPT `lm_rewrite__ly5l6tf8`: Python 3.11 not installed; pytest collected no
  tests (progress 0.47).
- Gemini `lm_rewrite__ezxnd6h4`: original DST repro still prints FAILURE;
  only mocked clocks verified (progress 0.43).

**44/55** unresolved graphs have progress > 0.5. On those, **31** are weak only
on localization and/or handoff; **10** are fully verified. Remaining-work there
is mostly generic or circular (“instance marked unresolved”). The sandbox
`summary()` currently includes `resolved`, so that circular text can be **label
leakage**, not independent diagnosis.

High-progress failures can still record **coverage holes** (no repo tests in the
checkout, 3.11 path untested on 3.10, ad-hoc scripts deleted before submit).
They do not reliably say the patch is semantically wrong.

## Recommended research direction

Good direction: **partial credit for failed attempts**, not “graphs predict
`resolved`.” The claim to test is that failed trajectories are not all zeros,
and loc/repro credit is closer to useful leftover work than in-trace pytest.

To make that a result rather than a dashboard:

1. **Metric.** Report verified loc and loc∧repro on unresolved traces. Report
   node presence separately. Keep mean progress for the resolved/unresolved
   contrast, not as the failure score.
2. **Scale.** Claude n=5 on this cohort is too small. Full test unresolved
   pools (76 / 154 / 174 / 196) are large enough. The 50-ID set spans only three
   SWE-Smith repos (`async-timeout`, `schedule`, `channels`).
3. **Do not drop decompose failures.** Count “no valid graph” as its own bin
   so DeepSeek/Gemini incompletes are not selected out.
4. **Blind `resolved`.** Remove it from sandbox summaries shown to decomposer
   and judge.
5. **Validate loc/repro** against gold-patch file overlap or blinded humans.
   Judge “verified loc” is still “grep/read succeeded,” not “cited the gold
   hunk.”
6. **Construct check.** Run the continuation experiment in
   `reports/handoff-protocol.md`: does a verified loc+repro graph beat a
   token-matched raw prefix on unresolved tasks?

Go/no-go for this slice: loc/repro partial credit should (a) separate never-edited
failures from ritual-complete wrong patches, (b) agree with gold-file overlap or
annotators, and (c) lift handoff vs a raw summary. If only (a) holds, the graphs
are a diagnostic representation, not measured completed work.
