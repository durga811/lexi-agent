# Retrieval Experiments — Tracking Log

Running log of retrieval/agent improvements: each is implemented, measured against
the verified gold set, recorded here, and committed. One variable at a time.

- **Gold set:** 7 verified queries (`docs/GOLDEN_TEST_SET.md`).
- **Two evals:**
  - **Retriever-level** (`src/eval/retrieval_eval.py`, fast, ~seconds) — runs gold
    queries straight through the retriever; measures whether gold docs appear in
    retrieved chunks (recall@k). Isolates retrieval from the agent. Use for tight
    A/B loops.
  - **End-to-end agent** (`src/eval/run_eval.py`, ~13 min) — drives the live agent
    + Gemini judges; the headline metrics. Run for milestones / winners only.
- **Durable interpretation:** `ADR.md` §6. **Per-query labels:** `GOLDEN_TEST_SET.md`.

---

## Roadmap (from the agreed table)

| # | Improvement | Targets | Status |
|---|---|---|---|
| **I8** | Retriever-level eval (ContextPrecision/Recall) | measurement infra | ✅ done |
| **I1** | Cross-encoder reranker (MiniLM, over-retrieve 40 → top-8) | precision + recall | ✅ merged (+0.19 prec, +0.09 rec) |
| **I2** | Counter-query / adverse-retrieval step (prompt-only) | adverse_recall | ❌ null — reverted (retrieval is the bottleneck → I7) |
| **I3** | Raise top_k / over-retrieve | recall | ⏳ |
| **I4** | Per-document summary index (parent-doc retrieval) | doc-level recall (Q3/Q7) | ⏳ |
| **I5** | Tune ensemble BM25/dense weights | lexical vs semantic balance | ⏳ |
| **I6** | Structure-aware chunking (Held/Ratio) | Q2 reasoning (0.30) | ⏳ |
| **I7** | Metadata filtering (court/vehicle/year) | precision, doc-level | ⏳ |
| **I9** | Eval hardening — faithfulness-vs-context probe | eval correctness | ✅ added (faithfulness); multi-sample/abstention still ⏳ |

Legend: ⏳ todo · 🔄 in progress · ✅ merged · ❌ rejected (kept for the record).

---

## Results at a glance (master comparison)

| Exp | Change | Metric moved | Result | Why |
|---|---|---|---|---|
| I8 | retriever-level eval | — | ✅ infra | gives deterministic recall@k in seconds; enables fast A/B |
| **I1** | **reranker — MiniLM** | precision/recall | ✅ **SHIPPED** | gold docs were in the pool but ranked low (r@8 0.32→r@40 0.71); cross-encoder re-floats them → **+0.19 prec / +0.09 rec** |
| I1 | reranker — bge-base | precision/recall | ❌ rejected | +0.02 r@8 only, broke the lexical query (Q4), 10× slower |
| I1 | reranker — RRF fusion | precision/recall | ➖ fallback | +0.04 r@8; preserves deep recall but gives up half the r@8 gain |
| I2 | counter-query prompt | adverse_recall | ❌ null | 0.20 vs 0.20 (3-run A/B); adverse docs never reach top-8, so prompting can't help |
| I7a | vehicle metadata tag | Q3 doc-recall | ✅ works (unwired) | objective attribute → reliably taggable; recall 0.95 → would lift Q3 0.38→~0.95 |
| I7b | outcome metadata tag | adverse_recall | ❌ fails | regex counts keywords, not the holding; adverse docs *discuss* "pay & recover" while rejecting it |

**The single pattern:** levers acting on *content relevance* (I1) or *objective
attributes* (I7a) **work**; levers needing to read the *operative holding* —
adverse-vs-supporting — **all fail** (I2, I7b), because that signal is in the
ruling, not the vocabulary or embedding. That is the architectural ceiling on the
adverse dimension for this corpus.

## Experiment flow

```
                I8  measurement infra (deterministic recall@k, fast A/B)
                 │
                 ▼
        ┌─────  I1  reranker (over-retrieve 40 → rerank → top-8)  ─────┐
        │  A bge-base ❌  +0.02 r@8, broke Q4, 10× slow               │
        │  B MiniLM   ✅  +0.07 r@8, no per-query regression  ◀── ship │
        │  C RRF-fuse ➖  +0.04 r@8 (fallback if top_k raised)         │
        └──────────────────────────┬──────────────────────────────────┘
                                    ▼  end-to-end: +0.19 prec / +0.09 rec  → MERGED
        adverse_recall still ~0.20 → two attempts, both fail (same root cause):
            I2  counter-query prompt     ❌ null   (0.20 = 0.20)
            I7b outcome metadata (regex) ❌ fails  (keywords ≠ holding)
        objective-attribute axis instead:
            I7a vehicle metadata (regex) ✅ works  (0.95 recall → Q3 0.38→~0.95)  [exp branch, unwired]

  branches:  main ──▶ retrieval-improvements (I8, I1 shipped, I2 reverted, I7 notes)
                            └──▶ exp/i7-metadata (I7 experiment code, isolated)
```

---

## Baseline — end-to-end agent (2026-06-24, `gemini-3.5-flash`, commit 402d181)

| Query | precision | recall | f1 | adverse_recall | reasoning |
|---|---|---|---|---|---|
| Q1 case brief (deep) | 0.90 | 0.50 | 0.64 | **0.20** | 0.90 |
| Q2 compensation method (deep) | 0.43 | 0.33 | 0.38 | — | 0.30 |
| Q3 commercial vehicles | 0.75 | 0.43 | 0.55 | — | 0.90 |
| Q4 contributory negligence (deep) | 0.67 | 0.80 | 0.73 | 1.00 | 0.80 |
| Q5 pay-and-recover (lexical) | 0.63 | 0.50 | 0.56 | — | 0.50 |
| Q6 hit-and-run (negative) | n/a* | n/a* | — | — | 0.70 |
| Q7 passenger-in-vehicle | 0.43 | 0.60 | 0.50 | — | 0.50 |

\* empty gold; agent correctly abstained (metric artifact — see GOLDEN_TEST_SET §7).
**Wall-clock:** 13m19s. **Weak axes:** adverse_recall (0.20), overall recall (0.33–0.60), Q2 reasoning (0.30).

---

## Experiment log

### I8 · Retriever-level eval — baseline measured ✅

Added `src/eval/retrieval_eval.py`: runs gold queries straight through the
ensemble, measures doc-level recall@k (k = chunks the agent would see). Harness
takes any `ranking_fn`, so reranker configs reuse it.

**Baseline ensemble (BM25 0.4 + dense 0.6), recall@k:**

| Query | gold | r@8 | r@20 | r@40 |
|---|---|---|---|---|
| Q1 case brief | 18 | 0.167 | 0.333 | 0.556 |
| Q2 compensation method | 9 | 0.333 | 0.333 | 0.667 |
| Q3 commercial vehicles | 21 | 0.095 | 0.238 | 0.429 |
| Q4 contributory negligence | 5 | 0.600 | 0.800 | 0.800 |
| Q5 pay-and-recover | 10 | 0.300 | 0.600 | 0.800 |
| Q7 passenger-in-vehicle | 5 | 0.400 | 1.000 | 1.000 |
| **MEAN** | | **0.316** | **0.551** | **0.709** |

**Finding (decides I1):** mean recall more than **doubles from r@8 (0.316) to r@40
(0.709)** — the gold docs are in the candidate pool, just ranked too low. A
cross-encoder reranker that over-retrieves ~40 and refilters to 8 has real
headroom to lift agent recall. **Exception: Q3 commercial** is low even at r@40
(0.429) — its gold docs are genuinely hard to retrieve by content (doc-level
enumeration), pointing to I4 (per-doc summary index), not the reranker.

_Runtime: ~seconds (vs 13 min for the agent eval) — good for tight A/B loops._

### I1 · Cross-encoder reranker — attempt A: `bge-reranker-base`, pool=50 ⚠️ mixed

Over-retrieve 50 with the ensemble → cross-encoder rerank → measure recall@k.

| Query | r@8 Δ | r@20 Δ |
|---|---|---|
| Q1 case brief | 0.167→0.278 (+0.11) | 0.333→0.389 (+0.06) |
| Q2 compensation | 0.333→0.444 (+0.11) | 0.333→0.556 (+0.22) |
| Q3 commercial | 0.095→0.190 (+0.10) | 0.238→0.333 (+0.10) |
| Q4 contributory neg | 0.600→0.400 (**−0.20**) | 0.800→0.400 (**−0.40**) |
| Q5 pay-and-recover | 0.300→0.300 (0.00) | 0.600→0.500 (−0.10) |
| Q7 passenger | 0.400→0.400 (0.00) | 1.000→0.800 (−0.20) |
| **MEAN** | 0.316→**0.335 (+0.02)** | 0.551→**0.496 (−0.06)** |

**Decision: not a clear win — do not ship as-is.** Two problems:
1. **It helps semantic queries (Q1/Q2/Q3) but hurts lexical-strong ones (Q4/Q7).**
   The cross-encoder reorders away exact-keyword hits BM25 had surfaced well. Net
   r@8 gain is only +0.02; r@20 regresses.
2. **Latency 492s for 6 queries (~82s/query on CPU)** — unshippable; every
   `search_corpus` call would stall the agent.

Hypotheses to test next: (a) a faster reranker (MiniLM) for latency; (b) the
reranker shouldn't fully discard the ensemble order — fuse reranker rank WITH
ensemble rank (RRF) so lexical wins survive; (c) smaller pool. → attempt B.

### I1 · attempt B: `ms-marco-MiniLM-L-6-v2`, pool=50 ✅ better

| Query | r@8 Δ | r@20 Δ |
|---|---|---|
| Q1 case brief | 0.167→0.278 (+0.11) | 0.333→0.333 (0.00) |
| Q2 compensation | 0.333→0.333 (0.00) | 0.333→0.444 (+0.11) |
| Q3 commercial | 0.095→0.190 (+0.10) | 0.238→0.381 (+0.14) |
| Q4 contributory neg | 0.600→0.600 (0.00) | 0.800→0.800 (0.00) |
| Q5 pay-and-recover | 0.300→0.300 (0.00) | 0.600→0.300 (**−0.30**) |
| Q7 passenger | 0.400→0.600 (+0.20) | 1.000→0.800 (−0.20) |
| **MEAN** | 0.316→**0.383 (+0.07)** | 0.551→**0.510 (−0.04)** |

**MiniLM beats bge-reranker-base here:** bigger r@8 gain (+0.07 vs +0.02), it
**does not destroy Q4** (lexical query preserved), and it's **10× faster**
(45s vs 492s for 6 queries). Remaining wart: r@20 regresses on Q5 (lexical
"pay and recover"). → test fusion (attempt C) to keep the lexical ordering at
deeper k while keeping MiniLM's r@8 gains.

### I1 · attempt C: MiniLM pure-rerank vs RRF-fusion — head to head

Mean recall (reranker = MiniLM, pool 50):

| config | r@8 | r@20 | per-query @8 regression? |
|---|---|---|---|
| baseline | 0.316 | 0.551 | — |
| **pure-rerank** | **0.383** (+0.07) | 0.510 (−0.04) | **none** (Q1 +0.11, Q3 +0.10, Q7 +0.20, rest flat) |
| fused-RRF | 0.357 (+0.04) | 0.552 (≈ base) | none |

**DECISION — ship pure-rerank (MiniLM, pool 40, return top-8).** It is a Pareto
improvement at r@8 (the agent's window): **no query is hurt at r@8** and mean
recall rises +0.07 (+21% relative). The r@20 dip doesn't reach the agent (it reads
~8). Fusion preserves deep recall but sacrifices half the r@8 gain — kept as the
fallback if top_k is raised later (I3). Next: wire into `retriever.py`, run the
full agent eval to confirm the end-to-end effect, commit.

### I1 · END-TO-END agent eval — CONFIRMED ✅ merged

Wired into `retriever.py` (`use_reranker=True`, MiniLM, pool 40 → top 8) and ran
the full agent eval (15m33s, 7/7 queries, no errors).

| Query | precision Δ | recall Δ | adv_recall Δ |
|---|---|---|---|
| Q1 case brief | 0.90 → **1.00** | 0.50 → 0.50 | 0.20 → **0.40** |
| Q2 compensation | 0.43 → **0.60** | 0.33 → 0.33 | — |
| Q3 commercial | 0.75 → **0.80** | 0.43 → 0.38 | — |
| Q4 contributory neg | 0.67 → **1.00** | 0.80 → **1.00** | 1.00 → 1.00 |
| Q5 pay-and-recover | 0.63 → **0.71** | 0.50 → 0.50 | — |
| Q7 passenger | 0.43 → **0.83** | 0.60 → **1.00** | — |
| **MEAN (Q1–Q7, ex-Q6)** | **0.64 → 0.83 (+0.19)** | **0.53 → 0.62 (+0.09)** | Q1 **0.20 → 0.40** |

**Verdict: clear win, merged.** Precision up on *every* query (the reranker pushes
topically-adjacent-but-off-gold docs below the cut), recall up overall. The
retriever-level +0.07 r@8 translated to +0.19 agent precision / +0.09 recall.
Cost: +~2 min eval wall-clock; ~3–4s added per `search_corpus` call (acceptable
for a research agent; tunable via `rerank_pool`).

> **Correction (after I2 variance testing):** the single-run "adverse_recall
> 0.20→0.40" reported here is within noise — over 3 runs adverse_recall is ~0.20
> with or without the reranker (it ranges 0.00–0.40 per run). The reranker's
> robust, repeatable gain is **precision/recall**, anchored by the deterministic
> retriever-level r@8 (+0.07, no agent noise). Adverse_recall needs a structural
> fix (I7), not the reranker.

**Caveat:** the G-Eval *reasoning* scores swung erratically (e.g. Q4 0.80→0.10
despite now-perfect precision/recall) — judge noise, not signal. The deterministic
backbone is the trustworthy delta; the LLM-judge layer needs multi-sample
averaging to be A/B-usable (folds into I9).

### I2 · Counter-query / adverse-retrieval step 🔄

**Diagnostic first (fast, retriever-level).** Case-brief adverse_recall was 0.40
(2/5) after I1. Checked whether the 5 adverse docs are reachable under explicit
adverse-framed queries (pool 60):

| Adverse doc | Best rank under a counter-query | Reachable? |
|---|---|---|
| DOC_029 | 1 | yes (already found) |
| DOC_014 | 5 | yes (counter-query) |
| DOC_030 | 7 | yes ("permit/use breach" framing) |
| DOC_028 | 12 | yes (deeper / union of queries) |
| **DOC_002** | **None (not in top 60)** | **no — content too far from the brief** |

**Finding:** the counter-query approach has a realistic **ceiling of ~0.80 (4/5)**.
DOC_002 (agricultural tractor used non-agriculturally; a scooterist victim) is
semantically too distant from "commercial truck, unlicensed driver" to retrieve by
content — it needs **metadata tagging + filtering (I7)**, not query phrasing. Also:
a *single* adverse phrasing only finds 1–2; the docs differ in facts (tractor/jeep/
passenger), so the agent must run **several** adverse-framed searches and union them.

**Change tested:** added a MANDATORY ADVERSE-SEARCH PASS to the deep-research
prompt — run multiple counter-queries in the opponent's vocabulary (exoneration /
policy void / gratuitous passenger / permit breach). Prompt-only.

**Result: NULL — reverted.** A/B on case-brief adverse_recall, 3 matched runs each
(no judges, agent driven directly to isolate the metric):

| prompt | adverse_recall runs | mean | precision | recall |
|---|---|---|---|---|
| old (baseline) | 0.40, 0.20, 0.00 | **0.20** | 0.84 | 0.43 |
| new (counter-query) | 0.20, 0.00, 0.40 | **0.20** | 0.87 | 0.46 |

Identical distributions — **no measurable effect.** Root cause (confirmed by the
retriever diagnostic above): the adverse docs sit at ranks 5–16 and DOC_002 is
unreachable, so they don't survive into the agent's reranked top-8 — prompting the
agent to "search adversarially" can't cite docs it never retrieves. **The
bottleneck is retrieval, not agent strategy.** Reverted the prompt; re-scoped the
adverse-recall fix to **I7 (metadata tagging + filtering)** — tag each doc by
insurer-outcome and retrieve adverse docs by that label, not by content similarity
to the brief. The prompt directive can return once retrieval can surface them.

**Methodology finding (→ I9):** adverse_recall (denominator 5) is too noisy for
single-run A/B — it ranges 0.00–0.40 run to run. The "I1 doubled adverse_recall
0.20→0.40" claim was a single-sample artifact; the true value is ~0.20 and the
reranker did **not** reliably move it (its robust win is precision/recall). Future
A/B of small-denominator or LLM-judge metrics must average ≥3 runs.

### I7 · Metadata tagging — experiment on branch `exp/i7-metadata`

Tested whether **independent** ingest-time heuristics (no gold-label leakage) can
tag docs well enough that metadata filtering beats semantic retrieval. Two axes:

**(a) Vehicle-type tag — WORKS.** Regex counts commercial vs private vs tractor
keywords; `is_commercial = ≥2 commercial hits, not outweighed by private/tractor`.
Scored vs the gold Q3 commercial set: **recall 0.952 / precision 0.833** (24
flagged, 20 of 21 caught; 1 miss = DOC_010, single "auto" mention; FPs are
criminal-bus cases 015/017 and stray mentions). → a vehicle-metadata filter would
lift **Q3 doc-level recall from 0.38 → ~0.95**. Objective attributes ARE reliably
taggable by heuristic.

**(b) Insurer-outcome tag — FAILS.** Regex for exoneration vs pay-and-recover does
NOT separate the adverse (exonerated) docs from the supporting (liable/pay-recover)
docs — DOC_028/029/030 (adverse) show payrec≫exon (look supporting), while several
support docs (032/034/041) show exon≫payrec (look adverse). The adverse docs
discuss "pay and recover" heavily *because they reject/distinguish it*; support
docs discuss "exoneration" *because they reject the insurer's argument*. **Keyword
presence ≠ the operative holding** — the same reason semantic retrieval fails on
these and I2 was null.

**Decisions:**
- **Vehicle metadata: worth shipping** for doc-level queries (Q3-type) — clean,
  honest win. Needs a re-ingest + a metadata-filter path. Kept on `exp/i7-metadata`;
  NOT merged into `retrieval-improvements` (structure unchanged there per plan).
- **Adverse_recall is NOT fixable by heuristic metadata.** It needs an LLM
  outcome-classifier at ingest (read the operative order; heavier, watch eval
  circularity) or it stays a documented limitation — the honest ceiling on the
  adverse dimension for this corpus.

### I9 · Faithfulness-vs-context probe ✅ added

The eval previously judged only `input + answer` — it never checked the answer
against the chunks the agent actually retrieved (the canonical RAG "faithfulness"
probe). Added it:
- **Retrieval capture** (`tools.py`): `search_corpus` logs the chunks it returns;
  the eval brackets each run with `reset_retrieval_log()` / `get_retrieval_log()`
  and passes the union as `retrieval_context`.
- **Custom Grounding G-Eval** (not stock `FaithfulnessMetric`). **Why:** DeepEval's
  `FaithfulnessMetric` only penalises *contradictions* — an unsupported/fabricated
  claim gets verdict "idk" and counts as faithful. Verified: a fabricated answer
  ("aspirin causes kidney failure", context says only "stomach irritation") scored
  **1.0** on the stock metric. For a legal agent, **fabrication is the failure
  mode**, so the G-Eval rubric penalises any claim not traceable to a chunk —
  grounded → 1.0, fabricated → 0.0.

**First signal (Q1, single run):** **faithfulness 0.40** over 79 retrieved chunks,
while precision 0.9 / reasoning 1.0. The classic confident-hallucination signature
— the answer looks good but ~40% of its claims (landmark case names, legal
principles) aren't traceable to the retrieved text. Previously invisible. *(Like
adverse_recall, a single faithfulness score is noisy — the multi-sample harness is
still on the I9 backlog before A/B-ing it.)*
