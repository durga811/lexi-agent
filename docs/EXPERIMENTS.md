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
| **I8** | Retriever-level eval (ContextPrecision/Recall) | measurement infra | 🔄 in progress |
| **I1** | Cross-encoder reranker (`bge-reranker`, over-retrieve → top-k) | precision + recall | ⏳ |
| **I2** | Counter-query / adverse-retrieval step | adverse_recall (0.20) | ⏳ |
| **I3** | Raise top_k / over-retrieve | recall | ⏳ |
| **I4** | Per-document summary index (parent-doc retrieval) | doc-level recall (Q3/Q7) | ⏳ |
| **I5** | Tune ensemble BM25/dense weights | lexical vs semantic balance | ⏳ |
| **I6** | Structure-aware chunking (Held/Ratio) | Q2 reasoning (0.30) | ⏳ |
| **I7** | Metadata filtering (court/vehicle/year) | precision, doc-level | ⏳ |
| **I9** | Refine negative-query abstention metric | eval correctness | ⏳ |

Legend: ⏳ todo · 🔄 in progress · ✅ merged · ❌ rejected (kept for the record).

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
