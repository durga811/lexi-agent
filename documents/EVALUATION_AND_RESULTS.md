# Evaluation Framework + Results

*How we measure the agent, what it scores today, where it fails, and what we'd fix
first. Plain language.*

**Covers exactly what the assessment asks for:** one automated eval for each of the
four required dimensions — **Precision, Recall, Reasoning, Adverse** — plus a written
failure analysis (§6).

## 1. How we measure (and why this way)

We score on **two layers**:

- **A deterministic backbone (no LLM, fully repeatable).** We compare the case IDs the
  agent actually **cites** against a hand-built answer key. Exact arithmetic → this
  gives **Precision** and **Recall**, and they don't wobble run-to-run.
- **An LLM-judge layer.** A separate model (Gemini) scores the qualitative things a
  set-comparison can't: is the **reasoning** sound, did it **honestly surface the bad
  cases**, is the **strategy** grounded, is every claim **backed by the retrieved
  text**.

Two things we learned the hard way and built in:

- **LLM judges are noisy.** A single run is unreliable — the reasoning score swung by
  ±0.33. So we run **every question 5 times** and report the **average ± spread**. A
  change smaller than the spread isn't real, and we don't claim it.
- **Not every metric fits every question.** "Adverse" is meaningless for "list the
  commercial-vehicle cases." So each metric only runs where it makes sense (the rest
  show `n/a`), instead of polluting the averages with fake zeros.

## 2. The answer key (how we know what's "right")

We didn't guess what counts as relevant. We **read all 56 judgments**, recorded the
objective facts (court, vehicle, licence defect, and the actual ruling), and **derived
the labels from a written rubric** — then double-checked them with a second,
adversarial pass. (Full detail in [`GOLD_SET.md`](GOLD_SET.md).)

The **8 test questions are weighted toward the core skill** (deep research):

| # | Question | Type |
|---|---|---|
| A1 | The Lakshmi Devi case brief (with her real numbers) | advocacy |
| A2 | Contributory negligence — advise the claimant | advocacy |
| A3 | A passenger claim where **the law is against the client** | advocacy (hard) |
| E1 | How is death compensation calculated? | explanation |
| L1 | Which cases involve commercial vehicles? | lookup |
| L2 | Which cases apply "pay and recover"? | lookup (keyword trap) |
| L3 | Which cases are about **trademark/IP**? | lookup (non-motor) |
| N1 | Hit-and-run / untraced vehicle | **trap** (answer: corpus has none) |

Standard: **strict** for advocacy (a case counts only if its *ruling* truly supports
the position), **inclusive** for lookups.

## 3. The four required dimensions — how each is scored

| Dimension | What it asks | How we score it |
|---|---|---|
| **Precision** | of the cases it cited, how many were right? | deterministic (cited IDs vs key) |
| **Recall** | of the cases it should find, how many did it? | deterministic |
| **Reasoning** | does its "why this case applies" hold up? | LLM judge |
| **Adverse** | did it find the cases that **hurt** the client, honestly? | deterministic (found the known adverse cases) **+** an LLM honesty check |

We also added two extras a legal tool needs: **Faithfulness** (is every claim
traceable to the retrieved text — the "don't hallucinate" check) and **Strategy** (does
the advocacy answer give prioritised arguments + a **realistic, grounded compensation
range** + the risks).

## 4. Results (current, average of 5 runs each)

| Question | Precision | Recall | Adverse | Reasoning | Honesty | Strategy | Faithful |
|---|---|---|---|---|---|---|---|
| A1 Lakshmi Devi brief | 0.67 | 0.49 | **0.28** | 0.85\* | 1.00 | 1.00 ✓range | 0.65 |
| A2 contributory negligence | 0.83 | 0.88 | 1.00 | 0.80\* | 1.00 | 1.00 | 0.56 |
| A3 passenger (hard) | 0.81 | 0.90 | 0.85 | 1.00 | 1.00 | 1.00 | 0.54 |
| E1 compensation method | 0.70 | 0.71 | — | 1.00 | — | — | 0.70 |
| L1 commercial vehicles | 0.95 | 0.39 | — | 1.00 | — | — | 0.64 |
| L2 pay-and-recover | 0.62 | 0.46 | — | 1.00 | — | — | 0.54 |
| L3 trademark (non-motor) | 1.00 | 1.00 | — | 1.00 | — | — | 0.64 |
| N1 hit-and-run (trap) | —† | —† | — | 1.00 | — | — | ~0.5 |

\* Reasoning is the noisiest judge (±0.33 over the 5 runs); read it as "high," not as a
precise number. † N1 has no right answers by design — see §6.

**The short read:** strong on reasoning, honesty, strategy, generalization (it nails a
non-motor topic, L3), and grounding. On the Lakshmi Devi brief it also produces a
**grounded compensation range** (≈₹49–52 lakh, traced to the corpus's multiplier
method) every run. **The one real weak spot is finding the adverse cases for the main
brief (0.28)** — §6.

## 5. We use the eval to make decisions (a worked example)

The eval isn't just a report card — we use it to decide changes, one variable at a
time. Example: the "pay and recover" lookup (L2) scored only ~0.51 precision because
the agent listed cases that *mention* the phrase but actually **reject** it. We changed
**one instruction** — *"list a case only if its ruling matches, not if the term merely
appears or is rejected"* — re-ran, and precision rose (**L2 0.51→0.62, L1 0.86→0.95**)
with no harm to anything else. Kept it. That's the loop: **measure → change one thing →
measure → keep or revert.**

## 6. Where it fails, and what we'd fix first

**Honest answer: finding the *adverse* precedents for the core brief is weak (0.28).**

**Why** (and we proved this, didn't guess): whether a case helps or hurts is decided by
its **ruling**, not its words. The cases that hurt our client (insurer escaped) talk
about "pay and recover" *more* than the helpful ones — because they **reject** it. So
keyword and semantic search literally can't tell them apart, and the adverse cases sit
just below the cut-off where the agent looks. (We confirmed it: prompting the agent to
"search adversarially" didn't help — it can't cite what never gets retrieved.)

**The fix — our #1 next step:** read each judgment's **ruling once at load time** with
an LLM and tag its outcome (insurer liable / pay-and-recover / insurer escaped). Then
the agent pulls adverse cases **by that label** instead of by similarity. This turns an
unsolvable similarity problem into a simple filter and should roughly **double** the
adverse score. It's the #1 item in the ADR's "another week" list.

**Smaller issues:**
- **Lookup recall on long lists** (L1 = 0.39): chunk search under-counts complete sets;
  a metadata index would fix it (a prototype already hit ~0.95 on this query).
- **The trap question (N1) reads as 0** only because the agent politely names the
  closest-but-wrong case while correctly saying "the corpus has nothing on this." That's
  an eval-scoring quirk, **not a hallucination** — a small scorer fix away.

## 7. How to run it

```bash
# full eval (5 runs/question, the headline numbers)
EVAL_SAMPLES=5 EVAL_MAX_WORKERS=8 PYTHONPATH=. uv run python -m src.eval.run_eval

# fast retriever-only check (recall@k, no LLM, seconds)
PYTHONPATH=. uv run python -m src.eval.retrieval_eval
```
Each run writes a full `eval_results*.md`; the headline scoreboard is §4 above.
