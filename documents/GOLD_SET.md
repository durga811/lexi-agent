# The Test Gold Set — How the Eval's Answer Key Was Built

*The companion to [`EVALUATION_AND_RESULTS.md`](EVALUATION_AND_RESULTS.md). It
explains the test data: how we decided what the "right" answer is for each
question, so the precision/recall numbers mean something.*

## Why a hand-built answer key

You can't measure retrieval without knowing the correct answer. For a legal
corpus there's no off-the-shelf label set, so we built one by hand — and the
quality of the whole eval rests on it being honest. The machine-readable form
lives in `src/eval/gold_set.py`; this is the plain-language account of how it was
made.

## How it was made (the process)

1. **Read all 56 judgments.** For each, we recorded the objective facts that decide
   relevance: the court, the vehicle type, the licence/coverage defect, who the
   victim was (third party vs passenger), and — crucially — **the actual ruling**.
2. **Wrote a rubric per question**, then derived each label from it rather than
   from gut feel. The rubric turns on the *ruling*, not the words: a case supports
   "the insurer must still pay" only if it actually *held* that, not merely
   discussed it.
3. **Did a second, adversarial pass** — re-checking every label by trying to argue
   the opposite — to catch wishful inclusions.
4. **Set the labelling standard explicitly:**
   - **Strict** for advocacy/adverse — a case counts as *supporting* only if its
     ruling genuinely advances the position, and as *adverse* only if its ruling
     genuinely cuts against it.
   - **Inclusive** for lookups — any case that fits the category is in.

The labels are one careful annotator's reading, scored pass/fail. We document that
honestly; a domain lawyer's sign-off on the few borderline calls is on the
"another week" list.

## The 8 questions and their keys

The set is **weighted toward the core skill** (deep precedent research): 3 advocacy
questions, 1 explanatory, 3 lookups, 1 trap. `kind` drives metric gating — each
metric only runs where it's meaningful (see the eval doc, §1).

| # | Question | Kind | Answer key (gold IDs) |
|---|---|---|---|
| **A1** | The Lakshmi Devi case brief — unlicensed commercial-truck driver, insurer says policy void | advocacy | **11 supporting** (insurer made to pay despite a licence defect) + **5 adverse** (insurer escaped on a *non-licence* ground) |
| **A2** | Contributory negligence — advise the claimant | advocacy | 4 supporting (plea rejected) + 1 adverse (plea applied) |
| **A3** | Gratuitous passenger in a goods vehicle (the law is mostly *against* the client) | advocacy (hard) | 2 supporting (fragile levers) + 4 adverse (insurer exonerated) |
| **E1** | How is death compensation calculated? | explanatory | 7 multiplier-method authorities; no adverse (not adversarial) |
| **L1** | Which cases involve commercial vehicles? | lookup | 21 cases |
| **L2** | Which cases *apply* "pay and recover"? | lookup | 10 cases |
| **L3** | Which cases concern trademark / IP? | lookup (non-motor) | 4 cases |
| **N1** | Hit-and-run / untraced vehicle | trap (negative) | **empty** — the corpus has none; the right answer is to abstain |

## Four design decisions worth calling out

- **A1 carries Mrs. Lakshmi Devi's real numbers** (husband 42, ₹35,000/month, widow
  + 2 minor children). That lets us check the *Strategy* section's compensation
  range against a corpus-grounded figure (≈₹49–52 lakh, from the multiplier method)
  instead of just eyeballing it.
- **A3 is deliberately a losing position.** Most precedents go against a gratuitous
  passenger. It tests whether the agent *honestly* reports bad news and still gives
  a realistic, cautious strategy — not false optimism.
- **L2 is a keyword trap.** Three cases use "pay and recover" heavily but only to
  **reject** it; they are *not* in the key. This catches an agent that pattern-matches
  on the phrase instead of reading the ruling.
- **L3 is non-motor on purpose.** The corpus is ~40 motor cases plus civil / banking
  / criminal / trademark. L3 verifies the agent generalises beyond motor law — and
  that the ~40 motor cases act as distractors it must *not* cite.

## How the key is used

- **Precision / Recall** compare the doc_ids the agent *cites* against this key —
  pure arithmetic, fully repeatable.
- **Adverse** checks whether the known adverse IDs were found.
- The `kind` label gates which metrics apply (e.g. "adverse" never runs on a lookup).
- `src/eval/gold_set.py` enforces consistency at import: every entry's declared
  `kind` must match its shape, so a mislabelled question fails fast.
