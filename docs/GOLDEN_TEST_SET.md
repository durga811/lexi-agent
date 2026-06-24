# Golden Test Set — Lexi Precedent Research Eval

This document is the **ground-truth answer key** behind `src/eval/gold_set.py`. It
records (1) the original task objective the queries are built around, (2) an
objective per-document classification of all 56 corpus judgments, (3) the explicit
rubric that turns those facts into query labels, and (4) the resulting labelled
queries with reasoning. See [`EVALUATION.md`](EVALUATION.md) for how the metrics
*use* this set.

> **Why this exists.** A retrieval metric is only as trustworthy as its answer key.
> Rather than guess which judgments are "relevant," every judgment was read and
> reduced to objective facts (court, vehicle, licence defect, insurer outcome,
> compensation method, contributory negligence, holding). Labels are then *derived*
> from those facts by a written rubric — reproducible, auditable, defensible.

---

## 1. The original task objective (from the assessment brief)

Build a legal precedent research agent over ~56 Indian court judgments that, given
a client case brief, produces **Supporting Precedents · Adverse Precedents ·
Strategy**, and also handles general queries. The anchor case brief:

> **Client:** Mrs. Lakshmi Devi. **Matter:** motor accident — death of spouse.
> Her husband (age 42, income ₹35,000/month, dependents = widow + two minor
> children aged 8 and 12) was killed in a road accident involving a **commercial
> truck** owned by a transport company, whose driver **had no valid driving
> licence**. The insurer (National Insurance Co.) denies the claim, arguing the
> **policy is void** because the driver was unlicensed, so it bears no liability.

The deceased was a **third party** (not a passenger in the offending truck) — this
is decisive for distinguishing the adverse precedents below.

The eval must measure four dimensions: **Precision, Recall, Reasoning quality,
Adverse identification**.

---

## 2. Per-document classification (all 56 judgments)

`MAC?` = is a motor-accident compensation claim/appeal. `Comm?` = involves a
commercial vehicle. **Insurer-on-licence** is the key axis: `liable` = held liable
despite the defect · `pay&rec` = pay the third party then recover from the owner ·
`exon` = insurer exonerated · `not-lic` = liability turned on something else ·
`na` = not a motor insurance case.

| Doc | Court | Domain | MAC? | Vehicle | Comm? | Licence defect | Insurer-on-licence | Multiplier? | Death? | Contrib? | One-line holding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 001 | HC | motor | ✓ | tanker/truck | ✓ | wrong-class | **pay&rec** | ✓ | ✓ | – | No hazardous-goods endorsement = breach; insurer pays, recovers from owner |
| 002 | HC | motor | ✓ | tractor | – | none | **exon** (use breach) | – | ✓ | – | Tractor used non-agriculturally = breach; insurer not liable, owner pays |
| 003 | HC | motor | ✓ | private car | – | expired | **pay&rec** | ✓ | ✓ | – | Licence expired >30d, no retro validity; insurer pays & recovers |
| 004 | HC | motor | ✓ | private car | – | none | not-lic | ✓ | ✓ | ✓ | Contributory-negligence plea fails; award enhanced (Pranay Sethi) |
| 005 | HC | motor | ✓ | dumper truck | ✓ | fake (unproven) | **liable** | – | ✓ | – | Insurer failed to prove fake licence + owner knowledge; remains liable |
| 006 | HC | motor | ✓ | tanker | ✓ | no licence | **pay&rec** | ✓ | ✓ | ✓ | No-licence tanker (National Ins.); pay & recover; contrib. set aside |
| 007 | HC | motor | ✓ | private car | – | none | not-lic | ✓ | ✓ | – | Sarla Verma: non-dependent LR excluded from dependency |
| 008 | HC | motor | ✓ | unknown | ? | none | na | ✓ | ✓ | – | Full Bench: HC "operative multiplier" not overruled |
| 009 | MACT | motor | ✓ | unknown | ? | none* | not-lic | ✓ | – | ✓ | 50% contributory cut: injured rode unlicensed, no helmet, triple |
| 010 | MACT | motor | ✓ | auto | ✓ | none | not-lic | ✓ | – | – | National Ins. auto rear-ends truck; insurer liable, no licence issue |
| 011 | MACT | motor | ✓ | bus | ✓ | none | not-lic | ✓ | ✓ | – | Death by bus; Sarla Verma multiplier computation |
| 012 | MACT | motor | ✓ | unknown | ? | none | not-lic | ✓ | ✓ | – | Death of breadwinner; widow + minors; Sarla Verma / Pranay Sethi |
| 013 | SC | criminal | – | bus | ✓ | none | na | – | ✓ | – | s.304A set aside; negligence not presumed from a passenger's fall |
| 014 | HC | motor | ✓ | tractor (agri) | – | none | **exon** (passenger) | ✓ | ✓ | – | Gratuitous passenger on a farm tractor that turned turtle; insurer exonerated |
| 015 | HC | criminal | – | unknown | ? | none | na | – | ✓ | – | Driver's acquittal upheld (retaining-wall collapse, reasonable doubt) |
| 016 | HC | criminal | – | bus | ✓ | no licence | na | – | – | – | Rash-driving convictions quashed; lack of licence ≠ negligence (s.181 upheld) |
| 017 | HC | criminal | – | unknown | ? | no licence | na | – | ✓ | – | s.279/304A conviction upheld, sentence reduced (compensation paid) |
| 018 | HC | motor | ✓ | mini truck | ✓ | none | not-lic | ✓ | ✓ | ✓ | Insurer liable (no breach); contrib. plea fails; quantum upheld |
| 019 | HC | motor | ✓ | truck | ✓ | none | not-lic | ✓ | ✓ | – | Four death appeals; multiplier + future prospects enhancement |
| 020 | HC | motor | ✓ | unknown | ? | none | not-lic | ✓ | – | – | Injury (amputation): functional disability + future prospects |
| 021 | MACT | motor | ✓ | truck | ✓ | none | not-lic | ✓ | ✓ | – | Truck hits car (1 death + 3 injured); insurer (valid DL) liable |
| 022 | MACT | motor | ✓ | motorcycle | – | unknown | not-lic | ✓ | ✓ | – | Child death by motorcycle; notional-income child formula; driver liable |
| 023 | HC | motor | ✓ | jeep (private) | – | none | not-lic | ✓ | ✓ | ✓ | Contrib. plea fails (res ipsa); multiplier upheld; s.170 bar on insurer |
| 024 | MACT | motor | ✓ | private car | – | no licence | **pay&rec** | ✓ | – | – | No-licence breach proven, but third party protected: pay & recover |
| 025 | HC | motor | ✓ | goods vehicle | ✓ | (any breach) | **pay&rec** | – | – | – | Full Bench: even a void/breached policy → pay & recover, not exoneration |
| 026 | HC | motor | ✓ | tempo (goods) | ✓ | no licence | not-lic† | – | ✓ | – | Gratuitous passenger uncovered; pay & recover ordered on discretion |
| 027 | HC | motor | ✓ | lorry | ✓ | wrong-class | **pay&rec** | ✓ | ✓ | – | HGV on LMV licence; pay & recover (Swaran Singh/Dhut/Iyyapan) |
| 028 | HC | motor | ✓ | private jeep | – | none | **exon** (passenger) | – | ✓ | – | Hire/reward passenger not a third party; insurer exonerated |
| 029 | HC | motor | ✓ | goods/tractor | ✓ | none | **exon** (passenger) | – | ✓ | – | Gratuitous passengers in goods vehicle; 149(4)(5) not attracted; exonerated |
| 030 | HC | motor | ✓ | tractor | ? | none | **exon** (passenger) | ✓ | – | – | Passenger on tractor mudguard; defence not in 149(2); no pay & recover |
| 031 | SC | motor | ✓ | unknown | ? | fake | **pay&rec** | – | – | – | **Laxmi Narain Dhut**: third-party claims → pay & recover; own-damage → defeated |
| 032 | SC | motor | ✓ | maxi-cab/taxi | ✓ | wrong-class | **liable** | – | ✓ | – | **S. Iyyapan**: LMV licence suffices for maxi-cab; insurer liable to third party |
| 033 | HC | motor | ✓ | goods vehicle | ✓ | wrong-class | **pay&rec** | ✓ | ✓ | – | LMV licence for goods vehicle = breach; pay & recover |
| 034 | HC | motor | ✓ | lorry | ✓ | expired | **liable** | ✓ | ✓ | – | Briefly-lapsed licence; no conscious breach proven; liable, no recovery |
| 035 | HC | motor | ✓ | maxi-cab | ✓ | wrong-class | **pay&rec** | ✓ | – | – | Maxi-cab on LMV licence w/o badge; pay & recover (injury) |
| 036 | Consumer | other | – | n/a | – | – | na | – | – | – | **Distractor**: Corona Rakshak health-insurance claim payable |
| 037 | Consumer | other | – | n/a | – | – | na | – | – | – | **Distractor**: companion of 036 (near-identical) |
| 038 | NCDRC | other | – | n/a | – | – | na | – | – | – | **Distractor**: stock-broker fidelity-insurance claim rejected |
| 039 | HC | other | – | n/a | – | none | na | ✓ | ✓ | – | **Distractor**: terror-blast death; State pays multiplier-based comp. |
| 040 | HC | other | – | n/a | – | none | na | – | ✓ | – | **Distractor**: medical-negligence death; interim compensation |
| 041 | MACT | motor | ✓ | mini bus | ✓ | no licence | **liable** | ✓ | ✓ | – | No-licence defence pleaded but unproven; insurer jointly liable |
| 042 | HC | motor‡ | – | bus | ✓ | none | not-lic | – | ✓ | – | Employer can't recover from driver without inquiry (uninsured govt bus) |
| 043 | HC | motor‡ | – | jeep (private) | – | none | not-lic | – | ✓ | – | Recovery from govt driver's salary invalid without procedure |
| 044 | CESTAT | tax | – | n/a | – | – | na | – | – | – | **Distractor**: Cenvat credit on input-service insurance |
| 045 | HC | motor | ✓ | bus | ✓ | none | not-lic | – | – | – | Requisitioned bus; owner/insurer vicariously liable (injury) |
| 046 | HC | trademark | – | n/a | – | – | na | – | – | – | **Distractor**: Ajay Devgan personality/publicity-rights injunction |
| 047 | HC | trademark | – | n/a | – | – | na | – | – | – | **Distractor**: New Balance counterfeiting decree |
| 048 | HC | trademark | – | n/a | – | – | na | – | – | – | **Distractor**: Azuga trademark/copyright injunction |
| 049 | HC | trademark | – | n/a | – | – | na | – | – | – | **Distractor**: Intel trademark injunction (consent) |
| 050 | SC | banking | – | n/a | – | – | na | – | – | – | **Distractor**: s.145 NI Act cheque-evidence procedure |
| 051 | HC | criminal | – | n/a | – | – | na | – | – | – | **Distractor**: credit-card fraud, s.482 CrPC quashing refused |
| 052 | HC | banking | – | n/a | – | – | na | – | – | – | **Distractor**: NI Act prosecution quashed (no leave for receiver) |
| 053 | HC | civil | – | n/a | – | – | na | – | – | – | **Distractor**: specific-performance temporary injunction |
| 054 | HC | civil | – | n/a | – | – | na | – | – | – | **Distractor**: specific performance, readiness & willingness |
| 055 | HC | civil | – | n/a | – | – | na | – | – | – | **Distractor**: ex-parte decree set aside, impleadment |
| 056 | HC | civil | – | n/a | – | – | na | – | – | – | **Distractor**: CPC amendment of pleadings |

\* DOC_009: the *injured claimant himself* lacked a licence — used to reduce his
award, not to exonerate any insurer.
† DOC_026: gratuitous-passenger bar means no statutory liability, but pay &
recover was ordered on equitable discretion — net outcome is the insurer pays.
‡ DOC_042/043: arise out of motor accidents but adjudicate employer-vs-driver
recovery (no insurer party), so `is_motor_accident_claim = false`.

---

## 3. The labelling rubric

**Q1 — Case brief (unlicensed driver, insurer denies third-party claim).**
- **Supporting** ⇐ `MAC ∧ licence_defect ∈ {no/fake/expired/wrong-class} ∧
  insurer-on-licence ∈ {liable, pay&rec}`. These defeat the "policy void"
  defence for a third party.
- **Adverse** ⇐ `MAC ∧ insurer-on-licence = exon` (insurer escaped third-party
  liability on a policy-breach / coverage ground). None is a *licence*
  exoneration — they are the insurer's best analogies, distinguishable because the
  deceased was a third party, not a gratuitous passenger.
- Everything else (criminal, distractors, methodology-only, not-lic liability) ⇒
  **irrelevant**.

**Q2 — Compensation methodology (death of earning member).**
- **Relevant** ⇐ `MAC ∧ death ∧ multiplier` where quantum methodology is a
  *substantive* holding. Excluded for precision: licence-primary cases (001, 003,
  027, 033, 034), child-death notional-income (022), injury-only (020, 035),
  and non-motor death-comp (039 terror, 040 medical — methodologically on-point
  but out of domain; citing them is only a mild precision miss).

**Q3 — Commercial vehicles (simple lookup).**
- **Relevant** ⇐ accident involves a goods carriage (truck/lorry/tanker/tempo/
  goods vehicle) or commercial passenger vehicle (bus/mini-bus/maxi-cab/taxi/
  auto). Agricultural tractors (002, 030) and vehicle-unknown docs (031) are
  excluded as ambiguous; criminal-but-bus cases (013, 016) are included (the
  question is about vehicle involvement, not claim type).

**Q4 — Contributory negligence.**
- From the claimant's side: **Supporting** = judgments *rejecting* a
  contributory-negligence plea; **Adverse** = judgments *applying* it to cut the
  award. Only motor cases that actually adjudicate the doctrine.

---

## 4. The labelled queries (and the reasoning)

### Q1 · Case brief — `deep: true`
- **Supporting (13):** 001, 003, 005, 006, 024, 025, 027, 031, 032, 033, 034,
  035, 041. Flagships: **006** (no-licence commercial tanker, *same insurer*,
  pay & recover — closest factual fit), **032** S. Iyyapan & **031** Laxmi Narain
  Dhut (binding SC authority), **025** (Full Bench: void policy still → pay &
  recover), **005/034/041** (insurer's licence defence fails on the burden of
  proof — directly rebuts National Insurance Co.'s stance).
- **Adverse (5):** 002, 014, 028, 029, 030 — insurer-escapes cases. **Risk &
  distinguishing:** each turns on a *gratuitous-passenger* or *use* breach, not an
  unlicensed driver, and the victim was inside/using the offending vehicle. Mrs.
  Devi's husband was a *third party*, for whom Swaran Singh / Iyyapan / Dhut route
  even a real licence breach to pay-and-recover, not exoneration.
- **Correction vs. the previous gold set:** it wrongly listed 005, 025, 031, 032
  as *adverse*. All four are pro-claimant (032 & 005 → insurer liable; 025 & 031 →
  pay-and-recover for third parties). Mislabelling the two SC landmarks as adverse
  was the most serious error and is now fixed.

### Q2 · Compensation methodology — `deep: true`
- **Relevant (9):** 004, 006, 007, 008, 011, 012, 018, 019, 023. **012** is the
  tightest factual mirror (widow + minor children, breadwinner death, Sarla Verma
  / Pranay Sethi). **Correction:** dropped 014 (an exoneration case, methodology
  incidental), 039 and 043 (non-motor / not a compensation methodology holding)
  from the previous set — they hurt precision.

### Q3 · Commercial vehicles — `deep: false`
- **Relevant (21):** 001, 005, 006, 010, 011, 013, 016, 018, 019, 021, 025,
  026, 027, 029, 032, 033, 034, 035, 041, 042, 045. **Correction:** added obvious
  ones the previous set missed (001 truck, 006 tanker, 013 & 016 buses); removed
  agricultural tractors (002, 014, 030), a vehicle-unknown doctrinal case (031),
  and a private jeep (043). (DOC_014 was removed during independent verification —
  §7 — once its vehicle was confirmed to be a farm tractor, not a goods carrier.)

### Q4 · Contributory negligence — `deep: true` (NEW)
- **Supporting (4):** 004, 006, 018, 023 (plea rejected). **Adverse (1):** 009
  (50% cut for the claimant's own unlicensed/no-helmet/triple-riding). Mirrors the
  assessment's example alternative prompt and adds a second adverse-bearing query.

### Q5 · "Pay and recover" doctrine — `deep: false` · LEXICAL/BM25 stress (NEW)
- **Relevant (10):** 001, 003, 006, 024, 025, 026, 027, 031, 033, 035 — judgments
  whose operative order is pay-the-third-party-then-recover-from-the-owner.
- **Why it exists:** every other query is conceptual/paraphrase-heavy, so the dense
  retriever carries them and the BM25 half of the hybrid is never measured. "Pay
  and recover" is an exact legal token → this query exercises lexical retrieval.
- **Built-in precision trap:** DOC_028/029/030 mention "pay and recover" with high
  term-frequency (18 / 39 / 12 hits) but *reject* it (insurer exonerated). A
  keyword retriever that ranks on raw term overlap will surface them; they are NOT
  in the gold. Tests *applying* a doctrine vs. merely *discussing* it.

### Q6 · Hit-and-run / untraced vehicle — `deep: false` · NEGATIVE trap (NEW)
- **Gold: EMPTY (by design).** Verified absent: 0 of 56 docs mention hit-and-run /
  untraced / unidentified vehicle / Section 161 / Solatium Fund (grep-confirmed;
  intoxication is likewise effectively absent — the only `drunk`/`liquor` hits are
  an incidental criminal acquittal and an unrelated quoted passage).
- **What it measures:** the most dangerous legal-AI failure — fabricating a
  plausible precedent. A perfect answer cites **nothing** and states the corpus
  has no such authority. Scored by the abstention rule in `metrics.py`: empty
  prediction → 1.0; any fabricated citation → 0.0.

### Q7 · Passenger in the offending vehicle — `deep: false` · theme + depth balance (NEW)
- **Relevant (5):** 014, 026, 028, 029, 030 — judgments whose victim was a
  gratuitous or hire-and-reward *passenger in the offending vehicle*, not a third
  party on the road. A doc-level lookup (parallel to Q3) that also exercises the
  passenger-coverage cluster the case brief only touches as "adverse."

---

## 5. Coverage design — why these seven

The set is deliberately spread across the axes a retrieval eval should span, so a
change to the system is tested broadly, not just against the brief:

| Query | Depth | Intent | Retrieval mode stressed | Adverse? |
|---|---|---|---|---|
| Q1 case brief | deep | support/adverse strategy | semantic (conceptual) | ✓ |
| Q2 compensation method | deep | explanatory "how is X calculated" | semantic | – |
| Q3 commercial vehicles | simple | doc-level enumeration | doc-level | – |
| Q4 contributory negligence | deep | support/adverse | semantic | ✓ |
| Q5 pay-and-recover | simple | doctrine lookup | **lexical / BM25 (exact term)** | – |
| Q6 hit-and-run | simple | **negative / abstention** | empty-answer + distractor noise | – |
| Q7 passenger-in-vehicle | simple | doc-level enumeration (theme) | doc-level | – |

Balance achieved: **3 deep / 4 simple** (proves the agent's dynamic workflow both
ways, not always-go-deep); all three retrieval modes (**semantic, lexical,
doc-level**) exercised; an adverse dimension in **2** queries; and a **negative**
query that nothing else covers. This intentionally stops at 7 — more same-shaped
queries would add eval cost without new signal, and per-query N is already small.

## 6. Known limits of this set

- **Single annotator, small N.** ~15 gold docs per deep query ⇒ a recall delta
  below ~0.05–0.07 is within noise. Good for detecting a reranker-sized change,
  not a 1% tweak.
- **Binary relevance.** Borderline-but-defensible citations (e.g. DOC_039 for Q2,
  DOC_006 cross-listed in Q1 & Q2) are scored hard 0/1. Documented above so a
  precision "miss" can be judged in context.
- **End-to-end, not retriever-level.** These labels score the agent's *cited* set;
  they don't isolate the retriever. Add RAGAS-style ContextRecall/ContextPrecision
  (run gold queries straight through `get_retriever()`) to attribute a delta to
  retrieval specifically — see [`EVALUATION.md`](EVALUATION.md) §5.

## 7. Verification status (how confident are these labels?)

Labels were built in two independent passes, not one:

1. **Extraction pass** — 7 agents read all 56 judgments and recorded the objective
   facts in §2. Labels were then derived by the §3 rubric.
2. **Adversarial verification pass** — 4 separate agents re-read the *source text*
   of every one of the 34 labelled docs, instructed to **REFUTE** each label unless
   the operative order/holding clearly supported it, quoting the deciding sentence.
   Plus grep spot-checks (pay-and-recover footprint, hit-and-run absence, distractor
   domains).

**Outcome: 33 / 34 labels confirmed; 1 corrected.**

| Check | Result |
|---|---|
| Q1 supporting (13) | all confirmed client-favorable (every order is liable / pay-and-recover) |
| Q1 adverse (5) | all confirmed: insurer exonerated on a **non-licence** ground (distinguishable) |
| Q5 pay-and-recover (10) | confirmed; DOC_028/029/030 confirmed as term-heavy NON-answers (traps) |
| Q7 passenger-in-vehicle (5) | all confirmed |
| Q2 methodology (9) | all confirmed as motor-death multiplier holdings |
| Q4 contributory negligence (4 + 1) | all confirmed (rejected vs applied) |
| Q3 commercial vehicles (22→**21**) | **DOC_014 removed** — verified as a farm tractor, not a goods carrier |
| Q6 hit-and-run | confirmed **absent** (0/56) → empty gold is honest |
| 17 distractors | confirmed out-of-domain; appear in no gold list |

**Documented nuances (verified, no label change):**
- **DOC_025** (Full Bench): its *own facts* were a gratuitous-passenger breach, but
  its holding establishes pay-and-recover for *any* s.149(2) breach incl. licence —
  so it genuinely supports the client and belongs in Q1/Q5 as a doctrine authority.
- **DOC_031** (Laxmi Narain Dhut, SC): establishes third-party pay-and-recover but
  disposes by remand, and expressly limits the rule to third-party (not own-damage)
  claims. Kept as the doctrinal authority for Q1/Q5.
- **DOC_009**: an *injury* case (not death), so it is only in Q4-adverse
  (contributory negligence applied), never in the death-methodology set Q2.

This is now a *verified* set rather than a single-pass draft. Residual limit: both
passes are LLM reads of the same text — strong, but not a substitute for a domain
lawyer's sign-off on the borderline doctrinal calls above.
