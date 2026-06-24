"""Hand-labelled ground truth for evaluation.

HOW THIS SET WAS BUILT (reproducible methodology, not a guess):
Every one of the 56 judgments was read and reduced to OBJECTIVE facts — court
level, vehicle type, any driving-licence defect, the insurer-liability outcome,
whether it applies the compensation multiplier, whether contributory negligence
was in issue, and a one-line holding. Those facts are recorded in
`docs/GOLDEN_TEST_SET.md`. The query labels below are then derived from those
facts by an EXPLICIT rubric (also documented there), so each label is defensible
and reproducible rather than subjective.

KEY FINDINGS that shape these labels:
- The corpus is UNIFORMLY PRO-CLAIMANT on the unlicensed-driver question: every
  judgment where a driver had a licence defect (none/fake/expired/wrong-class)
  routed to insurer liability or "pay and recover" for the THIRD-PARTY victim.
  NO judgment exonerates an insurer purely on an unlicensed-driver ground.
  => Supporting precedents for the case brief are plentiful; the genuine ADVERSE
     material is the policy-breach / gratuitous-passenger EXONERATION line, which
     the insurer would cite by analogy and the client must DISTINGUISH (the
     deceased was a third party, not a passenger in the offending vehicle).
- DOC_031 (Laxmi Narain Dhut, SC) and DOC_032 (S. Iyyapan, SC) are the two
  landmark authorities — and BOTH SUPPORT a third-party claimant (Iyyapan: insurer
  liable despite wrong-class licence; Dhut: pay-and-recover applies to third-party
  claims). They are labelled SUPPORTING, not adverse.
- ~17 of the 56 docs are DISTRACTORS with no motor-accident bearing (consumer /
  health insurance: 036/037/038; terror & medical death: 039/040; excise: 044;
  trademark/IP: 046/047/048/049; banking/NI Act: 050/052; criminal fraud: 051;
  civil property: 053/054/055/056). They are relevant to NO query and exist to
  test precision.
- DOC_015/016/017 are CRIMINAL driving cases (no insurer), excluded from the
  insurer-liability gold set. DOC_036/037 are near-identical companion judgments;
  DOC_024 is NOT a duplicate of DOC_022 (an earlier note to that effect was wrong).

Each entry: query -> {supporting: [...], adverse: [...], deep: bool}.
`deep` flags queries that should trigger the full three-section research workflow.
"""
from __future__ import annotations

GOLD: dict[str, dict] = {
    # ====================================================================
    # Q1 — Core case brief: unlicensed driver of a commercial truck, insurer
    # denies the third-party death claim as void.
    #   SUPPORTING = motor third-party claim with a licence defect where the
    #     insurer was held liable OR ordered to pay-and-recover (the pro-claimant
    #     line that defeats the "policy void" defence).
    #   ADVERSE    = motor case where the insurer ESCAPED third-party liability on
    #     a policy-breach / coverage ground (distinguishable, but the insurer's
    #     ammunition; the client distinguishes them as gratuitous-passenger /
    #     use-breach cases, not licence cases against a third party).
    # ====================================================================
    "Client's husband was killed by a commercial truck whose driver had no valid "
    "driving licence; the insurer denies the motor accident claim as void because "
    "the driver was unlicensed. Find supporting and adverse precedents and "
    "recommend a strategy.": {
        "supporting": [
            "DOC_001",  # wrong-class licence, commercial tanker -> pay & recover
            "DOC_003",  # expired licence -> pay & recover (Swaran Singh)
            "DOC_005",  # alleged fake licence unproven -> insurer liable
            "DOC_006",  # no licence, National Ins. tanker -> pay & recover (closest fit)
            "DOC_024",  # no licence proven -> still pay & recover (third party protected)
            "DOC_025",  # Full Bench: any s.149(2) breach -> pay & recover, no exoneration
            "DOC_027",  # HGV driven on LMV licence -> pay & recover
            "DOC_031",  # Laxmi Narain Dhut (SC): third-party claims -> pay & recover
            "DOC_032",  # S. Iyyapan (SC): wrong-class licence -> insurer liable
            "DOC_033",  # LMV licence for goods vehicle -> pay & recover
            "DOC_034",  # briefly-lapsed licence, breach unproven -> insurer liable
            "DOC_035",  # maxi-cab on LMV licence -> pay & recover
            "DOC_041",  # no-licence defence pleaded but unproven -> insurer liable
        ],
        "adverse": [
            "DOC_002",  # tractor used non-agriculturally -> insurer escapes (use breach)
            "DOC_014",  # gratuitous passenger in goods vehicle -> insurer exonerated
            "DOC_028",  # hire/reward passenger not a third party -> insurer exonerated
            "DOC_029",  # gratuitous passengers in goods vehicle -> insurer exonerated
            "DOC_030",  # gratuitous passenger on tractor -> no pay & recover, exonerated
        ],
        "deep": True,
    },
    # ====================================================================
    # Q2 — Compensation methodology for death of an earning member.
    #   relevant = motor-accident DEATH judgments whose substantive holding
    #     applies the multiplier method / future prospects / dependency calc
    #     (Sarla Verma, Pranay Sethi, Susamma Thomas). Licence-primary and
    #     child-death (notional income) judgments are excluded to keep precision;
    #     non-motor death-compensation cases (DOC_039 terror, DOC_040 medical) are
    #     excluded as out-of-domain (see GOLDEN_TEST_SET.md).
    # ====================================================================
    "How is compensation calculated for the death of an earning member with "
    "dependents? Find precedents on the multiplier method and future prospects.": {
        "supporting": [
            "DOC_004",  # Pranay Sethi / Sarla Verma enhancement
            "DOC_006",  # future prospects added under Pranay Sethi
            "DOC_007",  # Sarla Verma: dependency / multiplicand, LR excluded
            "DOC_008",  # Full Bench on choice of multiplier (fatal accidents)
            "DOC_011",  # Sarla Verma multiplier, death of motorcyclist
            "DOC_012",  # Sarla Verma / Pranay Sethi, widow + minor children
            "DOC_018",  # Sarla Verma / Pranay Sethi, quantum upheld
            "DOC_019",  # multiplier + future prospects, four death appeals
            "DOC_023",  # Susamma Thomas multiplier, death by jeep
        ],
        "adverse": [],
        "deep": True,
    },
    # ====================================================================
    # Q3 — Simple doc-level lookup: judgments involving a commercial vehicle.
    #   relevant = the accident/matter involves a goods carriage (truck / lorry /
    #     tanker / tempo / goods vehicle) or a commercial passenger vehicle (bus /
    #     mini-bus / maxi-cab / taxi / auto). Agricultural tractors are EXCLUDED as
    #     ambiguous (002, 014, 030), as are vehicle-type-unknown docs (031).
    #     Criminal cases that nonetheless involve a bus (013, 016) ARE included —
    #     the query asks about vehicle involvement, not claim type.
    #   (DOC_014 was removed here after independent verification confirmed its
    #    vehicle is an agricultural tractor, not a goods carrier — see
    #    GOLDEN_TEST_SET.md §7. It remains adverse in Q1 / a passenger case in Q7.)
    # ====================================================================
    "Which of these judgments involve commercial vehicles?": {
        "supporting": [
            "DOC_001", "DOC_005", "DOC_006", "DOC_010", "DOC_011", "DOC_013",
            "DOC_016", "DOC_018", "DOC_019", "DOC_021", "DOC_025", "DOC_026",
            "DOC_027", "DOC_029", "DOC_032", "DOC_033", "DOC_034", "DOC_035",
            "DOC_041", "DOC_042", "DOC_045",
        ],
        "adverse": [],
        "deep": False,
    },
    # ====================================================================
    # Q4 — Contributory negligence (mirrors the assessment's example alt-prompt
    # "find precedents that support our argument on contributory negligence").
    # From the CLAIMANT's side, "support" = judgments REJECTING a contributory-
    # negligence plea (no reduction); "adverse" = judgments APPLYING it to cut the
    # award. Only motor-accident judgments that actually adjudicate the doctrine.
    # ====================================================================
    "Find precedents on contributory negligence in motor accident claims — when "
    "is a claimant's compensation reduced for their own negligence, and when do "
    "courts reject that defence?": {
        "supporting": [
            "DOC_004",  # insurer's contributory-negligence plea fails (no evidence)
            "DOC_006",  # contributory-negligence finding set aside
            "DOC_018",  # contributory-negligence plea rejected
            "DOC_023",  # contributory-negligence plea fails on res ipsa loquitur
        ],
        "adverse": [
            "DOC_009",  # 50% cut: injured rode unlicensed, no helmet, triple-riding
        ],
        "deep": True,
    },
    # ====================================================================
    # Q5 (coverage-A) — LEXICAL / exact-term stress test. The phrase "pay and
    # recover" is a precise legal token: this query exercises the BM25 half of
    # the hybrid retriever that the semantic queries above never touch.
    #   relevant = judgments whose OPERATIVE ORDER is pay-and-recover (insurer
    #     pays the third party, then recovers from owner/driver).
    #   NOTE the built-in precision trap: DOC_028/029/030 mention "pay and
    #     recover" with high term-frequency but REJECT it (insurer exonerated),
    #     so a keyword retriever must not be fooled into citing them here.
    # ====================================================================
    "Which judgments apply the 'pay and recover' principle — directing the insurer "
    "to pay the third-party claimant first and then recover the amount from the "
    "vehicle owner or driver?": {
        "supporting": [
            "DOC_001", "DOC_003", "DOC_006", "DOC_024", "DOC_025",
            "DOC_026", "DOC_027", "DOC_031", "DOC_033", "DOC_035",
        ],
        "adverse": [],
        "deep": False,
    },
    # ====================================================================
    # Q6 (coverage-B) — NEGATIVE / hallucination trap. Verified absent from the
    # corpus (0 docs mention hit-and-run / untraced vehicle / s.161). The honest
    # answer is "no such precedent in this corpus"; the gold is intentionally
    # EMPTY. This tests whether the agent fabricates a plausible holding or
    # honestly reports the gap — the most dangerous failure mode in legal work.
    # A perfect answer cites NOTHING (precision is undefined/​n/a; the signal is
    # that the agent does not invent doc_ids).
    # ====================================================================
    "Find precedents on compensation for hit-and-run accidents where the offending "
    "vehicle was untraced or unidentified (e.g. Section 161 Motor Vehicles Act / "
    "the Solatium Fund).": {
        "supporting": [],
        "adverse": [],
        "deep": False,
    },
    # ====================================================================
    # Q7 (coverage-C) — second SIMPLE doc-level lookup (balances the depth skew)
    # and exercises a distinct corpus theme: persons travelling IN the offending
    # vehicle (gratuitous / hire-reward passengers), as opposed to third parties
    # on the road. This is the coverage cluster the brief only touches as adverse.
    #   relevant = judgments whose victim was a passenger in the offending vehicle.
    # ====================================================================
    "Which judgments involve a person travelling in the offending vehicle as a "
    "passenger (a gratuitous or hire-and-reward passenger), rather than a third "
    "party on the road?": {
        "supporting": [
            "DOC_014",  # gratuitous passenger on goods vehicle
            "DOC_026",  # gratuitous passengers in a tempo (goods vehicle)
            "DOC_028",  # hire/reward passenger in a private jeep
            "DOC_029",  # gratuitous passengers in a goods vehicle / tractor
            "DOC_030",  # passenger on a tractor mudguard
        ],
        "adverse": [],
        "deep": False,
    },
}
