"""Hand-labelled ground truth for evaluation — the eval's answer key.

This is the machine-readable form; the plain-language methodology (how the labels
were built and why) is in `documents/GOLD_SET.md`. The set is weighted toward
ADVOCACY (3 of 8 queries); the Lakshmi Devi brief carries its quantitative facts
so the Strategy's compensation range is checkable; labels use a STRICT standard for
advocacy/adverse and INCLUSIVE for lookups; and a non-motor (trademark) lookup
verifies the agent generalises across the mixed corpus.

Each entry: query -> {supporting, adverse, deep, kind, [strategy_basis,
strategy_must_mention, strategy_range_lakh]}.
  `deep`  flags the three-section / deep-research workflow.
  `kind`  drives metric gating (see metrics.METRIC_APPLICABILITY).
  `strategy_*` (advocacy A1 only) support the actionability check: which docs give
    the compensation-range basis, which concepts the strategy must cover, and the
    corpus-grounded range an answer should land near.
"""
from __future__ import annotations

# Query taxonomy — what behaviour the query expects, which gates the metrics.
KIND_ADVOCACY = "advocacy"        # supporting + adverse + strategy; adversarial
KIND_EXPLANATORY = "explanatory"  # deep, methodology/educational, non-adversarial
KIND_LOOKUP = "lookup"            # simple doc-level list retrieval
KIND_NEGATIVE = "negative"        # correct behaviour is to abstain (empty gold)
ALL_KINDS = (KIND_ADVOCACY, KIND_EXPLANATORY, KIND_LOOKUP, KIND_NEGATIVE)


def _infer_kind(g: dict) -> str:
    """Derive kind from the gold shape (used to validate the explicit `kind`)."""
    sup, adv, deep = g["supporting"], g["adverse"], g.get("deep", False)
    if not sup and not adv:
        return KIND_NEGATIVE
    if adv:
        return KIND_ADVOCACY
    return KIND_EXPLANATORY if deep else KIND_LOOKUP


GOLD: dict[str, dict] = {
    # ====================================================================
    # A1 — Core case brief (Mrs. Lakshmi Devi). Third-party death, unlicensed
    # commercial-truck driver, insurer pleads the policy is void. Facts carried so
    # the Strategy's compensation range is checkable.
    #   SUPPORTING (strict) = third-party + insurer made to pay despite a licence
    #     defect (liable / pay-and-recover) — defeats the "policy void" defence.
    #   ADVERSE = insurer escaped on a NON-licence ground (use breach / gratuitous
    #     passenger); the opponent's analogies, distinguishable (victim = third party).
    # ====================================================================
    "Our client is Mrs. Lakshmi Devi. Her husband — aged 42, monthly income "
    "Rs 35,000, survived by his widow and two minor children (8 and 12) — was "
    "killed by a commercial truck owned by a transport company whose driver had no "
    "valid driving licence. The insurer (National Insurance Co.) denies the claim, "
    "arguing the policy is void because the driver was unlicensed. Find supporting "
    "and adverse precedents and recommend a strategy, including a realistic "
    "compensation range.": {
        "supporting": [
            "DOC_001",  # wrong-class/no-endorsement tanker, road third party -> pay & recover
            "DOC_003",  # expired licence, pedestrian -> pay & recover
            "DOC_005",  # fake licence unproven + owner knowledge not shown -> liable
            "DOC_006",  # CLOSEST FIT: National Ins., no-licence tanker, 3rd-party death -> pay&rec
            "DOC_025",  # Full Bench: even a VOID policy -> still pay the third party
            "DOC_027",  # wrong-class HGV, motorcyclist third party -> pay & recover (Banumathi)
            "DOC_031",  # Dhut (SC) — CAVEAT: own-damage holding; cite only 3rd-party dictum
            "DOC_032",  # S. Iyyapan (SC): cyclist third party, wrong-class -> insurer liable
            "DOC_033",  # LMV licence for goods vehicle (wrong-class) -> pay & recover
            "DOC_034",  # expired/non-renewed licence, road victim, no breach proven -> liable
            "DOC_035",  # no-badge/wrong-class, pedestrian -> pay & recover (Banumathi)
        ],
        "adverse": [
            "DOC_002",  # tractor used non-agriculturally (use breach) -> insurer escapes
            "DOC_014",  # gratuitous passenger on farm tractor -> exonerated
            "DOC_028",  # hire/reward passenger in jeep -> exonerated (not a third party)
            "DOC_029",  # gratuitous passengers in goods vehicle -> exonerated (no pay&rec power)
            "DOC_030",  # passenger on tractor mudguard -> exonerated
        ],
        "deep": True,
        "kind": KIND_ADVOCACY,
        # Quantum authorities for the compensation-range basis (NOT scored as
        # supporting/adverse — used by the Strategy actionability check).
        "strategy_basis": ["DOC_004", "DOC_006", "DOC_018", "DOC_019"],
        # Concepts the Strategy section must cover (each group = acceptable synonyms).
        "strategy_must_mention": [
            ["multiplier"],                                  # the quantum method
            ["future prospect"],                             # +25-30% for age 40-50
            ["third party", "third-party"],                  # the liability theory
            ["pay and recover", "liable", "indemnif"],       # insurer still pays
            ["risk", "adverse", "distinguish"],              # honest risk assessment
        ],
        # Corpus-grounded loss-of-dependency range (multiplier 14 @ age 41-45, +25-30%
        # future prospects, 1/3 deduction). See documents/GOLD_SET.md.
        "strategy_range_lakh": (49, 52),
    },
    # ====================================================================
    # A2 — Contributory negligence (claimant side). Supporting = plea REJECTED;
    # Adverse = plea APPLIED to cut the award.
    # ====================================================================
    "Find precedents on contributory negligence in motor accident claims — when is "
    "a claimant's compensation reduced for their own negligence, and when do courts "
    "reject that defence? Advise for a claimant facing such a plea.": {
        "supporting": [
            "DOC_004",  # no evidence of deceased's negligence; insurer's burden unmet
            "DOC_006",  # eyewitness beats a site-plan inference; 50% cut reversed
            "DOC_018",  # insurer led no evidence; truck on wrong side -> no reduction
            "DOC_023",  # rear-end hit; res ipsa loquitur -> no contributory negligence
        ],
        "adverse": [
            "DOC_009",  # 50% cut: injured rode unlicensed, no helmet, triple-riding
        ],
        "deep": True,
        "kind": KIND_ADVOCACY,
    },
    # ====================================================================
    # A3 — Gratuitous-passenger claim (ADVERSE-HEAVY). Tests honest reporting of a
    # losing position + a cautious, realistic strategy.
    #   SUPPORTING (strict) = 026 (equitable pay&rec, fragile) + 025 (doctrine: the
    #     extra-premium/contractual-coverage theory, the one affirmative lever).
    #   ADVERSE = insurer exonerated against the passenger.
    # ====================================================================
    "Our client was a gratuitous passenger riding in a goods vehicle (a tempo / "
    "tractor) that overturned, and was killed. The insurer denies liability, "
    "saying passengers carried in a goods carriage are not covered. Find supporting "
    "and adverse precedents and recommend a strategy.": {
        "supporting": [
            "DOC_025",  # Full Bench doctrine: extra-premium/contractual coverage theory
            "DOC_026",  # equitable pay & recover ordered for goods-tempo passengers (fragile)
        ],
        "adverse": [
            "DOC_014",  # gratuitous passenger, tractor -> exonerated
            "DOC_028",  # hire/reward passenger, jeep -> exonerated
            "DOC_029",  # gratuitous passengers, goods vehicle -> exonerated, no pay&rec power
            "DOC_030",  # mudguard passenger, tractor -> exonerated, pay&rec refused
        ],
        "deep": True,
        "kind": KIND_ADVOCACY,
    },
    # ====================================================================
    # E1 — Compensation methodology (death of an earning member). Explanatory:
    # genuine Sarla Verma / Pranay Sethi multiplier-method authorities only.
    # ====================================================================
    "How is compensation calculated for the death of an earning member with "
    "dependents? Explain the multiplier method and future prospects with "
    "precedents.": {
        "supporting": [
            "DOC_004",  # full age-band multiplier table + future-prospects rules
            "DOC_006",  # best analogue: earning married man, 40-50 band, 30% prospects
            "DOC_007",  # dependency / deduction-fraction principle
            "DOC_011",  # Sarla Verma multiplier computation
            "DOC_012",  # widow + 2 minors fact pattern; Sarla Verma / Pranay Sethi
            "DOC_018",  # clean Sarla Verma 3-step recital; 1/3 deduction
            "DOC_019",  # full multiplier table + future-prospects chart (Constitution Bench)
        ],
        "adverse": [],
        "deep": True,
        "kind": KIND_EXPLANATORY,
    },
    # ====================================================================
    # L1 — Commercial vehicles (lookup, inclusive). Any goods carriage or commercial
    # passenger vehicle; criminal-but-bus cases included (vehicle involvement, not
    # claim type). Agri tractors / vehicle-unknown excluded.
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
        "kind": KIND_LOOKUP,
    },
    # ====================================================================
    # L2 — "Pay and recover" doctrine (lookup, inclusive — LEXICAL/BM25 stress).
    # Operative order is pay-the-third-party-then-recover. Precision trap:
    # DOC_028/029/030 are term-heavy but REJECT the doctrine (not in gold).
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
        "kind": KIND_LOOKUP,
    },
    # ====================================================================
    # L3 — Trademark / IP disputes (lookup, inclusive — NON-MOTOR DIVERSITY).
    # Verifies the generalised agent on a non-motor topic + reverse precision (the
    # ~40 motor docs are distractors and must NOT be cited). See GOLDEN_SET_V2 §6.
    # ====================================================================
    "Which of these judgments concern trademark or intellectual-property disputes?": {
        "supporting": ["DOC_046", "DOC_047", "DOC_048", "DOC_049"],
        "adverse": [],
        "deep": False,
        "kind": KIND_LOOKUP,
    },
    # ====================================================================
    # N1 — Hit-and-run / untraced vehicle (NEGATIVE / abstention). Verified absent
    # from the corpus (0/56). A perfect answer cites NOTHING and says so.
    # ====================================================================
    "Find precedents on compensation for hit-and-run accidents where the offending "
    "vehicle was untraced or unidentified (e.g. Section 161 Motor Vehicles Act / "
    "the Solatium Fund).": {
        "supporting": [],
        "adverse": [],
        "deep": False,
        "kind": KIND_NEGATIVE,
    },
}

# Fail fast if a `kind` was mislabelled or omitted: every explicit kind must be
# valid and must match what the gold shape implies.
for _q, _g in GOLD.items():
    assert _g.get("kind") in ALL_KINDS, f"bad/missing kind: {_q[:40]}"
    assert _g["kind"] == _infer_kind(_g), (
        f"kind {_g['kind']} != inferred {_infer_kind(_g)} for: {_q[:40]}"
    )
# strategy_* fields only make sense on advocacy queries.
for _q, _g in GOLD.items():
    if any(k.startswith("strategy_") for k in _g):
        assert _g["kind"] == KIND_ADVOCACY, f"strategy_* on non-advocacy: {_q[:40]}"
