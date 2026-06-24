"""Prototype normalization for Lexi judgments + before/after measurement.

Strips the repeating Indian Kanoon page furniture (running title, source URL,
page numbers, neutral-citation watermark lines) while preserving the leading
case title/date as document metadata.

Run:  uv run python scripts/clean_text.py
"""
from __future__ import annotations

import re

import pymupdf

from src.config import settings

KANOON_RE = re.compile(r"Indian Kanoon\s*-\s*http://indiankanoon\.org/doc/(\d+)/")
DATE_RE = re.compile(r"on (\d{1,2} [A-Z][a-z]+,? \d{4})")
PAGE_OF_RE = re.compile(r"^\s*\d+\s+of\s+\d+\s*$")            # "25 of 26"
PAGE_NUM_RE = re.compile(r"^\s*-?\s*\d{1,4}\s*-?\s*$")        # "1", "-1-"
NEUTRAL_LINE_RE = re.compile(r"^\s*Neutral Citation No.*$", re.I)
NEUTRAL_CODE_RE = re.compile(r"^\s*\d{4}:[A-Z]{2,}:\d+\s*$")       # 2023:PHHC:141930
DOWNLOADED_RE = re.compile(r":::.*Downloaded on.*:::", re.I)       # scan watermark


def extract_metadata(flat: list[str]) -> dict:
    first_k = next((i for i, l in enumerate(flat) if KANOON_RE.search(l)), None)
    title = ""
    if first_k is not None:
        for j in range(first_k - 1, -1, -1):
            if flat[j].strip():
                title = flat[j].strip()
                break
    header = " ".join(l.strip() for l in flat[: first_k or 0])
    m = KANOON_RE.search("\n".join(flat))
    d = DATE_RE.search(header)
    return {
        "title": title,
        "date": d.group(1) if d else "",
        "kanoon_id": m.group(1) if m else "",
    }


def clean(flat: list[str], meta: dict) -> str:
    """Drop the repeating page furniture; keep the body."""
    title = meta["title"]
    out: list[str] = []
    for l in flat:
        s = l.strip()
        if not s:
            out.append("")
            continue
        if KANOON_RE.search(s):
            continue
        if title and s == title:            # running header/footer title
            continue
        if PAGE_OF_RE.match(s) or PAGE_NUM_RE.match(s):
            continue
        if NEUTRAL_LINE_RE.match(s) or NEUTRAL_CODE_RE.match(s):  # citation watermark
            continue
        if DOWNLOADED_RE.search(s):         # "::: Downloaded on ... :::"
            continue
        out.append(l)
    # collapse the blank runs left behind by removed lines
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    pdfs = sorted(settings.raw_dir.glob("*.pdf"))
    raw_chars = clean_chars = 0
    print(f"{'doc':10s} {'raw':>8s} {'clean':>8s} {'saved':>6s}  date / id")
    print("-" * 70)
    for pdf in pdfs:
        with pymupdf.open(pdf) as f:
            flat = [l for p in f for l in p.get_text().splitlines()]
        meta = extract_metadata(flat)
        raw = "\n".join(flat)
        cleaned = clean(flat, meta)
        raw_chars += len(raw)
        clean_chars += len(cleaned)
        if pdf.stem in ("doc_001", "doc_025", "doc_056"):
            pct = 100 * (1 - len(cleaned) / len(raw))
            print(f"{pdf.stem:10s} {len(raw):8d} {len(cleaned):8d} {pct:5.1f}%  "
                  f"{meta['date']} / {meta['kanoon_id']}")
    print("-" * 70)
    print(f"{'TOTAL':10s} {raw_chars:8d} {clean_chars:8d} "
          f"{100*(1-clean_chars/raw_chars):5.1f}% removed")

    # Show a concrete before/after on doc_001's first page.
    print("\n" + "=" * 70 + "\nBEFORE (doc_001 head):")
    with pymupdf.open(pdfs[0]) as f:
        flat = [l for p in f for l in p.get_text().splitlines()]
    for l in flat[34:44]:
        print("  " + repr(l))
    print("\nAFTER (same region, cleaned):")
    meta = extract_metadata(flat)
    for l in clean(flat, meta).splitlines()[30:40]:
        print("  " + repr(l))
    print("\nExtracted metadata:", extract_metadata(flat))


if __name__ == "__main__":
    main()
