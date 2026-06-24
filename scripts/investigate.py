"""Deep empirical investigation of the first N docs to answer specific
preprocessing questions. Run: PYTHONPATH=. uv run python scripts/investigate.py
"""
from __future__ import annotations

import re

import pymupdf

from src.config import settings

KANOON_RE = re.compile(r"Indian Kanoon\s*-\s*http://indiankanoon\.org/doc/(\d+)/")
PAGE_NUM_RE = re.compile(r"^\s*-?\s*\d{1,4}\s*-?\s*$")
PAGE_OF_RE = re.compile(r"^\s*\d+\s+of\s+\d+\s*$")
COURT_RE = re.compile(r"IN THE (HIGH COURT|SUPREME COURT)[^\n]*", re.I)
NEUTRAL_RE = re.compile(r"Neutral Citation No:?=?\s*([0-9:A-Z]+)", re.I)


def pages_of(pdf):
    with pymupdf.open(pdf) as f:
        return [p.get_text().splitlines() for p in f], f.metadata if False else None


def main():
    pdfs = sorted(settings.raw_dir.glob("*.pdf"))

    # ---- Q1: does the title appear twice on page 1? -------------------------
    print("Q1 — title occurrences on PAGE 1 (fixed header repeat + main caption)")
    for pdf in pdfs[:5]:
        with pymupdf.open(pdf) as f:
            p1 = f[0].get_text().splitlines()
            first_k = next((i for i, l in enumerate(p1) if KANOON_RE.search(l)), None)
            title = ""
            if first_k:
                for j in range(first_k - 1, -1, -1):
                    if p1[j].strip():
                        title = p1[j].strip(); break
            count = sum(1 for l in p1 if l.strip() == title) if title else 0
            print(f"  {pdf.stem}: title={title[:45]!r}... appears {count}x on page1")

    # ---- Q5: is the page number ALWAYS immediately after the Kanoon link? ---
    print("\nQ5 — line(s) immediately AFTER each Kanoon footer line (is it a page #?)")
    for pdf in pdfs[:5]:
        with pymupdf.open(pdf) as f:
            flat = [l for p in f for l in p.get_text().splitlines()]
        after_kanoon = []
        for i, l in enumerate(flat):
            if KANOON_RE.search(l):
                nxt = flat[i + 1].strip() if i + 1 < len(flat) else "<EOF>"
                after_kanoon.append(nxt)
        is_pagenum = [bool(PAGE_NUM_RE.match(x)) for x in after_kanoon]
        print(f"  {pdf.stem}: {len(after_kanoon)} footers, "
              f"{sum(is_pagenum)}/{len(after_kanoon)} followed by a bare page#  "
              f"sample={after_kanoon[:4]}")

    # ---- Q4: how often does "N of M" appear & is it ever inside a sentence? -
    print("\nQ4 — 'N of M' page counters: count + are they standalone lines?")
    total_nofm = standalone = 0
    for pdf in pdfs:
        with pymupdf.open(pdf) as f:
            flat = [l for p in f for l in p.get_text().splitlines()]
        for l in flat:
            if re.search(r"\b\d+\s+of\s+\d+\b", l):
                total_nofm += 1
                if PAGE_OF_RE.match(l.strip()):
                    standalone += 1
    print(f"  corpus-wide: {total_nofm} lines contain 'N of M', "
          f"{standalone} are standalone page-counter lines "
          f"({100*standalone/max(total_nofm,1):.0f}%)")

    # ---- Q2: court name extraction reliability (all docs) -------------------
    print("\nQ2 — court name extraction across ALL docs")
    found = 0; missing = []
    for pdf in pdfs:
        with pymupdf.open(pdf) as f:
            txt = "\n".join(l for p in f for l in p.get_text().splitlines())
        m = COURT_RE.search(txt)
        if m: found += 1
        else: missing.append(pdf.stem)
    print(f"  'IN THE HIGH/SUPREME COURT...' found in {found}/{len(pdfs)}")
    if missing:
        print(f"  missing: {missing}")
        for stem in missing[:5]:
            with pymupdf.open(settings.raw_dir / f"{stem.lower()}.pdf") as f:
                head = [l.strip() for l in f[0].get_text().splitlines() if l.strip()][:12]
            print(f"    {stem} head: {head}")

    # ---- Q3: neutral citation — what do the codes look like? ----------------
    print("\nQ3 — neutral-citation codes present (keep as metadata?)")
    for pdf in pdfs[:5]:
        with pymupdf.open(pdf) as f:
            txt = "\n".join(l for p in f for l in p.get_text().splitlines())
        codes = set(NEUTRAL_RE.findall(txt))
        print(f"  {pdf.stem}: {sorted(codes)[:3] or 'NONE'}")

    # ---- Q6: other reliable metadata in first 5 docs ------------------------
    print("\nQ6 — candidate metadata fields (first 5 docs)")
    pats = {
        "CORAM (judge)": re.compile(r"CORAM:?-?\s*(.+)", re.I),
        "Author": re.compile(r"^Author:\s*(.+)", re.I),
        "Equivalent cite": re.compile(r"Equivalent citations?:\s*(.+)", re.I),
        "Case no (FAO/CA)": re.compile(r"\b((?:FAO|CA|MAC|MFA|WP|CR|SLP)[-\s/][\w\-/()&. ]{3,30})"),
        "Reserved/Pronounced": re.compile(r"(Reserved|Pronounced) on:?-?\s*(.+)", re.I),
    }
    for pdf in pdfs[:5]:
        with pymupdf.open(pdf) as f:
            lines = [l.strip() for p in f for l in p.get_text().splitlines()]
        txt = "\n".join(lines)
        print(f"  {pdf.stem}:")
        for name, rx in pats.items():
            m = rx.search(txt)
            print(f"     {name:22s}: {m.group(m.lastindex)[:50] if m else '—'}")

    # ---- Q8: do any docs need OCR? (is text actually extractable?) ----------
    print("\nQ8 — extractable-text check (chars/page; ~0 would imply scanned/OCR)")
    for pdf in pdfs[:6]:
        with pymupdf.open(pdf) as f:
            per_page = [len(p.get_text().strip()) for p in f]
            images = sum(len(p.get_images()) for p in f)
        print(f"  {pdf.stem}: pages={len(per_page)} "
              f"min_chars/pg={min(per_page)} avg={sum(per_page)//len(per_page)} "
              f"embedded_images={images}")


if __name__ == "__main__":
    main()
