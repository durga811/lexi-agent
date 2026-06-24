"""Corpus structure analysis for the Lexi judgment dataset.

Verifies the claimed structural patterns across *every* PDF in data/raw and
quantifies recurring boilerplate so we can design a preprocessing pipeline.

Run:  uv run python scripts/analyze_corpus.py
"""
from __future__ import annotations

import re
from collections import Counter

import pymupdf

from src.config import settings

KANOON_RE = re.compile(r"Indian Kanoon\s*-\s*http://indiankanoon\.org/doc/(\d+)/")
# "... on 6 November, 2023" — may appear mid-line; PyMuPDF wraps the header.
DATE_RE = re.compile(r"on \d{1,2} [A-Z][a-z]+,? \d{4}")
MONTH_RE = re.compile(r"\b(January|February|March|April|May|June|July|August|"
                      r"September|October|November|December),? \d{4}")
NEUTRAL_CIT_RE = re.compile(r"Neutral Citation No", re.I)
PAGE_OF_RE = re.compile(r"^\s*\d+\s+of\s+\d+\s*$")
PAGE_DASH_RE = re.compile(r"^\s*-?\s*\d+\s*-?\s*$")  # "-1-", "1"


def load(pdf) -> tuple[list[str], list[list[str]]]:
    """Return (flat_lines, per_page_lines) for a PDF."""
    pages = []
    with pymupdf.open(pdf) as f:
        for page in f:
            pages.append(page.get_text().splitlines())
    flat = [l for pg in pages for l in pg]
    return flat, pages


def main() -> None:
    pdfs = sorted(settings.raw_dir.glob("*.pdf"))
    print(f"Analyzing {len(pdfs)} PDFs\n" + "=" * 70)

    has_title_header = 0
    has_date_line = 0
    has_kanoon = 0
    footer_per_page_ok = 0
    doc_ids: dict[str, str] = {}
    total_lines = 0
    boilerplate_lines = 0
    page_counts = []
    failures = []

    for pdf in pdfs:
        name = pdf.stem.upper()
        flat, pages = load(pdf)
        total_lines += len(flat)
        page_counts.append(len(pages))

        # --- 1. Title header + date ---
        # Canonical title = the line that immediately precedes the first Kanoon
        # footer line (the footer block is: <title> / <kanoon url> / <page#>).
        # That same title also opens the document.
        nonempty = [l for l in flat if l.strip()]
        first_kanoon_idx = next((i for i, l in enumerate(flat) if KANOON_RE.search(l)), None)
        title = ""
        if first_kanoon_idx is not None:
            for j in range(first_kanoon_idx - 1, -1, -1):
                if flat[j].strip():
                    title = flat[j].strip()
                    break
        if not title and nonempty:
            title = nonempty[0].strip()
        # Header block = text before the first footer; the case date lives here.
        header_block = " ".join(l.strip() for l in flat[:first_kanoon_idx or 0])
        line2 = nonempty[1].strip() if len(nonempty) > 1 else ""
        title_ok = bool(title) and not KANOON_RE.search(title)
        date_ok = bool(DATE_RE.search(header_block) or MONTH_RE.search(header_block))
        has_title_header += title_ok
        has_date_line += date_ok

        # --- 2. Indian Kanoon link + doc id ---
        m = KANOON_RE.search("\n".join(flat))
        if m:
            has_kanoon += 1
            doc_ids[name] = m.group(1)

        # --- 3. Footer repeats on (almost) every page? ---
        # Footer block = title line + kanoon line + page number, at end of page.
        pages_with_footer = 0
        for pg in pages:
            joined = "\n".join(pg)
            if title and title in joined and KANOON_RE.search(joined):
                pages_with_footer += 1
        if pages and pages_with_footer >= len(pages) - 1:
            footer_per_page_ok += 1

        # --- 4. Count boilerplate lines (footer title repeats, kanoon, page nums, neutral cite) ---
        for l in flat:
            s = l.strip()
            if not s:
                continue
            if KANOON_RE.search(s) or (title and s == title) or s == line2 \
               or PAGE_OF_RE.match(s) or PAGE_DASH_RE.match(s) or NEUTRAL_CIT_RE.search(s):
                boilerplate_lines += 1

        if not (title_ok and date_ok and m):
            failures.append((name, title_ok, date_ok, bool(m)))

    n = len(pdfs)
    print(f"Title header present (line 1):        {has_title_header}/{n}")
    print(f"'on <date>' second line:              {has_date_line}/{n}")
    print(f"Indian Kanoon link + doc_id:          {has_kanoon}/{n}")
    print(f"Footer on ~every page:                {footer_per_page_ok}/{n}")
    print(f"Unique Kanoon doc_ids:                {len(set(doc_ids.values()))}")
    print(f"Pages/doc: min={min(page_counts)} max={max(page_counts)} avg={sum(page_counts)/n:.1f}")
    print(f"Boilerplate lines: {boilerplate_lines}/{total_lines} "
          f"({100*boilerplate_lines/total_lines:.1f}% of all lines)")

    if failures:
        print("\nDocs failing one of the 3 core checks (title/date/kanoon):")
        for name, t, d, k in failures:
            print(f"  {name}: title={t} date={d} kanoon={k}")
    else:
        print("\nAll docs pass title + date + kanoon checks.")

    # --- 5. Other recurring tokens worth normalizing (sampled) ---
    print("\n" + "=" * 70 + "\nRecurring structural tokens (top, across corpus):")
    tok = Counter()
    for pdf in pdfs:
        flat, _ = load(pdf)
        for l in flat:
            s = l.strip()
            for key, rx in [
                ("CORAM line", re.compile(r"^CORAM", re.I)),
                ("...Appellant", re.compile(r"\.\.\.\s*Appellant", re.I)),
                ("...Respondent", re.compile(r"\.\.\.\s*Respondent", re.I)),
                ("Versus", re.compile(r"^Versus$", re.I)),
                ("Neutral Citation", NEUTRAL_CIT_RE),
                ("Reserved/Pronounced", re.compile(r"^(Reserved|Pronounced) on", re.I)),
                ("JUDGMENT heading", re.compile(r"^J ?U ?D ?G ?M ?E ?N ?T", re.I)),
                ("IN THE HIGH COURT", re.compile(r"IN THE HIGH COURT", re.I)),
            ]:
                if rx.search(s):
                    tok[key] += 1
    for k, v in tok.most_common():
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
