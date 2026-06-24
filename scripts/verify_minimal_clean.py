"""Verify the minimal, 100%-reliable extraction + cleanup on ALL 56 docs.

Scope (deliberately conservative — only the furniture we proved is reliable):
  - per-page fixed HEADER  = running title line  -> extract title + date, strip it
  - per-page FOOTER        = 'Indian Kanoon - <url>' line -> extract url + id, strip it
  - the PAGE NUMBER line immediately AFTER the Kanoon line -> strip it
Nothing else is touched (no neutral-citation / 'N of M' guessing).

Checks that {doc_id, title, date, kanoon_url, kanoon_id} are extracted for 56/56.
Run: PYTHONPATH=. uv run python scripts/verify_minimal_clean.py
"""
from __future__ import annotations

import re

import pymupdf

from src.config import settings

KANOON_RE = re.compile(r"(Indian Kanoon\s*-\s*(http://indiankanoon\.org/doc/(\d+)/))")
DATE_RE = re.compile(r"\bon (\d{1,2} [A-Z][a-z]+,? \d{4})\b")
PAGE_NUM_RE = re.compile(r"^\s*-?\s*\d{1,4}\s*-?\s*$")


def extract_and_clean(flat: list[str], doc_id: str, source: str):
    """Return (metadata, cleaned_text)."""
    kanoon_idxs = [i for i, l in enumerate(flat) if KANOON_RE.search(l)]

    # --- footer: url + id (from the first Kanoon line) ---
    m = KANOON_RE.search(flat[kanoon_idxs[0]]) if kanoon_idxs else None
    kanoon_url = m.group(2) if m else ""
    kanoon_id = m.group(3) if m else ""

    # --- header: running title = the non-empty line right before a Kanoon line.
    # Take the most common such line (robust against the opening caption page). ---
    from collections import Counter
    cand = Counter()
    for k in kanoon_idxs:
        for j in range(k - 1, -1, -1):
            if flat[j].strip():
                cand[flat[j].strip()] += 1
                break
    title = cand.most_common(1)[0][0] if cand else ""

    # --- date: parse 'on <DATE>' out of the running title ---
    d = DATE_RE.search(title)
    date = d.group(1) if d else ""

    # --- strip the three furniture lines per page ---
    drop = set()
    for k in kanoon_idxs:
        drop.add(k)                                   # the Kanoon line
        if k + 1 < len(flat) and PAGE_NUM_RE.match(flat[k + 1].strip()):
            drop.add(k + 1)                           # page number after it
        # the running-title line immediately before it
        for j in range(k - 1, -1, -1):
            if flat[j].strip():
                if flat[j].strip() == title:
                    drop.add(j)
                break
    cleaned = "\n".join(l for i, l in enumerate(flat) if i not in drop)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    meta = {
        "doc_id": doc_id, "source": source, "title": title,
        "date": date, "kanoon_url": kanoon_url, "kanoon_id": kanoon_id,
    }
    return meta, cleaned


def main():
    pdfs = sorted(settings.raw_dir.glob("*.pdf"))
    fields = ["doc_id", "title", "date", "kanoon_url", "kanoon_id"]
    ok = {f: 0 for f in fields}
    ids = set()
    removed_chars = total_chars = 0
    failures = []

    for pdf in pdfs:
        with pymupdf.open(pdf) as f:
            flat = [l for p in f for l in p.get_text().splitlines()]
        raw = "\n".join(flat)
        meta, cleaned = extract_and_clean(flat, pdf.stem.upper(), pdf.name)
        for fld in fields:
            if meta[fld]:
                ok[fld] += 1
        ids.add(meta["kanoon_id"])
        total_chars += len(raw)
        removed_chars += len(raw) - len(cleaned)
        if not all(meta[f] for f in fields):
            failures.append((pdf.stem, {f: meta[f] for f in fields if not meta[f]}))

    n = len(pdfs)
    print(f"Reliability across {n} docs:")
    for fld in fields:
        flag = "OK " if ok[fld] == n else "!! "
        print(f"  {flag}{fld:12s} {ok[fld]}/{n}")
    print(f"  unique kanoon_id: {len(ids - {''})}")
    print(f"  furniture removed: {removed_chars} chars "
          f"({100*removed_chars/total_chars:.2f}% of corpus)")
    if failures:
        print("\nFAILURES:")
        for stem, miss in failures:
            print(f"  {stem}: missing {miss}")
    else:
        print("\n=> 100% reliable for all 5 fields.")

    # show one full example
    with pymupdf.open(pdfs[0]) as f:
        flat = [l for p in f for l in p.get_text().splitlines()]
    meta, cleaned = extract_and_clean(flat, pdfs[0].stem.upper(), pdfs[0].name)
    print("\nExample metadata (doc_001):")
    for k, v in meta.items():
        print(f"  {k:12s}: {v}")


if __name__ == "__main__":
    main()
