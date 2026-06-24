"""PDF -> cleaned plain text + metadata. One dict per judgment, keyed by doc_id.

The raw Indian Kanoon PDFs carry fixed page furniture on every page: a running
HEADER (the case title) and a FOOTER (the Indian Kanoon source URL followed by a
bare page number). We strip exactly that furniture — the only boilerplate proven
100% reliable across all 56 docs — and lift four fields out of it as metadata:
title, date, kanoon_url, kanoon_id. Nothing else is touched (neutral-citation
watermarks etc. are left in place to avoid corrupting body text).
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import pymupdf

from src.config import settings

# Footer: "Indian Kanoon - http://indiankanoon.org/doc/<id>/"
KANOON_RE = re.compile(
    r"(Indian Kanoon\s*-\s*(http://indiankanoon\.org/doc/(\d+)/))"
)
# Case date embedded in the running title: "... on 6 November, 2023"
DATE_RE = re.compile(r"\bon (\d{1,2} [A-Z][a-z]+,? \d{4})\b")
# A bare page-number line: "1", "-1-"
PAGE_NUM_RE = re.compile(r"^\s*-?\s*\d{1,4}\s*-?\s*$")


def _extract_and_clean(flat: list[str]) -> tuple[dict, str]:
    """Pull (title, date, kanoon_url, kanoon_id) and strip the page furniture.

    Returns ({title, date, kanoon_url, kanoon_id}, cleaned_text).
    """
    kanoon_idxs = [i for i, l in enumerate(flat) if KANOON_RE.search(l)]

    # --- footer -> url + id (from the first Kanoon line) ---
    first = KANOON_RE.search(flat[kanoon_idxs[0]]) if kanoon_idxs else None
    kanoon_url = first.group(2) if first else ""
    kanoon_id = first.group(3) if first else ""

    # --- header -> running title = most common line preceding a Kanoon line.
    # (Taking the mode makes this robust to the opening-caption page.) ---
    cand: Counter[str] = Counter()
    for k in kanoon_idxs:
        for j in range(k - 1, -1, -1):
            if flat[j].strip():
                cand[flat[j].strip()] += 1
                break
    title = cand.most_common(1)[0][0] if cand else ""

    # --- date parsed out of the running title ---
    d = DATE_RE.search(title)
    date = d.group(1) if d else ""

    # --- strip the furniture: title line + Kanoon line + the page# after it ---
    drop: set[int] = set()
    for k in kanoon_idxs:
        drop.add(k)
        if k + 1 < len(flat) and PAGE_NUM_RE.match(flat[k + 1].strip()):
            drop.add(k + 1)
        for j in range(k - 1, -1, -1):
            if flat[j].strip():
                if flat[j].strip() == title:
                    drop.add(j)
                break

    cleaned = "\n".join(l for i, l in enumerate(flat) if i not in drop)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    meta = {
        "title": title,
        "date": date,
        "kanoon_url": kanoon_url,
        "kanoon_id": kanoon_id,
    }
    return meta, cleaned


@lru_cache(maxsize=1)
def load_documents() -> list[dict]:
    """Return [{doc_id, source, title, date, kanoon_url, kanoon_id, text}] per PDF.

    Cached: the PDFs never change at runtime, and both the BM25 retriever and the
    get_document tool call this — no point re-parsing 56 PDFs each time.
    """
    docs: list[dict] = []
    for pdf in sorted(settings.raw_dir.glob("*.pdf")):
        with pymupdf.open(pdf) as f:
            flat = [l for page in f for l in page.get_text().splitlines()]
        meta, text = _extract_and_clean(flat)
        docs.append(
            {
                "doc_id": pdf.stem.upper(),  # normalise: DOC_001
                "source": pdf.name,
                **meta,
                "text": text,
            }
        )
    return docs
