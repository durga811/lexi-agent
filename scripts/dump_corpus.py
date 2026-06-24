"""Dump the cleaned corpus to one text file per judgment for human/agent review.

Writes scratchpad files: DOC_xxx.txt with the metadata header + full cleaned text.
Used to build the gold set: a reader can scan every judgment's facts + holding
without re-parsing PDFs.
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.ingest.parse import load_documents

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("corpus_dump")
OUT.mkdir(parents=True, exist_ok=True)

docs = load_documents()
index_lines = []
for d in docs:
    p = OUT / f"{d['doc_id']}.txt"
    header = (
        f"doc_id: {d['doc_id']}\n"
        f"title: {d['title']}\n"
        f"date: {d['date']}\n"
        f"source: {d['source']}\n"
        f"kanoon_url: {d['kanoon_url']}\n"
        f"{'=' * 70}\n"
    )
    p.write_text(header + d["text"])
    index_lines.append(f"{d['doc_id']} | {d['title']} | {len(d['text'])} chars")

(OUT / "_INDEX.txt").write_text("\n".join(index_lines))
print(f"Wrote {len(docs)} docs to {OUT.resolve()}")
print("\n".join(index_lines))
