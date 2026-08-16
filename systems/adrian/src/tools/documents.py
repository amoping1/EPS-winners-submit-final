"""Agent-facing document tools.

These are what the research agent calls. `list_index` first (cheap, precise, reads the
organizer's pre-built catalog), then `search` (BM25) when the catalog is not enough, then
`read_document` / `extract_table` to pull actual numbers.

Every return value carries provenance. Nothing here invents a fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .corpus import Corpus, _parse_date

# INDEX.md rows: | Published | Type | Period | Title | [Open](relative/path.md) |
_INDEX_ROW = re.compile(
    r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*([^|]+?)\s*\|"
    r"\s*\[Open\]\(([^)]+)\)\s*\|",
    re.MULTILINE,
)

# Filename slug carries the *reliable* period: 2026-05-20__adi-us-20260520-q2-10q__x.md
_SLUG_PERIOD = re.compile(r"-(q[1-4]|fy|h[12])-", re.IGNORECASE)

_ZERO_WIDTH = "​‌‍﻿"


def _clean_table(rows: list[str]) -> list[str]:
    """Strip zero-width padding and collapse spacer columns.

    Filing tables interleave real values with empty cells and zero-width spaces, which
    makes them unreadable to a model and inflates token cost. Drop the empties and the
    separator rows, keep rows that still carry content.
    """
    out = []
    for row in rows:
        cells = [
            c.strip().strip(_ZERO_WIDTH).strip() for c in row.split("|")[1:-1]
        ]
        cells = [c for c in cells if c and not set(c) <= set("-: ")]
        if len(cells) >= 2:
            out.append("| " + " | ".join(cells) + " |")
    return out if len(out) >= 2 else []


@dataclass
class IndexEntry:
    """One row of a company's INDEX.md catalog."""

    published_at: date
    doc_type: str
    period_label: str      # organizer metadata - UNRELIABLE, see period_from_slug
    title: str
    rel_path: str
    doc_id: str

    @property
    def period_from_slug(self) -> str | None:
        """Period parsed from the filename slug.

        The organizer `period` column is wrong on real documents - ADI's Q2 10-Q is
        labelled "Q3 2026", Deere's Q2 earnings call is labelled "Q3 2026". The filename
        slug (`-q2-10q-`) matches the actual content. Prefer this, and flag disagreement.
        """
        m = _SLUG_PERIOD.search(self.doc_id)
        return m.group(1).upper() if m else None

    @property
    def period_conflict(self) -> bool:
        """True when the catalog label disagrees with the filename slug. Worth flagging."""
        slug = self.period_from_slug
        if not slug or not self.period_label:
            return False
        if slug.startswith("Q"):
            return slug not in self.period_label.upper().replace(" ", "")
        return False


class DocumentTools:
    """The research agent's document toolbelt for one company."""

    def __init__(self, corpus_root: str | Path, company_dir: str, as_of: date | str):
        self.root = Path(corpus_root)
        self.company_dir = company_dir
        self.base = self.root / company_dir
        self.as_of = _parse_date(as_of) if isinstance(as_of, str) else as_of
        if self.as_of is None:
            raise ValueError("as_of is required and must be YYYY-MM-DD")
        self._entries: list[IndexEntry] | None = None
        self._corpus: Corpus | None = None

    # ---------------------------------------------------------------- catalog

    def _load_index(self) -> list[IndexEntry]:
        if self._entries is not None:
            return self._entries
        text = (self.base / "INDEX.md").read_text(encoding="utf-8", errors="replace")
        entries = []
        for published, doc_type, period, title, rel in _INDEX_ROW.findall(text):
            pub = _parse_date(published)
            if pub is None:
                continue
            entries.append(
                IndexEntry(
                    published_at=pub,
                    doc_type=doc_type.strip(),
                    period_label=period.strip(),
                    title=title.strip(),
                    rel_path=rel.strip(),
                    doc_id=Path(rel.strip()).stem,
                )
            )
        self._entries = entries
        return entries

    def list_index(
        self,
        doc_type: str | None = None,
        period: str | None = None,
        title_contains: str | None = None,
        since: date | str | None = None,
        limit: int = 25,
    ) -> list[IndexEntry]:
        """Browse the company catalog, newest first, never past `as_of`.

        Cheaper and more precise than search when you know what you want -
        "the Q2 earnings release", "the most recent 10-Q".
        """
        if isinstance(since, str):
            since = _parse_date(since)

        out = []
        for e in self._load_index():
            if e.published_at > self.as_of:
                continue                                    # the as_of guard
            if since and e.published_at < since:
                continue
            if doc_type and doc_type.lower() not in e.doc_type.lower():
                continue
            if period and period.upper().replace(" ", "") not in (
                (e.period_label + " " + (e.period_from_slug or "")).upper().replace(" ", "")
            ):
                continue
            if title_contains and title_contains.lower() not in e.title.lower():
                continue
            out.append(e)
        out.sort(key=lambda e: e.published_at, reverse=True)
        return out[:limit]

    # ---------------------------------------------------------------- search

    def search(self, query: str, limit: int = 8, doc_type: str | None = None):
        """BM25 fallback when the catalog is not specific enough. as_of enforced."""
        if self._corpus is None:
            self._corpus = Corpus(self.root, self.company_dir).build()
        return self._corpus.search(
            query, as_of=self.as_of, limit=limit, document_type=doc_type
        )

    # ---------------------------------------------------------------- reading

    def read_document(
        self, doc_id: str, contains: str | None = None, window: int = 2500
    ) -> dict:
        """Full text, or the region around `contains`. Always returns provenance."""
        path = self._resolve(doc_id)
        raw = path.read_text(encoding="utf-8", errors="replace")
        fm, body = self._split_frontmatter(raw)

        excerpt, matched = body, None
        if contains:
            idx = body.lower().find(contains.lower())
            if idx >= 0:
                start = max(0, idx - window // 2)
                excerpt = body[start : start + window]
                matched = True
            else:
                excerpt = body[:window]
                matched = False

        return {
            "doc_id": doc_id,
            "published_at": fm.get("published_at"),
            "document_type": fm.get("document_type"),
            "period_label": fm.get("period"),
            "source_url": fm.get("source_url"),
            "matched": matched,
            "text": excerpt,
            "truncated": len(excerpt) < len(body),
        }

    def extract_table(self, doc_id: str, near: str | None = None, limit: int = 3) -> list[dict]:
        """Markdown tables from a filing, cleaned and ranked by relevance.

        Segment operating profit and income-statement lines live in tables, not prose.
        Filing tables are heavily padded with zero-width spaces and empty spacer cells,
        and the phrase you are looking for usually sits *inside* the table rather than
        near it - so rank by content, not by distance.
        """
        path = self._resolve(doc_id)
        raw = path.read_text(encoding="utf-8", errors="replace")
        fm, body = self._split_frontmatter(raw)

        tables, current = [], []
        for line in body.splitlines():
            if line.lstrip().startswith("|"):
                current.append(line.rstrip())
            elif current:
                if len(current) >= 2:
                    tables.append(current)
                current = []
        if len(current) >= 2:
            tables.append(current)

        scored = []
        for rows in tables:
            cleaned = _clean_table(rows)
            if not cleaned:
                continue
            text = "\n".join(cleaned)
            numbers = len(re.findall(r"\d[\d,]*\.?\d*", text))
            score = float(numbers)
            if near:
                if near.lower() in text.lower():
                    score += 1000.0          # the anchor is inside this table
                else:
                    score -= 500.0
            scored.append((score, text))

        scored.sort(key=lambda pair: -pair[0])
        if near:
            scored = [pair for pair in scored if pair[0] > 0]

        return [
            {
                "doc_id": doc_id,
                "published_at": fm.get("published_at"),
                "source_url": fm.get("source_url"),
                "table": text,
            }
            for _, text in scored[:limit]
        ]

    # ---------------------------------------------------------------- internals

    def _resolve(self, doc_id: str) -> Path:
        for entry in self._load_index():
            if entry.doc_id == doc_id:
                path = self.base / entry.rel_path
                if path.exists():
                    self._assert_as_of(entry.published_at, doc_id)
                    return path
        hits = list(self.base.rglob(f"{doc_id}.md"))
        if not hits:
            raise FileNotFoundError(f"No document {doc_id!r} for {self.company_dir}")
        return hits[0]

    def _assert_as_of(self, published: date, doc_id: str) -> None:
        if published > self.as_of:
            raise PermissionError(
                f"as_of violation: {doc_id} published {published} > as_of {self.as_of}. "
                "Reading it would leak the answer into a backtest."
            )

    @staticmethod
    def _split_frontmatter(raw: str) -> tuple[dict, str]:
        m = re.match(r"\A---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
        if not m:
            return {}, raw
        fields = dict(re.findall(r'^(\w+):\s*"?(.*?)"?\s*$', m.group(1), re.MULTILINE))
        return fields, raw[m.end() :]
