"""Corpus indexing and as_of-filtered retrieval over the frozen document set.

The `as_of` parameter is mandatory on every search, by design. Without it a backtest
retrieves the document that contains the answer, scores brilliantly, and tells us nothing.
See STRATEGY.md section 6.

Stdlib only - no install step, deterministic, fast.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# Filenames lead with the publication date: 2026-07-10__has-ln-...md
_FILENAME_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})__")
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FM_FIELD = re.compile(r'^(\w+):\s*"?(.*?)"?\s*$', re.MULTILINE)
_TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]*")


def _parse_date(value: str) -> date | None:
    try:
        y, m, d = value.strip().split("-")
        return date(int(y), int(m), int(d))
    except (ValueError, AttributeError):
        return None


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Document:
    """One corpus document plus the provenance fields judges expect us to carry."""

    doc_id: str
    path: Path
    company: str
    ticker: str
    published_at: date | None
    document_type: str
    period: str
    source_url: str
    _body: str | None = field(default=None, repr=False)

    @property
    def body(self) -> str:
        if self._body is None:
            raw = self.path.read_text(encoding="utf-8", errors="replace")
            self._body = _FRONTMATTER.sub("", raw, count=1)
        return self._body

    def cite(self) -> dict:
        """Provenance stamp. Attach to every extracted value."""
        return {
            "doc_id": self.doc_id,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "document_type": self.document_type,
            "period": self.period,
            "source_url": self.source_url,
        }


def _read_document(path: Path) -> Document | None:
    head = path.read_text(encoding="utf-8", errors="replace")[:2048]
    match = _FRONTMATTER.search(head)
    fields = dict(_FM_FIELD.findall(match.group(1))) if match else {}

    published = _parse_date(fields.get("published_at", ""))
    if published is None:
        # Fall back to the filename date rather than dropping the document.
        fn = _FILENAME_DATE.match(path.name)
        if fn:
            published = date(int(fn.group(1)), int(fn.group(2)), int(fn.group(3)))

    return Document(
        doc_id=path.stem,
        path=path,
        company=fields.get("company", ""),
        ticker=fields.get("ticker", ""),
        published_at=published,
        document_type=fields.get("document_type", ""),
        period=fields.get("period", ""),
        source_url=fields.get("source_url", ""),
    )


class Corpus:
    """BM25 index over one company's documents, with a hard as_of cutoff on every query."""

    K1 = 1.5
    B = 0.75

    def __init__(self, root: Path, company_dir: str):
        self.root = Path(root)
        self.company_dir = company_dir
        self.docs: list[Document] = []
        self._tf: list[Counter] = []
        self._len: list[int] = []
        self._df: Counter = Counter()
        self._avg_len: float = 0.0
        self._built = False

    def build(self) -> "Corpus":
        base = self.root / self.company_dir
        if not base.is_dir():
            raise FileNotFoundError(
                f"Corpus dir not found: {base}\n"
                "Set corpus.root in config/settings.json to the organizer "
                "challenge/offline-data directory."
            )

        for path in sorted(base.rglob("*.md")):
            if path.name == "INDEX.md":
                continue
            doc = _read_document(path)
            if doc is not None:
                self.docs.append(doc)

        for doc in self.docs:
            tokens = tokenize(doc.body)
            tf = Counter(tokens)
            self._tf.append(tf)
            self._len.append(len(tokens))
            self._df.update(tf.keys())

        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0
        self._built = True
        return self

    def search(
        self,
        query: str,
        as_of: date | str,
        limit: int = 8,
        document_type: str | None = None,
    ) -> list[tuple[Document, float]]:
        """Top-`limit` documents for `query`, excluding anything published after `as_of`.

        `as_of` is required. Passing the real forecast date for a live run and a historical
        date for a backtest is what makes the two comparable.
        """
        if not self._built:
            raise RuntimeError("Call build() before search().")
        if isinstance(as_of, str):
            parsed = _parse_date(as_of)
            if parsed is None:
                raise ValueError(f"as_of must be YYYY-MM-DD, got {as_of!r}")
            as_of = parsed

        q_tokens = [t for t in tokenize(query) if t in self._df]
        if not q_tokens:
            return []

        n = len(self.docs)
        idf = {
            t: math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5)) for t in q_tokens
        }

        scored: list[tuple[Document, float]] = []
        for i, doc in enumerate(self.docs):
            # The guard. Unknown dates are excluded rather than trusted.
            if doc.published_at is None or doc.published_at > as_of:
                continue
            if document_type and doc.document_type.upper() != document_type.upper():
                continue

            tf, dl = self._tf[i], self._len[i]
            score = 0.0
            for t in q_tokens:
                f = tf.get(t, 0)
                if not f:
                    continue
                denom = f + self.K1 * (1 - self.B + self.B * dl / (self._avg_len or 1))
                score += idf[t] * (f * (self.K1 + 1)) / denom
            if score > 0:
                scored.append((doc, score))

        scored.sort(key=lambda pair: (-pair[1], pair[0].published_at or date.min))
        return scored[:limit]

    def latest(self, as_of: date | str, document_type: str | None = None) -> Document | None:
        """Most recent document at or before `as_of`. Useful for 'what did they last guide?'"""
        if isinstance(as_of, str):
            as_of = _parse_date(as_of)
        candidates = [
            d
            for d in self.docs
            if d.published_at
            and d.published_at <= as_of
            and (not document_type or d.document_type.upper() == document_type.upper())
        ]
        return max(candidates, key=lambda d: d.published_at) if candidates else None


def load_settings(path: str | Path = "config/settings.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
