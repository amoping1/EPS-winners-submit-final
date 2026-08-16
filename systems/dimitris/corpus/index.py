"""index.py — an index over the frozen offline corpus.

The corpus is the primary evidence base: 1,139 first-party documents (filings,
call-transcript sections, slide decks) for the four challenge companies, frozen
2026-08-14. We do NOT fetch filings from SEC EDGAR — everything needed is here,
and the frozen set is what the organisers intend us to reason from.

Two facts drive the design:

  1. Every document carries YAML front-matter with `published_at`. That is the
     only safe field for a point-in-time cutoff -- `period` diverges from it
     (Home Depot has a doc published 2026-05-21 carrying period "Q2 2027"), so
     filtering on `period` leaks the future.

  2. `source_url` is frequently null. Cite the FILENAME, or the evidence trail
     breaks exactly where the rubric's "data approach" and "model quality"
     criteria look.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_ROOT = REPO_ROOT / "starter" / "challenge" / "offline-data"

# Company directory <-> ticker used by the challenge.
COMPANY_DIRS: dict[str, str] = {
    "HD": "home-depot",
    "ADI": "analog-devices",
    "HAS": "hays",
    "DE": "deere",
}
TICKERS = tuple(COMPANY_DIRS)

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FM_LINE = re.compile(r'^(\w+):\s*(?:"(.*)"|(.*))\s*$')
# 2026-05-19__hd-us-20260519-q1-8k-2__1038584.md
_FILENAME = re.compile(
    r"^(?P<pub>\d{4}-\d{2}-\d{2})__"
    r"(?P<ticker>[a-z]+)-(?P<market>[a-z]+)-(?P<fdate>\d{8})-"
    r"(?P<doctype>.+?)__(?P<docid>\d+)\.md$"
)


@dataclass(frozen=True)
class Document:
    path: Path
    ticker: str
    company: str
    published_at: date
    document_type: str  # FILING | CALL_TRANSCRIPT | SLIDES (from front-matter)
    period: str
    title: str
    doc_kind: str  # 10q | 10k | 8k | filing | ... (from the filename)
    doc_id: str
    source_url: str | None

    @property
    def rel(self) -> str:
        """Repo-relative path — this is the citation key."""
        return self.path.relative_to(REPO_ROOT).as_posix()

    @property
    def name(self) -> str:
        return self.path.name

    def text(self) -> str:
        return self.path.read_text(encoding="utf-8", errors="replace")

    def summary_line(self) -> str:
        return (
            f"{self.published_at.isoformat()} | {self.ticker} | "
            f"{self.doc_kind or self.document_type} | {self.period} | "
            f"{self.title[:70]} | {self.name}"
        )


def _parse_front_matter(text: str) -> dict[str, str]:
    m = _FRONT_MATTER.match(text)
    if not m:
        return {}
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        fm = _FM_LINE.match(line.strip())
        if fm:
            value = fm.group(2) if fm.group(2) is not None else (fm.group(3) or "")
            out[fm.group(1)] = value.strip()
    return out


def _first_heading(text: str) -> str:
    body = _FRONT_MATTER.sub("", text, count=1)
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line and not line.startswith("<!--"):
            return line[:120]
    return ""


def _load(path: Path) -> Document | None:
    # Read only the head: front-matter plus enough for a title. Full text is
    # loaded lazily via Document.text(). Indexing 102 MB eagerly is wasteful.
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        head = fh.read(4096)

    fm = _parse_front_matter(head)
    fn = _FILENAME.match(path.name)

    pub_raw = fm.get("published_at") or (fn.group("pub") if fn else "")
    try:
        published = date.fromisoformat(pub_raw)
    except ValueError:
        return None

    ticker = (fm.get("ticker") or "").upper().replace("LSE:", "")
    if ticker not in COMPANY_DIRS:
        # Fall back to the parent directory, which is authoritative.
        for tk, d in COMPANY_DIRS.items():
            if d in path.parts:
                ticker = tk
                break
    if ticker not in COMPANY_DIRS:
        return None

    src = fm.get("source_url", "")
    return Document(
        path=path,
        ticker=ticker,
        company=fm.get("company", ""),
        published_at=published,
        document_type=fm.get("document_type", ""),
        period=fm.get("period", ""),
        title=_first_heading(head),
        doc_kind=(fn.group("doctype") if fn else ""),
        doc_id=(fn.group("docid") if fn else ""),
        source_url=None if src in ("", "null", "None") else src,
    )


@lru_cache(maxsize=1)
def load_index() -> tuple[Document, ...]:
    """Index every corpus document. Cached for the process."""
    if not CORPUS_ROOT.exists():
        raise FileNotFoundError(
            f"Corpus not found at {CORPUS_ROOT}. Clone the starter repo first:\n"
            f"  git clone https://github.com/kernlai/agents-vs-wall-street-starter.git starter"
        )
    docs: list[Document] = []
    for company_dir in COMPANY_DIRS.values():
        root = CORPUS_ROOT / company_dir
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.md")):
            if path.name == "INDEX.md":
                continue
            doc = _load(path)
            if doc is not None:
                docs.append(doc)
    docs.sort(key=lambda d: (d.ticker, d.published_at), reverse=True)
    return tuple(docs)


def find(
    ticker: str | None = None,
    *,
    before: date | str | None = None,
    after: date | str | None = None,
    doc_kinds: tuple[str, ...] | None = None,
    document_type: str | None = None,
    period_contains: str | None = None,
    limit: int | None = None,
) -> list[Document]:
    """Filter the index. Newest first.

    `before` is the point-in-time cutoff and is applied to `published_at`,
    never to `period` -- see the module docstring for why.
    """
    if isinstance(before, str):
        before = date.fromisoformat(before)
    if isinstance(after, str):
        after = date.fromisoformat(after)

    out = []
    for d in load_index():
        if ticker and d.ticker != ticker.upper():
            continue
        if before and d.published_at >= before:
            continue
        if after and d.published_at < after:
            continue
        if doc_kinds and not any(k in d.doc_kind for k in doc_kinds):
            continue
        if document_type and d.document_type.upper() != document_type.upper():
            continue
        if period_contains and period_contains.lower() not in d.period.lower():
            continue
        out.append(d)
        if limit and len(out) >= limit:
            break
    return out


def get(name_or_path: str) -> Document:
    """Look a document up by filename, repo-relative path, or document id."""
    key = name_or_path.strip().replace("\\", "/")
    docs = load_index()
    for d in docs:
        if key in (d.name, d.rel, d.doc_id):
            return d
    # Tolerate a bare stem or a partial path.
    for d in docs:
        if key in d.rel:
            return d
    raise KeyError(f"No corpus document matching {name_or_path!r}")
