"""reader.py — navigate a corpus document without dumping it into context.

Documents range from a few KB to several hundred. Reading one whole is both
wasteful and self-defeating: the agent loses the thread. So the reader exposes
the same moves a human makes — outline, jump, search, pull the table — and
every return value is bounded and carries its citation.

Document shape (verified against the real corpus):
  - YAML front-matter
  - a plain-text rendering of the document
  - a `<!-- PAGE n -->` body with markdown headings and clean pipe tables

The pipe tables are the reason numbers do not need to be read by an LLM. A
statement line looks like:
  | Net sales | $ 41,765 | $ 39,856 | 4.8% |
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .index import Document, get

_HEADING = re.compile(r"^(#{1,4})\s+(.*?)\s*$")
_PAGE = re.compile(r"^<!--\s*PAGE\s+(\d+)\s*-->\s*$", re.IGNORECASE)
_TABLE_SEP = re.compile(r"^\|[\s:|-]+\|\s*$")
# Numbers as filings write them: 41,765  (3.0)  $ 1,234.5  4.8%  (1,234)
_NUMBER = re.compile(r"\(?\$?\s?-?\d[\d,]*\.?\d*\s?%?\)?")


@dataclass(frozen=True)
class Section:
    index: int
    level: int
    title: str
    start: int  # 1-based, inclusive
    end: int  # 1-based, inclusive
    chars: int

    def label(self) -> str:
        indent = "  " * (self.level - 1)
        return (
            f"[{self.index}] {indent}{self.title[:80]} "
            f"(lines {self.start}-{self.end}, {self.chars:,} chars)"
        )


def _lines(doc: Document) -> list[str]:
    return doc.text().splitlines()


def outline(name: str, max_level: int = 3) -> tuple[Document, list[Section]]:
    """Headings with line ranges, so an agent can choose where to look."""
    doc = get(name)
    lines = _lines(doc)

    marks: list[tuple[int, int, str]] = []  # (line_no, level, title)
    for i, line in enumerate(lines, start=1):
        h = _HEADING.match(line)
        if h and len(h.group(1)) <= max_level:
            title = h.group(2).strip()
            if title:
                marks.append((i, len(h.group(1)), title))
            continue
        p = _PAGE.match(line)
        if p:
            marks.append((i, max_level, f"(page {p.group(1)})"))

    if not marks:
        return doc, [
            Section(0, 1, "(whole document)", 1, len(lines), len(doc.text()))
        ]

    sections: list[Section] = []
    if marks[0][0] > 1:
        head = "\n".join(lines[: marks[0][0] - 1])
        sections.append(Section(0, 1, "(front matter / preamble)", 1, marks[0][0] - 1, len(head)))

    for n, (line_no, level, title) in enumerate(marks):
        end = marks[n + 1][0] - 1 if n + 1 < len(marks) else len(lines)
        body = "\n".join(lines[line_no - 1 : end])
        sections.append(Section(len(sections), level, title, line_no, end, len(body)))
    return doc, sections


def read_lines(name: str, start: int = 1, count: int = 200) -> tuple[Document, str]:
    """Read an explicit line window. 1-based, inclusive."""
    doc = get(name)
    lines = _lines(doc)
    start = max(1, start)
    end = min(len(lines), start + max(1, count) - 1)
    body = "\n".join(lines[start - 1 : end])
    header = f"{doc.name} lines {start}-{end} of {len(lines)}"
    return doc, f"{header}\n\n{body}"


def read_section(name: str, section_index: int) -> tuple[Document, str]:
    """Read one section from `outline()`."""
    doc, sections = outline(name)
    match = next((s for s in sections if s.index == section_index), None)
    if match is None:
        available = ", ".join(str(s.index) for s in sections)
        raise KeyError(f"No section {section_index} in {doc.name}. Available: {available}")
    _, body = read_lines(name, match.start, match.end - match.start + 1)
    return doc, body


def search_in_document(
    name: str, query: str, context: int = 2, max_hits: int = 20
) -> tuple[Document, list[tuple[int, str]]]:
    """Case-insensitive search inside one document.

    `query` may be a regex; a bad pattern degrades to a literal search rather
    than raising, because the caller is usually a model.
    """
    doc = get(name)
    lines = _lines(doc)
    try:
        pat = re.compile(query, re.IGNORECASE)
    except re.error:
        pat = re.compile(re.escape(query), re.IGNORECASE)

    hits: list[tuple[int, str]] = []
    for i, line in enumerate(lines, start=1):
        if pat.search(line):
            lo = max(1, i - context)
            hi = min(len(lines), i + context)
            hits.append((i, "\n".join(lines[lo - 1 : hi])))
            if len(hits) >= max_hits:
                break
    return doc, hits


def tables(name: str, contains: str | None = None, max_tables: int = 6) -> tuple[Document, list[str]]:
    """Extract markdown pipe tables, optionally only those mentioning `contains`.

    Financial statements in this corpus are clean pipe tables, so this is
    deterministic extraction — no model reads the digits.
    """
    doc = get(name)
    lines = _lines(doc)

    found: list[str] = []
    block: list[str] = []
    for line in lines + [""]:
        if line.lstrip().startswith("|"):
            block.append(line)
            continue
        if block:
            # A real table needs a header separator row.
            if len(block) >= 3 and any(_TABLE_SEP.match(b.strip()) for b in block):
                text = "\n".join(block)
                if contains is None or contains.lower() in text.lower():
                    found.append(text)
                    if len(found) >= max_tables:
                        break
            block = []
    return doc, found


def numbers_near(name: str, query: str, context: int = 1, max_hits: int = 15) -> list[dict]:
    """Lines matching `query` plus the numbers on them — search leads, not truth.

    `starter/README.md` is blunt about this: results are "leads, not verified
    financial history". Always confirm in the cited document, and keep
    reported / adjusted / quarterly / annual values apart.
    """
    _, hits = search_in_document(name, query, context=context, max_hits=max_hits)
    out = []
    for line_no, excerpt in hits:
        nums = [n.strip() for n in _NUMBER.findall(excerpt) if any(c.isdigit() for c in n)]
        out.append({"line": line_no, "excerpt": excerpt, "numbers": nums})
    return out
