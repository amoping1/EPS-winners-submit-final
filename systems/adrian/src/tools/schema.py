"""Agent-facing tool layer.

Turns the Python toolbelt into something a model can actually call: JSON-serializable
returns, hard token ceilings, descriptions that carry the traps, and errors returned as
text so a failed call teaches the agent to retry differently instead of killing the run.

The descriptions matter as much as the code. Everything the agent needs to know about
this corpus - that `period_label` lies, that `as_of` is enforced - has to reach the model
here, because nothing else does.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date

from .documents import DocumentTools

# ~4 chars per token. Caps are deliberately tight: one uncapped 10-Q is ~52k tokens and
# would evict the agent's working context in a single call.
MAX_DOC_CHARS = 8_000
MAX_TABLE_CHARS = 6_000
MAX_RESULTS = 25


def _jsonable(obj):
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "list_index",
            "description": (
                "Browse this company's document catalog, newest first. ALWAYS TRY THIS "
                "FIRST - it is cheaper and more precise than search when you know what "
                "you want (e.g. 'the Q2 earnings release', 'the most recent 10-Q'). "
                "IMPORTANT: the catalog's 'period_label' field is unreliable - it "
                "disagrees with the filename on 76 of 1139 documents, and the errors "
                "cluster on recent earnings documents. Trust 'period_from_slug' instead, "
                "and treat 'period_conflict': true as a warning to verify against the "
                "document body. Documents published after as_of are never returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_type": {
                        "type": "string",
                        "description": "Filing | Call Transcript | Slide",
                    },
                    "period": {"type": "string", "description": "e.g. 'Q2 2026', 'FY 2026'"},
                    "title_contains": {"type": "string"},
                    "since": {"type": "string", "description": "YYYY-MM-DD lower bound"},
                    "limit": {"type": "integer", "default": 15},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "BM25 full-text search across this company's documents. Use when "
                "list_index is not specific enough - e.g. hunting a phrase like "
                "'company compiled consensus' or 'comparable sales'. Returns ranked "
                "documents with provenance, not text; follow up with read_document. "
                "Documents published after as_of are never returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 8},
                    "doc_type": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "Read a document. ALWAYS pass 'contains' with a distinctive phrase - "
                "filings run to 200,000+ characters and an unfocused read wastes most of "
                "your context. The response is capped and reports whether the phrase was "
                "found. If matched is false, try a different phrase rather than reading "
                "more."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "contains": {
                        "type": "string",
                        "description": "Phrase to centre the excerpt on. Strongly recommended.",
                    },
                    "window": {"type": "integer", "default": 2500},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_table",
            "description": (
                "Pull financial tables out of a filing, cleaned and ranked. Segment "
                "operating profit, income-statement lines and margin tables live in "
                "tables, not prose - use this rather than read_document for any figure "
                "you expect in a table. Pass 'near' with the row or section label you "
                "want (e.g. 'Production & Precision Ag'); tables containing that phrase "
                "rank first. Returns [] if the phrase appears in no table."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "near": {"type": "string"},
                    "limit": {"type": "integer", "default": 2},
                },
                "required": ["doc_id"],
            },
        },
    },
]

TOOL_NAMES = [spec["function"]["name"] for spec in TOOL_SPECS]


def dispatch(name: str, args: dict, tools: DocumentTools) -> dict:
    """Execute one tool call. Never raises - errors come back as data the agent can read."""
    try:
        if name == "list_index":
            entries = tools.list_index(
                doc_type=args.get("doc_type"),
                period=args.get("period"),
                title_contains=args.get("title_contains"),
                since=args.get("since"),
                limit=min(int(args.get("limit", 15)), MAX_RESULTS),
            )
            return {
                "count": len(entries),
                "documents": [
                    {
                        "doc_id": e.doc_id,
                        "published_at": e.published_at.isoformat(),
                        "doc_type": e.doc_type,
                        "title": e.title,
                        "period_label": e.period_label,
                        "period_from_slug": e.period_from_slug,
                        "period_conflict": e.period_conflict,
                    }
                    for e in entries
                ],
            }

        if name == "search":
            hits = tools.search(
                args["query"],
                limit=min(int(args.get("limit", 8)), MAX_RESULTS),
                doc_type=args.get("doc_type"),
            )
            return {
                "count": len(hits),
                "documents": [
                    {
                        "doc_id": d.doc_id,
                        "published_at": d.published_at.isoformat() if d.published_at else None,
                        "doc_type": d.document_type,
                        "score": round(score, 2),
                        "source_url": d.source_url,
                    }
                    for d, score in hits
                ],
            }

        if name == "read_document":
            result = tools.read_document(
                args["doc_id"],
                contains=args.get("contains"),
                window=int(args.get("window", 2500)),
            )
            text = result["text"]
            if len(text) > MAX_DOC_CHARS:
                text = text[:MAX_DOC_CHARS] + "\n...[truncated - narrow with 'contains']"
            result["text"] = text
            if result.get("matched") is False:
                result["hint"] = (
                    "Phrase not found. Showing the start of the document. "
                    "Try a different phrase, or extract_table if you expect a table."
                )
            return _jsonable(result)

        if name == "extract_table":
            tables = tools.extract_table(
                args["doc_id"],
                near=args.get("near"),
                limit=min(int(args.get("limit", 2)), 5),
            )
            for t in tables:
                if len(t["table"]) > MAX_TABLE_CHARS:
                    t["table"] = t["table"][:MAX_TABLE_CHARS] + "\n...[truncated]"
            if not tables:
                return {
                    "count": 0,
                    "tables": [],
                    "hint": (
                        "No table contained that phrase. Try a shorter or different "
                        "label, or drop 'near' to see the most numeric tables."
                    ),
                }
            return {"count": len(tables), "tables": _jsonable(tables)}

        return {"error": f"Unknown tool {name!r}. Available: {', '.join(TOOL_NAMES)}"}

    except PermissionError as exc:
        # as_of violation - a real guard firing, not a crash.
        return {"error": str(exc), "kind": "as_of_violation"}
    except FileNotFoundError as exc:
        return {"error": str(exc), "kind": "not_found",
                "hint": "Use list_index or search to get a valid doc_id."}
    except Exception as exc:  # noqa: BLE001 - the agent should see any failure as text
        return {"error": f"{type(exc).__name__}: {exc}", "kind": "tool_error"}


def dispatch_json(name: str, args: dict, tools: DocumentTools) -> str:
    return json.dumps(dispatch(name, args, tools), ensure_ascii=False)
