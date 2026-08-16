"""make_entry.py — generate entry.json from the live system, not from memory.

ENTRY.md requires a private entry record naming the agent, every team member,
the technical setup and the exact version used for the final run. `finalCommit`,
`repositoryUrl`, `primaryModels` and `preExistingComponents` are all facts about
the repository as it stands, so they are read from it rather than typed. A
hand-written entry drifts from the system it describes; a generated one cannot.

RULES.md also makes an incomplete or false declaration no protection against the
pre-made-work disqualification, so honesty here is not optional.

    python -m submission.make_entry --name "Dimitris Koutsoumpos" --email dimknaf@gmail.com
    python -m submission.make_entry --check          # validate an existing entry.json

entry.json is gitignored: it carries email addresses.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STARTER = REPO_ROOT / "starter"
ENTRY_PATH = STARTER / "entry.json"
TEMPLATE_PATH = STARTER / "entry.template.json"

VALID_BUILD_STYLES = ("headless-agent", "coding-harness", "hybrid", "other")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return ""


def detect_models() -> list[str]:
    """The models the final system actually uses, from live config."""
    from agent_core.config import _LLM_PROFILES, settings

    used = {settings.llm_profile, settings.bulk_profile}
    return sorted({_LLM_PROFILES[p]["model"].split("/", 1)[-1] for p in used if p in _LLM_PROFILES})


def detect_pre_existing() -> list[str]:
    """Public libraries that existed before the event. Declared, per RULES.md."""
    req = REPO_ROOT / "requirements.txt"
    out: list[str] = []
    if req.exists():
        for line in req.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    # The harness itself is pre-existing and must be declared.
    out.append("Claude Code (generic coding harness, unmodified)")
    return out


def detect_repo_url() -> str:
    url = _git("config", "--get", "remote.origin.url")
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split(":", 1)[1]
    return url.removesuffix(".git")


def build_entry(
    agent_name: str,
    description: str,
    members: list[dict[str, str]],
    final_command: str,
    build_style: str,
    email_confirmed: bool,
) -> dict:
    return {
        "agentName": agent_name,
        "oneLineDescription": description,
        "teamMembers": members,
        "technicalSetup": {
            "buildStyle": build_style,
            "harnessOrFramework": "Claude Code; openai-agents[litellm] runtime built during the event",
            "primaryModels": detect_models(),
            "languagesAndFrameworks": [
                "Python 3.11",
                "openai-agents[litellm]",
                "pydantic / pydantic-settings",
                "SQLite (stdlib sqlite3)",
                "openpyxl",
                "yfinance",
                "Firecrawl v2 REST",
            ],
            "preExistingComponents": detect_pre_existing(),
            "humanInputDuringFinalRun": (
                "none for forecast generation; a team member manually uploads the four "
                "workbooks to OpenStocks, as required by SUBMISSION.md"
            ),
        },
        "submission": {
            "repositoryUrl": detect_repo_url(),
            "finalCommit": _git("rev-parse", "HEAD"),
            "finalCommand": final_command,
        },
        "emailUseConfirmed": email_confirmed,
    }


def validate(entry: dict) -> list[str]:
    """Mirror check-entry.mjs so problems surface here, not at 17:55."""
    problems: list[str] = []

    def need(value, label):
        if not value or (isinstance(value, str) and not value.strip()):
            problems.append(f"{label} is empty")

    need(entry.get("agentName"), "agentName")
    need(entry.get("oneLineDescription"), "oneLineDescription")

    members = entry.get("teamMembers") or []
    if not isinstance(members, list) or not members:
        problems.append("teamMembers is empty")
    elif len(members) > 4:
        problems.append(f"teamMembers has {len(members)} people; the maximum is 4")
    else:
        for i, m in enumerate(members):
            need(m.get("name"), f"teamMembers[{i}].name")
            email = (m.get("email") or "").strip()
            if not EMAIL_RE.match(email):
                problems.append(f"teamMembers[{i}].email is not a valid address: {email!r}")

    tech = entry.get("technicalSetup") or {}
    style = tech.get("buildStyle")
    if style not in VALID_BUILD_STYLES:
        problems.append(f"buildStyle {style!r} must be one of {VALID_BUILD_STYLES}")
    need(tech.get("harnessOrFramework"), "harnessOrFramework")
    need(tech.get("humanInputDuringFinalRun"), "humanInputDuringFinalRun")
    for key in ("primaryModels", "languagesAndFrameworks", "preExistingComponents"):
        if not isinstance(tech.get(key), list):
            problems.append(f"technicalSetup.{key} must be a list")

    sub = entry.get("submission") or {}
    need(sub.get("repositoryUrl"), "submission.repositoryUrl")
    need(sub.get("finalCommit"), "submission.finalCommit")
    need(sub.get("finalCommand"), "submission.finalCommand")
    if len(sub.get("finalCommit") or "") < 7:
        problems.append("submission.finalCommit does not look like a commit hash")

    if entry.get("emailUseConfirmed") is not True:
        problems.append(
            "emailUseConfirmed must be true — every member must know and agree "
            "their address is included (ENTRY.md)"
        )

    blob = json.dumps(entry)
    if len(blob.encode()) > 64_000:
        problems.append("entry.json exceeds the 64 KB limit")
    for marker in ("sk-", "fc-", "API_KEY"):
        if marker in blob:
            problems.append(f"possible secret in entry.json (found {marker!r}) — remove it")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate or validate entry.json")
    ap.add_argument("--name", help="Primary contact full name")
    ap.add_argument("--email", help="Primary contact email")
    ap.add_argument("--member", action="append", default=[],
                    help='Additional member as "Name <email>" (repeatable, max 4 total)')
    ap.add_argument("--agent-name", default="Corpus-First Forecaster")
    ap.add_argument("--description", default=(
        "Three specialist agents read the frozen filings corpus, recent value-investing "
        "news, and long-run financial history; a central agent reconciles them into 12 "
        "cited forecasts."
    ))
    ap.add_argument("--final-command", default="python -m analysts.pipeline --tickers HD,ADI,HAS,DE")
    ap.add_argument("--build-style", default="headless-agent", choices=VALID_BUILD_STYLES)
    ap.add_argument("--check", action="store_true", help="Validate the existing entry.json only")
    ap.add_argument("--confirm-emails", action="store_true",
                    help="Set emailUseConfirmed=true (every member must have agreed)")
    a = ap.parse_args()

    if a.check:
        if not ENTRY_PATH.exists():
            print(f"No entry.json at {ENTRY_PATH}")
            return 2
        entry = json.loads(ENTRY_PATH.read_text(encoding="utf-8"))
    else:
        if not (a.name and a.email):
            print("--name and --email are required to generate (or use --check).")
            return 2
        members = [{"name": a.name.strip(), "email": a.email.strip()}]
        for raw in a.member:
            m = re.match(r"^\s*(.+?)\s*<\s*(.+?)\s*>\s*$", raw)
            if not m:
                print(f'--member must look like "Name <email>", got: {raw!r}')
                return 2
            members.append({"name": m.group(1), "email": m.group(2)})
        entry = build_entry(
            a.agent_name, a.description, members, a.final_command,
            a.build_style, a.confirm_emails,
        )
        ENTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        ENTRY_PATH.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {ENTRY_PATH}")

    print(json.dumps(entry, indent=2))
    problems = validate(entry)
    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("entry.json is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
