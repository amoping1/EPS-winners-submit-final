# Build Plan — phased

Written 12:15. Final run opens **17:15**. Hard deadline **18:00**.

Ordering principle: **the submission path is built before the intelligence.** A mediocre
number in a valid workbook scores. A brilliant number that never reaches a workbook scores
5.0. Everything in Phase 1 exists to guarantee we can submit *something* valid by 13:15,
after which every phase only improves the numbers.

---

## Phase 0 — Foundation ✅ done 12:12

- Strategy, metrics config with unit/basis trap guards
- `src/tools/corpus.py` — BM25 + mandatory `as_of` cutoff, leak-tested (0 leaks)
- Agent-shaped directory structure, first commit landed

---

## Phase 1 — Toolbelt + submission path  (12:15 → 13:15)

**Goal: four valid .xlsx passing `npm run check:submission`, with placeholder numbers.**

1. Vendor organizer contract files (~100 KB, not the 100 MB corpus):
   `scripts/`, `package.json`, `challenge/templates/`, `challenge/companies.json`
2. `src/tools/documents.py`
   - `list_index(company, doc_type=None, period=None, since=None)` — reads the per-company
     `INDEX.md` catalog. Cheaper and more precise than search; use it first.
   - `read_document(doc_id, section=None)` — body + provenance stamp
   - `extract_table(doc_id, near=None)` — pull markdown tables out of filings
3. `src/rails/workbook.py` — load template, write **C7:C9 only**, save. Never build a
   workbook from scratch.
4. Wire a placeholder run end-to-end. Run `check:submission`. **Must print PASS ×4.**

**Exit test:** `npm run check:submission` → 4 PASS. If this is not green by 13:15, stop
everything else and fix it.

---

## Phase 2 — The research agent  (13:15 → 14:30)

The actual agent: an LLM in a loop with the toolbelt, deciding its own next move.

- `src/agents/research.py` — loop: classify industry → plan queries from the profile →
  search / read / extract → assess gaps → re-query → emit a cited **evidence pack**
- `src/agents/profiles.py` — `IndustryProfile` registry: retail, semiconductors,
  industrial, staffing, fallback. Profiles are **data, not code paths**.
- Every fact in the evidence pack carries `{doc_id, published_at, source_url}`.

**Exit test:** run on Deere alone, print the full tool-call trace and evidence pack.
The trace is also the demo for the judge conversation.

---

## Phase 3 — Forecasters + critic  (14:30 → 15:30)

- Three **independent** forecaster agents, separate context, same evidence pack:
  guidance-anchored · statistical/seasonal · qualitative/transcript
- `src/agents/critic.py` — adversarial, sees numbers *without* forecaster reasoning, has
  tools, tries to refute. Refutation sends the research agent one more round (max 2).
- `src/rails/reconcile.py` — **confidence-weighted, not a flat median.** ADI Q3 is
  explicitly guided and Hays has published consensus; flat-averaging those against a naive
  trend model throws away our two biggest edges.

---

## Phase 4 — Rails + provenance  (15:30 → 16:00) → **FEATURE FREEZE 16:00**

- `src/rails/validate.py` — hard assertions the agent cannot argue past:
  - units (**Hays EPS in pence**; percentages as `4.5` not `0.045`)
  - basis (Deere = GAAP; HD/ADI = adjusted; Hays = pre-exceptional)
  - period cross-check — **the corpus `period` metadata is unreliable**; verify against
    filename slug, `published_at`, and body
  - magnitude bounds vs extracted history
  - identity checks (EPS x shares ~ net income)
  - never emit null — fall back to the anchor rather than leave a cell empty
- Provenance report: all 12 numbers, citation chain, validation flags. This is the judge
  conversation prop and scores under tooling.

---

## Phase 5 — Architecture HTML  (14:00 → 16:00, parallel, one person full-time)

30 of 100 points. Not a leftover task.

- Plain-English workflow a technical outsider gets in 5 minutes
- Diagram that **matches the real code** (10 pts scores exactly this)
- Honesty section — abandoned approaches, known weaknesses, what is untested.
  6 pts, and most teams will score ~1 by writing marketing copy.
- Self-contained, < 2 MB, no scripts / external assets / network, no secrets
- **Locked 17:15**

---

## Phase 6 — Rehearsal  (16:00 → 17:15)

- Judge conversation 16:00–17:15, 5 min — "the most important part" of architecture scoring
- Rehearse the pitch twice
- **Two timed full dry runs.** Target under 15 min end-to-end so the 45-min final window
  allows a retry.
- Fill remaining `entry.json` fields: models, libraries (openpyxl declared), final command

---

## Phase 7 — Final run  (17:15 → 18:00)

| Time | Action |
|---|---|
| 17:00 | Social posts already up (do these by 13:00 — $1,000, highest ROI in the event) |
| 17:15 | HTML locks. Final run starts. Capture clear-run log to `logs/`. |
| 17:30 | Uploads open. **Upload immediately** — last valid upload counts, so overwrite later. |
| 17:50 | Final commit, `entry.json` submitted via the private form |
| 18:00 | Hard deadline |

---

## Cut list if we fall behind

Drop in this order — each keeps a valid submission:

1. Backtest harness (nice evidence, not scored directly)
2. Critic agent → replace with assertions only
3. Three forecasters → one guidance-anchored forecaster
4. Research agent loop → `list_index` + fixed queries

**Never cut:** workbook writer, validation assertions, architecture HTML, the four uploads.

---

## Standing risks

- **`period` metadata in the corpus is wrong on real documents** (ADI Q2 10-Q labelled
  Q3; Deere Q2 call labelled Q3). Trusting it produces whole-company 5.0s.
- **Hays divestitures** — 6 countries sold 16 June 2026, c.£15m of FY26 net fees; 13
  countries ~breakeven on c.£85m. Naive YoY extrapolation will be wrong.
- **HD acquisitions** — SRS and GMS inflate net sales; sales and comps diverge.
- **Repo is private.** Judges must verify history. Resolve before 18:00.
