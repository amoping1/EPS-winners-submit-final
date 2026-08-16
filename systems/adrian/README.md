# EPS-Winners

Forecasting agent for **Agents vs Wall Street** (16 Aug 2026). Produces 12 metrics across
four companies: Home Depot, Analog Devices, Hays plc, Deere & Company.

Read [STRATEGY.md](STRATEGY.md) first — it covers what we optimise for, the scoring
function, the traps that produce capped 5.0s, and the time plan.

## Layout

```
STRATEGY.md          the plan
config/
  metrics.json       12 metrics + unit/basis assertions (the trap guards)
  settings.json      corpus path, as_of date
src/
  corpus/            indexing + as_of-filtered BM25 retrieval
  extract/           history extraction with provenance      [to build]
  forecast/          driver decomposition + 3-method ensemble [to build]
  validate/          unit, basis, bounds, identity checks     [to build]
  workbook/          template-preserving xlsx writer          [to build]
backtest/            as_of harness                            [to build]
architecture/        the judged HTML write-up                 [to build]
submission/          four final .xlsx land here
logs/                clear-run log lands here
```

## The corpus lives outside this repo

100 MB of organizer-supplied documents, kept out of here on purpose. Path is set in
`config/settings.json`:

```
/Users/spax/agents-vs-wall-street-starter/challenge/offline-data
```

Also mirrored privately at `github.com/amoping1/agents-vs-wall-street-starter`.

## The one rule that shapes all retrieval

Every search takes a mandatory `as_of` date and hard-excludes documents published after it.
Without that, a backtest reads the answer and reports a fantastic score. See STRATEGY.md §6.

```python
from src.corpus.corpus import Corpus

hays = Corpus("/path/to/offline-data", "hays").build()
for doc, score in hays.search("net fees full year", as_of="2026-08-16", limit=5):
    print(doc.published_at, doc.document_type, doc.doc_id)
```

## Submission contract (organizer-owned)

Four workbooks in `submission/`, built from the organizer templates without altering the
`Summary` sheet:

`HD-FY2026Q2.xlsx` · `ADI-FY2026Q3.xlsx` · `HAS-FY2026.xlsx` · `DE-FY2026Q3.xlsx`

Validated with the organizer's own checker before upload:

```bash
npm run check:submission
```

Uploads to OpenStocks are **manual** — the agent must never upload programmatically.
