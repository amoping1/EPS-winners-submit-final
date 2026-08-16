# EPS-Winners — Strategy

**Event:** Agents vs Wall Street, 16 Aug 2026. Build 11:15 → final run 17:15 → deadline 18:00.

## 1. What we optimise for

Two independent $4,500 pools. They are not equally winnable.

| | Architecture & Design | Forecast Accuracy |
|---|---|---|
| Decided | Today, on a published 100-pt rubric | Post-event, weeks out |
| We control | ~100% | ~30% |

**Primary target: Architecture.** 30 of its 100 points are the HTML write-up alone, and the
5-minute judge conversation is called "the most important part". Accuracy falls out of the
same build, so we are not trading one for the other — we are choosing where to spend the
last hour.

## 2. The accuracy scoring function dictates the forecast style

Score per metric = `min(5.0, our_error / max(wall_street_error, floor))`, averaged over 12.
Lower wins. Missing = 5.0.

Consequences, in priority order:

1. **Never leave a cell blank.** A crude guess beats an empty cell by 4.5 points.
2. **Minimise variance, not bias.** One blow-up (5.0) erases four excellent metrics (~0.4).
3. **Prefer the boring number.** Guidance-anchored and trend-consistent beats clever and
   contrarian, every time, under a capped-ratio score.
4. **Clamp every output** to a plausible band before it reaches the workbook.

Floors: percentage metrics 0.5pp; money/EPS 0.5% of reported result. So on percentage
metrics, anything inside ~0.5pp is already near-optimal — do not over-invest there.

## 3. The 12 metrics

| Company | Period | Metric | Units | Basis |
|---|---|---|---|---|
| Home Depot | FY2026 Q2 | Net sales | USDm | — |
| | | Adjusted diluted EPS | USD/share | **adjusted** |
| | | Comparable sales, total company | % | — |
| Analog Devices | FY2026 Q3 | Revenue | USDm | — |
| | | Adjusted diluted EPS | USD/share | **adjusted** |
| | | Adjusted gross margin | % | **adjusted** |
| Hays plc | FY2026 (full year) | Net fees | GBPm | — |
| | | Pre-exceptional basic EPS | **GBp (pence)** | **pre-exceptional** |
| | | Pre-exceptional operating profit | GBPm | **pre-exceptional** |
| Deere | FY2026 Q3 | Worldwide net sales and revenues | USDm | — |
| | | Diluted EPS **(GAAP)** | USD/share | **gaap** |
| | | Production & Precision Ag operating profit | USDm | — |

## 4. Three traps that produce 5.0s

1. **Hays EPS is in pence, not pounds.** Everything else is millions or dollars-per-share.
   A £/p slip is a 100× error — capped 5.0, and it looks plausible on the way past.
2. **Deere EPS is GAAP; HD and ADI are Adjusted.** Mixing basis is a silent miss.
3. **Fiscal calendars are offset.** None of these are calendar quarters. Verify every
   `period_end` against the corpus before forecasting anything. Wrong period = 5.0 across
   the board.

Secondary: three of the four companies' third metric is a non-GAAP or segment measure
(comparable sales, net fees, pre-exceptional, segment operating profit). These live in MD&A
and segment notes, not the face of the income statement.

## 5. System design (built for the 70 system points)

The 16-point question is literally *"how does the system reason its way to a forecast
instead of simply asking an AI model for a number?"* So we do not ask for a number.

```
corpus ──▶ retrieve(as_of) ──▶ extract history ──▶ decompose drivers
                                    │                     │
                                    ▼                     ▼
                              provenance          3 forecast methods
                                    │                     │
                                    └──────▶ reconcile (median) ──▶ validate ──▶ workbook
```

- **Retrieval** — BM25/TF-IDF over the frozen corpus. Deterministic, no embedding setup,
  fast. Takes a mandatory `as_of` date (see §6).
- **Extraction** — last 8–12 periods per metric into a strict schema. Every value carries
  `{doc_id, section, date}`. Provenance threaded end-to-end = the 12-pt model-quality score.
- **Decomposition** — forecast the drivers, not the headline. EPS = net income / diluted
  shares. Revenue = segments, or comps × base. Show the arithmetic.
- **Ensemble** — three independent methods per metric: (a) guidance-anchored,
  (b) seasonal/trend statistical, (c) transcript-signal qualitative. Reconcile by **median**
  — variance-robust, which is exactly what the capped score rewards.
- **Validation gate** — unit assertions, basis assertions, magnitude bounds vs history,
  identity checks (EPS × shares ≈ NI), period assertion, cross-method disagreement flags.
  Cheapest 12 points on the board.

## 6. Backtesting — the leakage trap

Every corpus document header carries a publication date. **Retrieval must take an `as_of`
date and hard-filter anything published later.** Without it, a backtest retrieves the
document containing the answer, scores brilliantly, and tells us nothing — and we would not
find out until results publish.

A backtest is then: set `as_of` to the day before a past earnings date, run the *unmodified*
pipeline, score against the known actual.

Build the filter on day one. Retrofitting it voids every backtest run before it.

This also scores directly — "data approach" is 12 points on source validation and currency
checks.

## 7. Corpus coverage

| Company | Docs | Coverage |
|---|---:|---|
| Home Depot | 319 | 2012-05-15 → 2026-05-21 |
| Analog Devices | 271 | 2015-01-29 → 2026-06-02 |
| Hays plc | 239 | 2015-09-18 → **2026-08-03** |
| Deere | 310 | 2012-05-16 → 2026-05-28 |

Frozen 2026-08-14. HD/ADI/DE cut off in late May–early June, so the corpus holds **last
quarter's results plus management guidance** — our strongest single signal.

**Hays runs to 2026-08-03.** Hays' FY2026 ended ~30 June, so a July trading update or
prelim is very likely in there — the highest-value document in the corpus. Check it first.

## 8. Time plan

| Time | Milestone |
|---|---|
| 11:15–11:45 | Verify all four `period_end` dates. Freeze `config/metrics.json`. |
| 11:45–13:00 | **End-to-end with fake numbers.** 4 valid .xlsx that pass `check:submission`. |
| 13:00–15:00 | Real engine: extraction → decomposition → ensemble. |
| 15:00–16:00 | Validation layer + provenance report. **Feature freeze 16:00.** |
| 14:00–16:00 | *Parallel, one person full-time:* architecture HTML. |
| 16:00–17:15 | Judge conversation. Rehearse twice. Dry-run pipeline twice, **time it**. |
| 17:15–18:00 | Final run. Upload from 17:30 — last valid upload counts. |

The 11:45→13:00 milestone is the highest-value tactic of the day. The final run is only
45 minutes; a pipeline that has never run end-to-end will not start working at 17:15.
Target **under 15 minutes** end-to-end so we get a retry.

## 9. Free points most teams drop

- **Social prizes: $1,000 for two posts.** Best ROI per minute in the event. Post by 13:00.
  X needs `@_openstocks @primerapp_ @AITinkerers @openai`; LinkedIn needs Primer, OpenAI,
  AI Tinkerers. Closes 17:00.
- **"Honesty and self-knowledge" — 6 pts.** Explicitly list what we abandoned and what is
  broken. Most teams write a marketing page and score ~1.
- **Clear-run log** — timestamped, secrets stripped. Mandatory component.

## 10. Rules we must not trip

- Everything challenge-specific is built after 11:15. Generic libraries are fine, and must
  be declared in `entry.json`.
- No hand-made forecasts outside the system.
- Uploads to OpenStocks are manual — the agent must not upload programmatically.
- No secrets in repo, HTML, or logs.
- Repo history and final commit are mandatory verification. Commit continuously.
