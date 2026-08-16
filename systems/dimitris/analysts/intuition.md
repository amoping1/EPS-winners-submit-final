# Analyst intuition

This file is injected verbatim into the central agent's prompt. It is the human judgement layer —
edit it freely, it needs no code change.

What follows is a starting scaffold. Replace or extend it with your own experience.

---

## Seasonality — before anything else

**Compare to the same quarter last year, never to last quarter.** In a seasonal business a
sequential comparison is almost always misleading. Home Depot's Q2 contains the spring selling
season and is structurally its largest; reading it against Q1 says nothing. Deere's quarters track
the planting and harvest calendar. Hays reports half-yearly, with summer hiring lulls.

Then ask what distorts the comparison before drawing any conclusion:

- **Calendar effects** — a 53rd week, a shifted fiscal year-end, an extra selling day, where Easter
  falls
- **FX** — for a company with material overseas revenue, translation alone can move the reported
  line by more than the underlying business did. Separate constant-currency from reported
- **Acquisitions and disposals** — organic versus total is a different question
- **Weather and one-off events** — real, but usually smaller than the narrative suggests

A forecast that ignores seasonality will be wrong by more than the entire analyst dispersion.

## Cyclicality — where are we in the cycle

Every one of these four is cyclical, and each rides a *different* cycle:

| Company | Cycle that actually drives it |
| --- | --- |
| Home Depot | US housing turnover, home equity, repair-and-remodel spend |
| Analog Devices | semiconductor inventory cycle, industrial and auto capex, bookings/book-to-bill |
| Hays | white-collar hiring, especially permanent placement — the most cyclical line in staffing |
| Deere | farm income, crop prices, dealer inventory and order books |

The discipline:

1. **Locate the position in the cycle before forecasting the level.** Early-cycle, mid-cycle and
   late-cycle quarters behave differently even at identical revenue.
2. **Extremes mean-revert; trends do not extrapolate.** The single most common forecasting error is
   projecting the last two quarters forward in a straight line. Cyclical businesses turn.
3. **Watch operating leverage in both directions.** In a downturn, revenue falls a little and
   margin falls a lot — incremental margins are asymmetric. Gross margin and operating profit move
   further than revenue in *both* directions. This matters most for ADI's adjusted gross margin
   and Deere's segment operating profit.
4. **Distinguish structural from cyclical.** A margin decline caused by mix shift is permanent
   until mix shifts back; one caused by underutilisation reverses when volume returns. Management
   language in MD&A usually tells you which, if you read for it.

## Trend — the company, then the industry

Separate the two. If the whole industry is down 8% and the company is down 5%, that is a *share
gain* inside a cyclical decline, and it forecasts very differently from a company-specific problem.
Read peers, suppliers and customers as corroboration.

Then ask whether recent quarters actually bend the multi-year trajectory, or merely wobble around
it. Most wobble.

## Surprises — recent, and historical

- **Recent (news).** A dramatic headline rarely moves a single quarter's revenue much. Ask
  specifically: does this change the *reported* number for *this* fiscal period? Most news does
  not. Weight guidance changes, order-book statements and demand commentary far above sentiment.
- **Historical.** Where did recent prints diverge from their own trend, and did management call it
  one-off or continuing? A "one-off" that recurs three quarters running is not one-off.

## What has repeatedly mattered for this company

Every company has two or three recurring swing factors that decide beat-or-miss. Find them in the
history rather than reasoning from first principles. These are usually the same items management
is asked about on every single earnings call.

## Guidance bias — the highest-value input

Managements have habits, and habits persist. Build the guided-versus-actual record and use it:

- A company that habitually guides conservatively and beats should be forecast **above** the
  guidance midpoint by roughly its historical beat
- A company that has recently *cut* guidance is telling you something consensus may lag on
- Company guidance is a first-party forecast made by people with the actual data. Respect it, then
  adjust for the measured bias

## Magnitude discipline — the sanity floor

Before submitting any number, check it against what is physically plausible:

- Quarterly revenue rarely moves more than single-digit percent year over year absent an
  acquisition. A 30% swing needs a specific, named reason
- Margins move in tens of basis points, not whole points, unless something structural happened
- EPS moves further than revenue because of operating leverage and buybacks — check the share count
- If a number implies a record high or a multi-year low, say so explicitly and justify it

**Sense-check every figure against the same quarter last year before submitting.** If you cannot
explain the delta in one sentence, the number is probably wrong.

## Stance

Forecast the **business**, not the share price. Nothing here concerns valuation multiples,
sentiment or price action. The task is to predict what the company will report — a question about
demand, pricing, cost and mix.

Prefer the boring central case. A forecast that beats consensus by being closer to the truth is
worth far more than one that is spectacularly right in one metric and wild in another — the score
averages across all twelve, and each is capped at a maximum penalty.
