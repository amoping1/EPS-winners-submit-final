"""Reconciliation rail.

Deterministic on purpose. The three forecasters disagree; something has to choose, and
that something must not be persuadable. An LLM asked to pick between its own three answers
will rationalise. Arithmetic will not.

Not a flat median. Guidance quality varies enormously across these four companies - ADI
guides the target quarter with a point estimate, Deere guides only the full year, Hays has
published analyst consensus. Weighting all three methods equally throws away the strongest
signals available.
"""

from __future__ import annotations

from statistics import median

# Method weights. Guidance wins where guidance exists; the statistical model is the
# reliable floor; qualitative is a tie-breaker that should rarely dominate.
BASE_WEIGHTS = {"guidance": 1.0, "statistical": 0.8, "qualitative": 0.6}
CONFIDENCE_WEIGHTS = {"high": 1.0, "medium": 0.7, "low": 0.4}

# How far a single forecaster may sit from the group median before it is treated as an
# outlier and down-weighted. Capped scoring punishes blow-ups far more than it rewards
# precision, so we lean hard against outliers.
OUTLIER_RATIO = 0.35

# Fallback half-width when guidance is given as a point with no stated range.
GUIDANCE_BAND = 0.05


def _relative_spread(value: float, ref: float) -> float:
    if ref == 0:
        return abs(value)
    return abs(value - ref) / abs(ref)


def _period_matches(anchor_period, target_period) -> bool:
    """True only when an anchor describes the SAME period we are forecasting.

    Full-year guidance is not a quarterly anchor. Deere guides FY2026 net income, which
    implies ~$16.67 of annual EPS; applying that to a Q3 forecast dragged a sensible $5.25
    up to $10.21. Periods must match before an anchor is allowed to pull.
    """
    import re

    if not anchor_period or not target_period:
        return False

    def parse(value: str):
        text = str(value).upper()
        year = re.search(r"(20\d{2})", text)
        quarter = re.search(r"Q([1-4])", text)
        return (year.group(1) if year else None, quarter.group(1) if quarter else None)

    a_year, a_q = parse(anchor_period)
    t_year, t_q = parse(target_period)

    if a_year != t_year:
        return False
    # A quarterly target needs the same quarter; an annual target needs no quarter at all.
    return a_q == t_q


def reconcile_metric(
    proposals: dict[str, dict],
    anchor: dict | None = None,
    history: list[float] | None = None,
    target_period: str | None = None,
) -> dict:
    """Combine method proposals for one metric into a single number.

    Returns the chosen value plus the audit trail: weights applied, outliers found, and
    whether a clamp fired. Everything here is explainable to a judge.
    """
    if not proposals:
        return {"value": None, "note": "no proposals"}

    values = {m: p["value"] for m, p in proposals.items()}
    ref = median(values.values())

    weights, outliers = {}, []
    for method, proposal in proposals.items():
        weight = BASE_WEIGHTS.get(method, 0.5) * CONFIDENCE_WEIGHTS.get(
            proposal.get("confidence", "medium"), 0.7
        )
        spread = _relative_spread(proposal["value"], ref)
        if spread > OUTLIER_RATIO:
            weight *= 0.25
            outliers.append({"method": method, "value": proposal["value"],
                             "spread_from_median": round(spread, 3)})
        weights[method] = weight

    # A high-confidence company guidance anchor outranks any model. Management sees the
    # quarter from the inside; our trend fit does not.
    anchor_pull = None
    anchor_rejected = None
    if anchor and anchor.get("kind") in ("guidance", "consensus", "last_actual"):
        if (anchor.get("kind") != "last_actual" and target_period
                and not _period_matches(anchor.get("period"), target_period)):
            anchor_rejected = {
                "reason": "period mismatch",
                "anchor_period": anchor.get("period"),
                "target_period": target_period,
                "value": anchor.get("value"),
            }
            anchor = None
    if anchor and anchor.get("kind") in ("guidance", "consensus", "last_actual"):
        if isinstance(anchor.get("value"), (int, float)):
            anchor_weight = {"guidance": 1.2, "consensus": 0.9, "last_actual": 0.7}[anchor["kind"]]
            if anchor.get("confidence") == "low":
                anchor_weight *= 0.5
            weights["_anchor"] = anchor_weight
            values["_anchor"] = float(anchor["value"])
            anchor_pull = {"kind": anchor["kind"], "value": float(anchor["value"]),
                           "weight": round(anchor_weight, 2)}

    total = sum(weights.values()) or 1.0
    combined = sum(values[m] * w for m, w in weights.items()) / total

    # Hard bound to explicit company guidance. Management publishing "revenue $3.9bn +/-
    # $100m" for the exact period we are forecasting is not one opinion among three - it is
    # the tightest information anyone outside the company has. A model that lands outside
    # that band has not found an edge, it has drifted.
    guidance_bound = None
    if anchor and anchor.get("kind") == "guidance" and isinstance(anchor.get("value"), (int, float)):
        if isinstance(anchor.get("low"), (int, float)) and isinstance(anchor.get("high"), (int, float)):
            low, high = sorted((float(anchor["low"]), float(anchor["high"])))
        else:
            # The agent did not return the guided range, only the point. Guidance bands on
            # these companies run roughly +/-3% (ADI guided $3.9bn +/- $100m, ~2.6%), so
            # bound generously rather than not at all - the failure we are preventing is a
            # forecast drifting well outside a range management published.
            point = float(anchor["value"])
            low, high = point * (1 - GUIDANCE_BAND), point * (1 + GUIDANCE_BAND)
        if not low <= combined <= high:
            guidance_bound = {"from": round(combined, 4), "low": low, "high": high}
            combined = min(max(combined, low), high)

    # Clamp against observed history. A forecast far outside everything ever reported is
    # more likely a unit error than an insight.
    clamp = None
    if history:
        lo, hi = min(history), max(history)
        span = (hi - lo) or abs(hi) or 1.0
        floor, ceiling = lo - span, hi + span
        if combined < floor or combined > ceiling:
            clamp = {"from": round(combined, 4), "floor": round(floor, 4),
                     "ceiling": round(ceiling, 4)}
            combined = min(max(combined, floor), ceiling)

    return {
        "value": round(combined, 4),
        "method_values": {m: round(v, 4) for m, v in values.items()},
        "weights": {m: round(w, 3) for m, w in weights.items()},
        "median_of_methods": round(ref, 4),
        "outliers": outliers,
        "anchor": anchor_pull,
        "anchor_rejected": anchor_rejected,
        "guidance_bound": guidance_bound,
        "clamped": clamp,
        "agreement": round(1.0 - min(
            _relative_spread(max(values.values()), ref),
            1.0,
        ), 3),
    }


def _normalise(label: str) -> str:
    """Strip decoration so 'Revenue (USDm, basis: reported)' matches 'Revenue'."""
    label = str(label).split("(")[0]
    return "".join(ch for ch in label.lower() if ch.isalnum())


def align_label(returned: str, canonical: list[str]) -> str | None:
    """Map whatever a model returned onto the canonical metric label.

    Prompts drift no matter how firmly they are worded, and a label mismatch silently
    drops a metric - which scores 5.0. This is the deterministic backstop.
    """
    target = _normalise(returned)
    if not target:
        return None
    for label in canonical:
        if _normalise(label) == target:
            return label
    for label in canonical:
        norm = _normalise(label)
        if target.startswith(norm) or norm.startswith(target):
            return label
    return None


def align_all(by_label: dict, canonical: list[str]) -> dict:
    """Re-key a {label: ...} mapping onto canonical labels, dropping unmatchable keys."""
    out: dict = {}
    for key, value in by_label.items():
        matched = align_label(key, canonical)
        if matched is None:
            continue
        if isinstance(value, dict) and isinstance(out.get(matched), dict):
            out[matched].update(value)
        else:
            out[matched] = value
    return out


def pick_anchor(anchors: list[dict], metric_label: str) -> dict | None:
    """Best anchor for a metric: company guidance beats consensus beats last actual."""
    ranked = {"guidance": 3, "consensus": 2, "last_actual": 1, "derived": 0}
    candidates = [a for a in anchors if align_label(a.get("metric", ""), [metric_label])
                  and isinstance(a.get("value"), (int, float))]
    if not candidates:
        return None
    return max(candidates, key=lambda a: ranked.get(a.get("kind", "derived"), 0))


def is_quarterly(period) -> bool:
    """True when a period label denotes a quarter rather than a full year."""
    import re

    return bool(re.search(r"Q[1-4]", str(period).upper()))


def quarter_of(period):
    """Which quarter a label denotes, or None for an annual period."""
    import re

    match = re.search(r"Q([1-4])", str(period).upper())
    return match.group(1) if match else None


def same_period_shape(candidate, target) -> bool:
    """Same KIND of period: same quarter number, or both annual.

    Filtering only on quarter-vs-annual is not enough for a seasonal business. Deere's Q2
    is its spring peak and Q3 is materially smaller; trending Q3 against Q2 actuals put
    worldwide net sales at 12,055 against a consensus of 10,732. The comparable period for
    a Q3 forecast is prior Q3s, nothing else.
    """
    return quarter_of(candidate) == quarter_of(target)


def history_for(history: list[dict], metric_label: str,
                target_period: str | None = None) -> list[float]:
    """Historical values for a metric, restricted to the target's period TYPE.

    A quarterly forecast must never be trended against full-year actuals. When a research
    pass returned Deere's FY totals instead of Q3 comparables, the clamp had a full-year
    range to work with and passed 40,516 for a quarter that runs around 11,000 - three
    capped 5.0s on one company. Period type is filtered here, deterministically, rather
    than hoped for in a prompt.
    """
    rows = [
        h for h in history
        if align_label(h.get("metric", ""), [metric_label])
        and isinstance(h.get("value"), (int, float))
    ]
    if target_period:
        rows = [h for h in rows if same_period_shape(h.get("period"), target_period)]
    return [float(h["value"]) for h in rows]
