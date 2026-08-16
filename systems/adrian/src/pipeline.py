"""Orchestrator: evidence -> forecasts -> critique -> reconcile -> validated numbers.

Companies run concurrently. The final-run window is 45 minutes and a retry has to fit
inside it, so wall-clock matters more than tidiness here.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from .agents.aggregator import EvidenceAggregator, followup_brief
from .agents.forecast import run_critic, run_forecaster
from .agents.profiles import PROFILES, classify
from .agents.research import run_research
from .rails.reconcile import align_all, history_for, pick_anchor, reconcile_metric
from .tools.consensus import scan_consensus, scan_last_actual
from .tools.market import consensus_anchors, fetch_snapshot
from .tools.documents import DocumentTools

METHODS = ("guidance", "statistical", "qualitative")


def classify_company(corpus_root: str, company: dict, as_of):
    """Pick an IndustryProfile from filing text. No hardcoded ticker->industry map."""
    tools = DocumentTools(corpus_root, company["corpusDir"], as_of=as_of)
    text = ""
    for entry in tools.list_index(doc_type="Filing", limit=3):
        text += tools.read_document(entry.doc_id, window=12_000)["text"]
    profile, confidence, hits = classify(text)
    return profile, confidence, hits


def run_company(client, corpus_root: str, company: dict, as_of, max_steps: int = 20) -> dict:
    """Full chain for one company. Returns forecasts plus the full audit trail."""
    started = time.time()
    labels = [m["label"] for m in company["metrics"]]

    profile, confidence, hits = classify_company(corpus_root, company, as_of)
    pack = run_research(client, corpus_root, company, profile, as_of,
                        max_steps=max_steps, verbose=False)

    # Aggregate into four channels and check depth. Thin history means the statistical
    # forecaster is guessing and the clamp cannot engage, so go back for exactly what is
    # missing rather than accepting whatever the first pass happened to find.
    # Deterministic consensus hunt. The accuracy score is our error over Wall Street's
    # error, so a published consensus is the number we are measured against. Too valuable
    # to leave to the agent noticing it.
    consensus_tools = DocumentTools(corpus_root, company["corpusDir"], as_of=as_of)
    scanned = scan_consensus(consensus_tools, labels)
    for anchor in scanned:
        anchor.setdefault("period", company["period"])
        pack.anchors.append(anchor)

    # Market channel: public analyst consensus. This is literally the denominator of the
    # accuracy score, so forecasting without reading it is forecasting blind against a
    # benchmark we could simply look up. Yahoo covers the US filers; the UK filer is
    # covered by its own published consensus in the corpus scan above.
    snapshot = fetch_snapshot(company["ticker"])
    market = consensus_anchors(snapshot, labels, company["period"])
    for anchor in market:
        pack.anchors.append(anchor)

    # Drop history whose period type does not match the target before anything reasons
    # over it. A quarterly target trended against full-year actuals is an order-of-
    # magnitude error, not a rounding one.
    from .rails.reconcile import same_period_shape
    dropped = [h for h in pack.history
               if not same_period_shape(h.get("period"), company["period"])]
    pack.history = [h for h in pack.history
                    if same_period_shape(h.get("period"), company["period"])]

    # Last-resort anchor for metrics the research pass returned NO history for. With an
    # empty series the three forecasters have nothing to reason over, agree on a guess,
    # and every rail passes it - ADI adjusted gross margin came back 65.0 that way against
    # a true 73.0. Deliberately narrow: percentage metrics only, where the table-based
    # read is reliable. The money path misreads prose (Home Depot net sales came back as
    # 7.0) and is left switched off rather than shipped half-working.
    aggregator = EvidenceAggregator(labels)
    aggregated = aggregator.aggregate(pack)
    rescued = []
    for metric in company["metrics"]:
        label = metric["label"]
        if aggregated["metrics"][label]["finance"]:
            continue
        if "%" not in metric["units"]:
            continue
        found = scan_last_actual(consensus_tools, label, metric["units"])
        if found:
            found["period"] = company["period"]
            pack.anchors.append(found)
            rescued.append(found)
    if rescued:
        aggregated = aggregator.aggregate(pack)
    followups = 0
    if aggregated["thin_metrics"]:
        followups = 1
        brief = followup_brief(aggregated["thin_metrics"], aggregated, company["period"])
        extra = run_research(client, corpus_root, company, profile, as_of,
                             max_steps=max(8, max_steps // 2), verbose=False,
                             followup=brief)
        seen = {(h.get("metric"), h.get("period")) for h in pack.history}
        for row in extra.history:
            if (row.get("metric"), row.get("period")) not in seen:
                pack.history.append(row)
        for anchor in extra.anchors:
            pack.anchors.append(anchor)
        pack.trace.extend(extra.trace)
        aggregated = aggregator.aggregate(pack)

    pack.cyclical = {
        label: data["cyclical"] for label, data in aggregated["metrics"].items()
    }

    proposals: dict[str, dict] = {}
    for method in METHODS:
        aligned = align_all(run_forecaster(client, company, pack, method, profile), labels)
        for label, forecast in aligned.items():
            proposals.setdefault(label, {})[method] = forecast

    # Reconcile FIRST, then critique the value that will actually be submitted.
    results = {}
    for metric in company["metrics"]:
        label = metric["label"]
        results[label] = reconcile_metric(
            proposals.get(label, {}),
            pick_anchor(pack.anchors, label),
            history_for(pack.history, label, company["period"]),
            target_period=company["period"],
        )
        results[label]["units"] = metric["units"]
        results[label]["basis"] = metric.get("basis")

    verdicts = align_all(
        run_critic(client, company, pack, proposals, profile, reconciled=results), labels
    )

    # Give the critic teeth. A refusal with a concrete range now binds the value instead
    # of being recorded and ignored - that gap was the standing weakness in this design.
    for metric in company["metrics"]:
        label = metric["label"]
        res = results[label]
        verdict = verdicts.get(label)
        res["verdict"] = verdict
        if not verdict or verdict.get("plausible") is not False:
            continue
        low, high = verdict.get("suggested_low"), verdict.get("suggested_high")
        value = res.get("value")
        if (isinstance(low, (int, float)) and isinstance(high, (int, float))
                and isinstance(value, (int, float)) and low <= high):
            if not low <= value <= high:
                res["critic_bound"] = {"from": round(value, 4), "low": low, "high": high,
                                       "concern": verdict.get("concern")}
                res["value"] = round(min(max(value, low), high), 4)

    return {
        "ticker": company["ticker"],
        "company": company["company"],
        "period": company["period"],
        "profile": profile.key,
        "profile_confidence": confidence,
        "profile_signals": hits,
        "profile_label": profile.label,
        "predictors": profile.predictors,
        "catalysts": profile.catalysts,
        "kpis": profile.kpis,
        "elapsed_s": round(time.time() - started, 1),
        "tool_calls": len(pack.trace),
        "history_rows": len(pack.history),
        "followup_passes": followups,
        "channels": aggregated["coverage"],
        "evidence_gaps": aggregated["gaps"],
        "thin_metrics": aggregated["thin_metrics"],
        "consensus_found": scanned,
        "rescued_metrics": rescued,
        "market": snapshot.to_dict(),
        "market_anchors": market,
        "history_dropped_wrong_period": len(dropped),
        "aggregated": aggregated["metrics"],
        "anchors": pack.anchors,
        "gaps": pack.gaps,
        "consensus": pack.consensus,
        "results": results,
        "evidence": pack.to_dict(),
    }


def run_all(client, corpus_root: str, companies: list[dict], as_of, max_steps: int = 20,
            workers: int = 4) -> list[dict]:
    """All four companies concurrently."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_company, client, corpus_root, c, as_of, max_steps)
            for c in companies
        ]
        return [f.result() for f in futures]
