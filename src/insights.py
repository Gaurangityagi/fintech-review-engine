"""
AI executive brief + exportable Markdown report.

The LLM only sees pre-computed statistics (never asked to count), so numbers in
the brief are grounded in the analytics layer rather than hallucinated.
"""
import json
from datetime import datetime

import pandas as pd

from src import analytics as A
from src.llm_classify import call_llm


def build_fact_sheet(df: pd.DataFrame, ws) -> dict:
    tax = ws.taxonomy
    score = A.app_scorecard(df, tax)
    bench = A.comparative_benchmark(df, tax)
    prio = A.compute_priority_scores(df, tax)
    drivers = A.rating_drivers(df, tax)
    spikes = A.detect_spikes(df, tax)
    versions = A.version_impact(df, tax)
    frs = A.feature_request_clusters(df)

    def rnd(x):
        return x.round(2).to_dict("records")

    facts = {
        "workspace": ws.name, "category": ws.category, "country": ws.country,
        "overview": A.get_overview_stats(df),
        "app_scorecard": rnd(score),
        "top_priorities": rnd(prio.head(6)[["theme_label", "reviews", "share_pct", "high_sev_pct",
                                            "priority_score", "quick_win"]]) if not prio.empty else [],
        "rating_drivers": rnd(drivers.head(5)[["theme_label", "share_pct", "theme_avg_rating", "rating_drag"]]
                              .rename(columns={"rating_drag": "potential_star_gain_if_fixed"})) if not drivers.empty else [],
        "feature_requests": rnd(frs.head(6)[["request", "mentions", "apps"]]) if not frs.empty else [],
    }
    if not bench.empty:
        sig = bench[bench["significant"] & ~bench["theme"].isin(A.NON_ISSUE_THEMES)]
        facts["significant_gaps_vs_peers"] = rnd(sig.sort_values("delta", ascending=False)
                                                 [["app_name", "theme_label", "app_percentage",
                                                   "peer_percentage", "q_value"]].head(8))
    if not spikes.empty:
        facts["spike_alerts"] = rnd(spikes[["app_name", "theme_label", "latest_share", "baseline_share", "z"]])
    if not versions.empty:
        flagged = versions[versions["flag"] != ""]
        facts["release_flags"] = rnd(flagged[["app_name", "version", "avg_rating", "delta_vs_prev", "flag"]])
    # Real, verbatim quotes for the top issues so the brief never has to invent "what users say"
    if not prio.empty:
        facts["verbatim_quotes"] = {
            r.theme_label: [f"[{q.app_name}, {int(q.rating)}★] {str(q.review_text)[:220]}"
                            for q in A.voice_of_customer(df, r.theme, n=2).itertuples()]
            for r in prio.head(4).itertuples() if r.theme not in A.NON_ISSUE_THEMES
        }
    return facts


def generate_ai_brief(df: pd.DataFrame, ws) -> str:
    facts = build_fact_sheet(df, ws)
    prompt = f"""You are a Head of Product writing a crisp executive brief from app-review analytics.
Use ONLY the numbers in the fact sheet. Do not invent statistics. Be specific and name apps.
When quoting users, use ONLY the verbatim_quotes provided (you may shorten with "…"); never write your own quotes.

Write Markdown with exactly these sections:
## TL;DR
3 bullets, each one sentence with a number.
## Competitive Position
Who leads, who lags and why (2-4 bullets). Mention only statistically significant gaps as "significant".
## Biggest Risks
Top 3 issues by priority / rating drag, each with: what users say, evidence (numbers), likely root cause hypothesis.
## Recommended Actions (next 30-60-90 days)
A short table: Horizon | Action | Owner (team) | Success metric.
## Opportunities
Feature requests or competitor weaknesses worth exploiting (2-3 bullets).
## Caveats
1-2 bullets on data limitations (sample window, LLM classification, store sampling).

Fact sheet (JSON):
{json.dumps(facts, default=str)[:24000]}"""
    text = call_llm(prompt, json_mode=False)
    if not text or text.strip() in ("{}", ""):
        return "_The AI brief could not be generated — check your LLM API key in `.env`._"
    ws.brief_path.write_text(text, encoding="utf-8")
    return text


def build_markdown_report(df: pd.DataFrame, ws, brief: str | None = None) -> str:
    tax = ws.taxonomy
    o = A.get_overview_stats(df)
    out = [f"# {ws.name} — App Review Intelligence Report",
           f"_Category: {ws.category} · Market: {ws.country.upper()} · Generated {datetime.now():%d %b %Y %H:%M}_", "",
           f"**{o['total_reviews']:,} reviews** across **{o['apps']} apps**, "
           f"{o['date_range'][0]} → {o['date_range'][1]}.", "",
           f"> {A.get_headline_finding(df, tax)}", ""]
    if brief:
        out += [brief, ""]

    def table(d: pd.DataFrame, cols: dict, fmt: dict | None = None):
        fmt = fmt or {}
        d = d[list(cols)].copy()
        for c, f in fmt.items():
            d[c] = d[c].map(lambda v: f.format(v) if pd.notna(v) else "—")
        return d.rename(columns=cols).to_markdown(index=False)

    score = A.app_scorecard(df, tax)
    out += ["## App Scorecard", table(score, {
        "app_name": "App", "reviews": "Reviews", "avg_rating": "Avg ★", "nps_proxy": "NPS proxy",
        "pct_1_2_star": "% 1-2★", "high_sev_pct": "% High sev", "top_issue": "Top issue"},
        {"avg_rating": "{:.2f}", "nps_proxy": "{:+.0f}", "pct_1_2_star": "{:.1f}%", "high_sev_pct": "{:.1f}%"}), ""]

    prio = A.compute_priority_scores(df, tax)
    if not prio.empty:
        out += ["## Priority Action Plan", table(prio, {
            "theme_label": "Theme", "reviews": "Reviews", "high_sev_pct": "% High sev", "impact": "Impact",
            "effort": "Effort", "priority_score": "Priority", "justification": "Why"},
            {"high_sev_pct": "{:.0f}%", "impact": "{:.0f}", "effort": "{:.0f}", "priority_score": "{:.1f}"}), ""]

    bench = A.comparative_benchmark(df, tax)
    if not bench.empty:
        sig = bench[bench["significant"]].sort_values("delta", ascending=False)
        if not sig.empty:
            out += ["## Statistically Significant Gaps vs Peers (BH-corrected, q < 0.05)", table(sig, {
                "app_name": "App", "theme_label": "Theme", "app_percentage": "App %",
                "peer_percentage": "Peers %", "q_value": "q", "status": "Status"},
                {"app_percentage": "{:.1f}%", "peer_percentage": "{:.1f}%", "q_value": "{:.3f}"}), ""]

    drv = A.rating_drivers(df, tax)
    if not drv.empty:
        out += ["## Rating Drivers (all apps)", table(drv.head(8), {
            "theme_label": "Theme", "share_pct": "Share", "theme_avg_rating": "Avg ★ in theme",
            "rating_drag": "Rating drag (★)"},
            {"share_pct": "{:.1f}%", "theme_avg_rating": "{:.2f}", "rating_drag": "{:.3f}"}), ""]

    frs = A.feature_request_clusters(df)
    if not frs.empty:
        out += ["## Top Feature Requests", table(frs.head(10), {
            "request": "Request", "mentions": "Mentions", "helpful_votes": "Helpful votes", "apps": "Apps"}), ""]

    out += ["## Methodology",
            "- Reviews scraped from Google Play / Apple App Store (newest first unless stated).",
            "- Each review classified by an LLM into the workspace taxonomy with severity and sentiment.",
            "- Benchmarks: two-proportion z-test of each app vs pooled peers, Benjamini-Hochberg FDR correction.",
            "- Rating drag = theme share × (avg ★ of other reviews − avg ★ in theme).",
            "- Priority = Impact×4 + Frequency/Severity gap×4 − Effort×2 (impact/effort editable per workspace)."]
    return "\n".join(out)
