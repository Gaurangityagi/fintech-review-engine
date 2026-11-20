"""
Analytics layer. Everything is taxonomy-driven so it works for any app category.

Classic PM views:   theme frequency, severity-weighted score, benchmark, priority plan
Analyst additions:  significance testing (two-proportion z + Benjamini-Hochberg),
                    Wilson confidence intervals, rating-drag key-driver analysis,
                    release/version regression detection, developer-response analytics,
                    spike alerts, distinctive keywords (weighted log-odds), feature-request
                    clustering, rating-vs-sentiment mismatch.
"""
import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# Small groups with identical ratings (e.g. all 1★) make scipy warn about precision loss; the
# resulting p-values are still valid for our purposes, so keep the console clean.
warnings.filterwarnings("ignore", message="Precision loss occurred", category=RuntimeWarning)

SEVERITY_WEIGHTS = {"high": 3.0, "medium": 1.5, "low": 1.0}
NON_ISSUE_THEMES = {"praise_no_issue", "other"}


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def theme_labels(taxonomy: list[dict]) -> dict:
    return {t["key"]: t["label"] for t in taxonomy}


def _label(df: pd.DataFrame, taxonomy: list[dict], col: str = "theme") -> pd.Series:
    return df[col].map(theme_labels(taxonomy)).fillna(df[col].astype(str).str.replace("_", " ").str.title())


def wilson_ci(k, n, z: float = 1.96):
    """Wilson score interval for a proportion. Works on scalars or arrays; returns (lo, hi) in [0,1]."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, 0)
        denom = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return np.nan_to_num(centre - half), np.nan_to_num(centre + half)


def bh_adjust(pvals) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def pick_time_freq(dates: pd.Series) -> tuple[str, str]:
    """Choose a bucket size that gives a readable number of points for the data's span."""
    d = pd.to_datetime(dates, errors="coerce").dropna()
    if d.empty:
        return "D", "Day"
    span = (d.max() - d.min()).days
    if span <= 3:
        return "6h", "6-hour window"
    if span <= 45:
        return "D", "Day"
    if span <= 240:
        return "W", "Week"
    return "M", "Month"


def _bucket(dates: pd.Series, freq: str) -> pd.Series:
    d = pd.to_datetime(dates, errors="coerce")
    if freq in ("W", "M"):
        return d.dt.to_period(freq).dt.start_time
    return d.dt.floor(freq)


def version_key(v) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", str(v))[:4]) or (0,)


# ═══════════════════════════════════════════════════════════════
#  OVERVIEW
# ═══════════════════════════════════════════════════════════════
def get_overview_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total_reviews": 0, "apps": 0, "date_range": (None, None), "avg_rating": None,
                "avg_sentiment": None, "pct_negative": None, "high_severity_pct": None}
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    return {
        "total_reviews": int(len(df)),
        "apps": int(df["app_name"].nunique()),
        "date_range": (d.min().strftime("%d %b %Y"), d.max().strftime("%d %b %Y")) if not d.empty else (None, None),
        "avg_rating": float(df["rating"].mean()),
        "avg_sentiment": float(df["sentiment_score"].mean()) if "sentiment_score" in df else None,
        "pct_negative": float((df["rating"] <= 2).mean() * 100),
        "high_severity_pct": float((df["severity"] == "high").mean() * 100) if "severity" in df else None,
    }


def app_scorecard(df: pd.DataFrame, taxonomy: list[dict]) -> pd.DataFrame:
    """One row per app: volume, rating, sentiment, detractor share, NPS proxy, high-severity %, top issue."""
    labels = theme_labels(taxonomy)
    rows = []
    for app, g in df.groupby("app_name"):
        issues = g[~g["theme"].isin(NON_ISSUE_THEMES)] if "theme" in g else g.iloc[0:0]
        top = issues["theme"].value_counts()
        promoters = (g["rating"] == 5).mean()
        detractors = (g["rating"] <= 3).mean()
        rows.append({
            "app_name": app,
            "reviews": len(g),
            "avg_rating": g["rating"].mean(),
            "avg_sentiment": g["sentiment_score"].mean() if "sentiment_score" in g else np.nan,
            "pct_1_2_star": (g["rating"] <= 2).mean() * 100,
            "nps_proxy": (promoters - detractors) * 100,
            "high_sev_pct": (g["severity"] == "high").mean() * 100 if "severity" in g else np.nan,
            "issue_share": len(issues) / len(g) * 100 if len(g) else 0,
            "top_issue": labels.get(top.index[0], top.index[0]) if len(top) else "—",
            "top_issue_pct": top.iloc[0] / len(g) * 100 if len(top) else 0,
        })
    return pd.DataFrame(rows).sort_values("avg_rating", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
#  THEMES / SEVERITY / TRENDS
# ═══════════════════════════════════════════════════════════════
def theme_frequency(df: pd.DataFrame, taxonomy: list[dict], app_name: Optional[str] = None) -> pd.DataFrame:
    if df.empty or "theme" not in df:
        return pd.DataFrame(columns=["theme", "theme_label", "count", "percentage"])
    if app_name:
        df = df[df["app_name"] == app_name]
    c = df["theme"].value_counts().rename_axis("theme").reset_index(name="count")
    c["percentage"] = c["count"] / c["count"].sum() * 100
    c["theme_label"] = _label(c, taxonomy)
    return c[["theme", "theme_label", "count", "percentage"]]


def severity_weighted_score(df: pd.DataFrame, taxonomy: list[dict]) -> pd.DataFrame:
    """(high*3 + medium*1.5 + low*1) per app x theme, normalised per 100 reviews of that app. Issues only."""
    if df.empty or "severity" not in df:
        return pd.DataFrame()
    issues = df[~df["theme"].isin(NON_ISSUE_THEMES)]
    sc = issues.groupby(["app_name", "theme", "severity"]).size().unstack(fill_value=0)
    for lvl in SEVERITY_WEIGHTS:
        if lvl not in sc:
            sc[lvl] = 0
    sc["raw_score"] = sum(sc[l] * w for l, w in SEVERITY_WEIGHTS.items())
    sc = sc.reset_index()
    totals = df.groupby("app_name").size()
    sc["normalized_score"] = sc["raw_score"] / sc["app_name"].map(totals) * 100
    sc["theme_label"] = _label(sc, taxonomy)
    return sc


def trend_over_time(df: pd.DataFrame, theme: Optional[str] = None, freq: Optional[str] = None) -> pd.DataFrame:
    """Share of each app's reviews in `theme` per time bucket (share, not count, so volume swings don't mislead)."""
    if df.empty or "date" not in df:
        return pd.DataFrame()
    freq = freq or pick_time_freq(df["date"])[0]
    t = df.assign(bucket=_bucket(df["date"], freq)).dropna(subset=["bucket"])
    tot = t.groupby(["bucket", "app_name"]).size().rename("total")
    hit = (t[t["theme"] == theme] if theme else t).groupby(["bucket", "app_name"]).size().rename("count")
    out = pd.concat([tot, hit], axis=1).fillna(0).reset_index()
    out["share"] = out["count"] / out["total"] * 100
    return out


def rating_trend(df: pd.DataFrame, freq: Optional[str] = None) -> pd.DataFrame:
    freq = freq or pick_time_freq(df["date"])[0]
    t = df.assign(bucket=_bucket(df["date"], freq)).dropna(subset=["bucket"])
    return (t.groupby(["bucket", "app_name"])
             .agg(avg_rating=("rating", "mean"), avg_sentiment=("sentiment_score", "mean"), reviews=("rating", "size"))
             .reset_index())


# ═══════════════════════════════════════════════════════════════
#  BENCHMARK WITH SIGNIFICANCE
# ═══════════════════════════════════════════════════════════════
def comparative_benchmark(df: pd.DataFrame, taxonomy: list[dict], alpha: float = 0.05) -> pd.DataFrame:
    """
    For each app x theme: app share vs peer share (all other apps pooled).
    Two-proportion z-test, Benjamini-Hochberg corrected across all cells,
    Wilson 95% CI on the app share.
    """
    if df.empty or "theme" not in df or df["app_name"].nunique() < 2:
        return pd.DataFrame()
    ct = pd.crosstab(df["app_name"], df["theme"])
    n_app = ct.sum(axis=1)
    rows = []
    for app in ct.index:
        for theme in ct.columns:
            k1, n1 = ct.at[app, theme], n_app[app]
            k2, n2 = ct[theme].sum() - k1, n_app.sum() - n1
            p1, p2 = k1 / n1, (k2 / n2 if n2 else 0)
            pooled = (k1 + k2) / (n1 + n2)
            se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2)) if n2 else 0
            z = (p1 - p2) / se if se > 0 else 0.0
            rows.append({"app_name": app, "theme": theme, "count": int(k1), "app_total": int(n1),
                         "app_percentage": p1 * 100, "peer_percentage": p2 * 100,
                         "category_avg_percentage": ct[theme].sum() / n_app.sum() * 100,
                         "delta": (p1 - p2) * 100, "z": z, "p_value": 2 * stats.norm.sf(abs(z))})
    b = pd.DataFrame(rows)
    b["q_value"] = bh_adjust(b["p_value"].values)
    lo, hi = wilson_ci(b["count"], b["app_total"])
    b["ci_low"], b["ci_high"] = lo * 100, hi * 100
    b["significant"] = b["q_value"] < alpha

    def status(r):
        if not r.significant:
            return "≈ in line with peers"
        worse = r.delta > 0
        if r.theme == "praise_no_issue":
            worse = not worse
        return "▲ worse than peers" if worse else "▼ better than peers"

    b["status"] = b.apply(status, axis=1)
    b["theme_label"] = _label(b, taxonomy)
    return b


def get_headline_finding(df: pd.DataFrame, taxonomy: list[dict]) -> str:
    b = comparative_benchmark(df, taxonomy)
    if b.empty:
        if df.empty:
            return "No data available yet."
        f = theme_frequency(df[~df["theme"].isin(NON_ISSUE_THEMES)], taxonomy)
        if f.empty:
            return "Reviews are overwhelmingly positive — no dominant issue theme."
        return f"{f.iloc[0]['theme_label']} is the leading issue, at {f.iloc[0]['percentage']:.1f}% of all complaints."
    issues = b[(~b["theme"].isin(NON_ISSUE_THEMES)) & b["significant"] & (b["delta"] > 0)]
    if issues.empty:
        issues = b[~b["theme"].isin(NON_ISSUE_THEMES)]
        r = issues.loc[issues["delta"].idxmax()]
        return (f"No app is a statistically significant outlier. Largest gap: {r.app_name} on "
                f"{r.theme_label} ({r.app_percentage:.1f}% vs {r.peer_percentage:.1f}% for peers).")
    r = issues.loc[issues["delta"].idxmax()]
    ratio = r.app_percentage / r.peer_percentage if r.peer_percentage > 0 else np.inf
    ratio_txt = f"{ratio:.1f}× the peer rate" if np.isfinite(ratio) else "while peers have almost none"
    return (f"{r.app_percentage:.1f}% of {r.app_name}'s reviews are about {r.theme_label} — "
            f"{ratio_txt} ({r.peer_percentage:.1f}%), statistically significant (q = {r.q_value:.3f}).")


# ═══════════════════════════════════════════════════════════════
#  KEY DRIVER ANALYSIS
# ═══════════════════════════════════════════════════════════════
def rating_drivers(df: pd.DataFrame, taxonomy: list[dict], app_name: Optional[str] = None) -> pd.DataFrame:
    """
    'Rating drag' per issue theme = share_of_reviews x (avg rating of other reviews - avg rating in theme).
    Read it as: stars the app's average would gain if this theme's reviewers rated like everyone else.
    Welch t-test gives a p-value for the gap.
    """
    d = df if app_name is None else df[df["app_name"] == app_name]
    if d.empty or "theme" not in d:
        return pd.DataFrame()
    rows = []
    for theme, g in d.groupby("theme"):
        if theme in NON_ISSUE_THEMES or len(g) < 3:
            continue
        rest = d[d["theme"] != theme]["rating"]
        gap = rest.mean() - g["rating"].mean()
        share = len(g) / len(d)
        p = stats.ttest_ind(g["rating"], rest, equal_var=False).pvalue if len(rest) > 2 else np.nan
        rows.append({"theme": theme, "reviews": len(g), "share_pct": share * 100,
                     "theme_avg_rating": g["rating"].mean(), "others_avg_rating": rest.mean(),
                     "rating_gap": gap, "rating_drag": share * gap, "p_value": p})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["theme_label"] = _label(out, taxonomy)
    return out.sort_values("rating_drag", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
#  RELEASE IMPACT
# ═══════════════════════════════════════════════════════════════
def version_impact(df: pd.DataFrame, taxonomy: list[dict], min_reviews: int = 15) -> pd.DataFrame:
    """
    Per app x version: avg rating, % 1-2 star, top issue. Each version is compared to the
    previous version with enough reviews (Mann-Whitney U on ratings) to flag regressions.
    """
    if "app_version" not in df or df["app_version"].isna().all():
        return pd.DataFrame()
    labels = theme_labels(taxonomy)
    rows = []
    for app, g in df.dropna(subset=["app_version"]).groupby("app_name"):
        g = g.assign(app_version=g["app_version"].astype(str))
        versions = [v for v, n in g["app_version"].value_counts().items() if n >= min_reviews]
        versions.sort(key=version_key)
        prev = None
        for v in versions:
            vg = g[g["app_version"] == v]
            issues = vg[~vg["theme"].isin(NON_ISSUE_THEMES)]["theme"].value_counts()
            row = {"app_name": app, "version": v, "reviews": len(vg),
                   "first_seen": pd.to_datetime(vg["date"]).min(),
                   "avg_rating": vg["rating"].mean(), "pct_1_2_star": (vg["rating"] <= 2).mean() * 100,
                   "high_sev_pct": (vg["severity"] == "high").mean() * 100,
                   "top_issue": labels.get(issues.index[0], issues.index[0]) if len(issues) else "—",
                   "delta_vs_prev": np.nan, "p_value": np.nan, "flag": ""}
            if prev is not None:
                row["delta_vs_prev"] = row["avg_rating"] - prev["rating"].mean()
                try:
                    row["p_value"] = stats.mannwhitneyu(vg["rating"], prev["rating"]).pvalue
                except ValueError:
                    pass
                if row["p_value"] < 0.05:
                    row["flag"] = "🔻 regression" if row["delta_vs_prev"] < 0 else "🟢 improvement"
            rows.append(row)
            prev = vg
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
#  DEVELOPER RESPONSES
# ═══════════════════════════════════════════════════════════════
def response_analytics(df: pd.DataFrame, taxonomy: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reply rate, reply rate on 1-2★, median hours to reply per app; reply rate by theme."""
    if "reply_text" not in df:
        return pd.DataFrame(), pd.DataFrame()
    d = df[df["platform"].fillna("play") == "play"].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    d["replied"] = d["reply_text"].notna() & (d["reply_text"].astype(str).str.strip() != "")
    d["hours_to_reply"] = (pd.to_datetime(d["replied_at"], errors="coerce")
                           - pd.to_datetime(d["date"], errors="coerce")).dt.total_seconds() / 3600
    per_app = d.groupby("app_name").apply(lambda g: pd.Series({
        "reviews": len(g),
        "reply_rate": g["replied"].mean() * 100,
        "reply_rate_negative": g.loc[g["rating"] <= 2, "replied"].mean() * 100 if (g["rating"] <= 2).any() else np.nan,
        "reply_rate_high_sev": g.loc[g["severity"] == "high", "replied"].mean() * 100 if (g["severity"] == "high").any() else np.nan,
        "median_hours": g.loc[g["replied"], "hours_to_reply"].median(),
    }), include_groups=False).reset_index()
    by_theme = (d.groupby(["app_name", "theme"])["replied"].agg(["mean", "size"]).reset_index()
                 .rename(columns={"mean": "reply_rate", "size": "reviews"}))
    by_theme["reply_rate"] *= 100
    by_theme["theme_label"] = _label(by_theme, taxonomy)
    return per_app, by_theme


# ═══════════════════════════════════════════════════════════════
#  SPIKE ALERTS
# ═══════════════════════════════════════════════════════════════
def detect_spikes(df: pd.DataFrame, taxonomy: list[dict], z_threshold: float = 2.0, min_count: int = 5) -> pd.DataFrame:
    """
    For each app x issue theme, compare the latest time bucket's share against the
    baseline of earlier buckets (z-score). Needs at least 4 buckets of history.
    """
    if df.empty or "date" not in df:
        return pd.DataFrame()
    freq, unit = pick_time_freq(df["date"])
    t = df.assign(bucket=_bucket(df["date"], freq)).dropna(subset=["bucket"])
    rows = []
    for app, g in t.groupby("app_name"):
        totals = g.groupby("bucket").size()
        if len(totals) < 4:
            continue
        latest = totals.index.max()
        for theme in g["theme"].dropna().unique():
            if theme in NON_ISSUE_THEMES:
                continue
            counts = g[g["theme"] == theme].groupby("bucket").size().reindex(totals.index, fill_value=0)
            share = counts / totals
            base = share.drop(latest)
            sd = base.std(ddof=1)
            z = (share[latest] - base.mean()) / sd if sd > 0 else 0
            if z >= z_threshold and counts[latest] >= min_count:
                rows.append({"app_name": app, "theme": theme, "bucket": latest, "unit": unit,
                             "latest_share": share[latest] * 100, "baseline_share": base.mean() * 100,
                             "latest_count": int(counts[latest]), "z": z})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["theme_label"] = _label(out, taxonomy)
        out = out.sort_values("z", ascending=False)
    return out


# ═══════════════════════════════════════════════════════════════
#  TEXT ANALYTICS
# ═══════════════════════════════════════════════════════════════
def distinctive_terms(df: pd.DataFrame, top_n: int = 15, max_rating: int = 3) -> pd.DataFrame:
    """
    Words/phrases over-represented in each app's negative reviews vs other apps.
    Weighted log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008),
    reported as z-scores — robust for rare terms, unlike raw frequency or TF-IDF.
    """
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

    d = df[(df["rating"] <= max_rating) & df["review_text"].notna()]
    if d["app_name"].nunique() < 2 or len(d) < 30:
        return pd.DataFrame()
    stop = list(ENGLISH_STOP_WORDS | {"app", "apps", "please", "use", "using", "just", "dont", "don", "im", "ve", "nhi", "hai", "ka", "ki", "ke"})
    vec = CountVectorizer(ngram_range=(1, 2), min_df=3, stop_words=stop, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    try:
        X = vec.fit_transform(d["review_text"].astype(str).str.lower())
    except ValueError:
        return pd.DataFrame()
    terms = np.array(vec.get_feature_names_out())
    apps = d["app_name"].values
    prior = np.asarray(X.sum(axis=0)).ravel() + 0.01
    a0 = prior.sum()
    rows = []
    for app in np.unique(apps):
        yi = np.asarray(X[apps == app].sum(axis=0)).ravel()
        yj = np.asarray(X[apps != app].sum(axis=0)).ravel()
        ni, nj = yi.sum(), yj.sum()
        delta = (np.log((yi + prior) / (ni + a0 - yi - prior)) - np.log((yj + prior) / (nj + a0 - yj - prior)))
        z = delta / np.sqrt(1 / (yi + prior) + 1 / (yj + prior))
        for i in np.argsort(-z)[:top_n]:
            if yi[i] >= 3:
                rows.append({"app_name": app, "term": terms[i], "z": z[i], "mentions": int(yi[i])})
    return pd.DataFrame(rows)


def feature_request_clusters(df: pd.DataFrame, threshold: float = 0.45) -> pd.DataFrame:
    """Group near-duplicate feature requests (TF-IDF cosine, greedy) and rank by mentions + helpful votes."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if "feature_request" not in df:
        return pd.DataFrame()
    d = df[df["feature_request"].notna() & (df["feature_request"].astype(str).str.strip() != "")].copy()
    if d.empty:
        return pd.DataFrame()
    d["fr"] = d["feature_request"].astype(str).str.strip().str.lower().str.rstrip(".")
    uniq = d["fr"].unique()
    if len(uniq) == 1:
        assign = {uniq[0]: 0}
    else:
        sim = cosine_similarity(TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(uniq))
        assign, cid = {}, 0
        for i in np.argsort(-d["fr"].value_counts().reindex(uniq).values):
            if uniq[i] in assign:
                continue
            for j in np.where(sim[i] >= threshold)[0]:
                assign.setdefault(uniq[j], cid)
            cid += 1
    d["cluster"] = d["fr"].map(assign)
    out = d.groupby("cluster").agg(
        request=("feature_request", lambda s: s.value_counts().index[0]),
        mentions=("fr", "size"),
        helpful_votes=("thumbs_up_count", "sum"),
        apps=("app_name", lambda s: ", ".join(f"{a} ({n})" for a, n in s.value_counts().items())),
        avg_rating=("rating", "mean"),
        examples=("review_text", lambda s: " | ".join(str(x)[:160] for x in s.head(3))),
    ).reset_index(drop=True)
    out["demand_score"] = out["mentions"] + np.log1p(out["helpful_votes"].fillna(0))
    return out.sort_values("demand_score", ascending=False).reset_index(drop=True)


def rating_sentiment_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """'Hidden detractors' (4-5★ but negative text) and 'grumpy fans' (1-2★ but positive text)."""
    if "sentiment_score" not in df:
        return pd.DataFrame()
    m = df[((df["rating"] >= 4) & (df["sentiment_score"] <= -0.3)) |
           ((df["rating"] <= 2) & (df["sentiment_score"] >= 0.3))].copy()
    m["type"] = np.where(m["rating"] >= 4, "Hidden detractor (high stars, negative text)",
                         "Grumpy fan (low stars, positive text)")
    return m


def voice_of_customer(df: pd.DataFrame, theme: str, app_name: Optional[str] = None, n: int = 5) -> pd.DataFrame:
    """Most-helpful, substantive reviews for a theme — quotable evidence for a PRD or exec deck."""
    d = df[df["theme"] == theme]
    if app_name:
        d = d[d["app_name"] == app_name]
    d = d[d["review_text"].astype(str).str.len() > 40]
    sev = d["severity"].map({"high": 2, "medium": 1, "low": 0}).fillna(0)
    return d.assign(_s=d["thumbs_up_count"].fillna(0) * 2 + sev).sort_values("_s", ascending=False).head(n)


# ═══════════════════════════════════════════════════════════════
#  PRIORITY PLAN
# ═══════════════════════════════════════════════════════════════
def compute_priority_scores(df: pd.DataFrame, taxonomy: list[dict]) -> pd.DataFrame:
    """
    Priority Score = Impact x 4 + Frequency/Severity Gap x 4 - Effort x 2
      Impact / Effort: from the workspace taxonomy (editable)
      Freq/Severity Gap: severity-weighted volume, scaled 0-10 against the worst theme
    Reach adds context: helpful votes = other users who agree.
    """
    if df.empty or "severity" not in df:
        return pd.DataFrame()
    tax = {t["key"]: t for t in taxonomy}
    issues = df[~df["theme"].isin({"praise_no_issue"})]
    sc = issues.groupby(["theme", "severity"]).size().unstack(fill_value=0)
    for lvl in SEVERITY_WEIGHTS:
        if lvl not in sc:
            sc[lvl] = 0
    raw = sum(sc[l] * w for l, w in SEVERITY_WEIGHTS.items())
    res = pd.DataFrame({
        "theme": sc.index,
        "reviews": sc[["high", "medium", "low"]].sum(axis=1).values,
        "high_sev_pct": (sc["high"] / sc[["high", "medium", "low"]].sum(axis=1) * 100).values,
        "frequency_severity_gap": (raw / raw.max() * 10).values if raw.max() > 0 else 0,
    })
    reach = issues.groupby("theme")["thumbs_up_count"].sum()
    res["helpful_votes"] = res["theme"].map(reach).fillna(0).astype(int)
    res["share_pct"] = res["reviews"] / len(df) * 100
    res["impact"] = res["theme"].map(lambda k: tax.get(k, {}).get("impact", 3))
    res["effort"] = res["theme"].map(lambda k: tax.get(k, {}).get("effort", 5))
    res["priority_score"] = res["impact"] * 4 + res["frequency_severity_gap"] * 4 - res["effort"] * 2
    res["quick_win"] = (res["impact"] >= 6) & (res["effort"] <= 4)
    res["theme_label"] = _label(res, taxonomy)

    def why(r):
        if r.quick_win:
            return "High impact, low effort — ship first."
        if r.frequency_severity_gap >= 7:
            return "Highest volume x severity in the dataset — a top user pain."
        if r.impact >= 8:
            return "High business impact; plan a focused investment."
        if r.frequency_severity_gap < 2:
            return "Low volume — monitor, don't prioritise yet."
        return "Standard backlog priority."

    res["justification"] = res.apply(why, axis=1)
    return res.sort_values("priority_score", ascending=False).reset_index(drop=True)
