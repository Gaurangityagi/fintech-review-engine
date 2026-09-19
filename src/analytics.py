import pandas as pd
import numpy as np
from typing import Optional

# ─── Theme Display Names ──────────────────────────────────────────
THEME_LABELS = {
    "failed_transaction": "Failed Transaction",
    "kyc_friction": "KYC Friction",
    "refund_delay": "Refund Delay",
    "customer_support": "Customer Support",
    "app_crash_bug": "App Crash / Bug",
    "ui_confusion": "UI Confusion",
    "fraud_security_concern": "Fraud / Security",
    "praise_no_issue": "Praise / No Issue",
    "other": "Other"
}

SEVERITY_WEIGHTS = {"high": 3.0, "medium": 1.5, "low": 1.0}

# ─── Impact and Effort Assumptions for Priority Score ─────────────
THEME_IMPACT = {
    "failed_transaction": 10,   # Blocks core money movement
    "fraud_security_concern": 9, # Trust/safety critical
    "refund_delay": 8,           # Money stuck
    "kyc_friction": 7,           # Blocks onboarding
    "customer_support": 6,       # Degrades trust
    "app_crash_bug": 7,          # Blocks usage
    "ui_confusion": 5,           # Degrades experience
    "praise_no_issue": 0,        # Not an issue
    "other": 3                   # Generic
}

THEME_EFFORT = {
    "failed_transaction": 9,    # Backend infra, payment gateway
    "fraud_security_concern": 8, # Security team, fraud detection
    "refund_delay": 6,           # Process + backend
    "kyc_friction": 5,           # UX + compliance workflow
    "customer_support": 4,       # Training + tooling
    "app_crash_bug": 6,          # Engineering debugging
    "ui_confusion": 3,           # UX redesign, relatively low effort
    "praise_no_issue": 0,
    "other": 5
}

def theme_frequency(df: pd.DataFrame, app_name: Optional[str] = None) -> pd.DataFrame:
    """
    Count and percentage share of each theme.
    If app_name is provided, filter to that app.
    Returns DataFrame with columns: theme, theme_label, count, percentage
    Sorted by count descending.
    """
    if df.empty or "theme" not in df.columns:
        return pd.DataFrame(columns=["theme", "theme_label", "count", "percentage"])
    
    if app_name:
        df = df[df["app_name"] == app_name]
        
    counts = df["theme"].value_counts().reset_index()
    counts.columns = ["theme", "count"]
    
    total = counts["count"].sum()
    counts["percentage"] = (counts["count"] / total * 100) if total > 0 else 0.0
    counts["theme_label"] = counts["theme"].map(THEME_LABELS).fillna(counts["theme"])
    
    return counts[["theme", "theme_label", "count", "percentage"]].sort_values("count", ascending=False).reset_index(drop=True)

def severity_weighted_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute severity-weighted score per app and theme.
    Formula: (high_count * 3) + (medium_count * 1.5) + (low_count * 1)
    Normalize per 100 reviews for fair comparison.
    Returns DataFrame with columns: app_name, theme, theme_label, 
    high_count, medium_count, low_count, raw_score, normalized_score
    """
    if df.empty or "app_name" not in df.columns or "theme" not in df.columns or "severity" not in df.columns:
        return pd.DataFrame(columns=["app_name", "theme", "theme_label", "high_count", "medium_count", "low_count", "raw_score", "normalized_score"])

    # Aggregate counts of severity per app and theme
    severity_counts = df.groupby(["app_name", "theme", "severity"]).size().unstack(fill_value=0).reset_index()
    
    for level in ["high", "medium", "low"]:
        if level not in severity_counts.columns:
            severity_counts[level] = 0
            
    severity_counts = severity_counts.rename(columns={"high": "high_count", "medium": "medium_count", "low": "low_count"})
    
    # Raw score
    severity_counts["raw_score"] = (
        severity_counts["high_count"] * SEVERITY_WEIGHTS["high"] +
        severity_counts["medium_count"] * SEVERITY_WEIGHTS["medium"] +
        severity_counts["low_count"] * SEVERITY_WEIGHTS["low"]
    )
    
    # Normalize per 100 reviews per app
    total_reviews_per_app = df.groupby("app_name").size().reset_index(name="total_app_reviews")
    severity_counts = severity_counts.merge(total_reviews_per_app, on="app_name", how="left")
    
    severity_counts["normalized_score"] = np.where(
        severity_counts["total_app_reviews"] > 0,
        (severity_counts["raw_score"] / severity_counts["total_app_reviews"]) * 100,
        0.0
    )
    
    severity_counts["theme_label"] = severity_counts["theme"].map(THEME_LABELS).fillna(severity_counts["theme"])
    
    cols = ["app_name", "theme", "theme_label", "high_count", "medium_count", "low_count", "raw_score", "normalized_score"]
    # return only columns that are present, just in case
    return severity_counts[[c for c in cols if c in severity_counts.columns]]

def trend_over_time(df: pd.DataFrame, theme: Optional[str] = None) -> pd.DataFrame:
    """
    Theme frequency by month.
    If theme provided, filter to that theme.
    Returns DataFrame with columns: month, app_name, theme, count
    The date column should be parsed and grouped by month (YYYY-MM format).
    """
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=["month", "app_name", "theme", "count"])
        
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['date'], errors='coerce')
    df_temp = df_temp.dropna(subset=['date'])
    df_temp['month'] = df_temp['date'].dt.to_period('M').astype(str)
    
    if theme:
        df_temp = df_temp[df_temp["theme"] == theme]
        
    if df_temp.empty:
        return pd.DataFrame(columns=["month", "app_name", "theme", "count"])
        
    counts = df_temp.groupby(["month", "app_name", "theme"]).size().reset_index(name="count")
    return counts

def comparative_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare each app's theme distribution to the category average.
    Returns DataFrame with columns: app_name, theme, theme_label,
    app_percentage, category_avg_percentage, delta (app - avg),
    status ('above_avg', 'below_avg', 'at_avg')
    """
    if df.empty or "app_name" not in df.columns or "theme" not in df.columns:
        return pd.DataFrame(columns=["app_name", "theme", "theme_label", "app_percentage", "category_avg_percentage", "delta", "status"])
        
    app_theme_counts = df.groupby(["app_name", "theme"]).size().reset_index(name="count")
    app_totals = app_theme_counts.groupby("app_name")["count"].sum().reset_index(name="app_total")
    
    app_stats = app_theme_counts.merge(app_totals, on="app_name")
    app_stats["app_percentage"] = (app_stats["count"] / app_stats["app_total"]) * 100
    
    # Category average
    category_counts = df.groupby("theme").size().reset_index(name="cat_count")
    total_reviews = len(df)
    category_counts["category_avg_percentage"] = (category_counts["cat_count"] / total_reviews) * 100 if total_reviews > 0 else 0
    
    benchmark = app_stats.merge(category_counts[["theme", "category_avg_percentage"]], on="theme")
    benchmark["delta"] = benchmark["app_percentage"] - benchmark["category_avg_percentage"]
    
    def get_status(val):
        if val > 1.0: return "above_avg"
        elif val < -1.0: return "below_avg"
        else: return "at_avg"
        
    benchmark["status"] = benchmark["delta"].apply(get_status)
    benchmark["theme_label"] = benchmark["theme"].map(THEME_LABELS).fillna(benchmark["theme"])
    
    return benchmark[["app_name", "theme", "theme_label", "app_percentage", "category_avg_percentage", "delta", "status"]]

def compute_priority_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute priority scores per theme.
    
    Priority Score = (Impact × 4) + (Frequency_Severity_Gap × 4) − (Effort × 2)
    """
    if df.empty or "theme" not in df.columns or "severity" not in df.columns:
        return pd.DataFrame(columns=["theme", "theme_label", "impact", "frequency_severity_gap", "effort", "priority_score", "quick_win", "justification"])
        
    # Calculate frequency_severity_gap (normalized 0-10)
    severity_counts = df.groupby(["theme", "severity"]).size().unstack(fill_value=0).reset_index()
    for level in ["high", "medium", "low"]:
        if level not in severity_counts.columns:
            severity_counts[level] = 0
            
    severity_counts["raw_score"] = (
        severity_counts["high"] * SEVERITY_WEIGHTS["high"] +
        severity_counts["medium"] * SEVERITY_WEIGHTS["medium"] +
        severity_counts["low"] * SEVERITY_WEIGHTS["low"]
    )
    
    max_score = severity_counts["raw_score"].max()
    if max_score > 0:
        severity_counts["frequency_severity_gap"] = (severity_counts["raw_score"] / max_score) * 10
    else:
        severity_counts["frequency_severity_gap"] = 0
        
    res = severity_counts[["theme", "frequency_severity_gap"]].copy()
    res = res[res["theme"] != "praise_no_issue"]
    
    res["theme_label"] = res["theme"].map(THEME_LABELS).fillna(res["theme"])
    res["impact"] = res["theme"].map(THEME_IMPACT).fillna(3)
    res["effort"] = res["theme"].map(THEME_EFFORT).fillna(5)
    
    res["priority_score"] = (res["impact"] * 4) + (res["frequency_severity_gap"] * 4) - (res["effort"] * 2)
    res["quick_win"] = (res["impact"] >= 6) & (res["effort"] <= 4)
    
    def generate_justification(row):
        if row["quick_win"]:
            return "High impact, low effort. Optimal target for immediate action."
        elif row["priority_score"] >= 40:
            return "Critical priority due to high frequency and severity."
        elif row["impact"] >= 8:
            return "High impact issue, requires focused investment."
        else:
            return "Standard prioritization."
            
    res["justification"] = res.apply(generate_justification, axis=1)
    
    res = res.sort_values("priority_score", ascending=False).reset_index(drop=True)
    
    return res[["theme", "theme_label", "impact", "frequency_severity_gap", "effort", "priority_score", "quick_win", "justification"]]

def get_overview_stats(df: pd.DataFrame) -> dict:
    """
    Compute high-level overview statistics:
    - total_reviews: int
    - per_app_counts: dict of app_name -> count
    - date_range: (min_date, max_date)
    - avg_rating_per_app: dict of app_name -> float
    - top_theme_per_app: dict of app_name -> theme
    - severity_distribution: dict of severity -> count
    - avg_sentiment: float
    """
    if df.empty:
        return {
            "total_reviews": 0,
            "per_app_counts": {},
            "date_range": (None, None),
            "avg_rating_per_app": {},
            "top_theme_per_app": {},
            "severity_distribution": {},
            "avg_sentiment": 0.0
        }
        
    stats = {}
    stats["total_reviews"] = int(len(df))
    stats["per_app_counts"] = df["app_name"].value_counts().to_dict() if "app_name" in df.columns else {}
    
    if "date" in df.columns:
        df_date = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not df_date.empty:
            stats["date_range"] = (df_date.min().strftime("%Y-%m-%d"), df_date.max().strftime("%Y-%m-%d"))
        else:
            stats["date_range"] = (None, None)
    else:
        stats["date_range"] = (None, None)
        
    if "rating" in df.columns and "app_name" in df.columns:
        stats["avg_rating_per_app"] = df.groupby("app_name")["rating"].mean().to_dict()
    else:
        stats["avg_rating_per_app"] = {}
        
    if "theme" in df.columns and "app_name" in df.columns:
        top_themes = df.groupby(["app_name", "theme"]).size().reset_index(name="count")
        top_themes = top_themes.sort_values(["app_name", "count"], ascending=[True, False]).drop_duplicates("app_name")
        stats["top_theme_per_app"] = dict(zip(top_themes["app_name"], top_themes["theme"]))
    else:
        stats["top_theme_per_app"] = {}
        
    if "severity" in df.columns:
        stats["severity_distribution"] = df["severity"].value_counts().to_dict()
    else:
        stats["severity_distribution"] = {}
        
    if "sentiment_score" in df.columns:
        stats["avg_sentiment"] = float(df["sentiment_score"].mean())
    else:
        stats["avg_sentiment"] = 0.0
        
    return stats

def get_headline_finding(df: pd.DataFrame) -> str:
    """
    Generate a data-driven headline finding string.
    """
    if df.empty:
        return "No data available to generate insights."
    
    benchmark = comparative_benchmark(df)
    if benchmark.empty:
        return "Not enough data for insights."
        
    try:
        # Filter out 'praise_no_issue' and 'other' if we want real issues to headline
        issues = benchmark[~benchmark["theme"].isin(["praise_no_issue", "other"])]
        if issues.empty:
            issues = benchmark
            
        max_delta_row = issues.loc[issues['delta'].idxmax()]
        app = max_delta_row['app_name']
        theme_label = max_delta_row['theme_label']
        pct = max_delta_row['app_percentage']
        
        return f"{pct:.1f}% of {app}'s reviews are related to {theme_label}, which is significantly higher than the category average."
    except Exception:
        return "No distinct outliers found in the current dataset."
