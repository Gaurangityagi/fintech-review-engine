import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import time
from pathlib import Path
from datetime import datetime

# --- SETTINGS & CONFIG ---
st.set_page_config(
    page_title="Fintech Review Intelligence Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS ---
THEME_TAXONOMY = [
    "failed_transaction", "kyc_friction", "refund_delay",
    "customer_support", "app_crash_bug", "ui_confusion",
    "fraud_security_concern", "praise_no_issue", "other"
]

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

APP_COLORS = {
    "PhonePe": "#6366F1",
    "Paytm": "#06B6D4",
    "Google Pay": "#10B981"
}

SEVERITY_COLORS = {
    "high": "#EF4444",
    "medium": "#F59E0B",
    "low": "#22C55E"
}

CLASSIFIED_DATA_PATH = Path("data/classified/classified_reviews.csv")

# --- CUSTOM CSS ---
def inject_custom_css():
    st.markdown("""
    <style>
    /* Premium dark editorial design */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Cards */
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.5rem;
    }

    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    /* Buttons */
    .stButton>button {
        background-color: #6366F1;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #4F46E5;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 0.875rem;
        font-weight: 600;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
    }
    .badge-green { background-color: rgba(34, 197, 94, 0.2); color: #4ADE80; }
    .badge-yellow { background-color: rgba(245, 158, 11, 0.2); color: #FBBF24; }
    .badge-red { background-color: rgba(239, 68, 68, 0.2); color: #F87171; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0B1120;
        border-right: 1px solid #1E293B;
    }

    /* Data tables */
    .dataframe {
        background-color: #1E293B;
        color: #F8FAFC;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #94A3B8;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- IMPORTS WITH GRACEFUL FALLBACK ---
try:
    from src.scraper import load_raw_reviews, scrape_all_apps
except ImportError:
    def load_raw_reviews():
        return pd.DataFrame()
    def scrape_all_apps(target_count=750):
        return {}

try:
    from src.llm_classify import classify_dataframe
except ImportError:
    def classify_dataframe(df, batch_size=15, progress_callback=None):
        return df

try:
    from src.analytics import (
        theme_frequency,
        severity_weighted_score,
        compute_priority_scores,
        comparative_benchmark,
        trend_over_time,
        get_overview_stats,
        get_headline_finding,
        THEME_LABELS as ANALYTICS_THEME_LABELS
    )
except ImportError:
    def theme_frequency(df, app_name=None):
        return pd.DataFrame()
    def severity_weighted_score(df):
        return pd.DataFrame()
    def compute_priority_scores(df):
        return pd.DataFrame()
    def comparative_benchmark(df):
        return pd.DataFrame()
    def trend_over_time(df, theme=None):
        return pd.DataFrame()
    def get_overview_stats(df):
        return {"total_reviews": 0, "per_app_counts": {}, "date_range": (None, None),
                "avg_rating_per_app": {}, "top_theme_per_app": {}, "severity_distribution": {},
                "avg_sentiment": 0.0}
    def get_headline_finding(df):
        return "Run the pipeline to generate insights."

try:
    from src.validation import compute_agreement, load_manual_labels
except ImportError:
    def compute_agreement(llm_df, manual_df):
        return {}
    def load_manual_labels():
        return pd.DataFrame()


# --- DATA LOADING ---
@st.cache_data
def load_data():
    """Load classified data if available, otherwise load raw data with cached classifications applied."""
    try:
        # 1. Try classified data first
        if CLASSIFIED_DATA_PATH.exists():
            df = pd.read_csv(CLASSIFIED_DATA_PATH)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            return df

        # 2. Fall back to raw scraped data + apply any cached classifications
        df = load_raw_reviews()
        if not df.empty:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # Apply cached LLM classifications if cache exists
            cache_path = Path("data/classified/cache.json")
            if cache_path.exists() and 'review_text' in df.columns:
                import hashlib
                try:
                    cache = json.load(open(cache_path, "r", encoding="utf-8"))
                    if cache:
                        themes, severities, sentiments, summaries = [], [], [], []
                        matched = 0
                        for _, row in df.iterrows():
                            h = hashlib.sha256(str(row['review_text']).encode('utf-8')).hexdigest()
                            if h in cache:
                                matched += 1
                                entry = cache[h]
                                themes.append(entry.get('theme', 'other'))
                                severities.append(entry.get('severity', 'medium'))
                                sentiments.append(entry.get('sentiment_score', 0.0))
                                summaries.append(entry.get('one_line_summary', ''))
                            else:
                                themes.append(None)
                                severities.append(None)
                                sentiments.append(None)
                                summaries.append(None)

                        if matched > 0:
                            df['theme'] = themes
                            df['severity'] = severities
                            df['sentiment_score'] = sentiments
                            df['one_line_summary'] = summaries
                            # Drop rows with no classification so analytics work correctly
                            df = df.dropna(subset=['theme']).reset_index(drop=True)
                            # Auto-save so future loads are fast
                            CLASSIFIED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                            df.to_csv(CLASSIFIED_DATA_PATH, index=False)
                except Exception:
                    pass  # If cache reading fails, just return raw data

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


@st.cache_data
def load_validation_metrics():
    """Load validation agreement metrics if manual labels and classified data exist."""
    try:
        manual_df = load_manual_labels()
        if manual_df.empty:
            return None

        # Check if manual labels are filled in
        if manual_df['manual_theme'].isna().all() or (manual_df['manual_theme'] == '').all():
            return None

        # Load classified data to compare
        if CLASSIFIED_DATA_PATH.exists():
            llm_df = pd.read_csv(CLASSIFIED_DATA_PATH)
            return compute_agreement(llm_df, manual_df)
    except Exception:
        pass
    return None


# --- PLOTLY HELPERS ---
def get_chart_layout(title=""):
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(size=16, color="#F8FAFC")),
        margin=dict(t=50, l=20, r=20, b=20),
        font=dict(color="#94A3B8", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )


# ═══════════════════════════════════════════════════════════════
#  PAGE 0 — OVERVIEW
# ═══════════════════════════════════════════════════════════════
def render_overview(df):
    st.title("Fintech Review Intelligence Engine")
    st.markdown("##### AI-Powered Analysis of Indian UPI App Reviews")

    if df.empty:
        st.info("👋 **No data available yet.** Click **Run Pipeline** in the sidebar to scrape reviews and classify them.")
        st.markdown("""
        ### How it works
        1. **Scrape** — Pull 500–1,000 recent reviews per app from Google Play Store
        2. **Classify** — LLM reads each review, assigns a theme + severity + sentiment
        3. **Validate** — Agreement metrics vs. 50 hand-labeled reviews
        4. **Analyze** — Severity-weighted scoring, trend detection, cross-app benchmarking
        5. **Prioritize** — Impact × Frequency − Effort → ranked action plan
        """)
        return

    stats = get_overview_stats(df)

    # --- Top-level metrics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reviews", f"{stats['total_reviews']:,}")
    with col2:
        st.metric("Apps Analyzed", len(stats.get('per_app_counts', {})))
    with col3:
        dr = stats.get('date_range', (None, None))
        date_str = f"{dr[0]} → {dr[1]}" if dr[0] else "N/A"
        st.metric("Date Range", date_str)

    st.markdown("---")

    # --- Average ratings + Validation ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### Average Rating per App")
        avg_ratings = stats.get('avg_rating_per_app', {})
        if avg_ratings:
            cols = st.columns(len(avg_ratings))
            for i, (app, rating) in enumerate(avg_ratings.items()):
                color = APP_COLORS.get(app, "#94A3B8")
                with cols[i]:
                    st.markdown(f"""
                    <div style="background-color: #1E293B; border-top: 4px solid {color};
                                padding: 1rem; border-radius: 8px; text-align: center;">
                        <p style="margin: 0; color: #94A3B8; font-size: 0.875rem;">{app}</p>
                        <h2 style="margin: 0; color: #F8FAFC;">{rating:.2f} ⭐</h2>
                    </div>
                    """, unsafe_allow_html=True)

    with col_right:
        st.markdown("#### LLM Validation Accuracy")
        val = load_validation_metrics()
        if val and val.get('theme_accuracy') is not None:
            acc = val['theme_accuracy']
            kappa = val.get('theme_kappa', 0)
            badge_class = "badge-green" if acc > 0.8 else ("badge-yellow" if acc >= 0.6 else "badge-red")
            st.markdown(f"""
            <div style="background-color: #1E293B; padding: 1.5rem; border-radius: 8px; text-align: center;">
                <div style="font-size: 2.5rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.25rem;">
                    {acc*100:.1f}%
                </div>
                <p style="color: #94A3B8; margin: 0.25rem 0;">Theme Agreement</p>
                <span class="badge {badge_class}">Cohen's κ: {kappa:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Validation metrics not yet computed. Classify reviews first, then run validation.")

    st.markdown("---")

    # --- Headline insight ---
    st.markdown("#### 💡 Key Insight")
    headline = get_headline_finding(df)
    st.info(headline)

    # --- Methodology ---
    with st.expander("📘 Methodology"):
        st.markdown("""
        **Pipeline**: Scrape → Classify → Validate → Analyze → Prioritize

        - **Scraping**: `google-play-scraper` pulls recent reviews from Google Play Store
        - **Classification**: Reviews batched (15 per call) to GPT-4o-mini / Claude, each tagged with theme + severity + sentiment
        - **Caching**: SHA-256 hash of review text → classified JSON on disk (no re-spending tokens)
        - **Validation**: 50 hand-labeled reviews compared via accuracy + Cohen's κ
        - **Priority Score**: `(Impact × 4) + (Freq/Severity Gap × 4) − (Fix Effort × 2)`
        """)


# ═══════════════════════════════════════════════════════════════
#  PAGE 1 — THEME BREAKDOWN
# ═══════════════════════════════════════════════════════════════
def render_theme_breakdown(df):
    st.title("Theme Breakdown")

    if df.empty or 'theme' not in df.columns:
        st.warning("⚠️ Classified data not available. Run the pipeline first.")
        return

    # App selector
    apps_available = sorted(df['app_name'].unique().tolist())
    selected_apps = st.multiselect("Select Apps", apps_available, default=apps_available)
    filtered = df[df['app_name'].isin(selected_apps)]

    if filtered.empty:
        st.info("No data for selected apps.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Theme Frequency per App")
        theme_counts = filtered.groupby(['app_name', 'theme']).size().reset_index(name='count')
        theme_counts['theme_label'] = theme_counts['theme'].map(THEME_LABELS).fillna(theme_counts['theme'])
        fig_bar = px.bar(
            theme_counts,
            y="theme_label",
            x="count",
            color="app_name",
            barmode="group",
            orientation='h',
            color_discrete_map=APP_COLORS,
            labels={"count": "Review Count", "theme_label": "Theme", "app_name": "App"}
        )
        fig_bar.update_layout(**get_chart_layout())
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("#### Theme Distribution (Treemap)")
        tree_data = filtered.groupby(['app_name', 'theme']).size().reset_index(name='count')
        tree_data['theme_label'] = tree_data['theme'].map(THEME_LABELS).fillna(tree_data['theme'])
        fig_tree = px.treemap(
            tree_data,
            path=['app_name', 'theme_label'],
            values='count',
            color='count',
            color_continuous_scale='Blues'
        )
        fig_tree.update_layout(**get_chart_layout())
        st.plotly_chart(fig_tree, use_container_width=True)

    # Review explorer
    st.markdown("#### 📋 Review Explorer")
    selected_theme = st.selectbox("Filter by Theme", ["All"] + THEME_TAXONOMY)
    display = filtered if selected_theme == "All" else filtered[filtered['theme'] == selected_theme]

    cols_to_show = ['app_name', 'rating', 'theme', 'severity', 'one_line_summary', 'review_text']
    available_cols = [c for c in cols_to_show if c in display.columns]

    st.dataframe(
        display[available_cols].head(200),
        use_container_width=True,
        height=400,
        column_config={
            "app_name": "App",
            "rating": st.column_config.NumberColumn("Rating", format="⭐ %d"),
            "theme": "Theme",
            "severity": "Severity",
            "one_line_summary": "Summary",
            "review_text": "Full Review"
        }
    )


# ═══════════════════════════════════════════════════════════════
#  PAGE 2 — SEVERITY & TREND
# ═══════════════════════════════════════════════════════════════
def render_severity_trend(df):
    st.title("Severity & Trends")

    if df.empty or 'severity' not in df.columns:
        st.warning("⚠️ Classified data not available. Run the pipeline first.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Severity-Weighted Scores")
        sws = severity_weighted_score(df)
        if not sws.empty:
            # Aggregate per app
            app_scores = sws.groupby('app_name')['normalized_score'].sum().reset_index()
            app_scores = app_scores.sort_values('normalized_score', ascending=True)
            fig_sev = px.bar(
                app_scores,
                y="app_name",
                x="normalized_score",
                orientation='h',
                color="app_name",
                color_discrete_map=APP_COLORS,
                labels={"normalized_score": "Severity Score (per 100 reviews)", "app_name": "App"}
            )
            fig_sev.update_layout(**get_chart_layout())
            st.plotly_chart(fig_sev, use_container_width=True)

        st.markdown("#### Severity Distribution")
        sev_counts = df.groupby(['app_name', 'severity']).size().reset_index(name='count')
        fig_stack = px.bar(
            sev_counts,
            x="app_name",
            y="count",
            color="severity",
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
            labels={"count": "Reviews", "app_name": "App", "severity": "Severity"}
        )
        fig_stack.update_layout(**get_chart_layout())
        st.plotly_chart(fig_stack, use_container_width=True)

    with col2:
        st.markdown("#### Theme Trends Over Time")
        if 'date' in df.columns:
            selected_theme = st.selectbox("Focus Theme", THEME_TAXONOMY, index=0)
            trend_data = trend_over_time(df, theme=selected_theme)
            if not trend_data.empty:
                fig_trend = px.line(
                    trend_data,
                    x="month",
                    y="count",
                    color="app_name",
                    color_discrete_map=APP_COLORS,
                    markers=True,
                    labels={"count": "Reviews", "month": "Month", "app_name": "App"}
                )
                fig_trend.update_layout(**get_chart_layout(f"Monthly trend: {THEME_LABELS.get(selected_theme, selected_theme)}"))
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No trend data available for this theme.")
        else:
            st.info("Date information not available for trend analysis.")

    # Heatmap
    st.markdown("#### Severity Heatmap (% High Severity by Theme)")
    heat_data = df.groupby(['app_name', 'theme']).apply(
        lambda x: (x['severity'] == 'high').mean() * 100
    ).reset_index(name='high_sev_pct')
    heat_data['theme_label'] = heat_data['theme'].map(THEME_LABELS).fillna(heat_data['theme'])
    heat_pivot = heat_data.pivot(index='theme_label', columns='app_name', values='high_sev_pct').fillna(0)

    fig_heat = px.imshow(
        heat_pivot,
        labels=dict(x="App", y="Theme", color="% High Severity"),
        color_continuous_scale="Reds",
        aspect="auto"
    )
    fig_heat.update_layout(**get_chart_layout())
    st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 3 — COMPARATIVE BENCHMARK
# ═══════════════════════════════════════════════════════════════
def render_benchmark(df):
    st.title("Comparative Benchmark")

    if df.empty or 'theme' not in df.columns:
        st.warning("⚠️ Data not available.")
        return

    benchmark_data = comparative_benchmark(df)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Radar: App vs Category Average")
        app_to_compare = st.selectbox("Select App to Benchmark", sorted(df['app_name'].unique()))

        if not benchmark_data.empty:
            app_bench = benchmark_data[benchmark_data['app_name'] == app_to_compare]
            # Get category averages for all themes
            avg_data = benchmark_data.groupby('theme')[['category_avg_percentage']].first().reset_index()

            radar_merged = app_bench.merge(avg_data[['theme', 'category_avg_percentage']], on='theme', how='right', suffixes=('', '_avg'))
            radar_merged['app_percentage'] = radar_merged['app_percentage'].fillna(0)
            radar_merged['theme_label'] = radar_merged['theme'].map(THEME_LABELS).fillna(radar_merged['theme'])

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=radar_merged['app_percentage'],
                theta=radar_merged['theme_label'],
                fill='toself',
                name=app_to_compare,
                line_color=APP_COLORS.get(app_to_compare, "#6366F1")
            ))
            fig.add_trace(go.Scatterpolar(
                r=radar_merged['category_avg_percentage'],
                theta=radar_merged['theme_label'],
                fill='toself',
                name='Category Average',
                line_color='rgba(148, 163, 184, 0.5)'
            ))
            max_val = max(radar_merged[['app_percentage', 'category_avg_percentage']].max().max(), 1)
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, max_val + 5]),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=True,
                **get_chart_layout()
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Sentiment Distribution")
        if 'sentiment_score' in df.columns:
            fig_box = px.box(
                df,
                x="app_name",
                y="sentiment_score",
                color="app_name",
                color_discrete_map=APP_COLORS,
                labels={"sentiment_score": "Sentiment Score", "app_name": "App"}
            )
            fig_box.update_layout(**get_chart_layout())
            st.plotly_chart(fig_box, use_container_width=True)
        elif 'rating' in df.columns:
            fig_box = px.box(
                df,
                x="app_name",
                y="rating",
                color="app_name",
                color_discrete_map=APP_COLORS,
                labels={"rating": "Rating", "app_name": "App"}
            )
            fig_box.update_layout(**get_chart_layout())
            st.plotly_chart(fig_box, use_container_width=True)

    # Delta table
    if not benchmark_data.empty:
        st.markdown("#### Delta Table: App vs Category Average")
        display_bench = benchmark_data.copy()
        display_bench['theme_label'] = display_bench['theme'].map(THEME_LABELS).fillna(display_bench['theme'])
        display_bench['delta'] = display_bench['delta'].round(2)
        display_bench['app_percentage'] = display_bench['app_percentage'].round(2)
        display_bench['category_avg_percentage'] = display_bench['category_avg_percentage'].round(2)

        st.dataframe(
            display_bench[['app_name', 'theme_label', 'app_percentage', 'category_avg_percentage', 'delta', 'status']],
            use_container_width=True,
            column_config={
                "app_name": "App",
                "theme_label": "Theme",
                "app_percentage": st.column_config.NumberColumn("App %", format="%.1f%%"),
                "category_avg_percentage": st.column_config.NumberColumn("Category Avg %", format="%.1f%%"),
                "delta": st.column_config.NumberColumn("Delta", format="%+.1f pp"),
                "status": "Status"
            },
            hide_index=True
        )


# ═══════════════════════════════════════════════════════════════
#  PAGE 4 — ACTION PLAN
# ═══════════════════════════════════════════════════════════════
def render_action_plan(df):
    st.title("Action Plan & Priorities")

    if df.empty or 'theme' not in df.columns:
        st.warning("⚠️ Data not available.")
        return

    priority_df = compute_priority_scores(df)

    if priority_df.empty:
        st.info("Not enough data to compute priorities.")
        return

    st.markdown("### 🎯 Prioritized Theme Ranking")

    # Add rank column
    display = priority_df.copy()
    display.insert(0, 'rank', range(1, len(display) + 1))
    display['quick_win_display'] = display['quick_win'].apply(lambda x: '🟢 Quick Win' if x else '⚪')
    display['priority_score'] = display['priority_score'].round(1)
    display['frequency_severity_gap'] = display['frequency_severity_gap'].round(1)

    st.dataframe(
        display[['rank', 'theme_label', 'priority_score', 'impact', 'frequency_severity_gap', 'effort', 'quick_win_display', 'justification']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": "Rank",
            "theme_label": "Theme",
            "priority_score": st.column_config.NumberColumn("Priority Score", format="%.1f"),
            "impact": st.column_config.NumberColumn("Impact", format="%d"),
            "frequency_severity_gap": st.column_config.NumberColumn("Freq/Severity Gap", format="%.1f"),
            "effort": st.column_config.NumberColumn("Fix Effort", format="%d"),
            "quick_win_display": "Quick Win?",
            "justification": "Justification"
        }
    )

    # Quick wins callout
    quick_wins = display[display['quick_win'] == True]
    if not quick_wins.empty:
        st.markdown("### 🏆 Quick Wins (High Impact, Low Effort)")
        for _, row in quick_wins.iterrows():
            st.success(f"**{row['theme_label']}** — Priority Score: {row['priority_score']:.1f} | {row['justification']}")

    # Download
    csv = priority_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Action Plan (CSV)",
        data=csv,
        file_name='fintech_action_plan.csv',
        mime='text/csv',
    )

    # Methodology
    with st.expander("📐 Priority Scoring Methodology & Assumptions"):
        st.markdown("""
        **Formula:**
        ```
        Priority Score = (Impact × 4) + (Frequency/Severity Gap × 4) − (Assumed Fix Effort × 2)
        ```

        **Impact Assumptions** (predefined per theme):
        | Theme | Impact | Rationale |
        |-------|--------|-----------|
        | Failed Transaction | 10 | Blocks core money movement |
        | Fraud / Security | 9 | Trust & safety critical |
        | Refund Delay | 8 | Money stuck in limbo |
        | KYC Friction | 7 | Blocks onboarding |
        | App Crash / Bug | 7 | Blocks usage |
        | Customer Support | 6 | Degrades trust |
        | UI Confusion | 5 | Degrades experience |

        **Effort Assumptions** (predefined per theme):
        | Theme | Effort | Rationale |
        |-------|--------|-----------|
        | Failed Transaction | 9 | Backend infra, payment gateway work |
        | Fraud / Security | 8 | Security team, fraud detection systems |
        | Refund Delay | 6 | Process + backend changes |
        | App Crash / Bug | 6 | Engineering debugging |
        | KYC Friction | 5 | UX + compliance workflow |
        | Customer Support | 4 | Training + tooling |
        | UI Confusion | 3 | UX redesign, relatively low effort |

        **Quick Win Flag:** Theme has Impact ≥ 6 AND Effort ≤ 4
        """)


# ═══════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════
def main():
    inject_custom_css()

    df = load_data()

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("## 🧠 Review Intelligence")
        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigate",
            ["📊 Overview", "🏷️ Theme Breakdown", "📈 Severity & Trend",
             "⚖️ Comparative Benchmark", "🎯 Action Plan"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Data status
        st.markdown("### 📁 Data Status")
        total_reviews = len(df)
        classified = len(df[df['theme'].notna()]) if 'theme' in df.columns else 0

        st.markdown(f"**Reviews loaded:** {total_reviews:,}")
        st.markdown(f"**Classified:** {classified:,}")

        if total_reviews > 0 and classified > 0:
            cache_rate = (classified / total_reviews) * 100
            st.markdown(f"**Classification rate:** {cache_rate:.0f}%")

        if 'app_name' in df.columns:
            for app in sorted(df['app_name'].unique()):
                count = len(df[df['app_name'] == app])
                color = APP_COLORS.get(app, "#94A3B8")
                st.markdown(f"<span style='color:{color}'>●</span> {app}: {count:,}", unsafe_allow_html=True)

        st.markdown("---")

        # Pipeline button
        if st.button("🚀 Run Pipeline", use_container_width=True):
            with st.status("Running pipeline...", expanded=True) as status:
                st.write("📥 Scraping reviews from Google Play Store...")
                try:
                    results = scrape_all_apps(target_count=750)
                    raw_df = load_raw_reviews()
                    st.write(f"✅ Scraped {len(raw_df):,} reviews")
                except Exception as e:
                    st.error(f"Scraping failed: {e}")
                    raw_df = load_raw_reviews()  # Try loading existing
                    if raw_df.empty:
                        status.update(label="Pipeline failed", state="error")
                        st.stop()

                st.write("🤖 Classifying reviews with LLM...")
                try:
                    progress_bar = st.progress(0)
                    def update_progress(current, total):
                        progress_bar.progress(current / total)

                    classified_df = classify_dataframe(raw_df, batch_size=15, progress_callback=update_progress)

                    # Save classified data
                    CLASSIFIED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                    classified_df.to_csv(CLASSIFIED_DATA_PATH, index=False)
                    st.write(f"✅ Classified {len(classified_df):,} reviews")
                except Exception as e:
                    st.warning(f"Classification encountered an error: {e}")
                    st.write("💾 Saving partial results from cache...")
                    # Save whatever we have from cache instead of losing everything
                    try:
                        import hashlib
                        cache_path = Path("data/classified/cache.json")
                        if cache_path.exists():
                            cache = json.load(open(cache_path, "r", encoding="utf-8"))
                            partial_df = raw_df.copy()
                            themes, severities, sentiments, summaries = [], [], [], []
                            for _, row in partial_df.iterrows():
                                h = hashlib.sha256(str(row['review_text']).encode('utf-8')).hexdigest()
                                if h in cache:
                                    entry = cache[h]
                                    themes.append(entry.get('theme', 'other'))
                                    severities.append(entry.get('severity', 'medium'))
                                    sentiments.append(entry.get('sentiment_score', 0.0))
                                    summaries.append(entry.get('one_line_summary', ''))
                                else:
                                    themes.append(None)
                                    severities.append(None)
                                    sentiments.append(None)
                                    summaries.append(None)
                            partial_df['theme'] = themes
                            partial_df['severity'] = severities
                            partial_df['sentiment_score'] = sentiments
                            partial_df['one_line_summary'] = summaries
                            partial_df = partial_df.dropna(subset=['theme']).reset_index(drop=True)
                            if not partial_df.empty:
                                CLASSIFIED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                                partial_df.to_csv(CLASSIFIED_DATA_PATH, index=False)
                                st.write(f"✅ Saved {len(partial_df):,} partially classified reviews")
                            else:
                                st.error("No cached classifications found to save.")
                                status.update(label="Pipeline failed", state="error")
                                st.stop()
                    except Exception as save_err:
                        st.error(f"Failed to save partial results: {save_err}")
                        status.update(label="Pipeline failed", state="error")
                        st.stop()

                status.update(label="Pipeline complete!", state="complete")
                st.cache_data.clear()
                time.sleep(1)
                st.rerun()

        st.markdown("---")
        st.caption("Built with Streamlit & Plotly")

    # --- PAGE ROUTING ---
    if "Overview" in page:
        render_overview(df)
    elif "Theme Breakdown" in page:
        render_theme_breakdown(df)
    elif "Severity & Trend" in page:
        render_severity_trend(df)
    elif "Comparative Benchmark" in page:
        render_benchmark(df)
    elif "Action Plan" in page:
        render_action_plan(df)


if __name__ == "__main__":
    main()
