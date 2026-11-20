"""
App Review Intelligence Engine
Analyse and benchmark ANY apps from Google Play and the Apple App Store.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analytics as A
from src.insights import build_fact_sheet, build_markdown_report, generate_ai_brief
from src.llm_classify import apply_cache, classify_dataframe, load_cache, taxonomy_fingerprint
from src.scraper import load_workspace_raw, scrape_workspace, search_apps
from src.taxonomy import CATEGORIES, df_to_taxonomy, generate_taxonomy_with_llm, get_preset, taxonomy_to_df, RESERVED_KEYS
from src.validation import workspace_agreement, workspace_label_sample
from src.workspace import AppRef, Workspace, delete_workspace, list_workspaces, load_workspace

st.set_page_config(page_title="App Review Intelligence Engine", page_icon=None,
                   layout="wide", initial_sidebar_state="expanded")

# --- CONSTANTS ---
SEVERITY_COLORS = {"high": "#b91c1c", "medium": "#a16207", "low": "#15803d"}
PLATFORM_LABEL = {"play": "Google Play", "appstore": "App Store"}
COUNTRIES = {"in": "India", "us": "United States", "gb": "United Kingdom", "ca": "Canada", "au": "Australia",
             "sg": "Singapore", "ae": "UAE", "de": "Germany", "fr": "France", "br": "Brazil", "id": "Indonesia",
             "jp": "Japan"}
NEW_WS = "＋ New workspace"
MAX_APPS = 8

TEMPLATES = {
    "Food delivery — Swiggy vs Zomato": ("Food & Grocery Delivery",
        [("Swiggy", "in.swiggy.android"), ("Zomato", "com.application.zomato")]),
    "Quick commerce — Blinkit vs Zepto": ("Food & Grocery Delivery",
        [("Blinkit", "com.grofers.customerapp"), ("Zepto", "com.zeptoconsumerapp")]),
    "Ride-hailing — Uber vs Ola vs Rapido": ("Ride-hailing & Travel",
        [("Uber", "com.ubercab"), ("Ola", "com.olacabs.customer"), ("Rapido", "com.rapido.passenger")]),
    "Streaming — Netflix vs Prime Video vs JioHotstar": ("Streaming / Media",
        [("Netflix", "com.netflix.mediaclient"), ("Prime Video", "com.amazon.avod.thirdpartyclient"),
         ("JioHotstar", "in.startv.hotstar")]),
    "E-commerce — Amazon vs Flipkart vs Meesho": ("E-commerce / Shopping",
        [("Amazon", "in.amazon.mShop.android.shopping"), ("Flipkart", "com.flipkart.android"),
         ("Meesho", "com.meesho.supply")]),
}

# --- CUSTOM CSS ---
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    /* Light frosted-glass editorial design */
    .stApp {
        background:
            radial-gradient(1150px 720px at 100% -6%, #fef4e4 0%, rgba(254,244,228,0) 55%),
            radial-gradient(760px 560px at 104% 8%, #fde7c4 0%, rgba(253,231,196,0) 48%),
            radial-gradient(880px 720px at -10% 110%, #e6d9ff 0%, rgba(230,217,255,0) 55%),
            radial-gradient(820px 700px at 114% 116%, #e0dcfb 0%, rgba(224,220,251,0) 55%),
            linear-gradient(170deg, #eef1fc 0%, #ebedf8 100%) !important;
        background-attachment: fixed !important;
        color: #1c2230;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 14px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #1c2230 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.3px;
    }
    h1 { font-size: 26px !important; font-weight: 500 !important; letter-spacing: -0.6px; }
    p, span, label, div { font-family: 'IBM Plex Sans', sans-serif; }

    /* Main content area -- frosted glass sheet */
    .block-container {
        background: rgba(255, 255, 255, 0.82) !important;
        backdrop-filter: blur(22px) saturate(150%) !important;
        -webkit-backdrop-filter: blur(22px) saturate(150%) !important;
        border: 1px solid rgba(255, 255, 255, 0.85) !important;
        border-radius: 20px !important;
        padding: 30px 36px 40px 36px !important;
        margin-top: 18px !important;
        margin-bottom: 24px !important;
        max-width: 1400px !important;
        box-shadow: 0 24px 64px rgba(23, 31, 56, 0.12), 0 2px 6px rgba(23, 31, 56, 0.04) !important;
    }

    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 29px !important;
        font-weight: 400 !important;
        color: #1c2230 !important;
        letter-spacing: -0.7px;
        font-variant-numeric: tabular-nums;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 13px !important;
        color: #6b7384 !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 12px !important;
    }

    /* Metric containers */
    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e6e8f2;
        border-radius: 14px;
        padding: 21px;
        box-shadow: 0 6px 18px rgba(23,31,56,0.07), 0 2px 6px rgba(23,31,56,0.04);
    }

    /* Buttons */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border: 1px solid #2563eb;
        border-radius: 9px;
        padding: 7px 14px;
        font-weight: 500;
        font-size: 13px;
        min-height: 37px;
        transition: background-color 0.15s, color 0.15s;
        box-shadow: 0 6px 16px rgba(37,99,235,0.28);
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        border-color: #1d4ed8;
    }

    /* Secondary / outline buttons */
    .stDownloadButton>button {
        background: #fff;
        color: #1c2230;
        border: 1px solid #dcdfeb;
        border-radius: 9px;
        font-weight: 500;
        font-size: 13px;
        box-shadow: none;
    }
    .stDownloadButton>button:hover {
        background: #f6f7fc;
        border-color: #cfd4e6;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 7px;
        font-size: 12px;
        font-weight: 500;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 4px;
    }
    .badge-green { background-color: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }
    .badge-yellow { background-color: #fffbeb; color: #a16207; border: 1px solid #fde68a; }
    .badge-red { background-color: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(170deg, #f4f6fd 0%, #eef0f9 100%) !important;
        border-right: 1px solid #e6e8f2 !important;
        padding-top: 12px;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1c2230 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #586468;
    }
    /* CloudVault-inspired sidebar navigation tabs */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 4px;
        display: flex;
        flex-direction: column;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        background: transparent;
        border-radius: 8px;
        padding: 9px 12px;
        color: #586468 !important;
        font-size: 13px;
        font-weight: 400;
        transition: background-color 0.15s, color 0.15s;
        cursor: pointer;
        display: flex;
        align-items: center;
        margin: 0;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.65);
        color: #1c2230 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: #eef2ff !important;
        color: #1d4ed8 !important;
        font-weight: 500;
    }
    /* Hide radio circle button for clean nav link look */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* Data tables */
    .stDataFrame {
        border: 1px solid #e6e8f2;
        border-radius: 14px;
        overflow: hidden;
        background: #ffffff;
    }

    /* Plotly charts wrapper */
    div[data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid #e6e8f2;
        border-radius: 14px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(23, 31, 56, 0.04);
        margin-bottom: 12px;
    }

    /* Expander */
    details[data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e6e8f2 !important;
        border-radius: 14px !important;
        box-shadow: 0 1px 2px rgba(23,31,56,0.05), 0 1px 3px rgba(23,31,56,0.06);
    }
    details[data-testid="stExpander"] summary {
        font-weight: 500;
        color: #1c2230;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #586468 !important;
        border-bottom: 2px solid transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1c2230 !important;
        border-bottom-color: #2563eb !important;
    }

    /* Selectbox / multiselect */
    div[data-baseweb="select"] {
        border-radius: 9px;
    }
    div[data-baseweb="select"] > div {
        background: white;
        border: 1px solid #d4d8e8;
        border-radius: 9px;
        color: #1c2230;
    }

    /* Info / warning / success / error boxes */
    div[data-testid="stAlert"] {
        border-radius: 8px;
        font-size: 13px;
    }

    /* Section headers */
    .section-header {
        font-size: 12px;
        font-weight: 500;
        color: #6b7384;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    /* Dividers */
    hr {
        border-color: #e6e8f2 !important;
    }

    /* Caption text */
    .stCaption, small {
        color: #6b7384 !important;
        font-size: 12px !important;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #2563eb) !important;
    }

    /* Status widget */
    div[data-testid="stStatusWidget"] {
        background: white;
        border: 1px solid #e6e8f2;
        border-radius: 14px;
    }

    /* Pill tags */
    .pill {
        font-size: 12px;
        background: #f0f3f3;
        color: #526166;
        border-radius: 4px;
        padding: 3px 7px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def get_chart_layout(title=""):
    return dict(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        title=dict(text=title, font=dict(size=14, color="#1c2230", family="IBM Plex Sans")),
        margin=dict(t=50 if title else 20, l=20, r=20, b=20),
        font=dict(color="#6b7384", size=12, family="IBM Plex Sans"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#e6e8f2", zerolinecolor="#e6e8f2"),
        yaxis=dict(gridcolor="#e6e8f2", zerolinecolor="#e6e8f2"),
    )


def show(fig, title="", height=None):
    fig.update_layout(**get_chart_layout(title))
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)


def card(label, value, sub="", color=None):
    dot = f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{color};"></span>' if color else ""
    st.markdown(f"""
    <div style="background:#fff;padding:16px 18px;border-radius:14px;border:1px solid #e6e8f2;
                box-shadow:0 1px 3px rgba(23,31,56,0.05);margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-size:13px;color:#6b7384;font-weight:500;">{label}</span>{dot}
        </div>
        <div style="font-size:24px;font-weight:500;color:#1c2230;letter-spacing:-0.5px;">{value}</div>
        <div style="font-size:12px;color:#8a92a6;margin-top:3px;">{sub}</div>
    </div>""", unsafe_allow_html=True)


def callout(text, kicker="Executive Observation", tone="blue"):
    bg, border, k = {"blue": ("linear-gradient(120deg,#f2f6ff 0%,#f6f1ff 100%)", "#e4e6fa", "#3b82f6"),
                     "red": ("#fef2f2", "#fecaca", "#b91c1c"),
                     "green": ("#f0fdf4", "#bbf7d0", "#15803d")}[tone]
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {border};border-radius:14px;padding:14px 18px;margin:8px 0 14px 0;">
        <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:{k};margin-bottom:4px;">{kicker}</div>
        <div style="font-size:14px;color:#1c2230;line-height:1.5;">{text}</div>
    </div>""", unsafe_allow_html=True)


def needs_data(df) -> bool:
    if df.empty or "theme" not in df:
        st.info("No classified reviews yet. Go to **Setup → Run Pipeline** to scrape and classify reviews.")
        return True
    return False


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _load(slug: str, _sig: str):
    ws = load_workspace(slug)
    raw = load_workspace_raw(ws)
    if raw.empty:
        return raw, raw
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["review_text"] = raw["review_text"].fillna("").astype(str)
    raw = raw[raw["review_text"].str.strip() != ""].reset_index(drop=True)
    raw["rating"] = pd.to_numeric(raw["rating"], errors="coerce")
    raw["thumbs_up_count"] = pd.to_numeric(raw["thumbs_up_count"], errors="coerce").fillna(0)
    df = apply_cache(raw, load_cache(ws.cache_path, ws.taxonomy))
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    return raw, df[df["theme"].notna()].reset_index(drop=True)


def get_data(ws: Workspace):
    """(raw, classified). Cache-busted by file mtimes + workspace config."""
    files = sorted(ws.raw_dir.glob("*.csv")) if ws.raw_dir.exists() else []
    sig = json.dumps([[f.name, _mtime(f)] for f in files] + [_mtime(ws.cache_path), _mtime(ws.root / "workspace.json")])
    return _load(ws.slug, sig)


def tax_options(ws):
    return [t["key"] for t in ws.taxonomy]


# ═══════════════════════════════════════════════════════════════
#  PIPELINE ACTIONS
# ═══════════════════════════════════════════════════════════════
def run_scrape(ws, n, sort, merge):
    with st.status("Scraping store data…", expanded=True) as s:
        msg, bar = st.empty(), st.progress(0.0)
        scrape_workspace(ws, n, sort=sort, merge=merge,
                         status_callback=lambda m: msg.write(m),
                         progress_callback=lambda c, t: bar.progress(min(c / max(t, 1), 1.0)))
        s.update(label="Scrape complete", state="complete")


def run_classify(ws):
    raw, _ = get_data(ws)
    if raw.empty:
        st.warning("Nothing to classify — scrape reviews first.")
        return
    with st.status("Classifying reviews with the LLM…", expanded=True) as s:
        bar = st.progress(0.0, text="Checking cache…")
        ctx = f"{ws.category} apps ({', '.join(a.name for a in ws.apps)}) in {COUNTRIES.get(ws.country, ws.country)}"
        out = classify_dataframe(raw, ws.taxonomy, ws.cache_path, context=ctx,
                                 progress_callback=lambda c, t: bar.progress(c / max(t, 1), text=f"{c:,}/{t:,} new reviews"))
        done = out.dropna(subset=["theme"])
        done.to_csv(ws.classified_path, index=False)
        s.update(label=f"Classified {len(done):,}/{len(out):,} reviews", state="complete")


# ═══════════════════════════════════════════════════════════════
#  PAGE — SETUP
# ═══════════════════════════════════════════════════════════════
def render_new_workspace():
    st.title("Create a Workspace")
    st.markdown("A workspace is a set of competing apps analysed together — any category, any store.")

    st.markdown("#### Start from a template")
    cols = st.columns(len(TEMPLATES))
    for col, (name, (cat, apps)) in zip(cols, TEMPLATES.items()):
        with col:
            if st.button(name, use_container_width=True, key=f"tpl_{name}"):
                ws = Workspace(name=name.split(" — ")[1], category=cat, country="in")
                ws.apps = [AppRef(name=n, platform="play", app_id=i) for n, i in apps]
                ws.taxonomy = get_preset(cat)
                ws.save()
                st.session_state.ws = ws.slug
                st.session_state.nav_to = "Setup"
                st.rerun()

    st.markdown("#### …or build your own")
    with st.form("new_ws"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Workspace name", placeholder="e.g. Neobanks UK")
        category = c2.selectbox("Category (sets the starting taxonomy)", CATEGORIES)
        c3, c4 = st.columns(2)
        country = c3.selectbox("Store country", list(COUNTRIES), format_func=lambda c: f"{COUNTRIES[c]} ({c})")
        lang = c4.text_input("Review language code", "en")
        if st.form_submit_button("Create workspace", type="primary"):
            if not name.strip():
                st.error("Give the workspace a name.")
            elif load_workspace(Workspace(name=name).slug):
                st.error("A workspace with that name already exists.")
            else:
                ws = Workspace(name=name.strip(), category=category, country=country, lang=lang.strip() or "en")
                ws.taxonomy = get_preset(category)
                ws.save()
                st.session_state.ws = ws.slug
                st.session_state.nav_to = "Setup"
                st.rerun()


def render_setup(ws: Workspace):
    if ws is None:
        render_new_workspace()
        return
    st.title("Workspace Setup")
    st.caption(f"**{ws.name}** · {ws.category} · {COUNTRIES.get(ws.country, ws.country)} store · "
               f"{len(ws.apps)} apps · {len(ws.taxonomy)} themes")
    t_apps, t_tax, t_run, t_settings = st.tabs(["1 · Apps", "2 · Taxonomy", "3 · Run Pipeline", "Settings"])
    with t_apps:
        render_setup_apps(ws)
    with t_tax:
        render_setup_taxonomy(ws)
    with t_run:
        render_setup_run(ws)
    with t_settings:
        render_setup_settings(ws)


def render_setup_apps(ws):
    st.markdown("#### Apps in this workspace")
    if not ws.apps:
        st.info("No apps yet — search below and add 2 or more to compare.")
    colors = ws.app_colors()
    for i, a in enumerate(list(ws.apps)):
        c0, c1, c2, c3 = st.columns([0.6, 3, 4, 1])
        with c0:
            if a.icon:
                st.image(a.icon, width=36)
            else:
                st.markdown(f"<div style='width:12px;height:12px;border-radius:50%;background:{colors[a.name]};margin-top:12px'></div>",
                            unsafe_allow_html=True)
        new_name = c1.text_input("Display name", a.name, key=f"nm_{a.key}", label_visibility="collapsed")
        c2.markdown(f"<div style='padding-top:8px;font-size:13px;color:#6b7384'>{PLATFORM_LABEL[a.platform]} · <code>{a.app_id}</code></div>",
                    unsafe_allow_html=True)
        if c3.button("Remove", key=f"rm_{a.key}"):
            ws.apps.pop(i)
            ws.save()
            st.rerun()
        if new_name.strip() and new_name.strip() != a.name:
            if any(o.name == new_name.strip() for o in ws.apps):
                st.error("Display names must be unique.")
            else:
                a.name = new_name.strip()
                ws.save()
                st.rerun()

    st.markdown("---")
    st.markdown("#### Add apps")
    with st.form("search"):
        c1, c2, c3 = st.columns([1.3, 4, 1])
        platform = c1.radio("Store", ["play", "appstore"], format_func=PLATFORM_LABEL.get, horizontal=False)
        q = c2.text_input("Search by name, or paste a store URL / package id",
                          placeholder="e.g. Revolut   ·   com.whatsapp   ·   https://apps.apple.com/.../id310633997")
        c3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        go_search = c3.form_submit_button("Search", use_container_width=True)
    if go_search and q.strip():
        with st.spinner("Searching the store…"):
            st.session_state.search_results = search_apps(q, platform, ws.country, ws.lang)
        if not st.session_state.search_results:
            st.warning("No results. Try a different name, or paste the store URL.")

    existing = {a.key for a in ws.apps}
    for r in st.session_state.get("search_results", []):
        ref = AppRef(name="", platform=r["platform"], app_id=r["app_id"], icon=r.get("icon") or "", developer=r.get("developer") or "")
        c0, c1, c2, c3 = st.columns([0.6, 5, 2.5, 1.2])
        if r.get("icon"):
            c0.image(r["icon"], width=40)
        c1.markdown(f"**{r['title']}**  \n<span style='font-size:12px;color:#6b7384'>{r.get('developer') or ''} · "
                    f"{r.get('genre') or ''} · {PLATFORM_LABEL[r['platform']]}</span>", unsafe_allow_html=True)
        score = f"★ {r['score']:.2f}" if r.get("score") else "★ —"
        vol = r.get("installs") or (f"{r['ratings']:,} ratings" if r.get("ratings") else "")
        c2.markdown(f"<div style='padding-top:6px;font-size:13px'>{score} · {vol}</div>", unsafe_allow_html=True)
        if ref.key in existing:
            c3.markdown("<div style='padding-top:6px;font-size:13px;color:#15803d'>Added ✓</div>", unsafe_allow_html=True)
        elif c3.button("Add", key=f"add_{ref.key}", disabled=len(ws.apps) >= MAX_APPS):
            short = r["title"].split(":")[0].split(" - ")[0].split(" – ")[0].strip()[:24]
            names = {a.name for a in ws.apps}
            if short in names:
                short = f"{short} ({'iOS' if ref.platform == 'appstore' else 'Android'})"
            ref.name = short
            ws.apps.append(ref)
            ws.save()
            st.rerun()
    if len(ws.apps) >= MAX_APPS:
        st.caption(f"Maximum of {MAX_APPS} apps per workspace keeps charts readable.")


def render_setup_taxonomy(ws):
    raw, _ = get_data(ws)
    st.markdown("Themes are what each review gets classified into. **Impact** and **Effort** (0-10) feed the "
                "priority score — tune them with your engineering lead.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Use a preset")
        preset = st.selectbox("Preset", CATEGORIES, index=CATEGORIES.index(ws.category) if ws.category in CATEGORIES else 0,
                              label_visibility="collapsed")
        if st.button("Load preset into editor"):
            st.session_state.proposed_tax = get_preset(preset)
    with c2:
        st.markdown("##### Generate from real reviews with AI")
        n_themes = st.slider("Number of issue themes", 5, 12, 8)
        if st.button("✨ Propose taxonomy", disabled=raw.empty,
                     help="Needs scraped reviews. The LLM reads a sample (low-star weighted) and proposes domain themes."):
            with st.spinner("Reading reviews and drafting themes…"):
                st.session_state.proposed_tax = generate_taxonomy_with_llm(
                    raw, ws.category, [a.name for a in ws.apps], n_themes=n_themes)
        if raw.empty:
            st.caption("Scrape reviews first (step 3) to enable this.")

    proposed = st.session_state.get("proposed_tax")
    source = proposed or ws.taxonomy
    if proposed:
        st.info("Showing a **proposed** taxonomy — review/edit, then save to apply.")
    editable = taxonomy_to_df([t for t in source if t["key"] not in RESERVED_KEYS])
    edited = st.data_editor(
        editable, num_rows="dynamic", use_container_width=True, hide_index=True, key=f"tax_{ws.slug}_{bool(proposed)}",
        column_config={
            "key": st.column_config.TextColumn("Key", help="snake_case id", required=True),
            "label": "Label",
            "description": st.column_config.TextColumn("Description (shown to the LLM)", width="large"),
            "impact": st.column_config.NumberColumn("Impact", min_value=0, max_value=10, step=1),
            "effort": st.column_config.NumberColumn("Effort", min_value=0, max_value=10, step=1),
        })
    st.caption("`praise_no_issue` and `other` are always included automatically.")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Save taxonomy", type="primary"):
        new_tax = df_to_taxonomy(edited)
        changed = taxonomy_fingerprint(new_tax) != taxonomy_fingerprint(ws.taxonomy)
        ws.taxonomy = new_tax
        ws.save()
        st.session_state.pop("proposed_tax", None)
        if changed:
            st.warning("Theme keys/descriptions changed — previous classifications no longer apply. "
                       "Run **Classify** in step 3.")
        else:
            st.success("Saved. Impact/effort changes apply instantly; no re-classification needed.")
    if proposed and c2.button("Discard proposal"):
        st.session_state.pop("proposed_tax", None)
        st.rerun()


def render_setup_run(ws):
    if not ws.apps:
        st.info("Add apps first.")
        return
    raw, df = get_data(ws)
    meta = ws.load_meta()

    st.markdown("#### Data status")
    rows = []
    for a in ws.apps:
        r = raw[raw["app_name"] == a.name] if not raw.empty else raw
        c = df[df["app_name"] == a.name] if not df.empty else df
        m = meta.get(a.key, {})
        rows.append({"App": a.name, "Store": PLATFORM_LABEL[a.platform], "Reviews scraped": len(r),
                     "Classified": len(c), "Oldest review": r["date"].min() if len(r) else None,
                     "Newest review": r["date"].max() if len(r) else None, "Last scraped": m.get("fetched_at")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 column_config={"Oldest review": st.column_config.DatetimeColumn(format="D MMM YYYY"),
                                "Newest review": st.column_config.DatetimeColumn(format="D MMM YYYY")})
    pending = len(raw) - len(df)

    st.markdown("#### Run")
    c1, c2, c3 = st.columns([2, 2, 1.4])
    n = c1.slider("Reviews per app", 100, 3000, 500, 100,
                  help="App Store public feed caps at 500 reviews per app.")
    sort = c2.radio("Sampling", ["newest", "relevant"], horizontal=True,
                    format_func={"newest": "Newest (monitoring)", "relevant": "Most relevant (longer history)"}.get)
    merge = c3.checkbox("Merge with existing", True, help="Keep earlier reviews and add new ones → builds history for trends.")
    est = int(np.ceil(pending / 15))
    st.caption(f"{pending:,} reviews awaiting classification ≈ {est} LLM calls. Cached reviews are never re-sent.")

    b1, b2, b3 = st.columns(3)
    if b1.button("① Scrape reviews", use_container_width=True):
        run_scrape(ws, n, sort, merge)
        st.cache_data.clear()
        st.rerun()
    if b2.button(f"② Classify ({pending:,} pending)", use_container_width=True, disabled=raw.empty or pending == 0):
        run_classify(ws)
        st.cache_data.clear()
        st.rerun()
    if b3.button("Run full pipeline", type="primary", use_container_width=True):
        run_scrape(ws, n, sort, merge)
        st.cache_data.clear()
        run_classify(ws)
        st.cache_data.clear()
        st.rerun()


def render_setup_settings(ws):
    with st.form("ws_settings"):
        category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(ws.category) if ws.category in CATEGORIES else 0)
        country = st.selectbox("Store country", list(COUNTRIES), index=list(COUNTRIES).index(ws.country) if ws.country in COUNTRIES else 0,
                               format_func=lambda c: f"{COUNTRIES[c]} ({c})")
        lang = st.text_input("Review language", ws.lang)
        if st.form_submit_button("Save settings"):
            ws.category, ws.country, ws.lang = category, country, lang
            ws.save()
            st.success("Saved. Changing country/language affects the next scrape.")
    st.markdown("---")
    st.markdown("##### Danger zone")
    confirm = st.checkbox(f"I understand this permanently deletes **{ws.name}** and its data")
    if st.button("Delete workspace", disabled=not confirm):
        delete_workspace(ws.slug)
        st.session_state.ws = None
        st.cache_data.clear()
        st.rerun()


# ═══════════════════════════════════════════════════════════════
#  PAGE — OVERVIEW
# ═══════════════════════════════════════════════════════════════
def render_overview(ws, df, raw):
    st.title(ws.name)
    st.markdown(f"##### {ws.category} · {', '.join(a.name for a in ws.apps)}")
    if needs_data(df):
        st.markdown("""
        ### How it works
        1. **Pick apps** — any app on Google Play or the App Store
        2. **Scrape** — newest (or most relevant) reviews + store metadata
        3. **Classify** — LLM tags every review with theme, severity, sentiment and feature requests
        4. **Analyse** — significance-tested benchmarks, rating drivers, release impact, spike alerts
        5. **Act** — priority matrix, AI executive brief, exportable report
        """)
        return

    o = A.get_overview_stats(df)
    c = st.columns(5)
    c[0].metric("Reviews analysed", f"{o['total_reviews']:,}",
                help=f"{len(raw) - len(df):,} more scraped but not yet classified" if len(raw) > len(df) else None)
    c[1].metric("Apps", o["apps"])
    c[2].metric("Avg rating", f"{o['avg_rating']:.2f} ★")
    c[3].metric("1-2 ★ share", f"{o['pct_negative']:.1f}%")
    c[4].metric("High-severity", f"{o['high_severity_pct']:.1f}%")
    st.caption(f"Review window: {o['date_range'][0]} → {o['date_range'][1]}")

    callout(A.get_headline_finding(df, ws.taxonomy))

    st.markdown("#### App Scorecard")
    sc = A.app_scorecard(df, ws.taxonomy)
    colors = ws.app_colors()
    cols = st.columns(min(len(sc), 4))
    for i, r in sc.iterrows():
        with cols[i % len(cols)]:
            card(r.app_name, f"{r.avg_rating:.2f} ★",
                 f"NPS proxy {r.nps_proxy:+.0f} · {r.pct_1_2_star:.1f}% 1-2★<br>Top issue: {r.top_issue} ({r.top_issue_pct:.1f}%)",
                 colors.get(r.app_name))

    st.markdown("#### Alerts")
    spikes = A.detect_spikes(df, ws.taxonomy)
    versions = A.version_impact(df, ws.taxonomy)
    regress = versions[versions["flag"].str.contains("regression")] if not versions.empty else versions
    if spikes.empty and regress.empty:
        callout("No statistically unusual spikes or release regressions in the current window.", "All clear", "green")
    for r in spikes.head(5).itertuples():
        callout(f"<b>{r.app_name}</b>: {r.theme_label} jumped to {r.latest_share:.1f}% of reviews in the latest {r.unit.lower()} "
                f"vs a {r.baseline_share:.1f}% baseline (z = {r.z:.1f}, {r.latest_count} reviews).", "Spike alert", "red")
    for r in regress.itertuples():
        callout(f"<b>{r.app_name} v{r.version}</b>: average rating {r.avg_rating:.2f}★ "
                f"({r.delta_vs_prev:+.2f} vs previous version, p = {r.p_value:.3f}). Top issue: {r.top_issue}.",
                "Release regression", "red")

    with st.expander("Full scorecard table"):
        st.dataframe(sc, use_container_width=True, hide_index=True, column_config={
            "app_name": "App", "reviews": "Reviews",
            "avg_rating": st.column_config.NumberColumn("Avg ★", format="%.2f"),
            "avg_sentiment": st.column_config.NumberColumn("Sentiment", format="%.2f"),
            "pct_1_2_star": st.column_config.NumberColumn("% 1-2★", format="%.1f%%"),
            "nps_proxy": st.column_config.NumberColumn("NPS proxy", format="%+.0f",
                                                       help="% 5★ minus % 1-3★ — a rating-based proxy, not a true NPS survey"),
            "high_sev_pct": st.column_config.NumberColumn("% High sev", format="%.1f%%"),
            "issue_share": st.column_config.NumberColumn("% Issues", format="%.1f%%"),
            "top_issue": "Top issue", "top_issue_pct": st.column_config.NumberColumn("Top issue %", format="%.1f%%"),
        })


# ═══════════════════════════════════════════════════════════════
#  PAGE — MARKET SNAPSHOT
# ═══════════════════════════════════════════════════════════════
def render_market(ws, df):
    st.title("Market Snapshot")
    st.caption("Store-level metadata vs what recent reviewers are saying.")
    meta = ws.load_meta()
    rows = []
    for a in ws.apps:
        m = meta.get(a.key)
        if not m:
            continue
        recent = df[df["app_name"] == a.name]["rating"].mean() if not df.empty else np.nan
        rows.append({"icon": m.get("icon"), "app": a.name, "store": PLATFORM_LABEL[a.platform],
                     "store_rating": m.get("score"), "recent_rating": recent,
                     "momentum": recent - m["score"] if m.get("score") and pd.notna(recent) else np.nan,
                     "ratings": m.get("ratings"), "installs": m.get("installs") or "—", "version": m.get("version"),
                     "updated": pd.to_datetime(m.get("updated"), errors="coerce", utc=True),
                     "genre": m.get("genre"), "ads": m.get("contains_ads"), "iap": m.get("offers_iap"),
                     "histogram": m.get("histogram")})
    if not rows:
        st.info("No store metadata yet — run the scraper.")
        return
    t = pd.DataFrame(rows)
    st.dataframe(t.drop(columns=["histogram"]), use_container_width=True, hide_index=True, column_config={
        "icon": st.column_config.ImageColumn("", width="small"), "app": "App", "store": "Store",
        "store_rating": st.column_config.NumberColumn("Lifetime ★", format="%.2f"),
        "recent_rating": st.column_config.NumberColumn("Recent ★ (sample)", format="%.2f"),
        "momentum": st.column_config.NumberColumn("Momentum", format="%+.2f",
                                                  help="Recent sample rating minus lifetime store rating. Negative = recent experience is worse."),
        "ratings": st.column_config.NumberColumn("Ratings", format="%d"), "installs": "Installs",
        "version": "Version", "updated": st.column_config.DatetimeColumn("Last update", format="D MMM YYYY"),
        "genre": "Genre", "ads": "Ads?", "iap": "IAP?"})

    c1, c2 = st.columns(2)
    with c1:
        m = t.melt(id_vars="app", value_vars=["store_rating", "recent_rating"], var_name="kind", value_name="rating")
        m["kind"] = m["kind"].map({"store_rating": "Lifetime (store)", "recent_rating": "Recent reviews"})
        fig = px.bar(m, x="app", y="rating", color="kind", barmode="group",
                     color_discrete_sequence=["#94a3b8", "#2563eb"], labels={"app": "", "rating": "★", "kind": ""})
        fig.update_yaxes(range=[1, 5])
        show(fig, "Lifetime vs recent rating")
    with c2:
        h = t.dropna(subset=["histogram"])
        if not h.empty:
            recs = []
            for r in h.itertuples():
                tot = sum(r.histogram) or 1
                recs += [{"app": r.app, "stars": f"{i + 1}★", "pct": v / tot * 100} for i, v in enumerate(r.histogram)]
            fig = px.bar(pd.DataFrame(recs), y="app", x="pct", color="stars", orientation="h",
                         color_discrete_sequence=["#b91c1c", "#ea580c", "#eab308", "#84cc16", "#15803d"],
                         labels={"pct": "% of all ratings", "app": ""})
            show(fig, "Lifetime star distribution (Google Play)")
        else:
            st.info("Star histograms are only published by Google Play.")


# ═══════════════════════════════════════════════════════════════
#  PAGE — THEMES
# ═══════════════════════════════════════════════════════════════
def render_themes(ws, df):
    st.title("Theme Breakdown")
    if needs_data(df):
        return
    labels = ws.theme_labels()
    colors = ws.app_colors()
    c1, c2 = st.columns([3, 1])
    apps = c1.multiselect("Apps", list(colors), default=[a for a in colors if a in df["app_name"].unique()])
    issues_only = c2.toggle("Issues only", True, help="Hide praise and 'other'")
    d = df[df["app_name"].isin(apps)]
    if d.empty:
        return
    share = (d.groupby(["app_name", "theme"]).size() / d.groupby("app_name").size()).mul(100).rename("pct").reset_index()
    if issues_only:
        share = share[~share["theme"].isin(A.NON_ISSUE_THEMES)]
    share["theme_label"] = share["theme"].map(labels).fillna(share["theme"])
    order = share.groupby("theme_label")["pct"].mean().sort_values().index.tolist()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(share, y="theme_label", x="pct", color="app_name", barmode="group", orientation="h",
                     color_discrete_map=colors, category_orders={"theme_label": order[::-1]},
                     labels={"pct": "% of app's reviews", "theme_label": "", "app_name": "App"})
        show(fig, "Theme share per app (normalised)", height=460)
    with c2:
        tree = d if not issues_only else d[~d["theme"].isin(A.NON_ISSUE_THEMES)]
        tree = tree.groupby(["app_name", "theme"]).size().reset_index(name="count")
        tree["theme_label"] = tree["theme"].map(labels).fillna(tree["theme"])
        fig = px.treemap(tree, path=["app_name", "theme_label"], values="count", color="count", color_continuous_scale="Blues")
        show(fig, "Volume treemap", height=460)

    st.markdown("#### Voice of the Customer")
    c1, c2 = st.columns(2)
    issue_keys = [k for k in tax_options(ws) if k in d["theme"].unique()]
    theme = c1.selectbox("Theme", issue_keys, format_func=lambda k: labels.get(k, k))
    app = c2.selectbox("App", ["All"] + apps)
    quotes = A.voice_of_customer(d, theme, None if app == "All" else app)
    if quotes.empty:
        st.caption("No substantive quotes for this selection.")
    for q in quotes.itertuples():
        st.markdown(f"""<div style="background:#fff;border:1px solid #e6e8f2;border-left:4px solid {colors.get(q.app_name, '#94a3b8')};
            border-radius:9px;padding:10px 14px;margin-bottom:8px;font-size:13px;color:#334155">
            “{str(q.review_text)[:500]}”<div style="font-size:12px;color:#8a92a6;margin-top:4px">
            {q.app_name} · {int(q.rating)}★ · {q.severity} severity · 👍 {int(q.thumbs_up_count)}</div></div>""",
                    unsafe_allow_html=True)

    st.markdown("#### Review Explorer")
    c1, c2, c3, c4 = st.columns([2, 1.3, 1.5, 2])
    f_theme = c1.multiselect("Themes", tax_options(ws), format_func=lambda k: labels.get(k, k))
    f_sev = c2.multiselect("Severity", ["high", "medium", "low"])
    f_rating = c3.slider("Rating", 1, 5, (1, 5))
    f_text = c4.text_input("Search text", placeholder="e.g. refund, otp, crash")
    v = d[d["rating"].between(*f_rating)]
    if f_theme:
        v = v[v["theme"].isin(f_theme)]
    if f_sev:
        v = v[v["severity"].isin(f_sev)]
    if f_text:
        v = v[v["review_text"].str.contains(f_text, case=False, na=False, regex=False)]
    st.caption(f"{len(v):,} reviews match")
    cols = ["app_name", "date", "rating", "theme", "severity", "sentiment_score", "one_line_summary", "review_text", "thumbs_up_count"]
    st.dataframe(v[cols].sort_values("date", ascending=False).head(1000), use_container_width=True, height=380, hide_index=True,
                 column_config={"app_name": "App", "date": st.column_config.DatetimeColumn("Date", format="D MMM, HH:mm"),
                                "rating": st.column_config.NumberColumn("★", format="%d"), "theme": "Theme",
                                "severity": "Severity", "sentiment_score": st.column_config.NumberColumn("Sent.", format="%.2f"),
                                "one_line_summary": "Summary", "review_text": st.column_config.TextColumn("Review", width="large"),
                                "thumbs_up_count": st.column_config.NumberColumn("👍", format="%d")})
    st.download_button("Download filtered reviews (CSV)", v[cols].to_csv(index=False).encode("utf-8"),
                       f"{ws.slug}_reviews.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════
#  PAGE — TRENDS & SEVERITY
# ═══════════════════════════════════════════════════════════════
def render_trends(ws, df):
    st.title("Trends & Severity")
    if needs_data(df):
        return
    colors = ws.app_colors()
    labels = ws.theme_labels()
    freq, unit = A.pick_time_freq(df["date"])
    st.caption(f"Data spans {df['date'].min():%d %b %Y} → {df['date'].max():%d %b %Y}; bucketed by **{unit.lower()}**. "
               "Enable *Merge with existing* and re-run the pipeline periodically to build longer history.")

    t1, t2 = st.tabs(["Rating & sentiment", "Theme share over time"])
    with t1:
        rt = A.rating_trend(df, freq)
        c1, c2 = st.columns(2)
        with c1:
            show(px.line(rt, x="bucket", y="avg_rating", color="app_name", markers=True, color_discrete_map=colors,
                         labels={"bucket": unit, "avg_rating": "Avg ★", "app_name": "App"}), "Average rating")
        with c2:
            show(px.line(rt, x="bucket", y="avg_sentiment", color="app_name", markers=True, color_discrete_map=colors,
                         labels={"bucket": unit, "avg_sentiment": "Avg sentiment", "app_name": "App"}), "Average sentiment")
    with t2:
        keys = [k for k in tax_options(ws) if k in df["theme"].unique() and k not in A.NON_ISSUE_THEMES]
        theme = st.selectbox("Theme", keys, format_func=lambda k: labels.get(k, k))
        tr = A.trend_over_time(df, theme, freq)
        show(px.line(tr, x="bucket", y="share", color="app_name", markers=True, color_discrete_map=colors,
                     hover_data=["count", "total"],
                     labels={"bucket": unit, "share": "% of reviews", "app_name": "App"}),
             f"{labels.get(theme, theme)} — share of reviews")

    c1, c2 = st.columns(2)
    with c1:
        sws = A.severity_weighted_score(df, ws.taxonomy)
        if not sws.empty:
            fig = px.bar(sws, x="normalized_score", y="app_name", color="theme_label", orientation="h",
                         labels={"normalized_score": "Severity points per 100 reviews", "app_name": "", "theme_label": "Theme"})
            show(fig, "Friction index (severity-weighted)")
    with c2:
        sev = df.groupby(["app_name", "severity"]).size().reset_index(name="n")
        sev["pct"] = sev["n"] / sev.groupby("app_name")["n"].transform("sum") * 100
        show(px.bar(sev, x="app_name", y="pct", color="severity", color_discrete_map=SEVERITY_COLORS,
                    category_orders={"severity": ["high", "medium", "low"]},
                    labels={"pct": "% of reviews", "app_name": ""}), "Severity mix")

    heat = (df[~df["theme"].isin(A.NON_ISSUE_THEMES)].groupby(["theme", "app_name"])["severity"]
            .apply(lambda s: (s == "high").mean() * 100).unstack())
    if not heat.empty:
        heat.index = heat.index.map(lambda k: labels.get(k, k))
        show(px.imshow(heat.round(0), text_auto=True, color_continuous_scale="Reds", aspect="auto",
                       labels=dict(x="App", y="Theme", color="% High")), "% of each theme's reviews that are high severity")


# ═══════════════════════════════════════════════════════════════
#  PAGE — BENCHMARK
# ═══════════════════════════════════════════════════════════════
def render_benchmark(ws, df):
    st.title("Competitive Benchmark")
    if needs_data(df):
        return
    b = A.comparative_benchmark(df, ws.taxonomy)
    if b.empty:
        st.info("Add at least two apps to benchmark.")
        return
    colors = ws.app_colors()
    st.caption("Each app is compared with its peers pooled. Two-proportion z-test, Benjamini-Hochberg corrected "
               "(q < 0.05). ★ = statistically significant.")

    iss = b[~b["theme"].isin(A.NON_ISSUE_THEMES)]
    pv = iss.pivot(index="theme_label", columns="app_name", values="delta")
    sig = iss.pivot(index="theme_label", columns="app_name", values="significant")
    txt = pv.round(1).astype(str).where(~sig, pv.round(1).astype(str) + " ★")
    lim = max(abs(pv.min().min()), abs(pv.max().max()), 1)
    fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, text=txt.values, texttemplate="%{text}",
                               colorscale="RdBu_r", zmin=-lim, zmax=lim, colorbar=dict(title="pp vs peers")))
    show(fig, "Issue share vs peers (percentage points; red = worse)", height=420)

    c1, c2 = st.columns(2)
    with c1:
        labels = ws.theme_labels()
        keys = [k for k in tax_options(ws) if k in b["theme"].unique()]
        theme = st.selectbox("Theme — share with 95% confidence interval", keys, format_func=lambda k: labels.get(k, k))
        s = b[b["theme"] == theme]
        fig = go.Figure()
        for r in s.itertuples():
            fig.add_trace(go.Scatter(x=[r.app_percentage], y=[r.app_name], mode="markers", marker=dict(size=12, color=colors.get(r.app_name)),
                                     error_x=dict(type="data", symmetric=False, array=[r.ci_high - r.app_percentage],
                                                  arrayminus=[r.app_percentage - r.ci_low]), name=r.app_name, showlegend=False))
        fig.add_vline(x=s["category_avg_percentage"].iloc[0], line_dash="dot", line_color="#94a3b8",
                      annotation_text="category avg")
        fig.update_xaxes(title="% of reviews")
        show(fig, height=320)
    with c2:
        app = st.selectbox("Radar — app vs category", sorted(df["app_name"].unique()))
        r = b[b["app_name"] == app]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=r["app_percentage"], theta=r["theme_label"], fill="toself", name=app,
                                      line_color=colors.get(app, "#6366f1")))
        fig.add_trace(go.Scatterpolar(r=r["category_avg_percentage"], theta=r["theme_label"], fill="toself",
                                      name="Category avg", line_color="rgba(148,163,184,0.6)"))
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)"))
        show(fig, height=320)

    st.markdown("#### Full significance table")
    only_sig = st.toggle("Only significant differences", False)
    t = b[b["significant"]] if only_sig else b
    st.dataframe(t.sort_values(["significant", "delta"], ascending=[False, False])[
        ["app_name", "theme_label", "count", "app_percentage", "ci_low", "ci_high", "peer_percentage", "delta", "p_value", "q_value", "status"]],
        use_container_width=True, hide_index=True, column_config={
            "app_name": "App", "theme_label": "Theme", "count": "n",
            "app_percentage": st.column_config.NumberColumn("App %", format="%.1f%%"),
            "ci_low": st.column_config.NumberColumn("CI low", format="%.1f%%"),
            "ci_high": st.column_config.NumberColumn("CI high", format="%.1f%%"),
            "peer_percentage": st.column_config.NumberColumn("Peers %", format="%.1f%%"),
            "delta": st.column_config.NumberColumn("Δ pp", format="%+.1f"),
            "p_value": st.column_config.NumberColumn("p", format="%.4f"),
            "q_value": st.column_config.NumberColumn("q (BH)", format="%.4f"), "status": "Status"})


# ═══════════════════════════════════════════════════════════════
#  PAGE — RATING DRIVERS
# ═══════════════════════════════════════════════════════════════
def render_drivers(ws, df):
    st.title("Rating Drivers")
    if needs_data(df):
        return
    st.caption("**Rating drag** = theme share × (avg ★ of other reviews − avg ★ within theme). It estimates how many stars an "
               "app's average would gain if that theme's reviewers were as happy as everyone else — the ROI lens on fixing it.")
    app = st.selectbox("Scope", ["All apps"] + sorted(df["app_name"].unique()))
    drv = A.rating_drivers(df, ws.taxonomy, None if app == "All apps" else app)
    if drv.empty:
        st.info("Not enough issue reviews to compute drivers.")
        return
    drv["significance"] = np.where(drv["p_value"] < 0.05, "p < 0.05", "not significant")
    top = drv.iloc[0]
    callout(f"Fixing <b>{top.theme_label}</b> is the biggest rating lever{'' if app == 'All apps' else ' for ' + app}: "
            f"worth ≈ <b>+{top.rating_drag:.2f}★</b> on the average. Reviews in this theme average {top.theme_avg_rating:.2f}★ "
            f"vs {top.others_avg_rating:.2f}★ for the rest.", "Biggest lever")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(drv.sort_values("rating_drag"), x="rating_drag", y="theme_label", orientation="h", color="significance",
                     color_discrete_map={"p < 0.05": "#2563eb", "not significant": "#cbd5e1"},
                     labels={"rating_drag": "Potential ★ gain", "theme_label": ""})
        show(fig, "Rating drag by theme", height=420)
    with c2:
        fig = px.scatter(drv, x="share_pct", y="theme_avg_rating", size="reviews", text="theme_label", color="rating_drag",
                         color_continuous_scale="Reds", labels={"share_pct": "% of reviews", "theme_avg_rating": "Avg ★ in theme"})
        fig.update_traces(textposition="top center", textfont_size=10)
        show(fig, "Frequency vs. pain", height=420)

    if df["app_name"].nunique() > 1:
        rows = [A.rating_drivers(df, ws.taxonomy, a).assign(app_name=a) for a in df["app_name"].unique()]
        m = pd.concat([r for r in rows if not r.empty])
        if not m.empty:
            pv = m.pivot(index="theme_label", columns="app_name", values="rating_drag").fillna(0)
            show(px.imshow(pv.round(3), text_auto=True, color_continuous_scale="Reds", aspect="auto",
                           labels=dict(color="★ drag")), "Rating drag by app — where each competitor bleeds stars")


# ═══════════════════════════════════════════════════════════════
#  PAGE — RELEASE IMPACT
# ═══════════════════════════════════════════════════════════════
def render_releases(ws, df):
    st.title("Release Impact")
    if needs_data(df):
        return
    min_n = st.slider("Minimum reviews per version", 5, 100, 15)
    v = A.version_impact(df, ws.taxonomy, min_n)
    if v.empty:
        st.info("No version information yet (older scrapes didn't capture it) or no version meets the minimum. "
                "Re-scrape, or lower the threshold.")
        return
    st.caption("Each version is compared with the previous one (Mann-Whitney U on star ratings). "
               "Newest-review sampling usually covers only the last few releases; merge scrapes over time for more.")
    colors = ws.app_colors()
    v["version_order"] = v.groupby("app_name").cumcount()
    fig = px.line(v, x="version", y="avg_rating", color="app_name", markers=True, color_discrete_map=colors,
                  hover_data=["reviews", "pct_1_2_star", "top_issue", "flag"],
                  labels={"avg_rating": "Avg ★", "version": "App version", "app_name": "App"})
    fig.update_xaxes(type="category")
    show(fig, "Average rating by app version")
    st.dataframe(v.drop(columns=["version_order"]), use_container_width=True, hide_index=True, column_config={
        "app_name": "App", "version": "Version", "reviews": "Reviews",
        "first_seen": st.column_config.DatetimeColumn("First seen", format="D MMM YYYY"),
        "avg_rating": st.column_config.NumberColumn("Avg ★", format="%.2f"),
        "pct_1_2_star": st.column_config.NumberColumn("% 1-2★", format="%.1f%%"),
        "high_sev_pct": st.column_config.NumberColumn("% High sev", format="%.1f%%"),
        "top_issue": "Top issue", "delta_vs_prev": st.column_config.NumberColumn("Δ★ vs prev", format="%+.2f"),
        "p_value": st.column_config.NumberColumn("p", format="%.3f"), "flag": "Flag"})


# ═══════════════════════════════════════════════════════════════
#  PAGE — DEVELOPER RESPONSES
# ═══════════════════════════════════════════════════════════════
def render_responses(ws, df):
    st.title("Developer Responses")
    if needs_data(df):
        return
    per_app, by_theme = A.response_analytics(df, ws.taxonomy)
    if per_app.empty or per_app["reply_rate"].sum() == 0:
        st.info("No developer replies found. Replies are only available from Google Play, and older scrapes didn't capture them.")
        return
    st.caption("Google Play only — Apple's public feed doesn't expose developer replies.")
    colors = ws.app_colors()
    cols = st.columns(len(per_app))
    for col, r in zip(cols, per_app.itertuples()):
        with col:
            hrs = f"{r.median_hours:.0f} h" if pd.notna(r.median_hours) else "—"
            card(r.app_name, f"{r.reply_rate:.0f}% replied",
                 f"1-2★ reply rate: {r.reply_rate_negative:.0f}%<br>Median time to reply: {hrs}", colors.get(r.app_name))
    c1, c2 = st.columns(2)
    with c1:
        m = per_app.melt(id_vars="app_name", value_vars=["reply_rate", "reply_rate_negative", "reply_rate_high_sev"])
        m["variable"] = m["variable"].map({"reply_rate": "All reviews", "reply_rate_negative": "1-2★ reviews",
                                           "reply_rate_high_sev": "High-severity"})
        show(px.bar(m, x="app_name", y="value", color="variable", barmode="group",
                    color_discrete_sequence=["#94a3b8", "#2563eb", "#b91c1c"],
                    labels={"value": "% replied", "app_name": "", "variable": ""}), "Reply coverage")
    with c2:
        pv = by_theme[by_theme["reviews"] >= 3].pivot(index="theme_label", columns="app_name", values="reply_rate")
        if not pv.empty:
            show(px.imshow(pv.round(0), text_auto=True, color_continuous_scale="Blues", aspect="auto",
                           labels=dict(color="% replied")), "Reply rate by theme")


# ═══════════════════════════════════════════════════════════════
#  PAGE — FEATURE REQUESTS
# ═══════════════════════════════════════════════════════════════
def render_features(ws, df):
    st.title("Feature Requests")
    if needs_data(df):
        return
    app = st.selectbox("Scope", ["All apps"] + sorted(df["app_name"].unique()))
    d = df if app == "All apps" else df[df["app_name"] == app]
    fr = A.feature_request_clusters(d)
    if fr.empty:
        st.info("No feature requests extracted yet. Requests are captured for reviews classified with the new pipeline.")
        return
    st.caption("Requests extracted by the LLM, grouped by text similarity. Demand score = mentions + log(1 + helpful votes).")
    top = fr.head(15)
    fig = px.bar(top.iloc[::-1], x="mentions", y="request", orientation="h", color="helpful_votes",
                 color_continuous_scale="Blues", labels={"mentions": "Mentions", "request": "", "helpful_votes": "👍"})
    show(fig, "Top requested features", height=max(300, 28 * len(top)))
    st.dataframe(fr, use_container_width=True, hide_index=True, column_config={
        "request": "Request", "mentions": "Mentions", "helpful_votes": "👍 votes", "apps": "Apps",
        "avg_rating": st.column_config.NumberColumn("Avg ★ of requesters", format="%.2f"),
        "examples": st.column_config.TextColumn("Example reviews", width="large"),
        "demand_score": st.column_config.NumberColumn("Demand", format="%.1f")})


# ═══════════════════════════════════════════════════════════════
#  PAGE — TEXT LENS
# ═══════════════════════════════════════════════════════════════
def render_text(ws, df):
    st.title("Text Lens")
    if needs_data(df):
        return
    colors = ws.app_colors()
    st.markdown("#### What's distinctive about each app's complaints")
    st.caption("Words and phrases over-represented in an app's ≤3★ reviews relative to competitors — "
               "weighted log-odds with an informative Dirichlet prior (z-scores; > 1.96 is significant).")
    terms = A.distinctive_terms(df)
    if terms.empty:
        st.info("Needs at least two apps and ~30 negative reviews.")
    else:
        apps = terms["app_name"].unique()
        cols = st.columns(min(len(apps), 3))
        for i, app in enumerate(apps):
            t = terms[terms["app_name"] == app].sort_values("z")
            with cols[i % len(cols)]:
                show(px.bar(t, x="z", y="term", orientation="h", color_discrete_sequence=[colors.get(app, "#2563eb")],
                            hover_data=["mentions"], labels={"z": "z-score", "term": ""}), app, height=420)

    st.markdown("#### Rating ↔ sentiment mismatches")
    st.caption("**Hidden detractors** leave 4-5★ but write complaints — at risk of churning silently. "
               "**Grumpy fans** leave 1-2★ with positive text — often mis-taps or one-off frustrations; a low-cost win-back.")
    mm = A.rating_sentiment_mismatch(df)
    if mm.empty:
        st.success("No mismatches found.")
        return
    cnt = mm.groupby(["app_name", "type"]).size().reset_index(name="n")
    cnt["pct"] = cnt["n"] / cnt["app_name"].map(df.groupby("app_name").size()) * 100
    show(px.bar(cnt, x="app_name", y="pct", color="type", barmode="group",
                color_discrete_sequence=["#b91c1c", "#15803d"], labels={"pct": "% of app's reviews", "app_name": "", "type": ""}),
         height=320)
    st.dataframe(mm[["app_name", "type", "rating", "sentiment_score", "theme", "one_line_summary", "review_text"]],
                 use_container_width=True, hide_index=True, height=320,
                 column_config={"review_text": st.column_config.TextColumn("Review", width="large")})


# ═══════════════════════════════════════════════════════════════
#  PAGE — ACTION PLAN
# ═══════════════════════════════════════════════════════════════
def render_action_plan(ws, df):
    st.title("Action Plan")
    if needs_data(df):
        return
    app = st.selectbox("Build the backlog for", ["All apps"] + sorted(df["app_name"].unique()))
    d = df if app == "All apps" else df[df["app_name"] == app]
    p = A.compute_priority_scores(d, ws.taxonomy)
    if p.empty:
        st.info("Not enough data.")
        return

    st.markdown("#### Prioritisation matrix")
    fig = px.scatter(p, x="effort", y="frequency_severity_gap", size=p["reviews"].clip(lower=1), color="impact",
                     text="theme_label", color_continuous_scale="Reds", size_max=50, range_x=[-0.5, 10.5], range_y=[-0.5, 10.8],
                     hover_data=["reviews", "high_sev_pct", "priority_score"],
                     labels={"effort": "Fix effort →", "frequency_severity_gap": "User pain (volume × severity) →", "impact": "Impact"})
    fig.update_traces(textposition="top center", textfont_size=11)
    for x, y, t in [(2.5, 10.4, "QUICK WINS"), (7.5, 10.4, "BIG BETS"), (2.5, 0.0, "FILL-INS"), (7.5, 0.0, "DEPRIORITISE")]:
        fig.add_annotation(x=x, y=y, text=t, showarrow=False, font=dict(size=11, color="#94a3b8"))
    fig.add_hline(y=5, line_dash="dot", line_color="#cbd5e1")
    fig.add_vline(x=5, line_dash="dot", line_color="#cbd5e1")
    show(fig, height=520)

    st.markdown("#### Ranked backlog")
    t = p.copy()
    t.insert(0, "rank", range(1, len(t) + 1))
    st.dataframe(t[["rank", "theme_label", "priority_score", "reviews", "share_pct", "high_sev_pct", "helpful_votes",
                    "impact", "effort", "quick_win", "justification"]],
                 use_container_width=True, hide_index=True, column_config={
                     "rank": "#", "theme_label": "Theme",
                     "priority_score": st.column_config.ProgressColumn("Priority", format="%.1f", min_value=0,
                                                                       max_value=float(max(t["priority_score"].max(), 1))),
                     "reviews": "Reviews", "share_pct": st.column_config.NumberColumn("% reviews", format="%.1f%%"),
                     "high_sev_pct": st.column_config.NumberColumn("% high sev", format="%.0f%%"),
                     "helpful_votes": st.column_config.NumberColumn("👍 reach", format="%d"),
                     "impact": st.column_config.NumberColumn("Impact", format="%.0f"),
                     "effort": st.column_config.NumberColumn("Effort", format="%.0f"),
                     "quick_win": st.column_config.CheckboxColumn("Quick win"), "justification": "Why"})

    qw = p[p["quick_win"] & (p["reviews"] > 0)]
    if not qw.empty:
        st.markdown("#### Quick wins")
        for r in qw.itertuples():
            st.markdown(f"""<div style="background:#fff;border:1px solid #bbf7d0;border-left:4px solid #16a34a;border-radius:9px;
                padding:12px 16px;margin-bottom:8px;"><div style="display:flex;justify-content:space-between">
                <span style="font-weight:600;color:#166534;font-size:13px">{r.theme_label}</span>
                <span class="badge badge-green">Score {r.priority_score:.1f}</span></div>
                <div style="font-size:13px;color:#475569;margin-top:4px">{r.reviews} reviews · {r.high_sev_pct:.0f}% high severity · {r.justification}</div></div>""",
                        unsafe_allow_html=True)

    st.download_button("Export action plan (CSV)", p.to_csv(index=False).encode("utf-8"), f"{ws.slug}_action_plan.csv", "text/csv")
    with st.expander("Scoring methodology & assumptions"):
        st.markdown("`Priority = Impact × 4 + Frequency/Severity Gap × 4 − Effort × 2`  \n"
                    "Impact & Effort come from the workspace taxonomy (edit them in **Setup → Taxonomy**). "
                    "Gap = severity-weighted volume (high 3, medium 1.5, low 1) scaled 0-10 against the worst theme. "
                    "Quick win = Impact ≥ 6 and Effort ≤ 4.")
        st.dataframe(taxonomy_to_df(ws.taxonomy)[["label", "impact", "effort", "description"]], hide_index=True,
                     use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE — AI BRIEF & EXPORT
# ═══════════════════════════════════════════════════════════════
def render_brief(ws, df):
    st.title("AI Brief & Export")
    if needs_data(df):
        return
    st.caption("The LLM sees only pre-computed statistics (the fact sheet), never raw counts to estimate — "
               "so every number in the brief traces back to the analytics.")
    if st.button("✨ Generate executive brief", type="primary"):
        with st.spinner("Writing the brief…"):
            generate_ai_brief(df, ws)
    brief = ws.brief_path.read_text(encoding="utf-8") if ws.brief_path.exists() else None
    if brief:
        st.caption(f"Last generated: {pd.Timestamp(_mtime(ws.brief_path), unit='s'):%d %b %Y %H:%M} (UTC)")
        with st.container(border=True):
            st.markdown(brief)
    else:
        st.info("No brief yet — click the button above.")

    st.markdown("#### Export")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Full report (Markdown)", build_markdown_report(df, ws, brief).encode("utf-8"),
                       f"{ws.slug}_report.md", "text/markdown", use_container_width=True)
    c2.download_button("Classified reviews (CSV)", df.to_csv(index=False).encode("utf-8"),
                       f"{ws.slug}_classified.csv", "text/csv", use_container_width=True)
    c3.download_button("Fact sheet (JSON)", json.dumps(build_fact_sheet(df, ws), default=str, indent=2).encode("utf-8"),
                       f"{ws.slug}_facts.json", "application/json", use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE — VALIDATION
# ═══════════════════════════════════════════════════════════════
def render_validation(ws, df):
    st.title("Classification Validation")
    if needs_data(df):
        return
    st.caption("Hand-label a random, app-stratified sample; we compare it with the LLM's labels. "
               "LLM labels are hidden while you label to avoid anchoring bias. κ > 0.6 = substantial agreement.")
    path = ws.root / "validation.csv"
    if not path.exists():
        n = st.slider("Sample size", 20, 150, 50, 10)
        if st.button("Create labelling sample", type="primary"):
            workspace_label_sample(df, n).to_csv(path, index=False)
            st.rerun()
        return

    labels = pd.read_csv(path, dtype={"review_id": str})
    edited = st.data_editor(labels, use_container_width=True, hide_index=True, height=380, disabled=["review_id", "app_name", "rating", "review_text"],
                            column_config={
                                "review_id": None,
                                "review_text": st.column_config.TextColumn("Review", width="large"),
                                "manual_theme": st.column_config.SelectboxColumn("Your theme", options=tax_options(ws)),
                                "manual_severity": st.column_config.SelectboxColumn("Your severity", options=["high", "medium", "low"]),
                            })
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Save labels", type="primary"):
        edited.to_csv(path, index=False)
        st.success("Saved.")
    if c2.button("New sample"):
        path.unlink()
        st.rerun()

    res = workspace_agreement(df.assign(review_id=df["review_id"].astype(str)), edited)
    if res.get("n", 0) < 5:
        st.info(f"Label at least 5 reviews to see agreement metrics ({res.get('n', 0)} labelled so far).")
        return
    c = st.columns(4)
    c[0].metric("Labelled", res["n"])
    c[1].metric("Theme accuracy", f"{res['theme_accuracy']:.0%}")
    c[2].metric("Theme κ", f"{res['theme_kappa']:.2f}")
    c[3].metric("Severity accuracy", f"{res['severity_accuracy']:.0%}" if pd.notna(res["severity_accuracy"]) else "—")
    lbl = ws.theme_labels()
    names = [lbl.get(k, k) for k in res["labels"]]
    show(px.imshow(res["confusion"], x=names, y=names, text_auto=True, color_continuous_scale="Blues",
                   labels=dict(x="LLM", y="Human", color="n")), "Confusion matrix (rows = human)", height=460)
    if not res["disagreements"].empty:
        st.markdown("#### Disagreements")
        st.dataframe(res["disagreements"], use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
PAGES = {
    "Setup": render_setup, "Overview": render_overview, "Market Snapshot": render_market,
    "Theme Breakdown": render_themes, "Trends & Severity": render_trends, "Competitive Benchmark": render_benchmark,
    "Rating Drivers": render_drivers, "Release Impact": render_releases, "Developer Responses": render_responses,
    "Feature Requests": render_features, "Text Lens": render_text, "Action Plan": render_action_plan,
    "AI Brief & Export": render_brief, "Validation": render_validation,
}


def main():
    inject_custom_css()
    workspaces = list_workspaces()
    slugs = [w.slug for w in workspaces]
    names = {w.slug: w.name for w in workspaces}
    if "ws" not in st.session_state:
        st.session_state.ws = slugs[0] if slugs else None

    with st.sidebar:
        st.markdown("""
        <div style="padding: 8px 4px 14px 4px;">
            <div style="font-size: 18px; font-weight: 600; color: #1c2230; letter-spacing: -0.5px;">Review Engine</div>
            <div style="font-size: 12px; color: #6b7384; font-weight: 400; margin-top: 2px;">App Review Intelligence</div>
        </div>
        <div class="section-header" style="margin-bottom: 6px;">Workspace</div>
        """, unsafe_allow_html=True)
        options = slugs + [NEW_WS]
        current = st.session_state.ws if st.session_state.ws in slugs else NEW_WS
        choice = st.selectbox("Workspace", options, index=options.index(current), label_visibility="collapsed",
                              format_func=lambda s: names.get(s, s))
        if choice != current:
            st.session_state.ws = None if choice == NEW_WS else choice
            st.session_state.pop("search_results", None)
            st.session_state.pop("proposed_tax", None)
            st.session_state.nav_to = "Setup" if choice == NEW_WS else "Overview"
            st.rerun()

        ws = load_workspace(st.session_state.ws) if st.session_state.ws else None
        st.markdown('<div class="section-header" style="margin:14px 0 8px 0;">Navigation</div>', unsafe_allow_html=True)
        page_names = list(PAGES) if ws else ["Setup"]
        if "nav_to" in st.session_state:
            st.session_state.page = st.session_state.pop("nav_to")
        if st.session_state.get("page") not in page_names:
            st.session_state.page = "Overview" if ws else "Setup"
        page = st.radio("Navigate", page_names, key="page", label_visibility="collapsed")

        raw = df = pd.DataFrame()
        if ws:
            raw, df = get_data(ws)
            st.markdown("<hr style='margin: 14px 0; border: 0; border-top: 1px solid #e6e8f2;'>", unsafe_allow_html=True)
            st.markdown('<div class="section-header" style="margin-bottom: 8px;">Dataset</div>', unsafe_allow_html=True)
            st.markdown(f"**Scraped:** {len(raw):,} &nbsp;·&nbsp; **Classified:** {len(df):,}")
            colors = ws.app_colors()
            for a in ws.apps:
                n = int((df["app_name"] == a.name).sum()) if not df.empty else 0
                st.markdown(f"""
                <div style="display:flex;align-items:center;justify-content:space-between;padding:3px 0;font-size:13px;color:#475569;">
                    <div style="display:flex;align-items:center;">
                        <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{colors[a.name]};margin-right:8px;"></span>
                        <span>{a.name}</span><span style="font-size:11px;color:#94a3b8;margin-left:6px">{'iOS' if a.platform == 'appstore' else 'Play'}</span>
                    </div>
                    <span style="font-variant-numeric:tabular-nums;font-weight:500;color:#1c2230;">{n:,}</span>
                </div>""", unsafe_allow_html=True)
            if len(raw) > len(df):
                st.caption(f"{len(raw) - len(df):,} reviews pending classification → Setup › Run Pipeline")
        st.markdown("---")
        st.caption("App Review Intelligence Engine")

    if page == "Setup":
        render_setup(ws)
    elif page == "Overview":
        render_overview(ws, df, raw)
    else:
        PAGES[page](ws, df)


if __name__ == "__main__":
    main()
