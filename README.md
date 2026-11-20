# App Review Intelligence Engine

> Pick any apps on Google Play or the Apple App Store → scrape their reviews → let an LLM classify every review → get significance-tested competitive benchmarks, rating drivers, release-regression alerts and a prioritised product roadmap.

Started as a fintech-only analysis of PhonePe, Paytm and Google Pay; now works for **any category, any app, any store market**.

## The Problem
Product teams drown in app-store reviews. Star ratings tell you *if* users are unhappy, not *why*, not *how you compare to competitors*, and not *what to fix first*. Manual triage doesn't scale, and naive dashboards mistake noise for signal.

## What it does
| Step | What happens |
|---|---|
| **1. Workspace** | Create a workspace (e.g. *Swiggy vs Zomato*), search either store by name / URL / package id, add up to 8 apps — mixing Android and iOS is fine. One-click templates for food delivery, quick commerce, ride-hailing, streaming and e-commerce. |
| **2. Taxonomy** | Start from a category preset (10 built in) **or let the LLM propose a domain-specific taxonomy from a sample of real reviews**. Edit themes, impact and effort in the UI. |
| **3. Scrape** | Newest or most-relevant reviews plus store metadata (rating histogram, installs, version, ads/IAP). Re-runs **merge** with past scrapes, so the dataset grows into a time series. |
| **4. Classify** | Batched LLM calls tag each review with theme, severity, sentiment, a one-line summary and any **feature request**. SHA-256 cache, invalidated automatically when the taxonomy changes. |
| **5. Analyse & act** | 14 dashboard pages (below), an AI executive brief, and a Markdown report export. |

## Dashboard pages
- **Overview** — KPIs, app scorecards (avg ★, NPS proxy, top issue), headline insight, automatic **spike** and **release-regression alerts**
- **Market Snapshot** — store metadata; *lifetime vs recent rating momentum*; lifetime star distribution
- **Theme Breakdown** — normalised theme share per app, treemap, **voice-of-customer quotes** (ranked by helpful votes and severity), filterable review explorer with CSV export
- **Trends & Severity** — rating, sentiment and theme-share trends (time bucket adapts to the data span), friction index, severity heatmap
- **Competitive Benchmark** — each app vs its pooled peers: two-proportion z-tests with **Benjamini-Hochberg FDR correction**, **Wilson 95% CIs**, delta heatmap, radar
- **Rating Drivers** — key-driver analysis: *rating drag* = share × (★ of other reviews − ★ in theme), i.e. stars gained if the issue were fixed, with Welch t-tests
- **Release Impact** — rating by app version; each release is compared to the previous one (Mann-Whitney U) to flag regressions
- **Developer Responses** — reply rate overall / on 1-2★ / on high-severity reviews, median hours to reply, reply rate by theme
- **Feature Requests** — LLM-extracted requests clustered by TF-IDF similarity, ranked by mentions and helpful votes
- **Text Lens** — phrases distinctive to each app's complaints (**weighted log-odds with informative Dirichlet prior**, Monroe et al. 2008); *hidden detractors* (5★ with negative text) and *grumpy fans*
- **Action Plan** — impact/effort **2×2 prioritisation matrix**, ranked backlog per app or all apps, quick wins, CSV export
- **AI Brief & Export** — the LLM writes a TL;DR / risks / 30-60-90 plan **from a pre-computed fact sheet and verbatim quotes only**, so numbers aren't hallucinated; export Markdown report, classified CSV, fact-sheet JSON
- **Validation** — hand-label a stratified sample in the UI (LLM labels hidden to avoid anchoring) → accuracy, Cohen's κ, confusion matrix, disagreements

## Quick Start
```bash
pip install -r requirements.txt
cp .env.example .env          # add an LLM API key (Groq has a free tier)
streamlit run app.py
```
In the app: **＋ New workspace** → pick a template or add apps → **Setup › Run Pipeline**.

The original UPI dataset is migrated automatically into an *Indian UPI Apps* workspace on first launch.

## Architecture
```text
src/
├── workspace.py      # Workspace + AppRef model, on-disk layout, legacy migration
├── scraper.py        # Play + App Store search, metadata, reviews (one normalised schema)
├── taxonomy.py       # Category presets + LLM-generated taxonomies
├── llm_classify.py   # Batched, cached, taxonomy-aware classification (Groq/OpenAI/Anthropic/Gemini)
├── analytics.py      # All statistics: benchmarks, drivers, versions, spikes, text analytics, priorities
├── insights.py       # Fact sheet → AI executive brief, Markdown report
└── validation.py     # Human-vs-LLM agreement metrics
prompts/classify_review.txt   # Classification prompt template ({{CONTEXT}}, {{THEMES}}, {{REVIEWS}})
data/workspaces/<slug>/       # workspace.json, raw/*.csv, app_meta.json, cache.json, classified.csv, brief.md, validation.csv
```

## Methodology notes
- **Severity:** *high* blocks the core job, or causes money loss, a safety risk, data loss or lock-out. *Medium* is a significant degradation with a workaround. *Low* is cosmetic, minor, or praise.
- **Priority score:** `Impact × 4 + Frequency/Severity Gap × 4 − Effort × 2`. Impact and effort live in the workspace taxonomy. The gap is the severity-weighted volume (3 / 1.5 / 1), scaled 0–10. A quick win is Impact ≥ 6 and Effort ≤ 4.
- **NPS proxy:** % 5★ − % 1–3★. It is a rating-based proxy, not a survey NPS.
- **Significance:** results are FDR-controlled across every app × theme test, so a "significant" flag survives multiple comparisons.

## Limitations
- *Newest* sampling for popular apps covers only hours to days. Re-run with **Merge** periodically, or use *Most relevant* sampling, to build history for trends and release analysis.
- Apple's public review feed caps at 500 reviews per app and doesn't expose developer replies.
- LLM labels are probabilistic. Use the Validation page to measure agreement for each workspace.
- Mixed-language reviews (e.g. Hinglish) are classified by the LLM but excluded from English-tokenised text analytics.
