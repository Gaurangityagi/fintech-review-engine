# Fintech Review Intelligence Engine

> AI-powered analysis of 2,000+ customer reviews across India's top UPI payment apps — turning raw app-store feedback into a prioritized product improvement roadmap.

## The Problem
Fintech product teams drown in thousands of app store reviews. Star ratings tell you *if* users are unhappy, but not *why* — and not which issues to fix first. Manual review triage doesn't scale.

## The Solution
This engine scrapes Play Store reviews for PhonePe, Paytm, and Google Pay, then uses an LLM to classify each review into structured product signals: failure themes, severity levels, and sentiment scores. The output is a prioritized action plan that reads like a PM's quarterly review audit.

## Headline Finding
> "X% of negative reviews across Indian UPI apps trace to failed transactions, with PhonePe showing a measurable improvement trend while Paytm's transaction failure complaints worsened over the past two quarters."

*(Finding updates automatically based on actual scraped data)*

## Key Features
- **Automated Scraping**: 500-1,000 reviews per app via google-play-scraper
- **LLM Classification**: Batched GPT-4o-mini calls with 9-theme taxonomy and severity grading
- **Validation Layer**: 50-review manual label comparison with Cohen's κ agreement metrics — because LLM output needs proof, not blind trust
- **Severity-Weighted Scoring**: Composite friction index normalized per 100 reviews for fair cross-app comparison
- **Trend Analysis**: Monthly theme frequency tracking to detect improving vs worsening issues
- **Priority Action Plan**: Impact × Frequency − Effort scoring with quick-win flags

## Architecture
1. **Scraping**: `google-play-scraper` extracts recent reviews for PhonePe, Paytm, and Google Pay.
2. **Classification**: Reviews are processed in batches via LLM (GPT-4o-mini/Claude 3.5 Haiku) to assign a specific theme and severity level based on a predefined taxonomy.
3. **Validation**: A subset of reviews is compared against manual labels using Cohen's Kappa to ensure classification reliability.
4. **Analysis & Scoring**: Reviews are aggregated, and a Priority Score is calculated for each theme using the formula: `(Impact × 4) + (Frequency/Severity Gap × 4) − (Assumed Fix Effort × 2)`.
5. **Visualization**: A Streamlit dashboard presents interactive charts (Plotly) for trend analysis and cross-app comparisons.

## Tech Stack
- Python 3.10+
- Streamlit (dashboard)
- Plotly (interactive charts)
- OpenAI / Anthropic API (classification)
- google-play-scraper (data collection)
- scikit-learn (validation metrics)

## Quick Start
```bash
# Clone and install
git clone <repo-url>
cd fintech-review-engine
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API key

# Scrape reviews
python -c "from src.scraper import scrape_all_apps; scrape_all_apps()"

# Classify reviews  
python -c "from src.scraper import load_raw_reviews; from src.llm_classify import classify_dataframe; classify_dataframe(load_raw_reviews())"

# Launch dashboard
streamlit run app.py
```

## Methodology

### Theme Taxonomy
We use a 9-theme taxonomy tailored for Fintech applications:
- `failed_transaction`: Money deducted but not credited, failed UPI transfers.
- `kyc_friction`: Issues with identity verification and onboarding.
- `refund_delay`: Stuck refunds and delayed reversals.
- `customer_support`: Complaints about unresponsive or unhelpful support.
- `app_crash_bug`: Technical glitches, freezes, and crashes.
- `ui_confusion`: Poor UX, navigation issues, and confusing interfaces.
- `fraud_security_concern`: Unauthorized transactions, scams, and security fears.
- `praise_no_issue`: Positive feedback and general praise.
- `other`: Unrelated or unclassifiable reviews.

### Severity Framework
- **High**: Blocks core functionality (e.g., lost money, cannot transact, account locked).
- **Medium**: Significant friction but workaround exists (e.g., slow UI, delayed but eventual refund).
- **Low**: Minor annoyance or aesthetic issue (e.g., typo, ugly icon).

### Priority Score Formula
The Priority Action Plan uses a weighted formula to rank issues:
`Priority Score = (Impact × 4) + (Frequency/Severity Gap × 4) − (Assumed Fix Effort × 2)`
- **Impact**: Proportion of high-severity reviews for the theme.
- **Frequency/Severity Gap**: How often the issue occurs relative to others.
- **Assumed Fix Effort**: Estimated engineering/ops effort to resolve the root cause.
This ensures product managers focus on high-impact, low-effort wins first.

## Validation Approach
To ensure the LLM's classifications are reliable, we maintain a validation layer. A random sample of 50 reviews is manually labeled for both theme and severity. We compute simple accuracy and Cohen's Kappa agreement metrics between the manual labels and the LLM outputs. This approach proves that the automated triage is grounded in reality, addressing the common "black box" criticism of LLM data pipelines. A Kappa score above 0.6 indicates substantial agreement, providing confidence in the aggregate trends.

## Project Structure
```text
fintech-review-engine/
├── .env.example
├── .streamlit/
│   └── config.toml
├── README.md
├── validation_report.md
├── requirements.txt
├── app.py
├── data/
│   ├── raw/                    # Scraped CSVs per app
│   ├── classified/             # LLM cache + classified output
│   └── validation/             # Manual labels for validation
│       └── manual_labels.csv
├── prompts/
│   └── classify_review.txt     # LLM classification prompt
└── src/
    ├── __init__.py
    ├── scraper.py              # Play Store scraper
    ├── llm_classify.py         # Batched LLM classification + caching
    ├── analytics.py            # Scoring & aggregation functions
    └── validation.py           # Agreement metrics (accuracy, Cohen's κ)
```

## 60-Second Elevator Pitch
"We're turning unstructured app store complaints into a prioritized product roadmap. By scraping thousands of reviews across India's top UPI apps and running them through a custom LLM classification pipeline, our engine identifies exactly why users are frustrated—be it KYC friction or stuck refunds. We don't just count bad reviews; we grade them by severity and calculate a Priority Score based on impact and fix effort. The result is a Streamlit dashboard that tells Fintech PMs exactly what to fix next to move the needle on user satisfaction, backed by rigorous validation metrics."

## Limitations & Future Work
- Scraper may be rate-limited by Google Play
- LLM classification has ~X% disagreement with manual labels
- Taxonomy is English-focused; Hinglish reviews may be misclassified
- Future: Add iOS App Store reviews, multilingual support, automated periodic scraping
