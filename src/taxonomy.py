"""
Theme taxonomies. Each theme is a dict:
    key          snake_case id the LLM must return
    label        display name
    description  what belongs in the theme (goes into the prompt)
    impact       0-10, business impact if the issue goes unfixed (used in priority score)
    effort       0-10, assumed fix effort (used in priority score)

Every taxonomy always ends with `praise_no_issue` and `other`.
"""
import json
import re

import pandas as pd

PRAISE = {"key": "praise_no_issue", "label": "Praise / No Issue",
          "description": "Positive review, satisfied user, praise for features", "impact": 0, "effort": 0}
OTHER = {"key": "other", "label": "Other",
         "description": "Doesn't fit any other category, or is unintelligible", "impact": 3, "effort": 5}
RESERVED_KEYS = {PRAISE["key"], OTHER["key"]}


def _t(key, label, description, impact, effort):
    return {"key": key, "label": label, "description": description, "impact": impact, "effort": effort}


PRESETS: dict[str, list[dict]] = {
    "General (any app)": [
        _t("bugs_crashes", "Bugs / Crashes", "App crashes, freezes, errors, features not working", 8, 6),
        _t("performance", "Performance", "Slow loading, lag, battery drain, heavy app size", 6, 6),
        _t("login_account", "Login / Account", "Can't log in, OTP issues, account blocked or deleted", 8, 5),
        _t("ux_usability", "UX / Usability", "Confusing navigation, hard to find features, bad design", 5, 3),
        _t("pricing_value", "Pricing / Value", "Too expensive, subscription complaints, hidden charges", 6, 5),
        _t("ads_spam", "Ads / Spam", "Too many ads, notifications spam, unwanted promotions", 5, 2),
        _t("customer_support", "Customer Support", "No response, unhelpful support, can't reach a human", 6, 4),
        _t("missing_feature", "Missing Feature", "Requests for new features or functionality that doesn't exist", 4, 5),
        _t("privacy_security", "Privacy / Security", "Data misuse, permissions, hacked accounts, scams", 9, 7),
    ],
    "Fintech / Payments": [
        _t("failed_transaction", "Failed Transaction", "Payment failures, transaction timeouts, money debited but not credited, UPI errors", 10, 9),
        _t("kyc_friction", "KYC Friction", "KYC verification issues, document upload problems, account verification delays", 7, 5),
        _t("refund_delay", "Refund Delay", "Refund not received, cashback not credited, money stuck in processing", 8, 6),
        _t("customer_support", "Customer Support", "Poor support experience, no response from support, unhelpful agents", 6, 4),
        _t("app_crash_bug", "App Crash / Bug", "App crashes, freezes, slow loading, technical bugs, login issues", 7, 6),
        _t("ui_confusion", "UI Confusion", "Confusing navigation, hard to find features, poor UX design", 5, 3),
        _t("fraud_security_concern", "Fraud / Security", "Unauthorized transactions, phishing, security worries, account hacked", 9, 8),
    ],
    "Food & Grocery Delivery": [
        _t("late_delivery", "Late Delivery", "Order delivered late, ETA wrong, order never arrived", 9, 7),
        _t("wrong_missing_items", "Wrong / Missing Items", "Wrong order, missing items, spilled or damaged food", 8, 5),
        _t("food_quality", "Food / Product Quality", "Stale, cold, bad quality food or groceries, expired products", 7, 6),
        _t("refund_cancellation", "Refund / Cancellation", "Refund not given, cancellation charges, order cancelled by platform", 8, 5),
        _t("pricing_fees", "Pricing / Fees", "High delivery fees, surge, platform fee, price higher than menu", 7, 4),
        _t("delivery_partner", "Delivery Partner Behaviour", "Rude or unprofessional delivery agent, asking for extra money", 6, 5),
        _t("customer_support", "Customer Support", "Chatbot loops, no human support, unresolved complaints", 7, 4),
        _t("app_bug", "App Bug / Payment Issue", "Crashes, payment failures, coupon not applying, tracking broken", 7, 6),
    ],
    "E-commerce / Shopping": [
        _t("delivery_issue", "Delivery Issue", "Late, failed or fake delivery attempts, wrong address handling", 8, 7),
        _t("product_quality", "Product Quality / Fake", "Defective, counterfeit, not as described products", 8, 6),
        _t("returns_refunds", "Returns / Refunds", "Return rejected, refund delayed, pickup not done", 9, 5),
        _t("pricing_offers", "Pricing / Offers", "Misleading offers, price changes, hidden charges", 6, 4),
        _t("customer_support", "Customer Support", "No resolution, unhelpful agents, chatbot only", 7, 4),
        _t("payment_issue", "Payment Issue", "Payment failed, double charged, COD problems", 9, 7),
        _t("app_bug", "App Bug / Performance", "Crashes, slow app, search not working", 6, 6),
        _t("account_security", "Account / Security", "Account blocked, hacked, unauthorized orders", 8, 6),
    ],
    "Ride-hailing & Travel": [
        _t("driver_behaviour", "Driver Behaviour", "Rude drivers, cancellations by driver, asking extra cash, unsafe driving", 8, 6),
        _t("pricing_surge", "Pricing / Surge", "High fares, surge pricing, fare different from estimate", 7, 5),
        _t("availability_wait", "Availability / Wait Time", "No cabs available, long wait, booking failures", 8, 8),
        _t("safety", "Safety", "Safety incidents, harassment, SOS not working", 10, 7),
        _t("refund_cancellation", "Refund / Cancellation Fee", "Unfair cancellation fees, refunds not processed", 7, 4),
        _t("booking_bug", "Booking / App Bug", "App crashes, GPS/location issues, payment failures", 7, 6),
        _t("customer_support", "Customer Support", "No response, unresolved complaints", 6, 4),
    ],
    "Social / Messaging": [
        _t("bugs_crashes", "Bugs / Crashes", "App crashes, messages not sending, media not loading", 8, 6),
        _t("account_ban", "Account Ban / Login", "Account banned, suspended, can't log in, OTP issues", 9, 5),
        _t("content_moderation", "Content / Moderation", "Harmful content, unfair moderation, spam or bots", 7, 7),
        _t("privacy", "Privacy", "Data collection, privacy settings, unwanted visibility", 8, 6),
        _t("ads", "Ads", "Too many or irrelevant ads", 5, 3),
        _t("feed_algorithm", "Feed / Algorithm", "Bad recommendations, reach dropped, feed changes", 6, 7),
        _t("ui_changes", "UI Changes", "Dislike of new design or removed features", 5, 3),
    ],
    "Streaming / Media": [
        _t("playback_issue", "Playback / Buffering", "Buffering, playback errors, video quality, audio sync", 8, 6),
        _t("content_library", "Content Library", "Missing shows/songs, content removed, poor catalogue", 7, 8),
        _t("subscription_billing", "Subscription / Billing", "Pricing, auto-renew, charged after cancel, plan changes", 8, 4),
        _t("ads", "Ads", "Too many ads, ads on paid plan", 6, 3),
        _t("downloads_offline", "Downloads / Offline", "Download failures, offline mode problems", 5, 5),
        _t("ux_navigation", "UX / Navigation", "Hard to find content, bad search, confusing UI", 5, 3),
        _t("account_login", "Account / Login", "Login issues, device limits, profile problems", 6, 4),
    ],
    "EdTech / Learning": [
        _t("content_quality", "Content Quality", "Poor or outdated lessons, wrong answers, shallow content", 7, 6),
        _t("bugs_crashes", "Bugs / Crashes", "Videos not loading, progress lost, app crashes", 7, 6),
        _t("pricing_subscription", "Pricing / Subscription", "Expensive, paywalled, refund issues", 7, 4),
        _t("pushy_sales", "Pushy Sales", "Aggressive sales calls, misleading marketing", 6, 3),
        _t("learning_experience", "Learning Experience", "Gamification, streaks, difficulty, motivation issues", 5, 5),
        _t("customer_support", "Customer Support", "No response, unresolved issues", 6, 4),
    ],
    "Gaming": [
        _t("bugs_crashes", "Bugs / Crashes", "Crashes, freezes, glitches, lost progress", 8, 6),
        _t("pay_to_win", "Pay-to-win / Monetisation", "Too expensive, pay-to-win, lootboxes", 7, 5),
        _t("ads", "Ads", "Too many or forced ads", 6, 2),
        _t("balance_difficulty", "Balance / Difficulty", "Unfair matchmaking, too hard, nerfs", 5, 5),
        _t("performance_lag", "Performance / Lag", "Lag, FPS drops, server issues, high ping", 7, 7),
        _t("cheaters", "Cheaters / Hackers", "Cheating, hackers, bots in matches", 7, 7),
        _t("content_updates", "Content / Updates", "Not enough content, bad updates", 4, 6),
    ],
    "Health & Fitness": [
        _t("tracking_accuracy", "Tracking Accuracy", "Wrong steps, sleep, heart-rate or calorie data", 7, 7),
        _t("sync_device", "Sync / Device", "Wearable sync failing, integration with other apps broken", 8, 6),
        _t("subscription_paywall", "Subscription / Paywall", "Features moved behind paywall, pricing", 7, 4),
        _t("bugs_crashes", "Bugs / Crashes", "Crashes, data loss, features not working", 8, 6),
        _t("content_programs", "Content / Programs", "Workouts, plans or advice quality", 5, 5),
        _t("ux_usability", "UX / Usability", "Confusing UI, too many steps", 5, 3),
    ],
}

CATEGORIES = list(PRESETS.keys())


def finalize(themes: list[dict]) -> list[dict]:
    """Validate/normalise a taxonomy and append the reserved themes."""
    out, seen = [], set()
    for t in themes:
        key = re.sub(r"[^a-z0-9_]+", "_", str(t.get("key", "")).strip().lower()).strip("_")
        if not key or key in seen or key in RESERVED_KEYS:
            continue
        seen.add(key)
        out.append({
            "key": key,
            "label": str(t.get("label") or key.replace("_", " ").title()),
            "description": str(t.get("description") or ""),
            "impact": float(_clamp(t.get("impact", 5))),
            "effort": float(_clamp(t.get("effort", 5))),
        })
    return out + [dict(PRAISE), dict(OTHER)]


def _clamp(v, lo=0, hi=10):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return 5


def get_preset(category: str) -> list[dict]:
    return finalize(PRESETS.get(category, PRESETS["General (any app)"]))


def taxonomy_to_df(taxonomy: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(taxonomy, columns=["key", "label", "description", "impact", "effort"])


def df_to_taxonomy(df: pd.DataFrame) -> list[dict]:
    rows = df.dropna(subset=["key"]).to_dict("records")
    return finalize([r for r in rows if r.get("key") not in RESERVED_KEYS])


def generate_taxonomy_with_llm(reviews: pd.DataFrame, category: str, app_names: list[str],
                               n_sample: int = 120, n_themes: int = 8) -> list[dict]:
    """
    Ask the LLM to propose a domain-specific taxonomy from a sample of real reviews.
    Low-star reviews are over-sampled because they carry most of the issue signal.
    Falls back to the closest preset if the LLM call fails.
    """
    from src.llm_classify import call_llm

    df = reviews.dropna(subset=["review_text"])
    df = df[df["review_text"].astype(str).str.len() > 25]
    if df.empty:
        return get_preset(category)

    neg = df[df["rating"] <= 3]
    pos = df[df["rating"] > 3]
    n_neg = min(len(neg), int(n_sample * 0.75))
    sample = pd.concat([
        neg.sample(n_neg, random_state=7) if n_neg else neg,
        pos.sample(min(len(pos), n_sample - n_neg), random_state=7),
    ])
    lines = "\n".join(f"- [{int(r.rating)}★ {r.app_name}] {str(r.review_text)[:300]}" for r in sample.itertuples())

    prompt = f"""You are a senior product analyst. Below are real app-store reviews for these apps: {", ".join(app_names)} (category: {category}).

Design a taxonomy of {n_themes} mutually-exclusive ISSUE themes that a product team could act on.
Rules:
- Themes must be specific to this product domain and grounded in the reviews below.
- Do NOT include a praise/positive theme or an "other" theme (they are added automatically).
- key: short snake_case. label: 1-4 words. description: one sentence listing typical complaints.
- impact (0-10): business damage if unfixed (money lost / safety / churn = high, cosmetic = low).
- effort (0-10): typical engineering + ops effort to fix (infra/partners = high, copy/UX tweaks = low).

Return ONLY JSON: {{"themes": [{{"key": "...", "label": "...", "description": "...", "impact": 7, "effort": 4}}]}}

Reviews:
{lines}"""

    raw = call_llm(prompt)
    try:
        data = json.loads(_strip_fences(raw))
        themes = data.get("themes") if isinstance(data, dict) else data
        if isinstance(themes, list) and len(themes) >= 3:
            return finalize(themes)
    except Exception:
        pass
    return get_preset(category)


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text
