"""
Store scrapers for Google Play and the Apple App Store.

- search_apps(): find any app by name on either store
- get_app_details(): store metadata snapshot (rating, installs, histogram, version...)
- scrape_reviews(): newest reviews in one normalised schema

Normalised review columns:
    app_name, platform, review_id, rating, review_text, date,
    thumbs_up_count, app_version, reply_text, replied_at
"""
import glob
import logging
import random
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from google_play_scraper import Sort, reviews as gp_reviews, app as gp_app, search as gp_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REVIEW_COLUMNS = ["app_name", "platform", "review_id", "rating", "review_text", "date",
                  "thumbs_up_count", "app_version", "reply_text", "replied_at"]
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (review-intelligence-engine)"}
APPSTORE_MAX_PAGES = 10  # Apple's public RSS feed caps at 10 pages x 50 reviews


# ═══════════════════════════════════════════════════════════════
#  SEARCH
# ═══════════════════════════════════════════════════════════════
def search_apps(query: str, platform: str = "play", country: str = "in", lang: str = "en", n: int = 6) -> list[dict]:
    """Search a store. Returns list of {platform, app_id, title, developer, icon, score, ratings, genre}."""
    query = (query or "").strip()
    if not query:
        return []
    try:
        if platform == "appstore":
            return _search_appstore(query, country, n)
        return _search_play(query, country, lang, n)
    except Exception as e:
        logging.error(f"Search failed ({platform}, {query}): {e}")
        return []


def _search_play(query, country, lang, n):
    # Accept a pasted Play Store URL or package id directly
    m = re.search(r"id=([\w\.]+)", query)
    if m or re.fullmatch(r"[a-zA-Z][\w]*(\.[\w]+)+", query):
        ids = [m.group(1) if m else query]
    else:
        # google-play-scraper's search() returns appId=None for the featured top hit,
        # so read package ids straight from the search page and enrich with app().
        html = requests.get("https://play.google.com/store/search",
                            params={"q": query, "c": "apps", "hl": lang, "gl": country},
                            headers=HTTP_HEADERS, timeout=15).text
        ids = list(dict.fromkeys(re.findall(r"/store/apps/details\?id=([\w\.]+)", html)))[:n]
        if not ids:
            ids = [r["appId"] for r in gp_search(query, lang=lang, country=country, n_hits=n) if r.get("appId")]

    out = []
    for app_id in ids[:n]:
        try:
            d = gp_app(app_id, lang=lang, country=country)
            out.append({
                "platform": "play", "app_id": app_id, "title": d.get("title"),
                "developer": d.get("developer"), "icon": d.get("icon"), "score": d.get("score"),
                "ratings": d.get("ratings"), "genre": d.get("genre"), "installs": d.get("installs"),
            })
        except Exception as e:
            logging.warning(f"Could not fetch details for {app_id}: {e}")
    return out


def _search_appstore(query, country, n):
    m = re.search(r"id(\d{6,})", query)
    if m or query.isdigit():
        params = {"id": m.group(1) if m else query, "country": country}
        url = "https://itunes.apple.com/lookup"
    else:
        params = {"term": query, "entity": "software", "country": country, "limit": n}
        url = "https://itunes.apple.com/search"
    results = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=15).json().get("results", [])
    return [{
        "platform": "appstore", "app_id": str(r["trackId"]), "title": r.get("trackName"),
        "developer": r.get("sellerName"), "icon": r.get("artworkUrl100"),
        "score": r.get("averageUserRating"), "ratings": r.get("userRatingCount"),
        "genre": r.get("primaryGenreName"), "installs": None,
    } for r in results[:n] if r.get("trackId")]


# ═══════════════════════════════════════════════════════════════
#  APP METADATA
# ═══════════════════════════════════════════════════════════════
def get_app_details(platform: str, app_id: str, country: str = "in", lang: str = "en") -> dict:
    """Store-level snapshot for the Market Snapshot page."""
    try:
        if platform == "appstore":
            r = requests.get("https://itunes.apple.com/lookup", params={"id": app_id, "country": country},
                             headers=HTTP_HEADERS, timeout=15).json().get("results", [])
            if not r:
                return {}
            r = r[0]
            return {
                "title": r.get("trackName"), "developer": r.get("sellerName"), "icon": r.get("artworkUrl100"),
                "score": r.get("averageUserRating"), "ratings": r.get("userRatingCount"),
                "installs": None, "histogram": None, "genre": r.get("primaryGenreName"),
                "version": r.get("version"), "updated": r.get("currentVersionReleaseDate"),
                "released": r.get("releaseDate"), "price": r.get("price"),
                "size_mb": round(int(r.get("fileSizeBytes", 0)) / 1e6, 1) if r.get("fileSizeBytes") else None,
                "content_rating": r.get("contentAdvisoryRating"),
            }
        d = gp_app(app_id, lang=lang, country=country)
        updated = d.get("updated")
        return {
            "title": d.get("title"), "developer": d.get("developer"), "icon": d.get("icon"),
            "score": d.get("score"), "ratings": d.get("ratings"), "reviews": d.get("reviews"),
            "installs": d.get("installs"), "real_installs": d.get("realInstalls"),
            "histogram": d.get("histogram"), "genre": d.get("genre"), "version": d.get("version"),
            "updated": datetime.fromtimestamp(updated, tz=timezone.utc).isoformat() if updated else None,
            "released": d.get("released"), "price": d.get("price"), "free": d.get("free"),
            "contains_ads": d.get("containsAds"), "offers_iap": d.get("offersIAP"),
            "content_rating": d.get("contentRating"),
        }
    except Exception as e:
        logging.error(f"Metadata fetch failed for {platform}:{app_id}: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════
#  REVIEWS
# ═══════════════════════════════════════════════════════════════
def scrape_reviews(platform: str, app_id: str, app_name: str, target_count: int = 500,
                   country: str = "in", lang: str = "en", sort: str = "newest",
                   progress_callback=None) -> pd.DataFrame:
    """sort: 'newest' (recent window, best for monitoring) or 'relevant' (longer history, biased to helpful reviews)."""
    if platform == "appstore":
        df = _scrape_appstore(app_id, app_name, target_count, country, sort, progress_callback)
    else:
        df = _scrape_play(app_id, app_name, target_count, country, lang, sort, progress_callback)
    if df.empty:
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    df["app_name"] = app_name
    df["platform"] = platform
    for c in REVIEW_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[REVIEW_COLUMNS]


def _scrape_play(app_id, app_name, target_count, country, lang, sort, progress_callback):
    logging.info(f"Scraping Play Store: {app_name} ({app_id})")
    rows, token, retries = [], None, 0
    while len(rows) < target_count:
        try:
            result, token = gp_reviews(app_id, lang=lang, country=country, sort=Sort.MOST_RELEVANT if sort == "relevant" else Sort.NEWEST,
                                       count=min(199, target_count - len(rows)), continuation_token=token)
        except Exception as e:
            retries += 1
            logging.error(f"Error scraping {app_name}: {e} (retry {retries}/3)")
            if retries >= 3:
                break
            time.sleep(5 * 2 ** retries)
            continue
        if not result:
            break
        rows.extend(result)
        if progress_callback:
            progress_callback(min(len(rows), target_count), target_count)
        if token is None:
            break
        time.sleep(random.uniform(1.0, 2.5))

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={
        "reviewId": "review_id", "score": "rating", "content": "review_text", "at": "date",
        "thumbsUpCount": "thumbs_up_count", "replyContent": "reply_text", "repliedAt": "replied_at",
    })
    df["app_version"] = df.get("reviewCreatedVersion", df.get("appVersion"))
    return df


def _scrape_appstore(app_id, app_name, target_count, country, sort, progress_callback):
    order = "mosthelpful" if sort == "relevant" else "mostrecent"
    logging.info(f"Scraping App Store: {app_name} ({app_id})")
    rows = []
    for page in range(1, APPSTORE_MAX_PAGES + 1):
        if len(rows) >= target_count:
            break
        url = f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby={order}/json"
        try:
            entries = requests.get(url, headers=HTTP_HEADERS, timeout=15).json().get("feed", {}).get("entry", [])
        except Exception as e:
            logging.error(f"App Store page {page} failed for {app_name}: {e}")
            break
        if isinstance(entries, dict):
            entries = [entries]
        entries = [e for e in entries if "im:rating" in e]  # first entry can be app info
        if not entries:
            break
        for e in entries:
            title = e.get("title", {}).get("label", "")
            body = e.get("content", {}).get("label", "")
            rows.append({
                "review_id": e.get("id", {}).get("label"),
                "rating": int(e["im:rating"]["label"]),
                "review_text": f"{title}. {body}".strip(". ") if title else body,
                "date": pd.to_datetime(e.get("updated", {}).get("label"), utc=True).tz_convert(None),
                "thumbs_up_count": int(e.get("im:voteSum", {}).get("label", 0) or 0),
                "app_version": e.get("im:version", {}).get("label"),
                "reply_text": None, "replied_at": None,  # not exposed by Apple's public feed
            })
        if progress_callback:
            progress_callback(min(len(rows), target_count), target_count)
        time.sleep(random.uniform(0.5, 1.5))
    return pd.DataFrame(rows[:target_count])


# ═══════════════════════════════════════════════════════════════
#  WORKSPACE-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════
def scrape_workspace(ws, target_count: int = 500, sort: str = "newest", merge: bool = True,
                     status_callback=None, progress_callback=None) -> pd.DataFrame:
    """
    Scrape metadata + reviews for every app in a workspace and save to its raw dir.
    merge=True appends to previously scraped reviews (deduped by review_id), so running
    the pipeline weekly builds up a longitudinal dataset for trend analysis.
    """
    ws.raw_dir.mkdir(parents=True, exist_ok=True)
    meta = ws.load_meta()
    for i, a in enumerate(ws.apps):
        if status_callback:
            status_callback(f"Scraping {a.name} ({'App Store' if a.platform == 'appstore' else 'Google Play'}) "
                            f"— app {i + 1}/{len(ws.apps)}")
        details = get_app_details(a.platform, a.app_id, ws.country, ws.lang)
        if details:
            details["fetched_at"] = datetime.now().isoformat(timespec="seconds")
            meta[a.key] = {**details, "name": a.name, "platform": a.platform, "app_id": a.app_id}
        df = scrape_reviews(a.platform, a.app_id, a.name, target_count, ws.country, ws.lang, sort, progress_callback)
        path = ws.raw_dir / f"{a.key}.csv"
        if merge and path.exists() and not df.empty:
            old = pd.read_csv(path)
            df = pd.concat([old, df], ignore_index=True).drop_duplicates("review_id", keep="last")
        if not df.empty:
            df.to_csv(path, index=False)
    ws.save_meta(meta)
    return load_workspace_raw(ws)


def load_workspace_raw(ws) -> pd.DataFrame:
    """Load raw reviews for the apps currently in the workspace (ignores removed apps)."""
    frames = []
    for a in ws.apps:
        p = ws.raw_dir / f"{a.key}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["app_name"] = a.name  # honour renames
            if "platform" not in df.columns:
                df["platform"] = a.platform
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    for c in REVIEW_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df


# ── Legacy API (kept for backwards compatibility with the original README commands) ──
def load_raw_reviews() -> pd.DataFrame:
    """Load all raw CSVs from data/raw/ and combine into one DataFrame."""
    files = glob.glob("data/raw/*_reviews.csv")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
