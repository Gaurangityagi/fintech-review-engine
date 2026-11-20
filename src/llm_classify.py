"""
Batched, cached, taxonomy-aware LLM classification of reviews.

Each review gets: theme, severity, sentiment_score, one_line_summary, feature_request.
The cache is per workspace and keyed by SHA-256 of the review text. It also stores a
fingerprint of the taxonomy, so editing the taxonomy automatically invalidates it.
"""
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

VALID_SEVERITIES = ["low", "medium", "high"]
TAXONOMY_KEY = "__taxonomy__"
RESULT_FIELDS = ["theme", "severity", "sentiment_score", "one_line_summary", "feature_request"]
PROMPT_PATH = Path("prompts/classify_review.txt")


# ═══════════════════════════════════════════════════════════════
#  CACHE
# ═══════════════════════════════════════════════════════════════
def taxonomy_fingerprint(taxonomy: list[dict]) -> str:
    """Only keys + descriptions change classification output; impact/effort edits keep the cache."""
    sig = json.dumps([[t["key"], t.get("description", "")] for t in taxonomy], sort_keys=True)
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]


def hash_review(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def load_cache(path: Path, taxonomy: list[dict]) -> dict:
    fp = taxonomy_fingerprint(taxonomy)
    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
            if cache.get(TAXONOMY_KEY) == fp:
                return cache
        except json.JSONDecodeError:
            pass
    return {TAXONOMY_KEY: fp}


def save_cache(path: Path, cache: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ═══════════════════════════════════════════════════════════════
#  LLM CALL
# ═══════════════════════════════════════════════════════════════
def _parse_retry_delay(error_msg: str) -> float:
    m = re.search(r'retry(?:Delay)?["\s:]*["\s]*(\d+(?:\.\d+)?)\s*s', str(error_msg), re.IGNORECASE) \
        or re.search(r"Please (?:retry|try again) in (\d+(?:\.\d+)?)s", str(error_msg))
    return float(m.group(1)) if m else 0


def call_llm(prompt: str, provider: str = None, model: str = None, json_mode: bool = True) -> str:
    """
    Call the configured LLM (env LLM_PROVIDER / LLM_MODEL). Supports groq, openai,
    anthropic, gemini. Retries 5x with backoff that respects API retry hints.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
    model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    fmt = {"response_format": {"type": "json_object"}} if json_mode else {}

    max_retries = 5
    for attempt in range(max_retries):
        try:
            if provider == "groq":
                from groq import Groq
                client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], **fmt)
                return r.choices[0].message.content
            if provider == "openai":
                from openai import OpenAI
                r = OpenAI().chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], **fmt)
                return r.choices[0].message.content
            if provider == "anthropic":
                from anthropic import Anthropic
                r = Anthropic().messages.create(model=model, max_tokens=8192,
                                                messages=[{"role": "user", "content": prompt}])
                return r.content[0].text
            if provider == "gemini":
                from google import genai
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                cfg = {"response_mime_type": "application/json"} if json_mode else {}
                return client.models.generate_content(model=model, contents=prompt, config=cfg).text
            raise ValueError(f"Unknown provider: {provider}")
        except ValueError:
            raise
        except Exception as e:
            err = str(e)
            if attempt == max_retries - 1:
                print(f"API Error after {max_retries} attempts: {e}")
                return "{}"
            if any(c in err for c in ["429", "RESOURCE_EXHAUSTED", "rate_limit"]):
                wait = max(_parse_retry_delay(err) + random.uniform(2, 5), 20)
            elif any(c in err for c in ["503", "UNAVAILABLE", "overloaded", "529"]):
                wait = min(15 * 2 ** attempt + random.uniform(1, 5), 120)
            else:
                wait = 2 ** attempt + random.uniform(0, 1)
            print(f"LLM attempt {attempt + 1} failed ({err[:120]}). Retrying in {wait:.0f}s...")
            time.sleep(wait)
    return "{}"


# ═══════════════════════════════════════════════════════════════
#  PROMPT + PARSING
# ═══════════════════════════════════════════════════════════════
def build_prompt(reviews: list[str], taxonomy: list[dict], context: str) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else "{{THEMES}}\n\n{{REVIEWS}}"
    themes = "\n".join(f"- {t['key']}: {t.get('description', '')}" for t in taxonomy)
    body = "\n\n".join(f"Review {i}:\n{str(r)[:1500]}" for i, r in enumerate(reviews, 1))
    return template.replace("{{CONTEXT}}", context).replace("{{THEMES}}", themes).replace("{{REVIEWS}}", body)


def _clean_result(res: dict, valid_themes: set) -> dict:
    theme = str(res.get("theme", "other")).strip()
    severity = str(res.get("severity", "medium")).strip().lower()
    try:
        sentiment = max(-1.0, min(1.0, float(res.get("sentiment_score", 0.0))))
    except (TypeError, ValueError):
        sentiment = 0.0
    fr = res.get("feature_request")
    fr = str(fr).strip() if fr and str(fr).strip().lower() not in {"null", "none", "n/a", ""} else None
    return {
        "theme": theme if theme in valid_themes else "other",
        "severity": severity if severity in VALID_SEVERITIES else "medium",
        "sentiment_score": sentiment,
        "one_line_summary": str(res.get("one_line_summary") or ""),
        "feature_request": fr,
    }


def parse_llm_response(text: str, n_expected: int, valid_themes: set) -> list[dict | None]:
    """Return a list aligned to the batch (index i = review i+1). Missing items are None."""
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip()).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [None] * n_expected

    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        items = lists[0] if lists else [data]
    elif isinstance(data, list):
        items = data
    else:
        items = []

    out: list[dict | None] = [None] * n_expected
    for pos, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("review_number", pos + 1)) - 1
        except (TypeError, ValueError):
            idx = pos
        if 0 <= idx < n_expected and out[idx] is None:
            out[idx] = _clean_result(item, valid_themes)
    return out


# ═══════════════════════════════════════════════════════════════
#  DATAFRAME CLASSIFICATION
# ═══════════════════════════════════════════════════════════════
def apply_cache(df: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """Attach cached classifications to df (rows not in cache get NaN)."""
    df = df.copy()
    hits = df["review_text"].map(lambda t: cache.get(hash_review(t)))
    for f in RESULT_FIELDS:
        df[f] = hits.map(lambda h: h.get(f) if isinstance(h, dict) else None)
    return df


def classify_dataframe(df: pd.DataFrame, taxonomy: list[dict], cache_path: Path, context: str = "mobile apps",
                       batch_size: int = 15, progress_callback=None) -> pd.DataFrame:
    """
    Classify every review in df. Only texts not already in the cache hit the LLM;
    identical texts are classified once. Cache is saved after every batch so an
    interrupted run loses nothing.
    """
    if df.empty or "review_text" not in df.columns:
        return df
    df = df.copy()
    df["review_text"] = df["review_text"].fillna("").astype(str)

    cache = load_cache(cache_path, taxonomy)
    valid = {t["key"] for t in taxonomy}
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    delay = 1.5 if provider in ("groq", "openai", "anthropic") else 13

    todo = list(dict.fromkeys(t for t in df["review_text"] if t.strip() and hash_review(t) not in cache))
    total = len(todo)
    for start in range(0, total, batch_size):
        batch = todo[start:start + batch_size]
        parsed = parse_llm_response(call_llm(build_prompt(batch, taxonomy, context)), len(batch), valid)
        for text, res in zip(batch, parsed):
            if res is not None:  # leave failures uncached so the next run retries them
                cache[hash_review(text)] = res
        save_cache(cache_path, cache)
        if progress_callback:
            progress_callback(min(start + batch_size, total), total)
        if start + batch_size < total:
            time.sleep(delay)

    out = apply_cache(df, cache)
    empty = out["review_text"].str.strip() == ""
    out.loc[empty, ["theme", "severity", "sentiment_score"]] = ["other", "low", 0.0]
    return out
