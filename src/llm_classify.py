import os
import json
import hashlib
import time
from pathlib import Path
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

try:
    import openai
except ImportError:
    pass

try:
    import anthropic
except ImportError:
    pass

load_dotenv()

VALID_THEMES = ["failed_transaction", "kyc_friction", "refund_delay", "customer_support", 
                "app_crash_bug", "ui_confusion", "fraud_security_concern", "praise_no_issue", "other"]
VALID_SEVERITIES = ["low", "medium", "high"]

CACHE_DIR = Path("data/classified")
CACHE_FILE = CACHE_DIR / "cache.json"

def load_prompt_template() -> str:
    """Load the classification prompt from prompts/classify_review.txt"""
    prompt_path = Path("prompts/classify_review.txt")
    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Classify these reviews and return JSON format."

def get_cache() -> dict:
    """Load the classification cache from disk. Key = SHA-256 of review text."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_cache(cache: dict):
    """Persist the cache to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def hash_review(text: str) -> str:
    """SHA-256 hash of review text for cache key."""
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()

def format_review_batch(reviews: list[str]) -> str:
    """Format a batch of reviews as numbered list for the prompt."""
    formatted = []
    for i, review in enumerate(reviews, 1):
        formatted.append(f"Review {i}:\n{review}")
    return "\n\n".join(formatted)

def _parse_retry_delay(error_msg: str) -> float:
    """Extract retry delay from Gemini API error message if available."""
    import re
    match = re.search(r'retry(?:Delay)?["\s:]*["\s]*(\d+(?:\.\d+)?)\s*s', str(error_msg), re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r'Please retry in (\d+(?:\.\d+)?)s', str(error_msg))
    if match:
        return float(match.group(1))
    return 0

def call_llm(prompt: str, provider: str = None, model: str = None) -> str:
    """
    Call the LLM API. Supports OpenAI, Anthropic, and Google Gemini.
    - provider from env LLM_PROVIDER (default: 'gemini')
    - model from env LLM_MODEL (default: 'gemini-3.6-flash')
    - Handle rate limits with proper backoff respecting API retry hints
    - 5 retries with exponential backoff + jitter
    """
    import random
    provider = provider or os.getenv("LLM_PROVIDER", "gemini").lower()
    model = model or os.getenv("LLM_MODEL", "gemini-3.6-flash")
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            if provider == "openai":
                from openai import OpenAI
                client = OpenAI()
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            elif provider == "anthropic":
                from anthropic import Anthropic
                client = Anthropic()
                response = client.messages.create(
                    model=model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            elif provider == "gemini":
                from google import genai
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                    }
                )
                return response.text
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            error_str = str(e)
            is_rate_limit = any(code in error_str for code in ["429", "RESOURCE_EXHAUSTED"])
            is_overloaded = any(code in error_str for code in ["503", "UNAVAILABLE"])
            
            if attempt < max_retries - 1:
                if is_rate_limit:
                    # Respect the API's suggested retry delay
                    api_delay = _parse_retry_delay(error_str)
                    wait = max(api_delay + random.uniform(2, 5), 60)
                    print(f"Attempt {attempt+1} rate limited. Waiting {wait:.0f}s (API suggested {api_delay:.0f}s)...")
                elif is_overloaded:
                    wait = min(15 * (2 ** attempt) + random.uniform(1, 5), 120)
                    print(f"Attempt {attempt+1} service unavailable. Retrying in {wait:.0f}s...")
                else:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    print(f"Attempt {attempt+1} failed: {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"API Error after {max_retries} attempts: {e}")
                return "{}"
    return "{}"

def parse_llm_response(response_text: str) -> list[dict]:
    """
    Parse the LLM JSON response. Handle edge cases:
    - Strip markdown code fences if present
    - Validate each result has required fields
    - Clamp sentiment_score to [-1.0, 1.0]
    - Default invalid themes to 'other'
    - Default invalid severities to 'medium'
    """
    try:
        # Strip markdown code fences if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
            
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        data = json.loads(response_text.strip())
        
        # Unpack depending on how the LLM returned it
        if isinstance(data, dict):
            # Sometimes LLMs wrap the array in a dict key like {"reviews": [...]}
            keys = list(data.keys())
            if len(keys) == 1 and isinstance(data[keys[0]], list):
                results = data[keys[0]]
            else:
                results = [data]
        else:
            results = data
            
        if not isinstance(results, list):
            results = []
            
        parsed_results = []
        for res in results:
            if not isinstance(res, dict):
                continue
            
            theme = res.get("theme", "other")
            if theme not in VALID_THEMES:
                theme = "other"
                
            severity = res.get("severity", "medium")
            if severity not in VALID_SEVERITIES:
                severity = "medium"
                
            try:
                sentiment_score = float(res.get("sentiment_score", 0.0))
                sentiment_score = max(-1.0, min(1.0, sentiment_score))
            except (ValueError, TypeError):
                sentiment_score = 0.0
                
            parsed_results.append({
                "theme": theme,
                "severity": severity,
                "sentiment_score": sentiment_score,
                "one_line_summary": str(res.get("one_line_summary", "No summary provided."))
            })
            
        return parsed_results
    except json.JSONDecodeError:
        print("Failed to parse JSON")
        return []

def classify_batch(reviews: list[str]) -> list[dict]:
    """
    Classify a batch of reviews:
    1. Check cache for each review
    2. Send uncached reviews to LLM
    3. Parse response
    4. Update cache with new results
    5. Return all results (cached + new) in original order
    """
    cache = get_cache()
    
    uncached_indices = []
    uncached_reviews = []
    
    results = [None] * len(reviews)
    
    # 1. Check cache for each review
    for i, review in enumerate(reviews):
        h = hash_review(review)
        if h in cache:
            results[i] = cache[h]
        else:
            uncached_indices.append(i)
            uncached_reviews.append(review)
            
    # 2. Send uncached reviews to LLM
    if uncached_reviews:
        prompt_template = load_prompt_template()
        formatted_reviews = format_review_batch(uncached_reviews)
        prompt = prompt_template.replace("{{REVIEWS}}", formatted_reviews)
        if "{{REVIEWS}}" not in prompt_template:
            prompt = prompt_template + "\n\n" + formatted_reviews
            
        # Ensure we ask for json format specifically for the batch
        prompt += "\n\nReturn ONLY a JSON array of objects with keys: theme, severity, sentiment_score, one_line_summary."
            
        response_text = call_llm(prompt)
        
        # 3. Parse response
        parsed = parse_llm_response(response_text)
        
        # 4. Update cache and results
        for idx_in_batch, original_idx in enumerate(uncached_indices):
            if idx_in_batch < len(parsed):
                result = parsed[idx_in_batch]
                results[original_idx] = result
                h = hash_review(reviews[original_idx])
                cache[h] = result
            else:
                # API didn't return enough results
                results[original_idx] = {
                    "theme": "other", "severity": "medium", 
                    "sentiment_score": 0.0, "one_line_summary": "Failed to parse API response"
                }
                
        save_cache(cache)
        
    return results

def classify_dataframe(df: pd.DataFrame, batch_size: int = 15, progress_callback=None) -> pd.DataFrame:
    """
    Classify all reviews in a DataFrame:
    1. Extract review_text column
    2. Split into batches of batch_size
    3. Classify each batch with classify_batch()
    4. Add columns: theme, severity, sentiment_score, one_line_summary
    5. Use progress_callback(current, total) for Streamlit progress bars
    6. Respect free-tier rate limits (5 RPM = 13s between API calls)
    7. Return the augmented DataFrame
    """
    df = df.copy()
    if 'review_text' not in df.columns:
        return df
        
    reviews = df['review_text'].tolist()
    total_reviews = len(reviews)
    total_batches = (total_reviews + batch_size - 1) // batch_size
    
    themes, severities, sentiments, summaries = [], [], [], []
    api_calls_made = 0
    
    for i in range(0, total_reviews, batch_size):
        batch = reviews[i:i+batch_size]
        
        cache = get_cache()
        cached_count = sum(1 for r in batch if hash_review(r) in cache)
        batch_num = i // batch_size + 1
        print(f"Classifying batch {batch_num}/{total_batches} ({cached_count}/{len(batch)} cached)")
        
        # If entire batch is cached, no API call needed — skip delay
        needs_api = cached_count < len(batch)
        
        results = classify_batch(batch)
        
        for res in results:
            if res:
                themes.append(res.get("theme", "other"))
                severities.append(res.get("severity", "medium"))
                sentiments.append(res.get("sentiment_score", 0.0))
                summaries.append(res.get("one_line_summary", ""))
            else:
                themes.append("other")
                severities.append("medium")
                sentiments.append(0.0)
                summaries.append("")
                
        if progress_callback:
            progress_callback(min(i+batch_size, total_reviews), total_reviews)
            
        # Rate limiting: free tier = 5 RPM, so wait ~13s between API calls
        # Skip delay if the batch was fully cached (no API call made)
        if needs_api:
            api_calls_made += 1
            wait_time = 13  # Safe interval for 5 RPM free tier
            print(f"  Rate limit delay: {wait_time}s (API call #{api_calls_made})")
            time.sleep(wait_time)
            
    df['theme'] = themes[:total_reviews]
    df['severity'] = severities[:total_reviews]
    df['sentiment_score'] = sentiments[:total_reviews]
    df['one_line_summary'] = summaries[:total_reviews]
    
    return df
