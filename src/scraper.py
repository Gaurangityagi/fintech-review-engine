import pandas as pd
import time
import random
import logging
import os
from google_play_scraper import Sort, reviews

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def scrape_app_reviews(app_id: str, app_name: str, target_count: int = 750, lang: str = 'en', country: str = 'in') -> pd.DataFrame:
    """
    Scrape reviews from Google Play Store.
    - Use google_play_scraper.reviews and reviews_all
    - Use Sort.NEWEST to get recent reviews
    - Target 500-1000 reviews per app
    - Handle rate limiting with exponential backoff
    - Return DataFrame with columns: app_name, review_id, rating, review_text, date, thumbs_up_count
    """
    logging.info(f"Starting to scrape {app_name} ({app_id})")
    
    all_reviews = []
    continuation_token = None
    scraped_count = 0
    
    while scraped_count < target_count:
        try:
            result, continuation_token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=min(199, target_count - scraped_count), # API limit is often 199/200 per page
                continuation_token=continuation_token
            )
            
            if not result:
                logging.info(f"No more reviews found for {app_name}.")
                break
                
            all_reviews.extend(result)
            scraped_count += len(result)
            logging.info(f"Scraped {scraped_count}/{target_count} reviews for {app_name}")
            
            if continuation_token is None:
                logging.info(f"No continuation token returned for {app_name}, stopping.")
                break
                
            # Random delay
            delay = random.uniform(2.0, 5.0)
            logging.info(f"Sleeping for {delay:.2f}s...")
            time.sleep(delay)
            
        except Exception as e:
            logging.error(f"Error scraping {app_name}: {e}")
            logging.info("Attempting to backoff...")
            time.sleep(10) # Simple backoff
            break

    df = pd.DataFrame(all_reviews)
    
    # Check if df is empty before processing
    if df.empty:
        logging.warning(f"No reviews scraped for {app_name}.")
        return pd.DataFrame(columns=['app_name', 'review_id', 'rating', 'review_text', 'date', 'thumbs_up_count'])
        
    df['app_name'] = app_name
    
    # Map to expected columns if present, otherwise set to NA
    cols_to_extract = {
        'reviewId': 'review_id', 
        'score': 'rating', 
        'content': 'review_text', 
        'at': 'date', 
        'thumbsUpCount': 'thumbs_up_count'
    }
    
    # Rename columns that exist
    df = df.rename(columns={k: v for k, v in cols_to_extract.items() if k in df.columns})
    
    # Select only required columns, adding missing ones as NA
    expected_cols = ['app_name', 'review_id', 'rating', 'review_text', 'date', 'thumbs_up_count']
    
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA
            
    df = df[expected_cols]
    
    return df

def scrape_all_apps(target_count: int = 750) -> dict:
    """
    Scrape reviews for all 3 apps:
    - PhonePe: com.phonepe.app
    - Paytm: net.one97.paytm
    - Google Pay: com.google.android.apps.nbu.paisa.user
    Save each to data/raw/{app_name}_reviews.csv
    If scraping fails or is rate-limited, log the error and note the limitation.
    Return dict of {app_name: DataFrame}
    """
    apps = {
        'PhonePe': 'com.phonepe.app',
        'Paytm': 'net.one97.paytm',
        'Google Pay': 'com.google.android.apps.nbu.paisa.user'
    }
    
    results = {}
    
    os.makedirs('data/raw', exist_ok=True)
    
    for app_name, app_id in apps.items():
        logging.info(f"Processing app: {app_name}")
        df = scrape_app_reviews(app_id, app_name, target_count)
        
        output_path = f"data/raw/{app_name.lower().replace(' ', '_')}_reviews.csv"
        
        if not df.empty:
            df.to_csv(output_path, index=False)
            logging.info(f"Saved {len(df)} reviews to {output_path}")
        else:
            logging.warning(f"No data to save for {app_name}")
            
        results[app_name] = df
        
        # Delay between apps
        if app_name != list(apps.keys())[-1]:
            time.sleep(5)
            
    return results

def load_raw_reviews() -> pd.DataFrame:
    """Load all raw CSVs from data/raw/ and combine into one DataFrame."""
    import glob
    csv_files = glob.glob('data/raw/*_reviews.csv')
    
    if not csv_files:
        logging.warning("No CSV files found in data/raw/")
        return pd.DataFrame()
        
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            logging.error(f"Error reading {f}: {e}")
            
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

if __name__ == "__main__":
    pass
