"""
Utility functions for loading and processing articles
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def load_all_articles(data_dir: str = "data/raw") -> List[Dict]:
    """
    Load all JSON articles from the data directory.
    
    Args:
        data_dir: Root directory containing raw data
        
    Returns:
        List of article dictionaries
    """
    articles = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logging.error(f"Data directory not found: {data_dir}")
        return articles
    
    # Walk through all JSON files in data/raw/bangla and data/raw/english
    json_files = list(data_path.rglob("*.json"))
    logging.info(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                article = json.load(f)
                
                # Add filepath for reference
                article['filepath'] = str(json_file)
                
                # Basic validation
                if 'body' in article and article['body']:
                    articles.append(article)
                else:
                    logging.warning(f"Skipping article with empty body: {json_file}")
                    
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON {json_file}: {e}")
        except Exception as e:
            logging.error(f"Error loading {json_file}: {e}")
    
    logging.info(f"Successfully loaded {len(articles)} articles")
    return articles


def count_tokens(text: str) -> int:
    """
    Count tokens (words) in text.
    Simple whitespace-based tokenization.
    
    Args:
        text: Input text
        
    Returns:
        Number of tokens
    """
    if not text:
        return 0
    
    # Split by whitespace and filter empty strings
    tokens = [t for t in text.split() if t.strip()]
    return len(tokens)


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    return text.strip()


def filter_articles(articles: List[Dict], min_tokens: int = 50) -> List[Dict]:
    """
    Filter articles based on criteria.
    
    Args:
        articles: List of articles
        min_tokens: Minimum token count threshold
        
    Returns:
        Filtered list of articles
    """
    filtered = []
    
    for article in articles:
        # Skip if body is too short
        if 'token_count' in article and article['token_count'] < min_tokens:
            logging.debug(f"Skipping short article: {article.get('url', 'unknown')}")
            continue
        
        # Skip if no title or body
        if not article.get('title') or not article.get('body'):
            logging.debug(f"Skipping article with missing fields: {article.get('url', 'unknown')}")
            continue
            
        filtered.append(article)
    
    logging.info(f"Filtered from {len(articles)} to {len(filtered)} articles")
    return filtered


def deduplicate_articles(articles: List[Dict]) -> List[Dict]:
    """
    Remove duplicate articles based on URL.
    
    Args:
        articles: List of articles
        
    Returns:
        Deduplicated list of articles
    """
    seen_urls = set()
    unique_articles = []
    
    for article in articles:
        url = article.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)
    
    duplicates_removed = len(articles) - len(unique_articles)
    if duplicates_removed > 0:
        logging.info(f"Removed {duplicates_removed} duplicate articles")
    
    return unique_articles
