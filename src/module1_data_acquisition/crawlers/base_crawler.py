import logging
import time
import json
import os
import requests
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class BaseCrawler(ABC):
    def __init__(self, base_url, language, source_name, delay=1.0):
        self.base_url = base_url
        self.language = language
        self.source_name = source_name
        self.delay = delay
        self.data_dir = os.path.join("data", "raw", language, source_name)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(source_name)

    def setup_logging(self):
        log_file = os.path.join("logs", "crawler.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            encoding='utf-8' #utf-8 for Bangla logs
        )

    def fetch_page(self, url, retries=3):
        time.sleep(self.delay) # Respectful crawling
        for i in range(retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=15)
                
                # Fail fast on 403/401/404 - no point retrying usually
                if response.status_code in [401, 403, 404]:
                    self.logger.warning(f"Failed to fetch {url}: Status {response.status_code}. No retry.")
                    return None
                    
                response.raise_for_status()
                # Handle encoding
                if response.encoding is None or response.encoding == 'ISO-8859-1':
                     response.encoding = response.apparent_encoding
                
                return BeautifulSoup(response.text, 'lxml')
            except Exception as e:
                self.logger.error(f"Error fetching {url}: {e}. Retry {i+1}/{retries}")
                time.sleep(2 * (i + 1)) # Exponential backoff
        return None

    def save_article(self, article_data):
        """
        Saves article data to a JSON file.
        Required keys in article_data: url, title, body, date, language
        """
        if not article_data or 'url' not in article_data:
            self.logger.warning("Attempted to save empty article or missing URL")
            return False
            
        # Create a safe filename from the URL
        safe_filename = article_data['url'].split('/')[-1]
        # Fallback if split gives empty (e.g. trailing slash)
        if not safe_filename or len(safe_filename) < 5:
             safe_filename = str(hash(article_data['url']))
        
        # Sanitize filename
        safe_filename = "".join([c for c in safe_filename if c.isalnum() or c in ('-','_')]).strip()
        if not safe_filename:
             safe_filename = str(int(time.time()))
             
        filepath = os.path.join(self.data_dir, f"{safe_filename}.json")
        
        # Add timestamp if not present
        if 'crawled_at' not in article_data:
            article_data['crawled_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(article_data, f, ensure_ascii=False, indent=4)
            self.logger.info(f"Saved article: {article_data.get('title', 'No Title')} ({safe_filename})")
            return True
        except Exception as e:
            self.logger.error(f"Error saving file {filepath}: {e}")
            return False

    @abstractmethod
    def crawl(self, limit=100):
        """
        Main method to start crawling.
        Should call self.fetch_page() and self.save_article().
        """
        pass
