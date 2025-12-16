from .base_crawler import BaseCrawler
import logging
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ..utils import clean_text, parse_date

class SeleniumCrawler(BaseCrawler):
    def __init__(self, base_url, language, source_name, selectors, start_urls=None, delay=2.0, pagination_param=None):
        super().__init__(base_url=base_url, language=language, source_name=source_name, delay=delay)
        self.selectors = selectors
        self.start_urls = start_urls if start_urls else [base_url]
        self.pagination_param = pagination_param
        self.driver = None
        
    def _setup_driver(self):
        if self.driver:
            return
            
        chrome_options = Options()
        chrome_options.add_argument("--headless") 
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # Suppress logging
        chrome_options.add_argument("--log-level=3")
        
        # Anti-detection measures
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(options=chrome_options)
        
        # Execute CDP commands to prevent detection
        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            """
        })
        self.logger.info("Selenium driver initialized")

    def _teardown_driver(self):
        if self.driver:
            self.driver.quit()
            self.driver = None
            self.logger.info("Selenium driver closed")

    def fetch_page_selenium(self, url):
        self._setup_driver()
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(self.delay)
            
            return BeautifulSoup(self.driver.page_source, 'lxml')
        except Exception as e:
            self.logger.error(f"Selenium error fetching {url}: {e}")
            return None

    def crawl(self, limit=100):
        self.logger.info(f"Starting crawl for {self.source_name} with Selenium...")
        count = 0
        visited_urls = set()
        
        start_urls = self.start_urls
        self._setup_driver()
        
        try:
            for url in start_urls:
                if count >= limit:
                    break
                    
                # Pagination loop
                page_num = 1
                max_pages = 50
                
                while page_num <= max_pages:
                    if count >= limit:
                        break
                        
                    current_url = url
                    if hasattr(self, 'pagination_param') and self.pagination_param and page_num > 1:
                         sep = '&' if '?' in url else '?'
                         current_url = f"{url}{sep}{self.pagination_param}={page_num}"
                    
                    self.logger.info(f"Processing list page: {current_url}")
                    soup = self.fetch_page_selenium(current_url)
                    if not soup:
                        if page_num > 1: 
                            break
                        continue
                        
                    article_links = self.extract_links(soup, current_url)
                    
                    if not article_links:
                        self.logger.info(f"No links found on {current_url}. Ending pagination.")
                        break
                        
                    self.logger.info(f"Found {len(article_links)} links on {current_url}")
                    
                    links_processed_on_page = 0
                    for link in article_links:
                        if count >= limit:
                            break
                        if link in visited_urls:
                            continue
                            
                        visited_urls.add(link)
                        print(f"[{self.source_name}] Fetching ({count+1}/{limit}): {link}") 
                        article_data = self.parse_article(link)
                        if article_data:
                            if self.save_article(article_data):
                                count += 1
                                links_processed_on_page += 1
                                if count % 10 == 0:
                                    self.logger.info(f"Progress: {count}/{limit}")
                                    
                    if links_processed_on_page == 0 and page_num > 1:
                         self.logger.info("No new links processed on this page. Stopping pagination.")
                         break

                    if not getattr(self, 'pagination_param', None):
                        break
                        
                    page_num += 1
        finally:
            self._teardown_driver()
                        
        self.logger.info(f"Finished crawling {self.source_name}. Total articles: {count}")

    def extract_links(self, soup, base_url):
        links = set()
        selector = self.selectors.get('article_links', 'a')
        
        for a in soup.select(selector):
            href = a.get('href')
            if not href:
                continue
            
            full_url = urljoin(base_url, href)
            
            if self.base_url.replace('https://', '').replace('http://', '').split('/')[0] not in full_url:
                continue

            # Heuristic for generic 'a' selector
            if selector == 'a' and len(full_url) < len(base_url) + 15:
                continue
                
            links.add(full_url)
                
        return list(links)

    def parse_article(self, url):
        soup = self.fetch_page_selenium(url)
        if not soup:
            return None
            
        data = {
            'url': url,
            'language': self.language,
            'source': self.source_name
        }
        
        # Title
        title_selector = self.selectors.get('title', 'h1')
        title_tag = soup.select_one(title_selector)
        if title_tag:
            data['title'] = clean_text(title_tag.text)
        else:
            self.logger.warning(f"No title found for {url}")
            return None

        # Body
        body_selector = self.selectors.get('body', 'article')
        body_tags = soup.select(body_selector)
        if body_tags:
            body_text = " ".join([clean_text(tag.text) for tag in body_tags])
            data['body'] = body_text
        else:
             p_tags = soup.find_all('p')
             body_text = " ".join([clean_text(p.text) for p in p_tags])
             data['body'] = body_text

        if not data.get('body') or len(data['body']) < 100: # Ensure valid body
            self.logger.warning(f"No valid body found for {url}")
            return None

        # Date
        date_selector = self.selectors.get('date', 'time')
        date_tag = soup.select_one(date_selector)
        if date_tag:
            date_attr = self.selectors.get('date_attr')
            if date_attr and date_tag.has_attr(date_attr):
                raw_date = date_tag[date_attr]
            else:
                raw_date = date_tag.text
            data['date'] = parse_date(raw_date)
        else:
             data['date'] = None

        return data
