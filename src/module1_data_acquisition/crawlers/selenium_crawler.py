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
    def __init__(self, base_url, language, source_name, selectors, start_urls=None, delay=2.0, pagination_param=None, load_more_selector=None, max_load_more_clicks=50):
        super().__init__(base_url=base_url, language=language, source_name=source_name, delay=delay)
        self.selectors = selectors
        self.start_urls = start_urls if start_urls else [base_url]
        self.pagination_param = pagination_param
        self.load_more_selector = load_more_selector
        self.max_load_more_clicks = max_load_more_clicks
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
        self._setup_driver() #here opens headless browser
        
        try:
            for url in start_urls:
                if count >= limit:
                    break
                    
                self.logger.info(f"Processing list page: {url}")
                
                if self.load_more_selector:
                    self.driver.get(url)
                    
                    # Wait for initial articles to load
                    try:
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.TAG_NAME, "body"))
                        )
                        time.sleep(self.delay)
                    except Exception as e:
                        self.logger.error(f"Failed to load {url}: {e}")
                        continue
                    
                    # Keep clicking "আরও" (More) button until we have enough articles
                    clicks_without_progress = 0
                    max_clicks_without_progress = 3
                    
                    while count < limit:
                        # Get current articles
                        soup = BeautifulSoup(self.driver.page_source, 'lxml')
                        article_links = self.extract_links(soup, url)
                        
                        if not article_links:
                            self.logger.info("No more articles found.")
                            break
                        
                        # Count how many are NEW (not visited)
                        new_links = [link for link in article_links if link not in visited_urls]
                        self.logger.info(f"Found {len(article_links)} total links on page, {len(new_links)} are new")
                        
                        # If no new links after clicking button, we might be done
                        if not new_links:
                            clicks_without_progress += 1
                            if clicks_without_progress >= max_clicks_without_progress:
                                self.logger.info(f"No new articles after {clicks_without_progress} button clicks, ending pagination")
                                break
                        else:
                            clicks_without_progress = 0  # Reset counter when we find new articles
                        
                        # Process each NEW article
                        articles_saved_this_round = 0
                        for link in new_links:
                            if count >= limit:
                                break
                                
                            visited_urls.add(link)
                            print(f"[{self.source_name}] Fetching ({count+1}/{limit}): {link}") 
                            article_data = self.parse_article(link)
                            if article_data:
                                if self.save_article(article_data):
                                    count += 1
                                    articles_saved_this_round += 1
                                    if count % 10 == 0:
                                        self.logger.info(f"Progress: {count}/{limit}")
                        
                        self.logger.info(f"Saved {articles_saved_this_round} articles this round")
                        
                        # If we've reached limit, break
                        if count >= limit:
                            break
                        
                        # Try to click "আরও" (More) button to load more articles
                        try:
                            # Scroll to bottom to trigger lazy loading
                            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                            time.sleep(1.5)
                            
                            # Try multiple methods to find the "আরও" button
                            more_button = None
                            
                            # Method 1: Try CSS selector if provided
                            try:
                                more_button = WebDriverWait(self.driver, 5).until(
                                    EC.presence_of_element_located((By.CSS_SELECTOR, self.load_more_selector))
                                )
                                self.logger.info(f"Found button with selector: {self.load_more_selector}")
                            except Exception as e:
                                self.logger.debug(f"CSS selector failed: {str(e)[:50]}")
                            
                            # Method 2: Find by text content "আরও"
                            if not more_button:
                                try:
                                    more_button = self.driver.find_element(By.XPATH, "//button[contains(., 'আরও')] | //a[contains(., 'আরও')] | //*[contains(@class, 'load') and contains(., 'আরও')]")
                                    self.logger.info("Found button by text 'আরও'")
                                except Exception as e:
                                    self.logger.debug(f"XPath text search failed: {str(e)[:50]}")
                            
                            # Method 3: Find by ID pattern
                            if not more_button:
                                try:
                                    more_button = self.driver.find_element(By.CSS_SELECTOR, "button[id*='ajax_load_more'], a[id*='ajax_load_more'], button[id*='load_more'], a[id*='load_more']")
                                    self.logger.info("Found button by ID pattern")
                                except Exception as e:
                                    self.logger.debug(f"ID pattern search failed: {str(e)[:50]}")
                            
                            # Method 4: Find by class pattern
                            if not more_button:
                                try:
                                    more_button = self.driver.find_element(By.CSS_SELECTOR, "button[class*='load-more'], a[class*='load-more'], button[class*='more'], a[class*='more']")
                                    self.logger.info("Found button by class pattern")
                                except Exception as e:
                                    self.logger.debug(f"Class pattern search failed: {str(e)[:50]}")
                            
                            if not more_button:
                                self.logger.info("আরও button not found after trying all methods, ending pagination for this page")
                                break
                            
                            # Scroll to button and wait (even if not visible, try to make it visible)
                            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", more_button)
                            time.sleep(1.0)
                            
                            # Try clicking - use JavaScript which works even on hidden elements
                            try:
                                # First try regular click if visible
                                if more_button.is_displayed():
                                    try:
                                        more_button.click()
                                        self.logger.info("Clicked 'আরও' button with regular click")
                                    except Exception as click_error:
                                        self.logger.debug(f"Regular click failed: {str(click_error)[:50]}, trying JS click")
                                        self.driver.execute_script("arguments[0].click();", more_button)
                                        self.logger.info("Clicked 'আরও' button with JavaScript")
                                else:
                                    # If not visible, use JavaScript directly
                                    self.driver.execute_script("arguments[0].click();", more_button)
                                    self.logger.info("Clicked hidden 'আরও' button with JavaScript")
                                
                                time.sleep(3)  # Wait for new articles to load
                            except Exception as click_exc:
                                self.logger.info(f"Failed to click আরও button: {str(click_exc)[:150]}")
                                break
                        except Exception as e:
                            self.logger.info(f"Exception while trying to click আরও button: {str(e)[:150]}")
                            break
                            self.logger.info(f"No more 'আরও' button or not clickable: {str(e)[:100]}")
                            break
                else:
                    # Original pagination logic for non-AJAX sites
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
            
            # Handle protocol-relative URLs (//example.com)
            if href.startswith('//'):
                href = 'https:' + href
            
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
