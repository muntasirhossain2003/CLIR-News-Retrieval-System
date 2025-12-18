"""
Simplified Selenium Crawler - Main orchestration class.
Driver management and button strategies extracted to separate files.
"""
from .base_crawler import BaseCrawler
from .selenium_driver import SeleniumDriverManager
from .selenium_buttons import ButtonClickStrategy
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from ..utils import clean_text, parse_date


class SeleniumCrawler(BaseCrawler):
    def __init__(self, base_url, language, source_name, selectors, start_urls=None, delay=2.0, pagination_param=None, load_more_selector=None, max_load_more_clicks=50):
        super().__init__(base_url=base_url, language=language, source_name=source_name, delay=delay)
        self.selectors = selectors
        self.start_urls = start_urls if start_urls else [base_url]
        self.pagination_param = pagination_param
        self.load_more_selector = load_more_selector
        self.max_load_more_clicks = max_load_more_clicks
        
        # Use driver manager instead of direct driver
        self.driver_manager = SeleniumDriverManager(headless=True)
        self.button_strategy = None  # Will be initialized when driver is ready
        
    def _setup_driver(self):
        """Initialize driver and button strategy."""
        self.driver_manager.setup()
        if not self.button_strategy:
            self.button_strategy = ButtonClickStrategy(self.driver_manager.driver)

    def _teardown_driver(self):
        """Cleanup driver."""
        self.driver_manager.teardown()

    def fetch_page_selenium(self, url):
        """Fetch page using Selenium."""
        return self.driver_manager.get_page(url, delay=self.delay)

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
                    self.driver_manager.driver.get(url)
                    time.sleep(self.delay)
                    
                    # Main pagination loop
                    clicks_without_progress = 0
                    last_link_count = 0
                    scroll_attempts = 0
                    max_scroll_attempts = 100  # Limit scrolls per category
                    
                    while count < limit and scroll_attempts < max_scroll_attempts:
                        # Handle pagination BEFORE extracting links (for infinite scroll)
                        if self.load_more_selector == '__infinite_scroll__':
                            # On first iteration, don't scroll yet - get initial links
                            if last_link_count > 0:
                                self.logger.info(f"Attempting infinite scroll on {url} (attempt {scroll_attempts + 1})")
                                self.button_strategy.handle_infinite_scroll(wait_time=3.0)  # Increased from self.delay to 3.0 seconds
                                scroll_attempts += 1
                        
                        # Get current page articles
                        soup = self.driver_manager.get_current_soup()
                        article_links = self.extract_links(soup, url)
                        
                        if not article_links:
                            break
                        
                        # Find new links
                        new_links = [link for link in article_links if link not in visited_urls]
                        self.logger.info(f"Found {len(article_links)} total links on page, {len(new_links)} are new")
                        
                        # Check progress
                        if not new_links:
                            clicks_without_progress += 1
                            if clicks_without_progress >= 3:  # Reduced from 10 to 3 for faster detection
                                self.logger.info("No new articles after 3 scroll attempts, moving to next start URL")
                                break
                        else:
                            clicks_without_progress = 0
                        
                        # Process articles
                        articles_saved = self._process_article_links(new_links, count, limit, visited_urls)
                        count += articles_saved
                        self.logger.info(f"Saved {articles_saved} articles this round")
                        
                        last_link_count = len(article_links)
                        
                        if count >= limit:
                            break
                        
                        # For button-based pagination
                        if self.load_more_selector and self.load_more_selector != '__infinite_scroll__':
                            if not self._click_load_more_button():
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

    def _process_article_links(self, links, current_count, limit, visited_urls):
        """Process a batch of article links and return count of saved articles."""
        saved = 0
        for link in links:
            if current_count + saved >= limit:
                break
            
            visited_urls.add(link)
            print(f"[{self.source_name}] Fetching ({current_count + saved + 1}/{limit}): {link}")
            article_data = self.parse_article(link)
            if article_data and self.save_article(article_data):
                saved += 1
                if (current_count + saved) % 10 == 0:
                    self.logger.info(f"Progress: {current_count + saved}/{limit}")
        return saved
    
    def _click_load_more_button(self):
        """Try to find and click the load more button. Returns True if successful."""
        try:
            self.driver_manager.scroll_to_bottom()
            time.sleep(1.5)
            
            # Find button using strategy
            button = self.button_strategy.find_button(self.load_more_selector)
            if not button:
                self.logger.info("Load more button not found, ending pagination")
                return False
            
            # Click button using strategy
            if self.button_strategy.click_button(button):
                time.sleep(3)
                return True
            else:
                return False
        except Exception as e:
            self.logger.info(f"Exception while clicking button: {str(e)[:150]}")
            return False
    
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
            body_paragraphs = []
            for tag in body_tags:
                text = clean_text(tag.text)
                
                # For Daily Star, stop if we encounter Bangla text (indicates related news section)
                if self.source_name == 'daily_star':
                    # Check if text contains Bangla Unicode characters (U+0980 to U+09FF)
                    has_bangla = any('\u0980' <= char <= '\u09FF' for char in text)
                    if has_bangla:
                        break  # Stop collecting paragraphs when we hit Bangla content
                
                # For New Age, skip paragraphs with footer content
                if self.source_name == 'new_age':
                    footer_markers = ['Editor:', 'PABX:', 'Fax:', 'adnewage@gmail.com', 
                                     'For Advertisement', 'Cell: +880', 'Copyright', 'Nurul Kabir']
                    if any(marker in text for marker in footer_markers):
                        continue  # Skip this paragraph if it contains footer content
                
                # Filter out unwanted content
                if (len(text) > 50 and 
                    'বিজ্ঞাপন' not in text and 
                    'আরও পড়ুন' not in text and
                    'Site use implies' not in text and
                    'Privacy Policy' not in text and
                    'PRIVACY POLICY' not in text and
                    'TERMS OF USE' not in text and
                    'SAMAKAL ALL RIGHTS RESERVED' not in text and
                    'ফোন :' not in text and
                    'বিজ্ঞাপন :' not in text and
                    'ই-মেইল:' not in text and
                    'samakalad@gmail.com' not in text and
                    'marketingonline@samakal.com' not in text and
                    'উন্নয়নে ইমিথমেকারস.কম' not in text):
                    body_paragraphs.append(text)
            data['body'] = " ".join(body_paragraphs)
        else:
             p_tags = soup.find_all('p')
             body_paragraphs = []
             for p in p_tags:
                 text = clean_text(p.text)
                 
                 # For New Age, skip paragraphs with footer content (fallback path)
                 if self.source_name == 'new_age':
                     footer_markers = ['Editor:', 'PABX:', 'Fax:', 'adnewage@gmail.com', 
                                      'For Advertisement', 'Cell: +880', 'Copyright', 'Nurul Kabir']
                     if any(marker in text for marker in footer_markers):
                         continue  # Skip this paragraph if it contains footer content
                 
                 if (len(text) > 50 and 
                     'বিজ্ঞাপন' not in text and 
                     'আরও পড়ুন' not in text and
                     'Site use implies' not in text and
                     'Privacy Policy' not in text):
                     body_paragraphs.append(text)
             data['body'] = " ".join(body_paragraphs)

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
