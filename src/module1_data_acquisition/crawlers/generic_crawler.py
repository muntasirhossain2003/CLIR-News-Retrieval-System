from .base_crawler import BaseCrawler
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from ..utils import clean_text, parse_date

class GenericNewsCrawler(BaseCrawler):
    def __init__(self, base_url, language, source_name, selectors, start_urls=None, delay=1.0):
        super().__init__(base_url=base_url, language=language, source_name=source_name, delay=delay)
        self.selectors = selectors
        self.start_urls = start_urls if start_urls else [base_url]
        # selectors should be a dict:
        # {
        #   'article_links': 'css_selector_for_links',
        #   'title': 'css_selector',
        #   'body': 'css_selector',
        #   'date': 'css_selector',
        #   'date_attr': 'attribute_name_optional' 
        # }

    def crawl(self, limit=100):
        self.logger.info(f"Starting crawl for {self.source_name}...")
        count = 0
        visited_urls = set()

        start_urls = self.start_urls
        
        for url in start_urls:
            if count >= limit:
                break
                
            soup = self.fetch_page(url)
            if not soup:
                continue
                
            article_links = self.extract_links(soup, url)
            self.logger.info(f"Found {len(article_links)} links on {url}")
            
            for link in article_links:
                if count >= limit:
                    break
                if link in visited_urls:
                    continue
                    
                visited_urls.add(link)
                article_data = self.parse_article(link)
                if article_data:
                    if self.save_article(article_data):
                        count += 1
                        
        self.logger.info(f"Finished crawling {self.source_name}. Total articles: {count}")

    def extract_links(self, soup, base_url):
        links = set()
        selector = self.selectors.get('article_links', 'a')
        
        for a in soup.select(selector):
            href = a.get('href')
            if not href:
                continue
            
            full_url = urljoin(base_url, href)
            
            # Domain check
            if self.base_url.replace('https://', '').replace('http://', '').split('/')[0] not in full_url:
                continue

            # Heuristic: Article URLs usually have decent length and often contain hyphens
            # Skip short links (likely navigation) unless they are clearly strictly filtered by selector
            if selector == 'a' and len(full_url) < len(base_url) + 15:
                continue
                
            links.add(full_url)
                
        return list(links)

    def parse_article(self, url):
        soup = self.fetch_page(url)
        if not soup:
            return None
            
        data = {
            'url': url,
            'language': self.language,
            'source': self.source_name
        }
        
        # Title
        title_tag = soup.select_one(self.selectors.get('title', 'h1'))
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
             # Fallback: try p tags if body selector failed
             p_tags = soup.find_all('p')
             body_text = " ".join([clean_text(p.text) for p in p_tags])
             data['body'] = body_text

        if not data.get('body'):
            self.logger.warning(f"No body found for {url}")
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
