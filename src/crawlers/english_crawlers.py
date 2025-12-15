from .generic_crawler import GenericNewsCrawler
from .selenium_crawler import SeleniumCrawler

def get_english_crawlers():
    crawlers = []
    
    # 1. The Daily Star
    crawlers.append(GenericNewsCrawler(
        base_url="https://www.thedailystar.net/",
        language="english",
        source_name="daily_star",
        selectors={
            'article_links': 'a',
            'title': 'h1',
            'body': 'div.field-body, div.pb-20, article',
            'date': 'div.date, span.date'
        }
    ))
    
    # 2. New Age
    crawlers.append(GenericNewsCrawler(
        base_url="https://www.newagebd.net/",
        language="english",
        source_name="new_age",
        selectors={
            'article_links': 'a',
            'title': 'h1',
            'body': 'div.content-details, div.story-content',
            'date': 'span.posted-on, div.date'
        }
    ))
    
    # 3. Daily Observer
    crawlers.append(GenericNewsCrawler(
        base_url="https://www.observerbd.com/",
        language="english",
        source_name="daily_observer",
        selectors={
            'article_links': 'a[href*="news/"]',
            'title': 'h1',
            'body': 'div#mp1, article',
            'date': 'span.date' 
        }
    ))

    # 4. Daily Sun
    crawlers.append(GenericNewsCrawler(
        base_url="https://www.daily-sun.com/",
        language="english",
        source_name="daily_sun",
        selectors={
            'article_links': 'a',
            'title': 'h1',
            'body': 'div.news-content, div.content',
            'date': 'div.news-time'
        }
    ))
    
    # 5. Dhaka Tribune
    crawlers.append(GenericNewsCrawler(
        base_url="https://www.dhakatribune.com/",
        language="english",
        source_name="dhaka_tribune",
        selectors={
            'article_links': 'a',
            'title': 'h1',
            'body': 'div.content_body, div.story-element-text',
            'date': 'span.time',
            'date_attr': 'title'
        }
    ))

    # 6. Financial Express
    crawlers.append(GenericNewsCrawler(
        base_url="https://thefinancialexpress.com.bd/",
        language="english",
        source_name="financial_express",
        selectors={
            'article_links': 'a[href*="/views/"]',
            'title': 'h1',
            'body': 'div.article-body, div.content',
            'date': 'span.date'
        }
    ))
    
    return crawlers
