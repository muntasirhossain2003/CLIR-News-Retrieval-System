from .generic_crawler import GenericNewsCrawler
from .selenium_crawler import SeleniumCrawler

def get_english_crawlers():
    crawlers = []
    
    # 1. The Daily Star - Using Selenium with numbered pagination
    crawlers.append(SeleniumCrawler(
        base_url="https://www.thedailystar.net/",
        start_urls=[
            "https://www.thedailystar.net/news/bangladesh",
            "https://www.thedailystar.net/news/world",
            "https://www.thedailystar.net/news/asia",
            "https://www.thedailystar.net/sports",
            "https://www.thedailystar.net/business",
            "https://www.thedailystar.net/opinion",
            "https://www.thedailystar.net/lifestyle",
            "https://www.thedailystar.net/entertainment"
        ],
        language="english",
        source_name="daily_star",
        selectors={
            'article_links': 'a[href*="/news/"]',
            'title': 'h1',
            'body': 'div.field-body p, div.pb-20 p',
            'date': 'div.date, span.date, time'
        },
        pagination_param='page',  # Enables ?page=1, ?page=2, etc.
        delay=2.0
    ))
    
    # 2. New Age - Using Selenium with numbered pagination
    crawlers.append(SeleniumCrawler(
        base_url="https://www.newagebd.net/",
        start_urls=[
            "https://www.newagebd.net/articlelist/41/bangladesh",
            "https://www.newagebd.net/articlelist/44/business-economy",
            "https://www.newagebd.net/articlelist/42/politics",
            "https://www.newagebd.net/articlelist/45/world",
            "https://www.newagebd.net/articlelist/46/sports",
            "https://www.newagebd.net/articlelist/47/entertainment",
            "https://www.newagebd.net/articlelist/48/education",
            "https://www.newagebd.net/articlelist/49/opinion"
        ],
        language="english",
        source_name="new_age",
        selectors={
            'article_links': 'a[href*="/post/"]',
            'title': 'h1',
            'body': 'div.content-details p',
            'date': 'span.posted-on, div.date'
        },
        pagination_param='page',  # Enables ?page=1, ?page=2, etc.
        delay=2.0
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
    crawlers.append(SeleniumCrawler(
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
    crawlers.append(SeleniumCrawler(
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

    # 6. Financial Express - Infinite Scroll  
    crawlers.append(SeleniumCrawler(
        base_url="https://thefinancialexpress.com.bd/",
        language="english",
        source_name="financial_express",
        start_urls=[
            "https://thefinancialexpress.com.bd/all-news",
            "https://thefinancialexpress.com.bd/national",
            "https://thefinancialexpress.com.bd/trade",
            "https://thefinancialexpress.com.bd/stock",
            "https://thefinancialexpress.com.bd/views",
            "https://thefinancialexpress.com.bd/world",
            "https://thefinancialexpress.com.bd/sports",
        ],
        selectors={
            'article_links': 'h3 a',
            'title': 'h1',
            'body': 'article > p:not([class*="share"]):not([class*="publish"]):not([class*="meta"]), div.article-content > p',
            'date': 'time, span.date'
        },
        load_more_selector='__infinite_scroll__',  # Special marker 
        max_load_more_clicks=50,
        delay=3.0
    ))
    
    return crawlers
