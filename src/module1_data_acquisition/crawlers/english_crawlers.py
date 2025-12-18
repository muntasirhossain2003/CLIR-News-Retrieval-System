from .generic_crawler import GenericNewsCrawler
from .selenium_crawler import SeleniumCrawler

def get_english_crawlers():
    crawlers = []
    
    # 1. The Daily Star - Using Selenium with numbered pagination
    crawlers.append(SeleniumCrawler(
        base_url="https://www.thedailystar.net/",
        start_urls=[
            "https://www.thedailystar.net/news",
            "https://www.thedailystar.net/news/bangladesh",
            "https://www.thedailystar.net/news/investigative-stories",
            "https://www.thedailystar.net/tech-startup",
            "https://www.thedailystar.net/news/world",
            "https://www.thedailystar.net/news/asia",
            "https://www.thedailystar.net/sports",
            "https://www.thedailystar.net/business",
            "https://www.thedailystar.net/opinion",
            "https://www.thedailystar.net/lifestyle",
            "https://www.thedailystar.net/entertainment",
            "https://www.thedailystar.net/country-news"
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

    # 4. Prothom Alo English
    crawlers.append(SeleniumCrawler(
        base_url="https://en.prothomalo.com/",
        language="english",
        source_name="prothom_alo",
        start_urls=[
            "https://en.prothomalo.com/bangladesh",
            "https://en.prothomalo.com/international",
            "https://en.prothomalo.com/sports",
            "https://en.prothomalo.com/opinion",
            "https://en.prothomalo.com/business",
            "https://en.prothomalo.com/youth",
            "https://en.prothomalo.com/entertainment",
            "https://en.prothomalo.com/lifestyle"
        ],
        selectors={
            'article_links': 'a[href*="/bangladesh/"], a[href*="/international/"], a[href*="/sports/"], a[href*="/opinion/"], a[href*="/business/"], a[href*="/youth/"], a[href*="/entertainment/"], a[href*="/lifestyle/"]',
            'title': 'h1',
            'body': 'p',
            'date': 'time'
        },
        load_more_selector='__infinite_scroll__',
        delay=3.0
    ))
    
    # 5. Dhaka Tribune
    crawlers.append(SeleniumCrawler(
        base_url="https://www.dhakatribune.com/",
        start_urls=[
            "https://www.dhakatribune.com/bangladesh",
            "https://www.dhakatribune.com/bangladesh/dhaka",
            "https://www.dhakatribune.com/bangladesh/education",
            "https://www.dhakatribune.com/bangladesh/foreign-affairs",
            "https://www.dhakatribune.com/world",
            "https://www.dhakatribune.com/bangladesh/health",
            "https://www.dhakatribune.com/bangladesh/technology",
            "https://www.dhakatribune.com/bangladesh/entertainment",
            "https://www.dhakatribune.com/bangladesh/sports",
        ],
        language="english",
        source_name="dhaka_tribune",
        selectors={
            'article_links': 'a',
            'title': 'h1',
            'body': 'div.content_body, div.story-element-text',
            'date': 'span.time',
            'date_attr': 'title'
        },
        #it has "+more" button to get extra news
        load_more_selector='button.ajax_load_btn',
        max_load_more_clicks=100,
        delay=3.0
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
        load_more_selector='__infinite_scroll__', 
        max_load_more_clicks=50,
        delay=3.0
    ))

    # 7. United News of Bangladesh (UNB) - Infinite Scroll
    crawlers.append(SeleniumCrawler(
        base_url="https://unb.com.bd/",
        language="english",
        source_name="unb",
        start_urls=[
            "https://unb.com.bd/category/14/Bangladesh",
            "https://unb.com.bd/category/15/Politics",
            "https://unb.com.bd/category/16/Business",
            "https://unb.com.bd/category/17/Sports",
            "https://unb.com.bd/category/18/World",
            "https://unb.com.bd/category/19/Tech",
            "https://unb.com.bd/category/20/Entertainment",
            "https://unb.com.bd/category/21/Lifestyle",
            "https://unb.com.bd/category/22/Opinion",
            "https://unb.com.bd/category/25/Environment",
        ],
        selectors={
            'article_links': 'h3 a, article a, .news-item a',
            'title': 'div.upper-box h1, div.inner-box h1, main h1',
            'body': 'article p, .content-details p, main p',
            'date': 'time, span.date, div.date, .date'
        },
        load_more_selector='__infinite_scroll__',
        delay=1.5
    ))
    
    # 8. NTV BD
    crawlers.append(SeleniumCrawler(
        base_url="https://en.ntvbd.com/",
        language="english",
        source_name="ntv_bd",
        start_urls=[
            "https://en.ntvbd.com/bangladesh",
            "https://en.ntvbd.com/world",
            "https://en.ntvbd.com/sports",
            "https://en.ntvbd.com/entertainment",
            "https://en.ntvbd.com/business",
            "https://en.ntvbd.com/education",
            "https://en.ntvbd.com/sci-tech"
        ],
        selectors={
            'article_links': 'h3 a, h2 a, h4 a, div.card-title a, div.views-row a', 
            'title': 'article.node-news h1, .view-mode-full h1, h1[itemprop="headline"]',
            'body': 'div.section-media p, div.node-article p',
            'date': 'div.date'
        },
        load_more_selector='li.pager-show-more-next a',
        max_load_more_clicks=100,
        delay=3.0
    ))

    return crawlers
