from .selenium_crawler import SeleniumCrawler

def get_bangla_crawlers():
    crawlers = []
    
    # 1. Prothom Alo (Selenium with AJAX load more button)
    crawlers.append(SeleniumCrawler(
        base_url="https://www.prothomalo.com/",
        start_urls=[
            "https://www.prothomalo.com/collection/latest",
            "https://www.prothomalo.com/bangladesh",
            "https://www.prothomalo.com/bangladesh/district",
            "https://www.prothomalo.com/bangladesh/capital",
            "https://www.prothomalo.com/politics",
            "https://www.prothomalo.com/world",
            "https://www.prothomalo.com/sports",
            "https://www.prothomalo.com/business",
            "https://www.prothomalo.com/entertainment",
            "https://www.prothomalo.com/opinion",
            "https://www.prothomalo.com/lifestyle"
        ],
        language="bangla",
        source_name="prothom_alo",
        selectors={
            'article_links': 'a[href*="/bangladesh/"], a[href*="/politics/"], a[href*="/world/"], a[href*="/sports/"], a[href*="/business/"], a[href*="/entertainment/"], a[href*="/opinion/"], a[href*="/lifestyle/"]',
            'title': 'h1, h3.headline-title',
            'body': 'div.story-element-text, div.story-content p, article p',
            'date': 'time',
            'date_attr': 'datetime'
        },
        load_more_selector='span.load-more-content, div.more, div._7ZpjE',
        max_load_more_clicks=300,
        delay=2.5
    ))
    
    # 2. Ittefaq - Using SeleniumCrawler for AJAX pagination
    crawlers.append(SeleniumCrawler(
        base_url="https://www.ittefaq.com.bd/",
        language="bangla",
        source_name="ittefaq",
        start_urls=[
            "https://www.ittefaq.com.bd/latest-news",
            "https://www.ittefaq.com.bd/topic/বিশেষ-সংবাদ",
            "https://www.ittefaq.com.bd/national",
            "https://www.ittefaq.com.bd/country",
            "https://www.ittefaq.com.bd/politics",
            "https://www.ittefaq.com.bd/world-news",
            "https://www.ittefaq.com.bd/sports",
            "https://www.ittefaq.com.bd/entertainment",
            "https://www.ittefaq.com.bd/news"
        ],
        selectors={
            'article_links': 'a.link_overlay, h2.title a',
            'title': 'h1.title',  
            'body': 'div.viewport p', 
            'date': 'span.tts_time'  
        },
        load_more_selector='button#ajax_load_more_476_btn, button.ajax_load_btn',
        max_load_more_clicks=200,
        delay=2.0
    ))

    # 4. Bangla Tribune (with AJAX load more button)
    crawlers.append(SeleniumCrawler(
        base_url="https://www.banglatribune.com/",
        start_urls=[
            "https://www.banglatribune.com/আজকের-খবর", 
            "https://www.banglatribune.com/exclusive",
            "https://www.banglatribune.com/politics",
            "https://www.banglatribune.com/tech-and-gadget",
            "https://www.banglatribune.com/country",
            "https://www.banglatribune.com/business",
            "https://www.banglatribune.com/sports",
            "https://www.banglatribune.com/entertainment",
            "https://www.banglatribune.com/opinion",
            "https://www.banglatribune.com/international",
            "https://www.banglatribune.com/life-style",
            "https://www.banglatribune.com/literature",
        ],
        language="bangla",
        source_name="bangla_tribune",
        selectors={
            'article_links': 'a.link_overlay',  
            'title': 'h1.title',   
            'body': 'div.jw_article_body', 
            'date': 'span.time.aitm',
            'date_attr': 'data-published'
        },
        load_more_selector='button[id^="ajax_load_more_"]',
        max_load_more_clicks=200,
        delay=1.5
    ))
    
    # 5. Dhaka Post - Requires Selenium for AJAX pagination
    # 5. Dhaka Post - Uses infinite scroll
    crawlers.append(SeleniumCrawler(
        base_url="https://www.dhakapost.com/",
        start_urls=[
            "https://www.dhakapost.com/latest-news",
            "https://www.dhakapost.com/national",
            "https://www.dhakapost.com/politics",
            "https://www.dhakapost.com/economy",
            "https://www.dhakapost.com/international",
            "https://www.dhakapost.com/country",
            "https://www.dhakapost.com/sports",
            "https://www.dhakapost.com/campus",
            "https://www.dhakapost.com/education",
            "https://www.dhakapost.com/entertainment",
            "https://www.dhakapost.com/tech",
            "https://www.dhakapost.com/religion",
            "https://www.dhakapost.com/opinion",
            "https://www.dhakapost.com/life-style",
            "https://www.dhakapost.com/crime"
        ],
        language="bangla",
        source_name="dhaka_post",
        selectors={
            'article_links': 'a[href*="/national/"], a[href*="/politics/"], a[href*="/economy/"], a[href*="/international/"], a[href*="/country/"], a[href*="/sports/"], a[href*="/campus/"], a[href*="/education/"], a[href*="/entertainment/"], a[href*="/tech/"], a[href*="/opinion/"], a[href*="/life-style/"], a[href*="/crime/"]',
            'title': 'article h1',
            'body': 'article > div:nth-child(3) p',
            'date': 'article time'
        },
        load_more_selector='__infinite_scroll__',  # Uses infinite scroll
        max_load_more_clicks=500,
        delay=2.5
    ))

    # 6. Jugantor (Selenium with AJAX load more button)
    crawlers.append(SeleniumCrawler(
        base_url="https://www.jugantor.com/",
        start_urls=[
            "https://www.jugantor.com/latest",
            "https://www.jugantor.com/national",
            "https://www.jugantor.com/politics",
            "https://www.jugantor.com/economics",
            "https://www.jugantor.com/international",
            "https://www.jugantor.com/country-news",
            "https://www.jugantor.com/sports",
            "https://www.jugantor.com/entertainment",
            "https://www.jugantor.com/tech",
            "https://www.jugantor.com/literature"
        ],
        language="bangla",
        source_name="jugantor",
        selectors={
            'article_links': 'a.linkOverlay, a[href*="/national/"], a[href*="/politics/"], a[href*="/economics/"], a[href*="/international/"], a[href*="/sports/"]',
            'title': 'h1',
            'body': 'div.desktopDetailBody p',
            'date': 'span.publish-time, p.desktopTime',
        },
        load_more_selector='span.loadMoreButton, span.clickLoadMoreDesktop',
        max_load_more_clicks=300,
        delay=2.5
    ))

    crawlers.append(SeleniumCrawler(
        base_url="https://samakal.com/",
        start_urls=[
            "https://samakal.com/latest/news",
            "https://samakal.com/bangladesh",
            "https://samakal.com/politics",
            "https://samakal.com/economics",
            "https://samakal.com/international",
            "https://samakal.com/sports",
            "https://samakal.com/entertainment",
            "https://samakal.com/crime",
            "https://samakal.com/opinion",
            "https://samakal.com/capital",
            "https://samakal.com/lifestyle",
            "https://samakal.com/whole-country",
            "https://samakal.com/literature",
            "https://samakal.com/health",
            "https://samakal.com/education"
        ],
        language="bangla",
        source_name="samakal",
        selectors={
            'article_links': 'a[href*="/article/"]',
            'title': 'main h1',
            'body': 'main > div > div:nth-child(2) > div:nth-child(1) > div:nth-child(4) p',
            'date': 'main > div > div:nth-child(2) > div:nth-child(1) > div:nth-child(3) > div:nth-child(1) p:nth-child(2)',
            'date_attr': None
        },
        load_more_selector='__infinite_scroll__',  # Uses infinite scroll
        max_load_more_clicks=500,
        delay=3.0
    ))
    
    return crawlers
