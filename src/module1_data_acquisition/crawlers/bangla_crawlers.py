from .generic_crawler import GenericNewsCrawler
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
        max_load_more_clicks=150,
        delay=2.5
    ))
    
    # 2. BD News 24 (Selenium)
    crawlers.append(SeleniumCrawler(
        base_url="https://bangla.bdnews24.com/",
        language="bangla",
        source_name="bdnews24",
        selectors={
            'article_links': 'div#app a, a', 
            'title': 'h1.Title, h1',
            'body': 'div.article_body, div.custombody, article, div#root',
            'date': 'span.timeago',
            'date_attr': 'title'
        }
    ))
    
    # 3. Ittefaq 
    crawlers.append(GenericNewsCrawler(
        base_url="https://www.ittefaq.com.bd/",
        language="bangla",
        source_name="ittefaq",
        selectors={
            'article_links': 'a',
            'title': 'h1',
            'body': 'div.detail-content, div.print-content, article', 
            'date': 'span.date'
        },
        pagination_param='page'
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
        max_load_more_clicks=100,
        delay=1.5
    ))
    
    # 5. Dhaka Post
    crawlers.append(GenericNewsCrawler(
        base_url="https://www.dhakapost.com/",
        language="bangla",
        source_name="dhaka_post",
        selectors={
            'article_links': 'a',
            'title': 'h1',
            'body': 'div.post-content, div.ab_content, article',
            'date': 'span.time'
        },
        pagination_param='page'
    ))

    # 6. Jugantor
    crawlers.append(GenericNewsCrawler(
        base_url="https://www.jugantor.com/",
        language="bangla",
        source_name="jugantor",
        selectors={
            'article_links': 'a',
            'title': 'h1',
            'body': 'div.details-content, article',
            'date': 'span.publish-time'
        }
    ))

    # 7. Samakal
    crawlers.append(SeleniumCrawler(
        base_url="https://samakal.com/latest/news",
        language="bangla",
        source_name="samakal",
        selectors={
            'article_links': 'a[href*="/article/"]',
            'title': 'h1',
            'body': 'div.article-details, div.content',
            'date': 'div.article-time'
        }
    ))
    
    return crawlers
