from .generic_crawler import GenericNewsCrawler
from .selenium_crawler import SeleniumCrawler

def get_bangla_crawlers():
    crawlers = []
    
    # 1. Prothom Alo
    crawlers.append(GenericNewsCrawler(
        base_url="https://www.prothomalo.com/",
        language="bangla",
        source_name="prothom_alo",
        selectors={
            'article_links': 'a', 
            'title': 'h1',
            'body': 'div.story-element-text, div.story-content, article',
            'date': 'time',
            'date_attr': 'datetime'
        }
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
    
    # 3. Ittefaq (Replacement for Kaler Kantho)
    crawlers.append(GenericNewsCrawler(
        base_url="https://www.ittefaq.com.bd/",
        language="bangla",
        source_name="ittefaq",
        selectors={
            'article_links': 'a',
            'title': 'h1',
            'body': 'div.detail-content, div.print-content, article', 
            'date': 'span.date'
        }
    ))

    # 4. Bangla Tribune
    crawlers.append(GenericNewsCrawler(
        base_url="https://www.banglatribune.com/",
        start_urls=[
            "https://www.banglatribune.com/আজকের-খবর", 
            "https://www.banglatribune.com/exclusive",
            "https://www.banglatribune.com/politics",
            "https://www.banglatribune.com/tech-and-gadget"
        ],
        language="bangla",
        source_name="bangla_tribune",
        selectors={
            'article_links': 'a',  
            'title': 'h1.title',   
            'body': 'div.viewport.jw_article_body', 
            'date': 'span.time_rel',
            'date_attr': 'title'
        }
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
        }
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
