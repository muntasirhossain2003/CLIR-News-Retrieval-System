import logging
import argparse
import sys
import os


PROJECT_ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

# Module A folder name is not a valid Python package name. Add it directly.
module_a_path = os.path.join(
    PROJECT_ROOT,
    'src',
    'Module A — Dataset Construction & Indexing',
)
sys.path.append(module_a_path)

from crawlers.bangla_crawlers import get_bangla_crawlers
from crawlers.english_crawlers import get_english_crawlers

def main():
    parser = argparse.ArgumentParser(description="CLIR Assignment Crawler")
    parser.add_argument('--lang', choices=['bangla', 'english', 'all'], default='all', help='Language to crawl')
    parser.add_argument('--limit', type=int, default=50, help='Number of articles to crawl per site')
    parser.add_argument('--source', help='Specific source to crawl')
    
    args = parser.parse_args()
    
    crawlers = []
    
    if args.lang in ['bangla', 'all']:
        crawlers.extend(get_bangla_crawlers())
        
    if args.lang in ['english', 'all']:
        crawlers.extend(get_english_crawlers())
        
    if args.source:
        crawlers = [c for c in crawlers if c.source_name == args.source]
        
    if not crawlers:
        print("No crawlers selected!")
        return
        
    print(f"Starting crawl with {len(crawlers)} crawlers. Limit per site: {args.limit}")
    
    for crawler in crawlers:
        try:
            print(f"Running {crawler.source_name}...")
            crawler.crawl(limit=args.limit)
        except Exception as e:
            print(f"Failed to crawl {crawler.source_name}: {e}")
            logging.error(f"Critical failure in {crawler.source_name}: {e}")

if __name__ == "__main__":
    main()
