import logging
import argparse

from src.module1_data_acquisition.crawlers.bangla_crawlers import get_bangla_crawlers
from src.module1_data_acquisition.crawlers.english_crawlers import get_english_crawlers
from src.module1_preprocessing_indexing.embedding_generator import generate_embeddings
from src.module1_preprocessing_indexing.indexer import build_indices


def main():
    parser = argparse.ArgumentParser(description="CLIR Project - Crawling and Indexing")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Crawl command
    crawl_parser = subparsers.add_parser('crawl', help='Crawl news articles')
    crawl_parser.add_argument('--lang', choices=['bangla', 'english', 'all'], default='all', help='Language to crawl')
    crawl_parser.add_argument('--limit', type=int, default=50, help='Number of articles to crawl per site')
    crawl_parser.add_argument('--source', help='Specific source to crawl')
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings for articles')
    embed_parser.add_argument('--data-dir', default='data/raw', help='Directory containing raw JSON files')
    embed_parser.add_argument('--output', default='data/embeddings/articles_with_embeddings.pkl', help='Output pickle file')
    embed_parser.add_argument('--model', default='sentence-transformers/LaBSE', help='Sentence transformer model name')
    embed_parser.add_argument('--min-tokens', type=int, default=50, help='Minimum token count for filtering')
    embed_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for embedding generation')
    # Index command
    index_parser = subparsers.add_parser('index', help='Build search indices (Whoosh + FAISS)')
    index_parser.add_argument('--pickle', default='data/embeddings/articles_with_embeddings.pkl', help='Input pickle file')
    index_parser.add_argument('--whoosh-dir', default='data/indices/whoosh', help='Whoosh index directory')
    index_parser.add_argument('--faiss-file', default='data/indices/faiss_index.bin', help='FAISS index file')
    
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'crawl':
        # Crawling logic
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
    
    elif args.command == 'embed':
        # Embedding generation logic
        print(f"Generating embeddings with {args.model}")
        print(f"Data directory: {args.data_dir}")
        print(f"Output file: {args.output}")
        print(f"Min tokens: {args.min_tokens}")
        print(f"Batch size: {args.batch_size}")
        print("-" * 50)
        
        try:
            output_file = generate_embeddings(
                data_dir=args.data_dir,
                output_file=args.output,
                model_name=args.model,
                min_tokens=args.min_tokens,
                batch_size=args.batch_size
            )
            
            if output_file:
                print("\n" + "="*50)
                print("SUCCESS! Embeddings generated and saved.")
                print(f"File: {output_file}")
                print("="*50)
            else:
                print("Failed to generate embeddings!")
                
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            logging.error(f"Embedding generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.command == 'index':
        # Index building logic
        print(f"Building search indices...")
        print(f"Input pickle: {args.pickle}")
        print(f"Whoosh directory: {args.whoosh_dir}")
        print(f"FAISS file: {args.faiss_file}")
        print("-" * 50)
        
        try:
            whoosh_index, faiss_index = build_indices(
                pickle_file=args.pickle,
                whoosh_dir=args.whoosh_dir,
                faiss_file=args.faiss_file
            )
            
            print("\n" + "="*50)
            print("SUCCESS! Indices built successfully.")
            print(f"Whoosh: {args.whoosh_dir}")
            print(f"FAISS: {args.faiss_file}")
            print("="*50)
            
        except Exception as e:
            print(f"Error building indices: {e}")
            logging.error(f"Index building failed: {e}")
            import traceback
            traceback.print_exc()
                
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            logging.error(f"Embedding generation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
