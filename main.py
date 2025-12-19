import logging
import argparse
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.module1_data_acquisition.crawlers.bangla_crawlers import get_bangla_crawlers
from src.module1_data_acquisition.crawlers.english_crawlers import get_english_crawlers


def crawl_news(args):
    """Crawl news articles from websites"""
    crawlers = []

    if args.lang in ["bangla", "all"]:
        crawlers.extend(get_bangla_crawlers())

    if args.lang in ["english", "all"]:
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


def index_documents(args):
    """Module A: Index documents"""
    from src.module_a_indexing import DocumentProcessor

    print(f"Processing documents from: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    processor = DocumentProcessor(data_dir=args.input_dir, output_dir=args.output_dir)

    stats = processor.process_all_documents(batch_size=args.batch_size)

    print("\n=== Processing Statistics ===")
    print(f"Total processed: {stats['processed']}")
    print(f"Errors: {stats['errors']}")
    print(f"English docs: {stats['by_language'].get('en', 0)}")
    print(f"Bangla docs: {stats['by_language'].get('bn', 0)}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Total entities: {stats['total_entities']}")
    print(f"Processing time: {stats.get('time_elapsed', 0):.2f} seconds")
    print(f"Speed: {stats.get('docs_per_second', 0):.2f} docs/sec")


def search_documents(args):
    """Module B+C: Search documents"""
    from src.module_a_indexing import InvertedIndex
    from src.module_b_query_processing import QueryProcessor
    from src.module_c_retrieval import RetrievalEngine

    # Load index
    print(f"Loading index from: {args.index_path}")
    index = InvertedIndex()
    index.load_index(args.index_path)
    print(f"Loaded {len(index.doc_metadata)} documents\n")

    # Prepare documents
    documents = []
    for doc_id, metadata in index.doc_metadata.items():
        documents.append(
            {
                "doc_id": doc_id,
                "tokens": metadata.get("tokens", []),
                "content": metadata.get("content", ""),
                "metadata": metadata,
            }
        )

    # Initialize
    query_processor = QueryProcessor(
        enable_translation=args.translate,
        enable_expansion=args.expand,
        enable_entity_mapping=args.entities,
    )

    retrieval_engine = RetrievalEngine(
        inverted_index=index,
        documents=documents,
        enable_semantic=args.semantic,
        enable_fuzzy=args.fuzzy,
    )

    # Process query
    print(f"Query: {args.query}")
    processed = query_processor.process_query(args.query)
    print(f"Language: {processed['language']}")
    print(f"Normalized: {processed['normalized']}")

    if args.translate:
        print(f"English: {processed['en_query']}")
        print(f"Bangla: {processed['bn_query']}")

    # Search
    search_query = processed["en_query"] if args.translate else processed["normalized"]
    results = retrieval_engine.retrieve(
        search_query, method=args.method, top_k=args.top_k
    )

    # Display results
    print(f"\n=== Top {len(results)} Results ({args.method.upper()}) ===\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('title', 'Untitled')[:70]}")
        print(
            f"   Score: {result['score']:.4f} | Source: {result.get('source', 'Unknown')} | Lang: {result.get('language', '?')}"
        )
        if "snippet" in result:
            print(f"   {result['snippet'][:100]}...")
        print()


def test_modules(args):
    """Test modules"""
    if args.module == "a":
        test_module_a()
    elif args.module == "b":
        test_module_b()
    elif args.module == "c":
        test_module_c()
    elif args.module == "all":
        test_module_a()
        test_module_b()
        test_module_c()


def test_module_a():
    """Test Module A"""
    from src.module_a_indexing import (
        LanguageDetector,
        MultilingualTokenizer,
        NERExtractor,
        InvertedIndex,
    )

    print("\n=== Testing Module A: Indexing ===")
    print("\n1. Language Detector...")
    detector = LanguageDetector()
    result = detector.detect("Prime Minister announces policy")
    print(
        f"   ✓ Detected: {result['language']} (confidence: {result['confidence']:.2f})"
    )

    print("\n2. Tokenizer...")
    tokenizer = MultilingualTokenizer()
    tokens = tokenizer.tokenize(
        "The Prime Minister announced new policies", language="en"
    )
    print(f"   ✓ Tokens: {tokens[:5]}... ({len(tokens)} total)")

    print("\n3. NER Extractor...")
    ner = NERExtractor()
    entities = ner.extract_entities("Sheikh Hasina met Modi in Delhi", language="en")
    print(f"   ✓ Entities: {[e['text'] for e in entities]}")

    print("\n4. Inverted Index...")
    index = InvertedIndex()
    index.add_document("doc1", ["test", "document"], {"title": "Test"})
    print(f"   ✓ Created index with {len(index.index)} terms")

    print("\n✅ Module A PASSED\n")


def test_module_b():
    """Test Module B"""
    from src.module_b_query_processing import (
        QueryLanguageDetector,
        QueryNormalizer,
        QueryExpander,
        NamedEntityMapper,
    )

    print("\n=== Testing Module B: Query Processing ===")
    print("\n1. Query Detector...")
    detector = QueryLanguageDetector()
    result = detector.detect_query_language("education policy")
    print(
        f"   ✓ Detected: {result['language']} (confidence: {result['confidence']:.2f})"
    )

    print("\n2. Query Normalizer...")
    normalizer = QueryNormalizer()
    normalized = normalizer.normalize("  EDUCATION  Policy  ", language="en")
    print(f"   ✓ Normalized: '{normalized}'")

    print("\n3. Query Expander...")
    expander = QueryExpander()
    expanded = expander.expand("education", language="en", max_synonyms=2)
    print(f"   ✓ Expanded: {expanded}")

    print("\n4. Entity Mapper...")
    mapper = NamedEntityMapper()
    mapped = mapper.map_entity("Bangladesh", source_lang="en", target_lang="bn")
    print(f"   ✓ Mapped: {mapped}")

    print("\n✅ Module B PASSED\n")


def test_module_c():
    """Test Module C"""
    from src.module_c_retrieval import LexicalRetriever, FuzzyMatcher, HybridRanker

    print("\n=== Testing Module C: Retrieval ===")
    docs = [
        {
            "doc_id": "doc1",
            "tokens": ["education", "policy"],
            "metadata": {"title": "Education"},
        },
        {
            "doc_id": "doc2",
            "tokens": ["health", "program"],
            "metadata": {"title": "Health"},
        },
    ]

    print("\n1. BM25 Retrieval...")
    retriever = LexicalRetriever(docs)
    results = retriever.search_bm25(["education"], top_k=2)
    print(f"   ✓ Found {len(results)} results")

    print("\n2. Fuzzy Matcher...")
    matcher = FuzzyMatcher()
    score = matcher.fuzzy_match(["educaton"], ["education"])
    print(f"   ✓ Fuzzy score: {score:.4f}")

    print("\n3. Hybrid Ranker...")
    scores = {"bm25": {"doc1": 2.0, "doc2": 1.0}}
    ranker = HybridRanker()
    results = ranker.rank_documents(scores, method="weighted")
    print(f"   ✓ Ranked {len(results)} results")

    print("\n✅ Module C PASSED\n")


def main():
    parser = argparse.ArgumentParser(
        description="CLIR News Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl data (original crawler)
  python main.py --lang bangla --limit 50
  python main.py --lang english --source daily_star --limit 10
  
  # Index documents
  python main.py --index
  python main.py --index --input-dir data/raw --output-dir processed_data
  
  # Search documents
  python main.py --search "Sheikh Hasina education"
  python main.py --search "education policy" --method hybrid --translate --semantic
  
  # Test modules
  python main.py --test
  python main.py --test --module a
        """,
    )

    # Original crawler arguments (UNCHANGED)
    parser.add_argument(
        "--lang",
        choices=["bangla", "english", "all"],
        default="all",
        help="Language to crawl",
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Number of articles to crawl per site"
    )
    parser.add_argument("--source", help="Specific source to crawl")

    # New module arguments
    parser.add_argument(
        "--index", action="store_true", help="Index documents (Module A)"
    )
    parser.add_argument(
        "--search", metavar="QUERY", help="Search documents (Modules B+C)"
    )
    parser.add_argument("--test", action="store_true", help="Test modules")

    # Index options
    parser.add_argument("--input-dir", default="data/raw", help="Input directory")
    parser.add_argument(
        "--output-dir", default="processed_data", help="Output directory"
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")

    # Search options
    parser.add_argument(
        "--index-path",
        default="processed_data/inverted_index.pkl",
        help="Index file path",
    )
    parser.add_argument(
        "--method",
        choices=["bm25", "tfidf", "semantic", "fuzzy", "hybrid"],
        default="hybrid",
        help="Retrieval method",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--translate", action="store_true", help="Enable translation")
    parser.add_argument("--expand", action="store_true", help="Enable query expansion")
    parser.add_argument("--entities", action="store_true", help="Enable entity mapping")
    parser.add_argument(
        "--semantic", action="store_true", help="Enable semantic search"
    )
    parser.add_argument("--fuzzy", action="store_true", help="Enable fuzzy matching")

    # Test options
    parser.add_argument(
        "--module", choices=["a", "b", "c", "all"], default="all", help="Module to test"
    )

    args = parser.parse_args()

    # Execute based on arguments
    if args.test:
        test_modules(args)
    elif args.index:
        index_documents(args)
    elif args.search:
        args.query = args.search
        search_documents(args)
    else:
        # Original crawler behavior (default)
        crawl_news(args)


if __name__ == "__main__":
    main()
