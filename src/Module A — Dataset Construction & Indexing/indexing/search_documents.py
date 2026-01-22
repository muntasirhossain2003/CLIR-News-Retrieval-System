"""Search indexed documents using lexical (BM25) or semantic (embeddings) methods."""

import argparse
import os
from pathlib import Path
from lexical_indexer import LexicalIndexer
from semantic_indexer import SemanticIndexer

# Get absolute path to indexes directory
SCRIPT_DIR = Path(__file__).parent.absolute()
INDEXES_DIR = SCRIPT_DIR.parent.parent.parent / "indexes"


def lexical_search(query, language=None, limit=10):
    """Search using WHOOSH/BM25 (keyword-based)."""
    print("=" * 80)
    print(f"LEXICAL SEARCH (BM25): '{query}'")
    print("=" * 80)

    indexer = LexicalIndexer(index_dir=str(INDEXES_DIR / "whoosh"))
    indexer.open_index()

    results = indexer.search(query, language=language, limit=limit)

    if not results:
        print("\nNo results found.")
        return

    print(f"\nFound {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Language: {result['language']}")
        print(f"   Source: {result['source']}")
        print(f"   ID: {result['doc_id']}")
        if result.get("body"):
            preview = result["body"][:150].replace("\n", " ")
            print(f"   Preview: {preview}...")
        print()


def semantic_search(query, top_k=10):
    """Search using multilingual embeddings (meaning-based)."""
    print("=" * 80)
    print(f"SEMANTIC SEARCH (Embeddings): '{query}'")
    print("=" * 80)

    indexer = SemanticIndexer(index_dir="../../../indexes/semantic")
    indexer.load_index()

    results = indexer.search(query, top_k=top_k)

    if not results:
        print("\nNo results found.")
        return

    print(f"\nFound {len(results)} results:\n")

    # Load lexical index to get document details
    lexical = LexicalIndexer(index_dir="../../../indexes/whoosh")
    lexical.open_index()

    for i, (doc_id, score) in enumerate(results, 1):
        # Get document details
        doc_results = lexical.search(f"doc_id:{doc_id}", limit=1)
        if doc_results:
            doc = doc_results[0]
            print(f"{i}. {doc['title']}")
            print(f"   Similarity: {score:.3f}")
            print(f"   Language: {doc['language']}")
            print(f"   Source: {doc['source']}")
            print(f"   ID: {doc_id}")
            if doc.get("body"):
                preview = doc["body"][:150].replace("\n", " ")
                print(f"   Preview: {preview}...")
            print()


def hybrid_search(query, language=None, top_k=10):
    """Combine both lexical and semantic search results."""
    print("=" * 80)
    print(f"HYBRID SEARCH (BM25 + Embeddings): '{query}'")
    print("=" * 80)

    # Lexical search
    lexical_indexer = LexicalIndexer(index_dir=str(INDEXES_DIR / "whoosh"))
    lexical_indexer.open_index()
    lexical_results = lexical_indexer.search(query, language=language, limit=top_k)

    # Semantic search
    semantic_indexer = SemanticIndexer(index_dir=str(INDEXES_DIR / "semantic"))
    semantic_indexer.load_index()
    semantic_results = semantic_indexer.search(query, top_k=top_k)

    # Combine results with weighted scoring
    combined = {}

    # Add lexical results (weight: 0.5)
    for result in lexical_results:
        doc_id = result["doc_id"]
        combined[doc_id] = {
            "score": result["score"] * 0.5,
            "title": result["title"],
            "language": result["language"],
            "source": result["source"],
            "body": result.get("body", ""),
        }

    # Add semantic results (weight: 0.5)
    for doc_id, sim_score in semantic_results:
        if doc_id in combined:
            combined[doc_id]["score"] += sim_score * 0.5
        else:
            # Get document details
            doc_results = lexical_indexer.search(f"doc_id:{doc_id}", limit=1)
            if doc_results:
                doc = doc_results[0]
                combined[doc_id] = {
                    "score": sim_score * 0.5,
                    "title": doc["title"],
                    "language": doc["language"],
                    "source": doc["source"],
                    "body": doc.get("body", ""),
                }

    # Sort by combined score
    sorted_results = sorted(combined.items(), key=lambda x: x[1]["score"], reverse=True)

    print(f"\nFound {len(sorted_results)} results:\n")

    for i, (doc_id, data) in enumerate(sorted_results[:top_k], 1):
        print(f"{i}. {data['title']}")
        print(f"   Combined Score: {data['score']:.3f}")
        print(f"   Language: {data['language']}")
        print(f"   Source: {data['source']}")
        print(f"   ID: {doc_id}")
        if data["body"]:
            preview = data["body"][:150].replace("\n", " ")
            print(f"   Preview: {preview}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search indexed documents")
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--method",
        "-m",
        choices=["lexical", "semantic", "hybrid"],
        default="hybrid",
        help="Search method (default: hybrid)",
    )
    parser.add_argument(
        "--language",
        "-l",
        choices=["bangla", "english", "bn", "en"],
        help="Filter by language (for lexical search)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )

    args = parser.parse_args()

    # Normalize language
    if args.language:
        if args.language in ["bn", "bangla"]:
            args.language = "bangla"
        elif args.language in ["en", "english"]:
            args.language = "english"

    print()

    if args.method == "lexical":
        lexical_search(args.query, language=args.language, limit=args.limit)
    elif args.method == "semantic":
        semantic_search(args.query, top_k=args.limit)
    else:  # hybrid
        hybrid_search(args.query, language=args.language, top_k=args.limit)

    print("=" * 80)
