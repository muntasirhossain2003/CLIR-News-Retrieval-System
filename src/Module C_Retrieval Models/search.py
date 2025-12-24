"""
CLIR Search Interface

Cross-Lingual Information Retrieval Search Tool

DEFAULT MODELS:
- BM25 (Lexical Retrieval)
- Semantic (Cross-lingual with embeddings)

Usage:
    python search.py "query"
    python search.py --models bm25 semantic tfidf "query"
    python search.py --models all "query"
    python search.py  (interactive mode)
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from tfidf_retrieval import build_tfidf_index, retrieve_tfidf
from bm25_retrieval import build_bm25_index, retrieve_bm25
from fuzzy_retrieval import retrieve_fuzzy_per_term

# Try to import semantic and hybrid
SEMANTIC_AVAILABLE = True
HYBRID_AVAILABLE = True

try:
    from semantic_retrieval import encode_documents, retrieve_semantic
except Exception as e:
    import traceback

    print(f"\n⚠️  Semantic retrieval not available: {e}")
    print(f"   Error type: {type(e).__name__}")
    traceback.print_exc()
    print("   Continuing without semantic...\n")
    SEMANTIC_AVAILABLE = False

try:
    from hybrid_retrieval import hybrid_rank
except Exception:
    HYBRID_AVAILABLE = False


def load_documents() -> Dict[str, str]:
    """Load documents from data/raw/ directory."""
    print("\n" + "=" * 80)
    print("LOADING DOCUMENTS")
    print("=" * 80)

    documents = {}

    # Get project root (3 levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "raw"

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return {}

    # Define sources
    english_sources = [
        "daily_star",
        "dhaka_tribune",
        "financial_express",
        "new_age",
        "ntv_bd",
        "prothom_alo",
        "unb",
    ]
    bangla_sources = [
        "bangla_tribune",
        "dhaka_post",
        "ittefaq",
        "jugantor",
        "prothom_alo",
        "samakal",
    ]

    # Load English
    print("\nLoading English documents...")
    for source in english_sources:
        source_dir = data_dir / "english" / source
        if source_dir.exists():
            files = list(source_dir.glob("*.json"))
            for f in files:
                try:
                    with open(f, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        doc_id = f"en_{source}_{f.stem}"
                        title = data.get("title", "")
                        content = data.get("content", "") or data.get("body", "")
                        documents[doc_id] = f"{title} {content}".strip()
                except Exception:
                    continue
            if files:
                print(f"  {source}: {len(files)} documents")

    # Load Bangla
    print("\nLoading Bangla documents...")
    for source in bangla_sources:
        source_dir = data_dir / "bangla" / source
        if source_dir.exists():
            files = list(source_dir.glob("*.json"))
            for f in files:
                try:
                    with open(f, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        doc_id = f"bn_{source}_{f.stem}"
                        title = data.get("title", "")
                        content = data.get("content", "") or data.get("body", "")
                        documents[doc_id] = f"{title} {content}".strip()
                except Exception:
                    continue
            if files:
                print(f"  {source}: {len(files)} documents")

    print(f"\nTotal: {len(documents)} documents loaded")
    print("=" * 80)
    return documents


def build_indexes(documents: Dict[str, str], models: Set[str]):
    """Build indexes for selected models."""
    print("\n" + "=" * 80)
    print("BUILDING INDEXES")
    print("=" * 80)

    indexes = {}

    if "tfidf" in models:
        print("\nBuilding TF-IDF index...")
        start = time.time()
        indexes["tfidf"] = build_tfidf_index(documents)
        print(f"✓ TF-IDF ready ({time.time()-start:.2f}s)")

    if "bm25" in models:
        print("\nBuilding BM25 index...")
        start = time.time()
        indexes["bm25"] = build_bm25_index(documents)
        print(f"✓ BM25 ready ({time.time()-start:.2f}s)")

    if "semantic" in models and SEMANTIC_AVAILABLE:
        print("\nBuilding Semantic embeddings...")
        print("(This may take a few minutes for large corpus...)")
        start = time.time()
        try:
            indexes["semantic"] = encode_documents(documents)
            print(f"✓ Semantic ready ({time.time()-start:.2f}s)")
        except Exception as e:
            print(f"✗ Semantic failed: {e}")
            indexes["semantic"] = None
    elif "semantic" in models:
        print("\n✗ Semantic skipped (not available)")
        indexes["semantic"] = None

    print("\n" + "=" * 80)
    return indexes


def search(
    query: str,
    documents: Dict[str, str],
    indexes: Dict,
    models: Set[str],
    top_k: int = 10,
):
    """Run search with selected models."""
    print("\n" + "=" * 80)
    print(f"SEARCH: '{query}'")
    print("=" * 80)
    print(f"Models: {', '.join(sorted(models)).upper()}")
    print("=" * 80)

    results = {}

    # TF-IDF
    if "tfidf" in models and "tfidf" in indexes:
        print("\n[TF-IDF]")
        start = time.time()
        tfidf_results = retrieve_tfidf(query, indexes["tfidf"], top_k=top_k)
        results["tfidf"] = tfidf_results
        display_results(
            query, tfidf_results, documents, f"TF-IDF ({time.time()-start:.3f}s)", top_k
        )

    # BM25
    if "bm25" in models and "bm25" in indexes:
        print("\n[BM25]")
        start = time.time()
        bm25_results = retrieve_bm25(query, indexes["bm25"], top_k=top_k)
        results["bm25"] = bm25_results
        display_results(
            query, bm25_results, documents, f"BM25 ({time.time()-start:.3f}s)", top_k
        )
    else:
        bm25_results = []

    # Semantic
    if "semantic" in models and indexes.get("semantic") is not None:
        print("\n[SEMANTIC]")
        start = time.time()
        semantic_results = retrieve_semantic(query, indexes["semantic"], top_k=top_k)
        results["semantic"] = semantic_results
        display_results(
            query,
            semantic_results,
            documents,
            f"Semantic ({time.time()-start:.3f}s)",
            top_k,
        )
    elif "semantic" in models:
        print("\n[SEMANTIC] - Skipped (not available)")
        semantic_results = []
    else:
        semantic_results = []

    # Hybrid
    if "hybrid" in models and HYBRID_AVAILABLE and bm25_results and semantic_results:
        print("\n[HYBRID]")
        start = time.time()
        hybrid_results = hybrid_rank(
            bm25_results,
            semantic_results,
            weights={"lexical": 0.5, "semantic": 0.5},
            top_k=top_k,
        )
        results["hybrid"] = hybrid_results
        display_results(
            query,
            hybrid_results,
            documents,
            f"Hybrid ({time.time()-start:.3f}s)",
            top_k,
        )
    elif "hybrid" in models:
        print("\n[HYBRID] - Skipped (requires BM25 + Semantic)")

    # Fuzzy
    if "fuzzy" in models:
        print("\n[FUZZY]")
        start = time.time()
        fuzzy_results = retrieve_fuzzy_per_term(
            query, documents, top_k=top_k, min_score=0.5
        )
        results["fuzzy"] = fuzzy_results
        display_results(
            query, fuzzy_results, documents, f"Fuzzy ({time.time()-start:.3f}s)", top_k
        )

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\n{'Method':<15} {'Results':<10} {'Top Score':<12} {'Status'}")
        print("-" * 60)
        for method, res in results.items():
            count = len(res)
            score = f"{res[0][1]:.4f}" if res else "N/A"
            status = "✓" if res else "✗"
            print(f"{method.upper():<15} {count:<10} {score:<12} {status}")
        print("=" * 80)


def display_results(
    query: str,
    results: List[Tuple[str, float]],
    documents: Dict[str, str],
    title: str,
    top_k: int,
):
    """Display search results."""
    print(f"\n{title}")
    print("-" * 80)

    if not results:
        print("❌ NO RESULTS (FAILURE CASE)")
        return

    print(f"Found: {len(results)} documents\n")

    for rank, (doc_id, score) in enumerate(results[:top_k], 1):
        lang = "🇧🇩 Bangla" if doc_id.startswith("bn_") else "🇬🇧 English"
        preview = documents.get(doc_id, "")[:100]
        print(f"{rank}. {doc_id}")
        print(f"   {lang} | Score: {score:.4f}")
        print(f"   {preview}...")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CLIR Search - Cross-Lingual Information Retrieval"
    )
    parser.add_argument(
        "--lang",
        choices=["bangla", "english", "all"],
        default="all",
        help="Language to search",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of results to return"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bm25", "semantic"],
        help="Models to use (default: bm25 semantic)",
    )
    parser.add_argument("query", nargs="+", help="Search query (required)")

    args = parser.parse_args()

    # Parse models
    valid = {"tfidf", "bm25", "semantic", "hybrid", "fuzzy"}
    if "all" in args.models:
        models = {"bm25", "semantic", "tfidf", "hybrid", "fuzzy"}
    else:
        models = {m.lower() for m in args.models if m.lower() in valid}
        if not models:
            models = {"bm25", "semantic"}

    # Ensure hybrid dependencies
    if "hybrid" in models:
        models.add("bm25")
        models.add("semantic")

    # Load corpus
    all_documents = load_documents()
    if not all_documents:
        print("ERROR: No documents loaded")
        return

    # Filter by language
    if args.lang == "bangla":
        documents = {k: v for k, v in all_documents.items() if k.startswith("bn_")}
        print(f"\n🇧🇩 Filtering: {len(documents)} Bangla documents")
    elif args.lang == "english":
        documents = {k: v for k, v in all_documents.items() if k.startswith("en_")}
        print(f"\n🇬🇧 Filtering: {len(documents)} English documents")
    else:
        documents = all_documents
        print(f"\n🌍 Using all {len(documents)} documents")

    if not documents:
        print(f"ERROR: No {args.lang} documents found")
        return

    # Build indexes
    indexes = build_indexes(documents, models)

    # Search (command-line only)
    if not args.query:
        print("\nERROR: Query required")
        print('Usage: python search.py "your query here"')
        print('       python search.py --models bm25 tfidf --limit 20 "query"')
        return

    query = " ".join(args.query)
    search(query, documents, indexes, models, top_k=args.limit)


if __name__ == "__main__":
    main()
