"""
Module C - Retrieval Pipeline (Gateway)

This is the main entry point for the retrieval system.
Orchestrates all retrieval models and provides a unified interface.

Usage:
    from retrieval_pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline()
    pipeline.build_indexes(documents)
    results = pipeline.search("climate change", method="hybrid")

Command Line:
    python retrieval_pipeline.py "climate change" --method hybrid
    python retrieval_pipeline.py "জলবায়ু পরিবর্তন" --method semantic --top-k 20
"""

import os
import sys
import logging
import json
import time
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root directory (two levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_INDEX_DIR = os.path.join(PROJECT_ROOT, "indexes")
METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "metadata.csv")

# Document metadata cache
_doc_metadata_cache = None


def _load_document_metadata():
    """Load document metadata from CSV for displaying titles."""
    global _doc_metadata_cache
    if _doc_metadata_cache is not None:
        return _doc_metadata_cache

    _doc_metadata_cache = {}
    if os.path.exists(METADATA_PATH):
        try:
            import csv

            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Extract doc_id from filename (remove .json extension)
                    doc_id = row.get("filename", "").replace(".json", "")
                    if doc_id:
                        _doc_metadata_cache[doc_id] = {
                            "title": row.get("title", ""),
                            "language": row.get("language", ""),
                            "source": row.get("source", ""),
                            "url": row.get("url", ""),
                        }
            logger.info(f"Loaded metadata for {len(_doc_metadata_cache)} documents")
        except Exception as e:
            logger.warning(f"Could not load document metadata: {e}")
    return _doc_metadata_cache


def get_document_title(doc_id: str) -> str:
    """Get document title from metadata cache."""
    metadata = _load_document_metadata()
    doc_info = metadata.get(doc_id, {})
    return doc_info.get("title", doc_id)


# Import retrieval modules
try:
    from bm25_retrieval import BM25Index, build_bm25_index, retrieve_bm25
except ImportError:
    BM25Index = None
    logger.warning("BM25 retrieval module not available")

try:
    from tfidf_retrieval import TFIDFIndex, build_tfidf_index, retrieve_tfidf
except ImportError:
    TFIDFIndex = None
    logger.warning("TF-IDF retrieval module not available")

try:
    from fuzzy_retrieval import FuzzyMatcher, retrieve_fuzzy, fuzzy_match
except ImportError:
    FuzzyMatcher = None
    logger.warning("Fuzzy retrieval module not available")

try:
    from semantic_retrieval import SemanticIndex, retrieve_semantic
except ImportError:
    SemanticIndex = None
    logger.warning("Semantic retrieval module not available")

try:
    from hybrid_retrieval import HybridRetriever, create_hybrid_retriever
except ImportError:
    HybridRetriever = None
    logger.warning("Hybrid retrieval module not available")


# Try to import query processing from Module B
MODULE_B_AVAILABLE = False
try:
    # Add Module B to path
    module_b_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "Module B — Query Processing & Cross-Lingual Handling",
    )
    if os.path.exists(module_b_path):
        sys.path.insert(0, module_b_path)
        from query_pipeline import process_complete_query

        MODULE_B_AVAILABLE = True
        logger.info("Module B (Query Processing) loaded successfully")
except ImportError as e:
    logger.warning(f"Module B not available: {e}")
    process_complete_query = None


@dataclass
class SearchResult:
    """Unified search result structure."""

    doc_id: str
    score: float
    rank: int
    method: str
    confidence: str = "high"
    warnings: List[str] = field(default_factory=list)
    scores_breakdown: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 4),
            "rank": self.rank,
            "method": self.method,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "scores_breakdown": {
                k: round(v, 4) for k, v in self.scores_breakdown.items()
            },
            "metadata": self.metadata,
        }


class RetrievalPipeline:
    """
    Unified Retrieval Pipeline for CLIR System.

    Provides:
    - Multiple retrieval methods (BM25, TF-IDF, Semantic, Fuzzy, Hybrid)
    - Query preprocessing integration with Module B
    - Index building and persistence
    - Method comparison and analysis

    Architecture:

        User Query
            │
            ▼
    ┌───────────────────┐
    │  Query Processing │  (Module B - optional)
    │  - Language detect│
    │  - Normalization  │
    │  - Translation    │
    └────────┬──────────┘
             │
             ▼
    ┌───────────────────┐
    │ Retrieval Models  │
    │                   │
    │  ┌─────┐ ┌─────┐  │
    │  │BM25 │ │TFIDF│  │  Lexical
    │  └─────┘ └─────┘  │
    │                   │
    │  ┌─────────────┐  │
    │  │  Semantic   │  │  Dense vectors
    │  │ (mE5-large) │  │
    │  └─────────────┘  │
    │                   │
    │  ┌─────────────┐  │
    │  │   Fuzzy +   │  │  Cross-script
    │  │Transliterate│  │
    │  └─────────────┘  │
    │                   │
    │  ┌─────────────┐  │
    │  │   Hybrid    │  │  Score fusion
    │  │  (Combined) │  │
    │  └─────────────┘  │
    └────────┬──────────┘
             │
             ▼
    ┌───────────────────┐
    │  Ranked Results   │
    │  with confidence  │
    │  scores & warnings│
    └───────────────────┘
    """

    SUPPORTED_METHODS = ["bm25", "tfidf", "semantic", "fuzzy", "hybrid", "all"]

    def __init__(
        self,
        index_dir: str = "indexes",
        use_query_processing: bool = True,
        hybrid_weights: Dict[str, float] = None,
    ):
        """
        Initialize retrieval pipeline.

        Args:
            index_dir: Directory for storing/loading indexes
            use_query_processing: Whether to use Module B for query preprocessing
            hybrid_weights: Custom weights for hybrid retrieval
        """
        self.index_dir = index_dir
        self.use_query_processing = use_query_processing and MODULE_B_AVAILABLE
        self.hybrid_weights = hybrid_weights

        # Index instances (lazy loaded)
        self.bm25_index = None
        self.tfidf_index = None
        self.semantic_index = None
        self.fuzzy_matcher = None
        self.hybrid_retriever = None

        # Documents cache
        self.documents = None

        # Ensure index directory exists
        os.makedirs(index_dir, exist_ok=True)

        logger.info(f"RetrievalPipeline initialized (index_dir={index_dir})")
        if self.use_query_processing:
            logger.info("Query preprocessing enabled (Module B)")

    def build_indexes(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "content",
        build_bm25: bool = True,
        build_tfidf: bool = True,
        build_semantic: bool = True,
        build_fuzzy: bool = True,
        save: bool = True,
    ) -> Dict[str, bool]:
        """
        Build all retrieval indexes from documents.

        Args:
            documents: List of document dicts with 'id' and text content
            text_field: Field containing document text
            build_bm25: Build BM25 index
            build_tfidf: Build TF-IDF index
            build_semantic: Build semantic index
            build_fuzzy: Build fuzzy matcher
            save: Save indexes to disk

        Returns:
            Dictionary of build status for each index type
        """
        self.documents = documents
        status = {}

        logger.info(f"Building indexes for {len(documents)} documents...")

        # BM25 Index
        if build_bm25 and BM25Index is not None:
            try:
                logger.info("Building BM25 index...")
                start = time.time()
                self.bm25_index = BM25Index()
                self.bm25_index.build(documents, text_field=text_field)
                if save:
                    self.bm25_index.save(os.path.join(self.index_dir, "bm25_index.pkl"))
                status["bm25"] = True
                logger.info(f"BM25 index built in {time.time()-start:.2f}s")
            except Exception as e:
                logger.error(f"BM25 build failed: {e}")
                status["bm25"] = False

        # TF-IDF Index
        if build_tfidf and TFIDFIndex is not None:
            try:
                logger.info("Building TF-IDF index...")
                start = time.time()
                self.tfidf_index = TFIDFIndex()
                self.tfidf_index.build(documents, text_field=text_field)
                if save:
                    self.tfidf_index.save(
                        os.path.join(self.index_dir, "tfidf_index.pkl")
                    )
                status["tfidf"] = True
                logger.info(f"TF-IDF index built in {time.time()-start:.2f}s")
            except Exception as e:
                logger.error(f"TF-IDF build failed: {e}")
                status["tfidf"] = False

        # Semantic Index (must be pre-built using module1)
        if build_semantic and SemanticIndex is not None:
            try:
                logger.info("Loading semantic index from disk...")
                start = time.time()
                self.semantic_index = SemanticIndex()
                semantic_path = os.path.join(self.index_dir, "semantic")
                if self.semantic_index.load(semantic_path):
                    status["semantic"] = True
                    logger.info(f"Semantic index loaded in {time.time()-start:.2f}s")
                else:
                    logger.warning(
                        "Semantic index not found. Build it first using module1 indexing: "
                        "python -m src.module1_data_acquisition.indexing.build_index"
                    )
                    status["semantic"] = False
            except Exception as e:
                logger.error(f"Semantic load failed: {e}")
                status["semantic"] = False

        # Fuzzy Matcher
        if build_fuzzy and FuzzyMatcher is not None:
            try:
                logger.info("Building fuzzy matcher...")
                start = time.time()
                self.fuzzy_matcher = FuzzyMatcher()
                self.fuzzy_matcher.build(documents)
                status["fuzzy"] = True
                logger.info(f"Fuzzy matcher built in {time.time()-start:.2f}s")
            except Exception as e:
                logger.error(f"Fuzzy build failed: {e}")
                status["fuzzy"] = False

        # Initialize hybrid retriever
        if HybridRetriever is not None:
            self.hybrid_retriever = HybridRetriever(
                bm25_index=self.bm25_index,
                semantic_index=self.semantic_index,
                fuzzy_matcher=self.fuzzy_matcher,
                weights=self.hybrid_weights,
            )
            self.hybrid_retriever.set_documents(documents)
            status["hybrid"] = True

        return status

    def load_indexes(
        self,
        load_bm25: bool = True,
        load_tfidf: bool = True,
        load_semantic: bool = True,
    ) -> Dict[str, bool]:
        """
        Load indexes from disk.

        Args:
            load_bm25: Load BM25 index
            load_tfidf: Load TF-IDF index
            load_semantic: Load semantic index

        Returns:
            Dictionary of load status for each index type
        """
        status = {}

        # BM25
        if load_bm25 and BM25Index is not None:
            try:
                self.bm25_index = BM25Index()
                path = os.path.join(self.index_dir, "bm25_index.pkl")
                status["bm25"] = self.bm25_index.load(path)
            except Exception as e:
                logger.error(f"BM25 load failed: {e}")
                status["bm25"] = False

        # TF-IDF
        if load_tfidf and TFIDFIndex is not None:
            try:
                self.tfidf_index = TFIDFIndex()
                path = os.path.join(self.index_dir, "tfidf_index.pkl")
                status["tfidf"] = self.tfidf_index.load(path)
            except Exception as e:
                logger.error(f"TF-IDF load failed: {e}")
                status["tfidf"] = False

        # Semantic
        if load_semantic and SemanticIndex is not None:
            try:
                self.semantic_index = SemanticIndex()
                path = os.path.join(self.index_dir, "semantic")
                status["semantic"] = self.semantic_index.load(path)
            except Exception as e:
                logger.error(f"Semantic load failed: {e}")
                status["semantic"] = False

        # Initialize hybrid retriever with loaded indexes
        if HybridRetriever is not None:
            self.hybrid_retriever = HybridRetriever(
                bm25_index=self.bm25_index if status.get("bm25") else None,
                semantic_index=self.semantic_index if status.get("semantic") else None,
                fuzzy_matcher=self.fuzzy_matcher,
                weights=self.hybrid_weights,
            )
            status["hybrid"] = True

        return status

    def search(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 10,
        target_lang: str = None,
        preprocess: bool = True,
    ) -> Dict[str, Any]:
        """
        Search documents with specified retrieval method.

        Args:
            query: User's search query
            method: Retrieval method ("bm25", "tfidf", "semantic", "fuzzy", "hybrid", "all")
            top_k: Number of results to return
            target_lang: Target language for cross-lingual search (triggers translation)
            preprocess: Whether to preprocess query with Module B

        Returns:
            Dictionary with query info, results, and timing
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. Use one of {self.SUPPORTED_METHODS}"
            )

        start_time = time.time()

        # Query preprocessing (Module B)
        query_info = {"original": query}
        search_query = query

        if preprocess and self.use_query_processing and process_complete_query:
            try:
                processed = process_complete_query(query, target_lang=target_lang)
                query_info = {
                    "original": query,
                    "language": processed.get("language", "en"),
                    "normalized": processed.get("normalized_query", query),
                    "entities": processed.get("entities", []),
                    "translated": processed.get("translated_query"),
                }

                # Use translated query for cross-lingual search
                if target_lang and processed.get("translated_query"):
                    search_query = processed["translated_query"]
                    query_info["search_query"] = search_query
                else:
                    search_query = processed.get("normalized_query", query)
                    query_info["search_query"] = search_query

            except Exception as e:
                logger.warning(f"Query preprocessing failed: {e}")
                query_info["preprocessing_error"] = str(e)

        # Execute retrieval
        results = []
        method_times = {}

        if method == "all":
            # Run all methods for comparison
            all_results = {}

            for m in ["bm25", "tfidf", "semantic", "fuzzy", "hybrid"]:
                m_start = time.time()
                # Don't preprocess again, already done above
                m_results = self._search_single_method(
                    search_query, m, top_k, preprocess=False
                )
                method_times[m] = (time.time() - m_start) * 1000
                all_results[m] = m_results

            return {
                "query": query_info,
                "method": "all",
                "results_by_method": all_results,
                "timing": {
                    "total_ms": (time.time() - start_time) * 1000,
                    "by_method_ms": method_times,
                },
            }
        else:
            m_start = time.time()
            # Don't preprocess again, already done above
            results = self._search_single_method(
                search_query, method, top_k, preprocess=False
            )
            method_times[method] = (time.time() - m_start) * 1000

        return {
            "query": query_info,
            "method": method,
            "results": results,
            "timing": {
                "total_ms": (time.time() - start_time) * 1000,
                "retrieval_ms": method_times.get(method, 0),
            },
        }

    def _search_single_method(
        self,
        query: str,
        method: str,
        top_k: int,
        preprocess: bool = False,
        target_lang: str = None,
    ) -> List[Dict[str, Any]]:
        """Execute search with a single retrieval method.

        Note: preprocess is False by default because query is already preprocessed
        by the search() method before calling this function.
        """

        if method == "bm25":
            if self.bm25_index is None:
                logger.warning("BM25 index not available")
                return []
            results = self.bm25_index.get_normalized_scores(
                query, top_k=top_k, preprocess=preprocess, target_lang=target_lang
            )
            return [self._format_result(r, "bm25") for r in results]

        elif method == "tfidf":
            if self.tfidf_index is None:
                logger.warning("TF-IDF index not available")
                return []
            results = self.tfidf_index.search(
                query, top_k=top_k, preprocess=preprocess, target_lang=target_lang
            )
            return [self._format_result(r, "tfidf") for r in results]

        elif method == "semantic":
            if self.semantic_index is None:
                logger.warning("Semantic index not available")
                return []
            results = self.semantic_index.search(
                query, top_k=top_k, preprocess=preprocess, target_lang=target_lang
            )
            return [self._format_result(r, "semantic") for r in results]

        elif method == "fuzzy":
            if self.fuzzy_matcher is not None:
                results = self.fuzzy_matcher.search(query, top_k=top_k)
            elif self.documents is not None:
                results = retrieve_fuzzy(query, self.documents, top_k=top_k)
            else:
                logger.warning("Fuzzy matcher not available")
                return []
            return [self._format_result(r, "fuzzy") for r in results]

        elif method == "hybrid":
            if self.hybrid_retriever is None:
                logger.warning("Hybrid retriever not available")
                return []
            results = self.hybrid_retriever.search(
                query, top_k=top_k, preprocess=preprocess, target_lang=target_lang
            )
            return [r.to_dict() for r in results]

        return []

    def _format_result(self, result: Dict, method: str) -> Dict[str, Any]:
        """Format result to standard structure."""
        return {
            "doc_id": result.get("doc_id", ""),
            "score": result.get("score_normalized", result.get("score", 0)),
            "rank": result.get("rank", 0),
            "method": method,
            "confidence": "high",
            "warnings": [],
            "scores_breakdown": {
                method: result.get("score_normalized", result.get("score", 0))
            },
            "metadata": result.get("metadata", {}),
        }

    def compare_methods(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Compare all retrieval methods on a single query.

        Returns analysis of overlap, unique results, and timing.
        """
        results = self.search(query, method="all", top_k=top_k)

        if "results_by_method" not in results:
            return results

        all_results = results["results_by_method"]

        # Analyze overlap
        doc_sets = {
            method: set(r["doc_id"] for r in method_results)
            for method, method_results in all_results.items()
            if method_results
        }

        # Find documents retrieved by all methods
        if doc_sets:
            common = (
                set.intersection(*doc_sets.values()) if len(doc_sets) > 1 else set()
            )
        else:
            common = set()

        analysis = {
            "query": query,
            "methods_compared": list(doc_sets.keys()),
            "results_per_method": {m: len(r) for m, r in all_results.items()},
            "common_to_all": list(common),
            "timing_ms": results.get("timing", {}).get("by_method_ms", {}),
            "unique_by_method": {},
        }

        # Find unique documents per method
        for method, docs in doc_sets.items():
            others = set().union(*(s for m, s in doc_sets.items() if m != method))
            analysis["unique_by_method"][method] = list(docs - others)

        return analysis

    def get_available_methods(self) -> List[str]:
        """Get list of currently available retrieval methods."""
        available = []

        if self.bm25_index is not None:
            available.append("bm25")
        if self.tfidf_index is not None:
            available.append("tfidf")
        if self.semantic_index is not None:
            available.append("semantic")
        if self.fuzzy_matcher is not None or self.documents is not None:
            available.append("fuzzy")
        if self.hybrid_retriever is not None:
            available.append("hybrid")

        return available


def main():
    """Command line interface for retrieval pipeline."""
    parser = argparse.ArgumentParser(
        description="CLIR Retrieval Pipeline - Unified search interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  bm25      - BM25 lexical retrieval (keyword matching)
  tfidf     - TF-IDF retrieval (term frequency weighting)
  semantic  - Semantic retrieval with mE5-large-instruct (meaning-based)
  fuzzy     - Fuzzy + transliteration matching (cross-script)
  hybrid    - Combined retrieval (BM25 + Semantic + Fuzzy)
  all       - Run all methods and compare

Examples:
  # Basic search with hybrid method (recommended)
  python retrieval_pipeline.py "climate change"
  
  # Semantic search for cross-lingual
  python retrieval_pipeline.py "জলবায়ু পরিবর্তন" --method semantic
  
  # Compare all methods
  python retrieval_pipeline.py "COVID-19 vaccine" --method all
  
  # Cross-lingual search (English query -> Bangla documents)
  python retrieval_pipeline.py "climate change" --target-lang bn
  
  # Build indexes from documents
  python retrieval_pipeline.py --build --data data/documents.json
  
  # Search with custom index directory
  python retrieval_pipeline.py "query" --index-dir my_indexes
        """,
    )

    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument(
        "--method",
        "-m",
        default="hybrid",
        choices=RetrievalPipeline.SUPPORTED_METHODS,
        help="Retrieval method (default: hybrid)",
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=10, help="Number of results (default: 10)"
    )
    parser.add_argument(
        "--target-lang",
        "-t",
        choices=["bn", "en"],
        help="Target language for cross-lingual search",
    )
    parser.add_argument(
        "--index-dir",
        "-i",
        default=DEFAULT_INDEX_DIR,
        help=f"Index directory (default: {DEFAULT_INDEX_DIR})",
    )
    parser.add_argument(
        "--build", action="store_true", help="Build indexes from documents"
    )
    parser.add_argument("--data", "-d", help="Path to documents JSON")
    parser.add_argument(
        "--compare", action="store_true", help="Compare all retrieval methods"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--no-preprocess", action="store_true", help="Skip query preprocessing"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = RetrievalPipeline(
        index_dir=args.index_dir, use_query_processing=not args.no_preprocess
    )

    # Build mode
    if args.build:
        if not args.data:
            print("Error: --data required when building indexes")
            exit(1)

        with open(args.data, "r", encoding="utf-8") as f:
            documents = json.load(f)

        status = pipeline.build_indexes(documents)
        print("\nIndex Build Status:")
        for index_type, success in status.items():
            status_str = "✓" if success else "✗"
            print(f"  {status_str} {index_type}")
        exit(0)

    # Search mode
    if not args.query:
        parser.print_help()
        exit(0)

    # Load indexes
    load_status = pipeline.load_indexes()
    available = pipeline.get_available_methods()

    if not available:
        print(
            "Error: No indexes available. Build indexes first with --build --data <file>"
        )
        exit(1)

    if args.method not in available and args.method != "all":
        print(f"Warning: {args.method} not available. Using: {available[0]}")
        args.method = available[0]

    # Execute search
    if args.compare:
        results = pipeline.compare_methods(args.query, top_k=args.top_k)
    else:
        results = pipeline.search(
            args.query,
            method=args.method,
            top_k=args.top_k,
            target_lang=args.target_lang,
        )

    # Output
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'='*60}")
        print(f"Query: {results['query'].get('original', args.query)}")

        if results["query"].get("search_query") and results["query"][
            "search_query"
        ] != results["query"].get("original"):
            print(f"Search Query: {results['query']['search_query']}")

        if results["query"].get("entities"):
            print(f"Entities: {results['query']['entities']}")

        print(f"Method: {results.get('method', args.method)}")
        print(f"Time: {results.get('timing', {}).get('total_ms', 0):.1f}ms")
        print(f"{'='*60}")

        if args.compare or args.method == "all":
            # Comparison output
            if "results_by_method" in results:
                for method, method_results in results["results_by_method"].items():
                    print(f"\n{method.upper()} ({len(method_results)} results):")
                    for r in method_results[:5]:
                        print(f"  [{r['rank']}] {r['doc_id']}: {r['score']:.4f}")
            elif "unique_by_method" in results:
                print("\nMethod Comparison:")
                print(f"  Common to all: {len(results.get('common_to_all', []))}")
                for method, unique in results.get("unique_by_method", {}).items():
                    print(f"  {method} unique: {len(unique)}")
        else:
            # Single method output
            print(f"\nResults:")
            for r in results.get("results", []):
                conf = "⚠️" if r.get("confidence") != "high" else ""
                # Handle different score field names
                score = r.get(
                    "score", r.get("score_normalized", r.get("final_score", 0))
                )
                doc_id = r["doc_id"]
                title = get_document_title(doc_id)

                # Truncate title if too long
                if len(title) > 60:
                    title = title[:57] + "..."

                print(f"  [{r['rank']}] {score:.4f} {conf}")
                print(f"      ID: {doc_id}")
                print(f"      Title: {title}")
                if r.get("warnings"):
                    for w in r["warnings"]:
                        print(f"      ⚠️ {w}")

        print()


if __name__ == "__main__":
    main()
