"""
Module C - Model 4: Hybrid Retrieval - Score Fusion
"""

import logging
import time
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
import os

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Try to import Module B query processing
MODULE_B_AVAILABLE = False
try:
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

# Default fusion weights
DEFAULT_WEIGHTS = {"bm25": 0.3, "semantic": 0.5, "fuzzy": 0.2}

# Parallel execution config
ENABLE_PARALLEL = True
MAX_WORKERS = 3

# Confidence thresholds
LOW_CONFIDENCE_THRESHOLD = 0.3
VERY_LOW_CONFIDENCE_THRESHOLD = 0.15


@dataclass
class RetrievalResult:
    """
    Structured retrieval result with scores from all methods.
    """

    doc_id: str
    final_score: float
    rank: int
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    fuzzy_score: float = 0.0
    confidence: str = "high"
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "final_score": round(self.final_score, 4),
            "rank": self.rank,
            "scores": {
                "bm25": round(self.bm25_score, 4),
                "semantic": round(self.semantic_score, 4),
                "fuzzy": round(self.fuzzy_score, 4),
            },
            "confidence": self.confidence,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class HybridRetriever:
    """
    Hybrid Retrieval combining BM25, Semantic, and Fuzzy matching.

    Provides:
    - Configurable fusion weights
    - Score normalization
    - Confidence scoring
    - Low-confidence warnings
    """

    def __init__(
        self,
        bm25_index=None,
        semantic_index=None,
        fuzzy_matcher=None,
        weights: Dict[str, float] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_index: BM25Index instance
            semantic_index: SemanticIndex instance
            fuzzy_matcher: FuzzyMatcher instance or document list for ad-hoc fuzzy
            weights: Fusion weights dict {"bm25": w1, "semantic": w2, "fuzzy": w3}
        """
        self.bm25_index = bm25_index
        self.semantic_index = semantic_index
        self.fuzzy_matcher = fuzzy_matcher
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.documents = None  # For fuzzy matching without pre-built matcher

        # Validate weights sum to 1
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Weights sum to {weight_sum}, normalizing to 1.0")
            for k in self.weights:
                self.weights[k] /= weight_sum

    def set_weights(
        self, bm25: float = None, semantic: float = None, fuzzy: float = None
    ):
        """
        Update fusion weights.
        """
        if bm25 is not None:
            self.weights["bm25"] = bm25
        if semantic is not None:
            self.weights["semantic"] = semantic
        if fuzzy is not None:
            self.weights["fuzzy"] = fuzzy

        # Normalize
        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total

    def set_documents(self, documents: List[Dict[str, Any]]):
        """Set documents for fuzzy matching when no pre-built matcher."""
        self.documents = documents

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_bm25: bool = True,
        use_semantic: bool = True,
        use_fuzzy: bool = True,
        min_score: float = 0.0,
        preprocess: bool = True,
        target_lang: str = None,
        cross_lingual: bool = True,  # NEW: Enable true CLIR
    ) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval with TRUE cross-lingual support.

        For CLIR: Searches with BOTH original query AND translated query,
        then merges results to find documents in both languages.
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        start_time = time.time()

        # Apply query preprocessing
        original_query = query
        translated_query = None
        source_lang = "en"
        query_info = {}

        if preprocess and MODULE_B_AVAILABLE and process_complete_query:
            try:
                # First, detect language and normalize
                processed = process_complete_query(query, target_lang=None)
                query_info = processed
                source_lang = processed.get("language", "en")
                original_query = processed.get("normalized_query", query)

                # For CLIR: Always translate to the OTHER language
                if cross_lingual:
                    other_lang = "bn" if source_lang == "en" else "en"
                    logger.warning(
                        f"🌐 CLIR ENABLED: Detected {source_lang}, will translate to {other_lang}"
                    )
                    processed_with_translation = process_complete_query(
                        query, target_lang=other_lang
                    )
                    translated_query = processed_with_translation.get(
                        "translated_query"
                    )
                    if translated_query and translated_query.strip():
                        logger.warning(
                            f"✓ CLIR Translation: '{original_query}' ({source_lang}) -> '{translated_query}' ({other_lang})"
                        )
                    else:
                        logger.warning(
                            f"✗ CLIR Translation FAILED - translated_query is empty or None"
                        )
                        translated_query = None

            except Exception as e:
                logger.warning(f"Query preprocessing failed: {e}")
                original_query = query

        # Collect scores from each method
        bm25_scores = {}
        semantic_scores = {}
        fuzzy_scores = {}

        # === BM25: Search with BOTH queries for true CLIR ===
        if use_bm25 and self.bm25_index is not None:
            try:
                # Search with original query
                bm25_results_original = self.bm25_index.get_normalized_scores(
                    original_query, top_k * 2, False
                )
                for r in bm25_results_original:
                    doc_id = r["doc_id"]
                    score = r.get("score_normalized", r["score"])
                    bm25_scores[doc_id] = max(bm25_scores.get(doc_id, 0), score)

                logger.warning(
                    f"BM25 original query '{original_query}': {len(bm25_results_original)} results"
                )

                # CLIR: Also search with translated query
                if cross_lingual and translated_query:
                    logger.warning(
                        f"BM25 searching with translated query: '{translated_query}'"
                    )
                    bm25_results_translated = self.bm25_index.get_normalized_scores(
                        translated_query, top_k * 2, False
                    )
                    logger.warning(
                        f"BM25 translated query results: {len(bm25_results_translated)} docs"
                    )
                    for r in bm25_results_translated:
                        doc_id = r["doc_id"]
                        score = r.get("score_normalized", r["score"])
                        # Take max score from either query
                        bm25_scores[doc_id] = max(bm25_scores.get(doc_id, 0), score)
                else:
                    logger.warning(
                        f"BM25 skipped translated query: cross_lingual={cross_lingual}, translated_query={'None' if not translated_query else 'exists'}"
                    )

                logger.warning(
                    f"BM25 TOTAL: {len(bm25_scores)} unique candidates from dual-query search"
                )
            except Exception as e:
                logger.warning(f"BM25 failed: {e}")

        # === Semantic: Multilingual embeddings handle cross-lingual naturally ===
        # But we can boost by searching with both queries
        if use_semantic and self.semantic_index is not None:
            try:
                # Semantic search with original query
                semantic_results = self.semantic_index.search(
                    original_query, top_k * 2, 0.0, False
                )
                for r in semantic_results:
                    doc_id = r["doc_id"]
                    score = r.get("score_normalized", r["score"])
                    semantic_scores[doc_id] = max(semantic_scores.get(doc_id, 0), score)

                # CLIR: Also search with translated query for better coverage
                if cross_lingual and translated_query:
                    semantic_results_translated = self.semantic_index.search(
                        translated_query, top_k * 2, 0.0, False
                    )
                    for r in semantic_results_translated:
                        doc_id = r["doc_id"]
                        score = r.get("score_normalized", r["score"])
                        # Boost if found by both queries
                        if doc_id in semantic_scores:
                            semantic_scores[doc_id] = min(
                                1.0, semantic_scores[doc_id] * 1.2
                            )
                        else:
                            semantic_scores[doc_id] = score

                logger.info(f"Semantic: {len(semantic_scores)} candidates")
            except Exception as e:
                logger.warning(f"Semantic failed: {e}")

        # === Fuzzy: Search with both queries ===
        if use_fuzzy and self.fuzzy_matcher is not None:
            try:
                fuzzy_results = self.fuzzy_matcher.search(original_query, top_k * 2)
                for r in fuzzy_results:
                    doc_id = r["doc_id"]
                    score = r.get("score_normalized", r["score"])
                    fuzzy_scores[doc_id] = max(fuzzy_scores.get(doc_id, 0), score)

                if cross_lingual and translated_query:
                    fuzzy_results_translated = self.fuzzy_matcher.search(
                        translated_query, top_k * 2
                    )
                    for r in fuzzy_results_translated:
                        doc_id = r["doc_id"]
                        score = r.get("score_normalized", r["score"])
                        fuzzy_scores[doc_id] = max(fuzzy_scores.get(doc_id, 0), score)
            except Exception as e:
                logger.warning(f"Fuzzy failed: {e}")

        # Get all candidate documents
        all_doc_ids = (
            set(bm25_scores.keys())
            | set(semantic_scores.keys())
            | set(fuzzy_scores.keys())
        )

        if not all_doc_ids:
            logger.warning("No results from any retrieval method")
            return []

        # Calculate fused scores
        results = []

        for doc_id in all_doc_ids:
            bm25_s = bm25_scores.get(doc_id, 0.0)
            semantic_s = semantic_scores.get(doc_id, 0.0)
            fuzzy_s = fuzzy_scores.get(doc_id, 0.0)

            # Weighted fusion
            final_score = (
                self.weights["bm25"] * bm25_s
                + self.weights["semantic"] * semantic_s
                + self.weights["fuzzy"] * fuzzy_s
            )

            # Confidence assessment
            confidence, warnings = self._assess_confidence(
                final_score, bm25_s, semantic_s, fuzzy_s
            )

            if final_score >= min_score:
                results.append(
                    RetrievalResult(
                        doc_id=doc_id,
                        final_score=final_score,
                        rank=0,  # Will be set after sorting
                        bm25_score=bm25_s,
                        semantic_score=semantic_s,
                        fuzzy_score=fuzzy_s,
                        confidence=confidence,
                        warnings=warnings,
                    )
                )

        # Sort by final score and assign ranks
        results.sort(key=lambda x: x.final_score, reverse=True)
        results = results[:top_k]

        for i, r in enumerate(results, 1):
            r.rank = i

        elapsed = time.time() - start_time
        logger.info(
            f"Hybrid search completed in {elapsed*1000:.2f}ms ({len(results)} results)"
        )

        return results

    def _assess_confidence(
        self,
        final_score: float,
        bm25_score: float,
        semantic_score: float,
        fuzzy_score: float,
    ) -> Tuple[str, List[str]]:
        """Assess confidence level and generate warnings."""
        warnings = []

        # Very low overall score
        if final_score < VERY_LOW_CONFIDENCE_THRESHOLD:
            warnings.append(f"Very low relevance ({final_score:.3f})")
            return "very_low", warnings

        if final_score < LOW_CONFIDENCE_THRESHOLD:
            warnings.append(f"Low relevance ({final_score:.3f})")

        # Check for conflicting signals
        active_scores = [s for s in [bm25_score, semantic_score, fuzzy_score] if s > 0]

        if len(active_scores) >= 2:
            max_score = max(active_scores)
            min_score = min(active_scores)
            if max_score - min_score > 0.5:
                warnings.append("Conflicting relevance signals")

        # Check lexical overlap ONLY when BM25 enabled but zero score
        # (BM25=0 + Semantic>0 means truly no keyword overlap)
        if len(active_scores) == 1:
            if semantic_score > 0 and bm25_score == 0:
                # Only warn if BM25 was actually attempted (index exists)
                warnings.append("Semantic match only (no keyword overlap)")
            elif bm25_score > 0 and semantic_score == 0:
                warnings.append("Keyword match only (semantic unavailable)")

        # Confidence level
        if len(warnings) > 1 or final_score < LOW_CONFIDENCE_THRESHOLD:
            return "low", warnings
        elif warnings:
            return "medium", warnings
        return "high", []

    def search_with_analysis(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Perform hybrid search with detailed analysis.

        Returns comprehensive results including method comparison.
        """
        results = self.search(query, top_k=top_k)

        # Gather statistics
        bm25_only = sum(
            1 for r in results if r.bm25_score > 0 and r.semantic_score == 0
        )
        semantic_only = sum(
            1 for r in results if r.semantic_score > 0 and r.bm25_score == 0
        )
        both = sum(1 for r in results if r.bm25_score > 0 and r.semantic_score > 0)

        low_confidence = sum(1 for r in results if r.confidence in ["low", "very_low"])

        return {
            "query": query,
            "total_results": len(results),
            "results": [r.to_dict() for r in results],
            "analysis": {
                "bm25_only_matches": bm25_only,
                "semantic_only_matches": semantic_only,
                "both_methods_matches": both,
                "low_confidence_count": low_confidence,
                "weights_used": self.weights.copy(),
            },
        }


def create_hybrid_retriever(
    documents: List[Dict[str, Any]] = None,
    bm25_index_path: str = None,
    semantic_index_path: str = None,
    weights: Dict[str, float] = None,
    build_indexes: bool = False,
    text_field: str = "content",
) -> HybridRetriever:
    """
    Factory function to create a HybridRetriever.

    Args:
        documents: List of documents (required if build_indexes=True)
        bm25_index_path: Path to load/save BM25 index
        semantic_index_path: Path to load/save semantic index
        weights: Fusion weights
        build_indexes: Whether to build indexes from documents
        text_field: Field containing document text

    Returns:
        Configured HybridRetriever instance
    """
    bm25_index = None
    semantic_index = None
    fuzzy_matcher = None

    if build_indexes and documents:
        # Build BM25 index
        try:
            from bm25_retrieval import BM25Index

            bm25_index = BM25Index()
            bm25_index.build(documents, text_field=text_field)

            if bm25_index_path:
                bm25_index.save(bm25_index_path)
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}")

        # Build semantic index
        try:
            from semantic_retrieval import SemanticIndex

            semantic_index = SemanticIndex()
            semantic_index.build(documents, text_field=text_field)

            if semantic_index_path:
                semantic_index.save(semantic_index_path)
        except Exception as e:
            logger.warning(f"Failed to build semantic index: {e}")

        # Build fuzzy matcher
        try:
            from fuzzy_retrieval import FuzzyMatcher

            fuzzy_matcher = FuzzyMatcher()
            fuzzy_matcher.build(documents)
        except Exception as e:
            logger.warning(f"Failed to build fuzzy matcher: {e}")

    else:
        # Load existing indexes
        if bm25_index_path:
            try:
                from bm25_retrieval import BM25Index

                bm25_index = BM25Index()
                if not bm25_index.load(bm25_index_path):
                    bm25_index = None
            except Exception as e:
                logger.warning(f"Failed to load BM25 index: {e}")

        if semantic_index_path:
            try:
                from semantic_retrieval import SemanticIndex

                semantic_index = SemanticIndex()
                if not semantic_index.load(semantic_index_path):
                    semantic_index = None
            except Exception as e:
                logger.warning(f"Failed to load semantic index: {e}")

    retriever = HybridRetriever(
        bm25_index=bm25_index,
        semantic_index=semantic_index,
        fuzzy_matcher=fuzzy_matcher,
        weights=weights,
    )

    if documents:
        retriever.set_documents(documents)

    return retriever


def hybrid_search(
    query: str, retriever: HybridRetriever, top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Convenience function for hybrid search.

    Args:
        query: Search query
        retriever: HybridRetriever instance
        top_k: Number of results

    Returns:
        List of result dictionaries
    """
    results = retriever.search(query, top_k=top_k)
    return [r.to_dict() for r in results]


# Command line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid Retrieval for CLIR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """,
    )

    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--data", "-d", help="Path to documents JSON")
    parser.add_argument("--build", action="store_true", help="Build indexes from data")
    parser.add_argument("--bm25-index", help="Path to BM25 index")
    parser.add_argument("--semantic-index", help="Path to semantic index")
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of results")
    parser.add_argument(
        "--weights",
        "-w",
        nargs=3,
        type=float,
        metavar=("BM25", "SEMANTIC", "FUZZY"),
        help="Fusion weights (must sum to 1)",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--analyze", action="store_true", help="Include detailed analysis"
    )

    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        exit(0)

    # Load documents if provided
    documents = None
    if args.data:
        with open(args.data, "r", encoding="utf-8") as f:
            documents = json.load(f)

    # Parse weights
    weights = None
    if args.weights:
        weights = {
            "bm25": args.weights[0],
            "semantic": args.weights[1],
            "fuzzy": args.weights[2],
        }

    # Create retriever
    retriever = create_hybrid_retriever(
        documents=documents,
        bm25_index_path=args.bm25_index,
        semantic_index_path=args.semantic_index,
        weights=weights,
        build_indexes=args.build,
    )

    # Search
    if args.analyze:
        output = retriever.search_with_analysis(args.query, top_k=args.top_k)
    else:
        results = retriever.search(args.query, top_k=args.top_k)
        output = [r.to_dict() for r in results]

    if args.json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        if args.analyze:
            print(f"\nHybrid Search Analysis for: {args.query}")
            print("=" * 60)
            print(f"Total results: {output['total_results']}")
            print(f"BM25 only: {output['analysis']['bm25_only_matches']}")
            print(f"Semantic only: {output['analysis']['semantic_only_matches']}")
            print(f"Both methods: {output['analysis']['both_methods_matches']}")
            print(f"Low confidence: {output['analysis']['low_confidence_count']}")
            print(f"\nWeights: {output['analysis']['weights_used']}")
            print("\nResults:")
            for r in output["results"][:10]:
                conf_marker = "⚠️" if r["confidence"] != "high" else "✓"
                print(
                    f"  [{r['rank']}] {r['doc_id']}: {r['final_score']:.4f} {conf_marker}"
                )
                print(
                    f"      BM25={r['scores']['bm25']:.3f} SEM={r['scores']['semantic']:.3f} FUZ={r['scores']['fuzzy']:.3f}"
                )
                if r["warnings"]:
                    for w in r["warnings"]:
                        print(f"      ⚠️ {w}")
        else:
            print(f"\nHybrid Results for: {args.query}")
            print("=" * 50)
            for r in output:
                conf = "⚠️" if r["confidence"] != "high" else ""
                print(f"  [{r['rank']}] {r['doc_id']}: {r['final_score']:.4f} {conf}")
        print()
