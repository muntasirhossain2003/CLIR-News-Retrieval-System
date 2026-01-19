"""
Module C - Model 4: Hybrid Retrieval

This module implements hybrid retrieval that combines:
- BM25 (lexical matching)
- Semantic embeddings (meaning-based matching)
- Fuzzy/transliteration matching (cross-script support)

WHY HYBRID RETRIEVAL?
---------------------
No single retrieval method is perfect for all queries:

- BM25: Excellent for exact keyword matching, fails on synonyms
- Semantic: Great for meaning, may miss specific terms
- Fuzzy: Handles spelling variations, no semantic understanding

Hybrid combines strengths and mitigates weaknesses.

SCORE FUSION STRATEGY:
----------------------
Weighted linear combination:
    final_score = w1 * bm25_norm + w2 * semantic_norm + w3 * fuzzy_norm

Default weights (tunable):
    BM25:     0.3 (precise term matching)
    Semantic: 0.5 (meaning similarity - most important for CLIR)
    Fuzzy:    0.2 (cross-script and typo handling)

Semantic gets highest weight because:
- Most important for cross-lingual retrieval
- Handles vocabulary mismatch (the main CLIR challenge)

NORMALIZATION:
--------------
All scores normalized to [0, 1] before fusion:
- BM25: Min-max normalization
- Semantic: Already normalized (cosine similarity mapped to [0,1])
- Fuzzy: Already in [0, 1]

CONFIDENCE SCORING:
-------------------
Low confidence warnings when:
- Top score < 0.3
- Large gap between hybrid and individual scores
- Conflicting rankings from different methods
"""

import logging
import time
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
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

        Args:
            bm25: Weight for BM25 scores
            semantic: Weight for semantic scores
            fuzzy: Weight for fuzzy scores
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
    ) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval.

        Args:
            query: Search query string
            top_k: Number of results to return
            use_bm25: Enable BM25 retrieval
            use_semantic: Enable semantic retrieval
            use_fuzzy: Enable fuzzy matching
            min_score: Minimum final score threshold
            preprocess: Whether to apply Module B query preprocessing (default: True)
            target_lang: Target language for translation ('bn' or 'en')

        Returns:
            List of RetrievalResult objects with fused scores
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        start_time = time.time()

        # Apply query preprocessing if enabled
        search_query = query
        query_info = {}
        if preprocess and MODULE_B_AVAILABLE and process_complete_query:
            try:
                processed = process_complete_query(query, target_lang=target_lang)
                query_info = processed
                # Use translated query if available, else normalized query
                if target_lang and processed.get("translated_query"):
                    search_query = processed["translated_query"]
                    logger.info(f"Using translated query: {search_query}")
                else:
                    search_query = processed.get("normalized_query", query)
                    logger.info(f"Using normalized query: {search_query}")
            except Exception as e:
                logger.warning(f"Query preprocessing failed: {e}, using original query")
                search_query = query

        # Collect scores from each method
        bm25_scores = {}
        semantic_scores = {}
        fuzzy_scores = {}

        # BM25 retrieval
        if use_bm25 and self.bm25_index is not None:
            try:
                bm25_results = self.bm25_index.get_normalized_scores(
                    search_query,
                    top_k=top_k * 2,
                    preprocess=False,  # Already preprocessed
                )
                for r in bm25_results:
                    bm25_scores[r["doc_id"]] = r.get("score_normalized", r["score"])
                logger.debug(f"BM25: {len(bm25_results)} results")
            except Exception as e:
                logger.warning(f"BM25 retrieval failed: {e}")

        # Semantic retrieval
        if use_semantic and self.semantic_index is not None:
            try:
                semantic_results = self.semantic_index.search(
                    search_query,
                    top_k=top_k * 2,
                    preprocess=False,  # Already preprocessed
                )
                for r in semantic_results:
                    semantic_scores[r["doc_id"]] = r.get("score_normalized", r["score"])
                logger.debug(f"Semantic: {len(semantic_results)} results")
            except Exception as e:
                logger.warning(f"Semantic retrieval failed: {e}")

        # Fuzzy retrieval
        if use_fuzzy:
            try:
                if self.fuzzy_matcher is not None:
                    fuzzy_results = self.fuzzy_matcher.search(
                        search_query, top_k=top_k * 2
                    )
                elif self.documents is not None:
                    from fuzzy_retrieval import retrieve_fuzzy

                    fuzzy_results = retrieve_fuzzy(
                        search_query,
                        self.documents,
                        text_field="title",
                        top_k=top_k * 2,
                    )
                else:
                    fuzzy_results = []

                for r in fuzzy_results:
                    fuzzy_scores[r["doc_id"]] = r.get("score_normalized", r["score"])
                logger.debug(f"Fuzzy: {len(fuzzy_results)} results")
            except Exception as e:
                logger.warning(f"Fuzzy retrieval failed: {e}")

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
        """
        Assess confidence level and generate warnings.

        Args:
            final_score: Fused score
            bm25_score: BM25 component score
            semantic_score: Semantic component score
            fuzzy_score: Fuzzy component score

        Returns:
            Tuple of (confidence_level, list_of_warnings)
        """
        warnings = []

        # Very low overall score
        if final_score < VERY_LOW_CONFIDENCE_THRESHOLD:
            warnings.append(f"Very low relevance score ({final_score:.3f})")
            return "very_low", warnings

        if final_score < LOW_CONFIDENCE_THRESHOLD:
            warnings.append(f"Low relevance score ({final_score:.3f})")

        # Check for conflicting signals
        active_scores = [s for s in [bm25_score, semantic_score, fuzzy_score] if s > 0]

        if len(active_scores) >= 2:
            max_score = max(active_scores)
            min_score = min(active_scores)

            if max_score - min_score > 0.5:
                warnings.append("Conflicting relevance signals from different methods")

        # Only one method returned results
        if len(active_scores) == 1:
            if semantic_score > 0 and bm25_score == 0:
                warnings.append(
                    "Match based on semantic similarity only (no lexical overlap)"
                )
            elif bm25_score > 0 and semantic_score == 0:
                warnings.append(
                    "Match based on keyword only (semantic model unavailable)"
                )

        # Determine confidence level
        if warnings:
            if len(warnings) > 1 or final_score < LOW_CONFIDENCE_THRESHOLD:
                return "low", warnings
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
Examples:
  # Search with existing indexes
  python hybrid_retrieval.py "climate change" \\
      --bm25-index indexes/bm25_index.pkl \\
      --semantic-index indexes/semantic
  
  # Build and search
  python hybrid_retrieval.py "climate change" \\
      --data data/documents.json --build
  
  # Custom weights
  python hybrid_retrieval.py "climate" \\
      --data data/documents.json --build \\
      --weights 0.2 0.6 0.2
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
