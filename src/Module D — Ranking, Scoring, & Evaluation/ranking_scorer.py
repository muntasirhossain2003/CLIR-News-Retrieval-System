"""
Ranking and Scoring Module

Ranks documents, assigns confidence scores, and generates execution time reports.
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class QueryResult:
    """Represents a single query result."""
    doc_id: str
    title: str
    language: str
    source: str
    url: str
    body_preview: str
    score: float  # Normalized to [0, 1]
    ranking_position: int
    retrieval_method: str
    confidence_score: float  # Final matching score [0, 1]


@dataclass
class ExecutionMetrics:
    """Execution time metrics for a query."""
    total_time_ms: float
    translation_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    ranking_time_ms: float = 0.0
    lexical_search_time_ms: float = 0.0
    semantic_search_time_ms: float = 0.0


class RankingScorer:
    """
    Ranks documents, normalizes scores, and generates confidence metrics.
    """

    def __init__(self, low_confidence_threshold: float = 0.20):
        """
        Args:
            low_confidence_threshold: If top result score < this, show warning
        """
        self.low_confidence_threshold = low_confidence_threshold
        self.execution_metrics = None

    def normalize_scores(
        self, scores: Dict[str, float], method: str = "minmax"
    ) -> Dict[str, float]:
        """
        Normalize all scores to [0, 1] range.

        Args:
            scores: Dict mapping doc_id -> score
            method: "minmax" or "sigmoid"

        Returns:
            Normalized scores dict
        """
        if not scores:
            return {}

        score_values = list(scores.values())

        if method == "minmax":
            min_score = min(score_values)
            max_score = max(score_values)

            if max_score == min_score:
                # All scores are same
                return {doc_id: 0.5 for doc_id in scores.keys()}

            normalized = {}
            for doc_id, score in scores.items():
                norm_score = (score - min_score) / (max_score - min_score)
                normalized[doc_id] = max(0.0, min(1.0, norm_score))
            return normalized

        elif method == "sigmoid":
            # Sigmoid normalization for similarity scores
            import math

            normalized = {}
            for doc_id, score in scores.items():
                # Sigmoid: 1 / (1 + e^(-x))
                try:
                    sig_score = 1 / (1 + np.exp(-score))
                    normalized[doc_id] = max(0.0, min(1.0, sig_score))
                except:
                    normalized[doc_id] = 0.5
            return normalized

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def rank_documents(
        self,
        results: Dict[str, Any],
        method: str = "hybrid",
        top_k: int = 10,
        normalize_method: str = "minmax",
    ) -> Tuple[List[QueryResult], float]:
        """
        Rank and score documents from retrieval results.

        Args:
            results: Dict with doc_id -> document info
            method: Retrieval method used (bm25, semantic, hybrid, etc.)
            top_k: Return top-K results
            normalize_method: Normalization strategy

        Returns:
            Tuple of (ranked_results_list, top_confidence_score)
        """
        start_time = time.time()

        if not results:
            return [], 0.0

        # Extract scores from results
        scores = {}
        doc_info = {}

        for doc_id, doc_data in results.items():
            if isinstance(doc_data, dict):
                # Extract score (handle different formats)
                score = doc_data.get("score", 0.0)
                if score is None:
                    score = 0.0
                scores[doc_id] = float(score)
                doc_info[doc_id] = doc_data
            else:
                scores[doc_id] = 0.5
                doc_info[doc_id] = {"title": str(doc_data)}

        # Normalize scores
        normalized_scores = self.normalize_scores(scores, method=normalize_method)

        # Sort by normalized score (descending)
        sorted_docs = sorted(
            normalized_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Build ranked results
        ranked_results = []
        for rank, (doc_id, confidence_score) in enumerate(sorted_docs[:top_k], 1):
            doc_data = doc_info.get(doc_id, {})

            result = QueryResult(
                doc_id=doc_id,
                title=doc_data.get("title", "N/A"),
                language=doc_data.get("language", "unknown"),
                source=doc_data.get("source", "N/A"),
                url=doc_data.get("url", ""),
                body_preview=doc_data.get("body_preview", "")[:150],
                score=float(scores.get(doc_id, 0.0)),
                ranking_position=rank,
                retrieval_method=method,
                confidence_score=confidence_score,
            )
            ranked_results.append(result)

        ranking_time = (time.time() - start_time) * 1000  # ms

        # Store metrics
        self.execution_metrics = ExecutionMetrics(
            total_time_ms=ranking_time, ranking_time_ms=ranking_time
        )

        top_confidence = ranked_results[0].confidence_score if ranked_results else 0.0

        return ranked_results, top_confidence

    def generate_confidence_warning(self, top_confidence: float, query: str) -> Optional[str]:
        """
        Generate low-confidence warning if top result is below threshold.

        Args:
            top_confidence: Confidence score of top result
            query: Original query text

        Returns:
            Warning message if applicable, else None
        """
        if top_confidence < self.low_confidence_threshold:
            return (
                f"⚠️ Warning: Retrieved results may not be relevant. "
                f"Matching confidence is low (score: {top_confidence:.2f}). "
                f"Consider rephrasing your query or checking translation quality."
            )
        return None

    def format_results(self, ranked_results: List[QueryResult]) -> str:
        """
        Format ranked results for display.

        Args:
            ranked_results: List of QueryResult objects

        Returns:
            Formatted string for display
        """
        output = []
        output.append("=" * 80)
        output.append("RANKED RESULTS")
        output.append("=" * 80)

        if not ranked_results:
            output.append("\nNo results found.")
            return "\n".join(output)

        output.append(f"\nTotal Results: {len(ranked_results)}\n")

        for result in ranked_results:
            output.append(f"{result.ranking_position}. {result.title}")
            output.append(f"   Score: {result.score:.4f} | Confidence: {result.confidence_score:.2%}")
            output.append(f"   Language: {result.language} | Source: {result.source}")
            output.append(f"   Method: {result.retrieval_method}")
            output.append(f"   URL: {result.url}")
            if result.body_preview:
                output.append(f"   Preview: {result.body_preview}...")
            output.append("")

        return "\n".join(output)

    def combine_retrieval_scores(
        self,
        bm25_scores: Dict[str, float],
        semantic_scores: Dict[str, float],
        bm25_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> Dict[str, float]:
        """
        Combine scores from multiple retrieval methods (hybrid approach).

        Args:
            bm25_scores: BM25/TF-IDF scores
            semantic_scores: Embedding similarity scores
            bm25_weight: Weight for lexical retrieval
            semantic_weight: Weight for semantic retrieval

        Returns:
            Combined scores dict
        """
        # Normalize both
        norm_bm25 = self.normalize_scores(bm25_scores, method="minmax") if bm25_scores else {}
        norm_semantic = self.normalize_scores(semantic_scores, method="minmax") if semantic_scores else {}

        # Get all unique doc_ids
        all_docs = set(norm_bm25.keys()) | set(norm_semantic.keys())

        combined = {}
        for doc_id in all_docs:
            bm25_score = norm_bm25.get(doc_id, 0.0)
            semantic_score = norm_semantic.get(doc_id, 0.0)
            combined[doc_id] = (bm25_weight * bm25_score) + (semantic_weight * semantic_score)

        return combined

    def set_execution_metrics(self, metrics: ExecutionMetrics):
        """Set execution time metrics."""
        self.execution_metrics = metrics

    def format_execution_metrics(self) -> str:
        """Format execution metrics for display."""
        if not self.execution_metrics:
            return "No execution metrics available."

        metrics = self.execution_metrics
        output = []
        output.append("\n⏱️ EXECUTION TIME BREAKDOWN")
        output.append("─" * 40)
        output.append(f"Total Retrieval Time: {metrics.total_time_ms:.2f} ms")

        if metrics.translation_time_ms > 0:
            output.append(f"  └─ Translation: {metrics.translation_time_ms:.2f} ms")

        if metrics.lexical_search_time_ms > 0:
            output.append(f"  └─ Lexical Search: {metrics.lexical_search_time_ms:.2f} ms")

        if metrics.semantic_search_time_ms > 0:
            output.append(f"  └─ Semantic Search: {metrics.semantic_search_time_ms:.2f} ms")

        if metrics.embedding_time_ms > 0:
            output.append(f"  └─ Embedding Computation: {metrics.embedding_time_ms:.2f} ms")

        if metrics.ranking_time_ms > 0:
            output.append(f"  └─ Ranking: {metrics.ranking_time_ms:.2f} ms")

        return "\n".join(output)
