"""
Hybrid Ranker Module for Cross-Lingual Information Retrieval System.

Combines multiple retrieval scores for hybrid ranking.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class HybridRanker:
    """
    Combine BM25, fuzzy matching, and semantic scores for hybrid ranking.

    Uses weighted fusion to combine multiple scoring methods.

    Example:
        >>> ranker = HybridRanker()
        >>> results = ranker.rank_documents(
        ...     query, docs,
        ...     weights={'bm25': 0.3, 'semantic': 0.5, 'fuzzy': 0.2}
        ... )
        >>> print(results[0])
        {'doc_id': 'doc123', 'score': 0.87, 'confidence': 'high'}
    """

    def __init__(
        self,
        default_weights: Optional[Dict[str, float]] = None,
        low_confidence_threshold: float = 0.20,
    ):
        """
        Initialize the HybridRanker.

        Args:
            default_weights: Default weights for combining scores
            low_confidence_threshold: Threshold for low confidence warning
        """
        self.default_weights = default_weights or {
            "bm25": 0.3,
            "semantic": 0.5,
            "fuzzy": 0.2,
        }
        self.low_confidence_threshold = low_confidence_threshold

        # Validate weights
        total_weight = sum(self.default_weights.values())
        if not (0.99 <= total_weight <= 1.01):
            logger.warning(f"Weights sum to {total_weight}, should be 1.0")

        logger.info(f"HybridRanker initialized with weights: {self.default_weights}")

    def rank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        scores: Dict[str, Dict[str, float]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank documents using hybrid scoring.

        Args:
            query: Original query
            documents: List of document dictionaries
            scores: Dictionary of score type -> {doc_id: score}
                   e.g., {'bm25': {'doc1': 0.5}, 'semantic': {'doc1': 0.8}}
            weights: Optional custom weights (uses default if not provided)

        Returns:
            Ranked list of documents with combined scores
        """
        if not documents:
            return []

        weights = weights or self.default_weights

        # Normalize all score sets to [0, 1]
        normalized_scores = {}
        for score_type, score_dict in scores.items():
            normalized_scores[score_type] = self.normalize_scores(score_dict)

        # Combine scores
        combined = self.combine_scores(
            bm25_scores=normalized_scores.get("bm25", {}),
            semantic_scores=normalized_scores.get("semantic", {}),
            fuzzy_scores=normalized_scores.get("fuzzy", {}),
            weights=weights,
        )

        # Build result list
        results = []
        for doc in documents:
            doc_id = doc.get("doc_id", "")

            if not doc_id or doc_id not in combined:
                continue

            result = doc.copy()
            result["score"] = combined[doc_id]
            result["normalized_score"] = combined[doc_id]

            # Add confidence level
            result["confidence"] = self._get_confidence_level(combined[doc_id])

            # Add breakdown
            result["score_breakdown"] = {
                "bm25": normalized_scores.get("bm25", {}).get(doc_id, 0.0),
                "semantic": normalized_scores.get("semantic", {}).get(doc_id, 0.0),
                "fuzzy": normalized_scores.get("fuzzy", {}).get(doc_id, 0.0),
            }

            results.append(result)

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        # Add low confidence warning
        if results and results[0]["score"] < self.low_confidence_threshold:
            logger.warning(
                f"Low confidence results for query: '{query}' "
                f"(top score: {results[0]['score']:.4f})"
            )

        logger.debug(f"Hybrid ranking produced {len(results)} results")
        return results

    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize scores to [0, 1] range.

        Args:
            scores: Dictionary mapping doc_id to score

        Returns:
            Normalized scores dictionary
        """
        if not scores:
            return {}

        score_values = list(scores.values())
        max_score = max(score_values)
        min_score = min(score_values)
        score_range = max_score - min_score

        if score_range == 0:
            # All scores are the same
            return {
                doc_id: 1.0 if score > 0 else 0.0 for doc_id, score in scores.items()
            }

        normalized = {}
        for doc_id, score in scores.items():
            normalized[doc_id] = (score - min_score) / score_range

        return normalized

    def combine_scores(
        self,
        bm25_scores: Dict[str, float],
        semantic_scores: Dict[str, float],
        fuzzy_scores: Dict[str, float],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Combine multiple score dictionaries using weighted fusion.

        Args:
            bm25_scores: BM25 scores
            semantic_scores: Semantic similarity scores
            fuzzy_scores: Fuzzy matching scores
            weights: Weights for each score type

        Returns:
            Combined scores dictionary
        """
        # Get all document IDs
        all_doc_ids = set()
        all_doc_ids.update(bm25_scores.keys())
        all_doc_ids.update(semantic_scores.keys())
        all_doc_ids.update(fuzzy_scores.keys())

        combined = {}

        for doc_id in all_doc_ids:
            bm25 = bm25_scores.get(doc_id, 0.0)
            semantic = semantic_scores.get(doc_id, 0.0)
            fuzzy = fuzzy_scores.get(doc_id, 0.0)

            # Weighted combination
            score = (
                weights.get("bm25", 0.3) * bm25
                + weights.get("semantic", 0.5) * semantic
                + weights.get("fuzzy", 0.2) * fuzzy
            )

            combined[doc_id] = round(score, 4)

        return combined

    def _get_confidence_level(self, score: float) -> str:
        """
        Determine confidence level based on score.

        Args:
            score: Combined score

        Returns:
            Confidence level: 'high', 'medium', or 'low'
        """
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"

    def reciprocal_rank_fusion(
        self, rankings: List[List[str]], k: int = 60
    ) -> List[str]:
        """
        Combine multiple rankings using Reciprocal Rank Fusion (RRF).

        Args:
            rankings: List of ranked document ID lists
            k: Constant for RRF formula (default: 60)

        Returns:
            Fused ranking (list of doc_ids)
        """
        rrf_scores = {}

        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, 1):
                score = 1.0 / (k + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score

        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in sorted_docs]

    def linear_combination(
        self, score_dicts: List[Dict[str, float]], weights: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Linear combination of multiple score dictionaries.

        Args:
            score_dicts: List of score dictionaries
            weights: Optional weights (equal weights if not provided)

        Returns:
            Combined scores dictionary
        """
        if not score_dicts:
            return {}

        if weights is None:
            weights = [1.0 / len(score_dicts)] * len(score_dicts)

        # Get all document IDs
        all_doc_ids = set()
        for scores in score_dicts:
            all_doc_ids.update(scores.keys())

        combined = {}
        for doc_id in all_doc_ids:
            score = sum(
                weight * scores.get(doc_id, 0.0)
                for weight, scores in zip(weights, score_dicts)
            )
            combined[doc_id] = round(score, 4)

        return combined
