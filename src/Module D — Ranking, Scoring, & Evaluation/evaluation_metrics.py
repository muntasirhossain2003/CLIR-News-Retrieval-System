"""
Evaluation Metrics Module

Implements standard Information Retrieval metrics:
- Precision@K
- Recall@K
- nDCG@K (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
"""

from typing import List, Dict, Tuple
import math
import numpy as np


class EvaluationMetrics:
    """
    Calculates standard IR evaluation metrics for ranked retrieval results.
    """

    @staticmethod
    def precision_at_k(relevant_docs: List[str], retrieved_docs: List[str], k: int = 10) -> float:
        """
        Calculate Precision@K.

        Precision@K = (# relevant docs in top-K) / K

        Args:
            relevant_docs: List of document IDs that are relevant to query
            retrieved_docs: List of retrieved document IDs (in rank order)
            k: Cutoff position

        Returns:
            Precision@K score [0, 1]
        """
        if k <= 0:
            return 0.0

        retrieved_at_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)

        num_relevant_retrieved = len(retrieved_at_k & relevant_set)
        return num_relevant_retrieved / k

    @staticmethod
    def recall_at_k(relevant_docs: List[str], retrieved_docs: List[str], k: int = 50) -> float:
        """
        Calculate Recall@K.

        Recall@K = (# relevant docs in top-K) / (total # relevant docs)

        Args:
            relevant_docs: List of document IDs that are relevant to query
            retrieved_docs: List of retrieved document IDs (in rank order)
            k: Cutoff position

        Returns:
            Recall@K score [0, 1]
        """
        if not relevant_docs or len(relevant_docs) == 0:
            return 0.0

        retrieved_at_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)

        num_relevant_retrieved = len(retrieved_at_k & relevant_set)
        return num_relevant_retrieved / len(relevant_set)

    @staticmethod
    def average_precision(relevant_docs: List[str], retrieved_docs: List[str]) -> float:
        """
        Calculate Average Precision (AP).

        AP = (1/R) * Σ(Precision@k * rel(k))
        where R = total relevant docs, rel(k) = 1 if rank k is relevant, 0 otherwise

        Args:
            relevant_docs: List of document IDs that are relevant to query
            retrieved_docs: List of retrieved document IDs (in rank order)

        Returns:
            AP score [0, 1]
        """
        if not relevant_docs or len(relevant_docs) == 0:
            return 0.0

        relevant_set = set(relevant_docs)
        score = 0.0
        num_hits = 0

        for k, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_set:
                num_hits += 1
                precision_at_k = num_hits / k
                score += precision_at_k

        return score / len(relevant_set)

    @staticmethod
    def reciprocal_rank(relevant_docs: List[str], retrieved_docs: List[str]) -> float:
        """
        Calculate Reciprocal Rank (RR).

        RR = 1 / (rank of first relevant document)
        Returns 0 if no relevant document found.

        Args:
            relevant_docs: List of document IDs that are relevant to query
            retrieved_docs: List of retrieved document IDs (in rank order)

        Returns:
            RR score [0, 1]
        """
        if not relevant_docs or len(relevant_docs) == 0:
            return 0.0

        relevant_set = set(relevant_docs)

        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_set:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def dcg(relevant_scores: List[float], k: int = 10) -> float:
        """
        Calculate Discounted Cumulative Gain (DCG).

        DCG@K = Σ(rel_i / log₂(i+1)) for i in 1..K
        where rel_i is the relevance score at position i

        Args:
            relevant_scores: List of relevance scores (usually 0 or 1 for binary relevance)
            k: Cutoff position

        Returns:
            DCG score
        """
        dcg_score = 0.0

        for i, rel in enumerate(relevant_scores[:k], 1):
            dcg_score += rel / math.log2(i + 1)

        return dcg_score

    @staticmethod
    def idcg(num_relevant: int, k: int = 10) -> float:
        """
        Calculate Ideal Discounted Cumulative Gain (IDCG).

        IDCG = DCG for ideal ranking (all relevant docs at top)

        Args:
            num_relevant: Total number of relevant documents
            k: Cutoff position

        Returns:
            IDCG score
        """
        ideal_scores = [1.0] * min(num_relevant, k)
        return EvaluationMetrics.dcg(ideal_scores, k)

    @staticmethod
    def ndcg(
        relevant_docs: List[str], retrieved_docs: List[str], k: int = 10
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (nDCG@K).

        nDCG@K = DCG@K / IDCG@K

        Args:
            relevant_docs: List of document IDs that are relevant to query
            retrieved_docs: List of retrieved document IDs (in rank order)
            k: Cutoff position

        Returns:
            nDCG score [0, 1]
        """
        if not relevant_docs or len(relevant_docs) == 0:
            return 0.0

        relevant_set = set(relevant_docs)

        # Create binary relevance scores
        rel_scores = [1.0 if doc_id in relevant_set else 0.0 for doc_id in retrieved_docs]

        dcg_score = EvaluationMetrics.dcg(rel_scores, k)
        idcg_score = EvaluationMetrics.idcg(len(relevant_set), k)

        if idcg_score == 0:
            return 0.0

        return dcg_score / idcg_score

    @staticmethod
    def mean_average_precision(
        queries_relevant: Dict[str, List[str]], queries_retrieved: Dict[str, List[str]]
    ) -> float:
        """
        Calculate Mean Average Precision (MAP) across multiple queries.

        MAP = (1/Q) * Σ AP_q for all queries

        Args:
            queries_relevant: Dict mapping query_id -> list of relevant doc_ids
            queries_retrieved: Dict mapping query_id -> list of retrieved doc_ids

        Returns:
            MAP score [0, 1]
        """
        if not queries_relevant:
            return 0.0

        aps = []

        for query_id, relevant_docs in queries_relevant.items():
            retrieved_docs = queries_retrieved.get(query_id, [])
            ap = EvaluationMetrics.average_precision(relevant_docs, retrieved_docs)
            aps.append(ap)

        return np.mean(aps) if aps else 0.0

    @staticmethod
    def mean_reciprocal_rank(
        queries_relevant: Dict[str, List[str]], queries_retrieved: Dict[str, List[str]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) across multiple queries.

        MRR = (1/Q) * Σ RR_q for all queries

        Args:
            queries_relevant: Dict mapping query_id -> list of relevant doc_ids
            queries_retrieved: Dict mapping query_id -> list of retrieved doc_ids

        Returns:
            MRR score [0, 1]
        """
        if not queries_relevant:
            return 0.0

        rrs = []

        for query_id, relevant_docs in queries_relevant.items():
            retrieved_docs = queries_retrieved.get(query_id, [])
            rr = EvaluationMetrics.reciprocal_rank(relevant_docs, retrieved_docs)
            rrs.append(rr)

        return np.mean(rrs) if rrs else 0.0

    @staticmethod
    def evaluate_query(
        relevant_docs: List[str],
        retrieved_docs: List[str],
        query_id: str = "",
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation for a single query.

        Args:
            relevant_docs: List of relevant doc_ids
            retrieved_docs: List of retrieved doc_ids (in rank order)
            query_id: Query identifier (for logging)
            k: Cutoff position for metrics

        Returns:
            Dict with metric names and scores
        """
        metrics = {
            "query_id": query_id,
            "precision_at_10": EvaluationMetrics.precision_at_k(relevant_docs, retrieved_docs, 10),
            "precision_at_5": EvaluationMetrics.precision_at_k(relevant_docs, retrieved_docs, 5),
            "recall_at_50": EvaluationMetrics.recall_at_k(relevant_docs, retrieved_docs, 50),
            "recall_at_10": EvaluationMetrics.recall_at_k(relevant_docs, retrieved_docs, 10),
            "ndcg_at_10": EvaluationMetrics.ndcg(relevant_docs, retrieved_docs, 10),
            "average_precision": EvaluationMetrics.average_precision(relevant_docs, retrieved_docs),
            "mrr": EvaluationMetrics.reciprocal_rank(relevant_docs, retrieved_docs),
        }

        return metrics

    @staticmethod
    def evaluate_batch(
        batch_queries: Dict[str, Dict],  # query_id -> {relevant: [...], retrieved: [...]}
        k: int = 10,
    ) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
        """
        Comprehensive evaluation for batch of queries.

        Args:
            batch_queries: Dict mapping query_id -> {relevant: [...], retrieved: [...]}
            k: Cutoff position

        Returns:
            Tuple of (per_query_metrics, aggregate_metrics)
        """
        per_query_results = {}
        aggregates = {
            "precision_at_10": [],
            "precision_at_5": [],
            "recall_at_50": [],
            "recall_at_10": [],
            "ndcg_at_10": [],
            "map": [],
            "mrr": [],
        }

        for query_id, query_data in batch_queries.items():
            relevant = query_data.get("relevant", [])
            retrieved = query_data.get("retrieved", [])

            metrics = EvaluationMetrics.evaluate_query(relevant, retrieved, query_id, k)
            per_query_results[query_id] = metrics

            # Aggregate
            aggregates["precision_at_10"].append(metrics["precision_at_10"])
            aggregates["precision_at_5"].append(metrics["precision_at_5"])
            aggregates["recall_at_50"].append(metrics["recall_at_50"])
            aggregates["recall_at_10"].append(metrics["recall_at_10"])
            aggregates["ndcg_at_10"].append(metrics["ndcg_at_10"])
            aggregates["map"].append(metrics["average_precision"])
            aggregates["mrr"].append(metrics["mrr"])

        # Calculate means
        summary = {
            "mean_precision_at_10": np.mean(aggregates["precision_at_10"]),
            "mean_precision_at_5": np.mean(aggregates["precision_at_5"]),
            "mean_recall_at_50": np.mean(aggregates["recall_at_50"]),
            "mean_recall_at_10": np.mean(aggregates["recall_at_10"]),
            "mean_ndcg_at_10": np.mean(aggregates["ndcg_at_10"]),
            "mean_average_precision": np.mean(aggregates["map"]),
            "mean_reciprocal_rank": np.mean(aggregates["mrr"]),
            "num_queries": len(batch_queries),
        }

        return per_query_results, summary

    @staticmethod
    def format_metrics(metrics: Dict[str, float]) -> str:
        """Format metrics for display."""
        output = []
        output.append("\n📊 EVALUATION METRICS")
        output.append("─" * 50)

        for metric, value in metrics.items():
            if metric != "query_id" and isinstance(value, (int, float)):
                if isinstance(value, float):
                    output.append(f"{metric:.<40} {value:.4f}")
                else:
                    output.append(f"{metric:.<40} {value}")

        return "\n".join(output)
