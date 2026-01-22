"""
Module D — Ranking, Scoring, & Evaluation

Core evaluation framework for CLIR system including:
- Ranking and scoring with confidence metrics
- Query execution time tracking
- Standard IR evaluation metrics (Precision, Recall, nDCG, MRR)
- Error analysis and failure case detection
- Relevance labeling utilities
"""

from .ranking_scorer import RankingScorer, QueryResult
from .evaluation_metrics import EvaluationMetrics
from .error_analysis import ErrorAnalyzer
from .relevance_labeling import RelevanceLabeler

__all__ = [
    "RankingScorer",
    "QueryResult",
    "EvaluationMetrics",
    "ErrorAnalyzer",
    "RelevanceLabeler",
]
