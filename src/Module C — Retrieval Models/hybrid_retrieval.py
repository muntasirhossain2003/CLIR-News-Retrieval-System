"""
Module C - Model 4: Hybrid Ranking with Weighted Fusion

PURPOSE:
Combine lexical, semantic, and fuzzy retrieval scores using weighted fusion.
This allows leveraging the strengths of different retrieval methods:
- Lexical: Exact keyword matching
- Semantic: Meaning-based cross-lingual retrieval
- Fuzzy: Spelling variation tolerance

APPROACH:
- Score normalization (min-max scaling)
- Weighted linear combination
- Configurable weights for experimentation

This demonstrates how different retrieval signals complement each other.
"""

import logging
from typing import List, Tuple, Dict, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_scores(
    results: List[Tuple[str, float]], method: str = "minmax"
) -> Dict[str, float]:
    """
    Normalize retrieval scores to [0, 1] range.

    Args:
        results: List of (doc_id, score) tuples
        method: Normalization method - 'minmax' or 'standard'

    Returns:
        Dictionary mapping doc_id -> normalized_score

    Methods:
        - 'minmax': Scale to [0, 1] using (score - min) / (max - min)
        - 'standard': Z-score normalization, then sigmoid to [0, 1]

    Example:
        >>> results = [('doc1', 0.8), ('doc2', 0.5), ('doc3', 0.2)]
        >>> normalized = normalize_scores(results, method='minmax')
        >>> # doc1: 1.0, doc2: 0.5, doc3: 0.0

    Notes:
        - Returns empty dict if results is empty
        - Handles single-result case (all scores become 1.0)
        - Preserves only positive scores by default
    """
    if not results:
        return {}

    # Extract scores
    doc_ids = [doc_id for doc_id, _ in results]
    scores = np.array([score for _, score in results])

    if len(scores) == 1:
        # Single result: assign max score
        return {doc_ids[0]: 1.0}

    if method == "minmax":
        # Min-max normalization to [0, 1]
        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            # All scores are equal
            normalized = np.ones_like(scores)
        else:
            normalized = (scores - min_score) / (max_score - min_score)

    elif method == "standard":
        # Z-score normalization, then sigmoid
        mean_score = scores.mean()
        std_score = scores.std()

        if std_score == 0:
            normalized = np.ones_like(scores)
        else:
            z_scores = (scores - mean_score) / std_score
            # Sigmoid to map to [0, 1]
            normalized = 1 / (1 + np.exp(-z_scores))

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Create mapping
    return {
        doc_id: float(norm_score) for doc_id, norm_score in zip(doc_ids, normalized)
    }


def combine_scores(
    score_dicts: List[Dict[str, float]],
    weights: Optional[List[float]] = None,
    aggregation: str = "weighted_sum",
) -> Dict[str, float]:
    """
    Combine multiple score dictionaries using weighted aggregation.

    Args:
        score_dicts: List of dictionaries mapping doc_id -> score
        weights: List of weights for each score dict (default: equal weights)
        aggregation: Aggregation method - 'weighted_sum', 'max', 'min', 'avg'

    Returns:
        Dictionary mapping doc_id -> combined_score

    Aggregation Methods:
        - 'weighted_sum': Σ(weight_i × score_i)
        - 'max': max(scores) across all methods
        - 'min': min(scores) across all methods
        - 'avg': average of scores (ignores weights)

    Example:
        >>> lexical_scores = {'doc1': 0.8, 'doc2': 0.5}
        >>> semantic_scores = {'doc1': 0.6, 'doc3': 0.9}
        >>> combined = combine_scores(
        ...     [lexical_scores, semantic_scores],
        ...     weights=[0.6, 0.4],
        ...     aggregation='weighted_sum'
        ... )
        >>> # doc1: 0.8*0.6 + 0.6*0.4 = 0.72
        >>> # doc2: 0.5*0.6 + 0.0*0.4 = 0.30
        >>> # doc3: 0.0*0.6 + 0.9*0.4 = 0.36

    Notes:
        - Documents missing from a score dict get score 0.0
        - Weights are normalized to sum to 1.0
        - All documents from all dicts are included in output
    """
    if not score_dicts:
        return {}

    # Default: equal weights
    if weights is None:
        weights = [1.0 / len(score_dicts)] * len(score_dicts)
    else:
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

    if len(weights) != len(score_dicts):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match number of score dicts ({len(score_dicts)})"
        )

    # Collect all document IDs
    all_doc_ids = set()
    for score_dict in score_dicts:
        all_doc_ids.update(score_dict.keys())

    # Combine scores
    combined = {}

    for doc_id in all_doc_ids:
        scores = [score_dict.get(doc_id, 0.0) for score_dict in score_dicts]

        if aggregation == "weighted_sum":
            combined[doc_id] = sum(w * s for w, s in zip(weights, scores))

        elif aggregation == "max":
            combined[doc_id] = max(scores)

        elif aggregation == "min":
            combined[doc_id] = min(scores)

        elif aggregation == "avg":
            combined[doc_id] = sum(scores) / len(scores)

        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    return combined


def hybrid_rank(
    lexical_results: List[Tuple[str, float]],
    semantic_results: List[Tuple[str, float]],
    fuzzy_results: Optional[List[Tuple[str, float]]] = None,
    weights: Optional[Dict[str, float]] = None,
    top_k: int = 10,
    normalization: str = "minmax",
    aggregation: str = "weighted_sum",
) -> List[Tuple[str, float]]:
    """
    Combine lexical, semantic, and optionally fuzzy retrieval results using hybrid ranking.

    Args:
        lexical_results: Results from TF-IDF or BM25 retrieval
        semantic_results: Results from semantic retrieval
        fuzzy_results: Optional results from fuzzy matching
        weights: Dictionary with keys 'lexical', 'semantic', 'fuzzy' (default: equal)
        top_k: Number of top results to return (default: 10)
        normalization: Score normalization method - 'minmax' or 'standard'
        aggregation: Combination method - 'weighted_sum', 'max', 'min', 'avg'

    Returns:
        List of (doc_id, combined_score) tuples, sorted by score descending

    Default Weights:
        - If fuzzy_results provided: lexical=0.4, semantic=0.4, fuzzy=0.2
        - If only lexical+semantic: lexical=0.5, semantic=0.5

    Example:
        >>> from tfidf_retrieval import build_tfidf_index, retrieve_tfidf
        >>> from semantic_retrieval import encode_documents, retrieve_semantic
        >>> from hybrid_retrieval import hybrid_rank
        >>>
        >>> # Get results from different methods
        >>> tfidf_results = retrieve_tfidf(query, tfidf_index, top_k=20)
        >>> semantic_results = retrieve_semantic(query, doc_embeddings, top_k=20)
        >>>
        >>> # Combine using hybrid ranking
        >>> hybrid_results = hybrid_rank(
        ...     tfidf_results,
        ...     semantic_results,
        ...     weights={'lexical': 0.6, 'semantic': 0.4},
        ...     top_k=10
        ... )

    Strategy Recommendations:
        - Keyword-heavy queries: Higher lexical weight (0.6-0.7)
        - Cross-lingual queries: Higher semantic weight (0.6-0.7)
        - Noisy queries: Include fuzzy with weight 0.2-0.3
        - Balanced: Equal weights (0.5, 0.5)

    Notes:
        - Scores are normalized before combination
        - Missing documents in one method get score 0.0
        - All unique documents from all methods are considered
        - Re-ranked by combined score
    """
    logger.info("Starting hybrid ranking...")
    logger.info(f"  Lexical results: {len(lexical_results)}")
    logger.info(f"  Semantic results: {len(semantic_results)}")
    if fuzzy_results:
        logger.info(f"  Fuzzy results: {len(fuzzy_results)}")

    # Default weights
    if weights is None:
        if fuzzy_results:
            weights = {"lexical": 0.4, "semantic": 0.4, "fuzzy": 0.2}
        else:
            weights = {"lexical": 0.5, "semantic": 0.5}

    logger.info(f"  Weights: {weights}")
    logger.info(f"  Normalization: {normalization}")
    logger.info(f"  Aggregation: {aggregation}")

    # Step 1: Normalize scores
    lexical_norm = normalize_scores(lexical_results, method=normalization)
    semantic_norm = normalize_scores(semantic_results, method=normalization)

    score_dicts = [lexical_norm, semantic_norm]
    weight_list = [weights.get("lexical", 0.5), weights.get("semantic", 0.5)]

    if fuzzy_results:
        fuzzy_norm = normalize_scores(fuzzy_results, method=normalization)
        score_dicts.append(fuzzy_norm)
        weight_list.append(weights.get("fuzzy", 0.0))

    # Step 2: Combine scores
    combined = combine_scores(score_dicts, weights=weight_list, aggregation=aggregation)

    # Step 3: Sort and return top-K
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    results = sorted_results[:top_k]

    logger.info(f"Hybrid ranking complete: {len(results)} documents")
    if results:
        logger.info(f"  Top score: {results[0][1]:.4f}")
        logger.info(f"  Lowest score: {results[-1][1]:.4f}")

    return results


def analyze_fusion(
    lexical_results: List[Tuple[str, float]],
    semantic_results: List[Tuple[str, float]],
    fuzzy_results: Optional[List[Tuple[str, float]]] = None,
    top_k: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Analyze different fusion strategies by trying multiple weight combinations.

    Args:
        lexical_results: Results from lexical retrieval
        semantic_results: Results from semantic retrieval
        fuzzy_results: Optional fuzzy results
        top_k: Number of top results for each strategy

    Returns:
        Dictionary mapping strategy_name -> results list

    Strategies Tested:
        - 'lexical_only': 100% lexical
        - 'semantic_only': 100% semantic
        - 'balanced': 50-50 lexical-semantic
        - 'lexical_heavy': 70% lexical, 30% semantic
        - 'semantic_heavy': 30% lexical, 70% semantic
        - 'fuzzy_included': With fuzzy matching if provided

    Example:
        >>> strategies = analyze_fusion(tfidf_res, semantic_res)
        >>> for name, results in strategies.items():
        ...     print(f"{name}: {results[:3]}")

    Use Case:
        Experiment to find optimal weights for your dataset and query type.
    """
    logger.info("Analyzing fusion strategies...")

    strategies = {}

    # Pure strategies
    strategies["lexical_only"] = hybrid_rank(
        lexical_results,
        semantic_results,
        fuzzy_results,
        weights={"lexical": 1.0, "semantic": 0.0, "fuzzy": 0.0},
        top_k=top_k,
    )

    strategies["semantic_only"] = hybrid_rank(
        lexical_results,
        semantic_results,
        fuzzy_results,
        weights={"lexical": 0.0, "semantic": 1.0, "fuzzy": 0.0},
        top_k=top_k,
    )

    # Balanced
    strategies["balanced"] = hybrid_rank(
        lexical_results,
        semantic_results,
        fuzzy_results,
        weights={"lexical": 0.5, "semantic": 0.5, "fuzzy": 0.0},
        top_k=top_k,
    )

    # Lexical-heavy
    strategies["lexical_heavy"] = hybrid_rank(
        lexical_results,
        semantic_results,
        fuzzy_results,
        weights={"lexical": 0.7, "semantic": 0.3, "fuzzy": 0.0},
        top_k=top_k,
    )

    # Semantic-heavy (good for cross-lingual)
    strategies["semantic_heavy"] = hybrid_rank(
        lexical_results,
        semantic_results,
        fuzzy_results,
        weights={"lexical": 0.3, "semantic": 0.7, "fuzzy": 0.0},
        top_k=top_k,
    )

    # With fuzzy if provided
    if fuzzy_results:
        strategies["fuzzy_included"] = hybrid_rank(
            lexical_results,
            semantic_results,
            fuzzy_results,
            weights={"lexical": 0.4, "semantic": 0.4, "fuzzy": 0.2},
            top_k=top_k,
        )

    logger.info(f"Analyzed {len(strategies)} fusion strategies")

    return strategies
