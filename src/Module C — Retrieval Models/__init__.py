"""
Module C — Retrieval Models

This module implements various retrieval models for the CLIR system.

Models:
- Model 1A: TF-IDF based lexical retrieval (tfidf_retrieval.py)
- Model 1B: BM25 based lexical retrieval (bm25_retrieval.py)
- Model 2: Fuzzy and transliteration-based matching (fuzzy_retrieval.py)
- Model 3: Semantic retrieval with multilingual embeddings (semantic_retrieval.py)
- Model 4: Hybrid ranking with weighted fusion (hybrid_retrieval.py)
"""

from .tfidf_retrieval import build_tfidf_index, retrieve_tfidf, TFIDFIndex
from .bm25_retrieval import build_bm25_index, retrieve_bm25, BM25Index
from .fuzzy_retrieval import (
    fuzzy_match,
    retrieve_fuzzy,
    retrieve_fuzzy_per_term,
)
from .semantic_retrieval import (
    encode_documents,
    retrieve_semantic,
    retrieve_semantic_with_query_embedding,
    compute_similarity,
)
from .hybrid_retrieval import (
    normalize_scores,
    combine_scores,
    hybrid_rank,
    analyze_fusion,
)

__all__ = [
    "build_tfidf_index",
    "retrieve_tfidf",
    "TFIDFIndex",
    "build_bm25_index",
    "retrieve_bm25",
    "BM25Index",
    "fuzzy_match",
    "retrieve_fuzzy",
    "retrieve_fuzzy_per_term",
    "encode_documents",
    "retrieve_semantic",
    "retrieve_semantic_with_query_embedding",
    "compute_similarity",
    "normalize_scores",
    "combine_scores",
    "hybrid_rank",
    "analyze_fusion",
]
