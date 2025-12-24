"""
Module C — Retrieval Models

This module implements various retrieval models for the CLIR system.

Models:
- Model 1A: TF-IDF based lexical retrieval (tfidf_retrieval.py)
- Model 1B: BM25 based lexical retrieval (bm25_retrieval.py)
- Model 2: Fuzzy and transliteration-based matching (fuzzy_retrieval.py)
- Model 3: Semantic retrieval with embeddings (coming soon)
- Model 4: Hybrid retrieval combining lexical + semantic (coming soon)
"""

from .tfidf_retrieval import build_tfidf_index, retrieve_tfidf, TFIDFIndex
from .bm25_retrieval import build_bm25_index, retrieve_bm25, BM25Index
from .fuzzy_retrieval import (
    fuzzy_match,
    retrieve_fuzzy,
    retrieve_fuzzy_per_term,
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
]
