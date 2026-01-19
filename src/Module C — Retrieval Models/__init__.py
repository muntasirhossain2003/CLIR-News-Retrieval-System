"""
Module C — Retrieval Models

This module implements various retrieval models for the CLIR (Cross-Lingual Information Retrieval) system.

Models:
- Model 1A: BM25 Lexical Retrieval (bm25_retrieval.py)
- Model 1B: TF-IDF Lexical Retrieval (tfidf_retrieval.py)
- Model 2: Fuzzy + Transliteration Matching (fuzzy_retrieval.py)
- Model 3: Semantic Retrieval with FAISS (semantic_retrieval.py)
- Model 4: Hybrid Retrieval - Score Fusion (hybrid_retrieval.py)
- Gateway: Unified Retrieval Pipeline (retrieval_pipeline.py)

Usage:
    # Option 1: Use the unified pipeline (recommended)
    from retrieval_pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline()
    pipeline.build_indexes(documents)
    results = pipeline.search("climate change", method="hybrid")

    # Option 2: Use individual retrievers
    from bm25_retrieval import BM25Index, retrieve_bm25
    from semantic_retrieval import SemanticIndex, retrieve_semantic
    from hybrid_retrieval import HybridRetriever
"""

from .bm25_retrieval import (
    BM25Index,
    build_bm25_index,
    retrieve_bm25,
    compare_bm25_queries,
)

from .tfidf_retrieval import (
    TFIDFIndex,
    build_tfidf_index,
    retrieve_tfidf,
    compare_bm25_tfidf,
)

from .fuzzy_retrieval import (
    FuzzyMatcher,
    fuzzy_match,
    retrieve_fuzzy,
    retrieve_fuzzy_per_term,
    levenshtein_similarity,
    ngram_jaccard_similarity,
    get_transliteration_variants,
)

from .semantic_retrieval import (
    SemanticIndex,
    retrieve_semantic,
    encode_query,
)

from .hybrid_retrieval import (
    HybridRetriever,
    RetrievalResult,
    create_hybrid_retriever,
    hybrid_search,
)

from .retrieval_pipeline import RetrievalPipeline, SearchResult

__all__ = [
    # BM25
    "BM25Index",
    "build_bm25_index",
    "retrieve_bm25",
    "compare_bm25_queries",
    # TF-IDF
    "TFIDFIndex",
    "build_tfidf_index",
    "retrieve_tfidf",
    "compare_bm25_tfidf",
    # Fuzzy
    "FuzzyMatcher",
    "fuzzy_match",
    "retrieve_fuzzy",
    "retrieve_fuzzy_per_term",
    "levenshtein_similarity",
    "ngram_jaccard_similarity",
    "get_transliteration_variants",
    # Semantic
    "SemanticIndex",
    "retrieve_semantic",
    "encode_query",
    # Hybrid
    "HybridRetriever",
    "RetrievalResult",
    "create_hybrid_retriever",
    "hybrid_search",
    # Pipeline
    "RetrievalPipeline",
    "SearchResult",
]
