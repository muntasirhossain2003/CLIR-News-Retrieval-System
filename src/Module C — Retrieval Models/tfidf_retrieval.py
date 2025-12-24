"""
Module C - Model 1A: TF-IDF Based Lexical Retrieval

PURPOSE:
Retrieve and rank documents in the same language as the query
using TF-IDF cosine similarity.

This serves as a lexical baseline for comparison with semantic models.
No translation or embeddings are used - pure term-based matching.
"""

import logging
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFIDFIndex:
    """
    TF-IDF index for lexical document retrieval.

    Attributes:
        vectorizer: Trained TfidfVectorizer
        doc_vectors: TF-IDF matrix for all documents
        doc_ids: List of document IDs corresponding to matrix rows
    """

    def __init__(self, vectorizer, doc_vectors, doc_ids):
        self.vectorizer = vectorizer
        self.doc_vectors = doc_vectors
        self.doc_ids = doc_ids


def build_tfidf_index(docs: Dict[str, str]) -> TFIDFIndex:
    """
    Build TF-IDF index from document collection.

    Args:
        docs: Dictionary mapping doc_id -> document_text
              Example: {'doc1': 'climate change impacts...', 'doc2': '...'}

    Returns:
        TFIDFIndex object containing vectorizer, document vectors, and doc IDs

    Raises:
        ValueError: If docs is empty or invalid

    Example:
        >>> docs = {'doc1': 'climate change', 'doc2': 'global warming'}
        >>> index = build_tfidf_index(docs)
        >>> print(f"Indexed {len(index.doc_ids)} documents")
    """
    if not docs:
        raise ValueError("Cannot build index from empty document collection")

    logger.info(f"Building TF-IDF index for {len(docs)} documents...")

    # Extract doc_ids and texts in consistent order
    doc_ids = list(docs.keys())
    doc_texts = [docs[doc_id] for doc_id in doc_ids]

    # Initialize TF-IDF vectorizer
    # - Use L2 normalization (default)
    # - No max_df/min_df filtering to preserve all terms for baseline
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents=None,  # Preserve language-specific characters
        max_features=None,  # No feature limit for baseline
        sublinear_tf=True,  # Use log scaling for term frequency
        use_idf=True,
        smooth_idf=True,
    )

    # Fit vectorizer and transform documents
    try:
        doc_vectors = vectorizer.fit_transform(doc_texts)
        logger.info(f"TF-IDF index built successfully")
        logger.info(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
        logger.info(f"  Document vectors shape: {doc_vectors.shape}")

        return TFIDFIndex(vectorizer, doc_vectors, doc_ids)

    except Exception as e:
        logger.error(f"Error building TF-IDF index: {e}")
        raise


def retrieve_tfidf(
    query: str, index: TFIDFIndex, top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Retrieve top-K documents for query using TF-IDF cosine similarity.

    Args:
        query: Normalized query string (from Module B)
        index: TFIDFIndex object from build_tfidf_index()
        top_k: Number of top documents to retrieve (default: 10)

    Returns:
        List of (doc_id, score) tuples, sorted by score descending
        Scores are cosine similarity values in range [0, 1]

    Example:
        >>> query = "climate change impacts"
        >>> results = retrieve_tfidf(query, index, top_k=5)
        >>> for doc_id, score in results:
        ...     print(f"{doc_id}: {score:.4f}")

    Notes:
        - Returns empty list if query has no matching terms
        - Assumes query is already normalized (Module B)
        - No language translation performed
    """
    if not query or not query.strip():
        logger.warning("Empty query provided, returning empty results")
        return []

    try:
        # Transform query to TF-IDF vector
        query_vector = index.vectorizer.transform([query])

        # Compute cosine similarity with all documents
        similarities = cosine_similarity(query_vector, index.doc_vectors).flatten()

        # Get top-K results
        # argsort returns indices in ascending order, so we reverse with [::-1]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results list with (doc_id, score)
        results = [
            (index.doc_ids[idx], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > 0  # Only include non-zero scores
        ]

        logger.info(f"Retrieved {len(results)} documents for query (top_k={top_k})")
        if results:
            logger.info(f"  Top score: {results[0][1]:.4f}")
            logger.info(f"  Lowest score: {results[-1][1]:.4f}")
        else:
            logger.warning("  No matching documents found")

        return results

    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return []
