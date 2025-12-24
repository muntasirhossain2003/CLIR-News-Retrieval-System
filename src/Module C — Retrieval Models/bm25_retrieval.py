"""
Module C - Model 1B: BM25 Based Lexical Retrieval

PURPOSE:
Rank documents using BM25 (Best Match 25) scoring algorithm.
BM25 is a probabilistic retrieval function that ranks documents
based on query term occurrences with saturation.

Compared to TF-IDF:
- BM25 has term frequency saturation (prevents over-weighting)
- Better handles document length normalization
- Tunable parameters (k1, b) for different collections

This serves as an improved lexical baseline for CLIR evaluation.
"""

import logging
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 index for lexical document retrieval.

    Attributes:
        bm25: Trained BM25Okapi object
        doc_ids: List of document IDs corresponding to corpus order
        tokenized_corpus: List of tokenized documents
    """

    def __init__(self, bm25, doc_ids, tokenized_corpus):
        self.bm25 = bm25
        self.doc_ids = doc_ids
        self.tokenized_corpus = tokenized_corpus


def build_bm25_index(docs: Dict[str, str]) -> BM25Index:
    """
    Build BM25 index from document collection.

    Args:
        docs: Dictionary mapping doc_id -> document_text
              Example: {'doc1': 'climate change impacts...', 'doc2': '...'}

    Returns:
        BM25Index object containing BM25 model, document IDs, and tokenized corpus

    Raises:
        ValueError: If docs is empty or invalid

    Example:
        >>> docs = {'doc1': 'climate change', 'doc2': 'global warming'}
        >>> index = build_bm25_index(docs)
        >>> print(f"Indexed {len(index.doc_ids)} documents")

    Notes:
        - Uses simple whitespace tokenization (no stemming)
        - Default BM25 parameters: k1=1.5, b=0.75
        - Lowercase normalization applied
    """
    if not docs:
        raise ValueError("Cannot build index from empty document collection")

    logger.info(f"Building BM25 index for {len(docs)} documents...")

    # Extract doc_ids and texts in consistent order
    doc_ids = list(docs.keys())
    doc_texts = [docs[doc_id] for doc_id in doc_ids]

    # Tokenize documents: lowercase + whitespace split
    # No stemming to keep scoring explainable
    tokenized_corpus = [doc.lower().split() for doc in doc_texts]

    try:
        # Initialize BM25Okapi with default parameters
        # k1=1.5: Controls term frequency saturation
        # b=0.75: Controls document length normalization
        bm25 = BM25Okapi(tokenized_corpus)

        logger.info(f"BM25 index built successfully")
        logger.info(f"  Number of documents: {len(tokenized_corpus)}")
        logger.info(f"  Average document length: {bm25.avgdl:.2f} tokens")

        # Calculate vocabulary size
        vocab = set()
        for tokens in tokenized_corpus:
            vocab.update(tokens)
        logger.info(f"  Vocabulary size: {len(vocab)} unique tokens")

        return BM25Index(bm25, doc_ids, tokenized_corpus)

    except Exception as e:
        logger.error(f"Error building BM25 index: {e}")
        raise


def retrieve_bm25(
    query: str, index: BM25Index, top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Retrieve top-K documents for query using BM25 scoring.

    Args:
        query: Normalized query string (from Module B)
        index: BM25Index object from build_bm25_index()
        top_k: Number of top documents to retrieve (default: 10)

    Returns:
        List of (doc_id, score) tuples, sorted by score descending
        Scores are BM25 scores (unbounded, typically 0-100 range)

    Example:
        >>> query = "climate change impacts"
        >>> results = retrieve_bm25(query, index, top_k=5)
        >>> for doc_id, score in results:
        ...     print(f"{doc_id}: {score:.4f}")

    Notes:
        - Returns empty list if query has no matching terms
        - Assumes query is already normalized (Module B)
        - No language translation performed
        - Uses same tokenization as indexing (lowercase + split)

    BM25 Scoring:
        - Higher scores = better match
        - Considers term frequency with saturation
        - Accounts for document length
        - Applies IDF weighting
    """
    if not query or not query.strip():
        logger.warning("Empty query provided, returning empty results")
        return []

    try:
        # Tokenize query using same method as corpus
        tokenized_query = query.lower().split()

        if not tokenized_query:
            logger.warning("Query tokenization resulted in empty list")
            return []

        # Get BM25 scores for all documents
        scores = index.bm25.get_scores(tokenized_query)

        # Get top-K results
        # argsort returns indices in ascending order, so we reverse with [::-1]
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results list with (doc_id, score)
        results = [
            (index.doc_ids[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0  # Only include non-zero scores
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
