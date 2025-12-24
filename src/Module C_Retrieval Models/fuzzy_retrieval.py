"""
Module C - Model 2: Fuzzy and Transliteration-Based Matching

PURPOSE:
Handle spelling variations and cross-script term mismatch using character-level
similarity measures. This demonstrates why pure lexical retrieval fails for
cross-lingual terms and provides a baseline for handling spelling variations.

METHODS:
- Character n-gram similarity (Jaccard coefficient)
- Fuzzy string matching (SequenceMatcher ratio)

This model is NOT for semantic understanding - it's purely character-level
matching to handle typos, transliterations, and spelling variations.
"""

import logging
from typing import List, Tuple, Dict
from difflib import SequenceMatcher
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _extract_character_ngrams(text: str, n: int = 3) -> set:
    """
    Extract character n-grams from text.

    Args:
        text: Input text
        n: N-gram size (default: 3 for trigrams)

    Returns:
        Set of character n-grams

    Example:
        >>> _extract_character_ngrams("hello", 3)
        {'hel', 'ell', 'llo'}
    """
    text = text.lower()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculate Jaccard coefficient between two sets.

    Args:
        set1: First set
        set2: Second set

    Returns:
        Jaccard similarity in range [0, 1]

    Formula:
        J(A, B) = |A ∩ B| / |A ∪ B|
    """
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def fuzzy_match(
    query: str, doc_text: str, method: str = "ngram", ngram_size: int = 3
) -> float:
    """
    Calculate fuzzy similarity between query and document text.

    Args:
        query: Query string
        doc_text: Document text
        method: Matching method - 'ngram' (character n-grams) or 'sequence' (SequenceMatcher)
        ngram_size: Size of character n-grams (default: 3)

    Returns:
        Similarity score in range [0, 1]
        - 1.0 = perfect match
        - 0.0 = no similarity

    Methods:
        - 'ngram': Character n-gram Jaccard similarity
          * Good for: Spelling variations, transliterations
          * Fast and efficient
          * Handles character-level overlap

        - 'sequence': SequenceMatcher ratio (Ratcliff/Obershelp)
          * Good for: Edit distance, typos
          * More sensitive to character order
          * Slower but more accurate for similar strings

    Example:
        >>> fuzzy_match("climate", "climat", method='ngram')
        0.7142857142857143
        >>> fuzzy_match("color", "colour", method='sequence')
        0.9090909090909091

    Notes:
        - Case-insensitive matching
        - No semantic understanding
        - Works on character-level only
    """
    if not query or not doc_text:
        return 0.0

    query = query.lower().strip()
    doc_text = doc_text.lower().strip()

    if method == "ngram":
        # Character n-gram similarity
        query_ngrams = _extract_character_ngrams(query, ngram_size)
        doc_ngrams = _extract_character_ngrams(doc_text, ngram_size)
        return _jaccard_similarity(query_ngrams, doc_ngrams)

    elif method == "sequence":
        # Sequence matcher (edit distance based)
        return SequenceMatcher(None, query, doc_text).ratio()

    else:
        raise ValueError(f"Unknown method: {method}. Use 'ngram' or 'sequence'")


def retrieve_fuzzy(
    query: str,
    docs: Dict[str, str],
    top_k: int = 10,
    method: str = "ngram",
    ngram_size: int = 3,
    min_score: float = 0.1,
) -> List[Tuple[str, float]]:
    """
    Retrieve documents using fuzzy character-level matching.

    Args:
        query: Query string
        docs: Dictionary mapping doc_id -> document_text
        top_k: Number of top documents to retrieve (default: 10)
        method: Matching method - 'ngram' or 'sequence' (default: 'ngram')
        ngram_size: Size of character n-grams (default: 3)
        min_score: Minimum similarity score threshold (default: 0.1)

    Returns:
        List of (doc_id, score) tuples, sorted by score descending
        Scores are similarity values in range [0, 1]

    Example:
        >>> docs = {
        ...     'doc1': 'climate change is a serious issue',
        ...     'doc2': 'climat chang affects the world'
        ... }
        >>> results = retrieve_fuzzy('climate change', docs, top_k=5)
        >>> for doc_id, score in results:
        ...     print(f"{doc_id}: {score:.4f}")

    Notes:
        - Computes average fuzzy match across all query terms
        - Each query term is matched against entire document
        - Returns only documents with score >= min_score
        - No stopword removal or stemming
        - Character-level matching only (no semantics)

    Use Cases:
        - Spelling variations: "organize" vs "organise"
        - Typos: "recieve" vs "receive"
        - Transliterations: "Beijing" vs "Peking"
        - Cross-script similarity (limited effectiveness)

    Limitations:
        - Does NOT understand meaning
        - Does NOT handle synonyms
        - Does NOT translate languages
        - Slow for large document collections
    """
    if not query or not query.strip():
        logger.warning("Empty query provided, returning empty results")
        return []

    if not docs:
        logger.warning("Empty document collection provided")
        return []

    query = query.strip()

    # Split query into terms for more granular matching
    query_terms = query.lower().split()

    logger.info(f"Computing fuzzy similarity for {len(docs)} documents...")
    logger.info(f"  Method: {method}")
    logger.info(f"  Query terms: {len(query_terms)}")

    doc_scores = []

    try:
        for doc_id, doc_text in docs.items():
            # Calculate average similarity across all query terms
            term_scores = []

            for term in query_terms:
                # Match term against entire document
                # (In practice, you might want to match against individual doc words)
                score = fuzzy_match(
                    term, doc_text, method=method, ngram_size=ngram_size
                )
                term_scores.append(score)

            # Average score across all query terms
            avg_score = sum(term_scores) / len(term_scores) if term_scores else 0.0

            # Only include documents above minimum threshold
            if avg_score >= min_score:
                doc_scores.append((doc_id, avg_score))

        # Sort by score descending and take top-K
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        results = doc_scores[:top_k]

        logger.info(f"Retrieved {len(results)} documents (top_k={top_k})")
        if results:
            logger.info(f"  Top score: {results[0][1]:.4f}")
            logger.info(f"  Lowest score: {results[-1][1]:.4f}")
        else:
            logger.warning("  No matching documents found above threshold")

        return results

    except Exception as e:
        logger.error(f"Error during fuzzy retrieval: {e}")
        return []


def retrieve_fuzzy_per_term(
    query: str,
    docs: Dict[str, str],
    top_k: int = 10,
    method: str = "ngram",
    ngram_size: int = 3,
    min_score: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Retrieve documents using per-term fuzzy matching against document terms.

    This variant matches each query term against individual document terms
    (rather than the entire document), which is more precise for detecting
    specific term variations.

    Args:
        query: Query string
        docs: Dictionary mapping doc_id -> document_text
        top_k: Number of top documents to retrieve (default: 10)
        method: Matching method - 'ngram' or 'sequence' (default: 'ngram')
        ngram_size: Size of character n-grams (default: 3)
        min_score: Minimum similarity score threshold (default: 0.5)

    Returns:
        List of (doc_id, score) tuples, sorted by score descending

    Example:
        >>> docs = {'doc1': 'climate change', 'doc2': 'climat chang'}
        >>> results = retrieve_fuzzy_per_term('climate', docs)
        >>> # Finds 'climate' matches with 'climat' at term level

    Notes:
        - More precise than retrieve_fuzzy()
        - Better for identifying specific spelling variations
        - Slower due to term-by-term comparison
    """
    if not query or not query.strip():
        logger.warning("Empty query provided, returning empty results")
        return []

    if not docs:
        logger.warning("Empty document collection provided")
        return []

    query_terms = query.lower().split()

    logger.info(f"Computing per-term fuzzy similarity for {len(docs)} documents...")

    doc_scores = []

    try:
        for doc_id, doc_text in docs.items():
            doc_terms = doc_text.lower().split()

            # For each query term, find best match in document terms
            term_scores = []
            for q_term in query_terms:
                best_score = 0.0
                for d_term in doc_terms:
                    score = fuzzy_match(
                        q_term, d_term, method=method, ngram_size=ngram_size
                    )
                    best_score = max(best_score, score)
                term_scores.append(best_score)

            # Average of best matches
            avg_score = sum(term_scores) / len(term_scores) if term_scores else 0.0

            if avg_score >= min_score:
                doc_scores.append((doc_id, avg_score))

        # Sort and return top-K
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        results = doc_scores[:top_k]

        logger.info(f"Retrieved {len(results)} documents (top_k={top_k})")
        if results:
            logger.info(f"  Top score: {results[0][1]:.4f}")

        return results

    except Exception as e:
        logger.error(f"Error during per-term fuzzy retrieval: {e}")
        return []
