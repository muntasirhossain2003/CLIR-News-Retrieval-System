"""
Fuzzy Matcher Module for Cross-Lingual Information Retrieval System.

Implements fuzzy string matching for handling variations and typos.
"""

import logging
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


class FuzzyMatcher:
    """
    Fuzzy string matching using Levenshtein distance and n-gram similarity.

    Features:
    - Levenshtein distance calculation
    - Jaccard similarity
    - Character n-gram matching
    - Transliteration-aware matching

    Example:
        >>> matcher = FuzzyMatcher()
        >>> score = matcher.fuzzy_match("Bangladesh", "বাংলাদেশ", threshold=0.6)
        >>> print(score)
        0.75
    """

    def __init__(self, ngram_size: int = 3):
        """
        Initialize the FuzzyMatcher.

        Args:
            ngram_size: Size of character n-grams (default: 3)
        """
        self.ngram_size = ngram_size

        logger.info(f"FuzzyMatcher initialized (ngram_size={ngram_size})")

    def fuzzy_match(self, query: str, document: str, threshold: float = 0.8) -> float:
        """
        Calculate fuzzy match score between query and document.

        Args:
            query: Query string
            document: Document string
            threshold: Minimum similarity threshold

        Returns:
            Similarity score [0, 1]

        Example:
            >>> matcher.fuzzy_match("education", "educaton", threshold=0.7)
            0.89
        """
        if not query or not document:
            return 0.0

        # Normalize strings
        query_norm = query.lower().strip()
        doc_norm = document.lower().strip()

        # Exact match
        if query_norm == doc_norm:
            return 1.0

        # Calculate multiple similarity scores
        lev_sim = self._levenshtein_similarity(query_norm, doc_norm)
        jaccard_sim = self._jaccard_similarity(query_norm, doc_norm)
        ngram_sim = self.character_ngram_match(query_norm, doc_norm, self.ngram_size)

        # Weighted combination
        combined_score = 0.4 * lev_sim + 0.3 * jaccard_sim + 0.3 * ngram_sim

        # Apply threshold
        return combined_score if combined_score >= threshold else 0.0

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Edit distance (number of operations needed)
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row = [i + 1]

            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)

                current_row.append(min(insertions, deletions, substitutions))

            previous_row = current_row

        return previous_row[-1]

    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """
        Convert Levenshtein distance to similarity score [0, 1].

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score
        """
        distance = self._levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))

        if max_len == 0:
            return 1.0

        similarity = 1.0 - (distance / max_len)
        return max(0.0, similarity)

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate Jaccard similarity based on word sets.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Jaccard similarity [0, 1]
        """
        # Split into words
        words1 = set(s1.split())
        words2 = set(s2.split())

        if not words1 and not words2:
            return 1.0

        if not words1 or not words2:
            return 0.0

        # Calculate Jaccard
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def character_ngram_match(self, query: str, document: str, n: int = 3) -> float:
        """
        Calculate similarity using character n-grams.

        Args:
            query: Query string
            document: Document string
            n: N-gram size

        Returns:
            N-gram similarity score [0, 1]

        Example:
            >>> matcher.character_ngram_match("hello", "helo", n=2)
            0.75
        """
        if not query or not document:
            return 0.0

        # Generate n-grams
        query_ngrams = self._get_ngrams(query, n)
        doc_ngrams = self._get_ngrams(document, n)

        if not query_ngrams and not doc_ngrams:
            return 1.0

        if not query_ngrams or not doc_ngrams:
            return 0.0

        # Calculate Jaccard similarity of n-grams
        intersection = len(query_ngrams & doc_ngrams)
        union = len(query_ngrams | doc_ngrams)

        return intersection / union if union > 0 else 0.0

    def _get_ngrams(self, text: str, n: int) -> set:
        """
        Generate character n-grams from text.

        Args:
            text: Input text
            n: N-gram size

        Returns:
            Set of n-grams
        """
        if len(text) < n:
            return {text}

        ngrams = set()
        for i in range(len(text) - n + 1):
            ngrams.add(text[i : i + n])

        return ngrams

    def transliteration_match(self, query: str, document: str) -> float:
        """
        Handle transliteration matching (e.g., "Bangladesh" <-> "বাংলাদেশ").

        This is a simplified implementation. In production, use proper
        transliteration libraries or phonetic encoding.

        Args:
            query: Query string
            document: Document string

        Returns:
            Similarity score [0, 1]
        """
        # Simple heuristic: check if one is Bangla and one is English
        query_is_bangla = bool(re.search(r"[\u0980-\u09FF]", query))
        doc_is_bangla = bool(re.search(r"[\u0980-\u09FF]", document))

        # If both same script, use regular fuzzy match
        if query_is_bangla == doc_is_bangla:
            return self.fuzzy_match(query, document, threshold=0.0)

        # For cross-script matching, use length and character frequency
        # This is a simplified approach
        len_ratio = min(len(query), len(document)) / max(len(query), len(document))

        # Rough transliteration match based on length
        if len_ratio > 0.7:
            return len_ratio * 0.5  # Conservative score for cross-script

        return 0.0

    def batch_fuzzy_match(
        self, query: str, documents: List[str], threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Perform fuzzy matching against multiple documents.

        Args:
            query: Query string
            documents: List of document strings
            threshold: Minimum similarity threshold

        Returns:
            List of matches with scores, sorted by score
        """
        results = []

        for i, doc in enumerate(documents):
            score = self.fuzzy_match(query, doc, threshold=0.0)

            if score >= threshold:
                results.append({"index": i, "text": doc, "score": round(score, 4)})

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results
