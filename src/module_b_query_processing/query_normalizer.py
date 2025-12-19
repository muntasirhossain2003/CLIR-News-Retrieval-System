"""
Query Normalizer Module for Cross-Lingual Information Retrieval System.

Normalizes search queries for consistent processing across languages.
"""

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)


class QueryNormalizer:
    """
    Normalize search queries for consistent processing.

    Features:
    - Lowercase transformation (English)
    - Unicode normalization (Bangla)
    - Whitespace normalization
    - Punctuation handling
    - Optional stopword removal

    Example:
        >>> normalizer = QueryNormalizer()
        >>> result = normalizer.normalize("BANGLADESH'S Cricket!!!", "en")
        >>> print(result)
        'bangladesh cricket'
    """

    def __init__(
        self,
        remove_stopwords: bool = False,
        remove_punctuation: bool = True,
        normalize_unicode: bool = True,
    ):
        """
        Initialize the QueryNormalizer.

        Args:
            remove_stopwords: Whether to remove stopwords (default: False for queries)
            remove_punctuation: Whether to remove punctuation
            normalize_unicode: Whether to normalize Unicode (important for Bangla)
        """
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.normalize_unicode = normalize_unicode

        # Common English stopwords (only used if remove_stopwords=True)
        self.english_stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "may",
            "might",
            "can",
        }

        # Common Bangla stopwords
        self.bangla_stopwords = {
            "এবং",
            "বা",
            "কিন্তু",
            "তবে",
            "যে",
            "যা",
            "যার",
            "এই",
            "ওই",
            "সেই",
            "একটি",
            "একজন",
            "দুটি",
            "কয়েক",
            "অনেক",
            "সব",
            "সকল",
            "প্রতি",
            "মধ্যে",
        }

        logger.info(
            f"QueryNormalizer initialized (remove_stopwords={remove_stopwords}, "
            f"remove_punctuation={remove_punctuation})"
        )

    def normalize(self, query: str, language: str = "en") -> str:
        """
        Normalize a query based on language.

        Args:
            query: Input query to normalize
            language: Language code ('en' or 'bn')

        Returns:
            Normalized query string

        Example:
            >>> normalizer.normalize("BANGLADESH'S Cricket!!!", "en")
            'bangladesh cricket'
            >>> normalizer.normalize("  বাংলাদেশ  ক্রিকেট  ", "bn")
            'বাংলাদেশ ক্রিকেট'
        """
        if not query:
            return ""

        if language == "en":
            return self._normalize_english(query)
        elif language == "bn":
            return self._normalize_bangla(query)
        else:
            logger.warning(
                f"Unsupported language: {language}, using English normalization"
            )
            return self._normalize_english(query)

    def _normalize_english(self, query: str) -> str:
        """
        Normalize English query.

        Args:
            query: English query text

        Returns:
            Normalized query
        """
        # Convert to lowercase
        query = query.lower()

        # Remove possessives
        query = re.sub(r"'s\b", "", query)
        query = re.sub(r"'s\b", "", query)  # Handle curly apostrophe

        # Remove punctuation if configured
        if self.remove_punctuation:
            # Keep hyphens in compound words, remove other punctuation
            query = re.sub(r"[^\w\s-]", " ", query)
            # Clean up multiple hyphens
            query = re.sub(r"-+", "-", query)

        # Normalize whitespace
        query = re.sub(r"\s+", " ", query)
        query = query.strip()

        # Remove stopwords if configured
        if self.remove_stopwords:
            words = query.split()
            words = [w for w in words if w not in self.english_stopwords]
            query = " ".join(words)

        return query

    def _normalize_bangla(self, query: str) -> str:
        """
        Normalize Bangla query.

        Args:
            query: Bangla query text

        Returns:
            Normalized query
        """
        # Unicode normalization (NFC form for Bangla)
        if self.normalize_unicode:
            query = unicodedata.normalize("NFC", query)

        # Remove Bangla punctuation
        if self.remove_punctuation:
            # Common Bangla punctuation marks
            bangla_punct = r"[।॥,;:!?\'\"…]"
            query = re.sub(bangla_punct, " ", query)

            # Also remove common English punctuation
            query = re.sub(r"[^\w\s-]", " ", query, flags=re.UNICODE)

        # Normalize whitespace
        query = re.sub(r"\s+", " ", query)
        query = query.strip()

        # Remove stopwords if configured
        if self.remove_stopwords:
            words = query.split()
            words = [w for w in words if w not in self.bangla_stopwords]
            query = " ".join(words)

        return query

    def normalize_mixed(self, query: str) -> str:
        """
        Normalize a query that may contain both English and Bangla.

        Args:
            query: Mixed-language query

        Returns:
            Normalized query
        """
        # Apply both normalizations but skip lowercase for Bangla parts

        # Unicode normalization
        if self.normalize_unicode:
            query = unicodedata.normalize("NFC", query)

        # Remove punctuation
        if self.remove_punctuation:
            # Remove both English and Bangla punctuation
            query = re.sub(r"[।॥,;:!?\'\"…]", " ", query)
            query = re.sub(r"[^\w\s-]", " ", query, flags=re.UNICODE)

        # Normalize whitespace
        query = re.sub(r"\s+", " ", query)
        query = query.strip()

        # Lowercase only English parts (preserve Bangla as-is)
        words = query.split()
        normalized_words = []

        for word in words:
            # Check if word contains Bangla characters
            if re.search(r"[\u0980-\u09FF]", word):
                normalized_words.append(word)
            else:
                normalized_words.append(word.lower())

        query = " ".join(normalized_words)

        return query

    def remove_extra_spaces(self, query: str) -> str:
        """
        Remove extra whitespace from query.

        Args:
            query: Input query

        Returns:
            Query with normalized whitespace
        """
        return re.sub(r"\s+", " ", query).strip()

    def remove_special_chars(self, query: str, keep_chars: str = "") -> str:
        """
        Remove special characters from query.

        Args:
            query: Input query
            keep_chars: String of characters to keep (e.g., '-_')

        Returns:
            Query with special characters removed
        """
        if keep_chars:
            pattern = f"[^\\w\\s{re.escape(keep_chars)}]"
        else:
            pattern = r"[^\w\s]"

        return re.sub(pattern, " ", query, flags=re.UNICODE)
