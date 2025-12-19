"""
Query Language Detection Module for Cross-Lingual Information Retrieval System.

Specialized language detection for search queries, handling short text and code-switching.
"""

import logging
from typing import Dict, List, Optional
import langdetect
from langdetect import detect_langs, DetectorFactory
import re

# Set seed for consistent results
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


class QueryLanguageDetector:
    """
    Detect language of search queries with support for short text and code-switching.

    Handles:
    - Short queries (1-3 words)
    - Mixed language queries (code-switching)
    - Dominant language detection

    Example:
        >>> detector = QueryLanguageDetector()
        >>> result = detector.detect_query_language("Sheikh Hasina")
        >>> print(result)
        {'language': 'en', 'confidence': 0.85, 'is_mixed': False}
    """

    def __init__(
        self, min_confidence: float = 0.6, code_switch_threshold: float = 0.25
    ):
        """
        Initialize the QueryLanguageDetector.

        Args:
            min_confidence: Minimum confidence for language detection
            code_switch_threshold: Threshold for detecting code-switching
        """
        self.min_confidence = min_confidence
        self.code_switch_threshold = code_switch_threshold
        self.supported_languages = {"en", "bn"}

        # Bangla Unicode range
        self.bangla_pattern = re.compile(r"[\u0980-\u09FF]")
        # English letters
        self.english_pattern = re.compile(r"[a-zA-Z]")

        logger.info(f"QueryLanguageDetector initialized")

    def detect_query_language(self, query: str) -> Dict[str, any]:
        """
        Detect the language of a search query.

        Args:
            query: Search query text

        Returns:
            Dictionary containing:
                - language: Detected language ('en', 'bn', or 'unknown')
                - confidence: Confidence score
                - is_mixed: Whether query contains mixed languages
                - languages: List of detected languages with scores

        Example:
            >>> detector.detect_query_language("Bangladesh cricket")
            {'language': 'en', 'confidence': 0.95, 'is_mixed': False,
             'languages': [{'lang': 'en', 'prob': 0.95}]}
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return {
                "language": "unknown",
                "confidence": 0.0,
                "is_mixed": False,
                "languages": [],
            }

        query = query.strip()

        # First try character-based detection (works better for short text)
        char_based = self._character_based_detection(query)

        # Then try langdetect
        try:
            lang_probs = detect_langs(query)

            # Get dominant language
            dominant = lang_probs[0]
            dominant_lang = dominant.lang
            dominant_prob = dominant.prob

            # Check for code-switching
            is_mixed = self.is_code_switched(query)

            # Use character-based detection for very short queries or low confidence
            if len(query.split()) <= 2 or dominant_prob < self.min_confidence:
                if char_based["language"] != "unknown":
                    logger.debug(
                        f"Using character-based detection for short query: {query}"
                    )
                    return char_based

            # Validate against supported languages
            if dominant_lang not in self.supported_languages:
                dominant_lang = "unknown"

            result = {
                "language": dominant_lang,
                "confidence": round(dominant_prob, 4),
                "is_mixed": is_mixed,
                "languages": [
                    {"lang": lp.lang, "prob": round(lp.prob, 4)} for lp in lang_probs
                ],
            }

            logger.debug(
                f"Query '{query}' detected as: {result['language']} ({result['confidence']:.2f})"
            )
            return result

        except Exception as e:
            logger.warning(
                f"langdetect failed for query '{query}': {e}, using character-based"
            )
            return char_based

    def is_code_switched(self, query: str) -> bool:
        """
        Check if query contains code-switching (mixed languages).

        Args:
            query: Search query

        Returns:
            True if code-switching detected, False otherwise

        Example:
            >>> detector.is_code_switched("Sheikh Hasina শেখ হাসিনা")
            True
        """
        if not query:
            return False

        # Check for both Bangla and English characters
        has_bangla = bool(self.bangla_pattern.search(query))
        has_english = bool(self.english_pattern.search(query))

        # If both present, it's code-switched
        if has_bangla and has_english:
            logger.debug(f"Code-switching detected in: {query}")
            return True

        # Also check language probabilities
        try:
            lang_probs = detect_langs(query)
            if len(lang_probs) >= 2:
                second_prob = lang_probs[1].prob
                if second_prob >= self.code_switch_threshold:
                    return True
        except Exception:
            pass

        return False

    def get_dominant_language(self, query: str) -> str:
        """
        Get the dominant language in a potentially mixed query.

        Args:
            query: Search query

        Returns:
            Language code of dominant language ('en', 'bn', or 'unknown')

        Example:
            >>> detector.get_dominant_language("Bangladesh শেখ হাসিনা")
            'bn'
        """
        result = self.detect_query_language(query)
        return result["language"]

    def _character_based_detection(self, text: str) -> Dict[str, any]:
        """
        Detect language based on character patterns.

        Useful for very short queries where statistical detection fails.

        Args:
            text: Input text

        Returns:
            Detection result dictionary
        """
        bangla_chars = len(self.bangla_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        total_chars = bangla_chars + english_chars

        if total_chars == 0:
            return {
                "language": "unknown",
                "confidence": 0.0,
                "is_mixed": False,
                "languages": [],
            }

        bangla_ratio = bangla_chars / total_chars
        english_ratio = english_chars / total_chars

        # Determine dominant language
        if bangla_ratio > english_ratio and bangla_ratio > 0.3:
            language = "bn"
            confidence = bangla_ratio
        elif english_ratio > bangla_ratio and english_ratio > 0.3:
            language = "en"
            confidence = english_ratio
        else:
            language = "unknown"
            confidence = 0.5

        # Check for mixing
        is_mixed = bangla_ratio > 0.2 and english_ratio > 0.2

        return {
            "language": language,
            "confidence": round(confidence, 4),
            "is_mixed": is_mixed,
            "languages": [
                {"lang": "bn", "prob": round(bangla_ratio, 4)},
                {"lang": "en", "prob": round(english_ratio, 4)},
            ],
        }

    def detect_all_languages(self, query: str) -> List[Dict[str, any]]:
        """
        Get all languages detected in query with probabilities.

        Args:
            query: Search query

        Returns:
            List of language dictionaries sorted by probability
        """
        result = self.detect_query_language(query)
        return result.get("languages", [])
