"""
Language Detection Module for Cross-Lingual Information Retrieval System.

This module provides language detection capabilities for text documents
and queries, supporting English and Bangla languages.
"""

import logging
from typing import Dict, List, Tuple, Optional
import langdetect
from langdetect import detect, detect_langs, DetectorFactory

# Set seed for consistent results
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Detect language of text using langdetect library.

    Supports detection of English (en) and Bangla (bn) with confidence scores.
    Can handle mixed-language text detection.

    Example:
        >>> detector = LanguageDetector()
        >>> result = detector.detect("This is an English sentence.")
        >>> print(result)
        {'language': 'en', 'confidence': 0.9999, 'is_mixed': False}
    """

    def __init__(self, min_confidence: float = 0.7, mixed_threshold: float = 0.3):
        """
        Initialize the LanguageDetector.

        Args:
            min_confidence: Minimum confidence threshold for language detection
            mixed_threshold: Threshold for considering text as mixed-language
        """
        self.min_confidence = min_confidence
        self.mixed_threshold = mixed_threshold
        self.supported_languages = {"en", "bn"}
        logger.info(
            f"LanguageDetector initialized with min_confidence={min_confidence}"
        )

    def detect(self, text: str) -> Dict[str, any]:
        """
        Detect the language of a single text.

        Args:
            text: Input text to detect language

        Returns:
            Dictionary containing:
                - language: Detected language code ('en', 'bn', or 'unknown')
                - confidence: Confidence score (0.0 to 1.0)
                - is_mixed: Boolean indicating if text contains mixed languages

        Example:
            >>> detector.detect("Bangladesh is a beautiful country")
            {'language': 'en', 'confidence': 0.99, 'is_mixed': False}
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for language detection")
            return {"language": "unknown", "confidence": 0.0, "is_mixed": False}

        try:
            # Get all language probabilities
            lang_probs = detect_langs(text)

            # Find the dominant language
            dominant_lang = lang_probs[0]
            dominant_code = dominant_lang.lang
            dominant_prob = dominant_lang.prob

            # Check if text is mixed-language
            is_mixed = self._check_mixed_language(lang_probs)

            # Map to our supported languages
            if dominant_code not in self.supported_languages:
                logger.debug(
                    f"Detected unsupported language: {dominant_code}, defaulting to 'unknown'"
                )
                return {
                    "language": "unknown",
                    "confidence": dominant_prob,
                    "is_mixed": is_mixed,
                }

            result = {
                "language": dominant_code,
                "confidence": round(dominant_prob, 4),
                "is_mixed": is_mixed,
            }

            logger.debug(f"Detected language: {result}")
            return result

        except langdetect.lang_detect_exception.LangDetectException as e:
            logger.error(f"Language detection failed: {e}")
            return {"language": "unknown", "confidence": 0.0, "is_mixed": False}
        except Exception as e:
            logger.error(f"Unexpected error in language detection: {e}")
            return {"language": "unknown", "confidence": 0.0, "is_mixed": False}

    def detect_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Detect language for a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            List of detection results, one per input text

        Example:
            >>> texts = ["Hello world", "বাংলাদেশ"]
            >>> detector.detect_batch(texts)
            [{'language': 'en', 'confidence': 0.99, 'is_mixed': False},
             {'language': 'bn', 'confidence': 0.99, 'is_mixed': False}]
        """
        if not texts:
            logger.warning("Empty text list provided for batch detection")
            return []

        results = []
        for i, text in enumerate(texts):
            try:
                result = self.detect(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error detecting language for text {i}: {e}")
                results.append(
                    {"language": "unknown", "confidence": 0.0, "is_mixed": False}
                )

        logger.info(f"Batch detection completed for {len(texts)} texts")
        return results

    def is_mixed_language(self, text: str) -> bool:
        """
        Check if text contains mixed languages.

        Args:
            text: Input text to check

        Returns:
            True if text contains mixed languages, False otherwise

        Example:
            >>> detector.is_mixed_language("Hello বাংলাদেশ")
            True
        """
        if not text or not text.strip():
            return False

        try:
            lang_probs = detect_langs(text)
            return self._check_mixed_language(lang_probs)
        except Exception as e:
            logger.error(f"Error checking mixed language: {e}")
            return False

    def _check_mixed_language(self, lang_probs: List) -> bool:
        """
        Internal method to check if language probabilities indicate mixed text.

        Args:
            lang_probs: List of language probability objects from langdetect

        Returns:
            True if mixed languages detected, False otherwise
        """
        if len(lang_probs) < 2:
            return False

        # Check if second language has significant probability
        second_prob = lang_probs[1].prob if len(lang_probs) > 1 else 0.0

        # If second language probability is above threshold, consider it mixed
        is_mixed = second_prob >= self.mixed_threshold

        return is_mixed

    def get_language_distribution(self, text: str) -> Dict[str, float]:
        """
        Get probability distribution across all detected languages.

        Args:
            text: Input text

        Returns:
            Dictionary mapping language codes to probabilities

        Example:
            >>> detector.get_language_distribution("Hello বাংলাদেশ")
            {'en': 0.57, 'bn': 0.43}
        """
        if not text or not text.strip():
            return {}

        try:
            lang_probs = detect_langs(text)
            distribution = {lang.lang: round(lang.prob, 4) for lang in lang_probs}
            return distribution
        except Exception as e:
            logger.error(f"Error getting language distribution: {e}")
            return {}
