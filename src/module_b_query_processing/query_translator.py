"""
Query Translator Module for Cross-Lingual Information Retrieval System.

Translates queries between English and Bangla using transformer models.
"""

import logging
from typing import Dict, Optional, Tuple
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)


class QueryTranslator:
    """
    Translate queries between English and Bangla.

    Uses Helsinki-NLP OPUS-MT models for translation with caching.

    Example:
        >>> translator = QueryTranslator()
        >>> result = translator.translate_to_bangla("education policy")
        >>> print(result)
        {'translation': 'শিক্ষা নীতি', 'confidence': 0.92, 'source': 'en', 'target': 'bn'}
    """

    def __init__(self, cache_size: int = 1000, use_gpu: bool = False):
        """
        Initialize the QueryTranslator.

        Args:
            cache_size: Size of translation cache (LRU)
            use_gpu: Whether to use GPU for translation (if available)
        """
        self.cache_size = cache_size
        self.use_gpu = use_gpu
        self.translation_cache = {}

        # Model names (using available models from HuggingFace)
        # EN->BN: Use NLLB (No Language Left Behind) for English-Bengali
        self.en_to_bn_model = "facebook/nllb-200-distilled-600M"  # Supports 200 languages including Bengali
        self.bn_to_en_model = "Helsinki-NLP/opus-mt-bn-en"  # Available and working

        # Language codes for NLLB
        self.nllb_lang_codes = {"en": "eng_Latn", "bn": "ben_Beng"}

        # Lazy loading of models
        self.en_bn_pipeline = None
        self.bn_en_pipeline = None

        logger.info(
            f"QueryTranslator initialized (cache_size={cache_size}, use_gpu={use_gpu})"
        )

    def _load_en_to_bn_model(self):
        """Load English to Bangla translation model (NLLB)."""
        if self.en_bn_pipeline is not None:
            return

        try:
            from transformers import pipeline

            device = 0 if self.use_gpu else -1
            # NLLB requires specific language codes
            self.en_bn_pipeline = pipeline(
                "translation",
                model=self.en_to_bn_model,
                device=device,
                src_lang="eng_Latn",
                tgt_lang="ben_Beng",
            )
            logger.info(f"Loaded {self.en_to_bn_model} model (EN->BN)")

        except Exception as e:
            logger.error(f"Failed to load English to Bangla model: {e}")
            logger.warning("Translation from English to Bangla will not be available")

    def _load_bn_to_en_model(self):
        """Load Bangla to English translation model."""
        if self.bn_en_pipeline is not None:
            return

        try:
            from transformers import pipeline

            device = 0 if self.use_gpu else -1
            self.bn_en_pipeline = pipeline(
                "translation", model=self.bn_to_en_model, device=device
            )
            logger.info(f"Loaded {self.bn_to_en_model} model")

        except Exception as e:
            logger.error(f"Failed to load Bangla to English model: {e}")
            logger.warning("Translation from Bangla to English will not be available")

    def translate_to_bangla(self, query: str) -> Dict[str, any]:
        """
        Translate English query to Bangla.

        Args:
            query: English query text

        Returns:
            Dictionary containing:
                - translation: Translated text
                - confidence: Confidence score (if available)
                - source: Source language ('en')
                - target: Target language ('bn')

        Example:
            >>> translator.translate_to_bangla("education policy")
            {'translation': 'শিক্ষা নীতি', 'confidence': 0.92,
             'source': 'en', 'target': 'bn'}
        """
        return self.translate(query, source_lang="en", target_lang="bn")

    def translate_to_english(self, query: str) -> Dict[str, any]:
        """
        Translate Bangla query to English.

        Args:
            query: Bangla query text

        Returns:
            Dictionary containing translation and metadata

        Example:
            >>> translator.translate_to_english("শিক্ষা নীতি")
            {'translation': 'education policy', 'confidence': 0.89,
             'source': 'bn', 'target': 'en'}
        """
        return self.translate(query, source_lang="bn", target_lang="en")

    def translate(
        self, query: str, source_lang: str, target_lang: str
    ) -> Dict[str, any]:
        """
        Translate query from source language to target language.

        Args:
            query: Query text to translate
            source_lang: Source language code ('en' or 'bn')
            target_lang: Target language code ('en' or 'bn')

        Returns:
            Dictionary containing translation result and metadata
        """
        if not query or not query.strip():
            logger.warning("Empty query provided for translation")
            return {
                "translation": "",
                "confidence": 0.0,
                "source": source_lang,
                "target": target_lang,
                "cached": False,
            }

        # Check cache first
        cached_result = self.get_cached_translation(query, source_lang, target_lang)
        if cached_result:
            logger.debug(f"Cache hit for query: {query}")
            return cached_result

        # Perform translation
        try:
            # Load appropriate model
            if source_lang == "en" and target_lang == "bn":
                self._load_en_to_bn_model()
                pipeline = self.en_bn_pipeline
            elif source_lang == "bn" and target_lang == "en":
                self._load_bn_to_en_model()
                pipeline = self.bn_en_pipeline
            else:
                logger.error(
                    f"Unsupported language pair: {source_lang} -> {target_lang}"
                )
                return {
                    "translation": query,  # Return original
                    "confidence": 0.0,
                    "source": source_lang,
                    "target": target_lang,
                    "cached": False,
                    "error": "Unsupported language pair",
                }

            if pipeline is None:
                logger.error(
                    f"Translation model not available for {source_lang} -> {target_lang}"
                )
                return {
                    "translation": query,  # Return original
                    "confidence": 0.0,
                    "source": source_lang,
                    "target": target_lang,
                    "cached": False,
                    "error": "Model not loaded",
                }

            # Translate
            result = pipeline(query, max_length=512)
            translation = result[0]["translation_text"]

            # Build result
            result_dict = {
                "translation": translation,
                "confidence": 0.85,  # Default confidence (models don't provide scores)
                "source": source_lang,
                "target": target_lang,
                "cached": False,
            }

            # Cache the result
            self._cache_translation(query, source_lang, target_lang, result_dict)

            logger.debug(f"Translated '{query}' -> '{translation}'")
            return result_dict

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {
                "translation": query,  # Return original on error
                "confidence": 0.0,
                "source": source_lang,
                "target": target_lang,
                "cached": False,
                "error": str(e),
            }

    def get_cached_translation(
        self, query: str, source_lang: str, target_lang: str
    ) -> Optional[Dict[str, any]]:
        """
        Get cached translation if available.

        Args:
            query: Query text
            source_lang: Source language
            target_lang: Target language

        Returns:
            Cached result dictionary or None
        """
        cache_key = self._get_cache_key(query, source_lang, target_lang)
        result = self.translation_cache.get(cache_key)

        if result:
            result = result.copy()
            result["cached"] = True

        return result

    def _cache_translation(
        self, query: str, source_lang: str, target_lang: str, result: Dict[str, any]
    ):
        """
        Cache a translation result.

        Args:
            query: Original query
            source_lang: Source language
            target_lang: Target language
            result: Translation result to cache
        """
        # Implement simple LRU-like behavior
        if len(self.translation_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            first_key = next(iter(self.translation_cache))
            del self.translation_cache[first_key]

        cache_key = self._get_cache_key(query, source_lang, target_lang)
        self.translation_cache[cache_key] = result

    def _get_cache_key(self, query: str, source_lang: str, target_lang: str) -> str:
        """
        Generate cache key for query.

        Args:
            query: Query text
            source_lang: Source language
            target_lang: Target language

        Returns:
            Cache key string
        """
        # Use hash to handle long queries
        query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
        return f"{source_lang}_{target_lang}_{query_hash}"

    def clear_cache(self):
        """Clear the translation cache."""
        self.translation_cache.clear()
        logger.info("Translation cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache size and capacity
        """
        return {
            "size": len(self.translation_cache),
            "capacity": self.cache_size,
            "hit_rate": 0.0,  # Would need to track hits/misses to calculate
        }
