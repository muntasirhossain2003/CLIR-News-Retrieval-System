"""
Query Pipeline Module for Cross-Lingual Information Retrieval System.

Orchestrates the complete query processing pipeline.
"""

import logging
import time
from typing import Dict, List, Optional, Any

from .query_detector import QueryLanguageDetector
from .query_normalizer import QueryNormalizer
from .query_translator import QueryTranslator
from .query_expander import QueryExpander
from .ne_mapper import NamedEntityMapper

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Orchestrate complete query processing pipeline.

    Pipeline steps:
    1. Detect language and check for code-switching
    2. Normalize query (clean, lowercase, etc.)
    3. Translate to both languages if needed
    4. Expand with synonyms
    5. Map named entities to other language
    6. Return processed queries for retrieval

    Example:
        >>> processor = QueryProcessor()
        >>> result = processor.process_query("Sheikh Hasina education policy")
        >>> print(result['en_query'], result['bn_query'])
        'sheikh hasina education learning policy' 'শেখ হাসিনা শিক্ষা নীতি'
    """

    def __init__(
        self,
        enable_translation: bool = True,
        enable_expansion: bool = True,
        enable_entity_mapping: bool = True,
        log_timing: bool = True,
    ):
        """
        Initialize the QueryProcessor.

        Args:
            enable_translation: Enable query translation
            enable_expansion: Enable query expansion with synonyms
            enable_entity_mapping: Enable named entity mapping
            log_timing: Log timing for each pipeline step
        """
        self.enable_translation = enable_translation
        self.enable_expansion = enable_expansion
        self.enable_entity_mapping = enable_entity_mapping
        self.log_timing = log_timing

        # Initialize components
        self.language_detector = QueryLanguageDetector()
        self.normalizer = QueryNormalizer(remove_stopwords=False)

        if enable_translation:
            self.translator = QueryTranslator()

        if enable_expansion:
            self.expander = QueryExpander(max_synonyms=2)

        if enable_entity_mapping:
            self.entity_mapper = NamedEntityMapper()

        logger.info(
            f"QueryProcessor initialized (translation={enable_translation}, "
            f"expansion={enable_expansion}, entity_mapping={enable_entity_mapping})"
        )

    def process_query(
        self, query: str, target_languages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process query through complete pipeline.

        Args:
            query: Original search query
            target_languages: Languages to generate queries for (['en', 'bn'] by default)

        Returns:
            Dictionary containing:
                - original: Original query
                - language: Detected language
                - confidence: Detection confidence
                - is_mixed: Whether query is code-switched
                - normalized: Normalized query
                - en_query: Processed English query
                - bn_query: Processed Bangla query
                - entities: Mapped named entities
                - timing: Processing time breakdown

        Example:
            >>> result = processor.process_query("Sheikh Hasina education")
            >>> print(result.keys())
            dict_keys(['original', 'language', 'normalized', 'en_query',
                      'bn_query', 'entities', 'timing'])
        """
        if target_languages is None:
            target_languages = ["en", "bn"]

        timing = {}
        start_time = time.time()

        # Step 1: Detect and normalize
        lang_result, normalized = self._detect_and_normalize(query)
        timing["detect_normalize"] = time.time() - start_time

        source_lang = lang_result["language"]

        # Step 2: Initialize result
        result = {
            "original": query,
            "language": source_lang,
            "confidence": lang_result["confidence"],
            "is_mixed": lang_result["is_mixed"],
            "normalized": normalized,
            "en_query": "",
            "bn_query": "",
            "entities": [],
            "timing": timing,
        }

        # Step 3: Translate to target languages
        step_start = time.time()

        if source_lang == "en":
            result["en_query"] = normalized

            # Translate to Bangla if needed
            if "bn" in target_languages and self.enable_translation:
                bn_translated = self._translate_query(normalized, "bn")
                result["bn_query"] = bn_translated

        elif source_lang == "bn":
            result["bn_query"] = normalized

            # Translate to English if needed
            if "en" in target_languages and self.enable_translation:
                en_translated = self._translate_query(normalized, "en")
                result["en_query"] = en_translated

        else:  # Unknown language
            logger.warning(f"Unknown language detected for query: {query}")
            result["en_query"] = normalized
            result["bn_query"] = normalized

        timing["translate"] = time.time() - step_start

        # Step 4: Expand queries
        if self.enable_expansion:
            step_start = time.time()

            if result["en_query"] and not self.expander.disable_expansion(
                result["en_query"]
            ):
                result["en_query"] = self._expand_query(result["en_query"], "en")

            if result["bn_query"] and not self.expander.disable_expansion(
                result["bn_query"]
            ):
                result["bn_query"] = self._expand_query(result["bn_query"], "bn")

            timing["expand"] = time.time() - step_start

        # Step 5: Map named entities
        if self.enable_entity_mapping:
            step_start = time.time()
            result["entities"] = self._map_entities(query, source_lang)
            timing["entity_mapping"] = time.time() - step_start

        # Total time
        timing["total"] = time.time() - start_time

        if self.log_timing:
            self._log_timing(timing)

        logger.info(
            f"Processed query: '{query}' ({source_lang}) -> EN: '{result['en_query'][:50]}...', "
            f"BN: '{result['bn_query'][:50]}...'"
        )

        return result

    def _detect_and_normalize(self, query: str) -> tuple:
        """
        Detect language and normalize query.

        Args:
            query: Original query

        Returns:
            Tuple of (language_result, normalized_query)
        """
        # Detect language
        lang_result = self.language_detector.detect_query_language(query)
        language = lang_result["language"]

        # Normalize based on detected language
        if lang_result["is_mixed"]:
            normalized = self.normalizer.normalize_mixed(query)
        else:
            normalized = self.normalizer.normalize(query, language)

        return lang_result, normalized

    def _translate_query(self, query: str, target_lang: str) -> str:
        """
        Translate query to target language.

        Args:
            query: Query to translate
            target_lang: Target language ('en' or 'bn')

        Returns:
            Translated query
        """
        if not self.enable_translation or not query:
            return query

        try:
            # Determine source language
            source_lang = "en" if target_lang == "bn" else "bn"

            # Translate
            result = self.translator.translate(query, source_lang, target_lang)
            translated = result.get("translation", query)

            # Normalize translated query
            normalized_translated = self.normalizer.normalize(translated, target_lang)

            return normalized_translated

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return query

    def _expand_query(self, query: str, language: str) -> str:
        """
        Expand query with synonyms.

        Args:
            query: Query to expand
            language: Query language

        Returns:
            Expanded query
        """
        if not self.enable_expansion or not query:
            return query

        try:
            return self.expander.expand(query, language)
        except Exception as e:
            logger.error(f"Expansion error: {e}")
            return query

    def _map_entities(self, query: str, source_lang: str) -> List[Dict[str, Any]]:
        """
        Map named entities to other language.

        Args:
            query: Original query
            source_lang: Source language

        Returns:
            List of mapped entities
        """
        if not self.enable_entity_mapping or source_lang == "unknown":
            return []

        try:
            # Find entities in query
            found_entities = self.entity_mapper.find_entities_in_query(
                query, source_lang
            )

            # Map to target language
            target_lang = "bn" if source_lang == "en" else "en"

            mapped = []
            for entity_info in found_entities:
                entity = entity_info["entity"]
                target_variants = self.entity_mapper.map_entity(
                    entity, source_lang, target_lang
                )

                mapped.append(
                    {
                        "text": entity,
                        "type": "ENTITY",
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "target_variants": target_variants,
                    }
                )

            return mapped

        except Exception as e:
            logger.error(f"Entity mapping error: {e}")
            return []

    def _log_timing(self, timing: Dict[str, float]):
        """
        Log timing information.

        Args:
            timing: Dictionary of timing measurements
        """
        logger.debug("Query processing timing:")
        for step, duration in timing.items():
            logger.debug(f"  {step}: {duration*1000:.2f}ms")
