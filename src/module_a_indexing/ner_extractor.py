"""
Named Entity Recognition (NER) Module for Cross-Lingual Information Retrieval System.

Extracts named entities (PERSON, ORG, GPE, LOC, DATE) using spaCy.
"""

import logging
from typing import List, Dict, Optional, Set
import spacy

logger = logging.getLogger(__name__)


class NERExtractor:
    """
    Extract named entities from text using spaCy.

    Supports extraction of:
    - PERSON: People's names
    - ORG: Organizations, companies, agencies
    - GPE: Geo-Political Entities (countries, cities, states)
    - LOC: Non-GPE locations (mountains, water bodies)
    - DATE: Dates and time expressions

    Example:
        >>> extractor = NERExtractor()
        >>> entities = extractor.extract_entities("Sheikh Hasina visited Dhaka.", "en")
        >>> print(entities)
        [{'text': 'Sheikh Hasina', 'type': 'PERSON', 'start': 0, 'end': 13},
         {'text': 'Dhaka', 'type': 'GPE', 'start': 22, 'end': 27}]
    """

    def __init__(self, entity_types: Optional[Set[str]] = None):
        """
        Initialize the NERExtractor.

        Args:
            entity_types: Set of entity types to extract.
                         If None, extracts PERSON, ORG, GPE, LOC, DATE
        """
        self.entity_types = entity_types or {"PERSON", "ORG", "GPE", "LOC", "DATE"}

        # Initialize spaCy models
        self.nlp_en = None
        self.nlp_bn = None
        self._load_models()

        logger.info(f"NERExtractor initialized for entity types: {self.entity_types}")

    def _load_models(self):
        """Load spaCy models with NER capabilities."""
        # Load English model with NER
        try:
            self.nlp_en = spacy.load("en_core_web_sm")
            logger.info("Loaded English spaCy model with NER: en_core_web_sm")
        except Exception as e:
            logger.error(f"Could not load English NER model: {e}")
            logger.warning("NER will not be available for English")

        # Load Bangla model with NER
        try:
            self.nlp_bn = spacy.load("bn_core_news_sm")
            logger.info("Loaded Bangla spaCy model with NER: bn_core_news_sm")
        except Exception as e:
            logger.warning(f"Could not load Bangla NER model: {e}")
            logger.warning("NER will not be available for Bangla")

    def extract_entities(self, text: str, language: str = "en") -> List[Dict[str, any]]:
        """
        Extract named entities from text.

        Args:
            text: Input text to extract entities from
            language: Language code ('en' or 'bn')

        Returns:
            List of entity dictionaries with keys:
                - text: Entity text
                - type: Entity type (PERSON, ORG, GPE, LOC, DATE)
                - start: Start position in text
                - end: End position in text
                - confidence: Confidence score (if available)

        Example:
            >>> extractor.extract_entities("Sheikh Hasina is the PM of Bangladesh", "en")
            [{'text': 'Sheikh Hasina', 'type': 'PERSON', 'start': 0, 'end': 13},
             {'text': 'Bangladesh', 'type': 'GPE', 'start': 28, 'end': 38}]
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for NER extraction")
            return []

        try:
            # Get appropriate NLP model
            nlp = self._get_nlp_model(language)

            if nlp is None:
                logger.error(f"No NER model available for language: {language}")
                return []

            # Process text
            doc = nlp(text)
            entities = []

            for ent in doc.ents:
                # Filter by entity type
                if ent.label_ not in self.entity_types:
                    continue

                entity = {
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }

                # Add confidence if available (some models provide it)
                if hasattr(ent, "confidence"):
                    entity["confidence"] = round(ent.confidence, 4)

                entities.append(entity)

            logger.debug(
                f"Extracted {len(entities)} entities from text: '{text[:50]}...'"
            )
            return entities

        except Exception as e:
            logger.error(f"Error during NER extraction: {e}")
            return []

    def extract_entities_batch(
        self, texts: List[str], language: str = "en"
    ) -> List[List[Dict[str, any]]]:
        """
        Extract entities from multiple texts efficiently.

        Args:
            texts: List of texts to process
            language: Language code ('en' or 'bn')

        Returns:
            List of entity lists, one per input text

        Example:
            >>> texts = ["Sheikh Hasina visited Dhaka", "Bill Gates founded Microsoft"]
            >>> extractor.extract_entities_batch(texts, "en")
            [[{'text': 'Sheikh Hasina', 'type': 'PERSON', ...}, ...],
             [{'text': 'Bill Gates', 'type': 'PERSON', ...}, ...]]
        """
        if not texts:
            logger.warning("Empty text list provided for batch NER extraction")
            return []

        try:
            nlp = self._get_nlp_model(language)

            if nlp is None:
                logger.error(f"No NER model available for language: {language}")
                return [[] for _ in texts]

            # Use spaCy's pipe for efficient batch processing
            results = []

            for doc in nlp.pipe(texts, batch_size=50):
                entities = []
                for ent in doc.ents:
                    if ent.label_ not in self.entity_types:
                        continue

                    entity = {
                        "text": ent.text,
                        "type": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                    entities.append(entity)

                results.append(entities)

            logger.info(f"Batch NER extraction completed for {len(texts)} texts")
            return results

        except Exception as e:
            logger.error(f"Error during batch NER extraction: {e}")
            return [[] for _ in texts]

    def filter_by_type(
        self, entities: List[Dict[str, any]], entity_type: str
    ) -> List[Dict[str, any]]:
        """
        Filter entities by type.

        Args:
            entities: List of entity dictionaries
            entity_type: Entity type to filter for (PERSON, ORG, GPE, LOC, DATE)

        Returns:
            Filtered list of entities matching the type

        Example:
            >>> entities = [{'text': 'Dhaka', 'type': 'GPE'},
            ...            {'text': 'Sheikh Hasina', 'type': 'PERSON'}]
            >>> extractor.filter_by_type(entities, 'PERSON')
            [{'text': 'Sheikh Hasina', 'type': 'PERSON'}]
        """
        if not entities:
            return []

        filtered = [ent for ent in entities if ent.get("type") == entity_type]
        logger.debug(f"Filtered {len(filtered)} entities of type {entity_type}")
        return filtered

    def get_unique_entities(
        self, entities: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Get unique entities (deduplicate by text).

        Args:
            entities: List of entity dictionaries

        Returns:
            List of unique entities
        """
        if not entities:
            return []

        seen = set()
        unique = []

        for entity in entities:
            entity_text = entity["text"].lower()
            if entity_text not in seen:
                seen.add(entity_text)
                unique.append(entity)

        logger.debug(
            f"Reduced {len(entities)} entities to {len(unique)} unique entities"
        )
        return unique

    def get_entity_counts(self, entities: List[Dict[str, any]]) -> Dict[str, int]:
        """
        Get count of entities by type.

        Args:
            entities: List of entity dictionaries

        Returns:
            Dictionary mapping entity types to counts

        Example:
            >>> entities = [{'type': 'PERSON'}, {'type': 'GPE'}, {'type': 'PERSON'}]
            >>> extractor.get_entity_counts(entities)
            {'PERSON': 2, 'GPE': 1}
        """
        counts = {}
        for entity in entities:
            entity_type = entity.get("type", "UNKNOWN")
            counts[entity_type] = counts.get(entity_type, 0) + 1
        return counts

    def _get_nlp_model(self, language: str):
        """
        Get the appropriate spaCy model for language.

        Args:
            language: Language code ('en' or 'bn')

        Returns:
            spaCy language model or None
        """
        if language == "en":
            return self.nlp_en
        elif language == "bn":
            return self.nlp_bn
        else:
            logger.warning(f"Unsupported language: {language}, defaulting to English")
            return self.nlp_en
