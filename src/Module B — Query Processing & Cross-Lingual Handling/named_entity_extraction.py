"""
Module B - Named Entity Extraction

This module extracts named entities from user queries to improve retrieval accuracy.
Named entities (people, places, organizations) are often the most important search terms.
"""

import logging

# Set up logging for debugging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Module-level lazy loading for NLP models (load once, reuse)
_spacy_nlp = None
_stanza_nlp = None


def extract_query_entities(query: str, lang: str) -> list:
    """
    Extract named entities from a query using language-specific NLP models.

    Supported entity types:
    - PERSON: Names of people
    - LOCATION / GPE: Geographic locations, cities, countries
    - ORGANIZATION: Companies, institutions, agencies

    Args:
        query: The text to extract entities from
        lang: Language code ('bn' for Bangla, 'en' for English)

    Returns:
        List of entity strings (empty list if none found or model unavailable)

    Example:
        >>> extract_query_entities("Dhaka University students protest", "en")
        ['Dhaka University']

        >>> extract_query_entities("ঢাকা বিশ্ববিদ্যালয়ের ছাত্র", "bn")
        ['ঢাকা বিশ্ববিদ্যালয়']
    """
    if not query or not query.strip():
        return []

    entities = []

    if lang == "en":
        # Use spaCy for English named entity recognition
        global _spacy_nlp

        try:
            import spacy

            # Load English model once (lazy initialization)
            if _spacy_nlp is None:
                try:
                    _spacy_nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning(
                        "spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm"
                    )
                    return []

            # Process the query
            doc = _spacy_nlp(query)

            # Extract relevant entity types
            for ent in doc.ents:
                # Filter for PERSON, GPE (Geo-Political Entity), ORG, LOC
                if ent.label_ in ["PERSON", "GPE", "ORG", "LOC", "FAC"]:
                    entities.append(ent.text)

        except ImportError:
            logger.warning("spaCy not installed. Install with: pip install spacy")
            return []
        except Exception as e:
            logger.error(f"Error in English entity extraction: {e}")
            return []

    elif lang == "bn":
        # Use Stanza for Bangla named entity recognition
        global _stanza_nlp

        try:
            import stanza

            # Load Bangla model once (lazy initialization)
            if _stanza_nlp is None:
                try:
                    _stanza_nlp = stanza.Pipeline("bn", processors="tokenize,ner")
                except Exception as e:
                    # If model not downloaded, provide graceful fallback
                    logger.warning(f"Stanza Bangla model not available: {e}")
                    logger.info("Download with: import stanza; stanza.download('bn')")
                    return []

            # Process the query
            doc = _stanza_nlp(query)

            # Extract named entities using sentence-level entity spans
            for sentence in doc.sentences:
                for ent in sentence.ents:
                    # Filter for relevant entity types (PER, LOC, ORG)
                    if ent.type in ["PER", "LOC", "ORG"]:
                        # Extract full entity text (preserves multi-word entities)
                        entities.append(ent.text)

        except ImportError:
            logger.warning("Stanza not installed. Install with: pip install stanza")
            return []
        except Exception as e:
            logger.error(f"Error in Bangla entity extraction: {e}")
            return []

    else:
        # Unsupported language
        logger.warning(f"Entity extraction not supported for language: {lang}")
        return []

    # Remove duplicates while preserving order
    seen = set()
    unique_entities = []
    for entity in entities:
        if entity not in seen:
            seen.add(entity)
            unique_entities.append(entity)

    return unique_entities


def process_query_with_entities(query_obj: dict) -> dict:
    """
    Add named entity extraction to query processing pipeline.

    Takes output from language detection/normalization and adds entity information.

    Args:
        query_obj: Dictionary from process_query() containing:
            - original_query: str
            - language: str ('bn' or 'en')
            - normalized_query: str

    Returns:
        Enhanced dictionary with additional field:
            - entities: list of named entity strings

    Example:
        >>> from language_detection_normalization import process_query
        >>> result = process_query("Prime Minister visits Dhaka")
        >>> enhanced = process_query_with_entities(result)
        >>> print(enhanced['entities'])
        ['Prime Minister', 'Dhaka']
    """
    # Validate input
    if not query_obj or not isinstance(query_obj, dict):
        logger.error(
            "Invalid input: expected dictionary from language detection module"
        )
        return {
            "original_query": "",
            "language": "en",
            "normalized_query": "",
            "entities": [],
        }

    # Get required fields (with defaults)
    normalized = query_obj.get("normalized_query", "")
    lang = query_obj.get("language", "en")

    # Extract entities from the normalized query
    # Use normalized version for consistency with indexing
    entities = extract_query_entities(normalized, lang)

    # Add entities to the query object
    result = query_obj.copy()
    result["entities"] = entities

    return result
