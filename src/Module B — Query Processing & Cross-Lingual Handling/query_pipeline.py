"""
Complete Query Processing Pipeline

Integrates language detection, normalization, named entity extraction, and translation.
"""

from language_detection_normalization import process_query
from named_entity_extraction import process_query_with_entities
from query_translation import process_query_with_translation


def process_complete_query(user_query: str, target_lang: str = None) -> dict:
    """
    Complete query processing pipeline.

    Step 1: Detect language and normalize
    Step 2: Extract named entities
    Step 3: Translate if target_lang specified and different from source

    Args:
        user_query: Raw user input string
        target_lang: Optional target language for translation ('bn' or 'en')
                    If None, no translation is performed

    Returns:
        Dictionary containing:
        - original_query: The raw input
        - language: Detected language ('bn' or 'en')
        - normalized_query: Normalized text
        - entities: List of extracted named entities
        - translated_query: Translated query (only if target_lang specified)

    Example:
        >>> # Without translation
        >>> result = process_complete_query("Sheikh Hasina visits India")
        >>> print(result)
        {
            'original_query': 'Sheikh Hasina visits India',
            'language': 'en',
            'normalized_query': 'sheikh hasina visits india',
            'entities': ['sheikh hasina', 'india']
        }

        >>> # With translation (English -> Bangla)
        >>> result = process_complete_query("Climate change", target_lang='bn')
        >>> print(result['translated_query'])
        'জলবায়ু পরিবর্তন'
    """
    # Step 1: Language detection & normalization
    result = process_query(user_query)

    # Step 2: Named entity extraction
    result = process_query_with_entities(result)

    # Step 3: Translation (optional)
    if target_lang is not None:
        result = process_query_with_translation(result, target_lang)

    return result
