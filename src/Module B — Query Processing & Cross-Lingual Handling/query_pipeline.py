"""
Complete Query Processing Pipeline

Integrates language detection, normalization, and named entity extraction.
"""

from language_detection_normalization import process_query
from named_entity_extraction import process_query_with_entities


def process_complete_query(user_query: str) -> dict:
    """
    Complete query processing pipeline.

    Step 1: Detect language and normalize
    Step 2: Extract named entities

    Args:
        user_query: Raw user input string

    Returns:
        Dictionary containing:
        - original_query: The raw input
        - language: Detected language ('bn' or 'en')
        - normalized_query: Normalized text
        - entities: List of extracted named entities

    Example:
        >>> result = process_complete_query("Sheikh Hasina visits India")
        >>> print(result)
        {
            'original_query': 'Sheikh Hasina visits India',
            'language': 'en',
            'normalized_query': 'sheikh hasina visits india',
            'entities': ['sheikh hasina', 'india']
        }
    """
    # Step 1: Language detection & normalization
    result = process_query(user_query)

    # Step 2: Named entity extraction
    enhanced_result = process_query_with_entities(result)

    return enhanced_result
