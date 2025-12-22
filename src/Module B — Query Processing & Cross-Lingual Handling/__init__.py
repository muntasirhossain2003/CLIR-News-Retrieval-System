"""
Module B: Query Processing & Cross-Lingual Handling

This module handles query preprocessing for the CLIR system.
- Language detection and normalization
- Named entity extraction
- Complete query processing pipeline
"""

from .language_detection_normalization import (
    detect_query_language,
    normalize_query,
    process_query,
)

from .named_entity_extraction import (
    extract_query_entities,
    process_query_with_entities,
)

from .query_pipeline import (
    process_complete_query,
)

__all__ = [
    "detect_query_language",
    "normalize_query",
    "process_query",
    "extract_query_entities",
    "process_query_with_entities",
    "process_complete_query",
]
