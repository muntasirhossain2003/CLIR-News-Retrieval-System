"""
Module B: Query Processing

This module provides query processing and translation capabilities for the CLIR system.
"""

from .query_detector import QueryLanguageDetector
from .query_normalizer import QueryNormalizer
from .query_translator import QueryTranslator
from .query_expander import QueryExpander
from .ne_mapper import NamedEntityMapper
from .query_pipeline import QueryProcessor

__all__ = [
    "QueryLanguageDetector",
    "QueryNormalizer",
    "QueryTranslator",
    "QueryExpander",
    "NamedEntityMapper",
    "QueryProcessor",
]
