"""
Module B: Query Processing & Cross-Lingual Handling

This module handles query preprocessing for the CLIR system.
Part 1: Language detection and normalization only.
"""

from .query_processor_part1 import (
    detect_query_language,
    normalize_query,
    process_query_part1,
)

__all__ = ["detect_query_language", "normalize_query", "process_query_part1"]
