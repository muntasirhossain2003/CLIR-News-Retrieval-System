"""
Module A: Indexing

This module provides indexing and preprocessing capabilities for the CLIR system.
"""

from .language_detector import LanguageDetector
from .tokenizer import MultilingualTokenizer
from .ner_extractor import NERExtractor
from .inverted_index import InvertedIndex
from .document_processor import DocumentProcessor

__all__ = [
    "LanguageDetector",
    "MultilingualTokenizer",
    "NERExtractor",
    "InvertedIndex",
    "DocumentProcessor",
]
