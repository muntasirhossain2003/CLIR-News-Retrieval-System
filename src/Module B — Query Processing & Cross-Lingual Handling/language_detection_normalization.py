"""
Module B - Language Detection & Normalization

This module provides query language detection and text normalization.
Designed to be simple and explainable for academic purposes.
"""

import unicodedata
import os


def detect_query_language(query: str) -> str:
    """
    Detect whether the query is Bangla ('bn') or English ('en').

    Uses two methods:
    1. fastText language identification (if model available)
    2. Unicode-range fallback for Bangla characters

    Args:
        query: User's search query string

    Returns:
        'bn' for Bangla or 'en' for English
    """
    if not query or not query.strip():
        return "en"  # Default to English for empty queries

    # Try fastText first (if available)
    try:
        import fasttext

        # Check if fastText model exists
        model_path = os.path.join("models", "lid.176.bin")
        if os.path.exists(model_path):
            model = fasttext.load_model(model_path)
            # fastText returns predictions like ('__label__bn',) with confidence
            predictions = model.predict(query.replace("\n", " "))
            detected_lang = predictions[0][0].replace("__label__", "")

            # Map fastText language codes to our format
            if detected_lang in ["bn", "bn-BD", "bengali"]:
                return "bn"
            else:
                return "en"
    except (ImportError, Exception):
        # If fastText not available, use Unicode-range fallback
        pass

    # Fallback: Check for Bangla Unicode characters (U+0980 to U+09FF)
    bangla_char_count = 0
    total_chars = 0

    for char in query:
        if char.isalpha():  # Only count alphabetic characters
            total_chars += 1
            # Check if character is in Bangla Unicode range
            if "\u0980" <= char <= "\u09ff":
                bangla_char_count += 1

    # If more than 30% of alphabetic characters are Bangla, classify as Bangla
    if total_chars > 0 and (bangla_char_count / total_chars) > 0.3:
        return "bn"
    else:
        return "en"


def normalize_query(query: str, lang: str) -> str:
    """
    Normalize the query text based on language.

    Steps:
    1. Trim leading/trailing whitespace
    2. Normalize Unicode to NFC form (canonical composition)
    3. Lowercase ONLY if English (preserve Bangla case)

    Does NOT remove:
    - Stopwords (needed for semantic search)
    - Punctuation (may be significant)
    - Numbers (could be dates, IDs, etc.)

    Args:
        query: Input query string
        lang: Language code ('bn' or 'en')

    Returns:
        Normalized query string
    """
    if not query:
        return ""

    # Step 1: Trim whitespace
    normalized = query.strip()

    # Step 2: Unicode normalization (NFC)
    # NFC: Canonical composition - combines characters where possible
    # Example: é (e + accent) -> é (single character)
    normalized = unicodedata.normalize("NFC", normalized)

    # Step 3: Lowercase only for English
    # Bangla doesn't have case, and lowercasing can corrupt Bangla text
    if lang == "en":
        normalized = normalized.lower()

    return normalized


def process_query(query: str) -> dict:
    """
    Process a user query: detect language and normalize.

    This is the main entry point for language detection and normalization.

    Args:
        query: Raw user input query

    Returns:
        Dictionary containing:
        - original_query: The input query as-is
        - language: Detected language ('bn' or 'en')
        - normalized_query: Processed query ready for retrieval

    Example:
        >>> process_query("  Climate Change  ")
        {
            'original_query': '  Climate Change  ',
            'language': 'en',
            'normalized_query': 'climate change'
        }

        >>> process_query("জলবায়ু পরিবর্তন")
        {
            'original_query': 'জলবায়ু পরিবর্তন',
            'language': 'bn',
            'normalized_query': 'জলবায়ু পরিবর্তন'
        }
    """
    # Handle edge cases
    if query is None:
        query = ""

    # Step 1: Detect language
    language = detect_query_language(query)

    # Step 2: Normalize based on detected language
    normalized = normalize_query(query, language)

    # Step 3: Return structured result
    return {
        "original_query": query,
        "language": language,
        "normalized_query": normalized,
    }
