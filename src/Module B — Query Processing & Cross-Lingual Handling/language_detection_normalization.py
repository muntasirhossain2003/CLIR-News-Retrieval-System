"""
Module B - Language Detection & Normalization
"""

import unicodedata
import os


def detect_query_language(query: str) -> str:
    """
    Detect whether the query is Bangla ('bn') or English ('en').
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
    Process the query: detect language and normalize.
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
