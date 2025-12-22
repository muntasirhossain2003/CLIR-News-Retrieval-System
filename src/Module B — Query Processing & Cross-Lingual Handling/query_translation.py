"""
Module B - Query Translation

This module provides cross-lingual query translation for CLIR.
Enables searching in one language while retrieving documents in another.
"""

import logging

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def translate_query(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate query text from source language to target language.

    Uses googletrans library (free Google Translate API wrapper).

    Args:
        text: The text to translate
        src_lang: Source language code ('bn' or 'en')
        tgt_lang: Target language code ('bn' or 'en')

    Returns:
        Translated text, or original text if translation fails

    Example:
        >>> translate_query("climate change", "en", "bn")
        'জলবায়ু পরিবর্তন'

        >>> translate_query("ঢাকা বিশ্ববিদ্যালয়", "bn", "en")
        'Dhaka University'
    """
    # If source and target are the same, no translation needed
    if src_lang == tgt_lang:
        return text

    # If text is empty, return as-is
    if not text or not text.strip():
        return text

    try:
        from googletrans import Translator

        # Map our language codes to Google Translate codes
        # 'bn' stays 'bn', 'en' stays 'en'
        lang_map = {"bn": "bn", "en": "en"}
        src = lang_map.get(src_lang, src_lang)
        tgt = lang_map.get(tgt_lang, tgt_lang)

        # Create translator instance
        translator = Translator()

        # Translate
        result = translator.translate(text, src=src, dest=tgt)

        # Explicit validation of translation result
        if not result or not result.text:
            logger.warning(f"Translation returned None or empty result for: '{text}'")
            logger.warning(f"  Reason: googletrans returned invalid response")
            logger.warning(f"  Fallback: Using original text")
            return text

        translated_text = result.text.strip()

        # Check if translation actually changed the text
        # If source != target but output == input, translation failed silently
        if translated_text.lower() == text.lower() and src_lang != tgt_lang:
            logger.warning(
                f"Translation failed silently for: '{text}' ({src_lang} -> {tgt_lang})"
            )
            logger.warning(f"  Reason: Translated text identical to source text")
            logger.warning(f"  Output: '{translated_text}'")
            logger.warning(f"  Fallback: Using original text")
            return text

        # Check if translation is suspiciously short (possible truncation)
        if len(translated_text) < len(text) * 0.3 and len(text) > 10:
            logger.warning(f"Translation suspiciously short for: '{text}'")
            logger.warning(
                f"  Input length: {len(text)}, Output length: {len(translated_text)}"
            )
            logger.warning(f"  Output: '{translated_text}'")
            logger.warning(f"  Fallback: Using original text")
            return text

        logger.info(
            f"Translation successful: '{text}' ({src_lang}) -> '{translated_text}' ({tgt_lang})"
        )
        return translated_text

    except ImportError:
        logger.warning(
            "googletrans not installed. Install with: pip install googletrans==4.0.0-rc1"
        )
        logger.warning("  Fallback: Using original text without translation")
        return text
    except AttributeError as e:
        logger.error(f"Translation API error for '{text}': {e}")
        logger.error("  Possible cause: googletrans API change or connection issue")
        logger.error("  Fallback: Using original text")
        return text
    except Exception as e:
        logger.error(
            f"Unexpected translation error for '{text}': {type(e).__name__}: {e}"
        )
        logger.error("  Fallback: Using original text")
        return text


def process_query_with_translation(query_obj: dict, target_lang: str) -> dict:
    """
    Add query translation to the processing pipeline.

    Takes output from previous stages and adds translated version if needed.

    Args:
        query_obj: Dictionary from previous processing stages containing:
            - original_query: str
            - language: str ('bn' or 'en')
            - normalized_query: str
            - entities: list (optional)

        target_lang: Target language for retrieval ('bn' or 'en')

    Returns:
        Enhanced dictionary with additional field:
            - translated_query: str (translated to target_lang, or normalized_query if same language)

    Example:
        >>> # English query, want to search Bangla documents
        >>> query_obj = {
        ...     'original_query': 'Climate Change',
        ...     'language': 'en',
        ...     'normalized_query': 'climate change',
        ...     'entities': []
        ... }
        >>> result = process_query_with_translation(query_obj, target_lang='bn')
        >>> print(result['translated_query'])
        'জলবায়ু পরিবর্তন'

        >>> # Bangla query, want to search English documents
        >>> query_obj = {
        ...     'original_query': 'জলবায়ু পরিবর্তন',
        ...     'language': 'bn',
        ...     'normalized_query': 'জলবায়ু পরিবর্তন',
        ...     'entities': []
        ... }
        >>> result = process_query_with_translation(query_obj, target_lang='en')
        >>> print(result['translated_query'])
        'climate change'
    """
    # Validate input
    if not query_obj or not isinstance(query_obj, dict):
        logger.error(
            "Invalid input: expected dictionary from previous processing stages"
        )
        return {
            "original_query": "",
            "language": "en",
            "normalized_query": "",
            "entities": [],
            "translated_query": "",
        }

    # Get required fields
    normalized = query_obj.get("normalized_query", "")
    src_lang = query_obj.get("language", "en")

    # Validate target language
    if target_lang not in ["bn", "en"]:
        logger.warning(f"Invalid target language '{target_lang}', defaulting to 'en'")
        target_lang = "en"

    # Translate only if source != target
    if src_lang == target_lang:
        # Same language - no translation needed
        translated = normalized
        logger.info(
            f"Source and target language both '{src_lang}', skipping translation"
        )
    else:
        # Different language - attempt translation
        logger.info(f"Attempting translation: {src_lang} -> {target_lang}")
        translated = translate_query(normalized, src_lang, target_lang)

        # Explicit check: Did translation actually work?
        # If output matches input AND languages differ, translation failed
        if translated == normalized:
            logger.warning(
                f"Translation failed: Output identical to input despite different languages"
            )
            logger.warning(f"  Input: '{normalized}' ({src_lang})")
            logger.warning(f"  Output: '{translated}' ({target_lang})")
            logger.warning(
                f"  Action: Using normalized_query as fallback for CLIR evaluation"
            )
            # Keep the normalized query as fallback - it's better than empty
            # This makes the failure explicit in logs but doesn't break the pipeline
        else:
            logger.info(
                f"Translation succeeded: Using translated query for {target_lang} retrieval"
            )

    # Add translation to query object
    result = query_obj.copy()
    result["translated_query"] = translated

    return result
