"""
Query Translation for Cross-Lingual Information Retrieval.
"""

import logging
import time

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# Translation cache (LRU with TTL)
class TranslationCache:
    def __init__(self, max_size=5000, ttl=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl  # seconds
        self.hits = 0
        self.misses = 0

    def get(self, text, src, tgt):
        key = f"{text}|{src}|{tgt}"
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["time"] < self.ttl:
                self.hits += 1
                return entry["translation"]
            else:
                del self.cache[key]
        self.misses += 1
        return None

    def put(self, text, src, tgt, translation):
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        key = f"{text}|{src}|{tgt}"
        self.cache[key] = {"translation": translation, "time": time.time()}

    def stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
        }


_translation_cache = TranslationCache()

# Common query translations for instant lookup (avoid API calls)
COMMON_TRANSLATIONS = {
    # English -> Bangla
    ("climate change", "en", "bn"): "জলবায়ু পরিবর্তন",
    ("weather", "en", "bn"): "আবহাওয়া",
    ("economy", "en", "bn"): "অর্থনীতি",
    ("politics", "en", "bn"): "রাজনীতি",
    ("covid", "en", "bn"): "কোভিড",
    ("bangladesh", "en", "bn"): "বাংলাদেশ",
    ("mobile phone", "en", "bn"): "মোবাইল ফোন",
    ("election", "en", "bn"): "নির্বাচন",
    ("government", "en", "bn"): "সরকার",
    # Bangla -> English
    ("জলবায়ু পরিবর্তন", "bn", "en"): "climate change",
    ("আবহাওয়া", "bn", "en"): "weather",
    ("অর্থনীতি", "bn", "en"): "economy",
    ("রাজনীতি", "bn", "en"): "politics",
    ("কোভিড", "bn", "en"): "covid",
    ("বাংলাদেশ", "bn", "en"): "bangladesh",
    ("মোবাইল ফোন", "bn", "en"): "mobile phone",
    ("নির্বাচন", "bn", "en"): "election",
    ("সরকার", "bn", "en"): "government",
}


def get_cache_stats():
    """Get translation cache statistics."""
    return _translation_cache.stats()


def translate_query(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate query with caching and fallback strategies."""
    if src_lang == tgt_lang or not text.strip():
        return text

    # Check common translations first (instant)
    common_key = (text.lower(), src_lang, tgt_lang)
    if common_key in COMMON_TRANSLATIONS:
        logger.info(
            f"Using common translation: {text} -> {COMMON_TRANSLATIONS[common_key]}"
        )
        return COMMON_TRANSLATIONS[common_key]

    # Check cache (instant if cached)
    cached = _translation_cache.get(text, src_lang, tgt_lang)
    if cached:
        return cached

    # Translate with fallback: deep-translator -> googletrans
    translated = None

    try:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source=src_lang, target=tgt_lang)
        translated = translator.translate(text)
        logger.debug("Translation: deep-translator succeeded")
    except ImportError:
        logger.debug("deep-translator not available, trying googletrans")
    except Exception as e:
        logger.warning(f"deep-translator failed: {e}")

    if not translated:
        try:
            from googletrans import Translator

            translator = Translator()
            result = translator.translate(text, src=src_lang, dest=tgt_lang)
            translated = result.text if result and result.text else None
            logger.debug("Translation: googletrans succeeded")
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text

    if not translated or translated.strip() == "":
        return text

    # Cache successful translation
    _translation_cache.put(text, src_lang, tgt_lang, translated)
    return translated


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
