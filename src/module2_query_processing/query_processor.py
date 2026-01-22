"""
Query Processing for Cross-Lingual Information Retrieval
"""

import logging
from typing import Dict, Optional

from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QueryProcessor:
    """Process queries for cross-lingual retrieval."""
    
    def __init__(self):
        self.supported_languages = {'en', 'bn'}
    
    def detect_language(self, text: str) -> Optional[str]:
        """Detect language ('en' or 'bn')."""
        if not text or not text.strip():
            return None
        
        try:
            detected_lang = detect(text)
            return detected_lang if detected_lang in self.supported_languages else None
        except LangDetectException as e:
            logging.error(f"Language detection failed: {e}")
            return None
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Translate text between languages."""
        if not text or not text.strip():
            return None
        
        if source_lang == target_lang:
            return text
        
        try:
            lang_map = {'en': 'english', 'bn': 'bengali'}
            source_full = lang_map.get(source_lang, source_lang)
            target_full = lang_map.get(target_lang, target_lang)
            
            translator = GoogleTranslator(source=source_full, target=target_full)
            return translator.translate(text)
        except Exception as e:
            logging.error(f"Translation failed ({source_lang} -> {target_lang}): {e}")
            return None
    
    def process_query(self, text: str) -> Dict[str, Optional[str]]:
        """Process query: detect language and translate to opposite language."""
        result = {
            'original_text': text,
            'translated_text': None,
            'source_lang': None,
            'target_lang': None
        }
        
        source_lang = self.detect_language(text)
        if source_lang is None:
            source_lang = 'en'  # Default to English if detection fails
        
        result['source_lang'] = source_lang
        target_lang = 'bn' if source_lang == 'en' else 'en'
        result['target_lang'] = target_lang
        
        result['translated_text'] = self.translate_text(text, source_lang, target_lang)
        
        return result
