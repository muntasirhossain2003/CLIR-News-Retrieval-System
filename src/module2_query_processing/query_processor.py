import logging
from typing import Dict, Optional
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator


class QueryProcessor:
    
    def __init__(self):
        self.supported_languages = {'en', 'bn'}
    
    def detect_language(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        
        try:
            detected_lang = detect(text)
            return detected_lang if detected_lang in self.supported_languages else None
        except LangDetectException:
            return None
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
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
        except Exception:
            return None
    
    def process_query(self, text: str) -> Dict[str, Optional[str]]:
        result = {
            'original_text': text,
            'translated_text': None,
            'source_lang': None,
            'target_lang': None
        }
        
        source_lang = self.detect_language(text)
        if source_lang is None:
            source_lang = 'en'
        
        result['source_lang'] = source_lang
        target_lang = 'bn' if source_lang == 'en' else 'en'
        result['target_lang'] = target_lang
        
        result['translated_text'] = self.translate_text(text, source_lang, target_lang)
        
        return result
