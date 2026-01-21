"""
Text Preprocessing Module

Handles language-specific NLP preprocessing:
- English: spaCy (en_core_web_sm)
- Bangla: Stanza
- Named Entity Recognition
- Tokenization and cleaning
"""

import re
import logging
from typing import Dict, List, Tuple
import spacy
import stanza

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Language-aware text preprocessing for English and Bangla.
    Extracts tokens, named entities, and cleaned text.
    """

    def __init__(self):
        """Initialize preprocessor with spaCy and Stanza models."""
        self.en_nlp = None
        self.bn_nlp = None
        self._load_models()

    def _load_models(self):
        """Load NLP models for English and Bangla."""
        logger.info("Loading NLP models...")

        # Load English spaCy model
        try:
            logger.info("Loading spaCy English model (en_core_web_sm)...")
            self.en_nlp = spacy.load("en_core_web_sm")
            logger.info("✓ English model loaded")
        except OSError:
            logger.warning(
                "English model not found. Install with: python -m spacy download en_core_web_sm"
            )
            raise

        # Load Bangla Stanza model
        try:
            logger.info("Loading Stanza Bangla model...")
            # Suppress Stanza's verbose logging
            import sys
            import io

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                self.bn_nlp = stanza.Pipeline(
                    lang="bn",
                    processors="tokenize,ner",
                    verbose=False,
                    download_method=None,  # Don't auto-download
                )
            finally:
                sys.stdout = old_stdout

            logger.info("✓ Bangla model loaded")
        except Exception as e:
            logger.warning(f"Bangla model not found: {e}")
            logger.warning("Install with: import stanza; stanza.download('bn')")
            # Don't raise - allow continuing with English only

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and special characters.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r"http\S+|www.\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def preprocess_english(self, text: str) -> Dict:
        """
        Preprocess English text using spaCy.

        Args:
            text: English text

        Returns:
            Dictionary with cleaned_text, tokens, token_count, entities
        """
        if not self.en_nlp:
            raise RuntimeError("English model not loaded")

        cleaned = self.clean_text(text)

        try:
            doc = self.en_nlp(cleaned)

            # Extract tokens (lemmatized, lowercased, no stop words/punct)
            tokens = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and token.is_alpha
            ]

            # Extract named entities (unique)
            entities = list(set([ent.text for ent in doc.ents]))

            return {
                "cleaned_text": cleaned,
                "tokens": tokens,
                "token_count": len(tokens),
                "entities": entities,
            }

        except Exception as e:
            logger.error(f"English preprocessing failed: {e}")
            return {
                "cleaned_text": cleaned,
                "tokens": cleaned.lower().split(),
                "token_count": len(cleaned.split()),
                "entities": [],
            }

    def preprocess_bangla(self, text: str) -> Dict:
        """
        Preprocess Bangla text using Stanza.

        Args:
            text: Bangla text

        Returns:
            Dictionary with cleaned_text, tokens, token_count, entities
        """
        if not self.bn_nlp:
            logger.warning("Bangla model not loaded, using basic tokenization")
            cleaned = self.clean_text(text)
            tokens = cleaned.split()
            return {
                "cleaned_text": cleaned,
                "tokens": tokens,
                "token_count": len(tokens),
                "entities": [],
            }

        cleaned = self.clean_text(text)

        try:
            doc = self.bn_nlp(cleaned)

            # Extract tokens
            tokens = []
            for sentence in doc.sentences:
                for word in sentence.words:
                    if word.text and len(word.text) > 1:  # Skip single chars
                        tokens.append(word.text)

            # Extract named entities
            entities = []
            for sentence in doc.sentences:
                for ent in sentence.ents:
                    entities.append(ent.text)
            entities = list(set(entities))  # Unique entities

            return {
                "cleaned_text": cleaned,
                "tokens": tokens,
                "token_count": len(tokens),
                "entities": entities,
            }

        except Exception as e:
            logger.error(f"Bangla preprocessing failed: {e}")
            tokens = cleaned.split()
            return {
                "cleaned_text": cleaned,
                "tokens": tokens,
                "token_count": len(tokens),
                "entities": [],
            }

    def preprocess(self, text: str, language: str) -> Dict:
        """
        Preprocess text based on detected language.

        Args:
            text: Input text
            language: Language code ('en' or 'bn')

        Returns:
            Dictionary with preprocessing results
        """
        if language == "bn":
            return self.preprocess_bangla(text)
        else:
            return self.preprocess_english(text)
