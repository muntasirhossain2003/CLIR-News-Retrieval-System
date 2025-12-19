"""
Multilingual Tokenizer Module for Cross-Lingual Information Retrieval System.

Supports tokenization for English and Bangla using spaCy with fallback mechanisms.
"""

import logging
import re
from typing import List, Dict, Tuple, Optional, Set
import spacy
from spacy.lang.en import English
from spacy.lang.bn import Bengali

logger = logging.getLogger(__name__)


class MultilingualTokenizer:
    """
    Tokenizer supporting English and Bangla languages using spaCy.

    Features:
    - Language-specific tokenization
    - Optional lowercase transformation
    - Stopword removal (optional)
    - Punctuation handling
    - Position tracking
    - Fallback for missing spaCy models

    Example:
        >>> tokenizer = MultilingualTokenizer()
        >>> tokens = tokenizer.tokenize("Hello world!", "en")
        >>> print(tokens)
        ['hello', 'world']
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = False,
        remove_punctuation: bool = True,
        min_token_length: int = 1,
    ):
        """
        Initialize the MultilingualTokenizer.

        Args:
            lowercase: Convert tokens to lowercase (English only)
            remove_stopwords: Remove stopwords from tokens
            remove_punctuation: Remove punctuation tokens
            min_token_length: Minimum length for tokens to keep
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.min_token_length = min_token_length

        # Initialize spaCy models
        self.nlp_en = None
        self.nlp_bn = None
        self._load_models()

        logger.info(
            f"MultilingualTokenizer initialized (lowercase={lowercase}, "
            f"remove_stopwords={remove_stopwords}, remove_punct={remove_punctuation})"
        )

    def _load_models(self):
        """Load spaCy models for English and Bangla with fallback."""
        # Load English model
        try:
            self.nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            logger.info("Loaded English spaCy model: en_core_web_sm")
        except Exception as e:
            logger.warning(f"Could not load en_core_web_sm, using blank English: {e}")
            try:
                self.nlp_en = English()
                logger.info("Using blank English tokenizer")
            except Exception as e2:
                logger.error(f"Failed to initialize English tokenizer: {e2}")

        # Load Bangla model
        try:
            self.nlp_bn = spacy.load("bn_core_news_sm", disable=["parser", "ner"])
            logger.info("Loaded Bangla spaCy model: bn_core_news_sm")
        except Exception as e:
            logger.warning(f"Could not load bn_core_news_sm, using blank Bengali: {e}")
            try:
                self.nlp_bn = Bengali()
                logger.info("Using blank Bengali tokenizer")
            except Exception as e2:
                logger.error(f"Failed to initialize Bengali tokenizer: {e2}")

    def tokenize(self, text: str, language: str = "en") -> List[str]:
        """
        Tokenize text based on language.

        Args:
            text: Input text to tokenize
            language: Language code ('en' or 'bn')

        Returns:
            List of tokens

        Example:
            >>> tokenizer.tokenize("Hello, World!", "en")
            ['hello', 'world']
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for tokenization")
            return []

        try:
            # Get appropriate nlp model
            nlp = self._get_nlp_model(language)

            if nlp is None:
                logger.error(f"No tokenizer available for language: {language}")
                return self._fallback_tokenize(text, language)

            # Process text
            doc = nlp(text)
            tokens = []

            for token in doc:
                # Skip punctuation if configured
                if self.remove_punctuation and token.is_punct:
                    continue

                # Skip stopwords if configured
                if self.remove_stopwords and token.is_stop:
                    continue

                # Get token text
                token_text = token.text

                # Lowercase for English
                if self.lowercase and language == "en":
                    token_text = token_text.lower()

                # Skip short tokens
                if len(token_text) < self.min_token_length:
                    continue

                # Skip whitespace-only tokens
                if not token_text.strip():
                    continue

                tokens.append(token_text)

            logger.debug(f"Tokenized '{text[:50]}...' -> {len(tokens)} tokens")
            return tokens

        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            return self._fallback_tokenize(text, language)

    def tokenize_with_positions(
        self, text: str, language: str = "en"
    ) -> List[Dict[str, any]]:
        """
        Tokenize text and return tokens with their positions.

        Args:
            text: Input text to tokenize
            language: Language code ('en' or 'bn')

        Returns:
            List of dictionaries containing token, start, end positions

        Example:
            >>> tokenizer.tokenize_with_positions("Hello world", "en")
            [{'token': 'hello', 'start': 0, 'end': 5},
             {'token': 'world', 'start': 6, 'end': 11}]
        """
        if not text or not text.strip():
            return []

        try:
            nlp = self._get_nlp_model(language)

            if nlp is None:
                logger.error(f"No tokenizer available for language: {language}")
                return []

            doc = nlp(text)
            tokens_with_pos = []

            for token in doc:
                # Apply same filters as tokenize()
                if self.remove_punctuation and token.is_punct:
                    continue
                if self.remove_stopwords and token.is_stop:
                    continue

                token_text = token.text
                if self.lowercase and language == "en":
                    token_text = token_text.lower()

                if len(token_text) < self.min_token_length:
                    continue
                if not token_text.strip():
                    continue

                tokens_with_pos.append(
                    {
                        "token": token_text,
                        "start": token.idx,
                        "end": token.idx + len(token.text),
                    }
                )

            return tokens_with_pos

        except Exception as e:
            logger.error(f"Error during tokenization with positions: {e}")
            return []

    def get_token_count(self, text: str, language: str = "en") -> int:
        """
        Get the count of tokens in text.

        Args:
            text: Input text
            language: Language code ('en' or 'bn')

        Returns:
            Number of tokens

        Example:
            >>> tokenizer.get_token_count("Hello world!", "en")
            2
        """
        return len(self.tokenize(text, language))

    def _get_nlp_model(self, language: str):
        """
        Get the appropriate spaCy model for language.

        Args:
            language: Language code ('en' or 'bn')

        Returns:
            spaCy language model or None
        """
        if language == "en":
            return self.nlp_en
        elif language == "bn":
            return self.nlp_bn
        else:
            logger.warning(f"Unsupported language: {language}, defaulting to English")
            return self.nlp_en

    def _fallback_tokenize(self, text: str, language: str) -> List[str]:
        """
        Fallback tokenization using simple regex splitting.

        Args:
            text: Input text
            language: Language code

        Returns:
            List of tokens
        """
        logger.info(f"Using fallback tokenization for language: {language}")

        # Simple whitespace and punctuation splitting
        tokens = re.findall(r"\w+", text, re.UNICODE)

        if self.lowercase and language == "en":
            tokens = [t.lower() for t in tokens]

        # Filter by length
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        return tokens

    def get_stopwords(self, language: str = "en") -> Set[str]:
        """
        Get stopwords for specified language.

        Args:
            language: Language code ('en' or 'bn')

        Returns:
            Set of stopwords
        """
        nlp = self._get_nlp_model(language)
        if nlp and hasattr(nlp.Defaults, "stop_words"):
            return nlp.Defaults.stop_words
        return set()

    def batch_tokenize(self, texts: List[str], language: str = "en") -> List[List[str]]:
        """
        Tokenize multiple texts efficiently.

        Args:
            texts: List of texts to tokenize
            language: Language code ('en' or 'bn')

        Returns:
            List of token lists
        """
        if not texts:
            return []

        results = []
        for text in texts:
            tokens = self.tokenize(text, language)
            results.append(tokens)

        logger.info(f"Batch tokenized {len(texts)} texts")
        return results
