"""
Language Detection Module using fastText

Detects whether text is in Bangla (bn) or English (en).
"""

import fasttext
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Detects language using fastText's language identification model.
    Supports Bangla (bn) and English (en).
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize language detector.

        Args:
            model_path: Path to fastText model. If None, downloads lid.176.bin
        """
        self.model = None
        self.model_path = model_path

        if model_path is None:
            # Default to lid.176.bin (176 languages)
            self.model_path = os.path.join("models", "lid.176.bin")

        self._load_model()

    def _load_model(self):
        """Load or download fastText language detection model."""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading fastText model from {self.model_path}")
                # Suppress fastText warnings
                fasttext.FastText.eprint = lambda x: None
                self.model = fasttext.load_model(self.model_path)
            else:
                logger.warning(f"fastText model not found at {self.model_path}")
                logger.info("Downloading fastText lid.176.bin model...")

                # Create models directory
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

                # Download using fasttext
                import urllib.request

                url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
                urllib.request.urlretrieve(url, self.model_path)

                logger.info("Model downloaded successfully")
                fasttext.FastText.eprint = lambda x: None
                self.model = fasttext.load_model(self.model_path)

        except Exception as e:
            logger.error(f"Failed to load fastText model: {e}")
            raise

    def detect(self, text: str, default: str = "en") -> str:
        """
        Detect language of text.

        Args:
            text: Input text to detect language
            default: Default language if detection fails

        Returns:
            Language code ('bn' for Bangla, 'en' for English, or default)
        """
        if not text or len(text.strip()) < 10:
            logger.warning("Text too short for reliable detection, using default")
            return default

        try:
            # Clean text for detection
            text = text.replace("\n", " ").strip()[:500]  # Use first 500 chars

            # Predict language
            predictions = self.model.predict(text, k=2)  # Get top 2 predictions
            labels = predictions[0]
            scores = predictions[1]

            # Extract language code (fastText returns '__label__en' format)
            top_lang = labels[0].replace("__label__", "")
            confidence = scores[0]

            # Map to our supported languages
            if top_lang in ["bn", "bn-BD"]:
                detected = "bn"
            elif top_lang in ["en", "en-US", "en-GB"]:
                detected = "en"
            else:
                # Check if second prediction is our target language
                if len(labels) > 1:
                    second_lang = labels[1].replace("__label__", "")
                    if second_lang in ["bn", "bn-BD"] and scores[1] > 0.3:
                        detected = "bn"
                    elif second_lang in ["en", "en-US", "en-GB"] and scores[1] > 0.3:
                        detected = "en"
                    else:
                        detected = default
                else:
                    detected = default

            logger.debug(
                f"Detected language: {detected} (confidence: {confidence:.3f})"
            )
            return detected

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return default

    def detect_batch(self, texts: list, default: str = "en") -> list:
        """
        Detect languages for multiple texts.

        Args:
            texts: List of texts
            default: Default language if detection fails

        Returns:
            List of language codes
        """
        return [self.detect(text, default) for text in texts]
