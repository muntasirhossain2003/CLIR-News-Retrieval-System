"""
Query Expander Module for Cross-Lingual Information Retrieval System.

Expands queries with synonyms and related terms to improve recall.
"""

import logging
from typing import List, Set, Dict, Optional

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Expand queries with synonyms and related terms.

    Features:
    - English synonym expansion using NLTK WordNet
    - Bangla word expansion using embeddings (fallback to simple expansion)
    - Configurable expansion limits

    Example:
        >>> expander = QueryExpander()
        >>> expanded = expander.expand("education policy", "en", max_synonyms=2)
        >>> print(expanded)
        'education learning school policy'
    """

    def __init__(self, max_synonyms: int = 3):
        """
        Initialize the QueryExpander.

        Args:
            max_synonyms: Maximum number of synonyms to add per word
        """
        self.max_synonyms = max_synonyms

        # WordNet for English synonyms
        self.wordnet = None
        self._load_wordnet()

        # Bangla expansion dictionary (simplified - in production, use embeddings)
        self.bangla_synonyms = self._load_bangla_synonyms()

        logger.info(f"QueryExpander initialized (max_synonyms={max_synonyms})")

    def _load_wordnet(self):
        """Load NLTK WordNet for English synonyms."""
        try:
            import nltk
            from nltk.corpus import wordnet

            # Download WordNet if not available
            try:
                wordnet.synsets("test")
            except LookupError:
                logger.info("Downloading WordNet data...")
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)

            self.wordnet = wordnet
            logger.info("WordNet loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load WordNet: {e}")
            logger.warning("English synonym expansion will be limited")

    def _load_bangla_synonyms(self) -> Dict[str, List[str]]:
        """
        Load Bangla synonym dictionary.

        In production, this would use word embeddings or a proper synonym database.
        For now, using a small manual dictionary as fallback.
        """
        return {
            "শিক্ষা": ["শিক্ষাদান", "পড়াশোনা", "অধ্যয়ন"],
            "নীতি": ["নিয়ম", "আইন", "বিধান"],
            "সরকার": ["প্রশাসন", "শাসন"],
            "ক্রিকেট": ["খেলা", "ক্রীড়া"],
            "বাংলাদেশ": ["বাংলা", "দেশ"],
            "প্রধানমন্ত্রী": ["পিএম", "মুখ্যমন্ত্রী"],
            "শেখ": ["নেতা"],
            "ঢাকা": ["রাজধানী", "শহর"],
        }

    def expand(
        self, query: str, language: str = "en", max_synonyms: Optional[int] = None
    ) -> str:
        """
        Expand query with synonyms and related terms.

        Args:
            query: Query text to expand
            language: Language code ('en' or 'bn')
            max_synonyms: Maximum synonyms per word (overrides default)

        Returns:
            Expanded query string

        Example:
            >>> expander.expand("education", "en", max_synonyms=2)
            'education learning school'
        """
        if not query or not query.strip():
            return ""

        max_syn = max_synonyms if max_synonyms is not None else self.max_synonyms

        if language == "en":
            return self._expand_english(query, max_syn)
        elif language == "bn":
            return self._expand_bangla(query, max_syn)
        else:
            logger.warning(f"Unsupported language for expansion: {language}")
            return query

    def _expand_english(self, query: str, max_synonyms: int) -> str:
        """
        Expand English query using WordNet.

        Args:
            query: English query
            max_synonyms: Maximum synonyms per word

        Returns:
            Expanded query
        """
        words = query.lower().split()
        expanded_words = []

        for word in words:
            # Keep original word
            expanded_words.append(word)

            # Get synonyms
            synonyms = self.get_synonyms_english(word, max_synonyms)
            expanded_words.extend(synonyms)

        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for w in expanded_words:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)

        expanded = " ".join(unique_words)
        logger.debug(f"Expanded '{query}' -> '{expanded}'")
        return expanded

    def _expand_bangla(self, query: str, max_synonyms: int) -> str:
        """
        Expand Bangla query using word embeddings or synonym dictionary.

        Args:
            query: Bangla query
            max_synonyms: Maximum synonyms per word

        Returns:
            Expanded query
        """
        words = query.split()
        expanded_words = []

        for word in words:
            # Keep original word
            expanded_words.append(word)

            # Get synonyms from dictionary
            synonyms = self.get_similar_words_bangla(word, max_synonyms)
            expanded_words.extend(synonyms)

        # Remove duplicates
        seen = set()
        unique_words = []
        for w in expanded_words:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)

        expanded = " ".join(unique_words)
        logger.debug(f"Expanded '{query}' -> '{expanded}'")
        return expanded

    def get_synonyms_english(self, word: str, max_count: int = 3) -> List[str]:
        """
        Get English synonyms for a word using WordNet.

        Args:
            word: Word to find synonyms for
            max_count: Maximum number of synonyms to return

        Returns:
            List of synonyms

        Example:
            >>> expander.get_synonyms_english("education", max_count=3)
            ['learning', 'instruction', 'teaching']
        """
        if not self.wordnet:
            return []

        synonyms = set()

        try:
            # Get synsets for the word
            synsets = self.wordnet.synsets(word)

            for synset in synsets[:3]:  # Check first 3 synsets
                # Get lemmas (word forms)
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace("_", " ").lower()

                    # Skip if same as original word
                    if synonym == word.lower():
                        continue

                    # Skip multi-word synonyms for simplicity
                    if " " in synonym:
                        continue

                    synonyms.add(synonym)

                    if len(synonyms) >= max_count:
                        break

                if len(synonyms) >= max_count:
                    break

            return list(synonyms)[:max_count]

        except Exception as e:
            logger.debug(f"Error getting synonyms for '{word}': {e}")
            return []

    def get_similar_words_bangla(self, word: str, max_count: int = 3) -> List[str]:
        """
        Get similar Bangla words using embeddings or dictionary.

        Args:
            word: Bangla word
            max_count: Maximum number of similar words

        Returns:
            List of similar words

        Example:
            >>> expander.get_similar_words_bangla("শিক্ষা", max_count=2)
            ['শিক্ষাদান', 'পড়াশোনা']
        """
        # Use synonym dictionary
        similar = self.bangla_synonyms.get(word, [])
        return similar[:max_count]

    def expand_with_weights(self, query: str, language: str = "en") -> Dict[str, float]:
        """
        Expand query and return terms with weights.

        Original terms get weight 1.0, synonyms get lower weights.

        Args:
            query: Query text
            language: Language code

        Returns:
            Dictionary mapping terms to weights

        Example:
            >>> expander.expand_with_weights("education", "en")
            {'education': 1.0, 'learning': 0.7, 'instruction': 0.7}
        """
        if not query:
            return {}

        words = query.lower().split() if language == "en" else query.split()
        weighted_terms = {}

        for word in words:
            # Original word gets weight 1.0
            weighted_terms[word] = 1.0

            # Get synonyms
            if language == "en":
                synonyms = self.get_synonyms_english(word, self.max_synonyms)
            else:
                synonyms = self.get_similar_words_bangla(word, self.max_synonyms)

            # Synonyms get decreasing weights
            for i, syn in enumerate(synonyms):
                weight = 0.7 - (i * 0.1)  # 0.7, 0.6, 0.5, ...
                weighted_terms[syn] = max(weight, 0.3)

        return weighted_terms

    def disable_expansion(self, query: str) -> bool:
        """
        Check if query should skip expansion.

        Skip expansion for:
        - Very short queries (< 2 words)
        - Queries with quotes (exact match intent)
        - Queries with special operators

        Args:
            query: Query text

        Returns:
            True if expansion should be disabled
        """
        # Skip very short queries
        if len(query.split()) < 2:
            return True

        # Skip quoted queries
        if '"' in query or '"' in query or '"' in query:
            return True

        # Skip queries with operators
        if any(op in query for op in ["AND", "OR", "NOT", "+", "-"]):
            return True

        return False
