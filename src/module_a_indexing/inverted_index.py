"""
Inverted Index Module for Cross-Lingual Information Retrieval System.

Implements an inverted index data structure for efficient document retrieval.
"""

import logging
import json
import pickle
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class InvertedIndex:
    """
    Inverted index for efficient document retrieval.

    Structure:
        - term -> list of (doc_id, frequency, positions)
        - Document metadata storage
        - Support for saving/loading index

    Example:
        >>> index = InvertedIndex()
        >>> tokens = ['hello', 'world', 'hello']
        >>> metadata = {'title': 'Test Doc', 'url': 'http://example.com'}
        >>> index.add_document('doc1', tokens, metadata)
        >>> posting_list = index.get_posting_list('hello')
        >>> print(posting_list)
        [{'doc_id': 'doc1', 'frequency': 2, 'positions': [0, 2]}]
    """

    def __init__(self):
        """Initialize an empty inverted index."""
        # term -> list of {doc_id, frequency, positions}
        self.index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # doc_id -> metadata
        self.doc_metadata: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.total_documents = 0
        self.total_terms = 0
        self.avg_doc_length = 0.0

        logger.info("InvertedIndex initialized")

    def add_document(
        self, doc_id: str, tokens: List[str], metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a document to the inverted index.

        Args:
            doc_id: Unique document identifier
            tokens: List of tokens from the document
            metadata: Optional metadata (title, url, date, language, etc.)

        Returns:
            True if document added successfully, False otherwise

        Example:
            >>> index.add_document('doc1', ['hello', 'world'],
            ...                    {'title': 'Test', 'language': 'en'})
            True
        """
        if not doc_id:
            logger.error("Document ID cannot be empty")
            return False

        if doc_id in self.doc_metadata:
            logger.warning(f"Document {doc_id} already exists, skipping")
            return False

        if not tokens:
            logger.warning(f"Empty token list for document {doc_id}")
            tokens = []

        try:
            # Build term frequency and position mapping
            term_info = defaultdict(lambda: {"frequency": 0, "positions": []})

            for position, token in enumerate(tokens):
                term_info[token]["frequency"] += 1
                term_info[token]["positions"].append(position)

            # Add to inverted index
            for term, info in term_info.items():
                self.index[term].append(
                    {
                        "doc_id": doc_id,
                        "frequency": info["frequency"],
                        "positions": info["positions"],
                    }
                )

            # Store metadata
            if metadata is None:
                metadata = {}

            metadata["doc_id"] = doc_id
            metadata["token_count"] = len(tokens)
            metadata["unique_terms"] = len(term_info)

            self.doc_metadata[doc_id] = metadata

            # Update statistics
            self.total_documents += 1
            self.total_terms = len(self.index)
            self._update_avg_doc_length()

            logger.debug(f"Added document {doc_id} with {len(tokens)} tokens")
            return True

        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {e}")
            return False

    def get_posting_list(self, term: str) -> List[Dict[str, Any]]:
        """
        Get the posting list for a term.

        Args:
            term: Term to look up

        Returns:
            List of dictionaries containing doc_id, frequency, positions
            Returns empty list if term not found

        Example:
            >>> index.get_posting_list('hello')
            [{'doc_id': 'doc1', 'frequency': 2, 'positions': [0, 5]}]
        """
        return self.index.get(term, [])

    def get_term_frequency(self, term: str, doc_id: str) -> int:
        """
        Get frequency of a term in a specific document.

        Args:
            term: Term to look up
            doc_id: Document ID

        Returns:
            Frequency of term in document, 0 if not found
        """
        posting_list = self.get_posting_list(term)
        for posting in posting_list:
            if posting["doc_id"] == doc_id:
                return posting["frequency"]
        return 0

    def get_document_frequency(self, term: str) -> int:
        """
        Get the number of documents containing the term.

        Args:
            term: Term to look up

        Returns:
            Number of documents containing the term

        Example:
            >>> index.get_document_frequency('hello')
            15
        """
        return len(self.get_posting_list(term))

    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a document.

        Args:
            doc_id: Document ID

        Returns:
            Metadata dictionary or None if not found
        """
        return self.doc_metadata.get(doc_id)

    def get_all_terms(self) -> Set[str]:
        """
        Get all unique terms in the index.

        Returns:
            Set of all terms
        """
        return set(self.index.keys())

    def get_document_length(self, doc_id: str) -> int:
        """
        Get the length (token count) of a document.

        Args:
            doc_id: Document ID

        Returns:
            Number of tokens in document, 0 if not found
        """
        metadata = self.get_document_metadata(doc_id)
        return metadata.get("token_count", 0) if metadata else 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary containing:
                - total_documents: Total number of documents
                - total_terms: Total number of unique terms
                - avg_doc_length: Average document length
                - index_size: Size of index in memory
        """
        return {
            "total_documents": self.total_documents,
            "total_terms": self.total_terms,
            "avg_doc_length": round(self.avg_doc_length, 2),
            "index_size_mb": self._estimate_size_mb(),
        }

    def save_index(self, filepath: str, format: str = "pickle") -> bool:
        """
        Save the inverted index to disk.

        Args:
            filepath: Path to save the index
            format: 'pickle' or 'json'

        Returns:
            True if saved successfully, False otherwise

        Example:
            >>> index.save_index('index.pkl', format='pickle')
            True
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            data = {
                "index": dict(self.index),
                "doc_metadata": self.doc_metadata,
                "total_documents": self.total_documents,
                "total_terms": self.total_terms,
                "avg_doc_length": self.avg_doc_length,
            }

            if format == "pickle":
                with open(filepath, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Index saved to {filepath} (pickle format)")

            elif format == "json":
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"Index saved to {filepath} (JSON format)")

            else:
                logger.error(f"Unsupported format: {format}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False

    def load_index(self, filepath: str, format: str = "pickle") -> bool:
        """
        Load the inverted index from disk.

        Args:
            filepath: Path to load the index from
            format: 'pickle' or 'json'

        Returns:
            True if loaded successfully, False otherwise

        Example:
            >>> index.load_index('index.pkl', format='pickle')
            True
        """
        try:
            if not Path(filepath).exists():
                logger.error(f"Index file not found: {filepath}")
                return False

            if format == "pickle":
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"Index loaded from {filepath} (pickle format)")

            elif format == "json":
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Index loaded from {filepath} (JSON format)")

            else:
                logger.error(f"Unsupported format: {format}")
                return False

            # Restore index data
            self.index = defaultdict(list, data["index"])
            self.doc_metadata = data["doc_metadata"]
            self.total_documents = data["total_documents"]
            self.total_terms = data["total_terms"]
            self.avg_doc_length = data["avg_doc_length"]

            logger.info(
                f"Loaded index with {self.total_documents} documents, {self.total_terms} terms"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

    def _update_avg_doc_length(self):
        """Update the average document length statistic."""
        if self.total_documents == 0:
            self.avg_doc_length = 0.0
        else:
            total_length = sum(
                meta.get("token_count", 0) for meta in self.doc_metadata.values()
            )
            self.avg_doc_length = total_length / self.total_documents

    def _estimate_size_mb(self) -> float:
        """Estimate the size of the index in MB."""
        try:
            import sys

            size_bytes = sys.getsizeof(self.index) + sys.getsizeof(self.doc_metadata)
            return round(size_bytes / (1024 * 1024), 2)
        except Exception:
            return 0.0

    def clear(self):
        """Clear the entire index."""
        self.index.clear()
        self.doc_metadata.clear()
        self.total_documents = 0
        self.total_terms = 0
        self.avg_doc_length = 0.0
        logger.info("Index cleared")
