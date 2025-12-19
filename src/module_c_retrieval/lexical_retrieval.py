"""
Lexical Retrieval Module for Cross-Lingual Information Retrieval System.

Implements BM25 and TF-IDF scoring for lexical matching.
"""

import logging
import math
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


class LexicalRetriever:
    """
    Lexical retrieval using BM25 and TF-IDF.

    Features:
    - BM25 ranking (Okapi BM25)
    - TF-IDF scoring
    - Score normalization
    - Support for large document collections

    Example:
        >>> retriever = LexicalRetriever()
        >>> retriever.index_documents(documents)
        >>> results = retriever.search_bm25(['education', 'policy'], top_k=10)
        >>> print(results[0])
        {'doc_id': 'doc123', 'score': 0.89, 'title': '...'}
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize the LexicalRetriever.

        Args:
            k1: BM25 parameter controlling term frequency saturation (default: 1.5)
            b: BM25 parameter controlling length normalization (default: 0.75)
        """
        self.k1 = k1
        self.b = b

        # Document collection
        self.documents = {}  # doc_id -> {tokens, metadata}
        self.doc_lengths = {}  # doc_id -> token count
        self.avg_doc_length = 0.0

        # Inverted index: term -> {doc_id: frequency}
        self.inverted_index = {}

        # IDF scores: term -> idf_score
        self.idf_scores = {}

        # Statistics
        self.total_docs = 0

        logger.info(f"LexicalRetriever initialized (k1={k1}, b={b})")

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents for retrieval.

        Args:
            documents: List of document dictionaries with keys:
                      - doc_id: Document identifier
                      - tokens: List of tokens
                      - metadata: Optional metadata dict

        Example:
            >>> docs = [
            ...     {'doc_id': 'doc1', 'tokens': ['hello', 'world'],
            ...      'metadata': {'title': 'Test'}},
            ... ]
            >>> retriever.index_documents(docs)
        """
        logger.info(f"Indexing {len(documents)} documents...")

        for doc in documents:
            doc_id = doc["doc_id"]
            tokens = doc.get("tokens", [])
            metadata = doc.get("metadata", {})

            # Store document
            self.documents[doc_id] = {"tokens": tokens, "metadata": metadata}

            # Store document length
            doc_length = len(tokens)
            self.doc_lengths[doc_id] = doc_length

            # Build inverted index
            term_counts = Counter(tokens)
            for term, freq in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_id] = freq

        # Update statistics
        self.total_docs = len(self.documents)
        self.avg_doc_length = sum(self.doc_lengths.values()) / max(self.total_docs, 1)

        # Calculate IDF scores
        self._calculate_idf()

        logger.info(
            f"Indexed {self.total_docs} documents, "
            f"{len(self.inverted_index)} unique terms, "
            f"avg doc length: {self.avg_doc_length:.2f}"
        )

    def _calculate_idf(self):
        """Calculate IDF scores for all terms."""
        for term, postings in self.inverted_index.items():
            df = len(postings)  # Document frequency
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
            self.idf_scores[term] = idf

    def search_bm25(
        self, query_tokens: List[str], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search using BM25 ranking.

        Args:
            query_tokens: List of query tokens
            top_k: Number of top results to return

        Returns:
            List of result dictionaries sorted by BM25 score

        Example:
            >>> results = retriever.search_bm25(['education', 'policy'], top_k=5)
            >>> print(results[0]['score'])
            0.892
        """
        if not query_tokens:
            return []

        # Calculate BM25 scores for all documents
        scores = {}

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            idf = self.idf_scores.get(term, 0.0)
            postings = self.inverted_index[term]

            for doc_id, tf in postings.items():
                doc_length = self.doc_lengths[doc_id]

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )
                score = idf * (numerator / denominator)

                scores[doc_id] = scores.get(doc_id, 0.0) + score

        # Sort by score
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build result list
        results = []
        for doc_id, score in ranked_docs:
            result = {
                "doc_id": doc_id,
                "score": round(score, 4),
                "normalized_score": 0.0,  # Will be set later
            }

            # Add metadata
            metadata = self.documents[doc_id].get("metadata", {})
            result.update(metadata)

            results.append(result)

        # Normalize scores
        results = self._normalize_scores(results)

        logger.debug(f"BM25 search returned {len(results)} results")
        return results

    def search_tfidf(
        self, query_tokens: List[str], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search using TF-IDF scoring.

        Args:
            query_tokens: List of query tokens
            top_k: Number of top results to return

        Returns:
            List of result dictionaries sorted by TF-IDF score
        """
        if not query_tokens:
            return []

        # Calculate TF-IDF scores
        scores = {}

        # Query term frequencies
        query_tf = Counter(query_tokens)

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            idf = self.idf_scores.get(term, 0.0)
            postings = self.inverted_index[term]

            for doc_id, doc_tf in postings.items():
                doc_length = self.doc_lengths[doc_id]

                # Normalized TF
                tf_normalized = doc_tf / doc_length if doc_length > 0 else 0.0

                # TF-IDF score
                score = tf_normalized * idf * query_tf[term]

                scores[doc_id] = scores.get(doc_id, 0.0) + score

        # Sort by score
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build result list
        results = []
        for doc_id, score in ranked_docs:
            result = {
                "doc_id": doc_id,
                "score": round(score, 4),
                "normalized_score": 0.0,
            }

            # Add metadata
            metadata = self.documents[doc_id].get("metadata", {})
            result.update(metadata)

            results.append(result)

        # Normalize scores
        results = self._normalize_scores(results)

        logger.debug(f"TF-IDF search returned {len(results)} results")
        return results

    def get_scores(
        self, query_tokens: List[str], doc_ids: List[str], method: str = "bm25"
    ) -> Dict[str, float]:
        """
        Get scores for specific documents.

        Args:
            query_tokens: List of query tokens
            doc_ids: List of document IDs to score
            method: Scoring method ('bm25' or 'tfidf')

        Returns:
            Dictionary mapping doc_id to score
        """
        if method == "bm25":
            all_results = self.search_bm25(query_tokens, top_k=len(self.documents))
        else:
            all_results = self.search_tfidf(query_tokens, top_k=len(self.documents))

        # Filter for requested doc_ids
        scores = {}
        for result in all_results:
            if result["doc_id"] in doc_ids:
                scores[result["doc_id"]] = result["score"]

        return scores

    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize scores to [0, 1] range.

        Args:
            results: List of result dictionaries

        Returns:
            Results with normalized_score field added
        """
        if not results:
            return results

        scores = [r["score"] for r in results]
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        score_range = max_score - min_score

        for result in results:
            if score_range > 0:
                normalized = (result["score"] - min_score) / score_range
            else:
                normalized = 1.0 if result["score"] > 0 else 0.0

            result["normalized_score"] = round(normalized, 4)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get retriever statistics.

        Returns:
            Dictionary containing statistics
        """
        return {
            "total_documents": self.total_docs,
            "unique_terms": len(self.inverted_index),
            "avg_doc_length": round(self.avg_doc_length, 2),
            "k1": self.k1,
            "b": self.b,
        }
