"""
Retrieval Engine Module for Cross-Lingual Information Retrieval System.

Main orchestrator for all retrieval models and ranking methods.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from .lexical_retrieval import LexicalRetriever
from .fuzzy_matcher import FuzzyMatcher
from .semantic_retrieval import SemanticRetriever
from .hybrid_ranker import HybridRanker

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Main orchestrator for document retrieval.

    Integrates:
    - Lexical retrieval (BM25, TF-IDF)
    - Fuzzy matching
    - Semantic retrieval (embeddings)
    - Hybrid ranking

    Example:
        >>> engine = RetrievalEngine(inverted_index, documents)
        >>> results = engine.retrieve("education policy", method='hybrid', top_k=10)
        >>> print(results[0])
        {'doc_id': 'doc123', 'title': '...', 'score': 0.89, 'snippet': '...'}
    """

    def __init__(
        self,
        inverted_index=None,
        documents: Optional[List[Dict[str, Any]]] = None,
        enable_semantic: bool = True,
        enable_fuzzy: bool = True,
        semantic_model: str = "paraphrase-multilingual-mpnet-base-v2",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the RetrievalEngine.

        Args:
            inverted_index: InvertedIndex instance
            documents: List of documents with tokens and metadata
            enable_semantic: Enable semantic retrieval
            enable_fuzzy: Enable fuzzy matching
            semantic_model: Name of semantic embedding model
            cache_dir: Directory for caching embeddings
        """
        self.inverted_index = inverted_index
        self.enable_semantic = enable_semantic
        self.enable_fuzzy = enable_fuzzy

        # Initialize retrievers
        self.lexical_retriever = LexicalRetriever()

        if enable_fuzzy:
            self.fuzzy_matcher = FuzzyMatcher()

        if enable_semantic:
            self.semantic_retriever = SemanticRetriever(
                model_name=semantic_model, cache_dir=cache_dir
            )

        self.hybrid_ranker = HybridRanker()

        # Document collection
        self.documents = {}

        # Index documents if provided
        if documents:
            self._index_documents(documents)

        logger.info(
            f"RetrievalEngine initialized (semantic={enable_semantic}, fuzzy={enable_fuzzy})"
        )

    def _index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents for all retrieval methods.

        Args:
            documents: List of document dictionaries
        """
        logger.info(f"Indexing {len(documents)} documents for retrieval...")

        # Index for lexical retrieval
        self.lexical_retriever.index_documents(documents)

        # Store documents
        for doc in documents:
            doc_id = doc.get("doc_id")
            if doc_id:
                self.documents[doc_id] = doc

        # Index for semantic retrieval
        if self.enable_semantic:
            # Prepare documents with combined text
            semantic_docs = []
            for doc in documents:
                doc_id = doc.get("doc_id")
                metadata = doc.get("metadata", {})

                # Combine title and content
                title = metadata.get("title", "")
                # Reconstruct text from tokens if needed
                tokens = doc.get("tokens", [])
                text = " ".join(tokens) if tokens else ""

                semantic_docs.append(
                    {"doc_id": doc_id, "text": f"{title} {text}".strip(), **metadata}
                )

            self.semantic_retriever.encode_documents(semantic_docs)

        logger.info("Indexing complete")

    def retrieve(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 10,
        query_tokens: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a query.

        Args:
            query: Query text
            method: Retrieval method ('bm25', 'tfidf', 'semantic', 'fuzzy', 'hybrid')
            top_k: Number of results to return
            query_tokens: Pre-tokenized query (optional)

        Returns:
            List of result dictionaries
        """
        if not query:
            logger.warning("Empty query provided")
            return []

        start_time = time.time()
        timing = {}

        # Default to hybrid method
        if method not in ["bm25", "tfidf", "semantic", "fuzzy", "hybrid"]:
            logger.warning(f"Unknown method '{method}', using 'hybrid'")
            method = "hybrid"

        # Single method retrieval
        if method == "bm25":
            tokens = query_tokens or query.lower().split()
            results = self.lexical_retriever.search_bm25(tokens, top_k)

        elif method == "tfidf":
            tokens = query_tokens or query.lower().split()
            results = self.lexical_retriever.search_tfidf(tokens, top_k)

        elif method == "semantic":
            if not self.enable_semantic:
                logger.error("Semantic retrieval not enabled")
                return []
            results = self.semantic_retriever.search_semantic(query, top_k)

        elif method == "fuzzy":
            if not self.enable_fuzzy:
                logger.error("Fuzzy matching not enabled")
                return []
            # Fuzzy match against all documents
            results = self._fuzzy_search(query, top_k)

        else:  # hybrid
            results = self._hybrid_search(query, query_tokens, top_k, timing)

        # Add snippets and enhance metadata
        results = self._enhance_results(results, query)

        # Measure total time
        elapsed = time.time() - start_time
        timing["total"] = elapsed

        # Add timing to results
        if results:
            results[0]["_timing"] = timing

        logger.info(
            f"Retrieved {len(results)} documents for query '{query}' "
            f"using {method} in {elapsed:.3f}s"
        )

        return results

    def retrieve_cross_lingual(
        self, query: str, source_lang: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents for cross-lingual query.

        This method should be called with processed queries from QueryProcessor.

        Args:
            query: Processed query (already translated/expanded)
            source_lang: Source language of original query
            top_k: Number of results to return

        Returns:
            List of result dictionaries
        """
        # Use hybrid retrieval
        results = self.retrieve(query, method="hybrid", top_k=top_k)

        # Add source language info
        for result in results:
            result["query_lang"] = source_lang

        return results

    def _hybrid_search(
        self,
        query: str,
        query_tokens: Optional[List[str]],
        top_k: int,
        timing: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining multiple methods.

        Args:
            query: Query text
            query_tokens: Pre-tokenized query
            top_k: Number of results
            timing: Timing dictionary to update

        Returns:
            Ranked results
        """
        tokens = query_tokens or query.lower().split()

        # Get scores from all methods
        scores = {}

        # BM25 scores
        t0 = time.time()
        bm25_results = self.lexical_retriever.search_bm25(tokens, top_k=top_k * 3)
        scores["bm25"] = {r["doc_id"]: r["score"] for r in bm25_results}
        timing["bm25"] = time.time() - t0

        # Semantic scores
        if self.enable_semantic:
            t0 = time.time()
            semantic_results = self.semantic_retriever.search_semantic(
                query, top_k=top_k * 3
            )
            scores["semantic"] = {r["doc_id"]: r["score"] for r in semantic_results}
            timing["semantic"] = time.time() - t0
        else:
            scores["semantic"] = {}

        # Fuzzy scores (only for top BM25 results to save computation)
        if self.enable_fuzzy:
            t0 = time.time()
            fuzzy_scores = {}
            for result in bm25_results[: top_k * 2]:
                doc_id = result["doc_id"]
                doc_text = self._get_document_text(doc_id)
                score = self.fuzzy_matcher.fuzzy_match(query, doc_text, threshold=0.0)
                if score > 0:
                    fuzzy_scores[doc_id] = score
            scores["fuzzy"] = fuzzy_scores
            timing["fuzzy"] = time.time() - t0
        else:
            scores["fuzzy"] = {}

        # Collect all candidate documents
        all_doc_ids = set()
        all_doc_ids.update(scores["bm25"].keys())
        all_doc_ids.update(scores["semantic"].keys())
        all_doc_ids.update(scores["fuzzy"].keys())

        documents = [
            self.documents.get(doc_id, {"doc_id": doc_id})
            for doc_id in all_doc_ids
            if doc_id in self.documents
        ]

        # Combine using hybrid ranker
        t0 = time.time()
        results = self.hybrid_ranker.rank_documents(query, documents, scores)
        timing["ranking"] = time.time() - t0

        return results[:top_k]

    def _fuzzy_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform fuzzy search across all documents.

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            Results sorted by fuzzy match score
        """
        results = []

        for doc_id, doc in self.documents.items():
            doc_text = self._get_document_text(doc_id)
            score = self.fuzzy_matcher.fuzzy_match(query, doc_text, threshold=0.0)

            if score > 0:
                result = doc.copy()
                result["doc_id"] = doc_id
                result["score"] = round(score, 4)
                result["normalized_score"] = round(score, 4)
                results.append(result)

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def _get_document_text(self, doc_id: str) -> str:
        """
        Get text representation of document for fuzzy matching.

        Args:
            doc_id: Document ID

        Returns:
            Document text
        """
        doc = self.documents.get(doc_id, {})
        metadata = doc.get("metadata", {})

        title = metadata.get("title", "")
        tokens = doc.get("tokens", [])
        text = " ".join(tokens[:100]) if tokens else ""  # Limit for efficiency

        return f"{title} {text}".strip()

    def _enhance_results(
        self, results: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """
        Enhance results with snippets and additional metadata.

        Args:
            results: List of results
            query: Original query

        Returns:
            Enhanced results
        """
        for result in results:
            doc_id = result.get("doc_id")

            if not doc_id:
                continue

            # Get full document
            doc = self.documents.get(doc_id, {})
            metadata = doc.get("metadata", {})

            # Add snippet (first 200 chars of content)
            tokens = doc.get("tokens", [])
            if tokens:
                snippet = " ".join(tokens[:30])
                if len(tokens) > 30:
                    snippet += "..."
                result["snippet"] = snippet

            # Ensure metadata fields
            result.setdefault("title", metadata.get("title", "Untitled"))
            result.setdefault("url", metadata.get("url", ""))
            result.setdefault("language", metadata.get("language", "unknown"))
            result.setdefault("date", metadata.get("date", ""))

        return results

    def get_matching_score(
        self, doc_id: str, query: str, method: str = "bm25"
    ) -> float:
        """
        Get matching score for a specific document.

        Args:
            doc_id: Document ID
            query: Query text
            method: Scoring method

        Returns:
            Matching score
        """
        tokens = query.lower().split()
        scores_dict = self.lexical_retriever.get_scores(tokens, [doc_id], method)
        return scores_dict.get(doc_id, 0.0)

    def measure_query_time(self) -> Dict[str, float]:
        """
        Get timing information from last query.

        Returns:
            Dictionary of timing measurements
        """
        # This would return timing from last query
        # For now, return empty dict
        return {}
