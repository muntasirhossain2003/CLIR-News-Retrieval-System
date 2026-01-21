"""
WHOOSH Retrieval Wrapper for Module C

Provides Module C-compatible interface to Module A's WHOOSH lexical index.
This bridges Module A (indexing) with Module C (retrieval) without duplicating indexes.
"""

import os
import logging
from typing import List, Dict, Any
from whoosh import index
from whoosh.qparser import MultifieldParser
from whoosh.searching import Results

logger = logging.getLogger(__name__)


class WHOOSHRetrieval:
    """
    WHOOSH-based retrieval compatible with Module C's interface.
    Uses Module A's WHOOSH index instead of separate BM25/TF-IDF pickles.
    """

    def __init__(self, index_dir: str = "indexes/whoosh"):
        """
        Initialize WHOOSH retrieval.

        Args:
            index_dir: Path to WHOOSH index directory
        """
        self.index_dir = index_dir
        self.index = None
        self._is_loaded = False

    def load(self, index_path: str = None) -> bool:
        """
        Load WHOOSH index from disk.

        Args:
            index_path: Optional override for index directory

        Returns:
            True if loaded successfully
        """
        if index_path:
            self.index_dir = index_path

        try:
            if not index.exists_in(self.index_dir):
                logger.error(f"WHOOSH index not found at {self.index_dir}")
                return False

            self.index = index.open_dir(self.index_dir)
            self._is_loaded = True
            logger.info(f"✓ WHOOSH index loaded ({self.index.doc_count()} documents)")
            return True

        except Exception as e:
            logger.error(f"Failed to load WHOOSH index: {e}")
            return False

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        preprocess: bool = True,
        target_lang: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Search WHOOSH index (Module C-compatible interface).

        Args:
            query: Search query string
            top_k: Number of results to return
            min_score: Minimum score threshold
            preprocess: Whether to apply query preprocessing (ignored for WHOOSH)
            target_lang: Target language (ignored for WHOOSH)

        Returns:
            List of results with doc_id, score, rank
        """
        if not self._is_loaded or self.index is None:
            logger.error("WHOOSH index not loaded")
            return []

        if not query or not query.strip():
            logger.warning("Empty query")
            return []

        try:
            with self.index.searcher() as searcher:
                # Search in title and body fields with title boost
                parser = MultifieldParser(["title", "body"], schema=self.index.schema)
                parsed_query = parser.parse(query)

                # Execute search
                hits = searcher.search(parsed_query, limit=top_k)

                # Convert to Module C format
                results = []
                for rank, hit in enumerate(hits, 1):
                    score = hit.score
                    if score >= min_score:
                        results.append(
                            {
                                "doc_id": hit["doc_id"],
                                "score": float(score),
                                "rank": rank,
                                "method": "whoosh",
                                "metadata": {
                                    "title": hit.get("title", ""),
                                    "source": hit.get("source", ""),
                                    "language": hit.get("language", ""),
                                },
                            }
                        )

                return results

        except Exception as e:
            logger.error(f"WHOOSH search failed: {e}")
            return []

    def get_normalized_scores(
        self,
        query: str,
        top_k: int = 10,
        preprocess: bool = True,
        target_lang: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Search with normalized scores [0, 1].

        WHOOSH scores are already relatively normalized, so we just apply min-max.
        """
        results = self.search(
            query, top_k=top_k * 2, preprocess=preprocess, target_lang=target_lang
        )

        if not results:
            return []

        # Normalize scores to [0, 1]
        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range == 0:
            for r in results:
                r["score_normalized"] = 0.5
        else:
            for r in results:
                r["score_normalized"] = (r["score"] - min_score) / score_range
                r["score"] = r["score_normalized"]  # Update score field

        return results[:top_k]
