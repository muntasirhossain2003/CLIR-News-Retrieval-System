"""
FAISS Retrieval Wrapper for Module C
"""

import os
import json
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Check FAISS availability
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, will use numpy fallback")

# Model loading (lazy)
_embedding_model = None
_model_name_loaded = None


def _load_embedding_model(model_name: str):
    """Lazy load sentence transformer model."""
    global _embedding_model, _model_name_loaded

    if _embedding_model is not None and _model_name_loaded == model_name:
        return _embedding_model

    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {model_name}")
        start = time.time()
        _embedding_model = SentenceTransformer(model_name)
        _model_name_loaded = model_name
        logger.info(f"Model loaded in {time.time()-start:.2f}s")
        return _embedding_model
    except ImportError:
        logger.error("sentence-transformers not installed")
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# Best multilingual model for CLIR
DEFAULT_MODEL = "intfloat/multilingual-e5-large-instruct"


class FAISSRetrieval:
    """
    FAISS-based semantic retrieval compatible with Module C's interface.
    Uses Module A's FAISS index instead of separate semantic pickle.
    Uses mE5-large-instruct for best Bangla-English cross-lingual performance.
    """

    def __init__(
        self,
        index_dir: str = "indexes/semantic",
        model_name: str = DEFAULT_MODEL,
    ):
        """
        Initialize FAISS retrieval.

        Args:
            index_dir: Directory containing embeddings.npy, doc_ids.json, metadata.json
            model_name: Sentence transformer model name
        """
        self.index_dir = index_dir
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.doc_ids = []
        self.faiss_index = None
        self._is_loaded = False

    def load(self, index_path: str = None) -> bool:
        """
        Load FAISS index from disk.

        Args:
            index_path: Optional override for index directory

        Returns:
            True if loaded successfully
        """
        if index_path:
            self.index_dir = index_path

        embeddings_path = os.path.join(self.index_dir, "embeddings.npy")
        doc_ids_path = os.path.join(self.index_dir, "doc_ids.json")
        metadata_path = os.path.join(self.index_dir, "metadata.json")

        # Check files exist
        if not os.path.exists(embeddings_path):
            logger.error(f"Embeddings not found: {embeddings_path}")
            return False
        if not os.path.exists(doc_ids_path):
            logger.error(f"Doc IDs not found: {doc_ids_path}")
            return False

        try:
            # Load embeddings
            import time

            start = time.time()
            self.embeddings = np.load(embeddings_path)
            logger.info(
                f"Loaded embeddings: {self.embeddings.shape} ({time.time()-start:.2f}s)"
            )

            # Load doc IDs
            with open(doc_ids_path, "r") as f:
                self.doc_ids = json.load(f)

            # Load metadata if exists
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    if "model_name" in metadata:
                        self.model_name = metadata["model_name"]

            # Build FAISS index
            if FAISS_AVAILABLE:
                self._build_faiss_index()
                logger.info(
                    f"✓ FAISS index loaded ({len(self.doc_ids)} docs, {self.embeddings.shape[1]}d)"
                )
            else:
                logger.info(
                    f"✓ Semantic index loaded ({len(self.doc_ids)} docs, numpy mode)"
                )

            self._is_loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    def _build_faiss_index(self):
        """Build FAISS index from loaded embeddings."""
        if not FAISS_AVAILABLE or self.embeddings is None:
            return

        dim = self.embeddings.shape[1]

        # Normalize for cosine similarity
        normalized = self.embeddings.copy()
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        normalized = normalized / (norms + 1e-10)

        # Use IndexFlatIP (inner product = cosine for normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(normalized.astype(np.float32))
        logger.info(f"FAISS index built: {self.faiss_index.ntotal} vectors")

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        preprocess: bool = True,
        target_lang: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search using FAISS (Module C-compatible interface).

        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum similarity threshold
            preprocess: Ignored (embeddings handle preprocessing)
            target_lang: Ignored (multilingual model)

        Returns:
            List of results with doc_id, score, rank
        """
        if not self._is_loaded:
            logger.error("FAISS index not loaded")
            return []

        if not query or not query.strip():
            logger.warning("Empty query")
            return []

        try:
            # Load model if needed
            if self.model is None:
                self.model = _load_embedding_model(self.model_name)

            # Encode query
            import time

            start = time.time()

            # mE5-instruct models require "query: " prefix for retrieval tasks
            encode_query = query
            if "e5" in self.model_name.lower():
                encode_query = f"query: {query}"

            query_embedding = self.model.encode(
                [encode_query], show_progress_bar=False
            )[0]
            query_embedding = query_embedding / (
                np.linalg.norm(query_embedding) + 1e-10
            )

            # Search
            if self.faiss_index is not None:
                # FAISS search
                scores, indices = self.faiss_index.search(
                    query_embedding.reshape(1, -1).astype(np.float32), top_k
                )
                scores = scores[0]
                indices = indices[0]
            else:
                # Numpy fallback
                similarities = np.dot(self.embeddings, query_embedding)
                indices = np.argsort(similarities)[::-1][:top_k]
                scores = similarities[indices]

            # Format results
            results = []
            for rank, (idx, score) in enumerate(zip(indices, scores), 1):
                if idx >= len(self.doc_ids):
                    continue
                if score >= min_score:
                    results.append(
                        {
                            "doc_id": self.doc_ids[idx],
                            "score": float(score),
                            "score_normalized": float(score),  # Already normalized
                            "rank": rank,
                            "method": "faiss",
                        }
                    )

            logger.debug(
                f"FAISS search: {time.time()-start:.3f}s, {len(results)} results"
            )
            return results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

    def get_normalized_scores(self, query: str, top_k: int = 10, **kwargs):
        """Alias for search (scores already normalized for cosine similarity)."""
        return self.search(query, top_k=top_k, **kwargs)
