"""
Semantic Retrieval Module for Cross-Lingual Information Retrieval System.

Implements semantic retrieval using multilingual embeddings.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Semantic retrieval using multilingual sentence embeddings.

    Uses models like:
    - LaBSE (Language-agnostic BERT Sentence Embedding)
    - multilingual-e5-large
    - paraphrase-multilingual-mpnet-base-v2

    Example:
        >>> retriever = SemanticRetriever(model_name='paraphrase-multilingual-mpnet-base-v2')
        >>> retriever.encode_documents(documents)
        >>> results = retriever.search_semantic("education policy", top_k=10)
        >>> print(results[0])
        {'doc_id': 'doc123', 'score': 0.89, 'title': '...'}
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        cache_dir: Optional[str] = None,
        use_gpu: bool = False,
    ):
        """
        Initialize the SemanticRetriever.

        Args:
            model_name: Name of the sentence transformer model
            cache_dir: Directory to cache document embeddings
            use_gpu: Whether to use GPU for encoding
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_gpu = use_gpu

        # Model (lazy loaded)
        self.model = None

        # Document storage
        self.documents = {}  # doc_id -> metadata
        self.doc_embeddings = {}  # doc_id -> embedding vector
        self.embedding_matrix = None  # numpy array of all embeddings
        self.doc_id_list = []  # ordered list of doc_ids

        logger.info(
            f"SemanticRetriever initialized (model={model_name}, use_gpu={use_gpu})"
        )

    def load_model(self, model_name: Optional[str] = None):
        """
        Load the sentence transformer model.

        Args:
            model_name: Model name (uses default if not provided)
        """
        if self.model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            model_to_load = model_name or self.model_name

            device = "cuda" if self.use_gpu else "cpu"
            self.model = SentenceTransformer(model_to_load, device=device)

            logger.info(f"Loaded model: {model_to_load} on {device}")

        except Exception as e:
            logger.error(f"Failed to load model {model_to_load}: {e}")
            raise

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query into an embedding vector.

        Args:
            query: Query text

        Returns:
            Embedding vector as numpy array
        """
        if not query:
            logger.warning("Empty query provided")
            return np.array([])

        # Load model if needed
        if self.model is None:
            self.load_model()

        try:
            embedding = self.model.encode(query, convert_to_numpy=True)
            logger.debug(f"Encoded query: '{query[:50]}...'")
            return embedding

        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            return np.array([])

    def encode_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32,
        text_field: str = "text",
    ):
        """
        Encode documents into embedding vectors.

        Args:
            documents: List of document dictionaries with doc_id and text
            batch_size: Batch size for encoding
            text_field: Field name containing text to encode
        """
        if not documents:
            logger.warning("No documents provided for encoding")
            return

        logger.info(f"Encoding {len(documents)} documents...")

        # Load model if needed
        if self.model is None:
            self.load_model()

        # Check cache first
        if self.cache_dir:
            self._load_cached_embeddings()

        # Prepare documents for encoding
        docs_to_encode = []
        doc_ids_to_encode = []

        for doc in documents:
            doc_id = doc["doc_id"]
            text = doc.get(text_field, "")

            if not text:
                logger.warning(f"Empty text for document {doc_id}")
                continue

            # Store metadata
            self.documents[doc_id] = doc.copy()

            # Check if already cached
            if doc_id not in self.doc_embeddings:
                docs_to_encode.append(text)
                doc_ids_to_encode.append(doc_id)

        # Encode new documents
        if docs_to_encode:
            try:
                logger.info(f"Encoding {len(docs_to_encode)} new documents...")
                embeddings = self.model.encode(
                    docs_to_encode,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                )

                # Store embeddings
                for doc_id, embedding in zip(doc_ids_to_encode, embeddings):
                    self.doc_embeddings[doc_id] = embedding

                logger.info(f"Encoded {len(docs_to_encode)} documents")

                # Cache embeddings
                if self.cache_dir:
                    self._cache_embeddings()

            except Exception as e:
                logger.error(f"Error encoding documents: {e}")
                raise

        # Build embedding matrix for efficient search
        self._build_embedding_matrix()

    def compute_similarity(
        self, query_embedding: np.ndarray, doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.

        Args:
            query_embedding: Query embedding vector
            doc_embeddings: Matrix of document embeddings

        Returns:
            Array of similarity scores
        """
        if query_embedding.size == 0 or doc_embeddings.size == 0:
            return np.array([])

        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(
            doc_embeddings, axis=1, keepdims=True
        )

        # Cosine similarity
        similarities = np.dot(doc_norms, query_norm)

        return similarities

    def search_semantic(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using semantic similarity.

        Args:
            query: Query text
            top_k: Number of top results to return

        Returns:
            List of results sorted by semantic similarity
        """
        if not query:
            return []

        if self.embedding_matrix is None or len(self.doc_id_list) == 0:
            logger.warning("No documents indexed")
            return []

        # Encode query
        query_embedding = self.encode_query(query)

        if query_embedding.size == 0:
            return []

        # Compute similarities
        similarities = self.compute_similarity(query_embedding, self.embedding_matrix)

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            doc_id = self.doc_id_list[idx]
            score = float(similarities[idx])

            result = {
                "doc_id": doc_id,
                "score": round(score, 4),
                "normalized_score": round(score, 4),  # Cosine sim already [0,1]
            }

            # Add metadata
            metadata = self.documents.get(doc_id, {})
            result.update(metadata)

            results.append(result)

        logger.debug(f"Semantic search returned {len(results)} results")
        return results

    def _build_embedding_matrix(self):
        """Build numpy matrix of all document embeddings for efficient search."""
        if not self.doc_embeddings:
            return

        self.doc_id_list = list(self.doc_embeddings.keys())
        embeddings_list = [self.doc_embeddings[doc_id] for doc_id in self.doc_id_list]
        self.embedding_matrix = np.array(embeddings_list)

        logger.info(f"Built embedding matrix: {self.embedding_matrix.shape}")

    def _cache_embeddings(self):
        """Cache document embeddings to disk."""
        if not self.cache_dir:
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / "doc_embeddings.pkl"

            with open(cache_file, "wb") as f:
                pickle.dump(self.doc_embeddings, f)

            logger.info(f"Cached {len(self.doc_embeddings)} embeddings to {cache_file}")

        except Exception as e:
            logger.error(f"Error caching embeddings: {e}")

    def _load_cached_embeddings(self):
        """Load cached document embeddings from disk."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / "doc_embeddings.pkl"

        if not cache_file.exists():
            return

        try:
            with open(cache_file, "rb") as f:
                cached_embeddings = pickle.load(f)

            self.doc_embeddings.update(cached_embeddings)
            logger.info(
                f"Loaded {len(cached_embeddings)} cached embeddings from {cache_file}"
            )

        except Exception as e:
            logger.error(f"Error loading cached embeddings: {e}")
