"""
Semantic Indexer using Sentence Transformers

Creates dense vector representations for semantic similarity search.
Uses multilingual embeddings for cross-lingual retrieval.
"""

import os
import json
import numpy as np
import logging
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SemanticIndexer:
    """
    Semantic index using multilingual sentence embeddings.
    Supports cross-lingual similarity search with cosine distance.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        index_dir: str = "indexes/semantic",
    ):
        """
        Initialize semantic indexer.

        Args:
            model_name: Name of sentence-transformers model
            index_dir: Directory to save embeddings
        """
        self.model_name = model_name
        self.index_dir = index_dir
        self.model = None
        self.embeddings = None
        self.doc_ids = []

        os.makedirs(self.index_dir, exist_ok=True)

        self._load_model()

    def _load_model(self):
        """Load sentence transformer model."""
        logger.info(f"Loading sentence transformer: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(
                f"✓ Model loaded (embedding dim: {self.model.get_sentence_embedding_dimension()})"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def encode_documents(
        self, documents: List[Dict], batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode documents into embeddings.

        Args:
            documents: List of dicts with keys: doc_id, title, body
            batch_size: Batch size for encoding

        Returns:
            NumPy array of embeddings (n_docs, embedding_dim)
        """
        logger.info(f"Encoding {len(documents)} documents...")

        # Prepare texts (title + body)
        texts = []
        self.doc_ids = []

        for doc in documents:
            # Combine title and body for richer representation
            title = doc.get("title", "")
            body = doc.get("body", "")

            # Truncate body to avoid exceeding model's max length
            # Most sentence transformers have 512 token limit
            combined = f"{title} {body}"[:2000]  # ~500 tokens

            texts.append(combined)
            self.doc_ids.append(doc["doc_id"])

        # Encode in batches with progress
        logger.info("Generating embeddings (this may take a few minutes)...")
        self.embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )

        logger.info(f"✓ Generated embeddings: shape {self.embeddings.shape}")
        return self.embeddings

    def save_index(self):
        """Save embeddings and doc_ids to disk."""
        if self.embeddings is None or not self.doc_ids:
            raise RuntimeError("No embeddings to save. Call encode_documents() first.")

        embeddings_path = os.path.join(self.index_dir, "embeddings.npy")
        doc_ids_path = os.path.join(self.index_dir, "doc_ids.json")
        metadata_path = os.path.join(self.index_dir, "metadata.json")

        logger.info(f"Saving semantic index to {self.index_dir}")

        # Save embeddings
        np.save(embeddings_path, self.embeddings)

        # Save doc IDs
        with open(doc_ids_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f)

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "num_documents": len(self.doc_ids),
            "embedding_dim": self.embeddings.shape[1],
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("✓ Semantic index saved")

    def load_index(self):
        """Load embeddings and doc_ids from disk."""
        embeddings_path = os.path.join(self.index_dir, "embeddings.npy")
        doc_ids_path = os.path.join(self.index_dir, "doc_ids.json")

        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"No embeddings found at {embeddings_path}")

        logger.info(f"Loading semantic index from {self.index_dir}")

        self.embeddings = np.load(embeddings_path)

        with open(doc_ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)

        logger.info(f"✓ Loaded {len(self.doc_ids)} document embeddings")

    def search(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query_text: Query string
            top_k: Number of results to return

        Returns:
            List of (doc_id, similarity_score) tuples, sorted by score descending
        """
        if self.embeddings is None:
            self.load_index()

        # Encode query
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)

        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return doc_ids and scores
        results = [(self.doc_ids[idx], float(similarities[idx])) for idx in top_indices]

        return results

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        return self.model.encode([text], convert_to_numpy=True)[0]

    def get_stats(self) -> Dict:
        """Get index statistics."""
        if self.embeddings is None:
            try:
                self.load_index()
            except FileNotFoundError:
                return {"status": "not_created"}

        return {
            "total_documents": len(self.doc_ids),
            "embedding_dimension": self.embeddings.shape[1],
            "model": self.model_name,
            "index_dir": self.index_dir,
        }
