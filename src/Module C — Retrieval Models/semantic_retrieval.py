import logging
import time
import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Try to import Module B query processing
MODULE_B_AVAILABLE = False
try:
    module_b_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "Module B — Query Processing & Cross-Lingual Handling",
    )
    if os.path.exists(module_b_path):
        sys.path.insert(0, module_b_path)
        from query_pipeline import process_complete_query

        MODULE_B_AVAILABLE = True
        logger.info("Module B (Query Processing) loaded successfully")
except ImportError as e:
    logger.warning(f"Module B not available: {e}")
    process_complete_query = None

# Module-level lazy loading
_embedding_model = None
_model_name_loaded = None


def _load_embedding_model(model_name: str):
    """
    Lazy load the embedding model.

    Args:
        model_name: Name of the sentence-transformers model
    """
    global _embedding_model, _model_name_loaded

    # Return cached model if same model is requested
    if _embedding_model is not None and _model_name_loaded == model_name:
        return _embedding_model

    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {model_name}")
        start_time = time.time()

        _embedding_model = SentenceTransformer(model_name)
        _model_name_loaded = model_name

        elapsed = time.time() - start_time
        logger.info(f"Model loaded in {elapsed:.2f}s")

        return _embedding_model

    except ImportError:
        logger.error(
            "sentence-transformers not installed. Install with: pip install sentence-transformers"
        )
        raise ImportError("sentence-transformers required for semantic retrieval")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise


def encode_query(
    query: str, model_name: str, show_progress: bool = False
) -> np.ndarray:
    """
    Encode a query into a dense embedding.

    Args:
        query: Query text to encode
        model_name: Embedding model name
        show_progress: Show progress bar

    Returns:
        NumPy array of shape (1, embedding_dim)
    """
    model = _load_embedding_model(model_name)

    embedding = model.encode(
        [query],
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # Normalize for cosine similarity
    )

    return embedding


class SemanticIndex:
    """
    Semantic Index using FAISS for dense vector retrieval.

    Supports loading from existing index format:
    - embeddings.npy: Pre-computed embeddings
    - doc_ids.json: Document identifiers
    - metadata.json: Index configuration

    Creates FAISS index dynamically from embeddings for efficient search.

    Attributes:
        index: FAISS index instance
        doc_ids: List of document IDs
        embedding_dim: Dimension of embeddings
        model_name: Name of embedding model used
    """

    def __init__(self, index_type: str = "flat"):
        """
        Initialize semantic index.

        Args:
            index_type: FAISS index type
                - "flat": Exact search (IndexFlatIP) - best accuracy
                - "hnsw": Approximate search (HNSW) - faster for large collections
        """
        self.index_type = index_type
        self.index = None
        self.doc_ids = []
        self.embeddings = None
        self.embedding_dim = None
        self.model_name = None
        self.num_documents = 0
        self._is_loaded = False

    def load(self, directory: str) -> bool:
        """
        Load semantic index from existing format (embeddings.npy, doc_ids.json, metadata.json).

        Automatically builds FAISS index from loaded embeddings.

        Args:
            directory: Path to index directory

        Returns:
            True if loaded successfully
        """
        try:
            import faiss
        except ImportError:
            logger.error("FAISS not installed. Install with: pip install faiss-cpu")
            return False

        if not os.path.exists(directory):
            logger.error(f"Index directory not found: {directory}")
            return False

        try:
            start_time = time.time()

            # Load embeddings
            embeddings_path = os.path.join(directory, "embeddings.npy")
            if not os.path.exists(embeddings_path):
                logger.error(f"Embeddings not found: {embeddings_path}")
                return False
            self.embeddings = np.load(embeddings_path).astype(np.float32)

            # Load document IDs
            doc_ids_path = os.path.join(directory, "doc_ids.json")
            if not os.path.exists(doc_ids_path):
                logger.error(f"Document IDs not found: {doc_ids_path}")
                return False
            with open(doc_ids_path, "r", encoding="utf-8") as f:
                self.doc_ids = json.load(f)

            # Load metadata
            metadata_path = os.path.join(directory, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    self.embedding_dim = metadata.get("embedding_dim")
                    self.model_name = metadata.get("model_name")
                    self.num_documents = metadata.get(
                        "num_documents", len(self.doc_ids)
                    )

            # Get embedding dimension from data if not in metadata
            if self.embedding_dim is None:
                self.embedding_dim = self.embeddings.shape[1]

            # Build FAISS index from embeddings
            logger.info(f"Building FAISS {self.index_type} index...")

            if self.index_type == "hnsw":
                # HNSW for approximate search (faster for large collections)
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 128
            else:
                # Flat index for exact search (inner product = cosine for normalized vectors)
                self.index = faiss.IndexFlatIP(self.embedding_dim)

            # Add embeddings to FAISS index
            self.index.add(self.embeddings)

            self._is_loaded = True
            elapsed = time.time() - start_time

            logger.info(
                f"Semantic index loaded from {directory} ({len(self.doc_ids)} docs, "
                f"{self.embedding_dim}d, {elapsed:.2f}s)"
            )
            print(
                f"✓ Semantic index loaded: {len(self.doc_ids)} documents, {self.embedding_dim}d embeddings"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading semantic index: {e}")
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
        Search index with semantic similarity using FAISS.

        Args:
            query: Search query string
            top_k: Number of results to return
            min_score: Minimum similarity threshold (cosine similarity, 0-1)
            preprocess: Whether to apply Module B query preprocessing (default: True)
            target_lang: Target language for translation ('bn' or 'en')

        Returns:
            List of results with doc_id, score, and rank
        """
        if not self._is_loaded or self.index is None:
            logger.error("Semantic index not loaded. Call load() first.")
            return []

        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        if not self.model_name:
            logger.error("Model name not found in index metadata")
            return []

        start_time = time.time()

        # Apply query preprocessing if enabled
        search_query = query
        if preprocess and MODULE_B_AVAILABLE and process_complete_query:
            try:
                processed = process_complete_query(query, target_lang=target_lang)
                # Use translated query if available, else normalized query
                if target_lang and processed.get("translated_query"):
                    search_query = processed["translated_query"]
                    logger.info(f"Using translated query: {search_query}")
                else:
                    search_query = processed.get("normalized_query", query)
                    logger.info(f"Using normalized query: {search_query}")
            except Exception as e:
                logger.warning(f"Query preprocessing failed: {e}, using original query")
                search_query = query

        # Encode query using same model as documents
        query_embedding = encode_query(search_query, self.model_name)

        # Search FAISS index
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        # Build results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx >= 0 and idx < len(self.doc_ids):  # Valid index
                # For normalized vectors, inner product = cosine similarity (range -1 to 1)
                # Normalize to 0-1 range
                score_normalized = float((score + 1) / 2)

                if score_normalized >= min_score:
                    results.append(
                        {
                            "doc_id": self.doc_ids[idx],
                            "score": float(score),
                            "score_normalized": score_normalized,
                            "rank": rank,
                            "method": "semantic",
                        }
                    )

        elapsed = time.time() - start_time
        logger.debug(f"Semantic search completed in {elapsed*1000:.2f}ms")

        return results

    def get_info(self) -> Dict[str, Any]:
        """Get information about the loaded index."""
        return {
            "loaded": self._is_loaded,
            "num_documents": len(self.doc_ids),
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "index_type": self.index_type,
        }


def retrieve_semantic(
    query: str,
    index: SemanticIndex,
    top_k: int = 10,
    preprocess: bool = True,
    target_lang: str = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve documents using semantic similarity.

    Args:
        query: Search query string
        index: Loaded SemanticIndex instance
        top_k: Number of results
        preprocess: Whether to apply query preprocessing (default: True)
        target_lang: Target language for cross-lingual search ('bn' or 'en')

    Returns:
        List of ranked results with scores
    """
    return index.search(
        query, top_k=top_k, preprocess=preprocess, target_lang=target_lang
    )


# Command line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Semantic Retrieval using FAISS for CLIR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search with semantic embeddings
  python semantic_retrieval.py "climate change impact" --index indexes/semantic
  
  # Search in Bangla
  python semantic_retrieval.py "জলবায়ু পরিবর্তন" --index indexes/semantic
  
  # Get more results
  python semantic_retrieval.py "election results" --index indexes/semantic --top-k 20
        """,
    )

    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument(
        "--index",
        "-i",
        default="indexes/semantic",
        help="Path to semantic index directory",
    )
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of results")
    parser.add_argument(
        "--index-type",
        choices=["flat", "hnsw"],
        default="flat",
        help="FAISS index type (flat=exact, hnsw=approximate)",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--info", action="store_true", help="Show index information")

    args = parser.parse_args()

    if args.info:
        # Show index info
        index = SemanticIndex(index_type=args.index_type)
        if index.load(args.index):
            info = index.get_info()
            print("\nSemantic Index Information:")
            print("=" * 40)
            for key, value in info.items():
                print(f"  {key}: {value}")
            print()
        else:
            print(f"Error: Could not load index from {args.index}")
            exit(1)

    elif args.query:
        # Search
        index = SemanticIndex(index_type=args.index_type)
        if not index.load(args.index):
            print(f"Error: Could not load index from {args.index}")
            exit(1)

        results = retrieve_semantic(args.query, index, top_k=args.top_k)

        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        else:
            print(f"\nSemantic Results for: {args.query}")
            print("=" * 50)
            if results:
                for r in results:
                    print(f"  [{r['rank']}] {r['doc_id']}: {r['score_normalized']:.4f}")
            else:
                print("  No results found")
            print()

    else:
        parser.print_help()
