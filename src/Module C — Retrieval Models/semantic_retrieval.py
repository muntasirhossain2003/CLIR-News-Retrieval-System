"""
Module C - Model 3: Semantic Retrieval Using Multilingual Embeddings

PURPOSE:
Enable cross-lingual semantic matching using multilingual sentence embeddings.
This model can match queries and documents based on meaning rather than exact
word overlap, enabling true cross-lingual information retrieval.

MODEL:
- sentence-transformers multilingual SBERT
- paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions)
- Supports 50+ languages including English and Bangla

SIMILARITY:
- Cosine similarity between query and document embeddings

This is the first model that can handle cross-lingual retrieval effectively.
"""

import logging
from typing import List, Tuple, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance (lazy loading)
_embedding_model = None


def _get_embedding_model():
    """
    Get or initialize the sentence transformer model.

    Uses lazy loading to avoid loading model until needed.
    Model is loaded once and reused for all subsequent calls.

    Returns:
        SentenceTransformer model instance
    """
    global _embedding_model

    if _embedding_model is None:
        logger.info("Loading multilingual sentence transformer model...")
        logger.info("  Model: paraphrase-multilingual-MiniLM-L12-v2")
        logger.info("  This may take a moment on first use...")

        try:
            _embedding_model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
            logger.info("  Model loaded successfully")
            logger.info(
                f"  Embedding dimension: {_embedding_model.get_sentence_embedding_dimension()}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    return _embedding_model


def encode_documents(
    docs: Dict[str, str], show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """
    Encode documents into multilingual embeddings.

    Args:
        docs: Dictionary mapping doc_id -> document_text
        show_progress: Whether to show progress bar (default: True)

    Returns:
        Dictionary mapping doc_id -> embedding vector (numpy array)
        Embeddings are 384-dimensional L2-normalized vectors

    Example:
        >>> docs = {
        ...     'doc1': 'Climate change is a serious issue',
        ...     'doc2': 'জলবায়ু পরিবর্তন একটি গুরুতর সমস্যা'  # Bangla
        ... }
        >>> embeddings = encode_documents(docs)
        >>> print(f"Encoded {len(embeddings)} documents")

    Notes:
        - Uses paraphrase-multilingual-MiniLM-L12-v2
        - Embeddings are automatically normalized for cosine similarity
        - Works with mixed-language collections
        - First call loads model (~400MB download if not cached)

    Performance:
        - ~100-500 docs/second depending on hardware
        - GPU acceleration supported if available
        - Batch processing for efficiency
    """
    if not docs:
        raise ValueError("Cannot encode empty document collection")

    logger.info(f"Encoding {len(docs)} documents...")

    try:
        # Get model
        model = _get_embedding_model()

        # Extract doc_ids and texts in consistent order
        doc_ids = list(docs.keys())
        doc_texts = [docs[doc_id] for doc_id in doc_ids]

        # Encode all documents at once (batch processing)
        embeddings = model.encode(
            doc_texts,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
            convert_to_numpy=True,
        )

        # Create mapping from doc_id to embedding
        doc_embeddings = {
            doc_id: embedding for doc_id, embedding in zip(doc_ids, embeddings)
        }

        logger.info(f"Successfully encoded {len(doc_embeddings)} documents")
        logger.info(f"  Embedding shape: {embeddings[0].shape}")

        return doc_embeddings

    except Exception as e:
        logger.error(f"Error encoding documents: {e}")
        raise


def retrieve_semantic(
    query: str,
    doc_embeddings: Dict[str, np.ndarray],
    top_k: int = 10,
    normalize_query: bool = True,
) -> List[Tuple[str, float]]:
    """
    Retrieve documents using semantic similarity (cosine similarity).

    Args:
        query: Query string (can be in any language)
        doc_embeddings: Dictionary mapping doc_id -> embedding vector
        top_k: Number of top documents to retrieve (default: 10)
        normalize_query: Whether to normalize query embedding (default: True)

    Returns:
        List of (doc_id, score) tuples, sorted by score descending
        Scores are cosine similarity values in range [-1, 1], typically [0, 1]

    Example:
        >>> query = "climate change"
        >>> results = retrieve_semantic(query, doc_embeddings, top_k=5)
        >>> for doc_id, score in results:
        ...     print(f"{doc_id}: {score:.4f}")

    Cross-Lingual Example:
        >>> # English query on Bangla documents
        >>> query_en = "climate change"
        >>> results = retrieve_semantic(query_en, bangla_embeddings, top_k=5)
        >>> # Returns Bangla documents about climate change!

    Notes:
        - Query is encoded using same multilingual model
        - Cosine similarity: sim(q, d) = q · d / (||q|| ||d||)
        - With normalized embeddings, this simplifies to dot product
        - Works across all 50+ supported languages
        - No translation needed - semantic matching handles it

    Advantages:
        - Handles synonyms (car ≈ automobile)
        - Handles paraphrases (climate change ≈ global warming)
        - Works cross-lingually (English ≈ Bangla)
        - No vocabulary mismatch problems

    Limitations:
        - Slower than lexical methods
        - Requires more memory (embeddings storage)
        - May miss exact keyword matches
    """
    if not query or not query.strip():
        logger.warning("Empty query provided, returning empty results")
        return []

    if not doc_embeddings:
        logger.warning("Empty document embeddings provided")
        return []

    try:
        # Get model and encode query
        model = _get_embedding_model()

        logger.info(f"Encoding query: '{query[:50]}...'")
        query_embedding = model.encode(
            [query], normalize_embeddings=normalize_query, convert_to_numpy=True
        )[0]

        # Compute cosine similarity with all documents
        doc_ids = list(doc_embeddings.keys())
        doc_vectors = np.array([doc_embeddings[doc_id] for doc_id in doc_ids])

        # Cosine similarity = dot product (when vectors are normalized)
        similarities = np.dot(doc_vectors, query_embedding)

        # Get top-K results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results list
        results = [
            (doc_ids[idx], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > 0  # Only positive similarities
        ]

        logger.info(f"Retrieved {len(results)} documents for query (top_k={top_k})")
        if results:
            logger.info(f"  Top score: {results[0][1]:.4f}")
            logger.info(f"  Lowest score: {results[-1][1]:.4f}")
        else:
            logger.warning("  No matching documents found with positive similarity")

        return results

    except Exception as e:
        logger.error(f"Error during semantic retrieval: {e}")
        return []


def retrieve_semantic_with_query_embedding(
    query_embedding: np.ndarray, doc_embeddings: Dict[str, np.ndarray], top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Retrieve documents using pre-computed query embedding.

    Useful when you want to reuse the same query embedding multiple times
    or when integrating with Module B (query processing pipeline).

    Args:
        query_embedding: Pre-computed query embedding vector
        doc_embeddings: Dictionary mapping doc_id -> embedding vector
        top_k: Number of top documents to retrieve (default: 10)

    Returns:
        List of (doc_id, score) tuples, sorted by score descending

    Example:
        >>> model = _get_embedding_model()
        >>> query_emb = model.encode(["climate change"], normalize_embeddings=True)[0]
        >>> results = retrieve_semantic_with_query_embedding(query_emb, doc_embeddings)

    Notes:
        - Assumes query_embedding is already normalized
        - Faster than retrieve_semantic() when reusing embeddings
        - Useful for batch query processing
    """
    if query_embedding is None or len(query_embedding) == 0:
        logger.warning("Empty query embedding provided")
        return []

    if not doc_embeddings:
        logger.warning("Empty document embeddings provided")
        return []

    try:
        # Compute cosine similarity with all documents
        doc_ids = list(doc_embeddings.keys())
        doc_vectors = np.array([doc_embeddings[doc_id] for doc_id in doc_ids])

        # Cosine similarity = dot product (when vectors are normalized)
        similarities = np.dot(doc_vectors, query_embedding)

        # Get top-K results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results list
        results = [
            (doc_ids[idx], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > 0
        ]

        logger.info(f"Retrieved {len(results)} documents (top_k={top_k})")

        return results

    except Exception as e:
        logger.error(
            f"Error during semantic retrieval with pre-computed embedding: {e}"
        )
        return []


def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts.

    Utility function for testing or comparing individual texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Cosine similarity score in range [-1, 1], typically [0, 1]

    Example:
        >>> sim1 = compute_similarity("climate change", "global warming")
        >>> sim2 = compute_similarity("climate change", "জলবায়ু পরিবর্তন")
        >>> print(f"Synonym similarity: {sim1:.4f}")
        >>> print(f"Cross-lingual similarity: {sim2:.4f}")
    """
    if not text1 or not text2:
        return 0.0

    try:
        model = _get_embedding_model()

        embeddings = model.encode(
            [text1, text2], normalize_embeddings=True, convert_to_numpy=True
        )

        # Cosine similarity
        similarity = float(np.dot(embeddings[0], embeddings[1]))

        return similarity

    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return 0.0
