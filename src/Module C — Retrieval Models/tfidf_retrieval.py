"""
Module C - Model 1B: TF-IDF Lexical Retrieval

This module implements TF-IDF (Term Frequency-Inverse Document Frequency) retrieval.
Included for comparison with BM25 to demonstrate ranking differences.

TF-IDF vs BM25 COMPARISON:
--------------------------
- TF-IDF: Linear term frequency weighting
- BM25: Saturating term frequency (diminishing returns for repeated terms)

TF-IDF Formula:
  tfidf(t, d) = tf(t, d) * log(N / df(t))

Where:
  tf(t, d) = frequency of term t in document d
  N = total number of documents
  df(t) = number of documents containing term t

WHEN TF-IDF BEATS BM25:
-----------------------
- Very short queries (1-2 terms)
- When term frequency isn't inflated (academic papers, news articles)

WHEN BM25 BEATS TF-IDF:
-----------------------
- Longer queries
- Documents with repetitive terms (spam-like content)
- Collections with varying document lengths

SAME FAILURE CASES AS BM25:
---------------------------
- Synonyms, paraphrases, cross-lingual, cross-script
- All lexical models share vocabulary mismatch problems
"""

import logging
import time
import sys
from typing import List, Dict, Any, Optional, Tuple
import pickle
import os
import numpy as np

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


class TFIDFIndex:
    """
    TF-IDF Index for lexical document retrieval.

    Uses scikit-learn's TfidfVectorizer for efficient sparse matrix operations.

    Attributes:
        vectorizer: TfidfVectorizer instance
        tfidf_matrix: Sparse TF-IDF matrix (documents x terms)
        doc_ids: List of document IDs
    """

    def __init__(
        self,
        max_features: Optional[int] = None,
        ngram_range: Tuple[int, int] = (1, 1),
        min_df: int = 1,
        max_df: float = 1.0,
    ):
        """
        Initialize TF-IDF index with parameters.

        Args:
            max_features: Maximum vocabulary size (None = unlimited)
            ngram_range: (min_n, max_n) for n-gram extraction
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency ratio (0.0-1.0)
        """
        self.vectorizer = None
        self.tfidf_matrix = None
        self.doc_ids = []
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self._is_built = False

    def build(
        self, documents: List[Dict[str, Any]], text_field: str = "content"
    ) -> None:
        """
        Build TF-IDF index from documents.

        Args:
            documents: List of document dicts with 'id' and text_field
            text_field: Field name containing document text
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            logger.error(
                "scikit-learn not installed. Install with: pip install scikit-learn"
            )
            raise ImportError("scikit-learn required for TF-IDF retrieval")

        if not documents:
            logger.warning("Empty document list provided to TFIDFIndex.build()")
            return

        logger.info(f"Building TF-IDF index with {len(documents)} documents...")
        start_time = time.time()

        # Extract document IDs and texts
        self.doc_ids = []
        texts = []

        for doc in documents:
            doc_id = doc.get("id", doc.get("doc_id", str(len(self.doc_ids))))
            text = doc.get(text_field, "")

            if not text:
                text = ""  # Handle empty documents

            self.doc_ids.append(doc_id)
            texts.append(text)

        # Build TF-IDF vectorizer and matrix
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=True,
            token_pattern=r"\b[\w-]+\b",  # Keep hyphens for terms like COVID-19
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self._is_built = True

        vocab_size = len(self.vectorizer.vocabulary_)
        elapsed = time.time() - start_time
        logger.info(
            f"TF-IDF index built in {elapsed:.2f}s ({len(documents)} docs, {vocab_size} terms)"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        preprocess: bool = True,
        target_lang: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Search index with TF-IDF cosine similarity.

        Args:
            query: Search query string
            top_k: Number of top results to return
            min_score: Minimum similarity threshold [0, 1]
            preprocess: Whether to apply Module B query preprocessing (default: True)
            target_lang: Target language for translation ('bn' or 'en')

        Returns:
            List of results with doc_id, score (already normalized), and rank
        """
        if not self._is_built or self.vectorizer is None:
            logger.error("TF-IDF index not built. Call build() first.")
            return []

        if not query or not query.strip():
            logger.warning("Empty query provided to TF-IDF search")
            return []

        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            logger.error("scikit-learn required for cosine similarity")
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

        # Transform query to TF-IDF vector
        query_vec = self.vectorizer.transform([search_query])

        # Compute cosine similarity with all documents
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices sorted by similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(similarities[idx])
            if score >= min_score:
                results.append(
                    {
                        "doc_id": self.doc_ids[idx],
                        "score": score,  # Already in [0, 1] due to cosine similarity
                        "score_normalized": score,
                        "rank": rank,
                        "method": "tfidf",
                    }
                )

        elapsed = time.time() - start_time
        logger.debug(f"TF-IDF search completed in {elapsed*1000:.2f}ms")

        return results

    def get_term_weights(self, query: str, doc_id: str) -> Dict[str, float]:
        """
        Get TF-IDF weights for query terms in a specific document.

        Useful for explaining why a document was retrieved.

        Args:
            query: Query string
            doc_id: Document ID to inspect

        Returns:
            Dictionary mapping terms to their TF-IDF weights
        """
        if not self._is_built:
            return {}

        try:
            doc_idx = self.doc_ids.index(doc_id)
        except ValueError:
            logger.warning(f"Document {doc_id} not found in index")
            return {}

        # Get query terms
        query_terms = query.lower().split()

        # Get term weights for document
        feature_names = self.vectorizer.get_feature_names_out()
        doc_vector = self.tfidf_matrix[doc_idx].toarray().flatten()

        weights = {}
        for term in query_terms:
            if term in feature_names:
                term_idx = list(feature_names).index(term)
                weights[term] = float(doc_vector[term_idx])
            else:
                weights[term] = 0.0

        return weights

    def save(self, filepath: str) -> None:
        """Save TF-IDF index to disk."""
        if not self._is_built:
            logger.error("Cannot save: index not built")
            return

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "tfidf_matrix": self.tfidf_matrix,
                    "doc_ids": self.doc_ids,
                    "max_features": self.max_features,
                    "ngram_range": self.ngram_range,
                    "min_df": self.min_df,
                    "max_df": self.max_df,
                },
                f,
            )
        logger.info(f"TF-IDF index saved to {filepath}")

    def load(self, filepath: str) -> bool:
        """Load TF-IDF index from disk."""
        if not os.path.exists(filepath):
            logger.error(f"Index file not found: {filepath}")
            return False

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            self.vectorizer = data["vectorizer"]
            self.tfidf_matrix = data["tfidf_matrix"]
            self.doc_ids = data["doc_ids"]
            self.max_features = data.get("max_features")
            self.ngram_range = data.get("ngram_range", (1, 1))
            self.min_df = data.get("min_df", 1)
            self.max_df = data.get("max_df", 1.0)
            self._is_built = True

            logger.info(
                f"TF-IDF index loaded from {filepath} ({len(self.doc_ids)} documents)"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading TF-IDF index: {e}")
            return False


def build_tfidf_index(
    documents: List[Dict[str, Any]], text_field: str = "content", **kwargs
) -> TFIDFIndex:
    """
    Convenience function to build a TF-IDF index.

    Args:
        documents: List of document dicts
        text_field: Field containing document text
        **kwargs: Additional TFIDFIndex parameters

    Returns:
        Built TFIDFIndex instance
    """
    index = TFIDFIndex(**kwargs)
    index.build(documents, text_field=text_field)
    return index


def retrieve_tfidf(
    query: str, index: TFIDFIndex, top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Retrieve documents using TF-IDF ranking.

    Args:
        query: Search query string
        index: Built TFIDFIndex instance
        top_k: Number of results to return

    Returns:
        List of ranked results with scores (already in [0, 1])
    """
    return index.search(query, top_k=top_k)


def compare_bm25_tfidf(
    query: str, bm25_results: List[Dict], tfidf_results: List[Dict]
) -> Dict[str, Any]:
    """
    Compare BM25 and TF-IDF retrieval results.

    Args:
        query: The search query
        bm25_results: Results from BM25 retrieval
        tfidf_results: Results from TF-IDF retrieval

    Returns:
        Comparison statistics and analysis
    """
    bm25_ids = [r["doc_id"] for r in bm25_results]
    tfidf_ids = [r["doc_id"] for r in tfidf_results]

    # Set operations for comparison
    bm25_set = set(bm25_ids)
    tfidf_set = set(tfidf_ids)

    overlap = bm25_set & tfidf_set
    bm25_only = bm25_set - tfidf_set
    tfidf_only = tfidf_set - bm25_set

    # Rank correlation (Spearman for overlapping documents)
    rank_diff = []
    for doc_id in overlap:
        bm25_rank = bm25_ids.index(doc_id) + 1
        tfidf_rank = tfidf_ids.index(doc_id) + 1
        rank_diff.append(abs(bm25_rank - tfidf_rank))

    avg_rank_diff = sum(rank_diff) / len(rank_diff) if rank_diff else 0

    return {
        "query": query,
        "bm25_count": len(bm25_results),
        "tfidf_count": len(tfidf_results),
        "overlap_count": len(overlap),
        "bm25_only": list(bm25_only),
        "tfidf_only": list(tfidf_only),
        "avg_rank_difference": avg_rank_diff,
        "overlap_ratio": (
            len(overlap) / max(len(bm25_set), len(tfidf_set))
            if bm25_set or tfidf_set
            else 0
        ),
    }


# Command line interface
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="TF-IDF Lexical Retrieval for CLIR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search with a query
  python tfidf_retrieval.py "climate change" --index indexes/tfidf_index.pkl
  
  # Build index from JSON documents
  python tfidf_retrieval.py --build --data data/documents.json --output indexes/tfidf_index.pkl
        """,
    )

    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument(
        "--index", "-i", default="indexes/tfidf_index.pkl", help="Path to TF-IDF index"
    )
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of results")
    parser.add_argument("--build", action="store_true", help="Build index from data")
    parser.add_argument("--data", "-d", help="Path to documents JSON file")
    parser.add_argument("--output", "-o", help="Output path for built index")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.build:
        if not args.data:
            print("Error: --data required when building index")
            exit(1)

        with open(args.data, "r", encoding="utf-8") as f:
            docs = json.load(f)

        index = build_tfidf_index(docs)
        output_path = args.output or "indexes/tfidf_index.pkl"
        index.save(output_path)
        print(f"Index built and saved to {output_path}")

    elif args.query:
        index = TFIDFIndex()
        if not index.load(args.index):
            print(f"Error: Could not load index from {args.index}")
            exit(1)

        results = retrieve_tfidf(args.query, index, top_k=args.top_k)

        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print(f"\nTF-IDF Results for: {args.query}")
            print("=" * 50)
            for r in results:
                print(f"  [{r['rank']}] {r['doc_id']}: {r['score']:.4f}")
            print()

    else:
        parser.print_help()
