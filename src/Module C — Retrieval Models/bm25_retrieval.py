"""
Module C - Model 1A: BM25 Lexical Retrieval

This module implements BM25 (Best Matching 25) ranking for lexical document retrieval.
BM25 is a bag-of-words retrieval function that ranks documents based on query term
frequencies, document lengths, and corpus statistics.

WHY BM25 FOR CLIR?
------------------
- Industry standard for lexical retrieval (used by Elasticsearch, Lucene)
- Better than TF-IDF for handling term frequency saturation
- Document length normalization prevents bias toward longer documents
- Well-understood parameters (k1, b) for tuning

FAILURE CASES (Important for IR evaluation):
--------------------------------------------
1. SYNONYMS: BM25 fails when query uses different words with same meaning
   - "car" vs "automobile" - BM25 won't match these
   - Solution: Query expansion or semantic search

2. PARAPHRASES: Different sentence structures with same meaning fail
   - "climate change effects" vs "how global warming impacts us"
   - BM25 relies on exact term overlap

3. CROSS-LINGUAL: Cannot bridge language gaps
   - English query won't match Bangla documents (different scripts)
   - Solution: Translation + semantic embeddings

4. CROSS-SCRIPT: English ↔ বাংলা matching impossible with BM25
   - "Dhaka" won't match "ঢাকা"
   - Solution: Transliteration + fuzzy matching

5. MORPHOLOGICAL VARIATIONS: Different word forms may not match
   - "running" vs "run" without stemming/lemmatization
   - Solution: Text preprocessing with stemming
"""

import logging
import time
import sys
from typing import List, Dict, Tuple, Optional, Any
import pickle
import os

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

# Module-level lazy loading for BM25 index
_bm25_index = None


class BM25Index:
    """
    BM25 Index for lexical document retrieval.

    Uses rank-bm25 library for efficient BM25 scoring.
    Supports language-specific tokenization.

    Attributes:
        bm25: The BM25Okapi instance
        doc_ids: List of document IDs in index order
        tokenized_corpus: Pre-tokenized documents
        k1: Term frequency saturation parameter (default: 1.5)
        b: Document length normalization parameter (default: 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index with tuning parameters.

        Args:
            k1: Controls term frequency saturation (1.2-2.0 typical)
                Higher k1 = more weight to term frequency
            b: Controls document length normalization (0.0-1.0)
                b=0: No length normalization
                b=1: Full length normalization
        """
        self.bm25 = None
        self.doc_ids = []
        self.tokenized_corpus = []
        self.k1 = k1
        self.b = b
        self._is_built = False

    def build(
        self, documents: List[Dict[str, Any]], text_field: str = "content"
    ) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: List of document dictionaries with 'id' and text_field
            text_field: Field name containing document text (default: "content")

        Example:
            >>> index = BM25Index()
            >>> docs = [
            ...     {"id": "doc1", "content": "climate change effects"},
            ...     {"id": "doc2", "content": "global warming impact"}
            ... ]
            >>> index.build(docs)
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error("rank-bm25 not installed. Install with: pip install rank-bm25")
            raise ImportError("rank-bm25 required for BM25 retrieval")

        if not documents:
            logger.warning("Empty document list provided to BM25Index.build()")
            return

        logger.info(f"Building BM25 index with {len(documents)} documents...")
        start_time = time.time()

        # Extract document IDs and tokenize content
        self.doc_ids = []
        self.tokenized_corpus = []

        for doc in documents:
            doc_id = doc.get("id", doc.get("doc_id", str(len(self.doc_ids))))
            text = doc.get(text_field, "")

            if not text:
                logger.warning(f"Document {doc_id} has empty text field '{text_field}'")
                text = ""

            # Simple whitespace tokenization (can be enhanced with NLP)
            tokens = self._tokenize(text)

            self.doc_ids.append(doc_id)
            self.tokenized_corpus.append(tokens)

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        self._is_built = True

        elapsed = time.time() - start_time
        logger.info(f"BM25 index built in {elapsed:.2f}s ({len(documents)} documents)")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Simple whitespace + punctuation tokenization.
        Lowercase for case-insensitive matching.

        Args:
            text: Input text string

        Returns:
            List of lowercase tokens
        """
        if not text:
            return []

        # Convert to lowercase and split on whitespace
        text = text.lower()

        # Remove common punctuation but preserve hyphens (COVID-19)
        import re

        tokens = re.findall(r"\b[\w-]+\b", text)

        return tokens

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        preprocess: bool = True,
        target_lang: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Search index with BM25 ranking.

        Args:
            query: Search query string
            top_k: Number of top results to return
            min_score: Minimum BM25 score threshold
            preprocess: Whether to apply Module B query preprocessing (default: True)
            target_lang: Target language for translation ('bn' or 'en')

        Returns:
            List of results with doc_id, score, and rank

        Example:
            >>> results = index.search("climate change", top_k=5)
            >>> for r in results:
            ...     print(f"{r['doc_id']}: {r['score']:.4f}")
        """
        if not self._is_built or self.bm25 is None:
            logger.error("BM25 index not built. Call build() first.")
            return []

        if not query or not query.strip():
            logger.warning("Empty query provided to BM25 search")
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

        # Tokenize query
        query_tokens = self._tokenize(search_query)

        if not query_tokens:
            logger.warning("Query produced no tokens after tokenization")
            return []

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Create (doc_id, score) pairs and sort by score descending
        doc_scores = list(zip(self.doc_ids, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Filter by min_score and limit to top_k
        results = []
        for rank, (doc_id, score) in enumerate(doc_scores[:top_k], 1):
            if score >= min_score:
                results.append(
                    {
                        "doc_id": doc_id,
                        "score": float(score),
                        "rank": rank,
                        "method": "bm25",
                    }
                )

        elapsed = time.time() - start_time
        logger.debug(f"BM25 search completed in {elapsed*1000:.2f}ms")

        return results

    def get_normalized_scores(
        self,
        query: str,
        top_k: int = 10,
        preprocess: bool = True,
        target_lang: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Search with scores normalized to [0, 1] range.

        Normalization: score_norm = (score - min) / (max - min)

        Args:
            query: Search query string
            top_k: Number of results
            preprocess: Whether to apply query preprocessing (default: True)
            target_lang: Target language for translation ('bn' or 'en')

        Returns:
            Results with normalized scores in [0, 1]
        """
        results = self.search(
            query, top_k=top_k * 2, preprocess=preprocess, target_lang=target_lang
        )  # Get more for normalization

        if not results:
            return []

        # Get min/max scores
        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)

        # Normalize scores
        score_range = max_score - min_score
        if score_range == 0:
            # All scores equal - set to 0.5
            for r in results:
                r["score_normalized"] = 0.5
        else:
            for r in results:
                r["score_normalized"] = (r["score"] - min_score) / score_range

        return results[:top_k]

    def save(self, filepath: str) -> None:
        """Save BM25 index to disk."""
        if not self._is_built:
            logger.error("Cannot save: index not built")
            return

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "doc_ids": self.doc_ids,
                    "tokenized_corpus": self.tokenized_corpus,
                    "k1": self.k1,
                    "b": self.b,
                },
                f,
            )
        logger.info(f"BM25 index saved to {filepath}")

    def load(self, filepath: str) -> bool:
        """Load BM25 index from disk."""
        if not os.path.exists(filepath):
            logger.error(f"Index file not found: {filepath}")
            return False

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            self.bm25 = data["bm25"]
            self.doc_ids = data["doc_ids"]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.k1 = data.get("k1", 1.5)
            self.b = data.get("b", 0.75)
            self._is_built = True

            logger.info(
                f"BM25 index loaded from {filepath} ({len(self.doc_ids)} documents)"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading BM25 index: {e}")
            return False


def build_bm25_index(
    documents: List[Dict[str, Any]],
    text_field: str = "content",
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Index:
    """
    Convenience function to build a BM25 index.

    Args:
        documents: List of document dicts with 'id' and text content
        text_field: Field containing document text
        k1: BM25 k1 parameter
        b: BM25 b parameter

    Returns:
        Built BM25Index instance

    Example:
        >>> docs = [{"id": "1", "content": "hello world"}]
        >>> index = build_bm25_index(docs)
        >>> results = index.search("hello")
    """
    index = BM25Index(k1=k1, b=b)
    index.build(documents, text_field=text_field)
    return index


def retrieve_bm25(
    query: str, index: BM25Index, top_k: int = 10, normalize: bool = True
) -> List[Dict[str, Any]]:
    """
    Retrieve documents using BM25 ranking.

    Args:
        query: Search query string
        index: Built BM25Index instance
        top_k: Number of results to return
        normalize: Whether to normalize scores to [0, 1]

    Returns:
        List of ranked results with scores
    """
    if normalize:
        return index.get_normalized_scores(query, top_k=top_k)
    else:
        return index.search(query, top_k=top_k)


def compare_bm25_queries(
    queries: List[str], index: BM25Index, top_k: int = 5
) -> Dict[str, List[Dict]]:
    """
    Compare BM25 retrieval across multiple queries.

    Useful for analyzing failure cases.

    Args:
        queries: List of query strings
        index: Built BM25Index
        top_k: Results per query

    Returns:
        Dictionary mapping queries to their results
    """
    comparison = {}
    total_time = 0

    for query in queries:
        start = time.time()
        results = index.get_normalized_scores(query, top_k=top_k)
        elapsed = time.time() - start
        total_time += elapsed

        comparison[query] = {
            "results": results,
            "retrieval_time_ms": elapsed * 1000,
            "num_results": len(results),
        }

    avg_time = (total_time / len(queries)) * 1000 if queries else 0
    logger.info(
        f"BM25 comparison: {len(queries)} queries, avg {avg_time:.2f}ms per query"
    )

    return comparison


# Command line interface
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="BM25 Lexical Retrieval for CLIR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search with a query (requires pre-built index)
  python bm25_retrieval.py "climate change" --index indexes/bm25_index.pkl
  
  # Build index from JSON documents
  python bm25_retrieval.py --build --data data/documents.json --output indexes/bm25_index.pkl
        """,
    )

    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument(
        "--index", "-i", default="indexes/bm25_index.pkl", help="Path to BM25 index"
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

        index = build_bm25_index(docs)
        output_path = args.output or "indexes/bm25_index.pkl"
        index.save(output_path)
        print(f"Index built and saved to {output_path}")

    elif args.query:
        index = BM25Index()
        if not index.load(args.index):
            print(f"Error: Could not load index from {args.index}")
            exit(1)

        results = retrieve_bm25(args.query, index, top_k=args.top_k)

        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print(f"\nBM25 Results for: {args.query}")
            print("=" * 50)
            for r in results:
                score = r.get("score_normalized", r["score"])
                print(f"  [{r['rank']}] {r['doc_id']}: {score:.4f}")
            print()

    else:
        parser.print_help()
