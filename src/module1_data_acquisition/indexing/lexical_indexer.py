"""
Lexical Indexer using WHOOSH

Creates an inverted index with BM25F scoring for fast keyword-based retrieval.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List
from whoosh import fields, index
from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.filedb.filestore import FileStorage

logger = logging.getLogger(__name__)


class LexicalIndexer:
    """
    WHOOSH-based inverted index with BM25F weighting.
    Supports language-specific analysis and boosted title field.
    """

    def __init__(self, index_dir: str = "indexes/whoosh"):
        """
        Initialize lexical indexer.

        Args:
            index_dir: Directory to store WHOOSH index
        """
        self.index_dir = index_dir
        self.index = None
        self.schema = self._create_schema()

        # Create index directory
        os.makedirs(self.index_dir, exist_ok=True)

    def _create_schema(self):
        """
        Create WHOOSH schema for documents.

        Schema fields:
        - doc_id: Unique document identifier
        - title: Article title (boosted)
        - body: Article body text
        - language: Language code (bn/en)
        - source: News source
        - date: Publication date
        - entities: Named entities (comma-separated)
        """
        return fields.Schema(
            doc_id=fields.ID(stored=True, unique=True),
            title=fields.TEXT(
                stored=True,
                analyzer=StandardAnalyzer(),
                field_boost=2.0,  # Boost title matches
            ),
            body=fields.TEXT(analyzer=StemmingAnalyzer()),
            language=fields.KEYWORD(stored=True),
            source=fields.KEYWORD(stored=True),
            date=fields.DATETIME(stored=True),
            entities=fields.KEYWORD(stored=True, commas=True),
        )

    def create_index(self):
        """Create a new empty index."""
        logger.info(f"Creating new WHOOSH index at {self.index_dir}")

        # Remove existing index if present
        if index.exists_in(self.index_dir):
            logger.warning("Existing index found, will be overwritten")

        self.index = index.create_in(self.index_dir, self.schema)
        logger.info("✓ Index created")

    def open_index(self):
        """Open existing index."""
        if not index.exists_in(self.index_dir):
            raise FileNotFoundError(f"No index found at {self.index_dir}")

        self.index = index.open_dir(self.index_dir)
        logger.info(f"✓ Opened existing index with {self.index.doc_count()} documents")

    def add_document(self, doc_id: str, doc_data: Dict):
        """
        Add a single document to the index.

        Args:
            doc_id: Unique document ID
            doc_data: Dictionary with keys: title, body, language, source, date, entities
        """
        if not self.index:
            raise RuntimeError("Index not initialized. Call create_index() first.")

        writer = self.index.writer()

        try:
            # Parse date if string
            date_val = doc_data.get("date")
            if isinstance(date_val, str):
                try:
                    date_val = datetime.fromisoformat(date_val)
                except:
                    date_val = None

            # Join entities list to comma-separated string
            entities = doc_data.get("entities", [])
            if isinstance(entities, list):
                entities = ",".join(entities)

            writer.add_document(
                doc_id=doc_id,
                title=doc_data.get("title", ""),
                body=doc_data.get("body", ""),
                language=doc_data.get("language", "en"),
                source=doc_data.get("source", ""),
                date=date_val,
                entities=entities,
            )

            writer.commit()

        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            writer.cancel()
            raise

    def add_documents_batch(self, documents: List[Dict]):
        """
        Add multiple documents in batch.

        Args:
            documents: List of dicts with keys: doc_id, title, body, language, source, date, entities
        """
        if not self.index:
            raise RuntimeError("Index not initialized. Call create_index() first.")

        logger.info(f"Indexing {len(documents)} documents...")

        writer = self.index.writer()

        try:
            for doc in documents:
                # Parse date
                date_val = doc.get("date")
                if isinstance(date_val, str):
                    try:
                        date_val = datetime.fromisoformat(date_val)
                    except:
                        date_val = None
                elif date_val is not None and str(date_val) == "nan":
                    date_val = None

                # Join entities
                entities = doc.get("entities", [])
                if isinstance(entities, list):
                    entities = ",".join(entities)

                writer.add_document(
                    doc_id=doc["doc_id"],
                    title=doc.get("title", ""),
                    body=doc.get("body", ""),
                    language=doc.get("language", "en"),
                    source=doc.get("source", ""),
                    date=date_val,
                    entities=entities,
                )

            logger.info("Committing changes to index...")
            writer.commit()
            logger.info(f"✓ Successfully indexed {len(documents)} documents")

        except Exception as e:
            logger.error(f"Batch indexing failed: {e}")
            writer.cancel()
            raise

    def search(
        self, query_text: str, language: str = None, limit: int = 10
    ) -> List[Dict]:
        """
        Search the index using BM25F ranking.

        Args:
            query_text: Query string
            language: Filter by language (optional)
            limit: Maximum number of results

        Returns:
            List of result dictionaries with doc_id, title, score
        """
        if not self.index:
            self.open_index()

        with self.index.searcher() as searcher:
            # Search in title and body fields
            parser = MultifieldParser(["title", "body"], schema=self.schema)
            query = parser.parse(query_text)

            # Add language filter if specified
            if language:
                from whoosh.query import Term, And

                lang_filter = Term("language", language)
                query = And([query, lang_filter])

            results = searcher.search(query, limit=limit)

            # Convert to list of dicts
            output = []
            for hit in results:
                output.append(
                    {
                        "doc_id": hit["doc_id"],
                        "title": hit.get("title", ""),
                        "language": hit.get("language", ""),
                        "source": hit.get("source", ""),
                        "score": hit.score,
                    }
                )

            return output

    def get_stats(self) -> Dict:
        """Get index statistics."""
        if not self.index:
            self.open_index()

        return {"total_documents": self.index.doc_count(), "index_dir": self.index_dir}
