"""
Document Processor Module for Cross-Lingual Information Retrieval System.

Orchestrates the complete preprocessing pipeline: language detection,
tokenization, NER extraction, and indexing.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from tqdm import tqdm
import time

from .language_detector import LanguageDetector
from .tokenizer import MultilingualTokenizer
from .ner_extractor import NERExtractor
from .inverted_index import InvertedIndex

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Orchestrate complete document preprocessing pipeline.

    Pipeline steps:
    1. Read JSON documents from data directory
    2. Detect language
    3. Tokenize content
    4. Extract named entities
    5. Add to inverted index
    6. Save enhanced metadata

    Example:
        >>> processor = DocumentProcessor('data/', 'processed_data/')
        >>> stats = processor.process_all_documents()
        >>> print(stats)
        {'processed': 5068, 'errors': 0, 'time_elapsed': 125.3}
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str = "processed_data",
        save_enhanced_metadata: bool = True,
    ):
        """
        Initialize the DocumentProcessor.

        Args:
            data_dir: Directory containing raw JSON documents
            output_dir: Directory to save processed data and index
            save_enhanced_metadata: Whether to save enhanced metadata with tokens/entities
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.save_enhanced_metadata = save_enhanced_metadata

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.language_detector = LanguageDetector()
        self.tokenizer = MultilingualTokenizer(
            lowercase=True, remove_punctuation=True, remove_stopwords=False
        )
        self.ner_extractor = NERExtractor()
        self.inverted_index = InvertedIndex()

        # Statistics
        self.stats = {
            "processed": 0,
            "errors": 0,
            "by_language": {"en": 0, "bn": 0, "unknown": 0},
            "total_tokens": 0,
            "total_entities": 0,
        }

        logger.info(
            f"DocumentProcessor initialized (data_dir={data_dir}, output_dir={output_dir})"
        )

    def process_single_document(self, doc_path: Path) -> Optional[Dict[str, Any]]:
        """
        Process a single document through the entire pipeline.

        Args:
            doc_path: Path to the JSON document file

        Returns:
            Dictionary containing processed document data, or None if error

        Example:
            >>> doc = processor.process_single_document(Path('data/doc1.json'))
            >>> print(doc['language'], doc['token_count'], len(doc['entities']))
            'en' 523 12
        """
        try:
            # Read document
            with open(doc_path, "r", encoding="utf-8") as f:
                doc_data = json.load(f)

            # Extract required fields
            doc_id = doc_path.stem  # Use filename without extension
            title = doc_data.get("title", "")
            content = doc_data.get("content", "")
            url = doc_data.get("url", "")
            date = doc_data.get("published_date", "")

            # Combine title and content for processing
            full_text = f"{title} {content}".strip()

            if not full_text:
                logger.warning(f"Empty document: {doc_id}")
                return None

            # Step 1: Detect language
            lang_result = self.language_detector.detect(full_text)
            language = lang_result["language"]
            lang_confidence = lang_result["confidence"]

            if language == "unknown":
                logger.warning(f"Unknown language for document {doc_id}")
                # Try to infer from file path
                if "english" in str(doc_path):
                    language = "en"
                elif "bangla" in str(doc_path):
                    language = "bn"

            # Step 2: Tokenize
            tokens = self.tokenizer.tokenize(full_text, language)

            if not tokens:
                logger.warning(f"No tokens extracted from document {doc_id}")
                return None

            # Step 3: Extract named entities
            entities = self.ner_extractor.extract_entities(full_text, language)

            # Step 4: Build metadata
            metadata = {
                "doc_id": doc_id,
                "title": title,
                "url": url,
                "date": date,
                "language": language,
                "lang_confidence": lang_confidence,
                "token_count": len(tokens),
                "unique_terms": len(set(tokens)),
                "entity_count": len(entities),
                "entities": entities,
                "source_file": str(doc_path),
            }

            # Step 5: Add to inverted index
            self.inverted_index.add_document(doc_id, tokens, metadata)

            # Update statistics
            self.stats["processed"] += 1
            self.stats["by_language"][language] = (
                self.stats["by_language"].get(language, 0) + 1
            )
            self.stats["total_tokens"] += len(tokens)
            self.stats["total_entities"] += len(entities)

            logger.debug(
                f"Processed document {doc_id}: {len(tokens)} tokens, {len(entities)} entities"
            )

            return metadata

        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
            self.stats["errors"] += 1
            return None

    def process_all_documents(
        self, batch_size: int = 100, file_pattern: str = "**/*.json"
    ) -> Dict[str, Any]:
        """
        Process all documents in the data directory.

        Args:
            batch_size: Number of documents to process before saving checkpoint
            file_pattern: Glob pattern for finding JSON files

        Returns:
            Dictionary containing processing statistics

        Example:
            >>> stats = processor.process_all_documents()
            >>> print(f"Processed {stats['processed']} documents")
            Processed 5068 documents
        """
        logger.info(f"Starting document processing from {self.data_dir}")
        start_time = time.time()

        # Find all JSON files
        json_files = list(self.data_dir.glob(file_pattern))
        logger.info(f"Found {len(json_files)} JSON files to process")

        if not json_files:
            logger.warning("No JSON files found!")
            return self.stats

        # Process documents with progress bar
        enhanced_metadata_list = []

        for doc_path in tqdm(json_files, desc="Processing documents"):
            metadata = self.process_single_document(doc_path)

            if metadata and self.save_enhanced_metadata:
                enhanced_metadata_list.append(metadata)

            # Save checkpoint every batch_size documents
            if (
                self.stats["processed"] % batch_size == 0
                and self.stats["processed"] > 0
            ):
                self._save_checkpoint(enhanced_metadata_list)
                enhanced_metadata_list = []

        # Save final checkpoint
        if enhanced_metadata_list:
            self._save_checkpoint(enhanced_metadata_list)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        self.stats["time_elapsed"] = round(elapsed_time, 2)
        self.stats["docs_per_second"] = round(self.stats["processed"] / elapsed_time, 2)

        # Save final index and statistics
        self._save_final_outputs()

        logger.info(f"Processing complete: {self.stats}")
        return self.stats

    def generate_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about processed documents.

        Returns:
            Dictionary containing detailed statistics
        """
        index_stats = self.inverted_index.get_statistics()

        stats = {
            "processing": self.stats,
            "index": index_stats,
            "language_distribution": self.stats["by_language"],
            "avg_tokens_per_doc": round(
                self.stats["total_tokens"] / max(self.stats["processed"], 1), 2
            ),
            "avg_entities_per_doc": round(
                self.stats["total_entities"] / max(self.stats["processed"], 1), 2
            ),
        }

        return stats

    def _save_checkpoint(self, metadata_list: List[Dict[str, Any]]):
        """
        Save checkpoint with enhanced metadata.

        Args:
            metadata_list: List of metadata dictionaries to save
        """
        try:
            checkpoint_file = (
                self.output_dir / f'metadata_checkpoint_{self.stats["processed"]}.json'
            )

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=2)

            logger.debug(f"Checkpoint saved: {checkpoint_file}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def _save_final_outputs(self):
        """Save final index and statistics."""
        try:
            # Save inverted index
            index_file = self.output_dir / "inverted_index.pkl"
            self.inverted_index.save_index(str(index_file), format="pickle")
            logger.info(f"Inverted index saved: {index_file}")

            # Save statistics
            stats_file = self.output_dir / "processing_statistics.json"
            final_stats = self.generate_statistics()

            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(final_stats, f, ensure_ascii=False, indent=2)

            logger.info(f"Statistics saved: {stats_file}")

            # Save document metadata
            metadata_file = self.output_dir / "document_metadata.json"
            all_metadata = {
                doc_id: meta
                for doc_id, meta in self.inverted_index.doc_metadata.items()
            }

            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(all_metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"Document metadata saved: {metadata_file}")

        except Exception as e:
            logger.error(f"Error saving final outputs: {e}")

    def load_existing_index(self, index_file: str) -> bool:
        """
        Load an existing inverted index.

        Args:
            index_file: Path to the index file

        Returns:
            True if loaded successfully, False otherwise
        """
        return self.inverted_index.load_index(index_file, format="pickle")
