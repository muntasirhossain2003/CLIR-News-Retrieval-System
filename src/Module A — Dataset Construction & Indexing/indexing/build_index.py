import os
import sys
import json
import argparse
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add current directory to path for local imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from language_detector import LanguageDetector
from preprocessor import TextPreprocessor
from lexical_indexer import LexicalIndexer
from semantic_indexer import SemanticIndexer

# Create logs directory in project root
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, "indexing.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_documents(metadata_path: str, limit: int = 0) -> list:
    # Resolve path relative to project root if not absolute
    if not os.path.isabs(metadata_path):
        metadata_path = os.path.join(project_root, metadata_path)

    logger.info(f"Loading documents from {metadata_path}")

    # Read metadata
    df = pd.read_csv(metadata_path, encoding="utf-8")

    if limit > 0:
        df = df.head(limit)
        logger.info(f"Limited to {limit} documents")

    logger.info(f"Found {len(df)} documents in metadata")

    # Load full document content
    documents = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading documents"):
        try:
            # Resolve file path relative to project root
            filepath = os.path.join(
                project_root,
                "data",
                "raw",
                row["language"],
                row["source"],
                row["filename"],
            )

            if not os.path.exists(filepath):
                logger.warning(
                    f"File not found: {filepath} (lang={row['language']}, source={row['source']}, file={row['filename']})"
                )
                continue

            # Load JSON
            with open(filepath, "r", encoding="utf-8") as f:
                doc_data = json.load(f)

            # Create document dict
            doc = {
                "doc_id": row["filename"].replace(".json", ""),
                "title": doc_data.get("title", ""),
                "body": doc_data.get("body", ""),
                "url": doc_data.get("url", ""),
                "source": row["source"],
                "date": row.get("date"),
                "language": row["language"],  # From metadata
            }

            documents.append(doc)

        except Exception as e:
            logger.error(f"Error loading document {row['filename']}: {e}")
            continue

    logger.info(f"✓ Loaded {len(documents)} documents successfully")
    return documents


def process_documents(
    documents: list, lang_detector: LanguageDetector, preprocessor: TextPreprocessor
) -> list:
    logger.info("Processing documents...")

    processed = []

    for doc in tqdm(documents, desc="Processing"):
        try:
            # Detect language (verify metadata)
            detected_lang = lang_detector.detect(doc["body"], default=doc["language"])

            # Use metadata language if different (trust our crawlers more)
            if detected_lang != doc["language"]:
                logger.debug(
                    f"Language mismatch: metadata={doc['language']}, detected={detected_lang} for {doc['doc_id']}"
                )

            # Preprocess text
            combined_text = f"{doc['title']} {doc['body']}"
            preprocessed = preprocessor.preprocess(combined_text, doc["language"])

            # Add preprocessing results to document
            doc["detected_language"] = detected_lang
            doc["token_count"] = preprocessed["token_count"]
            doc["entities"] = preprocessed["entities"]
            doc["cleaned_text"] = preprocessed["cleaned_text"]

            processed.append(doc)

        except Exception as e:
            logger.error(
                f"Error processing document {doc.get('doc_id', 'unknown')}: {e}"
            )
            continue

    logger.info(f"✓ Processed {len(processed)} documents")
    return processed


def build_lexical_index(documents: list, index_dir: str = "indexes/whoosh"):
    """
    Build WHOOSH lexical index.
    """
    logger.info("=" * 60)
    logger.info("Building LEXICAL INDEX (WHOOSH/BM25)")
    logger.info("=" * 60)

    indexer = LexicalIndexer(index_dir=index_dir)
    indexer.create_index()

    # Prepare documents for indexing
    docs_for_index = []
    for doc in documents:
        docs_for_index.append(
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "body": doc["body"],
                "language": doc["language"],
                "source": doc["source"],
                "date": doc.get("date"),
                "entities": doc.get("entities", []),
            }
        )

    # Index in batch
    indexer.add_documents_batch(docs_for_index)

    # Show stats
    stats = indexer.get_stats()
    logger.info(f"✓ Lexical index complete: {stats['total_documents']} documents")

    return indexer


def build_semantic_index(documents: list, index_dir: str = "indexes/semantic"):
    """
    Build semantic index with embeddings.
    """

    logger.info("=" * 60)
    logger.info("BUILDING SEMANTIC INDEX (Embeddings)")
    logger.info("=" * 60)

    # Use mE5-large-instruct for best Bangla-English cross-lingual performance
    indexer = SemanticIndexer(
        model_name="intfloat/multilingual-e5-large-instruct", index_dir=index_dir
    )

    # Encode documents
    indexer.encode_documents(documents, batch_size=32)

    # Save to disk
    indexer.save_index()

    # Show stats
    stats = indexer.get_stats()
    logger.info(f"✓ Semantic index complete: {stats['total_documents']} documents")
    logger.info(f"  Embedding dimension: {stats['embedding_dimension']}")

    return indexer


def build_indexes(
    metadata_path: str,
    limit: int = 0,
    index_base_dir: str = "indexes",
    build_lexical: bool = True,
    build_semantic: bool = True,
) -> bool:
    try:
        # Resolve index directory relative to project root if not absolute
        if not os.path.isabs(index_base_dir):
            index_base_dir = os.path.join(project_root, index_base_dir)

        # Load documents
        documents = load_documents(metadata_path, limit=limit)

        if not documents:
            logger.error("No documents loaded.")
            return False

        # Initialize language detector and preprocessor
        logger.info("Initializing language detection and preprocessing...")
        lang_detector = LanguageDetector()
        preprocessor = TextPreprocessor()

        # Process documents
        processed_docs = process_documents(documents, lang_detector, preprocessor)

        # Build indexes
        if build_lexical:
            lexical_dir = os.path.join(index_base_dir, "whoosh")
            build_lexical_index(processed_docs, index_dir=lexical_dir)

        if build_semantic:
            semantic_dir = os.path.join(index_base_dir, "semantic")
            build_semantic_index(processed_docs, index_dir=semantic_dir)

        # Summary
        total_docs = len(processed_docs)
        bn_docs = sum(1 for d in processed_docs if d["language"] == "bangla")
        en_docs = sum(1 for d in processed_docs if d["language"] == "english")

        logger.info(f"Total documents indexed: {total_docs}")
        logger.info(f"  Bangla: {bn_docs}")
        logger.info(f"  English: {en_docs}")

        return True

    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        return False


def main():
    """Main indexing pipeline (CLI)."""
    parser = argparse.ArgumentParser(description="Build CLIR indexes")
    parser.add_argument(
        "--data", default="data/metadata.csv", help="Path to metadata CSV"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of documents (0 = all)"
    )
    parser.add_argument(
        "--skip-lexical", action="store_true", help="Skip lexical index building"
    )
    parser.add_argument(
        "--skip-semantic", action="store_true", help="Skip semantic index building"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CLIR INDEXING PIPELINE")
    logger.info("=" * 60)

    success = build_indexes(
        metadata_path=args.data,
        limit=args.limit,
        build_lexical=not args.skip_lexical,
        build_semantic=not args.skip_semantic,
    )

    if success:
        logger.info("=" * 60)
        logger.info("✓ INDEXING COMPLETE")
        logger.info("=" * 60)
    else:
        logger.error("Indexing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
