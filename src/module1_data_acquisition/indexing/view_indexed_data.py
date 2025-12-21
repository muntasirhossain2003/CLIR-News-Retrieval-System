"""View all indexed documents in a readable format or save to file."""

import json
import numpy as np
import argparse
from pathlib import Path
from whoosh.index import open_dir
from datetime import datetime

# Get absolute path to indexes directory
SCRIPT_DIR = Path(__file__).parent.absolute()
INDEXES_DIR = SCRIPT_DIR.parent.parent.parent / "indexes"


def view_in_terminal():
    """Display indexed documents in terminal with preview."""
    print("=" * 80)
    print("INDEXED DOCUMENTS VIEWER")
    print("=" * 80)

    try:
        idx = open_dir(str(INDEXES_DIR / "whoosh"))

        idx = open_dir(str(INDEXES_DIR / "whoosh"))

        with idx.searcher() as searcher:
            total = searcher.doc_count_all()
            print(f"\nTotal documents indexed: {total}\n")

            # Get all documents
            docs = []
            for doc in searcher.documents():
                docs.append(
                    {
                        "doc_id": doc.get("doc_id", "N/A"),
                        "title": doc.get("title", "N/A"),
                        "language": doc.get("language", "N/A"),
                        "source": doc.get("source", "N/A"),
                        "date": doc.get("date", "N/A"),
                        "body_preview": (
                            doc.get("body", "")[:200] + "..."
                            if doc.get("body")
                            else "N/A"
                        ),
                        "entities": doc.get("entities", "N/A"),
                    }
                )

            # Display all documents
            for i, doc in enumerate(docs, 1):
                print(f"{'─' * 80}")
                print(f"Document {i}/{total}")
                print(f"{'─' * 80}")
                print(f"ID:       {doc['doc_id']}")
                print(f"Title:    {doc['title']}")
                print(f"Language: {doc['language']}")
                print(f"Source:   {doc['source']}")
                print(f"Date:     {doc['date']}")
                print(f"Entities: {doc['entities']}")
                print(f"\nBody Preview:")
                print(f"{doc['body_preview']}")
                print()

        idx.close()
        print("=" * 80)

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have run the indexing pipeline first:")
        print(
            "  python -m src.module1_data_acquisition.indexing.build_index --data data/metadata.csv"
        )


def generate_file(output_file=None):
    """Generate comprehensive view file with all indexed documents."""
    if output_file is None:
        output_file = str(INDEXES_DIR / "indexed_data_view.txt")

    print("=" * 80)
    print("GENERATING INDEX VIEW FILE")
    print("=" * 80)

    try:
        # Open indexes
        idx = open_dir(str(INDEXES_DIR / "whoosh"))

        with open(str(INDEXES_DIR / "semantic" / "metadata.json")) as f:
            semantic_meta = json.load(f)

        embeddings = np.load(str(INDEXES_DIR / "semantic" / "embeddings.npy"))

        with open(str(INDEXES_DIR / "semantic" / "doc_ids.json")) as f:
            semantic_doc_ids = json.load(f)

        # Create a mapping of doc_id to embedding index
        doc_to_embedding = {doc_id: i for i, doc_id in enumerate(semantic_doc_ids)}

        # Collect all documents
        documents = []
        with idx.searcher() as searcher:
            for doc in searcher.documents():
                doc_id = doc.get("doc_id", "N/A")
                documents.append(
                    {
                        "doc_id": doc_id,
                        "title": doc.get("title", "N/A"),
                        "body": doc.get("body", "N/A"),
                        "language": doc.get("language", "N/A"),
                        "source": doc.get("source", "N/A"),
                        "date": str(doc.get("date", "N/A")),
                        "entities": doc.get("entities", "N/A"),
                        "has_embedding": doc_id in doc_to_embedding,
                        "embedding_index": doc_to_embedding.get(doc_id, -1),
                    }
                )

        idx.close()

        # Generate view file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("INDEXED DOCUMENTS - COMPLETE VIEW\n")
            f.write("=" * 100 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Documents: {len(documents)}\n")
            f.write(
                f"Semantic Embeddings: {len(semantic_doc_ids)} (dimension: {semantic_meta['embedding_dim']})\n"
            )
            f.write(f"Model: {semantic_meta['model_name']}\n")
            f.write("=" * 100 + "\n\n")

            # Write each document
            for i, doc in enumerate(documents, 1):
                f.write("─" * 100 + "\n")
                f.write(f"DOCUMENT {i}/{len(documents)}\n")
                f.write("─" * 100 + "\n")
                f.write(f"Document ID:     {doc['doc_id']}\n")
                f.write(f"Title:           {doc['title']}\n")
                f.write(f"Language:        {doc['language']}\n")
                f.write(f"Source:          {doc['source']}\n")
                f.write(f"Date:            {doc['date']}\n")
                f.write(f"Has Embedding:   {'Yes' if doc['has_embedding'] else 'No'}\n")

                if doc["has_embedding"]:
                    f.write(f"Embedding Index: {doc['embedding_index']}\n")

                f.write(f"\nNamed Entities:\n")
                entities = doc["entities"]
                if entities and entities != "N/A":
                    entity_list = entities.split(",")
                    # Show first 20 entities
                    for j, entity in enumerate(entity_list[:20], 1):
                        f.write(f"  {j}. {entity.strip()}\n")
                    if len(entity_list) > 20:
                        f.write(f"  ... and {len(entity_list) - 20} more\n")
                else:
                    f.write("  None\n")

                f.write(f"\nFull Body Text:\n")
                f.write("─" * 100 + "\n")
                body = doc["body"]
                if body and body != "N/A":
                    f.write(body + "\n")
                else:
                    f.write("No content available\n")
                f.write("─" * 100 + "\n\n")

            # Summary statistics
            f.write("\n" + "=" * 100 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 100 + "\n")

            bangla_count = sum(1 for d in documents if d["language"] == "bangla")
            english_count = sum(1 for d in documents if d["language"] == "english")

            f.write(f"Total Documents:     {len(documents)}\n")
            f.write(f"Bangla Documents:    {bangla_count}\n")
            f.write(f"English Documents:   {english_count}\n")
            f.write(
                f"Documents with Embeddings: {sum(1 for d in documents if d['has_embedding'])}\n"
            )

            # Sources breakdown
            sources = {}
            for d in documents:
                source = d["source"]
                sources[source] = sources.get(source, 0) + 1

            f.write(f"\nDocuments by Source:\n")
            for source, count in sorted(sources.items()):
                f.write(f"  {source}: {count}\n")

            f.write("\n" + "=" * 100 + "\n")
            f.write("END OF INDEX VIEW\n")
            f.write("=" * 100 + "\n")

        print(f"\n✓ Index view generated successfully!")
        print(f"\nOutput saved to: {output_file}")
        print(f"Total documents: {len(documents)}")
        print(f"  - Bangla: {bangla_count}")
        print(f"  - English: {english_count}")
        print(f"\nOpen the file to view all indexed data:")
        print(f"  notepad {output_file}")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have run the indexing pipeline first:")
        print(
            "  python -m src.module1_data_acquisition.indexing.build_index --data data/metadata.csv"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="View indexed documents in terminal or save to file"
    )
    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Save complete view to file instead of displaying in terminal",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path (default: indexes/indexed_data_view.txt)",
    )

    args = parser.parse_args()

    if args.save:
        generate_file(args.output)
    else:
        view_in_terminal()
