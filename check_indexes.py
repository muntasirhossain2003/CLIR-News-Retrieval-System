"""Check the status of indexed documents."""

import json
import numpy as np
from whoosh.index import open_dir

print("=" * 60)
print("INDEX STATUS CHECK")
print("=" * 60)

# Check WHOOSH lexical index
try:
    idx = open_dir("indexes/whoosh")
    num_docs = idx.doc_count_all()
    print(f"\n✓ LEXICAL INDEX (WHOOSH/BM25)")
    print(f"  Documents indexed: {num_docs}")

    # Get sample documents
    with idx.searcher() as searcher:
        # Get first 5 documents
        for i, doc in enumerate(searcher.documents()):
            if i >= 5:
                break
            print(f"\n  Sample {i+1}:")
            print(f"    ID: {doc.get('doc_id', 'N/A')}")
            print(f"    Title: {doc.get('title', 'N/A')[:60]}...")
            print(f"    Language: {doc.get('language', 'N/A')}")
            print(f"    Source: {doc.get('source', 'N/A')}")

    idx.close()
except Exception as e:
    print(f"\n✗ LEXICAL INDEX ERROR: {e}")

# Check semantic index
try:
    with open("indexes/semantic/metadata.json") as f:
        metadata = json.load(f)

    embeddings = np.load("indexes/semantic/embeddings.npy")
    with open("indexes/semantic/doc_ids.json") as f:
        doc_ids = json.load(f)

    print(f"\n✓ SEMANTIC INDEX (Embeddings)")
    print(f"  Documents indexed: {metadata['num_documents']}")
    print(f"  Embedding dimension: {metadata['embedding_dim']}")
    print(f"  Model: {metadata['model_name']}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"\n  Sample document IDs:")
    for i, doc_id in enumerate(doc_ids[:5]):
        print(f"    {i+1}. {doc_id}")

except Exception as e:
    print(f"\n✗ SEMANTIC INDEX ERROR: {e}")

print("\n" + "=" * 60)
print(f"NOTE: Currently only 10 documents are indexed (test run)")
print(f"To index all 5,170 documents, run:")
print(
    f"  python -m src.module1_data_acquisition.indexing.build_index --data data/metadata.csv"
)
print("=" * 60)
