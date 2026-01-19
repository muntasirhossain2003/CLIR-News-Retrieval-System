# Module C — Retrieval Models

## Overview

This module implements four retrieval models for the CLIR (Cross-Lingual Information Retrieval) system, comparing lexical, fuzzy/transliteration, semantic, and hybrid approaches.

**Target Languages:** English ↔ Bangla (system remains multilingual)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          User Query                                      │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Module B: Query Processing                            │
│                    (Language Detection, Normalization, Translation)      │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Module C: Retrieval Models                            │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │ Model 1: Lexical │  │ Model 2: Fuzzy + │  │ Model 3: Semantic    │   │
│  │                  │  │ Transliteration  │  │                      │   │
│  │  ┌────────────┐  │  │                  │  │  mE5-large-instruct  │   │
│  │  │   BM25     │  │  │  Levenshtein     │  │  + FAISS             │   │
│  │  └────────────┘  │  │  N-gram Jaccard  │  │                      │   │
│  │  ┌────────────┐  │  │  Phonetic Match  │  │  Dense Vectors       │   │
│  │  │  TF-IDF    │  │  │                  │  │  Cosine Similarity   │   │
│  │  └────────────┘  │  │  বাংলা ↔ English │  │                      │   │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘   │
│           │                     │                       │               │
│           └─────────────────────┼───────────────────────┘               │
│                                 │                                        │
│                                 ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Model 4: Hybrid Retrieval                      │   │
│  │                                                                   │   │
│  │   final_score = 0.3 × BM25 + 0.5 × Semantic + 0.2 × Fuzzy        │   │
│  │                                                                   │   │
│  │   • Weighted score fusion                                         │   │
│  │   • Confidence scoring                                            │   │
│  │   • Low-confidence warnings                                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Ranked Results with Scores [0, 1]                     │
│                    + Confidence Levels + Warnings                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
Module C — Retrieval Models/
├── __init__.py              # Module exports
├── bm25_retrieval.py        # Model 1A: BM25 lexical retrieval
├── tfidf_retrieval.py       # Model 1B: TF-IDF lexical retrieval
├── fuzzy_retrieval.py       # Model 2: Fuzzy + transliteration
├── semantic_retrieval.py    # Model 3: Semantic with mE5-large-instruct
├── hybrid_retrieval.py      # Model 4: Hybrid score fusion
├── retrieval_pipeline.py    # Gateway: Unified interface
└── README.md                # This file
```

---

## Models Overview

### Model 1: Lexical Retrieval (BM25 + TF-IDF)

| Feature         | BM25                             | TF-IDF                |
| --------------- | -------------------------------- | --------------------- |
| Library         | `rank-bm25`                      | `scikit-learn`        |
| Term Weighting  | Saturating (diminishing returns) | Linear                |
| Document Length | Normalized (b parameter)         | Normalized via TF-IDF |
| Best For        | Long queries, varied doc lengths | Short queries         |

**File:** [bm25_retrieval.py](bm25_retrieval.py), [tfidf_retrieval.py](tfidf_retrieval.py)

**Failure Cases (commented in code):**

- Synonyms ("car" vs "automobile")
- Paraphrases (different sentence structures)
- Cross-lingual queries (English ↔ বাংলা)
- Cross-script terms ("Dhaka" vs "ঢাকা")

### Model 2: Fuzzy + Transliteration Matching

**Methods:**

- Levenshtein (edit) distance
- Character n-gram Jaccard similarity
- SequenceMatcher (longest contiguous subsequence)

**Transliteration:**

- English ↔ Bangla phonetic mapping
- Common place names: "Dhaka" ↔ "ঢাকা"

**File:** [fuzzy_retrieval.py](fuzzy_retrieval.py)

**Best For:** Spelling variations, typos, cross-script names

### Model 3: Semantic Retrieval

**Uses:** FAISS (Facebook AI Similarity Search) for efficient dense vector retrieval

**Index Format:**

- Loads from existing `indexes/semantic/` directory
- Files: `embeddings.npy`, `doc_ids.json`, `metadata.json`
- FAISS index built dynamically from embeddings at load time

**FAISS Index Types:**

- `IndexFlatIP` (default): Exact search using inner product - best accuracy
- `IndexHNSWFlat`: Approximate search (HNSW) - faster for large collections

**Model:** Uses embedding model specified in `metadata.json` (e.g., `paraphrase-multilingual-MiniLM-L12-v2`)

**File:** [semantic_retrieval.py](semantic_retrieval.py)

**Best For:**

- Cross-lingual queries (English ↔ Bangla)
- Synonyms and paraphrases
- Semantic similarity when exact terms don't match

### Model 4: Hybrid Retrieval

**Score Fusion Formula:**

```
final_score = 0.3 × bm25_norm + 0.5 × semantic_norm + 0.2 × fuzzy_norm
```

**Weights Rationale:**

- Semantic (0.5): Most important for cross-lingual retrieval
- BM25 (0.3): Precise keyword matching
- Fuzzy (0.2): Cross-script and typo handling

**File:** [hybrid_retrieval.py](hybrid_retrieval.py)

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies added:**

```
# Module C: Retrieval Models
rank-bm25==0.2.2              # BM25 implementation
faiss-cpu>=1.7.4              # FAISS for dense vector search
sentence-transformers>=2.3.0  # Embedding models (already in requirements)
scikit-learn==1.3.2           # TF-IDF (already in requirements)
```

---

## Usage

### Method 1: Unified Pipeline (Recommended)

```bash
# Navigate to module
cd "src/Module C — Retrieval Models"

# Build indexes from documents
python retrieval_pipeline.py --build --data ../../data/documents.json

# Search with hybrid method (default)
python retrieval_pipeline.py "climate change"

# Search with specific method
python retrieval_pipeline.py "COVID-19 vaccine" --method semantic

# Cross-lingual search (English → Bangla documents)
python retrieval_pipeline.py "climate change" --target-lang bn

# Compare all methods
python retrieval_pipeline.py "জলবায়ু পরিবর্তন" --method all

# JSON output
python retrieval_pipeline.py "query" --json --top-k 20
```

### Method 2: Python Import

```python
from retrieval_pipeline import RetrievalPipeline

# Initialize
pipeline = RetrievalPipeline(index_dir="indexes")

# Build indexes (first time)
documents = [
    {"id": "doc1", "content": "Climate change effects on Bangladesh..."},
    {"id": "doc2", "content": "জলবায়ু পরিবর্তনের প্রভাব..."},
]
pipeline.build_indexes(documents)

# Search
results = pipeline.search(
    query="climate change",
    method="hybrid",   # or "bm25", "semantic", "fuzzy", "all"
    top_k=10
)

for r in results["results"]:
    print(f"[{r['rank']}] {r['doc_id']}: {r['score']:.4f}")
```

### Method 3: Individual Models

```python
# BM25
from bm25_retrieval import BM25Index

index = BM25Index()
index.build(documents, text_field="content")
results = index.search("climate change", top_k=10)

# Semantic (uses existing index from module1)
from semantic_retrieval import SemanticIndex

index = SemanticIndex(index_type="flat")  # or "hnsw"
index.load("indexes/semantic")
results = index.search("জলবায়ু পরিবর্তন", top_k=10)

# Fuzzy
from fuzzy_retrieval import fuzzy_match, retrieve_fuzzy

score = fuzzy_match("Dhaka", "ঢাকা")  # Transliteration matching
results = retrieve_fuzzy("climate", documents, text_field="title")

# Hybrid
from hybrid_retrieval import HybridRetriever

retriever = HybridRetriever(
    bm25_index=bm25_index,
    semantic_index=semantic_index,
    weights={"bm25": 0.3, "semantic": 0.5, "fuzzy": 0.2}
)
results = retriever.search("climate change")
```

---

## Command Line Reference

### retrieval_pipeline.py (Gateway)

```bash
python retrieval_pipeline.py [query] [options]

Options:
  --method, -m     Retrieval method: bm25, tfidf, semantic, fuzzy, hybrid, all
  --top-k, -k      Number of results (default: 10)
  --target-lang    Target language for cross-lingual: bn, en
  --index-dir      Index directory (default: indexes)
  --build          Build indexes from documents
  --data, -d       Path to documents JSON
  --compare        Compare all methods
  --json           Output as JSON
  --no-preprocess  Skip query preprocessing (Module B)
```

### Individual Model CLIs

```bash
# BM25
python bm25_retrieval.py "query" --index indexes/bm25_index.pkl

# TF-IDF
python tfidf_retrieval.py "query" --index indexes/tfidf_index.pkl

# Semantic (with index type option)
python semantic_retrieval.py "query" --index indexes/semantic --index-type flat

# Semantic (show index info)
python semantic_retrieval.py --info --index indexes/semantic

# Fuzzy (compare strings)
python fuzzy_retrieval.py --compare "Dhaka" "ঢাকা"

# Fuzzy (transliterate)
python fuzzy_retrieval.py --transliterate "Bangladesh"

# Hybrid
python hybrid_retrieval.py "query" --bm25-index indexes/bm25_index.pkl \
    --semantic-index indexes/semantic --analyze
```

---

## Output Format

### Standard Result

```json
{
  "query": {
    "original": "climate change",
    "language": "en",
    "normalized": "climate change",
    "search_query": "climate change"
  },
  "method": "hybrid",
  "results": [
    {
      "doc_id": "doc123",
      "score": 0.8542,
      "rank": 1,
      "method": "hybrid",
      "confidence": "high",
      "warnings": [],
      "scores_breakdown": {
        "bm25": 0.7234,
        "semantic": 0.9123,
        "fuzzy": 0.6543
      }
    }
  ],
  "timing": {
    "total_ms": 45.2,
    "retrieval_ms": 38.1
  }
}
```

### Confidence Levels

| Level      | Condition                                |
| ---------- | ---------------------------------------- |
| `high`     | Score > 0.3, consistent signals          |
| `medium`   | Some warnings, score 0.15-0.3            |
| `low`      | Multiple warnings or conflicting signals |
| `very_low` | Score < 0.15                             |

### Warnings

- "Very low relevance score"
- "Low relevance score"
- "Conflicting relevance signals from different methods"
- "Match based on semantic similarity only (no lexical overlap)"

---

## Index Files

After building indexes:

```
indexes/
├── bm25_index.pkl           # BM25 inverted index
├── tfidf_index.pkl          # TF-IDF vectorizer + matrix
├── whoosh/                  # Whoosh lexical index (from module1)
└── semantic/                # Semantic index directory
    ├── embeddings.npy       # Document embeddings (np.float32)
    ├── doc_ids.json         # Document ID mapping
    └── metadata.json        # Index metadata (model_name, embedding_dim, etc.)
```

**Note:** FAISS index is built dynamically from `embeddings.npy` when loading. This allows the same index files to be used across modules.

---

## Integration with Module B

Module C automatically uses Module B for query preprocessing:

```python
# This happens automatically when using the pipeline:
# 1. Language detection (bn/en)
# 2. Query normalization
# 3. Named entity extraction
# 4. Translation (if target_lang specified)

results = pipeline.search(
    "climate change",
    target_lang="bn"  # Translate to Bangla for cross-lingual search
)
```

To disable preprocessing:

```bash
python retrieval_pipeline.py "query" --no-preprocess
```

---

## Performance

| Operation       | Time (1000 docs) | Time (10000 docs) |
| --------------- | ---------------- | ----------------- |
| BM25 Build      | ~0.5s            | ~2s               |
| TF-IDF Build    | ~0.3s            | ~1.5s             |
| Semantic Build  | ~30s             | ~5 min            |
| BM25 Search     | ~5ms             | ~15ms             |
| Semantic Search | ~50ms            | ~100ms            |
| Hybrid Search   | ~60ms            | ~120ms            |

_Times vary based on document length and hardware (GPU speeds up semantic encoding)_

---

## Failure Case Analysis

### When Lexical (BM25/TF-IDF) Fails

```python
# Query: "automobile accident"
# Document: "car crash injuries"
# BM25 Score: 0.0 (no term overlap)
# Semantic Score: 0.85 (same concept)
```

### When Semantic Fails

```python
# Query: "DOC-2024-XYZ123"
# Document with exact ID
# Semantic: May not match exact codes
# BM25: Exact match found
```

### When Fuzzy Helps

```python
# Query: "Dhaka University"
# Document: "ঢাকা বিশ্ববিদ্যালয়"
# BM25: 0.0 (different scripts)
# Fuzzy + Transliteration: 0.9+ match
```

---

## API Reference

### RetrievalPipeline

```python
class RetrievalPipeline:
    def __init__(
        self,
        index_dir: str = "indexes",
        use_query_processing: bool = True,
        hybrid_weights: Dict[str, float] = None
    )

    def build_indexes(
        self,
        documents: List[Dict],
        text_field: str = "content",
        build_bm25: bool = True,
        build_tfidf: bool = True,
        build_semantic: bool = True,
        save: bool = True
    ) -> Dict[str, bool]

    def load_indexes(
        self,
        load_bm25: bool = True,
        load_tfidf: bool = True,
        load_semantic: bool = True
    ) -> Dict[str, bool]

    def search(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 10,
        target_lang: str = None
    ) -> Dict[str, Any]

    def compare_methods(
        self,
        query: str,
        top_k: int = 10
    ) -> Dict[str, Any]
```

### BM25Index

```python
class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75)
    def build(self, documents: List[Dict], text_field: str = "content")
    def search(self, query: str, top_k: int = 10) -> List[Dict]
    def get_normalized_scores(self, query: str, top_k: int = 10) -> List[Dict]
    def save(self, filepath: str)
    def load(self, filepath: str) -> bool
```

### SemanticIndex

```python
class SemanticIndex:
    def __init__(
        self,
        embedding_dim: int = 1024,
        index_type: str = "flat",  # or "hnsw"
        model_name: str = "intfloat/multilingual-e5-large-instruct"
    )
    def build(self, documents: List[Dict], text_field: str = "content")
    def search(self, query: str, top_k: int = 10) -> List[Dict]
    def save(self, directory: str)
    def load(self, directory: str) -> bool
```

### HybridRetriever

```python
class HybridRetriever:
    def __init__(
        self,
        bm25_index=None,
        semantic_index=None,
        fuzzy_matcher=None,
        weights: Dict[str, float] = None  # Default: bm25=0.3, semantic=0.5, fuzzy=0.2
    )
    def set_weights(self, bm25: float, semantic: float, fuzzy: float)
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]
    def search_with_analysis(self, query: str, top_k: int = 10) -> Dict
```

---

## Troubleshooting

### Issue: "rank-bm25 not installed"

```bash
pip install rank-bm25
```

### Issue: "FAISS not installed"

```bash
pip install faiss-cpu
# or for GPU:
pip install faiss-gpu
```

### Issue: "sentence-transformers not installed"

```bash
pip install sentence-transformers
```

### Issue: Slow semantic search

- Use HNSW index type for large collections:
  ```python
  index = SemanticIndex(index_type="hnsw")
  ```
- Enable GPU if available

### Issue: Module B not found

Ensure Module B exists at:

```
src/Module B — Query Processing & Cross-Lingual Handling/
```

Or disable preprocessing:

```bash
python retrieval_pipeline.py "query" --no-preprocess
```

---

## Academic Notes (Viva Defense)

### Why These Models?

1. **BM25 (Mandatory)**: Industry standard, baseline for comparison
2. **mE5-large-instruct (over LaBSE)**:
   - Explicitly trained for retrieval with instruction tuning
   - SOTA on MIRACL (multilingual IR benchmark)
   - Better Bangla support
3. **Fuzzy + Transliteration**: Essential for cross-script CLIR
4. **Hybrid**: Research shows fusion outperforms single methods

### Score Normalization

All scores normalized to [0, 1] for fair fusion:

- BM25: Min-max normalization per query
- Semantic: Cosine similarity → [0, 1] mapping
- Fuzzy: Already in [0, 1]

### Evaluation Metrics (Module D)

- Precision@K
- Recall@K
- nDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)

---

## Next Steps

1. **Module D**: Implement evaluation metrics
2. **Error Analysis**: Document failure cases
3. **Parameter Tuning**: Optimize fusion weights
4. **Report**: Generate CLIR performance report

---

## License

Part of CLIR News Retrieval System - Academic Project
