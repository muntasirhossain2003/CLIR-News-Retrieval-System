# Module 3: Retrieval

## Purpose

Retrieve relevant documents using lexical (BM25), semantic (LaBSE), and fuzzy (transliteration) methods.

## Architecture

```
module3_retrieval/
└── retriever.py                # Retriever class with 3 search methods
```

## Component

### Retriever

- **File**: `retriever.py`
- **Class**: `Retriever`
- **Methods**:
  - `search()` - Main search entry point
  - `_whoosh_search()` - Lexical search (BM25)
  - `_faiss_search()` - Semantic search (LaBSE embeddings)
  - `_fuzzy_search()` - Fuzzy matching with transliteration

## Search Methods

### 1. Lexical Search (BM25)

- **Method**: `_whoosh_search()`
- **Index**: Whoosh
- **Algorithm**: BM25
- **Strategy**:
  - Phrase matching (exact match, higher boost)
  - OR keyword search (any term matches)
  - Title boosted 5x over body
  - Searches both original and translated query

### 2. Semantic Search (Vector)

- **Method**: `_faiss_search()`
- **Index**: FAISS
- **Algorithm**: Dense vector similarity (cosine)
- **Model**: LaBSE (768-dim embeddings)
- **Strategy**:
  - ✅ **CRITICAL**: Uses ORIGINAL query directly (no translation)
  - LaBSE is trained for cross-lingual matching - translation introduces drift
  - Encode query to vector
  - Normalize L2
  - Find k-nearest neighbors
- **Why No Translation?**: LaBSE maps Bangla and English to same vector space without translation

### 3. Fuzzy Search (Transliteration)

- **Method**: `_fuzzy_search()`
- **Algorithm**: Partial ratio matching
- **Optimization**: ✅ **TITLE-ONLY** matching (no body text scan)
- **Scalability**: O(n) but ~5x faster, scales to 50k+ docs
- **Features**:
  - Named entity transliteration map
  - Character-level fuzzy matching (70% weight)
  - Token-based matching - word order invariant (30% weight)
  - Threshold: 70/100 (configurable)
- **Why Title-Only?**: Scanning full body creates noise (false positives) and doesn't scale

#### Transliteration Map

```python
{
    'bangladesh': 'বাংলাদেশ',
    'dhaka': 'ঢাকা',
    'chittagong': 'চট্টগ্রাম',
    'sylhet': 'সিলেট',
    'rajshahi': 'রাজশাহী',
    'khulna': 'খুলনা',
    'barisal': 'বরিশাল',
    'rangpur': 'রংপুর',
    'mymensingh': 'ময়মনসিংহ'
}
```

## Data Flow

```
Query
  ↓
QueryProcessor (Module 2)
  ↓
Retriever.search()
  ├─→ _whoosh_search()    → Lexical Results
  ├─→ _faiss_search()     → Semantic Results
  └─→ _fuzzy_search()     → Fuzzy Results
  ↓
Return {
    'lexical': [...],
    'semantic': [...],
    'fuzzy': [...],
    'timing': {...}
}
```

## Timing Breakdown

Each search tracks:

- Query processing time
- Lexical search time
- Semantic search time
- Fuzzy search time
- Total time

## Result Format

```python
{
    'lexical': [
        {'title': '...', 'url': '...', 'score': 0.85, 'lang': 'bangla'},
        ...
    ],
    'semantic': [
        {'title': '...', 'url': '...', 'score': 0.92, 'lang': 'english'},
        ...
    ],
    'fuzzy': [
        {'title': '...', 'url': '...', 'score': 0.78, 'lang': 'bangla'},
        ...
    ],
    'timing': {
        'query_processing': 0.023,
        'lexical_search': 0.045,
        'semantic_search': 0.012,
        'fuzzy_search': 0.156,
        'total': 0.236
    }
}
```

## Usage

### Basic Usage

```python
from src.module3_retrieval import Retriever

# Initialize
retriever = Retriever()

# Search
results = retriever.search("Dhaka air pollution", k=10)

# Access results
lexical_results = results['lexical']
semantic_results = results['semantic']
fuzzy_results = results['fuzzy']
timing = results['timing']
```

### Custom Configuration

```python
retriever = Retriever(
    data_path="data/embeddings/articles_with_embeddings.pkl",
    whoosh_path="data/indices/whoosh",
    faiss_path="data/indices/faiss_index.bin",
    model_name="sentence-transformers/LaBSE",
    fuzzy_threshold=70
)
```

## Integration

Used by web app and evaluation script:

```python
# In app.py
search_results = retriever.search(query, k=10)
ranked_results = ranker.merge_and_rank(
    lexical_results=search_results['lexical'],
    semantic_results=search_results['semantic'],
    fuzzy_results=search_results['fuzzy'],
    alpha=0.5
)
```

## Dependencies

```
whoosh==2.7.4
faiss-cpu==1.7.4
sentence-transformers==2.2.2
fuzzywuzzy==0.18.0
python-Levenshtein==0.23.0
```

## Performance

- Lexical search: ~40-50ms
- Semantic search: ~10-15ms
- Fuzzy search: ~150-200ms (slowest, scans all docs)
- Total: ~200-300ms per query

## Strengths & Weaknesses

| Method   | Strengths                           | Weaknesses                          |
| -------- | ----------------------------------- | ----------------------------------- |
| Lexical  | Fast, exact matches, term frequency | Misses synonyms, translation issues |
| Semantic | Captures meaning, cross-lingual     | Slower, requires embeddings         |
| Fuzzy    | Handles typos, transliteration      | Slow, noisy results                 |
