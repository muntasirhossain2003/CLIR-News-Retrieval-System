# Module C — Retrieval Models

## Overview

This module implements retrieval models for the Cross-Lingual Information Retrieval (CLIR) system. Each model represents a different approach to document retrieval, enabling comparison and evaluation of various techniques.

## Models Implemented

### Model 1A: TF-IDF Based Lexical Retrieval (`tfidf_retrieval.py`)

**Purpose:** Baseline lexical retrieval using term frequency-inverse document frequency (TF-IDF) with cosine similarity.

**Features:**

- Pure term-based matching (no embeddings)
- Same-language retrieval only (no translation)
- Normalized TF-IDF vectors with L2 normalization
- Sublinear term frequency scaling
- Returns ranked list of (doc_id, score) tuples

**Characteristics:**

- Fast and efficient
- Works well for keyword-based queries
- Sensitive to vocabulary mismatch
- No semantic understanding

---

## Installation

Ensure required dependencies are installed:

```bash
pip install scikit-learn numpy rank-bm25
```

Or install all project requirements:

```bash
pip install -r requirements.txt
```

---

## Usage

### Method 1: Direct Function Calls (TF-IDF)

```python
from tfidf_retrieval import build_tfidf_index, retrieve_tfidf

# Prepare documents (doc_id -> text)
documents = {
    'doc1': 'climate change affects global temperatures',
    'doc2': 'renewable energy reduces carbon emissions',
    'doc3': 'economic impacts of environmental policies'
}

# Build TF-IDF index
index = build_tfidf_index(documents)

# Retrieve top-K documents
query = "climate change impacts"
results = retrieve_tfidf(query, index, top_k=10)

# Process results
for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")
```

### Method 1B: Direct Function Calls (BM25)

```python
from bm25_retrieval import build_bm25_index, retrieve_bm25

# Prepare documents (same as above)
documents = {
    'doc1': 'climate change affects global temperatures',
    'doc2': 'renewable energy reduces carbon emissions',
    'doc3': 'economic impacts of environmental policies'
}

# Build BM25 index
index = build_bm25_index(documents)

# Retrieve top-K documents
query = "climate change impacts"
results = retrieve_bm25(query, index, top_k=10)

# Process results
for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")
```

### Method 3: Using Module Import

```python
from Module_C___Retrieval_Models import (
    build_tfidf_index, retrieve_tfidf,
    build_bm25_index, retrieve_bm25,
    fuzzy_match, retrieve_fuzzy
)

# Use any of the modelsissions',  # Spelling variations
    'doc3': 'economic impacts of environmental policies'
}

# Method 1: Calculate similarity between two strings
similarity = fuzzy_match('climate', 'climat', method='ngram')
print(f"Similarity: {similarity:.4f}")

# Method 2: Retrieve documents with fuzzy matching
query = "climate change"
results = retrieve_fuzzy(query, documents, top_k=10, method='ngram')
for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")

# Method 3: Per-term fuzzy matching (more precise)
results = retrieve_fuzzy_per_term(query, documents, top_k=10, min_score=0.5)
for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")
```

### Method 2: Using Module Import

```python
from Module_C___Retrieval_Models import (
    build_tfidf_index, retrieve_tfidf,
    build_bm25_index, retrieve_bm25
)

# Same as above - use either TF-IDF or BM25
```

4

### Method 3: Integration with Module B (Query Processing)

```python
import sys
sys.path.insert(0, 'src')

# Import Module B for query processing
from Module_B___Query_Processing___Cross_Lingual_Handling import process_query

# Import Module C for retrieval
from Module_C___Retrieval_Models import build_tfidf_index, retrieve_tfidf

# Step 1: Load and prepare documents
documents = load_documents()  # Your document loading function

# Step 2: Build index
index = build_tfidf_index(documents)

# Step 3: Process user query
user_query = "climate change"
query_obj = process_query(user_query)
normalized_query = query_obj['normalized_query']

# Step 4: Retrieve documents
results = retrieve_tfidf(normalized_query, index, top_k=10)
```

---

## API Reference

### `build_tfidf_index(docs: Dict[str, str]) -> TFIDFIndex`

Build TF-IDF index from document collection.

**Parameters:**

- `docs` (dict): Dictionary mapping `doc_id` → `document_text`

**Returns:**

- `TFIDFIndex`: Object containing:
  - `vectorizer`: Trained `TfidfVectorizer`
  - `doc_vectors`: TF-IDF matrix (sparse)
  - `doc_ids`: List of document IDs

**Example:**

```python
docs = {
    'doc1': 'First document text',
    'doc2': 'Second document text'
}
index = build_tfidf_index(docs)
print(f"Vocabulary size: {len(index.vectorizer.vocabulary_)}")
```

---

### `retrieve_tfidf(query: str, index: TFIDFIndex, top_k: int = 10) -> List[Tuple[str, float]]`

Retrieve top-K documents using TF-IDF cosine similarity.

**Parameters:**

- `query` (str): Normalized query string
- `index` (TFIDFIndex): Index from `build_tfidf_index()`
- `top_k` (int): Number of results to return (default: 10)

**Returns:**

- `List[Tuple[str, float]]`: List of `(doc_id, score)` tuples sorted by score descending
  - Scores are cosine similarity values in `[0, 1]`
  - Only non-zero scores included

**Example:**

```python
query = "climate change"
results = retrieve_tfidf(query, index, top_k=5)

for rank, (doc_id, score) in enumerate(results, 1):
    print(f"{rank}. {doc_id}: {score:.4f}")
```

---

Fuzzy Matching Functions

#### `fuzzy_match(query: str, doc_text: str, method: str = 'ngram', ngram_size: int = 3) -> float`

Calculate fuzzy similarity between query and document text.

**Parameters:**

- `query` (str): Query string
- `doc_text` (str): Document text to compare
- `method` (str): Matching method - `'ngram'` (character n-grams) or `'sequence'` (SequenceMatcher)
- `ngram_size` (int): Size of character n-grams (default: 3)

**Returns:**

- `float`: Similarity score in range `[0, 1]`
  - 1.0 = perfect match
  - 0.0 = no similarity

**Methods:**

- **'ngram'**: Character n-gram Jaccard similarity

  - Good for: Spelling variations, transliterations
  - Fast and efficient
  - Handles character-level overlap

- **'sequence'**: SequenceMatcher ratio (Ratcliff/Obershelp)
  - Good for: Edit distance, typos
  - More sensitive to character order
  - Slower but more accurate for similar strings

**Example:**

```python
# N-gram similarity
score1 = fuzzy_match("climate", "climat", method='ngram')
print(f"N-gram: {score1:.4f}")  # 0.7143

# Sequence matching
score2 = fuzzy_match("color", "colour", method='sequence')
print(f"Sequence: {score2:.4f}")  # 0.9091
```

---

#### `retrieve_fuzzy(query: str, docs: Dict[str, str], top_k: int = 10, method: str = 'ngram', ngram_size: int = 3, min_score: float = 0.1) -> List[Tuple[str, float]]`

Retrieve documents using fuzzy character-level matching.

**Parameters:**

- `query` (str): Query string
- `docs` (dict): Dictionary mapping `doc_id` → `document_text`
- `top_k` (int): Number of results to return (default: 10)
- `method` (str): Matching method - `'ngram'` or `'sequence'` (default: `'ngram'`)
- `ngram_size` (int): Size of n-grams (default: 3)
- `min_score` (float): Minimum similarity threshold (default: 0.1)

**Returns:**

- `List[Tuple[str, float]]`: List of `(doc_id, score)` tuples sorted by score descending

**Example:**

```python
docs = {
    'doc1': 'climate change is serious',
    'doc2': 'climat chang affects world'
}
results = retrieve_fuzzy('climate change', docs, top_k=5)
for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")
```

**Notes:**

- Computes average fuzzy match across all query terms
- Each query term matched against entire document
- Returns only documents with score ≥ min_score

---

#### `retrieve_fuzzy_per_term(query: str, docs: Dict[str, str], top_k: int = 10, method: str = 'ngram', ngram_size: int = 3, min_score: float = 0.5) -> List[Tuple[str, float]]`

Retrieve documents using per-term fuzzy matching against document terms.

**Parameters:**

- Same as `retrieve_fuzzy()` but `min_score` default is 0.5

**Returns:**

- `List[Tuple[str, float]]`: List of `(doc_id, score)` tuples

**Difference from retrieve_fuzzy():**

- Matches each query term against individual document terms
- More precise for detecting specific term variations
- Slower due to term-by-term comparison

**Example:**

```python
docs = {'doc1': 'climate change', 'doc2': 'climat chang'}
results = retrieve_fuzzy_per_term('climate', docs)
# Finds 'climate' matches with 'climat' at term level
```

---

###

### BM25 Functions

#### `build_bm25_index(docs: Dict[str, str]) -> BM25Index`

Build BM25 index from document collection.

**Parameters:**

- `docs` (dict): Dictionary mapping `doc_id` → `document_text`

**Returns:**

- `BM25Index`: Object containing:
  - `bm25`: Trained `BM25Okapi` object
  - `doc_ids`: List of document IDs
  - `tokenized_corpus`: List of tokenized documents

**Example:**

```python
docs = {
    'doc1': 'First document text',
    'doc2': 'Second document text'
}
index = build_bm25_index(docs)
print(f"Average document length: {index.bm25.avgdl:.2f} tokens")
```

---

#### `retrieve_bm25(query: str, index: BM25Index, top_k: int = 10) -> List[Tuple[str, float]]`

Retrieve top-K documents using BM25 scoring.

**Parameters:**

- `query` (str): Normalized query string
- `index` (BM25Index): Index from `build_bm25_index()`
- `top_k` (int): Number of results to return (default: 10)

**Returns:**

- `List[Tuple[str, float]]`: List of `(doc_id, score)` tuples sorted by score descending
  - Scores are BM25 scores (unbounded, typically 0-100)
  - Only non-zero scores included

**Example:**

```python
query = "climate change"
results = retrieve_bm25(query, index, top_k=5)

for rank, (doc_id, score) in enumerate(results, 1):
    print(f"{rank}. {doc_id}: {score:.4f}")
```

---

### Why Fuzzy Matching for Failure Analysis?

1. **Demonstrates Limitations:** Shows why lexical retrieval fails cross-lingually
2. **Handles Variations:** Good for typos and spelling differences
3. **Interpretable:** Character-level overlap is easy to understand
4. **No Dependencies:** Uses only standard library (difflib)

**Key Insight:** Fuzzy matching works for spelling variations but fails for:

- Synonyms (no semantic understanding)
- Translation (no cross-lingual knowledge)
- Context (no word meaning awareness)

This demonstrates the need for semantic models (Model 3+).

### `TFIDFIndex` Class

Container for TF-IDF index components.

**Attributes:**

- `vectorizer` (TfidfVectorizer): Trained vectorizer with vocabulary
- `doc_vectors` (sparse matrix): TF-IDF vectors for all documents
- `doc_ids` (list): Document IDs in same order as matrix rows

---

### `BM25Index` Class

Container for BM25 index components.

**Attributes:**

- `bm25` (BM25Okapi): Trained BM25 model
- `doc_ids` (list): Document IDs in same order as corpus
- `tokenized_corpus` (list): List of tokenized documents

---

## BM25 Configuration

BM25Okapi uses default parameters:

```python
BM25Okapi(
    corpus,              # Tokenized documents
    k1=1.5,             # Term frequency saturation parameter
    b=0.75,             # Document length normalization
    epsilon=0.25        # IDF floor value
)
```

**Parameters Explained:**

- **k1 (1.5):** Controls how quickly term frequency saturates
  - Higher k1 = more weight to term frequency
  - Lower k1 = earlier saturation
  - Range: 1.2 - 2.0
- **b (0.75):** Controls document length normalization
  - b=1: Full normalization by document length
  - b=0: No length normalization
  - b=0.75: Standard balanced setting
- **epsilon:** Floor value for IDF (prevents negative IDF)

**Tokenization:**

- Lowercase normalization
- Whitespace splitting (no stemming)
- Preserves all tokens for explaina

### When to Use Fuzzy Matching

- ✅ Spelling variations in queries
- ✅ Handling typos
- ✅ Transliterated names (limited)
- ✅ Small document collections
- ✅ Demonstrating lexical limitations
- ⚠️ As a fallback after lexical methods failbility

---

## Performance Characteristics

### TF-IDF

| Aspect           | Performance                                     |
| ---------------- | ----------------------------------------------- |
| Index Build Time | O(N × V) where N = docs, V = vocab              |
| Query Time       | O(V + N) linear in collection size              |
| Memory           | Sparse matrix (efficient for large collections) |
| Scalability      | Good up to 100K documents                       |

### BM25

| Aspect           | Performance                        |
| ---------------- | ---------------------------------- |
| Index Build Time | O(N × V) where N = docs, V = vocab |
| Query Time       | O(V × N) needs full corpus scan    |
| Memory           | Tokenized corpus in memory         |
| Scalability      | Good up to 50K documents           |

**Note:** BM25 is slightly slower for large collections but often more effective.

---

## Design Philosophy

### Why TF-IDF as Baseline?

1. **Established Benchmark:** Standard baseline in IR research
2. **Interpretable:** Scores reflect term overlap
3. **No Training Required:** Works out-of-the-box
4. **Fast:** Efficient for real-time retrieval

### Why BM25 as Improved Baseline?

1. **Better Ranking:** Probabilistic foundation
2. **Saturation:** Handles repeated terms gracefully
3. **Length Normalization:** Better for mixed-length documents
4. **SOTA Lexical:** State-of-the-art among lexical methods

---

## TF-IDF Technical Details

The vectorizer uses the following settings:

```python
TfidfVectorizer(
    lowercase=True,           # Convert to lowercase
    strip_accents=None,       # Preserve language-specific characters
    max_features=None,        # No feature limit (baseline)
    sublinear_tf=True,        # Log scaling: 1 + log(tf)
    use_idf=True,             # Apply IDF weighting
    smooth_idf=True,          # Add 1 to document frequencies
    norm='l2'                 # L2 normalization (default)
)
```

**Rationale:**

- **Sublinear TF:** Reduces impact of high-frequency terms
- **Smooth IDF:** Prevents division by zero
- **L2 Norm:** Ensures fair comparison regardless of document length
- **No max_features:** Preserves all terms for baseline evaluation

---

## Performance Characteristics

| Aspect           | Performance                                     |
| ---------------- | ----------------------------------------------- |
| Index Build Time | O(N × V) where N = docs, V = vocab              |
| Query Time       | O(V + N) linear in collection size              |
| Memory           | Sparse matrix (efficient for large collections) |
| Scalability      | Good up to 100K documents                       |

---

## Design Philosophy

### Why TF-IDF as Baseline?

1. **Established Benchmark:** Standard baseline in IR research
2. **Interpretable:** Scores reflect term overlap
3. **No Training Required:** Works out-of-the-box
4. **Fast:** Efficient for real-time retrieval

### Limitations of Lexical Models (Both TF-IDF and BM25)

1. **Vocabulary Mismatch:** Fails on synonyms (e.g., "car" vs "automobile")
2. **No Semantics:** Cannot understand meaning
3. **Monolingual Only:** Requires query and documents in same language
4. **Word Order:** Bag-of-words ignores phrase structure

### When to Use TF-IDF

- ✅ Keyword-based queries
- ✅ Same-language retrieval
- ✅ Baseline comparison
- ✅ Fast prototyping
- ✅ Very large collections (sparse matrix efficient)

### When to Use BM25

- ✅ Better ranking quality needed
- ✅ Mixed document lengths
- ✅ Repeated query terms
- ✅ Standard IR baseline
- ✅ Collections up to 50K documents

### When NOT to Use Lexical Models

- ❌ Cross-lingual retrieval (use Model 3)
- ❌ Semantic queries (use Model 2)
- ❌ Synonym-rich domains
- ❌ Context-dependent queries

---

## Model Comparison

### TF-IDF vs BM25

| Aspect              | TF-IDF             | BM25              |
| ------------------- | ------------------ | ----------------- |
| **Ranking Quality** | Good               | Better            |
| **Speed**           | Faster             | Slightly slower   |
| **Memory**          | Sparse (efficient) | Tokenized corpus  |
| **Term Saturation** | Linear             | Saturates         |
| **Length Norm**     | L2 norm            | Parameterized (b) |
| **Tuning**          | Fixed              | Tunable (k1, b)   |
| **Best For**        | Large collections  | Mixed-length docs |

**Recommendation:** Start with BM25 for better ranking, use TF-IDF if speed/memory critical.

---

## Testing

### Run TF-IDF Demo

```bash
cd "src\Module C — Retrieval Models"
python tfidf_retrieval.py
```

**Expected Output:**

```
======================================================================
Module C - Model 1A: TF-IDF Retrieval Demo
======================================================================

1. Building TF-IDF index...
   Indexed 5 documents
   Vocabulary size: 34

2. Testing retrieval...
   Query: 'climate change impacts'
     1. doc1: 0.3186
     2. doc3: 0.3175
     3. doc4: 0.2491
...
```

### Run BM25 Demo

```bash
cd "src\Module C — Retrieval Models"
python bm25_retrieval.py
```

**Expected Output:**

```
======================================================================
Module C - Model 1B: BM25 Retrieval Demo
======================================================================

======================================================================

1. Building BM25 index...
   Indexed 5 documents
   Average document length: 8.40 tokens

2. Testing retrieval...
   Query: 'climate change impacts'
     1. doc1: 2.1234
     2. doc3: 1.8765
     3. doc2: 0.9432

3. Testing BM25 term frequency saturation...
   Query with 1 occurrence: 'climate'
     doc1: 1.2345
   Query with 3 occurrences: 'climate climate climate'
     doc1: 1.6789
   (Note: Scores increase but saturate - BM25 feature)
...
```

### Integration Test

```python
# Test both models
import sys
sys.path.insert(0, 'src')

from tfidf_retrieval import build_tfidf_index, retrieve_tfidf
from bm25_retrieval import build_bm25_index, retrieve_bm25

# Sample documents
docs = {
    'doc1': 'climate change',
    'doc2': 'global warming'
}

# Test TF-IDF
tfidf_index = build_tfidf_index(docs)
tfidf_results = retrieve_tfidf("climate", tfidf_index, top_k=5)
assert len(tfidf_results) > 0, "TF-IDF should retrieve documents"

# Test BM25
bm25_index = build_bm25_index(docs)
bm25_results = retrieve_bm25("climate", bm25_index, top_k=5)
assert len(bm25_results) > 0, "BM25 should retrieve documents"

print("✅ Both models passed integration test")
```

---

## Troubleshooting

### Issue: Empty Results

**Problem:** `retrieve_tfidf()` returns empty list

**Causes:**

1. Query has no terms in vocabulary
2. All similarity scores are zero

**Solutions:**

- Check query normalization (Module B)
- Verify documents are indexed correctly
- Try broader queries with common terms

### Issue: Low Scores

**Problem:** All cosine similarity scores < 0.1

**Explanation:** Normal behavior when:

- Query terms are rare in documents
- Documents are long (dilutes term weights)
- Limited term overlap

**Not a bug:** TF-IDF is sensitive to vocabulary match

### Issue: Unexpected Rankings

**Problem:** Expected document not in top results

**Reasons:**

1. **Vocabulary mismatch:** Query uses different terms (synonyms)
2. **IDF penalty:** Query terms are very common
3. **Document length:** Longer docs get normalized

**Solution:** Consider semantic models (Model 2) for better recall

---

## Future Enhancements

### Planned Models

- **Model 1B:** BM25 retrieval (Okapi BM25 with tunable parameters)
- **Model 2:** Semantic retrieval using sentence embeddings
- **Model 3:** Hybrid retrieval combining lexical + semantic signals

### Potential Improvements

1. **Query expansion:** Add synonyms to improve recall
2. **Custom tokenization:** Language-specific tokenizers
3. **Field weighting:** Boost title/headline matches
4. **Caching:** Store frequent query results

---

## Academic Notes

### TF-IDF Formula

**Term Frequency (sublinear):**
$$\text{tf}(t, d) = 1 + \log(\text{count}(t, d))$$

**Inverse Document Frequency (smooth):**
$$\text{idf}(t) = \log\left(\frac{N + 1}{\text{df}(t) + 1}\right) + 1$$

**TF-IDF Weight:**
$$\text{tfidf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)$$

**Cosine Similarity:**
$$\text{sim}(q, d) = \frac{q \cdot d}{\|q\| \|d\|} = \frac{\sum_{t} q_t \times d_t}{\sqrt{\sum q_t^2} \times \sqrt{\sum d_t^2}}$$

### Citation

For academic use, cite the original TF-IDF paper:

> Salton, G., & Buckley, C. (1988). _Term-weighting approaches in automatic text retrieval_. Information Processing & Management, 24(5), 513-523.

---

## Terminal Commands

```powershell
# Navigate to module
cd "src\Module C — Retrieval Models"

# Run demo
python tfidf_retrieval.py

# Test import
python -c "from tfidf_retrieval import build_tfidf_index, retrieve_tfidf; print('✅ Import successful')"

# Check dependencies
python -c "import sklearn; import numpy; print('✅ All dependencies installed')"
```

---

## Integration Examples

### Example 1: Compare TF-IDF vs BM25

```python
from tfidf_retrieval import build_tfidf_index, retrieve_tfidf
from bm25_retrieval import build_bm25_index, retrieve_bm25

# Load documents (example)
def load_documents():
    return {
        'doc1': 'Document 1 content...',
        'doc2': 'Document 2 content...',
        # ... more documents
    }

# Build both indexes
documents = load_documents()
tfidf_index = build_tfidf_index(documents)
bm25_index = build_bm25_index(documents)

# Compare retrieval for same query
query = "sample query"
tfidf_results = retrieve_tfidf(query, tfidf_index, top_k=10)
bm25_results = retrieve_bm25(query, bm25_index, top_k=10)

print("TF-IDF Top 3:")
for doc_id, score in tfidf_results[:3]:
    print(f"  {doc_id}: {score:.4f}")

print("\nBM25 Top 3:")
for doc_id, score in bm25_results[:3]:
    print(f"  {doc_id}: {score:.4f}")
```

# Retrieve for multiple queries

queries = ["query 1", "query 2", "query 3"]
for query in queries:
results = retrieve_tfidf(query, index, top_k=10)
print(f"Query: {query} -> {len(results)} results")

````

### Example 2: With Query Processing (Module B)

```python
import sys
sys.path.insert(0, 'src')

from tfidf_retrieval import build_tfidf_index, retrieve_tfidf

# Module B import (adjust path as needed)
sys.path.insert(0, r'src\Module B — Query Processing & Cross-Lingual Handling')
from language_detection_normalization import process_query
: TF-IDF
├── bm25_retrieval.py           # Model 1B: BM25
├── fuzzy_retrieval.py          # Model 2: Fuzzy matching
documents = load_documents()
index = build_tfidf_index(documents)

# Process and retrieve
user_query = "Climate Change"
query_result = process_query(user_query)
normalized_query = query_result['normalized_query']
retrieved_docs = retrieve_tfidf(normalized_query, index, top_k=10)

print(f"Original: {user_query}")
print(f"Normalized: {normalized_query}")
print(f"Retrieved: {len(retrieved_docs)} documents")
````

---

## File Structure

```
Module C — Retrieval Models/
├── __init__.py                 # Module exports
├── tfidf_retrieval.py          # Model 1A implementation
├── bm25_retrieval.py           # Model 1B implementation
└── README.md                   # This file
```

---

## Contact & Contribution

For questions or improvements, refer to the main project README or contact the development team.

---

**Status:**

- ✅ Model 1A (TF-IDF) - Complete
- ✅ Model 1B (BM25) - Complete
- ⏳ Model 2 (Semantic Retrieval) - Planned
- ⏳ Model 3 (Hybrid Retrieval) - Planned

**Next:** Model 2 (Semantic Retrieval with Embeddings)
