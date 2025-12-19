# CLIR System Architecture - Data Flow

## Complete Query Flow: "osman hadi" → Results

This document explains the complete execution flow when you search for a query.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER INPUT                              │
│              Query: "osman hadi" or "ওসমান হাদী"              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                  │
│  • Parses command-line arguments                                │
│  • --search "osman hadi" --translate --expand --method hybrid   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MODULE A: INDEXING (Pre-computed)              │
│                                                                  │
│  1. inverted_index.py                                           │
│     • Loads pre-built index: processed_data/inverted_index.pkl  │
│     • Contains: term → [doc1, doc2, ...] mappings               │
│                                                                  │
│  2. document_metadata.json                                       │
│     • Contains all document metadata (title, source, language)  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               MODULE B: QUERY PROCESSING                         │
│                                                                  │
│  Step 1: query_detector.py                                      │
│  ┌──────────────────────────────────────────┐                  │
│  │ Input:  "osman hadi"                     │                  │
│  │ Output: {language: 'en', confidence: 1.0}│                  │
│  └──────────────────────────────────────────┘                  │
│                     │                                            │
│                     ▼                                            │
│  Step 2: query_normalizer.py                                    │
│  ┌──────────────────────────────────────────┐                  │
│  │ Input:  "osman hadi"                     │                  │
│  │ Output: "osman hadi" (lowercase, trimmed)│                  │
│  └──────────────────────────────────────────┘                  │
│                     │                                            │
│                     ▼                                            │
│  Step 3: query_translator.py (if --translate enabled)           │
│  ┌──────────────────────────────────────────┐                  │
│  │ EN→BN: facebook/nllb-200-distilled-600M  │                  │
│  │ Input:  "osman hadi"                     │                  │
│  │ Output: "ওসমান হাদী"                   │                  │
│  │                                           │                  │
│  │ BN→EN: Helsinki-NLP/opus-mt-bn-en        │                  │
│  │ Input:  "ওসমান হাদী"                   │                  │
│  │ Output: "osman hadi"                     │                  │
│  └──────────────────────────────────────────┘                  │
│                     │                                            │
│                     ▼                                            │
│  Step 4: query_expander.py (if --expand enabled)                │
│  ┌──────────────────────────────────────────┐                  │
│  │ Uses WordNet for English synonyms:       │                  │
│  │ Input:  "osman hadi"                     │                  │
│  │ Output: "osman hadi" (names don't expand)│                  │
│  └──────────────────────────────────────────┘                  │
│                     │                                            │
│                     ▼                                            │
│  Step 5: ne_mapper.py (if --entities enabled)                   │
│  ┌──────────────────────────────────────────┐                  │
│  │ Maps named entities across languages:    │                  │
│  │ Input:  "osman hadi"                     │                  │
│  │ Output: Checks built-in entity dictionary│                  │
│  │         (No match found for this name)   │                  │
│  └──────────────────────────────────────────┘                  │
│                     │                                            │
│                     ▼                                            │
│  query_pipeline.py - Orchestrates all steps                     │
│  ┌──────────────────────────────────────────┐                  │
│  │ Final Output:                            │                  │
│  │ {                                         │                  │
│  │   original: "osman hadi",                │                  │
│  │   language: "en",                        │                  │
│  │   normalized: "osman hadi",              │                  │
│  │   en_query: "osman hadi",                │                  │
│  │   bn_query: "ওসমান হাদী",              │                  │
│  │   entities: [],                          │                  │
│  │   timing: {...}                          │                  │
│  │ }                                         │                  │
│  └──────────────────────────────────────────┘                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                MODULE C: RETRIEVAL ENGINE                        │
│                                                                  │
│  retrieval_engine.py - Main orchestrator                        │
│  ┌──────────────────────────────────────────┐                  │
│  │ Receives processed queries:              │                  │
│  │ • en_query: "osman hadi"                 │                  │
│  │ • bn_query: "ওসমান হাদী"               │                  │
│  │ • method: "hybrid"                       │                  │
│  └──────────────────────────────────────────┘                  │
│                     │                                            │
│            ┌────────┴────────┐                                  │
│            ▼                 ▼                                   │
│  ┌──────────────────┐ ┌──────────────────┐                     │
│  │  Search EN docs  │ │  Search BN docs  │                     │
│  │  "osman hadi"    │ │  "ওসমান হাদী"  │                     │
│  └──────────────────┘ └──────────────────┘                     │
│            │                 │                                   │
│            └────────┬────────┘                                  │
│                     ▼                                            │
│  Step 1: lexical_retrieval.py                                   │
│  ┌──────────────────────────────────────────┐                  │
│  │ BM25 Algorithm (k1=1.5, b=0.75):         │                  │
│  │ • Tokenizes: ["osman", "hadi"]           │                  │
│  │ • Looks up in inverted index:            │                  │
│  │   - "osman" → [doc_123, doc_456, ...]    │                  │
│  │   - "hadi" → [doc_789, doc_123, ...]     │                  │
│  │ • Calculates BM25 scores for each doc    │                  │
│  │ • Returns top documents with scores      │                  │
│  └──────────────────────────────────────────┘                  │
│                     │                                            │
│                     ▼                                            │
│  Step 2: semantic_retrieval.py (if --semantic enabled)          │
│  ┌──────────────────────────────────────────┐                  │
│  │ Model: paraphrase-multilingual-mpnet     │                  │
│  │ • Encodes query → 768-dim vector         │                  │
│  │ • Compares with document vectors         │                  │
│  │ • Finds semantically similar docs        │                  │
│  │   (e.g., "ওসমান হাদী" matches Bengali  │                  │
│  │    docs about this person)               │                  │
│  └──────────────────────────────────────────┘                  │
│                     │                                            │
│                     ▼                                            │
│  Step 3: fuzzy_matcher.py (if --fuzzy enabled)                  │
│  ┌──────────────────────────────────────────┐                  │
│  │ Levenshtein Distance matching:           │                  │
│  │ • Handles typos and variations:          │                  │
│  │   "osman hadi" ~ "ushman hadi" (≈0.85)   │                  │
│  │   "osman hadi" ~ "osmán hadi" (≈0.95)    │                  │
│  │ • Returns fuzzy-matched documents        │                  │
│  └──────────────────────────────────────────┘                  │
│                     │                                            │
│                     ▼                                            │
│  Step 4: hybrid_ranker.py                                       │
│  ┌──────────────────────────────────────────┐                  │
│  │ Combines all scores with weights:        │                  │
│  │ • BM25 score × 0.3                       │                  │
│  │ • Semantic score × 0.5                   │                  │
│  │ • Fuzzy score × 0.2                      │                  │
│  │                                           │                  │
│  │ Final Score = Σ(normalized_scores)       │                  │
│  │                                           │                  │
│  │ Ranks documents by combined score        │                  │
│  └──────────────────────────────────────────┘                  │
│                     │                                            │
│                     ▼                                            │
│  ┌──────────────────────────────────────────┐                  │
│  │ Returns Top K Results (default: 10)      │                  │
│  │ [                                         │                  │
│  │   {doc_id: 123, score: 0.87, title: ..}, │                  │
│  │   {doc_id: 456, score: 0.79, title: ..}, │                  │
│  │   ...                                     │                  │
│  │ ]                                         │                  │
│  └──────────────────────────────────────────┘                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       main.py (Display)                          │
│                                                                  │
│  Formats and displays results:                                  │
│  ┌──────────────────────────────────────────┐                  │
│  │ 1. Document Title (70 chars)             │                  │
│  │    Score: 0.8745 | Source: daily_star    │                  │
│  │    Lang: en                               │                  │
│  │    Snippet: "Osman Hadi announced..."    │                  │
│  │                                           │                  │
│  │ 2. Another Document Title                │                  │
│  │    Score: 0.7892 | Source: prothom_alo   │                  │
│  │    Lang: bn                               │                  │
│  │    Snippet: "ওসমান হাদী বলেছেন..."      │                  │
│  └──────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed File-by-File Flow

### 1. Entry Point: `main.py`

**Function:** `search_documents(args)`

```python
# Input: args.query = "osman hadi"
#        args.translate = True
#        args.expand = True
#        args.method = "hybrid"

# Loads index
index = InvertedIndex()
index.load_index('processed_data/inverted_index.pkl')

# Loads documents
with open('processed_data/document_metadata.json') as f:
    documents = json.load(f)

# Initializes query processor (Module B)
query_processor = QueryProcessor(
    enable_translation=True,
    enable_expansion=True,
    enable_entity_mapping=True
)

# Initializes retrieval engine (Module C)
retrieval_engine = RetrievalEngine(
    inverted_index=index,
    documents=documents,
    enable_semantic=True,
    enable_fuzzy=True
)
```

---

### 2. Module B: Query Processing Pipeline

#### File: `src/module_b_query_processing/query_pipeline.py`

**Function:** `process_query(query)`

**Orchestrates 5 steps:**

#### Step 2.1: Language Detection

**File:** `src/module_b_query_processing/query_detector.py`

```python
# Input: "osman hadi"
detector = QueryLanguageDetector()
result = detector.detect_query_language("osman hadi")

# Output:
{
    'language': 'en',
    'confidence': 1.0,
    'is_mixed': False
}

# For Bengali input "ওসমান হাদী":
{
    'language': 'bn',
    'confidence': 1.0,
    'is_mixed': False
}
```

**Technology:** Uses `langdetect` library to identify language based on character patterns.

---

#### Step 2.2: Normalization

**File:** `src/module_b_query_processing/query_normalizer.py`

```python
# Input: "  Osman HADI  "
normalizer = QueryNormalizer()
normalized = normalizer.normalize("  Osman HADI  ", language='en')

# Output: "osman hadi"
# - Converts to lowercase
# - Removes extra whitespace
# - Normalizes Unicode (NFC)
```

---

#### Step 2.3: Translation

**File:** `src/module_b_query_processing/query_translator.py`

```python
# English → Bengali
translator = QueryTranslator()
result = translator.translate_to_bangla("osman hadi")

# Uses: facebook/nllb-200-distilled-600M model
# Output:
{
    'translation': 'ওসমান হাদী',
    'source': 'en',
    'target': 'bn'
}

# Bengali → English
result = translator.translate_to_english("ওসমান হাদী")

# Uses: Helsinki-NLP/opus-mt-bn-en model
# Output:
{
    'translation': 'osman hadi',
    'source': 'bn',
    'target': 'en'
}
```

**Models Used:**

- **EN→BN**: `facebook/nllb-200-distilled-600M` (2.4GB, 600M parameters)
- **BN→EN**: `Helsinki-NLP/opus-mt-bn-en` (300MB)

**Caching:** Translations are cached in memory to avoid re-translating same queries.

---

#### Step 2.4: Query Expansion

**File:** `src/module_b_query_processing/query_expander.py`

```python
# Input: "education policy"
expander = QueryExpander(max_synonyms=2)
expanded = expander.expand("education policy", language='en')

# Uses WordNet to find synonyms
# Output: "education learning school policy regulation"

# For "osman hadi" (proper name):
# Output: "osman hadi" (no expansion for names)
```

**Technology:** Uses NLTK WordNet for English, manual dictionary for Bangla.

---

#### Step 2.5: Entity Mapping

**File:** `src/module_b_query_processing/ne_mapper.py`

```python
# Checks if entities exist in built-in dictionary
mapper = NamedEntityMapper()
entities = mapper.find_entities_in_query("Sheikh Hasina", 'en')

# Built-in entities include:
# - Sheikh Hasina → শেখ হাসিনা
# - Dhaka → ঢাকা
# - Bangladesh → বাংলাদেশ

# For "osman hadi":
# Output: [] (not in built-in dictionary)
```

**Output from Module B:**

```python
{
    'original': 'osman hadi',
    'language': 'en',
    'normalized': 'osman hadi',
    'en_query': 'osman hadi',
    'bn_query': 'ওসমান হাদী',
    'entities': [],
    'timing': {
        'detect_normalize': 0.0023,
        'translate': 0.1543,
        'expand': 0.0012,
        'entity_mapping': 0.0008,
        'total': 0.1586
    }
}
```

---

### 3. Module C: Retrieval & Ranking

#### File: `src/module_c_retrieval/retrieval_engine.py`

**Function:** `retrieve(query, method='hybrid', top_k=10)`

**Executes parallel retrieval:**

---

#### Step 3.1: BM25 Retrieval

**File:** `src/module_c_retrieval/lexical_retrieval.py`

```python
# Tokenizes query
tokens = ["osman", "hadi"]

# Looks up in inverted index
inverted_index.get_posting_list("osman")
# Returns: [
#   {doc_id: 'doc_123', frequency: 3, positions: [45, 67, 89]},
#   {doc_id: 'doc_456', frequency: 1, positions: [12]},
#   ...
# ]

inverted_index.get_posting_list("hadi")
# Returns: [
#   {doc_id: 'doc_123', frequency: 2, positions: [46, 90]},
#   {doc_id: 'doc_789', frequency: 1, positions: [5]},
#   ...
# ]

# BM25 Score Calculation:
# For each document:
score = Σ IDF(term) × (f(term) × (k1 + 1)) / (f(term) + k1 × (1 - b + b × (docLen / avgDocLen)))

# Where:
# - IDF = log((N - df + 0.5) / (df + 0.5))
# - f(term) = term frequency in document
# - k1 = 1.5 (tuning parameter)
# - b = 0.75 (length normalization)
# - N = total documents
# - df = document frequency of term

# Returns top scored documents
```

**Example BM25 Results:**

```python
[
    {'doc_id': 'doc_123', 'score': 8.45, 'title': 'Osman Hadi announces...'},
    {'doc_id': 'doc_456', 'score': 6.72, 'title': 'Interview with Hadi'},
    {'doc_id': 'doc_789', 'score': 5.23, 'title': 'Osman speaks...'}
]
```

---

#### Step 3.2: Semantic Retrieval

**File:** `src/module_c_retrieval/semantic_retrieval.py`

```python
# Uses sentence-transformers
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Encode query to 768-dimensional vector
query_embedding = model.encode("osman hadi")
# Shape: (768,)

# Encode all documents (cached)
doc_embeddings = model.encode([doc['text'] for doc in documents])
# Shape: (num_docs, 768)

# Calculate cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

# Returns top similar documents
```

**Example Semantic Results:**

```python
[
    {'doc_id': 'doc_123', 'score': 0.87, 'title': 'Osman Hadi announces...'},
    {'doc_id': 'doc_890', 'score': 0.79, 'title': 'ওসমান হাদী বলেছেন...'},
    {'doc_id': 'doc_456', 'score': 0.73, 'title': 'Interview with Hadi'}
]
```

**Why semantic is powerful:**

- Can match "ওসমান হাদী" (Bengali) with "Osman Hadi" (English)
- Understands context, not just exact words
- Multilingual embeddings work across languages

---

#### Step 3.3: Fuzzy Matching

**File:** `src/module_c_retrieval/fuzzy_matcher.py`

```python
# Handles typos and spelling variations
matcher = FuzzyMatcher(threshold=0.7)

# Levenshtein distance calculation
score = matcher.fuzzy_match("osman hadi", "ushman hadi")
# Returns: 0.82 (similar but not exact)

score = matcher.fuzzy_match("osman hadi", "osman hadi")
# Returns: 1.0 (exact match)

# Searches documents for fuzzy matches
```

**Example Fuzzy Results:**

```python
[
    {'doc_id': 'doc_123', 'score': 1.0, 'title': 'Osman Hadi...'},
    {'doc_id': 'doc_234', 'score': 0.85, 'title': 'Ushman Hadi...'},  # Typo
    {'doc_id': 'doc_345', 'score': 0.78, 'title': 'Osmán Hadi...'}   # Accent
]
```

---

#### Step 3.4: Hybrid Ranking

**File:** `src/module_c_retrieval/hybrid_ranker.py`

```python
# Combines scores from all methods
ranker = HybridRanker(weights={
    'bm25': 0.3,
    'semantic': 0.5,
    'fuzzy': 0.2
})

# Normalize each score to [0, 1]
normalized_bm25 = normalize(bm25_scores)
normalized_semantic = normalize(semantic_scores)
normalized_fuzzy = normalize(fuzzy_scores)

# Calculate weighted sum
for doc_id in all_docs:
    final_score = (
        normalized_bm25[doc_id] * 0.3 +
        normalized_semantic[doc_id] * 0.5 +
        normalized_fuzzy[doc_id] * 0.2
    )

# Sort by final_score descending
# Return top K documents
```

**Example Final Results:**

```python
[
    {
        'doc_id': 'doc_123',
        'score': 0.8745,
        'title': 'Osman Hadi announces new education policy',
        'source': 'daily_star',
        'language': 'en',
        'snippet': 'Osman Hadi, the education minister, announced...'
    },
    {
        'doc_id': 'doc_890',
        'score': 0.7892,
        'title': 'ওসমান হাদী শিক্ষা নীতি ঘোষণা করেছেন',
        'source': 'prothom_alo',
        'language': 'bn',
        'snippet': 'শিক্ষামন্ত্রী ওসমান হাদী বলেছেন...'
    },
    ...
]
```

---

## Cross-Lingual Search Example

### Query: "osman hadi" (English input)

**What happens:**

1. **Detects** language: `en`
2. **Normalizes**: `"osman hadi"`
3. **Translates**: `"osman hadi"` → `"ওসমান হাদী"`
4. **Searches BOTH**:
   - English documents with "osman hadi"
   - Bengali documents with "ওসমান হাদী"
5. **Returns** results from both languages ranked together

### Query: "ওসমান হাদী" (Bengali input)

**What happens:**

1. **Detects** language: `bn`
2. **Normalizes**: `"ওসমান হাদী"`
3. **Translates**: `"ওসমান হাদী"` → `"osman hadi"`
4. **Searches BOTH**:
   - Bengali documents with "ওসমান হাদী"
   - English documents with "osman hadi"
5. **Returns** results from both languages ranked together

---

## Why Each Module Exists

### Module A: Indexing

**When:** Runs once during setup (before searching)
**Purpose:** Pre-processes all documents for fast retrieval
**Output:**

- `inverted_index.pkl` - Fast term lookup
- `document_metadata.json` - All document info

### Module B: Query Processing

**When:** Runs every search query
**Purpose:** Transforms user query into searchable format
**Key Features:**

- Language detection
- Translation (cross-lingual)
- Synonym expansion (recall)
- Entity mapping (precision)

### Module C: Retrieval

**When:** Runs every search query
**Purpose:** Finds and ranks relevant documents
**Methods:**

- **BM25**: Exact keyword matching (precise)
- **Semantic**: Meaning-based matching (broad)
- **Fuzzy**: Typo-tolerant matching (robust)
- **Hybrid**: Combines all methods (best results)

---

## Performance Characteristics

### Query Processing Time

- Language detection: ~2-3ms
- Normalization: ~1ms
- Translation: ~150-200ms (first time), ~1ms (cached)
- Expansion: ~10-15ms
- Entity mapping: ~1ms

**Total:** ~200ms (first query), ~20ms (cached queries)

### Retrieval Time

- BM25: ~50-100ms (5000 docs)
- Semantic: ~200-500ms (5000 docs, first time)
- Fuzzy: ~100-200ms
- Hybrid ranking: ~10ms

**Total:** ~500-800ms (first query), ~200-400ms (subsequent queries)

### Memory Usage

- Inverted Index: ~50-100MB (5000 docs)
- Translation Models: ~3GB (loaded in memory)
- Semantic Model: ~500MB (loaded in memory)
- Document Metadata: ~10-20MB

**Total:** ~3.5-4GB RAM recommended

---

## Data Structures

### Inverted Index Structure

```python
{
    'osman': [
        {'doc_id': 'doc_123', 'frequency': 3, 'positions': [45, 67, 89]},
        {'doc_id': 'doc_456', 'frequency': 1, 'positions': [12]}
    ],
    'hadi': [
        {'doc_id': 'doc_123', 'frequency': 2, 'positions': [46, 90]},
        {'doc_id': 'doc_789', 'frequency': 1, 'positions': [5]}
    ]
}
```

### Document Metadata Structure

```python
{
    'doc_123': {
        'title': 'Osman Hadi announces policy',
        'url': 'http://example.com/article123',
        'source': 'daily_star',
        'language': 'en',
        'date': '2024-01-15',
        'tokens': ['osman', 'hadi', 'announces', ...],
        'entities': [
            {'text': 'Osman Hadi', 'label': 'PERSON', 'start': 0, 'end': 10}
        ]
    }
}
```

---

## Summary: Complete Data Flow

```
User Query "osman hadi"
    ↓
[main.py] Parse arguments
    ↓
[query_detector.py] Detect: en
    ↓
[query_normalizer.py] Normalize: "osman hadi"
    ↓
[query_translator.py] Translate: "ওসমান হাদী"
    ↓
[query_expander.py] Expand: "osman hadi" (no expansion)
    ↓
[query_pipeline.py] Output: {en_query, bn_query}
    ↓
[retrieval_engine.py] Orchestrate retrieval
    ├─→ [lexical_retrieval.py] BM25 scores
    ├─→ [semantic_retrieval.py] Semantic scores
    └─→ [fuzzy_matcher.py] Fuzzy scores
    ↓
[hybrid_ranker.py] Combine scores
    ↓
[main.py] Display top 10 results
    ↓
User sees results from BOTH languages
```

This architecture enables:
✓ Cross-lingual retrieval (search in one language, find in both)
✓ Robust matching (handles typos, synonyms, semantic similarity)
✓ Fast performance (pre-computed index, cached translations)
✓ Scalable (can handle thousands of documents efficiently)
