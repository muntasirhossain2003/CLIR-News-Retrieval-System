# Module 1.2: Preprocessing & Indexing

## Purpose

Transforms raw news articles into searchable indices using both lexical (BM25) and semantic (LaBSE) approaches for cross-lingual information retrieval.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Preprocessing & Indexing                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  metadata.csv (5,194 docs)                                      │
│         │                                                       │
│         ├──► Whoosh Indexer ──► BM25 Index (data/indices/whoosh/)│
│         │    (Schema: url, title, content, lang, date)         │
│         │                                                       │
│         └──► Embedding Generator ──► FAISS Index                │
│              (LaBSE 768-dim)         (data/indices/faiss/)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. **indexer.py**

- **WhooshIndexer**: Creates BM25 lexical index
  - Schema: url (ID), title (TEXT), content (TEXT), lang (TEXT), date (TEXT)
  - Analyzer: StandardAnalyzer with stopword filtering
  - Storage: data/indices/whoosh/
- **FAISSIndexer**: Builds semantic vector index
  - Model: LaBSE (sentence-transformers/LaBSE)
  - Dimensions: 768
  - Index Type: FAISS FlatL2 (exact L2 distance)
  - Storage: data/indices/faiss_index.bin + url_mapping.pkl

### 2. **embedding_generator.py**

- Generates embeddings from article content
- Batch processing with progress tracking
- Outputs: embeddings.pkl (list of 768-dim vectors)

### 3. **utils.py**

- `load_articles_from_metadata()`: Load articles from metadata.csv
- `count_tokens()`: Count words in text
- `clean_text()`: Basic text cleaning
- `filter_articles()`: Filter by language/date/keywords

## Data Flow

```
metadata.csv
    ↓
load_articles_from_metadata()
    ↓
┌───────────────┴──────────────┐
│                              │
WhooshIndexer.index()    Embedding Generator
    ↓                          ↓
BM25 Index              embeddings.pkl
(5,194 docs)                 ↓
                      FAISSIndexer.index()
                             ↓
                       FAISS Index
                      (5,194 vectors)
```

## Usage

### Build Both Indices

````bash
cd c:\Users\User\Videos\clir-project
python main.py index
```bash
cd c:\Users\User\Videos\clir-project
python main.py index
````

**Output:**

```
Loading articles from metadata...
Loaded 5194 articles from metadata

Building Whoosh index...
Indexed 5194 documents in Whoosh
Whoosh index created at: data\indices\whoosh

Building FAISS index...
Loading embeddings from data\embeddings\embeddings.pkl...
Loaded 5194 embeddings
FAISS index created at: data\indices\faiss_index.bin
URL mapping saved at: data\indices\url_mapping.pkl
```

### Generate Embeddings Only

```bash
python main.py embed
```

**Output:**

```
Generating embeddings...
Processing articles: 100%|████████████| 5194/5194
Embeddings saved to data\embeddings\embeddings.pkl
```

## Output Structure

### Whoosh Index

```
data/indices/whoosh/
├── _MAIN_1.toc          # Table of contents
├── MAIN_xxxxx.seg       # Segment files
└── MAIN_WRITELOCK       # Lock file
```

**Schema Fields:**

- `url` (ID): Unique document identifier
- `title` (TEXT): Article headline
- `content` (TEXT): Full article text
- `lang` (TEXT): Language code (bn/en)
- `date` (TEXT): Publication date

### FAISS Index

```
data/indices/
├── faiss_index.bin      # 5194 x 768 float32 vectors
└── url_mapping.pkl      # [url1, url2, ..., url5194]
```

## Dependencies

```
whoosh==2.7.4
faiss-cpu==1.7.4
sentence-transformers==2.2.2
transformers==4.36.2
torch==2.1.2
pandas==2.1.4
```

## Performance

- **Whoosh Indexing**: ~10-15 seconds for 5,194 docs
- **Embedding Generation**: ~3-5 minutes on CPU (batch_size=32)
- **FAISS Index Build**: ~2-3 seconds
- **Disk Space**:
  - Whoosh: ~15 MB
  - FAISS: ~16 MB (5194 × 768 × 4 bytes)
  - Embeddings: ~16 MB

```python
[
    {
        "url": "https://...",
        "title": "Article Title",
        "body": "Article content...",
        "language": "bangla",
        "source": "prothom_alo",
        "date": "2025-12-18",
        "crawled_at": "2025-12-18 06:37:25",
        "filepath": "data/raw/bangla/...",
        "token_count": 523,
        "embedding": np.array([0.23, -0.45, ...])  # 768 dimensions
    },
    ...
]
```

## Why LaBSE?

LaBSE (Language-agnostic BERT Sentence Embedding) is specifically designed for cross-lingual tasks:

- Trained on 109 languages including Bangla and English
- Maps semantically similar sentences to nearby points regardless of language
- Excellent for CLIR (Cross-Lingual Information Retrieval)
- 768-dimensional embeddings

## Performance

- **Processing time**: ~2-3 hours for 5,600+ articles
- **File size**: ~100-150MB pickle file
- **Memory**: ~4-6GB RAM during processing
