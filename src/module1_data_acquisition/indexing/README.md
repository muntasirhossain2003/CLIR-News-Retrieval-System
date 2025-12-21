# Indexing Pipeline

This module creates a **hybrid indexing system** for the CLIR (Cross-Lingual Information Retrieval) project.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Raw Documents (JSON)                    │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│         Language Detection (fastText)                │
│         • Detect 'bn' or 'en'                        │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│      Text Preprocessing (spaCy/Stanza)              │
│      • Tokenization                                  │
│      • Named Entity Recognition                      │
│      • Text cleaning                                 │
└───────┬─────────────────────────────┬───────────────┘
        │                             │
        ▼                             ▼
┌──────────────────┐         ┌──────────────────────┐
│ LEXICAL INDEX    │         │  SEMANTIC INDEX      │
│ (WHOOSH/BM25)    │         │  (Embeddings)        │
│                  │         │                      │
│ • Inverted index │         │ • Multilingual       │
│ • Fast keyword   │         │   embeddings         │
│   search         │         │ • Cosine similarity  │
│ • Language-aware │         │ • Cross-lingual      │
└──────────────────┘         └──────────────────────┘
```

## Components

### 1. Language Detector (`language_detector.py`)

- Uses **fastText lid.176.bin** model
- Detects Bangla (bn) and English (en)
- Fallback to metadata language if detection uncertain

### 2. Text Preprocessor (`preprocessor.py`)

- **English**: spaCy `en_core_web_sm`
  - Tokenization with lemmatization
  - Stop word removal
  - Named Entity Recognition
- **Bangla**: Stanza pipeline
  - Tokenization
  - Named Entity Recognition
- Outputs: cleaned text, tokens, token count, entities

### 3. Lexical Indexer (`lexical_indexer.py`)

- **WHOOSH** inverted index
- **BM25F** ranking algorithm
- Schema fields:
  - `doc_id` (unique ID)
  - `title` (boosted 2x)
  - `body` (full text)
  - `language` (bn/en)
  - `source` (news site)
  - `date` (publication date)
  - `entities` (named entities)

### 4. Semantic Indexer (`semantic_indexer.py`)

- **sentence-transformers**: `paraphrase-multilingual-MiniLM-L12-v2`
- Generates dense embeddings (384 dimensions)
- Supports cross-lingual similarity search
- Uses cosine similarity for ranking
- Stores embeddings as NumPy arrays

## Installation

### 1. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Download NLP Models

**spaCy (English):**

```powershell
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz
```

**Stanza (Bangla):**

```powershell
python -c "import stanza; stanza.download('bn')"
```

_Note: Bangla model may have compatibility issues. The system will use fallback tokenization if unavailable._

**fastText (Language Detection):**
The script will automatically download `lid.176.bin` on first run.

## Usage

### Build Complete Index

```powershell
python -m src.module1_data_acquisition.indexing.build_index --data data/metadata.csv
```

### Options

```powershell
# Limit to first 100 documents (for testing)
python -m src.module1_data_acquisition.indexing.build_index --data data/metadata.csv --limit 100

# Skip lexical index (build only semantic)
python -m src.module1_data_acquisition.indexing.build_index --data data/metadata.csv --skip-lexical

# Skip semantic index (build only lexical)
python -m src.module1_data_acquisition.indexing.build_index --data data/metadata.csv --skip-semantic
```

### View Indexed Data

```powershell
# Quick preview in terminal (shows document summaries)
python src/module1_data_acquisition/indexing/view_indexed_data.py

# Save complete view to file (includes full text, entities, embeddings)
python src/module1_data_acquisition/indexing/view_indexed_data.py --save

# Custom output location
python src/module1_data_acquisition/indexing/view_indexed_data.py --save --output indexes/my_custom_view.txt

# Check index status (document counts and sample data)
python check_indexes.py
```

**Output File Location:** `indexes/indexed_data_view.txt`

## Output Structure

```
indexes/
├── whoosh/                    # Lexical index
│   ├── _MAIN_*.toc
│   ├── _MAIN_*.seg
│   └── ...
└── semantic/                  # Semantic index
    ├── embeddings.npy         # Document embeddings
    ├── doc_ids.json          # Document ID mapping
    └── metadata.json         # Index metadata
```

## Performance

**On ~5,000 documents:**

- Language detection: ~5 seconds
- Text preprocessing: ~10-15 minutes (depends on NLP models)
- WHOOSH indexing: ~30 seconds
- Semantic embedding: ~5-10 minutes (GPU recommended)

**Total time:** ~20-30 minutes

## Searching Indexed Documents

### Quick Search (Command Line)

```powershell
# Hybrid search (combines BM25 + embeddings) - RECOMMENDED
python src/module1_data_acquisition/indexing/search_documents.py "climate change"
python src/module1_data_acquisition/indexing/search_documents.py "জলবায়ু পরিবর্তন"

# Lexical search only (BM25 keyword matching)
python src/module1_data_acquisition/indexing/search_documents.py "mobile phone" --method lexical

# Semantic search only (meaning-based with embeddings)
python src/module1_data_acquisition/indexing/search_documents.py "global warming" --method semantic

# Filter by language
python src/module1_data_acquisition/indexing/search_documents.py "technology" --language english --limit 20

# Get more results
python src/module1_data_acquisition/indexing/search_documents.py "মোবাইল" --limit 20
```

### Programmatic Usage

**Lexical Search (BM25):**

```python
from src.module1_data_acquisition.indexing import LexicalIndexer

indexer = LexicalIndexer(index_dir="indexes/whoosh")
indexer.open_index()

results = indexer.search("climate change", language="english", limit=10)
for result in results:
    print(f"{result['title']} (score: {result['score']:.2f})")
```

**Semantic Search (Embeddings):**

```python
from src.module1_data_acquisition.indexing import SemanticIndexer

indexer = SemanticIndexer(index_dir="indexes/semantic")
indexer.load_index()

results = indexer.search("জলবায়ু পরিবর্তন", top_k=10)
for doc_id, score in results:
    print(f"{doc_id}: {score:.3f}")
```

## Troubleshooting

### Issue: "English model not found"

**Solution:**

```powershell
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz
```

### Issue: "Bangla model not found"

**Solution:**

```powershell
python -c "import stanza; stanza.download('bn')"
```

_Note: If this fails, the system will automatically use fallback tokenization._

### Issue: "fastText model not found"

**Solution:** The script will auto-download. If it fails, manually download from:
https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

Place in: `models/lid.176.bin`

## Evaluation Metrics

This indexing system supports the following IR metrics (Module D):

- **Precision@K**: Fraction of relevant docs in top-K
- **Recall@K**: Fraction of relevant docs retrieved
- **nDCG**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank

See Module 4 (Evaluation) for implementation.

## Next Steps

After building indexes:

1. **Module 3**: Query Processing & Translation
2. **Module 4**: Ranking & Evaluation
3. **Module 5**: Error Analysis & Report

## References

- WHOOSH: https://whoosh.readthedocs.io/
- Sentence Transformers: https://www.sbert.net/
- fastText: https://fasttext.cc/
- spaCy: https://spacy.io/
- Stanza: https://stanfordnlp.github.io/stanza/
