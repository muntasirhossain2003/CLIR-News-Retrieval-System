# CLIR News Retrieval System - Implementation Guide

## Overview

Complete implementation of Cross-Lingual Information Retrieval (CLIR) system with three main modules:

- **Module A: Indexing** - Document preprocessing, tokenization, NER, and indexing
- **Module B: Query Processing** - Language detection, translation, expansion, entity mapping
- **Module C: Retrieval** - BM25, TF-IDF, semantic search, fuzzy matching, hybrid ranking

## Project Structure

```
src/
├── module_a_indexing/
│   ├── language_detector.py      # Language detection (en/bn)
│   ├── tokenizer.py               # Multilingual tokenization
│   ├── ner_extractor.py           # Named entity extraction
│   ├── inverted_index.py          # Inverted index data structure
│   └── document_processor.py      # Complete preprocessing pipeline
│
├── module_b_query_processing/
│   ├── query_detector.py          # Query language detection
│   ├── query_normalizer.py        # Query normalization
│   ├── query_translator.py        # EN<->BN translation
│   ├── query_expander.py          # Synonym expansion
│   ├── ne_mapper.py               # Named entity mapping
│   └── query_pipeline.py          # Complete query pipeline
│
└── module_c_retrieval/
    ├── lexical_retrieval.py       # BM25 & TF-IDF
    ├── fuzzy_matcher.py           # Fuzzy string matching
    ├── semantic_retrieval.py      # Embedding-based retrieval
    ├── hybrid_ranker.py           # Score fusion
    └── retrieval_engine.py        # Main retrieval orchestrator
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download SpaCy Models

```bash
# English model (required)
python -m spacy download en_core_web_sm

# Note: Bangla spaCy model (bn_core_news_sm) is not yet available
# The system will automatically use fallback tokenization for Bangla text
```

### 3. Download NLTK Data

```bash
# Download WordNet for query expansion
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 4. Download Translation Models

```bash
# Download NLLB and OPUS-MT models for English-Bengali translation
python download_translation_models.py
```

**Note**: Translation models are large files:

- **English → Bengali**: facebook/nllb-200-distilled-600M (~2.4GB)
- **Bengali → English**: Helsinki-NLP/opus-mt-bn-en (~300MB)

First time download will take several minutes depending on your internet connection.

## Usage Examples

### Module A: Document Indexing

```python
from src.module_a_indexing import DocumentProcessor

# Initialize processor
processor = DocumentProcessor(
    data_dir='data/raw',
    output_dir='processed_data'
)

# Process all documents
stats = processor.process_all_documents()

print(f"Processed {stats['processed']} documents")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Total entities: {stats['total_entities']}")

# Output files:
# - processed_data/inverted_index.pkl
# - processed_data/document_metadata.json
# - processed_data/processing_statistics.json
```

### Module B: Query Processing

```python
from src.module_b_query_processing import QueryProcessor

# Initialize processor
query_processor = QueryProcessor(
    enable_translation=True,
    enable_expansion=True,
    enable_entity_mapping=True
)

# Process a query
result = query_processor.process_query("Sheikh Hasina education policy")

print(f"Detected language: {result['language']}")
print(f"English query: {result['en_query']}")
print(f"Bangla query: {result['bn_query']}")
print(f"Mapped entities: {result['entities']}")
```

### Module C: Retrieval

```python
from src.module_c_retrieval import RetrievalEngine
from src.module_a_indexing import InvertedIndex

# Load inverted index
inverted_index = InvertedIndex()
inverted_index.load_index('processed_data/inverted_index.pkl')

# Prepare documents
documents = []
for doc_id, metadata in inverted_index.doc_metadata.items():
    doc = {
        'doc_id': doc_id,
        'tokens': metadata.get('tokens', []),
        'metadata': metadata
    }
    documents.append(doc)

# Initialize retrieval engine
engine = RetrievalEngine(
    inverted_index=inverted_index,
    documents=documents,
    enable_semantic=True,
    enable_fuzzy=True
)

# Search
results = engine.retrieve(
    query="Sheikh Hasina education policy",
    method='hybrid',
    top_k=10
)

# Display results
for i, result in enumerate(results, 1):
    print(f"{i}. {result['title']}")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Language: {result['language']}")
    print(f"   Snippet: {result['snippet'][:100]}...")
    print()
```

### Complete CLIR Pipeline

```python
from src.module_a_indexing import DocumentProcessor, InvertedIndex
from src.module_b_query_processing import QueryProcessor
from src.module_c_retrieval import RetrievalEngine

# Step 1: Index documents (run once)
print("Step 1: Indexing documents...")
processor = DocumentProcessor('data/raw', 'processed_data')
stats = processor.process_all_documents()

# Step 2: Load index
print("\nStep 2: Loading index...")
inverted_index = InvertedIndex()
inverted_index.load_index('processed_data/inverted_index.pkl')

# Prepare documents for retrieval
documents = []
for doc_id, metadata in inverted_index.doc_metadata.items():
    documents.append({
        'doc_id': doc_id,
        'tokens': metadata.get('tokens', []),
        'metadata': metadata
    })

# Step 3: Initialize query processor and retrieval engine
print("\nStep 3: Initializing retrieval system...")
query_processor = QueryProcessor()
retrieval_engine = RetrievalEngine(
    inverted_index=inverted_index,
    documents=documents,
    enable_semantic=True
)

# Step 4: Process query
print("\nStep 4: Processing query...")
query = "Sheikh Hasina education policy"
processed = query_processor.process_query(query)

print(f"Original query: {query}")
print(f"Detected language: {processed['language']}")
print(f"English query: {processed['en_query']}")
print(f"Bangla query: {processed['bn_query']}")

# Step 5: Retrieve documents
print("\nStep 5: Retrieving documents...")
en_results = retrieval_engine.retrieve(processed['en_query'], method='hybrid', top_k=10)
bn_results = retrieval_engine.retrieve(processed['bn_query'], method='hybrid', top_k=10)

# Step 6: Merge and rank results
print("\nStep 6: Results:")
print("\n=== English Documents ===")
for i, result in enumerate(en_results[:5], 1):
    if result.get('language') == 'en':
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")

print("\n=== Bangla Documents ===")
for i, result in enumerate(bn_results[:5], 1):
    if result.get('language') == 'bn':
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
```

## Retrieval Methods

### 1. BM25 (Lexical)

```python
results = engine.retrieve(query, method='bm25', top_k=10)
```

### 2. TF-IDF (Lexical)

```python
results = engine.retrieve(query, method='tfidf', top_k=10)
```

### 3. Semantic (Embeddings)

```python
results = engine.retrieve(query, method='semantic', top_k=10)
```

### 4. Fuzzy Matching

```python
results = engine.retrieve(query, method='fuzzy', top_k=10)
```

### 5. Hybrid (Recommended)

```python
# Combines BM25 (30%), Semantic (50%), Fuzzy (20%)
results = engine.retrieve(query, method='hybrid', top_k=10)
```

## Configuration

### Custom Hybrid Weights

```python
from src.module_c_retrieval import HybridRanker

ranker = HybridRanker(
    default_weights={
        'bm25': 0.2,
        'semantic': 0.7,
        'fuzzy': 0.1
    }
)
```

### BM25 Parameters

```python
from src.module_c_retrieval import LexicalRetriever

retriever = LexicalRetriever(
    k1=1.5,  # Term frequency saturation (default: 1.5)
    b=0.75   # Length normalization (default: 0.75)
)
```

### Semantic Model Selection

```python
from src.module_c_retrieval import SemanticRetriever

# Options:
# - 'paraphrase-multilingual-mpnet-base-v2' (default, good balance)
# - 'LaBSE' (better cross-lingual)
# - 'multilingual-e5-large' (highest quality, slower)

retriever = SemanticRetriever(
    model_name='paraphrase-multilingual-mpnet-base-v2',
    cache_dir='cache/embeddings',
    use_gpu=False  # Set to True if CUDA available
)
```

## Performance Optimization

### 1. Cache Embeddings

```python
# Embeddings are automatically cached to disk
# Cache location: cache/embeddings/doc_embeddings.pkl
```

### 2. Batch Processing

```python
processor = DocumentProcessor('data/raw', 'processed_data')
# Process in batches of 100 documents
stats = processor.process_all_documents(batch_size=100)
```

### 3. GPU Acceleration

```python
# Enable GPU for semantic retrieval and translation
retriever = SemanticRetriever(use_gpu=True)
translator = QueryTranslator(use_gpu=True)
```

## Output Files

### After Indexing

- `processed_data/inverted_index.pkl` - Inverted index (pickle format)
- `processed_data/document_metadata.json` - Enhanced document metadata
- `processed_data/processing_statistics.json` - Processing statistics

### Statistics Example

```json
{
  "processing": {
    "processed": 5068,
    "errors": 0,
    "by_language": { "en": 2534, "bn": 2534 },
    "total_tokens": 1523400,
    "total_entities": 15234
  },
  "avg_tokens_per_doc": 300.55,
  "avg_entities_per_doc": 3.01
}
```

## Error Handling

All modules include comprehensive error handling:

```python
try:
    processor = DocumentProcessor('data/raw', 'processed_data')
    stats = processor.process_all_documents()
except Exception as e:
    logger.error(f"Processing failed: {e}")
```

## Logging

Configure logging level:

```python
import logging

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Testing Individual Modules

Each module file includes `__main__` section for standalone testing:

```bash
# Test language detector
python src/module_a_indexing/language_detector.py

# Test query translator
python src/module_b_query_processing/query_translator.py

# Test retrieval engine
python src/module_c_retrieval/retrieval_engine.py
```

## Next Steps

1. **Indexing**: Run document processor on your crawled data
2. **Query Processing**: Test query translation and expansion
3. **Retrieval**: Experiment with different retrieval methods
4. **Evaluation**: Implement evaluation metrics (MRR, NDCG, MAP)
5. **Web Interface**: Build a search interface using Flask/FastAPI

## Dependencies Summary

### Core

- `langdetect` - Language detection
- `spacy` - NLP processing
- `transformers` - Translation models
- `sentence-transformers` - Semantic embeddings
- `nltk` - Synonym expansion
- `numpy` - Numerical operations

### Retrieval

- `rank-bm25` - BM25 implementation
- `scikit-learn` - ML utilities

### Optional

- `torch` - PyTorch (for GPU acceleration)
- `fuzzywuzzy` - Alternative fuzzy matching

## Support

For issues or questions:

1. Check module docstrings for detailed API documentation
2. Review example usage in `__main__` sections
3. Enable DEBUG logging for troubleshooting

## License

Part of CLIR News Retrieval System
