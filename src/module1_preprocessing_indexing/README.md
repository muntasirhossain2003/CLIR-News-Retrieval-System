# Module 2: Preprocessing and Indexing

This module handles text preprocessing, embedding generation, and index creation for the CLIR system.

## Files

- **`utils.py`**: Utility functions for loading articles, counting tokens, cleaning text, and filtering
- **`embedding_generator.py`**: Generate embeddings using LaBSE model and save as pickle file

## Usage

### Generate Embeddings

```bash
python main.py embed
```

### Custom Options

```bash
# Specify custom data directory
python main.py embed --data-dir data/raw

# Specify output file
python main.py embed --output data/embeddings/custom_embeddings.pkl

# Use different model
python main.py embed --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2

# Set minimum token threshold
python main.py embed --min-tokens 100

# Adjust batch size (smaller for less memory)
python main.py embed --batch-size 16
```

## What It Does

1. **Loads** all JSON articles from `data/raw/bangla` and `data/raw/english`
2. **Deduplicates** articles by URL
3. **Cleans** text (removes extra whitespace, null bytes)
4. **Counts tokens** (adds `token_count` field)
5. **Filters** articles (removes those with < 50 tokens by default)
6. **Generates embeddings** using LaBSE model (768-dimensional vectors)
7. **Saves** as pickle file with structure:

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
