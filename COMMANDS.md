# CLIR System - Command Reference

This document contains all available commands for the CLIR News Retrieval System.

---

## 🕷️ Module 1: Data Acquisition (Crawling)

### Crawl All Sources

```powershell
python main.py
```

### Crawl with Language Filter

```powershell
# Crawl only Bangla sources
python main.py --lang bangla

# Crawl only English sources
python main.py --lang english

# Crawl all sources (default)
python main.py --lang all
```

### Crawl with Limit

```powershell
# Crawl 50 articles per source (default)
python main.py --limit 50

# Crawl 10 articles per source
python main.py --limit 10

# Crawl 100 articles per source
python main.py --limit 100
```

### Crawl Specific Source

```powershell
# Crawl only Prothom Alo
python main.py --source prothom_alo --limit 20

# Crawl only Daily Star
python main.py --source daily_star --limit 15

# Crawl only Dhaka Tribune
python main.py --source dhaka_tribune --limit 30
```

### Combined Crawler Commands

```powershell
# Crawl only Bangla sources, 20 articles each
python main.py --lang bangla --limit 20

# Crawl only English sources, 10 articles each
python main.py --lang english --limit 10

# Crawl specific source with limit
python main.py --source prothom_alo --limit 20 --lang bangla
```

---

## 📑 Module A: Indexing

### Index All Documents

```powershell
# Index documents from default directory (data/raw)
python main.py --index
```

### Index with Custom Directories

```powershell
# Specify input and output directories
python main.py --index --input-dir data/raw --output-dir processed_data

# Index from custom directory
python main.py --index --input-dir my_data --output-dir my_index
```

### Index with Batch Size

```powershell
# Process 50 documents at a time
python main.py --index --batch-size 50

# Process 200 documents at a time
python main.py --index --batch-size 200
```

### Complete Indexing Example

```powershell
python main.py --index --input-dir data/raw --output-dir processed_data --batch-size 100
```

---

## 🔍 Module B+C: Search & Retrieval

### Basic Search

```powershell
# Simple search with default settings (hybrid method)
python main.py --search "Sheikh Hasina education"

# Search for education policy
python main.py --search "education policy"

# Search for Bangladesh economy
python main.py --search "Bangladesh economy"
```

### Search with Different Methods

```powershell
# BM25 retrievalpython main.py --search "education policy" --method hybrid --translate
python main.py --search "education policy" --method bm25

# TF-IDF retrieval
python main.py --search "education policy" --method tfidf

# Semantic retrieval
python main.py --search "education policy" --method semantic --semantic

# Fuzzy matching
python main.py --search "education policy" --method fuzzy --fuzzy

# Hybrid retrieval (recommended)
python main.py --search "education policy" --method hybrid
```

### Search with Translation

```powershell
# Enable query translation (EN ↔ BN)
python main.py --search "Sheikh Hasina" --translate

# Search with translation and hybrid method

```

### Search with Query Expansion

```powershell
# Enable synonym expansion
python main.py --search "education" --expand

# Expand and translate
python main.py --search "education policy" --translate --expand
```

### Search with Entity Mapping

```powershell
# Enable entity mapping
python main.py --search "Sheikh Hasina" --entities

# Full entity processing
python main.py --search "Sheikh Hasina Bangladesh" --entities --translate
```

### Search with Semantic & Fuzzy

```powershell
# Enable semantic retrieval
python main.py --search "school teaching" --semantic

# Enable fuzzy matching (handles typos)
python main.py --search "educaton polcy" --fuzzy

# Both semantic and fuzzy
python main.py --search "education policy" --semantic --fuzzy
```

### Advanced Search (All Features)

```powershell
# Full-featured search
python main.py --search "Sheikh Hasina education policy" --method hybrid --translate --expand --entities --semantic --fuzzy

# Cross-lingual search with all features
python main.py --search "প্রধানমন্ত্রী শিক্ষা" --method hybrid --translate --semantic
```

### Search with Custom Result Count

```powershell
# Get top 5 results
python main.py --search "education" --top-k 5

# Get top 20 results
python main.py --search "education" --top-k 20

# Get top 50 results
python main.py --search "Bangladesh" --top-k 50
```

### Search with Custom Index Path

```powershell
# Use custom index file
python main.py --search "education" --index-path my_index/inverted_index.pkl

# Search with all custom paths
python main.py --search "policy" --index-path processed_data/inverted_index.pkl --top-k 15
```

---

## 🧪 Testing Modules

### Test All Modules

```powershell
python main.py --test
```

### Test Individual Modules

```powershell
# Test Module A (Indexing)
python main.py --test --module a

# Test Module B (Query Processing)
python main.py --test --module b

# Test Module C (Retrieval)
python main.py --test --module c
```

---

## 🎯 Complete Workflow Examples

### Example 1: Quick Start (Small Dataset)

```powershell
# Step 1: Crawl 5 articles per source
python main.py --limit 5

# Step 2: Index the documents
python main.py --index

# Step 3: Search
python main.py --search "Sheikh Hasina education"
```

### Example 2: English Only Workflow

```powershell
# Step 1: Crawl English sources only
python main.py --lang english --limit 20

# Step 2: Index
python main.py --index

# Step 3: Search English documents
python main.py --search "prime minister visits India" --method bm25
```

### Example 3: Cross-Lingual Workflow

```powershell
# Step 1: Crawl both languages
python main.py --lang all --limit 30

# Step 2: Index
python main.py --index

# Step 3: Cross-lingual search
python main.py --search "education policy" --translate --semantic
```

### Example 4: Advanced Research Workflow

```powershell
# Step 1: Crawl large dataset
python main.py --limit 100

# Step 2: Index with large batches
python main.py --index --batch-size 200

# Step 3: Test everything
python main.py --test

# Step 4: Advanced search
python main.py --search "Sheikh Hasina economic development" --method hybrid --translate --expand --entities --semantic --fuzzy --top-k 20
```

### Example 5: Specific Source Research

```powershell
# Step 1: Crawl specific source
python main.py --source prothom_alo --limit 50

# Step 2: Index
python main.py --index

# Step 3: Search
python main.py --search "শিক্ষা নীতি" --translate
```

---

## 📊 Output Files

### After Crawling

- `data/raw/bangla/<source>/*.json` - Bangla articles
- `data/raw/english/<source>/*.json` - English articles
- `data/metadata.csv` - Article metadata

### After Indexing

- `processed_data/inverted_index.pkl` - Inverted index
- `processed_data/document_metadata.json` - Enhanced metadata
- `processed_data/processing_statistics.json` - Statistics

---

## 💡 Tips & Best Practices

### For Crawling

```powershell
# Start small to test
python main.py --limit 5

# Use specific sources if some sites are slow
python main.py --source daily_star --limit 10

# Crawl one language at a time for better control
python main.py --lang bangla --limit 20
python main.py --lang english --limit 20
```

### For Indexing

```powershell
# Use larger batch sizes for faster processing
python main.py --index --batch-size 200

# Keep outputs separate for different experiments
python main.py --index --output-dir experiment1
python main.py --index --output-dir experiment2
```

### For Searching

```powershell
# Start with simple BM25 to verify index
python main.py --search "test" --method bm25

# Add features incrementally
python main.py --search "test" --method hybrid
python main.py --search "test" --method hybrid --translate
python main.py --search "test" --method hybrid --translate --semantic

# Use semantic search for conceptual queries
python main.py --search "school teaching methods" --semantic

# Use fuzzy for typo-prone queries
python main.py --search "edukation pollicy" --fuzzy
```

---

## 🔧 Troubleshooting

### Index Not Found

```powershell
# Make sure you indexed first
python main.py --index

# Then search
python main.py --search "query"
```

### No Results Found

```powershell
# Try different methods
python main.py --search "query" --method bm25
python main.py --search "query" --method semantic --semantic

# Try with translation
python main.py --search "query" --translate

# Try with more results
python main.py --search "query" --top-k 50
```

### Slow Search

```powershell
# Disable semantic/fuzzy for faster results
python main.py --search "query" --method bm25

# Use fewer results
python main.py --search "query" --top-k 5

# Disable translation and expansion
python main.py --search "query" --method bm25
```

---

## 📝 Quick Reference

| Task              | Command                                                                  |
| ----------------- | ------------------------------------------------------------------------ |
| Crawl all         | `python main.py`                                                         |
| Crawl Bangla      | `python main.py --lang bangla`                                           |
| Crawl limited     | `python main.py --limit 10`                                              |
| Index             | `python main.py --index`                                                 |
| Search            | `python main.py --search "query"`                                        |
| Test              | `python main.py --test`                                                  |
| BM25 search       | `python main.py --search "query" --method bm25`                          |
| Hybrid search     | `python main.py --search "query" --method hybrid`                        |
| Translated search | `python main.py --search "query" --translate`                            |
| Full features     | `python main.py --search "query" --method hybrid --translate --semantic` |

---

## 🎓 Learning Examples

### Example 1: Compare Retrieval Methods

```powershell
# BM25
python main.py --search "education policy" --method bm25 --top-k 5

# TF-IDF
python main.py --search "education policy" --method tfidf --top-k 5

# Semantic
python main.py --search "education policy" --method semantic --semantic --top-k 5

# Hybrid
python main.py --search "education policy" --method hybrid --top-k 5
```

### Example 2: Test Cross-Lingual Retrieval

```powershell
# English query on all documents
python main.py --search "education policy" --translate --top-k 10

# Bangla query on all documents
python main.py --search "শিক্ষা নীতি" --translate --top-k 10
```

### Example 3: Evaluate Query Expansion

```powershell
# Without expansion
python main.py --search "school" --method bm25 --top-k 5

# With expansion
python main.py --search "school" --method bm25 --expand --top-k 5
```

---

For more details, see:

- [MODULE_IMPLEMENTATION_GUIDE.md](MODULE_IMPLEMENTATION_GUIDE.md) - Setup & installation
- [MODULE_DOCUMENTATION.md](MODULE_DOCUMENTATION.md) - Detailed module explanations
