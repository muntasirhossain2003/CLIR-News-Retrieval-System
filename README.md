# Cross-Lingual Information Retrieval (CLIR) System

A multilingual news retrieval system for Bangla and English documents using lexical, semantic, and hybrid retrieval methods.

## Summary

This project builds a cross-lingual information retrieval pipeline for Bangla and English news. It crawls and cleans articles, constructs lexical and semantic indexes, and supports multiple retrieval strategies (BM25, TF-IDF, fuzzy matching, semantic embeddings, and hybrid ranking). The system includes a query pipeline with language detection, named entity extraction, and optional translation, plus a Streamlit UI for interactive search and comparison across methods.

## 📊 Project Status

- **Documents Collected:** 5,170 (2,585 Bangla + 2,585 English)
- **Metadata Entries:** 5,600 (deduplicated)
- **Sources:** 6 Bangla + 7 English news outlets
- **Retrieval Models:** BM25, TF-IDF, Fuzzy, Semantic (mE5), Hybrid

---

## 🚀 Quick Start Guide

### **Step 1: Install Dependencies**

```powershell
# Install Python packages
pip install -r requirements.txt

# Download spaCy model for English NER
python -m spacy download en_core_web_sm

# Download Stanza model for Bangla NER
python -c "import stanza; stanza.download('bn')"
```

### **Step 2: Verify Dataset**

```powershell
# Check document count
Get-ChildItem -Path "data\raw" -Recurse -Filter "*.json" | Measure-Object | Select-Object Count

# Verify metadata
python -c "import pandas as pd; df = pd.read_csv('data/metadata.csv'); print(df.groupby('language').size())"
```

### **Step 3: Build Indexes** ⚠️ CRITICAL - Must run before retrieval!

```powershell
# Build WHOOSH (lexical) and FAISS (semantic) indexes
cd "src\Module A — Dataset Construction & Indexing\indexing"
python build_unified_indexes.py

# Return to project root
cd ..\..\..
```

**Expected Output:**

- `indexes/whoosh/` - Lexical index for BM25 retrieval
- `indexes/semantic/` - FAISS index with embeddings
- `indexes/faiss_index.bin` - FAISS binary index file

**Time:** ~10-20 minutes (semantic embeddings are slow)

### **Step 4: Verify Indexes**

```powershell
python check_indexes.py
```

Should show:

- ✓ LEXICAL INDEX (WHOOSH/BM25): ~5,600 documents
- ✓ SEMANTIC INDEX: ~5,600 documents with 1024-dim embeddings

---

## 🔍 Running Retrieval Queries

### **Option A: Command Line (Module C)**

```powershell
# Simple query
python "src\Module C — Retrieval Models\retrieval_pipeline.py" "climate change Bangladesh" --method hybrid

# Cross-lingual query (English → Bangla docs)
python "src\Module C — Retrieval Models\retrieval_pipeline.py" "জলবায়ু পরিবর্তন" --method semantic

# Compare multiple methods
python "src\Module C — Retrieval Models\retrieval_pipeline.py" "করোনা ভাইরাস" --method all --top 10
```

**Available Methods:**

- `bm25` - Lexical retrieval (BM25 algorithm)
- `tfidf` - Lexical retrieval (TF-IDF)
- `fuzzy` - Edit distance matching
- `semantic` - Multilingual embeddings (mE5-large)
- `hybrid` - Combined scoring (recommended)
- `all` - Run all methods and compare

### **Option B: Web Interface (Streamlit)**

```powershell
streamlit run src\frontend\app.py
```

Then open: http://localhost:8501

**Features:**

- 🌐 Cross-lingual search (Bangla ↔ English)
- 📊 Multiple retrieval models
- 🎯 Confidence scores
- ⏱️ Performance metrics
- 🔍 Document previews

---

## 🧪 Testing Query Pipeline (Module B)

```powershell
cd "src\Module B — Query Processing & Cross-Lingual Handling"

# Test language detection & normalization
python query_pipeline.py "COVID-19 pandemic in Bangladesh"

# Test with translation
python query_pipeline.py "climate change" --target bn

# Test Bangla query
python query_pipeline.py "জলবায়ু পরিবর্তন" --target en

# JSON output
python query_pipeline.py "coronavirus" --target bn --json
```

**Output includes:**

- Detected language
- Normalized query
- Named entities (PERSON, LOCATION, ORG)
- Translated query (if --target specified)

---

## 📁 Project Structure

```
clir-project/
├── data/
│   ├── raw/                         # 5,170 JSON documents
│   │   ├── bangla/                  # 6 sources
│   │   └── english/                 # 7 sources
│   └── metadata.csv                 # 5,600 entries (cleaned)
│
├── indexes/                         # ⚠️ Generated after build
│   ├── whoosh/                      # Lexical index
│   └── semantic/                    # FAISS embeddings
│
├── src/
│   ├── Module A — Dataset Construction & Indexing/
│   │   ├── crawlers/                # News site crawlers
│   │   ├── indexing/                # WHOOSH + FAISS builders
│   │   │   └── build_unified_indexes.py  # ← Run this first!
│   │   └── generate_metadata.py
│   │
│   ├── Module B — Query Processing & Cross-Lingual Handling/
│   │   ├── language_detection_normalization.py
│   │   ├── named_entity_extraction.py
│   │   ├── query_translation.py
│   │   └── query_pipeline.py        # ← Full pipeline
│   │
│   ├── Module C — Retrieval Models/
│   │   ├── bm25_retrieval.py        # Model 1: Lexical
│   │   ├── fuzzy_retrieval.py       # Model 2: Fuzzy
│   │   ├── semantic_retrieval.py    # Model 3: Semantic
│   │   ├── hybrid_retrieval.py      # Model 4: Hybrid
│   │   └── retrieval_pipeline.py    # ← Main entry point
│   │
│   └── frontend/
│       └── app.py                   # Streamlit interface
│
├── main.py                          # Crawler launcher
├── check_indexes.py                 # Verify indexes
└── requirements.txt
```

---

## 🛠️ Common Issues & Solutions

### **Issue 1: "No module named 'sentence_transformers'"**

```powershell
pip install sentence-transformers torch
```

### **Issue 2: "Index not found" errors**

**Solution:** You must build indexes first!

```powershell
cd "src\Module A — Dataset Construction & Indexing\indexing"
python build_unified_indexes.py
cd ..\..\..
```

### **Issue 3: Slow semantic indexing**

**Expected:** First-time semantic indexing takes 10-20 minutes

- Downloads mE5-large-instruct model (~2GB)
- Generates 5,600 embeddings (1024-dim each)

**Speed up:**

- Use smaller model in `semantic_retrieval.py`: Change `DEFAULT_MODEL` to `"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"` (faster, smaller)

### **Issue 4: Translation errors with googletrans**

**Known issue:** googletrans sometimes fails. Fallback behavior:

- Uses normalized query if translation fails
- Logs warning in console

**Alternative:** Uncomment `deep-translator` in Module B code

### **Issue 5: Port 8501 already in use**

```powershell
streamlit run src\frontend\app.py --server.port 8502
```

---

## 📝 Usage Examples

### **Example 1: Cross-lingual news search**

```powershell
# English query → Find Bangla documents
python "src\Module C — Retrieval Models\retrieval_pipeline.py" "Bangladesh politics" --method semantic
```

### **Example 2: Bangla query → Find English documents**

```powershell
python "src\Module C — Retrieval Models\retrieval_pipeline.py" "অর্থনীতি" --method hybrid
```

### **Example 3: Compare retrieval methods**

```powershell
python "src\Module C — Retrieval Models\retrieval_pipeline.py" "technology mobile phone" --method all
```

---

## 🎯 Evaluation & Testing

### **Coming Soon: Module D (Evaluation)**

```powershell
# Evaluate with labeled queries
python evaluate.py --queries test_queries.json --metrics all

# Metrics: Precision@10, Recall@50, nDCG@10, MRR
```

### **Error Analysis (Module F)**

```powershell
# Analyze failure cases
python error_analysis.py --query "specific query" --analyze
```

---

## 🔧 Development Commands

### **Regenerate Metadata**

```powershell
cd "src\Module A — Dataset Construction & Indexing"
python generate_metadata.py
```

### **Crawl More Data**

```powershell
# Crawl specific source
python main.py --lang bangla --source prothom_alo --limit 100

# Crawl all sources
python main.py --lang all --limit 50
```

### **Rebuild Indexes**

```powershell
# Delete old indexes
Remove-Item -Recurse -Force indexes

# Rebuild
cd "src\Module A — Dataset Construction & Indexing\indexing"
python build_unified_indexes.py
```

---

## 📚 Documentation

- **Module A:** Dataset & Indexing (see `src/Module A/README.md`)
- **Module B:** Query Processing ([README](src/Module B — Query Processing & Cross-Lingual Handling/README.md))
- **Module C:** Retrieval Models (see code comments)
- **Module D:** Evaluation (TODO)
- **Module E:** Report (see Module E folder)

---

## 🎓 Academic Integrity

This project uses:

- Open-source libraries (cited in requirements.txt)
- Pre-trained models (mE5-large-instruct, spaCy, Stanza)
- News articles crawled from public websites (fair use for research)

**AI Tool Usage:** Document all AI-generated code in `AI_USAGE_LOG.md` as required by assignment.

---

## 📊 System Requirements

- **Python:** 3.8+
- **RAM:** 8GB minimum (16GB recommended for semantic indexing)
- **Disk:** 5GB (3GB for data, 2GB for models)
- **OS:** Windows 10/11, Linux, macOS

---

## 🐛 Support & Issues

1. **Check indexes exist:** `ls indexes`
2. **Verify dependencies:** `pip list | Select-String -Pattern "sentence-transformers|whoosh|spacy"`
3. **Check logs:** Look for ERROR/WARNING messages in console

---

## 📈 Next Steps (TODO)

- [ ] Implement evaluation metrics (Module D)
- [ ] Create labeled query dataset
- [ ] Write literature review (Module E)
- [ ] Complete error analysis (Module F)
- [ ] Add innovation component (Module G)
- [ ] Generate final report

---

## 👥 Contributors

- Siyam Bhuiyan (210041215)
- Syed Huzzatullah Mihad (210041218)
- Rafid Ahmed (210041232)
- Muntasir Hossain (210041265)

## 📄 License

Educational project for CSE 4739 - Data Mining course.
