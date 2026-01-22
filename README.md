# CLIR System: Cross-Lingual Information Retrieval

A production-ready Cross-Lingual Information Retrieval system for Bangla-English news articles with hybrid ranking, semantic search, and interactive web interface.

## System Architecture

```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                          PRESENTATION LAYER (Frontend)                            ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐  ║
║  │                      STREAMLIT WEB INTERFACE (app.py)                       │  ║
║  │   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐      │  ║
║  │   │   Search Page    │  │  Evaluation &    │  │   Error Analysis     │      │  ║
║  │   │  • Query Input   │  │    Metrics       │  │  • 5 Error Types     │      │  ║
║  │   │  • Top 10 Docs   │  │  • P@10, R@50    │  │  • Distribution      │      │  ║
║  │   │  • Score Display │  │  • MRR, nDCG@10  │  │  • Recommendations   │      │  ║
║  │   └────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘      │  ║
║  └────────────┼─────────────────────┼─────────────────────────┼────────────────┘  ║
╚══════════════╪═════════════════════╪═════════════════════════╪════════════════════╝
                │                     │                         │
                └─────────────────────┼─────────────────────────┘
                                     ▼
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                        MODULE 2: QUERY PROCESSING LAYER                           ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐  ║
║  │                      QueryProcessor (query_processor.py)                    │  ║
║  │                                                                             │  ║
║  │   Step 1: Text Normalization          Step 2: Language Detection            │  ║
║  │   ┌────────────────────┐               ┌────────────────────┐               │  ║
║  │   │ • Remove extra     │               │ • Detect Bangla    │               │  ║
║  │   │   whitespace       │    ────────►  │   vs English       │               │  ║
║  │   │ • Clean special    │               │ • Unicode range    │               │  ║
║  │   │   characters       │               │   checking         │               │  ║
║  │   └────────────────────┘               └──────────┬─────────┘               │  ║
║  │                                                   │                        │  ║
║  │   Step 3: Cross-Lingual Translation (if Bangla)   ▼                         │  ║
║  │   ┌───────────────────────────────────────────────────────────┐             │  ║
║  │   │  LaBSE Multilingual Embeddings (sentence-transformers)    │             │  ║
║  │   │  • Direct semantic understanding (no explicit translation)│             │  ║
║  │   │  • 768-dimensional dense vectors                          │             │  ║
║  │   │  • Preserves cross-lingual semantic similarity            │             │  ║
║  │   └───────────────────────────────────────────────────────────┘             │  ║
║  │                                                                             │  ║
║  │   Output: Processed Query + Original Query + Language Info                  │  ║
║  └─────────────────────────────────────────────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════╪═══════════════════════════╝
                                                        ▼
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                          MODULE 3: RETRIEVAL LAYER                                ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐  ║
║  │                        Retriever (retriever.py)                             │  ║
║  │                                                                             │  ║
║  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │  ║
║  │  │  Lexical         │  │   Semantic       │  │   Fuzzy Search           │   │  ║
║  │  │  Retrieval       │  │  Retrieval       │  │  (Transliteration)       │   │  ║
║  │  ├──────────────────┤  ├──────────────────┤  ├──────────────────────────┤   │  ║
║  │  │ Technology:      │  │ Technology:      │  │ Technology:              │   │  ║
║  │  │ • Whoosh Index   │  │ • LaBSE + FAISS  │  │ • FuzzyWuzzy             │   │  ║
║  │  │ • BM25 Algorithm │  │ • Vector Search  │  │ • Title-only search      │   │  ║
║  │  │                  │  │ • k=50 nearest   │  │ • Named entity focus     │   │  ║
║  │  │ Strengths:       │  │                  │  │                          │   │  ║
║  │  │ • Exact keyword  │  │ Strengths:       │  │ Strengths:               │   │  ║
║  │  │   matching       │  │ • Cross-lingual  │  │ • Handles spelling       │   │  ║
║  │  │ • Term frequency │  │   understanding  │  │   variations             │   │  ║
║  │  │ • Fast lookup    │  │ • Semantic sim.  │  │ • Name transliteration   │   │  ║
║  │  │                  │  │ • Context aware  │  │ • Phonetic matching      │   │  ║
║  │  │ Returns:         │  │                  │  │                          │   │  ║
║  │  │ • Top 50 docs    │  │ Returns:         │  │ Returns:                 │   │  ║
║  │  │ • BM25 scores    │  │ • Top 50 docs    │  │ • Top 50 docs            │   │  ║
║  │  │                  │  │ • Cosine sim.    │  │ • Fuzzy scores           │   │  ║
║  │  └────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────────┘   │  ║
║  └───────────┼─────────────────────┼─────────────────────────┼─────────────────┘  ║
╚══════════════╪═════════════════════╪═════════════════════════╪════════════════════╝
               │                     │                         │
               └─────────────────────┼─────────────────────────┘
                                     ▼
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                       MODULE 4: RANKING & FUSION LAYER                            ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐  ║
║  │                          Ranker (ranker.py)                                 │  ║
║  │                                                                             │  ║
║  │  ┌──────────────────────────────────────────────────────────────────────┐  │  ║
║  │  │                     SCORE NORMALIZATION                              │  │  ║
║  │  │  • Min-Max scaling to [0, 1] range for each retrieval method        │  │  ║
║  │  │  • Ensures fair comparison across different scoring systems          │  │  ║
║  │  └──────────────────────────────────────────────────────────────────────┘  │  ║
║  │                                     │                                       │  ║
║  │                                     ▼                                       │  ║
║  │  ┌──────────────────────────────────────────────────────────────────────┐  │  ║
║  │  │                     WEIGHTED SCORE FUSION                            │  │  ║
║  │  │                                                                      │  │  ║
║  │  │   Final Score = (0.60 × Semantic) + (0.20 × Lexical) + (0.20 × Fuzzy) │  ║
║  │  │                                                                      │  │  ║
║  │  │   Rationale:                                                         │  │  ║
║  │  │   • Semantic (60%): Primary for cross-lingual understanding          │  │  ║
║  │  │   • Lexical  (20%): Boosts exact keyword matches                     │  │  ║
║  │  │   • Fuzzy    (20%): Handles named entities & transliteration         │  │  ║
║  │  └──────────────────────────────────────────────────────────────────────┘  │  ║
║  │                                     │                                       │  ║
║  │                                     ▼                                       │  ║
║  │  ┌──────────────────────────────────────────────────────────────────────┐  │  ║
║  │  │                    EVALUATION METRICS                                │  │  ║
║  │  │                                                                      │  │  ║
║  │  │  • P@10   : Precision at top 10 (relevance accuracy)                │  │  ║
║  │  │  • R@50   : Recall at top 50 (coverage of relevant docs)            │  │  ║
║  │  │  • MRR    : Mean Reciprocal Rank (first relevant position)          │  │  ║
║  │  │  • nDCG@10: Normalized Discounted Cumulative Gain (ranked quality)  │  │  ║
║  │  └──────────────────────────────────────────────────────────────────────┘  │  ║
║  │                                                                             │  ║
║  │  Output: Ranked Documents (Top 10) with Final Scores                       │  ║
║  └─────────────────────────────────────────────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
                                        ▲
                                        │ (reads from)
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                       MODULE 1: DATA & INDEXING LAYER                             ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐  ║
║  │                                                                             │  ║
║  │  ┌─────────────────────────────────────────────────────────────────────┐   │  ║
║  │  │                      RAW DATA COLLECTION                            │   │  ║
║  │  │  • 5,194 News Articles (Bangla: 2,500+ | English: 2,700+)          │   │  ║
║  │  │  • 14 Sources: 6 Bangla + 8 English newspapers                     │   │  ║
║  │  │  • JSON format: {title, body, url, source, language, date}         │   │  ║
║  │  │  • Web Crawlers: Selenium (dynamic) + BeautifulSoup (static)       │   │  ║
║  │  └────────────────────────────────┬────────────────────────────────────┘   │  ║
║  │                                   │                                         │  ║
║  │                                   ▼                                         │  ║
║  │  ┌────────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │  ║
║  │  │  Metadata CSV      │    │  Whoosh Index    │    │  FAISS Index     │   │  ║
║  │  │  (metadata.csv)    │    │  (indices/whoosh)│    │  (embeddings/)   │   │  ║
║  │  ├────────────────────┤    ├──────────────────┤    ├──────────────────┤   │  ║
║  │  │ • Document IDs     │    │ • Inverted Index │    │ • LaBSE vectors  │   │  ║
║  │  │ • Source info      │    │ • BM25 weights   │    │ • 5,194 × 768    │   │  ║
║  │  │ • Language labels  │    │ • Term positions │    │ • Cosine search  │   │  ║
║  │  │ • File paths       │    │ • Fast lexical   │    │ • GPU-optimized  │   │  ║
║  │  │ • Timestamps       │    │   retrieval      │    │ • Batch queries  │   │  ║
║  │  └────────────────────┘    └──────────────────┘    └──────────────────┘   │  ║
║  │                                                                             │  ║
║  │  Tools: Indexer, EmbeddingGenerator, MetadataGenerator                     │  ║
║  └─────────────────────────────────────────────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════════════════════════╝

                         DATA FLOW SUMMARY
    ┌────────────────────────────────────────────────────┐
    │  User Query (Bangla/English)                       │
    └────────────────┬───────────────────────────────────┘
                     ▼
    ┌────────────────────────────────────────────────────┐
    │  Query Processing (Normalization + Language ID)    │
    └────────────────┬───────────────────────────────────┘
                     ▼
    ┌────────────────────────────────────────────────────┐
    │  Parallel Retrieval (BM25 + LaBSE + Fuzzy)         │
    └────────────────┬───────────────────────────────────┘
                     ▼
    ┌────────────────────────────────────────────────────┐
    │  Score Fusion + Ranking (Weighted combination)     │
    └────────────────┬───────────────────────────────────┘
                     ▼
    ┌────────────────────────────────────────────────────┐
    │  Top 10 Results (With scores & metadata)           │
    └────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Clone repository
cd clir-project

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web Interface

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

### 3. Navigate the Interface

- **Search Page**: Cross-lingual search with Bangla/English queries
- **Evaluation & Metrics**: Interactive charts showing P@10, R@50, MRR, nDCG@10
- **Error Analysis**: Visualize 5 error categories with examples

## Features

### Core Capabilities

- **Cross-Lingual Search**: Query in Bangla, retrieve English documents (and vice versa)
- **Hybrid Ranking**: Combines semantic, lexical, and fuzzy matching
- **5,194 Documents**: News articles from 14 Bangladeshi sources
- **Multi-Page UI**: Professional Streamlit interface with interactive charts

### Retrieval Methods

| Method       | Technology      | Use Case                 |
| ------------ | --------------- | ------------------------ |
| **Lexical**  | BM25 (Whoosh)   | Keyword matching         |
| **Semantic** | LaBSE + FAISS   | Cross-lingual embeddings |
| **Fuzzy**    | Transliteration | Named entity matching    |

### Evaluation Metrics

- **P@10**: Precision at top 10 results
- **R@50**: Recall at top 50 results
- **MRR**: Mean Reciprocal Rank
- **nDCG@10**: Normalized Discounted Cumulative Gain

### Error Analysis (5 Categories)

1. Translation Drift
2. Tokenization Issues
3. Named Entity Failures
4. Domain/Topic Mismatch
5. Stopword/Function Word Issues

## Project Structure

```
clir-project/
├── app.py                              # Streamlit web interface
├── main.py                             # Data acquisition CLI
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── data/
│   ├── metadata.csv                    # 5,194 articles metadata
│   ├── embeddings/                     # LaBSE embeddings
│   ├── indices/
│   │   └── whoosh/                     # BM25 index
│   └── raw/                            # Original JSON articles
│       ├── bangla/                     # 6 Bangla sources
│       └── english/                    # 8 English sources
│
├── src/
│   ├── module1_data_acquisition/       # Web crawlers & metadata
│   ├── module1_preprocessing_indexing/ # Indexing & embeddings
│   ├── module2_query_processing/       # Query normalization
│   ├── module3_retrieval/              # BM25, FAISS, Fuzzy
│   └── module4_ranking/                # Hybrid fusion & metrics
│
├── evaluation_results/                 # HTML/CSV/PNG reports
├── evaluation_detailed.py              # Error analysis script
└── run_evaluation_with_visualization.py # Evaluation pipeline
```

## Technologies

### Core Stack

- **Python 3.12** - Programming language
- **Streamlit** - Web interface
- **Plotly** - Interactive charts
- **Pandas** - Data manipulation

### NLP & IR

- **LaBSE** (sentence-transformers) - Multilingual embeddings
- **Whoosh** - BM25 indexing
- **FAISS** - Vector similarity search
- **FuzzyWuzzy** - String matching

### Data Collection

- **Selenium** - Dynamic web scraping
- **BeautifulSoup** - HTML parsing

## Usage Examples

### Search from Command Line

```python
from src.module2_query_processing.query_processor import QueryProcessor
from src.module3_retrieval.retriever import Retriever
from src.module4_ranking.ranker import Ranker

query_processor = QueryProcessor()
retriever = Retriever()
ranker = Ranker()

query = "ঢাকার বায়ু দূষণ"  # Bangla query
processed = query_processor.process_query(query)
results = retriever.retrieve(processed)
ranked = ranker.rank(results, processed)

for doc in ranked[:10]:
    print(f"{doc['title']} - Score: {doc['final_score']:.4f}")
```

### Generate Evaluation Reports

```bash
python run_evaluation_with_visualization.py
```

Output: `evaluation_results/evaluation_summary.html`

### Run Error Analysis

```bash
python evaluation_detailed.py
```

Shows 5 error categories with detailed examples.

## Data Sources

### Bangla (6 sources)

- Prothom Alo, Ittefaq, Bangla Tribune, Dhaka Post, Samakal, Jugantor

### English (8 sources)

- Daily Star, New Age, Dhaka Tribune, Financial Express, NTV BD, UNB, Prothom Alo English, Daily Observer

## Performance

- **Dataset**: 5,194 documents
- **Average Query Time**: <1 second
- **P@10**: ~0.40-0.60 (varies by query)
- **MRR**: ~0.50-1.00
- **nDCG@10**: ~0.60-0.85

## Documentation

- [Module 1: Data Acquisition](src/module1_data_acquisition/)
- [Module 2: Query Processing](src/module2_query_processing/README.md)
- [Module 3: Retrieval](src/module3_retrieval/)
- [Module 4: Ranking & Metrics](src/module4_ranking/README.md)

## License

Academic project for cross-lingual information retrieval research.

---
