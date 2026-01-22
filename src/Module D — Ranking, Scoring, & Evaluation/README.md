# Module D — Ranking, Scoring, & Evaluation

Comprehensive evaluation framework for CLIR system with ranking, scoring, metrics calculation, error analysis, and relevance labeling utilities.

---

## 📁 Module Structure

```
Module D — Ranking, Scoring, & Evaluation/
├── __init__.py                          # Package initialization
├── ranking_scorer.py                    # Ranking & scoring with confidence metrics
├── evaluation_metrics.py                # IR evaluation metrics (P@10, R@50, nDCG, MRR)
├── error_analysis.py                    # Error analysis and failure case detection
├── relevance_labeling.py               # Relevance label management & statistics
├── evaluate.py                         # Main evaluation pipeline script
├── test_queries.json                   # Sample test queries with relevance labels
└── README.md                           # This file
```

---

## 🎯 Module Components

### 1. **Ranking & Scoring** (`ranking_scorer.py`)

#### Purpose
Ranks documents, normalizes scores to [0,1], and generates confidence metrics.

#### Key Classes

**`QueryResult`** - Represents a single ranked result
```python
@dataclass
class QueryResult:
    doc_id: str                    # Document identifier
    title: str                     # Document title
    language: str                  # Document language (bangla/english)
    source: str                    # News source
    url: str                       # Document URL
    body_preview: str              # Text preview (150 chars)
    score: float                   # Original score from retrieval
    ranking_position: int          # Rank (1, 2, 3, ...)
    retrieval_method: str          # Method used (bm25, semantic, hybrid, etc.)
    confidence_score: float        # Normalized confidence [0, 1]
```

**`ExecutionMetrics`** - Query execution timing
```python
@dataclass
class ExecutionMetrics:
    total_time_ms: float           # Total retrieval time
    translation_time_ms: float     # Translation component time
    embedding_time_ms: float       # Embedding computation time
    ranking_time_ms: float         # Ranking & scoring time
    lexical_search_time_ms: float  # BM25/TF-IDF search time
    semantic_search_time_ms: float # Embedding search time
```

**`RankingScorer`** - Main ranking class
```python
class RankingScorer:
    def normalize_scores(scores, method="minmax")
        # Normalize scores to [0,1] using minmax or sigmoid
    
    def rank_documents(results, method="hybrid", top_k=10)
        # Rank and score documents, returns (QueryResult[], confidence)
    
    def generate_confidence_warning(top_confidence, query)
        # Generate warning if confidence < threshold (default 0.20)
    
    def combine_retrieval_scores(bm25_scores, semantic_scores)
        # Combine scores from multiple methods for hybrid retrieval
    
    def format_results(ranked_results)
        # Format results for display
    
    def format_execution_metrics()
        # Format timing breakdown for display
```

#### Usage Example

```python
from ranking_scorer import RankingScorer

# Initialize scorer with low confidence threshold of 0.20
scorer = RankingScorer(low_confidence_threshold=0.20)

# Normalize scores from retrieval
raw_scores = {
    "doc1": 0.95,
    "doc2": 0.75,
    "doc3": 0.45,
}
normalized = scorer.normalize_scores(raw_scores, method="minmax")
# Output: {doc1: 1.0, doc2: 0.667, doc3: 0.0}

# Rank documents
results = {
    "doc1": {"title": "Climate News", "score": 0.95, ...},
    "doc2": {"title": "Weather Report", "score": 0.75, ...},
}
ranked_results, top_confidence = scorer.rank_documents(
    results, method="hybrid", top_k=10
)

# Check confidence
warning = scorer.generate_confidence_warning(top_confidence, "climate change")
if warning:
    print(warning)
    # Output: ⚠️ Warning: Retrieved results may not be relevant...

# Combine scores from multiple methods
bm25_scores = {"doc1": 15.5, "doc2": 12.3}
semantic_scores = {"doc1": 0.92, "doc2": 0.78}
hybrid = scorer.combine_retrieval_scores(
    bm25_scores, semantic_scores,
    bm25_weight=0.4,
    semantic_weight=0.6
)

# Display results
print(scorer.format_results(ranked_results))
print(scorer.format_execution_metrics())
```

---

### 2. **Evaluation Metrics** (`evaluation_metrics.py`)

#### Purpose
Implements standard Information Retrieval evaluation metrics.

#### Key Metrics

| Metric | Formula | Target | Meaning |
|--------|---------|--------|---------|
| **Precision@K** | # relevant in top-K / K | ≥ 0.6 | Of top K results, how many are relevant? |
| **Recall@K** | # relevant in top-K / total relevant | ≥ 0.5 | Of all relevant docs, how many in top-K? |
| **nDCG@K** | DCG@K / IDCG@K | ≥ 0.5 | Quality of ranking, penalizes lower-ranked relevance |
| **MRR** | 1 / (rank of first relevant) | ≥ 0.4 | How fast to first relevant result? |
| **MAP** | Average precision across queries | - | Overall effectiveness across queries |

#### Key Methods

```python
class EvaluationMetrics:
    # Single-query metrics
    @staticmethod
    def precision_at_k(relevant_docs, retrieved_docs, k=10) -> float
    
    @staticmethod
    def recall_at_k(relevant_docs, retrieved_docs, k=50) -> float
    
    @staticmethod
    def ndcg(relevant_docs, retrieved_docs, k=10) -> float
    
    @staticmethod
    def reciprocal_rank(relevant_docs, retrieved_docs) -> float
    
    @staticmethod
    def average_precision(relevant_docs, retrieved_docs) -> float
    
    # Batch evaluation
    @staticmethod
    def evaluate_query(relevant_docs, retrieved_docs, query_id, k=10) -> Dict
        # Comprehensive metrics for one query
    
    @staticmethod
    def evaluate_batch(batch_queries, k=10) -> Tuple[Dict, Dict]
        # Evaluate multiple queries, return per-query and aggregate metrics
    
    @staticmethod
    def format_metrics(metrics) -> str
        # Format metrics for display
```

#### Usage Example

```python
from evaluation_metrics import EvaluationMetrics

# Single query evaluation
relevant_docs = ["doc1", "doc3", "doc5"]
retrieved_docs = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]

p10 = EvaluationMetrics.precision_at_k(relevant_docs, retrieved_docs, k=10)
# P@10 = 3/10 = 0.30 (3 relevant in top 10)

r50 = EvaluationMetrics.recall_at_k(relevant_docs, retrieved_docs, k=50)
# R@50 = 3/3 = 1.0 (all 3 relevant docs found)

ndcg = EvaluationMetrics.ndcg(relevant_docs, retrieved_docs, k=10)
# nDCG@10 = DCG / IDCG (penalizes lower-ranked relevant docs)

mrr = EvaluationMetrics.reciprocal_rank(relevant_docs, retrieved_docs)
# MRR = 1/1 = 1.0 (first relevant doc at rank 1)

# Comprehensive single query
metrics = EvaluationMetrics.evaluate_query(
    relevant_docs, retrieved_docs, query_id="q001", k=10
)
# Returns: {
#   "query_id": "q001",
#   "precision_at_10": 0.30,
#   "recall_at_10": 1.0,
#   "recall_at_50": 1.0,
#   "ndcg_at_10": 0.85,
#   "average_precision": 0.75,
#   "mrr": 1.0
# }

# Batch evaluation
batch_queries = {
    "q001": {"relevant": ["doc1", "doc3"], "retrieved": ["doc1", "doc2", "doc3"]},
    "q002": {"relevant": ["doc5"], "retrieved": ["doc5", "doc6"]},
    "q003": {"relevant": ["doc7", "doc8"], "retrieved": ["doc6", "doc7"]},
}

per_query_results, summary = EvaluationMetrics.evaluate_batch(batch_queries, k=10)
# summary returns:
# {
#   "mean_precision_at_10": 0.35,
#   "mean_recall_at_50": 0.67,
#   "mean_ndcg_at_10": 0.58,
#   "mean_average_precision": 0.52,
#   "mean_reciprocal_rank": 0.68,
#   "num_queries": 3
# }

print(EvaluationMetrics.format_metrics(metrics))
```

---

### 3. **Error Analysis** (`error_analysis.py`)

#### Purpose
Identifies and categorizes retrieval failures for detailed debugging.

#### Error Types

1. **Translation Failures** - Query mistranslated to different word
   - Example: "চেয়ার" (chair) → "Chairman" ❌

2. **Named Entity Mismatches** - Entity not recognized across languages
   - Example: "ঢাকা" (Bangla) vs "Dhaka" (English)

3. **Cross-Script Issues** - Same word written differently
   - Example: "Bangladesh" vs "বাংলাদেশ" vs "Bangla Desh"

4. **Code-Switching** - Query mixes Bangla and English
   - Example: "আমরা COVID-19 এর বিরুদ্ধে লড়াই করছি"

5. **Semantic vs Lexical Wins** - One retrieval method significantly outperforms
   - Example: Query "শিক্ষা" (education) - BM25 returns 0, Semantic finds "স্কুল" (school)

#### Key Methods

```python
class ErrorAnalyzer:
    def add_translation_failure(query_id, original_query, mistranslated_query, ...)
    def add_ner_mismatch(query_id, entity_in_query, entity_in_docs, ...)
    def add_cross_script_issue(query_id, script_variant_1, script_variant_2, ...)
    def add_code_switching_issue(query_id, query_text, mixed_components, ...)
    def add_semantic_vs_lexical(query_id, winner, lexical_results, semantic_results, ...)
    
    def summarize_errors() -> Dict[str, int]
        # Get count by error type
    
    def get_errors_by_type(error_type: str) -> List[ErrorCase]
    
    def format_error_report() -> str
        # Detailed error analysis report
    
    def format_error_summary_table() -> str
        # Quick summary table
```

#### Usage Example

```python
from error_analysis import ErrorAnalyzer

analyzer = ErrorAnalyzer()

# Log translation failure
analyzer.add_translation_failure(
    query_id="q001",
    query_text="chairperson in Bangladesh",
    query_language="english",
    original_query="চেয়ার",
    mistranslated_query="Chairman",
    expected_docs=["doc_chair1", "doc_chair2"],
    retrieved_docs=["doc_chairman1", "doc_politics1"],
    example="Query intended furniture, but got political results"
)

# Log NER mismatch
analyzer.add_ner_mismatch(
    query_id="q002",
    query_text="news from Dhaka",
    query_language="english",
    entity_in_query="ঢাকা",
    entity_in_docs="Dhaka",
    expected_docs=["dhaka_news_1", "dhaka_news_2"],
    retrieved_docs=["dhaka_politics_1"],
    example="Entity not recognized across script boundaries"
)

# Log code-switching
analyzer.add_code_switching_issue(
    query_id="q003",
    query_text="আমরা COVID-19 এর বিরুদ্ধে লড়াই করছি",
    mixed_components=["Bangla", "English (COVID-19)"],
    expected_docs=["covid_bn_1", "covid_en_1"],
    retrieved_docs=["political_news_1"],
    example="Language detection failed on mixed query"
)

# Analyze
summary = analyzer.summarize_errors()
# Output: {"translation": 5, "ner_mismatch": 3, "code_switch": 2}

# Get specific error type
translation_errors = analyzer.get_translation_failures()

# Print reports
print(analyzer.format_error_summary_table())
print(analyzer.format_error_report())
```

---

### 4. **Relevance Labeling** (`relevance_labeling.py`)

#### Purpose
Manage relevance labels for evaluation dataset and track annotator agreement.

#### Key Classes

```python
@dataclass
class RelevanceLabel:
    query_id: str          # Query identifier
    query_text: str        # Query text
    doc_id: str            # Document ID
    doc_title: str         # Document title
    doc_url: str           # Document URL
    language: str          # Document language
    relevant: bool         # Is relevant (True/False)
    confidence: int        # Annotator confidence (0-3)
    annotator: str         # Annotator name
    notes: str             # Notes about labeling
```

#### Key Methods

```python
class RelevanceLabeler:
    def add_label(query_id, query_text, doc_id, doc_title, relevant, ...)
    def get_relevant_docs_for_query(query_id) -> List[str]
    def get_all_docs_for_query(query_id) -> List[str]
    def save_to_csv(filepath)
    def load_from_csv(filepath)
    def save_to_json(filepath)
    def load_from_json(filepath)
    def get_statistics() -> Dict
    def format_statistics() -> str
    def inter_annotator_agreement() -> float
        # Calculate agreement between multiple annotators
    
    @staticmethod
    def create_sample_labeling_csv(output_path)
        # Generate template for manual labeling
```

#### CSV Format

```csv
query_id,query_text,doc_id,doc_title,doc_url,language,relevant,confidence,annotator,notes
q001,climate change Bangladesh,doc_123,Climate Crisis in Bangladesh,https://example.com/article,english,yes,3,annotator_1,Directly addresses the query
q001,climate change Bangladesh,doc_456,Sports News,https://example.com/sports,english,no,3,annotator_1,Unrelated to query
```

#### Usage Example

```python
from relevance_labeling import RelevanceLabeler

labeler = RelevanceLabeler()

# Add labels manually
labeler.add_label(
    query_id="q001",
    query_text="climate change Bangladesh",
    doc_id="doc_123",
    doc_title="Climate Crisis in Bangladesh",
    doc_url="https://example.com/article",
    language="english",
    relevant=True,
    confidence=3,  # High confidence
    annotator="annotator_1",
    notes="Directly addresses climate crisis in Bangladesh"
)

labeler.add_label(
    query_id="q001",
    query_text="climate change Bangladesh",
    doc_id="doc_456",
    doc_title="Sports News",
    doc_url="https://example.com/sports",
    language="english",
    relevant=False,
    confidence=3,
    annotator="annotator_1",
    notes="Unrelated to query"
)

# Get relevant docs for query
relevant_for_q001 = labeler.get_relevant_docs_for_query("q001")
# Output: ["doc_123"]

# Save to CSV
labeler.save_to_csv("labels.csv")

# Load from CSV
labeler2 = RelevanceLabeler()
labeler2.load_from_csv("labels.csv")

# Get statistics
stats = labeler.get_statistics()
# {
#   "total_labels": 50,
#   "relevant_count": 32,
#   "relevant_percentage": 64.0,
#   "unique_queries": 10,
#   "unique_documents": 45,
#   "average_confidence": 2.8
# }

print(labeler.format_statistics())

# Inter-annotator agreement
agreement = labeler.inter_annotator_agreement()
print(f"Agreement: {agreement:.2%}")

# Create template for manual labeling
RelevanceLabeler.create_sample_labeling_csv("labeling_template.csv")
```

---

## 🚀 Complete Evaluation Workflow

### Step 1: Prepare Test Queries

Create `test_queries.json` with relevance labels:

```json
{
  "q001": {
    "query_text": "climate change Bangladesh",
    "language": "english",
    "relevant_docs": ["doc_123", "doc_456"],
    "retrieved_docs": ["doc_123", "doc_789", "doc_456"]
  },
  "q002": {
    "query_text": "COVID-19 pandemic",
    "language": "english",
    "relevant_docs": ["doc_covid_1", "doc_covid_2"],
    "retrieved_docs": ["doc_covid_1", "doc_other", "doc_covid_2"]
  }
}
```

### Step 2: Create Labeling Template (Optional)

```powershell
cd "src\Module D — Ranking, Scoring, & Evaluation"
python evaluate.py --create-template
```

This creates `sample_labeling_template.csv` for manual annotation.

### Step 3: Run Evaluation

```powershell
# Evaluate with default test queries
python evaluate.py

# Evaluate with custom queries
python evaluate.py --queries my_test_queries.json

# Evaluate with multiple methods
python evaluate.py --methods hybrid bm25 semantic tfidf

# Save to custom output file
python evaluate.py --output my_results.json
```

### Step 4: Analyze Results

```powershell
# Results saved to evaluation_results.json
# Contains:
# - Per-query metrics
# - Aggregate metrics for each method
# - Method comparison
```

---

## 📊 Example Workflows

### Workflow 1: Quick Evaluation

```python
from ranking_scorer import RankingScorer
from evaluation_metrics import EvaluationMetrics

# Get results from retrieval system
results = module_c_retrieval.retrieve("climate change Bangladesh", method="hybrid")

# Rank and score
scorer = RankingScorer()
ranked, confidence = scorer.rank_documents(results, method="hybrid", top_k=10)

# Check confidence
warning = scorer.generate_confidence_warning(confidence, "climate change Bangladesh")
if warning:
    print(warning)

# Evaluate against ground truth
relevant_docs = ["doc_123", "doc_456", "doc_789"]
retrieved_docs = [r.doc_id for r in ranked]

metrics = EvaluationMetrics.evaluate_query(relevant_docs, retrieved_docs)
print(EvaluationMetrics.format_metrics(metrics))
```

### Workflow 2: Full System Evaluation

```python
# Load evaluation script
from evaluate import load_test_queries, evaluate_batch, print_evaluation_report

# Load test queries
test_queries = load_test_queries("test_queries.json")

# Evaluate all methods
results = evaluate_batch(test_queries, methods=["bm25", "semantic", "hybrid"])

# Print report
report = print_evaluation_report(results)
print(report)

# Save results
import json
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Workflow 3: Error Analysis

```python
from error_analysis import ErrorAnalyzer

analyzer = ErrorAnalyzer()

# Identify failures
retrieved = ["doc_politics", "doc_sports"]  # Wrong results
expected = ["doc_climate_1", "doc_climate_2"]

if set(expected) & set(retrieved) == set():
    # Complete failure
    analyzer.add_translation_failure(
        query_id="q001",
        query_text="climate change",
        query_language="english",
        original_query="জলবায়ু",
        mistranslated_query="weather",
        expected_docs=expected,
        retrieved_docs=retrieved
    )

# Print error analysis
print(analyzer.format_error_summary_table())
print(analyzer.format_error_report())
```

### Workflow 4: Relevance Labeling

```python
from relevance_labeling import RelevanceLabeler

labeler = RelevanceLabeler()

# Load annotations from CSV
labeler.load_from_csv("annotations.csv")

# Get metrics
stats = labeler.get_statistics()
print(f"Total Labels: {stats['total_labels']}")
print(f"Agreement: {labeler.inter_annotator_agreement():.2%}")

# Get relevant docs for evaluation
for query_id in ["q001", "q002", "q003"]:
    relevant = labeler.get_relevant_docs_for_query(query_id)
    print(f"Query {query_id}: {len(relevant)} relevant docs")
```

---

## ✅ Success Checklist

Before considering Module D complete, verify:

- ✓ Ranking and scoring working with confidence metrics
- ✓ All IR metrics implemented (P@K, R@K, nDCG, MRR, MAP)
- ✓ Error analysis capturing translation, NER, script, and code-switching failures
- ✓ Relevance labeling system with CSV/JSON I/O
- ✓ Full evaluation pipeline runnable with `evaluate.py`
- ✓ Evaluation results show:
  - Precision@10 ≥ 0.6
  - Recall@50 ≥ 0.5
  - nDCG@10 ≥ 0.5
  - MRR ≥ 0.4
- ✓ Error analysis identifies real failure patterns
- ✓ Inter-annotator agreement calculated if multiple annotators

---

## 🔧 Advanced Usage

### Custom Evaluation Metrics

```python
from evaluation_metrics import EvaluationMetrics

# Implement custom metric
def f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# Use in evaluation
for query_id in results:
    p = results[query_id]["precision_at_10"]
    r = results[query_id]["recall_at_50"]
    f1 = f1_score(p, r)
```

### Combining Multiple Evaluation Sources

```python
# Compare with Google/Bing results if available
manual_eval = load_json("manual_evaluation.json")
clir_eval = load_json("evaluation_results.json")

# Compare metrics
for query in manual_eval:
    manual_p10 = manual_eval[query]["precision@10"]
    clir_p10 = clir_eval[query]["precision@10"]
    improvement = (clir_p10 - manual_p10) / manual_p10 * 100
    print(f"Query {query}: {improvement:+.1f}%")
```

---

## 📝 Notes

- All confidence scores normalized to [0, 1]
- Low confidence threshold adjustable in `RankingScorer`
- Error analysis extensible for custom error types
- Labeling system supports multiple annotators with agreement calculation
- All metrics follow standard IR definitions (TREC, CLEF standards)

---

## 📚 References

- TREC (Text REtrieval Conference) - standard evaluation framework
- CLEF (Conference and Labs of the Evaluation Forum) - multilingual IR evaluation
- Baeza-Yates & Ribeiro-Neto (2011) - Modern Information Retrieval

---

## 🐛 Troubleshooting

**Issue: "No labels found"**
- Create test_queries.json with proper format
- Or create labels using `RelevanceLabeler.add_label()`

**Issue: Metrics all zeros**
- Verify relevant_docs and retrieved_docs are non-empty
- Check doc_ids match between ground truth and results

**Issue: Low confidence scores**
- Adjust `low_confidence_threshold` in RankingScorer
- Check if retrieval results are actually relevant

**Issue: Error analysis not working**
- Ensure error types match predefined categories
- Use `add_translation_failure()`, `add_ner_mismatch()`, etc.

---

## 📞 Support

For questions or issues, check:
1. Test query format in `test_queries.json`
2. Metric calculations in `evaluation_metrics.py`
3. Error type definitions in `error_analysis.py`
4. Label format in `relevance_labeling.py`
