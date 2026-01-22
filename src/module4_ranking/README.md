# Module 4: Ranking & Evaluation

## Purpose

Merges and ranks search results from multiple retrieval methods (lexical, semantic, fuzzy) and calculates evaluation metrics for CLIR system performance.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      Ranking & Evaluation                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Lexical Results    Semantic Results    Fuzzy Results             │
│  (BM25 scores)      (L2 distances)      (Fuzzy ratios)            │
│         │                  │                   │                   │
│         └──────────────────┴───────────────────┘                   │
│                           │                                        │
│                           ↓                                        │
│                  normalize_scores()                                │
│                  (Min-Max to [0,1])                                │
│                           │                                        │
│                           ↓                                        │
│                  merge_and_rank()                                  │
│                  Formula:                                          │
│                  score = α×semantic + β×lexical + γ×fuzzy          │
│                  (α + β + γ = 1)                                   │
│                           │                                        │
│                           ↓                                        │
│              ┌────────────────────────┐                            │
│              │  Ranked Results:       │                            │
│              │  - url                 │                            │
│              │  - title               │                            │
│              │  - content_preview     │                            │
│              │  - language            │                            │
│              │  - score (0-1)         │                            │
│              │  - date                │                            │
│              └────────────────────────┘                            │
│                           │                                        │
│                           ↓                                        │
│                  calculate_metrics()                               │
│                  (P@10, R@50, MRR, nDCG@10)                        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. **Score Normalization**

```python
def normalize_scores(results: List[dict], score_key: str) -> List[dict]:
    # Min-Max normalization to [0, 1]
    normalized_score = (score - min_score) / (max_score - min_score)
```

- Handles division by zero (all identical scores)
- Preserves relative ordering
- Makes scores comparable across methods

### 2. **Result Fusion**

```python
def merge_and_rank(
    lexical_results: List[dict],
    semantic_results: List[dict],
    fuzzy_results: List[dict] = None,
    alpha: float = 0.6,        # Semantic weight
    fuzzy_weight: float = 0.2  # Fuzzy weight
) -> List[dict]:
    # Lexical weight = 1 - alpha - fuzzy_weight
    final_score = (alpha × semantic) + (lexical_weight × lexical) + (fuzzy_weight × fuzzy)
```

**Default Weights:**

- Semantic (LaBSE): 60%
- Lexical (BM25): 20%
- Fuzzy (Transliteration): 20%

**Low Confidence Detection:**

- Flags when top_score < 0.2
- Indicates weak match across all methods

### 3. **Evaluation Metrics**

#### Precision@K (P@K)

```python
P@10 = (Relevant docs in top 10) / 10
```

#### Recall@K (R@K)

```python
R@50 = (Relevant docs in top 50) / (Total relevant docs)
```

#### Mean Reciprocal Rank (MRR)

```python
MRR = 1 / (Rank of first relevant doc)
```

#### Normalized Discounted Cumulative Gain (nDCG@K)

```python
DCG = Σ (relevance / log₂(position + 1))
nDCG = DCG / IDCG
```

## Data Flow

```
Input Results:
  lexical: [{url, score}, ...]   # BM25 scores (raw)
  semantic: [{url, score}, ...]  # L2 distances (lower = better)
  fuzzy: [{url, score}, ...]     # Fuzzy ratios (0-100)
       ↓
normalize_scores() for each
       ↓
  lexical: [{url, normalized_score: 0.85}, ...]
  semantic: [{url, normalized_score: 0.92}, ...]
  fuzzy: [{url, normalized_score: 0.67}, ...]
       ↓
merge_and_rank(alpha=0.6, fuzzy_weight=0.2)
       ↓
Final: [
  {url: "...", score: 0.88, title: "...", ...},  # Rank 1
  {url: "...", score: 0.76, title: "...", ...},  # Rank 2
  ...
]
       ↓
Top 50 results returned
```

## Usage

### Merge and Rank Results

```python
from src.module4_ranking.ranker import merge_and_rank

# Get results from retriever
lexical_results = retriever._whoosh_search(query, top_k=100)
semantic_results = retriever._faiss_search(query, top_k=100)
fuzzy_results = retriever._fuzzy_search(query, top_k=100)

# Merge with default weights (60% semantic, 20% lexical, 20% fuzzy)
final_results = merge_and_rank(
    lexical_results=lexical_results,
    semantic_results=semantic_results,
    fuzzy_results=fuzzy_results,
    alpha=0.6,
    fuzzy_weight=0.2
)

print(f"Top result: {final_results[0]['title']}")
print(f"Score: {final_results[0]['score']:.3f}")
```

### Calculate Evaluation Metrics

```python
from src.module4_ranking.ranker import calculate_metrics

# Define relevant URLs for a query
relevant_urls = [
    "https://example.com/article1",
    "https://example.com/article2"
]

# Calculate metrics for top 50 results
metrics = calculate_metrics(
    results=final_results,
    relevant_urls=relevant_urls
)

print(f"P@10: {metrics['precision@10']:.3f}")
print(f"R@50: {metrics['recall@50']:.3f}")
print(f"MRR: {metrics['mrr']:.3f}")
print(f"nDCG@10: {metrics['ndcg@10']:.3f}")
```

### Custom Weight Tuning

```python
# Emphasize semantic search (70%)
results_semantic = merge_and_rank(
    lexical_results, semantic_results, fuzzy_results,
    alpha=0.7, fuzzy_weight=0.15
)

# Emphasize lexical search (50%)
results_lexical = merge_and_rank(
    lexical_results, semantic_results, fuzzy_results,
    alpha=0.3, fuzzy_weight=0.2
)

# No fuzzy matching
results_no_fuzzy = merge_and_rank(
    lexical_results, semantic_results,
    alpha=0.6, fuzzy_weight=0.0
)
```

## Output Format

### Ranked Results

```python
[
    {
        'url': 'https://www.prothomalo.com/bangladesh/article123',
        'title': 'বাংলাদেশের অর্থনীতিতে নতুন সম্ভাবনা',
        'content_preview': 'বাংলাদেশের অর্থনীতি গত এক দশকে...',
        'lang': 'bn',
        'date': '2024-01-15',
        'score': 0.8823,
        'low_confidence': False
    },
    {
        'url': 'https://www.thedailystar.net/business/news/economy-2024',
        'title': 'Bangladesh Economy Shows Growth',
        'content_preview': 'The economy of Bangladesh has shown...',
        'lang': 'en',
        'date': '2024-01-14',
        'score': 0.7645,
        'low_confidence': False
    },
    ...
]
```

### Evaluation Metrics

```python
{
    'precision@10': 0.29,
    'recall@50': 1.0,
    'mrr': 0.80,
    'ndcg@10': 0.85
}
```

## Evaluation Results

### Test Queries (10 labeled queries)

**English Queries:**

- Precision@10: 0.29
- Recall@50: 1.00
- MRR: 0.80
- nDCG@10: 0.85

**Bangla Queries:**

- Precision@10: 0.21
- Recall@50: 1.00
- MRR: 0.73
- nDCG@10: 0.62

**Interpretation:**

- High recall (1.00) = All relevant docs found in top 50
- Good MRR (0.73-0.80) = Relevant docs appear early in results
- Moderate precision = Some irrelevant docs in top 10

## Dependencies

```
numpy==1.26.2
pandas==2.1.4
```

## Performance

- **Score Normalization**: ~1 ms per 100 results
- **Result Fusion**: ~5-10 ms for 300 results (100 × 3 methods)
- **Metric Calculation**: ~1-2 ms per query
- **Total Ranking Time**: ~10-15 ms per query

## Tuning Guidelines

### Weight Selection

1. **Semantic-Heavy (α=0.7-0.8)**: Best for conceptual queries
   - Example: "climate change impact" → "জলবায়ু পরিবর্তনের প্রভাব"

2. **Balanced (α=0.5-0.6)**: General purpose
   - Works well for mixed query types

3. **Lexical-Heavy (α=0.2-0.3)**: Best for named entities
   - Example: "ঢাকা বিশ্ববিদ্যালয়" → "Dhaka University"

4. **Fuzzy-Heavy (fuzzy_weight=0.3-0.4)**: Best for transliterated names
   - Example: "রোহিঙ্গা" → "Rohingya"

### Low Confidence Threshold

- Default: 0.2
- Increase to 0.3 for stricter quality control
- Decrease to 0.1 to reduce false negatives
- **Recall@50**: Coverage of relevant documents in top 50
- **MRR**: Mean Reciprocal Rank of first relevant document
- **nDCG@10**: Normalized Discounted Cumulative Gain at 10

## Usage

```python
from src.module4_ranking import Ranker

ranker = Ranker()

# Normalize scores
normalized = ranker.normalize_scores(results)

# Merge and rank results
merged = ranker.merge_and_rank(
    lexical_results=whoosh_results,
    semantic_results=faiss_results,
    alpha=0.5  # 50% semantic, 50% lexical
)

# Calculate metrics
metrics = ranker.calculate_metrics(
    retrieved_docs=['url1', 'url2', 'url3', ...],
    relevant_docs_ids={'url1', 'url5', 'url10'}
)

print(metrics)
# {
#     'precision@10': 0.3,
#     'recall@50': 0.6,
#     'mrr': 1.0,
#     'ndcg@10': 0.75
# }
```

## Methods

### `normalize_scores(results)`

Applies Min-Max normalization to result scores.

### `merge_and_rank(lexical_results, semantic_results, alpha=0.5)`

Fuses lexical and semantic results with configurable weighting.

Returns:

```python
{
    'results': [...],  # Sorted by final_score
    'warning': "Low confidence: ..." or None
}
```

### `calculate_metrics(retrieved_docs, relevant_docs_ids)`

Computes evaluation metrics for retrieved results.

## Dependencies

- `numpy`: For nDCG calculations
