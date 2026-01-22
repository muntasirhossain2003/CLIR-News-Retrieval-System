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

### Example: Step-by-Step Metric Calculation

#### Scenario Setup

**Query**: "Dhaka air pollution" (বায়ু দূষণ ঢাকা)

**Ground Truth** (manually labeled relevant documents):

- doc_A: "Air quality crisis in Dhaka reaches dangerous levels"
- doc_C: "ঢাকার বায়ু দূষণ স্বাস্থ্যের জন্য হুমকি"
- doc_F: "Pollution monitoring in Bangladesh capital"
- doc_G: "বাংলাদেশের রাজধানীতে পরিবেশ দূষণ"

**Total relevant documents**: 4

**System Retrieved Results** (what our CLIR system returned, ranked by score):

| Rank | Document | Title                          | Relevant? | Score |
| ---- | -------- | ------------------------------ | --------- | ----- |
| 1    | doc_A    | Air quality crisis in Dhaka... | ✓ Yes     | 0.92  |
| 2    | doc_B    | Traffic congestion in Dhaka    | ✗ No      | 0.85  |
| 3    | doc_C    | ঢাকার বায়ু দূষণ...            | ✓ Yes     | 0.81  |
| 4    | doc_D    | Political news Bangladesh      | ✗ No      | 0.76  |
| 5    | doc_E    | Weather forecast Dhaka         | ✗ No      | 0.72  |
| 6    | doc_F    | Pollution monitoring...        | ✓ Yes     | 0.68  |
| 7    | doc_G    | বাংলাদেশের রাজধানীতে...        | ✓ Yes     | 0.65  |
| 8    | doc_H    | Sports news Bangladesh         | ✗ No      | 0.61  |
| 9    | doc_I    | Economic development           | ✗ No      | 0.58  |
| 10   | doc_J    | Entertainment news             | ✗ No      | 0.55  |
| ...  | ...      | ...                            | ...       | ...   |
| 50   | doc_Z    | Random article                 | ✗ No      | 0.12  |

---

#### 1. Precision@10 Calculation

**Definition**: Out of the top 10 results shown to the user, what fraction are actually relevant?

**Formula**: `P@10 = (Number of relevant documents in top 10) / 10`

**Purpose**: Measures the accuracy of top results. High precision means users see mostly relevant documents.

**Step-by-step**:

1. Look at top 10 results: doc_A, doc_B, doc_C, doc_D, doc_E, doc_F, doc_G, doc_H, doc_I, doc_J
2. Count relevant ones: doc_A ✓, doc_C ✓, doc_F ✓, doc_G ✓
3. Total relevant in top 10 = 4
4. P@10 = 4 / 10 = **0.40** (40%)

**Interpretation**:

- 40% of top results are relevant
- User sees 4 good results and 6 irrelevant ones
- Moderate precision - room for improvement

**Why it matters**: Users typically only look at top 10 results. If precision is low, they waste time on irrelevant documents.

---

#### 2. Recall@50 Calculation

**Definition**: Out of ALL relevant documents that exist, what fraction did we find in the top 50 results?

**Formula**: `R@50 = (Relevant documents found in top 50) / (Total relevant documents)`

**Purpose**: Measures coverage. High recall means we didn't miss important documents.

**Step-by-step**:

1. Total relevant documents in dataset = 4 (doc_A, doc_C, doc_F, doc_G)
2. Check which ones appear in top 50:
   - doc_A: ✓ Found at position 1
   - doc_C: ✓ Found at position 3
   - doc_F: ✓ Found at position 6
   - doc_G: ✓ Found at position 7
3. All 4 relevant docs are in top 50
4. R@50 = 4 / 4 = **1.00** (100%)

**Interpretation**:

- Found ALL relevant documents
- No relevant document was missed or ranked below position 50
- Perfect recall for this query

**Example of lower recall**: If doc_G appeared at position 75 instead:

- R@50 = 3 / 4 = 0.75 (75%)
- We missed one relevant document

**Why it matters**: Low recall means users miss important information even if they scroll through 50 results.

---

#### 3. MRR (Mean Reciprocal Rank) Calculation

**Definition**: How quickly does the user find the FIRST relevant result?

**Formula**: `MRR = 1 / (Position of first relevant document)`

**Purpose**: Measures immediate satisfaction. Users want relevant results immediately.

**Step-by-step**:

1. Scan results from top to find first relevant document
2. Position 1: doc_A → Relevant! ✓
3. First relevant position = 1
4. MRR = 1 / 1 = **1.00** (100%)

**Interpretation**:

- Perfect score (1.00) because first result is relevant
- User gets what they need immediately

**Examples with different first positions**:

- First relevant at position 1: MRR = 1/1 = 1.000 (perfect)
- First relevant at position 2: MRR = 1/2 = 0.500
- First relevant at position 3: MRR = 1/3 = 0.333
- First relevant at position 5: MRR = 1/5 = 0.200
- First relevant at position 10: MRR = 1/10 = 0.100

**Why it matters**: If first relevant result is at position 20, MRR = 0.05 (very poor). Users may give up before scrolling that far.

---

#### 4. nDCG@10 (Normalized Discounted Cumulative Gain) Calculation

**Definition**: Measures ranking quality by rewarding relevant documents at TOP positions more heavily.

**Key Concept**: A relevant document at position 1 is MORE valuable than at position 10 because users look at top results first.

**Formula**:

```
DCG@10 = Σ(i=1 to 10) relevance(i) / log₂(i + 1)
nDCG@10 = DCG@10 / IDCG@10
```

Where:

- `relevance(i)` = 1 if document at position i is relevant, 0 otherwise
- `log₂(i + 1)` = discount factor (makes lower positions worth less)
- `IDCG` = Ideal DCG (best possible DCG with perfect ranking)

---

**Step 1: Calculate DCG (Discounted Cumulative Gain)**

Go through each position and calculate: `relevance / log₂(position + 1)`

| Position | Doc   | Relevant? | Relevance | log₂(pos+1)      | Contribution    |
| -------- | ----- | --------- | --------- | ---------------- | --------------- |
| 1        | doc_A | ✓ Yes     | 1         | log₂(2) = 1.000  | 1/1.000 = 1.000 |
| 2        | doc_B | ✗ No      | 0         | log₂(3) = 1.585  | 0/1.585 = 0.000 |
| 3        | doc_C | ✓ Yes     | 1         | log₂(4) = 2.000  | 1/2.000 = 0.500 |
| 4        | doc_D | ✗ No      | 0         | log₂(5) = 2.322  | 0/2.322 = 0.000 |
| 5        | doc_E | ✗ No      | 0         | log₂(6) = 2.585  | 0/2.585 = 0.000 |
| 6        | doc_F | ✓ Yes     | 1         | log₂(7) = 2.807  | 1/2.807 = 0.356 |
| 7        | doc_G | ✓ Yes     | 1         | log₂(8) = 3.000  | 1/3.000 = 0.333 |
| 8        | doc_H | ✗ No      | 0         | log₂(9) = 3.170  | 0/3.170 = 0.000 |
| 9        | doc_I | ✗ No      | 0         | log₂(10) = 3.322 | 0/3.322 = 0.000 |
| 10       | doc_J | ✗ No      | 0         | log₂(11) = 3.459 | 0/3.459 = 0.000 |

**DCG@10 = 1.000 + 0.000 + 0.500 + 0.000 + 0.000 + 0.356 + 0.333 + 0.000 + 0.000 + 0.000**

**DCG@10 = 2.189**

---

**Step 2: Calculate IDCG (Ideal DCG)**

What if we had PERFECT ranking? All relevant docs at the top:

| Position | Ideal Doc | Relevant? | Relevance | log₂(pos+1)     | Contribution    |
| -------- | --------- | --------- | --------- | --------------- | --------------- |
| 1        | doc_A     | ✓ Yes     | 1         | log₂(2) = 1.000 | 1/1.000 = 1.000 |
| 2        | doc_C     | ✓ Yes     | 1         | log₂(3) = 1.585 | 1/1.585 = 0.631 |
| 3        | doc_F     | ✓ Yes     | 1         | log₂(4) = 2.000 | 1/2.000 = 0.500 |
| 4        | doc_G     | ✓ Yes     | 1         | log₂(5) = 2.322 | 1/2.322 = 0.431 |
| 5-10     | (others)  | ✗ No      | 0         | -               | 0.000           |

**IDCG@10 = 1.000 + 0.631 + 0.500 + 0.431 = 2.562**

---

**Step 3: Normalize**

**nDCG@10 = DCG@10 / IDCG@10 = 2.189 / 2.562 = 0.854** (85.4%)

**Interpretation**:

- 85.4% of ideal ranking quality achieved
- Good but not perfect (perfect would be 1.00)
- Lost some value because doc_F and doc_G are at positions 6-7 instead of 2-3

**Why not 1.00?**: If we had perfect ranking, doc_C would be at position 2 (contributing 0.631 instead of 0.500), and doc_F/G would be at positions 3-4 (contributing more).

**Comparison scenarios**:

- **Current ranking**: doc_A(1), doc_C(3), doc_F(6), doc_G(7) → nDCG = 0.854
- **Perfect ranking**: doc_A(1), doc_C(2), doc_F(3), doc_G(4) → nDCG = 1.000
- **Poor ranking**: doc_A(5), doc_C(8), doc_F(15), doc_G(20) → nDCG ≈ 0.45

---

#### Summary Table

| Metric      | Value | Meaning                                  | Quality                       |
| ----------- | ----- | ---------------------------------------- | ----------------------------- |
| **P@10**    | 0.40  | 4 out of 10 top results are relevant     | Moderate - could be better    |
| **R@50**    | 1.00  | Found all 4 relevant documents in top 50 | Perfect - no document missed  |
| **MRR**     | 1.00  | First result is relevant                 | Perfect - immediate answer    |
| **nDCG@10** | 0.854 | 85% of ideal ranking quality             | Good - relevant docs near top |

#### Overall System Assessment

**Strengths**:

- Excellent recall (1.00) - doesn't miss relevant documents
- Perfect MRR (1.00) - user gets answer immediately
- Good ranking quality (nDCG 0.854)

**Weaknesses**:

- Moderate precision (0.40) - 6 out of 10 results are irrelevant
- Could improve by filtering out doc_B, doc_D, doc_E (traffic, politics, weather)

**Actionable Improvements**:

1. Improve query understanding to filter topic-irrelevant results
2. Boost exact phrase matches (e.g., "air pollution" / "বায়ু দূষণ")
3. Add domain filtering (environment/pollution category)
4. Fine-tune fusion weights to reduce irrelevant results

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
