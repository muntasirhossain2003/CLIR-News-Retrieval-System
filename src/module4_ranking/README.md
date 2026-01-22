# Module 4: Ranking and Evaluation

This module handles result fusion, score normalization, and evaluation metrics for the CLIR system.

## Features

### Score Normalization

- **Min-Max Normalization**: Scales scores to [0, 1] range
- **Division by zero handling**: Safe normalization even with identical scores

### Result Fusion

- **Weighted Fusion**: Combines lexical and semantic results
- **Formula**: `FinalScore = (alpha × Semantic) + ((1-alpha) × Lexical)`
- **Low Confidence Warning**: Flags results with top score < 0.2

### Evaluation Metrics

- **Precision@10**: Relevant documents in top 10 results
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
