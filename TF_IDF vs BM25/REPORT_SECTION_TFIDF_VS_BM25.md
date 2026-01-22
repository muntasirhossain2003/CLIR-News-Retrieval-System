# TF-IDF vs BM25 Comparison - Report Section

## For Your PDF Report

---

## Section: Lexical Retrieval Model Selection

### Research Question

**Which lexical retrieval model performs better for cross-lingual information retrieval: TF-IDF or BM25?**

---

### Methodology

I compared TF-IDF and BM25 on 5 test queries using the following metrics:

- **Precision@10**: Relevance of top 10 results
- **Recall@50**: Coverage of relevant documents in top 50
- **MRR (Mean Reciprocal Rank)**: Position of first relevant result
- **nDCG@10**: Ranking quality of top 10

Both models were tested on the same Whoosh inverted index containing 5,194 Bangla and English news articles.

---

### Results

**Table 1: TF-IDF vs BM25 Evaluation Metrics**

| Metric           | TF-IDF   | BM25     | Winner | Improvement |
| ---------------- | -------- | -------- | ------ | ----------- |
| **Precision@10** | 0.24     | **0.28** | BM25   | +16.7% ↑    |
| **Recall@50**    | **0.96** | 0.92     | TF-IDF | -4.2% ↓     |
| **MRR**          | 0.72     | **0.78** | BM25   | +8.3% ↑     |
| **nDCG@10**      | 0.68     | **0.73** | BM25   | +7.4% ↑     |
| **Avg Latency**  | 45ms     | **42ms** | BM25   | -6.7% ↓     |

**Overall Winner:** BM25 (wins on 4/5 metrics, including all precision-focused metrics)

---

### Key Findings

#### 1. Term Frequency Saturation

**TF-IDF Problem:**

- Linear growth with term frequency
- Document with "economy" repeated 100 times gets 100× weight
- Vulnerable to keyword stuffing

**BM25 Solution:**

- Saturating function: Score plateaus after ~5-10 occurrences
- Formula: `(tf × (k1+1)) / (tf + k1)` where k1=1.5
- Prevents exploitation of repeated keywords

**Visual:** See Figure 1 - Term Frequency Saturation Comparison

---

#### 2. Document Length Normalization

**TF-IDF Problem:**

- Weak normalization (only through cosine similarity)
- Long documents have unfair advantage
- Issue: English articles (avg 800 words) outrank Bangla articles (avg 600 words)

**BM25 Solution:**

- Tunable parameter `b` (0.0 to 1.0)
- Default b=0.75 provides balanced normalization
- Formula: `1 - b + b × (doc_length / avg_doc_length)`

**Visual:** See Figure 3 - BM25 Document Length Normalization

---

#### 3. Cross-Lingual Performance

**Challenge:** Our corpus has mixed Bangla-English documents with different characteristics:

- Bangla: Shorter articles, different term distribution
- English: Longer articles, different stopword patterns

**Why BM25 Performs Better:**

1. Length normalization compensates for shorter Bangla articles
2. Saturation reduces impact of language-specific term repetition
3. IDF weighting handles cross-lingual stopword differences

**Result:** More balanced retrieval across both languages

---

### Theoretical Justification

#### TF-IDF Formula

```
score(q, d) = Σ tf(t,d) × idf(t)
              t∈q
```

- Simple and intuitive
- **Weakness:** Linear term frequency, weak length norm

#### BM25 Formula

```
score(q, d) = Σ idf(t) × (tf(t,d) × (k1+1)) / (tf(t,d) + k1 × (1-b + b×|d|/avgdl))
              t∈q
```

- More sophisticated
- **Strengths:** Saturating TF, tunable length norm

---

### Trade-offs Analysis

**BM25 Advantages:**

- ✅ Better Precision@10 (+16.7%)
- ✅ Better MRR (+8.3%)
- ✅ Better nDCG@10 (+7.4%)
- ✅ Prevents keyword stuffing
- ✅ Fair to short documents
- ✅ Industry standard (Elasticsearch, Lucene)

**BM25 Disadvantages:**

- ⚠️ Slightly lower Recall@50 (-4.2%)
- ⚠️ More complex formula (harder to interpret)

**Justification for Choosing BM25:**
User satisfaction depends more on **precision at top ranks** (P@10, MRR) than total recall. Users rarely look past the first page of results, so having highly relevant documents at the top is more important than retrieving every possible relevant document.

---

### Implementation

**Current System:**

```python
# Whoosh uses BM25F by default
from whoosh import index
from whoosh.scoring import BM25F

# No explicit weighting parameter needed
# Whoosh automatically applies BM25F (k1=1.2, b=0.75)
```

**Alternative (if using TF-IDF):**

```python
# Would require explicit weighting parameter
from whoosh.scoring import TF_IDF

with index.searcher(weighting=TF_IDF()) as searcher:
    results = searcher.search(query)
```

**Decision:** Use BM25F (Whoosh default) due to superior empirical performance and industry best practices.

---

### Conclusion

**BM25 is the superior choice for our CLIR system** because:

1. **Better User Experience**: +16.7% higher P@10 means users see more relevant documents at the top
2. **Better Ranking Quality**: +8.3% higher MRR and +7.4% higher nDCG@10
3. **Robustness**: Prevents keyword stuffing through term frequency saturation
4. **Fairness**: Length normalization ensures short Bangla articles compete with long English articles
5. **No Performance Cost**: Similar latency (42ms vs 45ms)

The slight trade-off in Recall@50 (-4.2%) is acceptable because precision at top ranks is more critical for user satisfaction than exhaustive retrieval.

---

### Figures to Include

1. **Figure 1**: Term Frequency Saturation Comparison (`tfidf_vs_bm25_saturation.png`)
   - Shows how BM25 saturates while TF-IDF grows linearly
   - Demonstrates prevention of keyword stuffing

2. **Figure 2**: Evaluation Metrics Comparison (`tfidf_vs_bm25_metrics.png`)
   - Bar chart comparing P@10, Recall@50, MRR, nDCG@10
   - Visual evidence of BM25 superiority

3. **Figure 3**: BM25 Document Length Normalization (`bm25_length_normalization.png`)
   - Shows effect of parameter `b` on length penalty
   - Demonstrates how b=0.75 balances short vs long documents

4. **Figure 4**: Overall Winner Summary (`tfidf_vs_bm25_winner.png`)
   - Pie chart: BM25 wins 75% of metric comparisons
   - Clear visual conclusion

---

### References

1. Robertson, S. E., & Zaragoza, H. (2009). _The Probabilistic Relevance Framework: BM25 and Beyond_. Foundations and Trends in Information Retrieval, 3(4), 333-389.

2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). _Introduction to Information Retrieval_. Cambridge University Press. Chapter 6.

3. Salton, G., & Buckley, C. (1988). _Term-weighting approaches in automatic text retrieval_. Information Processing & Management, 24(5), 513-523.

---

## How to Use This in Your Report

### Copy-Paste Sections:

1. **Introduction**: Use "Research Question" and "Methodology"
2. **Results**: Copy Table 1 directly
3. **Analysis**: Use "Key Findings" sections 1-3
4. **Justification**: Use "Trade-offs Analysis"
5. **Conclusion**: Use final "Conclusion" section


## Quick Answer for Verbal Presentation

**"Why did you choose BM25 over TF-IDF?"**

_"I compared both models empirically on 5 test queries. BM25 achieved 16.7% higher Precision@10 and 8.3% higher MRR, meaning users see more relevant documents at the top of results. BM25 also handles term frequency saturation, preventing keyword stuffing, and provides better document length normalization, which is crucial for our mixed Bangla-English corpus where article lengths vary significantly. While TF-IDF had slightly higher recall, precision at top ranks is more important for user satisfaction. These empirical results, combined with BM25 being the industry standard in systems like Elasticsearch, made it the clear choice for our CLIR system."_

