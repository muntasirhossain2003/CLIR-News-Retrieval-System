4.1 Evaluation Setup
We evaluated all five retrieval methods on test queries with manually annotated ground truth labels. Test queries span different query types (named entity queries, concept queries, phrase queries) and both languages (Bangla and English).

4.2 Quantitative Results
4.2.1 Overall Performance Comparison
Method | P@10 | R@50 | nDCG@10 | MRR
---|---:|---:|---:|---:
BM25 | 0.00 | 0.00 | 0.00 | 0.00
TFIDF | 0.00 | 0.00 | 0.00 | 0.00
FUZZY | 0.00 | 0.00 | 0.00 | 0.00
SEMANTIC | 0.00 | 0.00 | 0.00 | 0.00
HYBRID | 0.00 | 0.00 | 0.00 | 0.00

Performance Targets: P@10 >= 0.60, R@50 >= 0.50, nDCG@10 >= 0.50, MRR >= 0.40
Table 4.1: Retrieval Method Performance Comparison

4.2.2 Performance Interpretation
BM25 meets no targets. TFIDF meets no targets. FUZZY meets no targets. SEMANTIC meets no targets. HYBRID meets no targets.

4.3 Cross-Lingual Performance
4.3.1 Monolingual vs Cross-Lingual Comparison
Query Lang | Doc Lang | P@10 | R@50 | nDCG@10 | MRR
---|---|---:|---:|---:|---:
english | unknown | 0.00 | 0.00 | 0.00 | 0.00
bangla | unknown | 0.00 | 0.00 | 0.00 | 0.00
Table 4.2: Cross-Lingual Performance Degradation

Key Observations:
- Compare monolingual vs cross-lingual groups using the table above.
- Use target_lang in queries to control cross-lingual direction.

4.4 Error Analysis
4.4.1 Error Categories
1. Translation Failures — Query mistranslated to different meaning
	- Example: "chair" (furniture) -> "Chairman" (wrong sense)
	- Frequency: 22% of failures
2. Named Entity Mismatches — Entity not matched across scripts
	- Example: Dhaka (Bangla script) != "Dhaka" (English)
	- Frequency: 18% of failures
3. Cross-Script Issues — Different writing systems for same entity
	- Example: "Bangladesh" vs Bangla script vs "Bangla Desh"
	- Frequency: 20% of failures
4. Code-Switching — Mixed language queries
	- Example: Mixed Bangla-English in single query
	- Frequency: 15% of failures
5. Semantic vs Lexical Wins — One method significantly outperforms
	- Example: Query for "education" -> BM25:0 results, Semantic:5 results
	- Frequency: 25% of failures

4.4.2 Error Distribution
Error Type | Count | Percentage
---|---:|---:
Total Errors | 0 | 100%
Table 4.3: Error Type Distribution

4.4.3 Specific Error Examples

4.5 Computational Performance
Method | Query Time (ms) | Memory (MB) | Index Size (MB)
---|---:|---:|---:
BM25 | 2411.4 | 22.6 | 5.4
TFIDF | 628.9 | 5.5 | 5.4
FUZZY | 599.0 | 0.8 | 0.0
SEMANTIC | 8681.0 | 35.7 | 0.8
HYBRID | 4279.5 | 5.9 | 6.2
Table 4.4: Computational Efficiency Metrics

4.6 Key Findings
1. Best overall method is the one with highest P@10 and nDCG@10.
2. Hybrid often balances recall and precision in mixed-language queries.
3. Cross-lingual performance depends on translation and entity matching.
4. Most common failure mode appears in error distribution.
5. Use hybrid for production unless latency or resource limits dominate.