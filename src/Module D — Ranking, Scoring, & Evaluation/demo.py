"""
Sample Evaluation Script

Demonstrates complete evaluation workflow for Module D.
Run this to see how all components work together.
"""

import json
from ranking_scorer import RankingScorer, ExecutionMetrics
from evaluation_metrics import EvaluationMetrics
from error_analysis import ErrorAnalyzer
from relevance_labeling import RelevanceLabeler


def demo_ranking_and_scoring():
    """Demonstrate ranking and scoring functionality."""
    print("\n" + "=" * 80)
    print("DEMO 1: RANKING AND SCORING")
    print("=" * 80)

    # Simulated retrieval results
    results = {
        "doc_1": {
            "title": "Climate Crisis in Bangladesh",
            "language": "english",
            "source": "BBC News",
            "url": "https://bbc.com/climate-bangladesh",
            "body_preview": "Bangladesh faces severe climate challenges...",
            "score": 0.95,
        },
        "doc_2": {
            "title": "Weather Forecast for Today",
            "language": "english",
            "source": "Weather.com",
            "url": "https://weather.com/bd",
            "body_preview": "Dhaka weather: 28°C, partly cloudy...",
            "score": 0.65,
        },
        "doc_3": {
            "title": "Rising Sea Levels Threaten Coastal Areas",
            "language": "english",
            "source": "CNN",
            "url": "https://cnn.com/sea-levels",
            "body_preview": "Global warming causes sea level rise affecting Bangladesh...",
            "score": 0.88,
        },
    }

    # Initialize scorer
    scorer = RankingScorer(low_confidence_threshold=0.20)

    # Rank documents
    ranked_results, top_confidence = scorer.rank_documents(
        results, method="hybrid", top_k=10, normalize_method="minmax"
    )

    # Display results
    print(scorer.format_results(ranked_results))

    # Check confidence
    warning = scorer.generate_confidence_warning(top_confidence, "climate change Bangladesh")
    if warning:
        print(warning)
    else:
        print(f"✓ Confidence score {top_confidence:.2%} is above threshold")


def demo_evaluation_metrics():
    """Demonstrate evaluation metrics calculation."""
    print("\n" + "=" * 80)
    print("DEMO 2: EVALUATION METRICS")
    print("=" * 80)

    # Ground truth
    relevant_docs = ["doc_1", "doc_3", "doc_5"]

    # Retrieved results (in rank order)
    retrieved_docs = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]

    # Calculate metrics
    metrics = EvaluationMetrics.evaluate_query(
        relevant_docs, retrieved_docs, query_id="demo_query", k=10
    )

    print("\nMetrics for query: 'climate change Bangladesh'")
    print(f"Relevant docs: {relevant_docs}")
    print(f"Retrieved docs: {retrieved_docs}")
    print(EvaluationMetrics.format_metrics(metrics))

    # Interpretation
    print("\n📊 INTERPRETATION:")
    print(f"  P@10: {metrics['precision_at_10']:.2f} - Found {int(metrics['precision_at_10']*10)}/10 relevant docs")
    print(f"  R@50: {metrics['recall_at_50']:.2f} - Found {int(metrics['recall_at_50']*len(relevant_docs))}/{len(relevant_docs)} relevant docs")
    print(f"  nDCG@10: {metrics['ndcg_at_10']:.2f} - Ranking quality (1.0 = perfect)")
    print(f"  MRR: {metrics['mrr']:.2f} - First relevant at rank {1/metrics['mrr'] if metrics['mrr'] > 0 else 'N/A'}")


def demo_batch_evaluation():
    """Demonstrate batch evaluation."""
    print("\n" + "=" * 80)
    print("DEMO 3: BATCH EVALUATION")
    print("=" * 80)

    batch_queries = {
        "q001": {
            "relevant": ["doc_1", "doc_3"],
            "retrieved": ["doc_1", "doc_2", "doc_3", "doc_4"],
        },
        "q002": {
            "relevant": ["doc_5", "doc_6"],
            "retrieved": ["doc_5", "doc_6", "doc_7"],
        },
        "q003": {
            "relevant": ["doc_8"],
            "retrieved": ["doc_8", "doc_9", "doc_10"],
        },
    }

    per_query, summary = EvaluationMetrics.evaluate_batch(batch_queries, k=10)

    print("\nPer-Query Results:")
    for query_id, metrics in per_query.items():
        print(f"  {query_id}: P@10={metrics['precision_at_10']:.2f}, nDCG@10={metrics['ndcg_at_10']:.2f}")

    print("\n📊 AGGREGATE RESULTS:")
    print(f"  Mean P@10: {summary['mean_precision_at_10']:.4f}")
    print(f"  Mean Recall@50: {summary['mean_recall_at_50']:.4f}")
    print(f"  Mean nDCG@10: {summary['mean_ndcg_at_10']:.4f}")
    print(f"  Mean Average Precision: {summary['mean_average_precision']:.4f}")
    print(f"  Mean Reciprocal Rank: {summary['mean_reciprocal_rank']:.4f}")
    print(f"  Queries Evaluated: {summary['num_queries']}")

    # Check against targets
    print("\n✓ TARGET ACHIEVEMENT:")
    targets = {
        "P@10": (summary["mean_precision_at_10"], 0.6),
        "R@50": (summary["mean_recall_at_50"], 0.5),
        "nDCG@10": (summary["mean_ndcg_at_10"], 0.5),
        "MRR": (summary["mean_reciprocal_rank"], 0.4),
    }

    for metric_name, (actual, target) in targets.items():
        status = "✓ PASS" if actual >= target else "✗ FAIL"
        print(f"  {status} {metric_name}: {actual:.4f} (target: ≥{target})")


def demo_error_analysis():
    """Demonstrate error analysis functionality."""
    print("\n" + "=" * 80)
    print("DEMO 4: ERROR ANALYSIS")
    print("=" * 80)

    analyzer = ErrorAnalyzer()

    # Add sample errors
    analyzer.add_translation_failure(
        query_id="q001",
        query_text="chair furniture",
        query_language="english",
        original_query="চেয়ার",
        mistranslated_query="Chairman",
        expected_docs=["furniture_doc_1", "furniture_doc_2"],
        retrieved_docs=["political_doc_1", "political_doc_2"],
        example="Query intended furniture items but got political results due to mistranslation",
    )

    analyzer.add_ner_mismatch(
        query_id="q002",
        query_text="news from Dhaka",
        query_language="english",
        entity_in_query="ঢাকা",
        entity_in_docs="Dhaka",
        expected_docs=["dhaka_news_1", "dhaka_news_2"],
        retrieved_docs=["other_news_1"],
        example="Entity 'Dhaka' not recognized when transliterated from Bangla script",
    )

    analyzer.add_code_switching_issue(
        query_id="q003",
        query_text="আমরা COVID-19 বিরুদ্ধে লড়াই করছি",
        mixed_components=["Bangla", "English (COVID-19)"],
        expected_docs=["covid_bn_1", "covid_bn_2"],
        retrieved_docs=["unrelated_doc_1"],
        example="Language detection failed on code-switched query",
    )

    analyzer.add_semantic_vs_lexical(
        query_id="q004",
        query_text="education system",
        query_language="english",
        winner="semantic",
        lexical_results=[],
        semantic_results=["school_doc_1", "university_doc_1"],
        example="Semantic model understood 'education' ≈ 'school' but BM25 found nothing",
    )

    # Print reports
    print(analyzer.format_error_summary_table())
    print(analyzer.format_error_report())


def demo_relevance_labeling():
    """Demonstrate relevance labeling functionality."""
    print("\n" + "=" * 80)
    print("DEMO 5: RELEVANCE LABELING")
    print("=" * 80)

    labeler = RelevanceLabeler()

    # Add sample labels
    labels_data = [
        ("q001", "climate change", "doc_1", "Climate Crisis in Bangladesh", True, 3),
        ("q001", "climate change", "doc_2", "Sports News", False, 3),
        ("q001", "climate change", "doc_3", "Rising Sea Levels", True, 2),
        ("q002", "education", "doc_5", "School System", True, 3),
        ("q002", "education", "doc_6", "University Rankings", True, 2),
        ("q002", "education", "doc_7", "Political News", False, 3),
    ]

    for query_id, query_text, doc_id, doc_title, relevant, confidence in labels_data:
        labeler.add_label(
            query_id=query_id,
            query_text=query_text,
            doc_id=doc_id,
            doc_title=doc_title,
            doc_url=f"https://example.com/{doc_id}",
            language="english",
            relevant=relevant,
            confidence=confidence,
            annotator="demo_annotator",
            notes="Sample labeling for demo",
        )

    # Print statistics
    print(labeler.format_statistics())

    # Show relevant docs for each query
    print("\n📝 RELEVANT DOCUMENTS BY QUERY:")
    for query_id in ["q001", "q002"]:
        relevant = labeler.get_relevant_docs_for_query(query_id)
        print(f"  Query {query_id}: {relevant}")

    # Demonstrate CSV save/load
    print("\n💾 SAVING AND LOADING:")
    labeler.save_to_csv("demo_labels.csv")
    print("  ✓ Labels saved to demo_labels.csv")

    labeler2 = RelevanceLabeler()
    labeler2.load_from_csv("demo_labels.csv")
    print("  ✓ Labels loaded from demo_labels.csv")
    print(f"  ✓ Loaded {len(labeler2.labels)} labels")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "MODULE D - RANKING, SCORING, & EVALUATION DEMO".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        demo_ranking_and_scoring()
        demo_evaluation_metrics()
        demo_batch_evaluation()
        demo_error_analysis()
        demo_relevance_labeling()

        print("\n" + "=" * 80)
        print("✓ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Review the README.md for complete documentation")
        print("  2. Check QUICK_START.md for usage tutorials")
        print("  3. Run: python evaluate.py")
        print("  4. Customize test_queries.json with your data")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
