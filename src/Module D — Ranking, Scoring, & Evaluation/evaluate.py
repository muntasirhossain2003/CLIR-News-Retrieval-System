"""
Main Evaluation Script

Comprehensive evaluation pipeline for CLIR system.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add parent directories to path
script_dir = Path(__file__).parent
module_c_path = script_dir.parent / "Module C — Retrieval Models"
module_d_path = script_dir

sys.path.insert(0, str(module_c_path))
sys.path.insert(0, str(module_d_path))

from ranking_scorer import RankingScorer, ExecutionMetrics
from evaluation_metrics import EvaluationMetrics
from error_analysis import ErrorAnalyzer
from relevance_labeling import RelevanceLabeler


def load_test_queries(filepath: str) -> Dict[str, Dict]:
    """
    Load test queries with relevance labels from JSON.

    Format:
    {
        "q001": {
            "query_text": "climate change Bangladesh",
            "language": "english",
            "relevant_docs": ["doc_123", "doc_456"],
            "retrieved_docs": ["doc_123", "doc_789", "doc_456"]  # in rank order
        }
    }
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_single_query(
    query_id: str,
    query_text: str,
    relevant_docs: List[str],
    retrieved_docs: List[str],
    method: str = "hybrid",
) -> Dict:
    """
    Evaluate a single query comprehensively.
    """
    # Get metrics
    metrics = EvaluationMetrics.evaluate_query(
        relevant_docs, retrieved_docs, query_id, k=10
    )

    # Add query info
    metrics["query_text"] = query_text
    metrics["method"] = method
    metrics["num_relevant"] = len(relevant_docs)
    metrics["num_retrieved"] = len(retrieved_docs)
    metrics["recall_achieved"] = len(set(relevant_docs) & set(retrieved_docs[:50]))

    return metrics


def evaluate_batch(test_queries: Dict[str, Dict], methods: List[str] = None) -> Dict:
    """
    Evaluate system on batch of test queries.
    """
    if methods is None:
        methods = ["hybrid"]

    results = {
        "test_queries": len(test_queries),
        "methods": methods,
        "per_query_results": {},
        "aggregate_metrics": {},
    }

    for method in methods:
        print(f"\n{'='*80}")
        print(f"Evaluating with method: {method.upper()}")
        print(f"{'='*80}")

        batch_queries = {}

        for query_id, query_data in test_queries.items():
            query_text = query_data.get("query_text", "")
            relevant_docs = query_data.get("relevant_docs", [])
            retrieved_docs = query_data.get("retrieved_docs", [])

            batch_queries[query_id] = {
                "relevant": relevant_docs,
                "retrieved": retrieved_docs,
            }

            # Single query metrics
            if query_id not in results["per_query_results"]:
                results["per_query_results"][query_id] = {}

            query_metrics = evaluate_single_query(
                query_id, query_text, relevant_docs, retrieved_docs, method
            )
            results["per_query_results"][query_id][method] = query_metrics

        # Batch metrics
        per_query, summary = EvaluationMetrics.evaluate_batch(batch_queries, k=10)

        results["aggregate_metrics"][method] = summary

        # Print summary
        print(f"\n📊 Results for {method}:")
        print(f"  Precision@10: {summary['mean_precision_at_10']:.4f}")
        print(f"  Precision@5:  {summary['mean_precision_at_5']:.4f}")
        print(f"  Recall@50:    {summary['mean_recall_at_50']:.4f}")
        print(f"  nDCG@10:      {summary['mean_ndcg_at_10']:.4f}")
        print(f"  MAP:          {summary['mean_average_precision']:.4f}")
        print(f"  MRR:          {summary['mean_reciprocal_rank']:.4f}")

    return results


def print_evaluation_report(results: Dict):
    """Print comprehensive evaluation report."""
    output = []

    output.append("\n" + "=" * 80)
    output.append("CLIR SYSTEM EVALUATION REPORT")
    output.append("=" * 80)

    output.append(f"\nTest Queries: {results['test_queries']}")
    output.append(f"Methods Evaluated: {', '.join(results['methods'])}\n")

    # Comparison table
    output.append("OVERALL PERFORMANCE:")
    output.append("─" * 80)
    output.append(
        f"{'Metric':<20} {' | '.join(results['methods']):<50}"
    )
    output.append("─" * 80)

    metrics_to_compare = [
        "mean_precision_at_10",
        "mean_precision_at_5",
        "mean_recall_at_50",
        "mean_recall_at_10",
        "mean_ndcg_at_10",
        "mean_average_precision",
        "mean_reciprocal_rank",
    ]

    for metric in metrics_to_compare:
        values = []
        for method in results["methods"]:
            val = results["aggregate_metrics"][method].get(metric, 0)
            values.append(f"{val:.4f}")

        row = f"{metric:<20} {' | '.join(values)}"
        output.append(row)

    output.append("─" * 80)

    # Interpretation
    output.append("\nINTERPRETATION:")
    output.append("─" * 80)

    for method in results["methods"]:
        summary = results["aggregate_metrics"][method]

        p10 = summary["mean_precision_at_10"]
        r50 = summary["mean_recall_at_50"]
        ndcg = summary["mean_ndcg_at_10"]
        mrr = summary["mean_reciprocal_rank"]

        status = []
        if p10 >= 0.6:
            status.append("✓ P@10 meets target (≥0.6)")
        else:
            status.append(f"✗ P@10 below target: {p10:.2f}")

        if r50 >= 0.5:
            status.append("✓ R@50 meets target (≥0.5)")
        else:
            status.append(f"✗ R@50 below target: {r50:.2f}")

        if ndcg >= 0.5:
            status.append("✓ nDCG@10 meets target (≥0.5)")
        else:
            status.append(f"✗ nDCG@10 below target: {ndcg:.2f}")

        if mrr >= 0.4:
            status.append("✓ MRR meets target (≥0.4)")
        else:
            status.append(f"✗ MRR below target: {mrr:.2f}")

        output.append(f"\n{method.upper()}:")
        for s in status:
            output.append(f"  {s}")

    output.append("\n" + "=" * 80)

    return "\n".join(output)


def save_evaluation_results(results: Dict, output_path: str):
    """Save evaluation results to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIR system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with default test queries
  python evaluate.py

  # Evaluate with custom queries
  python evaluate.py --queries custom_queries.json

  # Evaluate with multiple methods
  python evaluate.py --methods hybrid bm25 semantic

  # Generate labeling template
  python evaluate.py --create-template
        """,
    )

    parser.add_argument(
        "--queries",
        default="test_queries.json",
        help="Path to test queries JSON file",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["hybrid"],
        help="Retrieval methods to evaluate",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Create sample labeling CSV template",
    )

    args = parser.parse_args()

    # Create labeling template if requested
    if args.create_template:
        labeler = RelevanceLabeler()
        labeler.create_sample_labeling_csv("sample_labeling_template.csv")
        return

    # Check if test queries exist
    if not os.path.exists(args.queries):
        print(f"\n❌ Error: Test queries file not found: {args.queries}")
        print("\nPlease create a test queries JSON file with the following format:")
        print("""
{
    "q001": {
        "query_text": "climate change Bangladesh",
        "language": "english",
        "relevant_docs": ["doc_123", "doc_456"],
        "retrieved_docs": ["doc_123", "doc_789", "doc_456"]
    }
}
        """)
        return

    # Load test queries
    print(f"\nLoading test queries from {args.queries}...")
    test_queries = load_test_queries(args.queries)

    # Run evaluation
    print(f"\nEvaluating {len(test_queries)} queries with methods: {', '.join(args.methods)}")
    results = evaluate_batch(test_queries, args.methods)

    # Print report
    report = print_evaluation_report(results)
    print(report)

    # Save results
    save_evaluation_results(results, args.output)


if __name__ == "__main__":
    main()
