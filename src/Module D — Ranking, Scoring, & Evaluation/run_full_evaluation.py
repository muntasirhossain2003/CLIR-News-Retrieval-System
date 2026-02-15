"""
Full evaluation runner for Module D.

Generates:
- Overall metrics table for multiple methods
- Cross-lingual breakdown by query/doc language
- Error distribution (optional input)
- Computational performance summary
- Markdown report draft for thesis section
"""

import argparse
import csv
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODULE_C_PATH = SCRIPT_DIR.parent / "Module C — Retrieval Models"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
INDEX_DIR = PROJECT_ROOT / "indexes"

sys.path.insert(0, str(MODULE_C_PATH))
sys.path.insert(0, str(SCRIPT_DIR))

from retrieval_pipeline import RetrievalPipeline
from evaluation_metrics import EvaluationMetrics
from error_analysis import ErrorAnalyzer


def load_metadata() -> Dict[str, Dict[str, str]]:
    doc_map: Dict[str, Dict[str, str]] = {}
    if not METADATA_PATH.exists():
        return doc_map

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row.get("filename", "").replace(".json", "")
            if not doc_id:
                continue
            doc_map[doc_id] = {
                "language": row.get("language", ""),
                "title": row.get("title", ""),
                "source": row.get("source", ""),
                "url": row.get("url", ""),
            }
    return doc_map


def warn_missing_relevant_docs(
    queries: Dict[str, Dict[str, Any]],
    doc_meta: Dict[str, Dict[str, str]],
) -> None:
    missing = []
    for qid, q in queries.items():
        for doc_id in q.get("relevant_docs", []):
            if doc_id not in doc_meta:
                missing.append((qid, doc_id))

    if not missing:
        return

    print(
        f"Warning: {len(missing)} relevant doc ids do not exist in the corpus. "
        "Metrics will be zero for those queries."
    )
    sample = ", ".join([f"{qid}:{doc_id}" for qid, doc_id in missing[:5]])
    print(f"Missing samples: {sample}")


def load_queries(query_path: Path) -> Dict[str, Dict[str, Any]]:
    with query_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        queries = {item["query_id"]: item for item in data}
    else:
        queries = data

    for qid, q in queries.items():
        q.setdefault("query_id", qid)
        q.setdefault("relevant_docs", [])
        q.setdefault("retrieved_docs", [])
        q.setdefault("language", "unknown")
    return queries


def load_relevant_from_labels(csv_path: Path) -> Dict[str, List[str]]:
    relevant_by_query: Dict[str, List[str]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row.get("query_id", "")
            doc_id = row.get("doc_id", "")
            relevant_raw = row.get("relevant", "").strip().lower()
            relevant = relevant_raw in ["true", "yes", "1"]
            if query_id and doc_id and relevant:
                relevant_by_query.setdefault(query_id, []).append(doc_id)
    return relevant_by_query


def write_labeling_pool_csv(
    output_path: Path,
    queries: Dict[str, Dict[str, Any]],
    doc_meta: Dict[str, Dict[str, str]],
    methods: List[str],
    pool_size: int,
) -> None:
    headers = [
        "query_id",
        "query_text",
        "doc_id",
        "doc_title",
        "doc_url",
        "language",
        "method",
        "rank",
        "relevant",
        "confidence",
        "annotator",
        "notes",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for qid, q in queries.items():
            query_text = q.get("query_text", "")
            retrieved_by_method = q.get("retrieved_docs_by_method", {})

            for method in methods:
                doc_ids = retrieved_by_method.get(method, [])[:pool_size]
                for rank, doc_id in enumerate(doc_ids, 1):
                    meta = doc_meta.get(doc_id, {})
                    writer.writerow(
                        {
                            "query_id": qid,
                            "query_text": query_text,
                            "doc_id": doc_id,
                            "doc_title": meta.get("title", ""),
                            "doc_url": meta.get("url", ""),
                            "language": meta.get("language", ""),
                            "method": method,
                            "rank": rank,
                            "relevant": "",
                            "confidence": "",
                            "annotator": "",
                            "notes": "",
                        }
                    )

    print(f"Wrote labeling pool CSV: {output_path}")


def infer_doc_language(relevant_docs: List[str], doc_meta: Dict[str, Dict[str, str]]) -> str:
    counts: Dict[str, int] = {}
    for doc_id in relevant_docs:
        lang = doc_meta.get(doc_id, {}).get("language", "unknown")
        counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return "unknown"
    return max(counts.items(), key=lambda x: x[1])[0]


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total / (1024 * 1024)


def run_retrieval(
    pipeline: RetrievalPipeline,
    query_text: str,
    method: str,
    top_k: int,
    target_lang: str,
    preprocess: bool,
) -> Tuple[List[str], float, float]:
    method_aliases = {
        "bm25": "whoosh",
        "tfidf": "whoosh",
    }
    pipeline_method = method_aliases.get(method, method)
    tracemalloc.start()
    start = time.time()
    result = pipeline.search(
        query_text,
        method=pipeline_method,
        top_k=top_k,
        target_lang=target_lang,
        preprocess=preprocess,
    )
    elapsed_ms = (time.time() - start) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if pipeline_method == "all":
        raise ValueError("method must not be 'all' in run_retrieval")

    results = result.get("results", [])
    doc_ids = [r.get("doc_id") for r in results if r.get("doc_id")]
    peak_mb = peak / (1024 * 1024)
    return doc_ids, elapsed_ms, peak_mb


def compute_metrics(
    queries: Dict[str, Dict[str, Any]],
    methods: List[str],
) -> Dict[str, Any]:
    results = {
        "test_queries": len(queries),
        "methods": methods,
        "per_query_results": {},
        "aggregate_metrics": {},
    }

    for method in methods:
        batch_queries = {}
        for qid, q in queries.items():
            relevant = q.get("relevant_docs", [])
            retrieved = q.get("retrieved_docs_by_method", {}).get(method, [])

            batch_queries[qid] = {
                "relevant": relevant,
                "retrieved": retrieved,
            }

            if qid not in results["per_query_results"]:
                results["per_query_results"][qid] = {}

            query_metrics = EvaluationMetrics.evaluate_query(relevant, retrieved, qid, 10)
            query_metrics["query_text"] = q.get("query_text", "")
            query_metrics["method"] = method
            query_metrics["num_relevant"] = len(relevant)
            query_metrics["num_retrieved"] = len(retrieved)
            results["per_query_results"][qid][method] = query_metrics

        _, summary = EvaluationMetrics.evaluate_batch(batch_queries, k=10)
        results["aggregate_metrics"][method] = summary

    return results


def compute_cross_lingual(
    queries: Dict[str, Dict[str, Any]],
    methods: List[str],
    doc_meta: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], List[str]] = {}

    for qid, q in queries.items():
        query_lang = q.get("language", "unknown")
        target_lang = q.get("target_lang")
        if not target_lang:
            target_lang = infer_doc_language(q.get("relevant_docs", []), doc_meta)
        key = (query_lang, target_lang)
        groups.setdefault(key, []).append(qid)

    group_metrics: Dict[str, Any] = {}
    for (query_lang, doc_lang), qids in groups.items():
        label = f"{query_lang}->{doc_lang}"
        group_metrics[label] = {
            "query_lang": query_lang,
            "doc_lang": doc_lang,
            "methods": {},
        }

        for method in methods:
            batch_queries = {}
            for qid in qids:
                q = queries[qid]
                batch_queries[qid] = {
                    "relevant": q.get("relevant_docs", []),
                    "retrieved": q.get("retrieved_docs_by_method", {}).get(method, []),
                }

            _, summary = EvaluationMetrics.evaluate_batch(batch_queries, k=10)
            group_metrics[label]["methods"][method] = {
                "p10": summary.get("mean_precision_at_10", 0.0),
                "r50": summary.get("mean_recall_at_50", 0.0),
                "ndcg10": summary.get("mean_ndcg_at_10", 0.0),
                "mrr": summary.get("mean_reciprocal_rank", 0.0),
                "num_queries": summary.get("num_queries", 0),
            }

    return group_metrics


def compute_error_analysis(error_cases_path: Path) -> Dict[str, Any]:
    analyzer = ErrorAnalyzer()

    if not error_cases_path.exists():
        return {
            "total_errors": 0,
            "error_summary": {},
            "examples": [],
        }

    with error_cases_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    for case in cases:
        error_type = case.get("error_type", "")
        query_id = case.get("query_id", "")
        query_text = case.get("query_text", "")
        query_language = case.get("query_language", "unknown")
        expected_docs = case.get("expected_docs", [])
        retrieved_docs = case.get("retrieved_docs", [])
        example_doc = case.get("example_doc", "")

        if error_type == "translation":
            analyzer.add_translation_failure(
                query_id,
                query_text,
                query_language,
                case.get("original_query", ""),
                case.get("mistranslated_query", ""),
                expected_docs,
                retrieved_docs,
                example_doc,
            )
        elif error_type == "ner_mismatch":
            analyzer.add_ner_mismatch(
                query_id,
                query_text,
                query_language,
                case.get("entity_in_query", ""),
                case.get("entity_in_docs", ""),
                expected_docs,
                retrieved_docs,
                example_doc,
            )
        elif error_type == "script":
            analyzer.add_cross_script_issue(
                query_id,
                query_text,
                query_language,
                case.get("script_variant_1", ""),
                case.get("script_variant_2", ""),
                expected_docs,
                retrieved_docs,
                example_doc,
            )
        elif error_type == "code_switch":
            analyzer.add_code_switching_issue(
                query_id,
                query_text,
                case.get("mixed_components", []),
                expected_docs,
                retrieved_docs,
                example_doc,
            )
        elif error_type in ["semantic_win", "lexical_win"]:
            winner = "semantic" if error_type == "semantic_win" else "lexical"
            analyzer.add_semantic_vs_lexical(
                query_id,
                query_text,
                query_language,
                winner,
                case.get("lexical_results", []),
                case.get("semantic_results", []),
                example_doc,
            )

    summary = analyzer.summarize_errors()
    examples = []
    for error_type in summary.keys():
        for case in analyzer.get_errors_by_type(error_type)[:2]:
            examples.append(
                {
                    "error_type": error_type,
                    "query_id": case.query_id,
                    "query_text": case.query_text,
                    "query_language": case.query_language,
                    "description": case.error_description,
                }
            )

    return {
        "total_errors": len(analyzer.error_cases),
        "error_summary": summary,
        "examples": examples,
    }


def compute_performance_summary(
    timing_by_method: Dict[str, List[float]],
    memory_by_method: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    index_sizes = {
        "bm25": dir_size_mb(INDEX_DIR / "whoosh"),
        "tfidf": dir_size_mb(INDEX_DIR / "whoosh"),
        "fuzzy": 0.0,
        "semantic": dir_size_mb(INDEX_DIR / "semantic"),
        "hybrid": dir_size_mb(INDEX_DIR / "whoosh") + dir_size_mb(INDEX_DIR / "semantic"),
    }

    summary = {}
    for method, times in timing_by_method.items():
        avg_time = sum(times) / len(times) if times else 0.0
        avg_mem = sum(memory_by_method.get(method, [])) / len(memory_by_method.get(method, [])) if memory_by_method.get(method) else 0.0
        summary[method] = {
            "avg_query_time_ms": avg_time,
            "avg_peak_mem_mb": avg_mem,
            "index_size_mb": index_sizes.get(method, 0.0),
        }
    return summary


def build_markdown_report(report: Dict[str, Any], output_path: Path) -> None:
    lines: List[str] = []

    lines.append("4.1 Evaluation Setup")
    lines.append(
        "We evaluated all five retrieval methods on test queries with manually annotated ground truth labels. "
        "Test queries span different query types (named entity queries, concept queries, phrase queries) "
        "and both languages (Bangla and English)."
    )
    lines.append("")

    lines.append("4.2 Quantitative Results")
    lines.append("4.2.1 Overall Performance Comparison")

    lines.append("Method | P@10 | R@50 | nDCG@10 | MRR")
    lines.append("---|---:|---:|---:|---:")

    for method, summary in report["overall"]["aggregate_metrics"].items():
        lines.append(
            f"{method.upper()} | "
            f"{summary.get('mean_precision_at_10', 0.0):.2f} | "
            f"{summary.get('mean_recall_at_50', 0.0):.2f} | "
            f"{summary.get('mean_ndcg_at_10', 0.0):.2f} | "
            f"{summary.get('mean_reciprocal_rank', 0.0):.2f}"
        )

    lines.append("")
    lines.append("Performance Targets: P@10 >= 0.60, R@50 >= 0.50, nDCG@10 >= 0.50, MRR >= 0.40")
    lines.append("Table 4.1: Retrieval Method Performance Comparison")
    lines.append("")

    lines.append("4.2.2 Performance Interpretation")
    lines.append(report["interpretation"]) 
    lines.append("")

    lines.append("4.3 Cross-Lingual Performance")
    lines.append("4.3.1 Monolingual vs Cross-Lingual Comparison")
    lines.append("Query Lang | Doc Lang | P@10 | R@50 | nDCG@10 | MRR")
    lines.append("---|---|---:|---:|---:|---:")

    for label, group in report["cross_lingual"].items():
        # pick hybrid if available, else first method
        method = "hybrid" if "hybrid" in group["methods"] else next(iter(group["methods"]))
        m = group["methods"][method]
        lines.append(
            f"{group['query_lang']} | {group['doc_lang']} | "
            f"{m['p10']:.2f} | {m['r50']:.2f} | {m['ndcg10']:.2f} | {m['mrr']:.2f}"
        )

    lines.append("Table 4.2: Cross-Lingual Performance Degradation")
    lines.append("")

    lines.append("Key Observations:")
    for obs in report.get("cross_lingual_observations", []):
        lines.append(f"- {obs}")
    lines.append("")

    lines.append("4.4 Error Analysis")
    lines.append("4.4.1 Error Categories")
    for item in report.get("error_categories", []):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("4.4.2 Error Distribution")
    lines.append("Error Type | Count | Percentage")
    lines.append("---|---:|---:")

    total_errors = report.get("error_analysis", {}).get("total_errors", 0)
    summary = report.get("error_analysis", {}).get("error_summary", {})
    for error_type, count in summary.items():
        pct = (count / total_errors * 100) if total_errors else 0.0
        lines.append(f"{error_type} | {count} | {pct:.1f}%")
    lines.append(f"Total Errors | {total_errors} | 100%")
    lines.append("Table 4.3: Error Type Distribution")
    lines.append("")

    lines.append("4.4.3 Specific Error Examples")
    for ex in report.get("error_analysis", {}).get("examples", []):
        lines.append(
            f"- {ex['error_type']}: Query {ex['query_id']} ({ex['query_language']}) - {ex['query_text']}"
        )
    lines.append("")

    lines.append("4.5 Computational Performance")
    lines.append("Method | Query Time (ms) | Memory (MB) | Index Size (MB)")
    lines.append("---|---:|---:|---:")
    for method, stats in report.get("performance", {}).items():
        lines.append(
            f"{method.upper()} | "
            f"{stats.get('avg_query_time_ms', 0.0):.1f} | "
            f"{stats.get('avg_peak_mem_mb', 0.0):.1f} | "
            f"{stats.get('index_size_mb', 0.0):.1f}"
        )
    lines.append("Table 4.4: Computational Efficiency Metrics")
    lines.append("")

    lines.append("4.6 Key Findings")
    for i, finding in enumerate(report.get("key_findings", []), 1):
        lines.append(f"{i}. {finding}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full evaluation report")
    parser.add_argument("--queries", default="test_queries.json", help="Query JSON file")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bm25", "tfidf", "fuzzy", "semantic", "hybrid"],
        help="Methods to evaluate",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K retrieved docs")
    parser.add_argument("--labels-csv", help="CSV with relevance labels")
    parser.add_argument("--error-cases", help="JSON file with error cases")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip query preprocessing")
    parser.add_argument("--output-json", default="evaluation_report.json", help="Output report JSON")
    parser.add_argument("--output-md", default="evaluation_report.md", help="Output report Markdown")
    parser.add_argument(
        "--export-labels",
        action="store_true",
        help="Write a CSV labeling pool from retrieval results",
    )
    parser.add_argument(
        "--label-pool-size",
        type=int,
        default=10,
        help="Max docs per query per method in labeling pool",
    )
    parser.add_argument(
        "--labels-out",
        default="labeling_pool.csv",
        help="Output CSV for labeling pool",
    )
    args = parser.parse_args()

    query_path = SCRIPT_DIR / args.queries
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")

    queries = load_queries(query_path)
    doc_meta = load_metadata()

    warn_missing_relevant_docs(queries, doc_meta)

    if args.labels_csv:
        labels_path = SCRIPT_DIR / args.labels_csv
        if labels_path.exists():
            relevant_by_query = load_relevant_from_labels(labels_path)
            for qid, docs in relevant_by_query.items():
                if qid in queries:
                    queries[qid]["relevant_docs"] = docs

    pipeline = RetrievalPipeline(index_dir=str(INDEX_DIR), use_query_processing=not args.no_preprocess)
    pipeline.load_indexes()

    timing_by_method: Dict[str, List[float]] = {m: [] for m in args.methods}
    memory_by_method: Dict[str, List[float]] = {m: [] for m in args.methods}

    for qid, q in queries.items():
        q.setdefault("retrieved_docs_by_method", {})
        for method in args.methods:
            target_lang = q.get("target_lang")
            doc_ids, elapsed_ms, peak_mb = run_retrieval(
                pipeline,
                q.get("query_text", ""),
                method,
                args.top_k,
                target_lang,
                preprocess=not args.no_preprocess,
            )
            q["retrieved_docs_by_method"][method] = doc_ids
            timing_by_method[method].append(elapsed_ms)
            memory_by_method[method].append(peak_mb)

    if args.export_labels:
        labels_out = SCRIPT_DIR / args.labels_out
        write_labeling_pool_csv(
            labels_out,
            queries,
            doc_meta,
            args.methods,
            args.label_pool_size,
        )

    overall = compute_metrics(queries, args.methods)
    cross_lingual = compute_cross_lingual(queries, args.methods, doc_meta)

    error_analysis = {}
    if args.error_cases:
        error_analysis = compute_error_analysis(SCRIPT_DIR / args.error_cases)
    else:
        error_analysis = compute_error_analysis(Path("__missing__"))

    performance = compute_performance_summary(timing_by_method, memory_by_method)

    interpretation_lines = []
    for method, summary in overall["aggregate_metrics"].items():
        p10 = summary.get("mean_precision_at_10", 0.0)
        r50 = summary.get("mean_recall_at_50", 0.0)
        ndcg10 = summary.get("mean_ndcg_at_10", 0.0)
        mrr = summary.get("mean_reciprocal_rank", 0.0)
        meets = []
        if p10 >= 0.6:
            meets.append("P@10")
        if r50 >= 0.5:
            meets.append("R@50")
        if ndcg10 >= 0.5:
            meets.append("nDCG@10")
        if mrr >= 0.4:
            meets.append("MRR")
        status = "meets " + ", ".join(meets) if meets else "meets no targets"
        interpretation_lines.append(f"{method.upper()} {status}.")

    report = {
        "overall": overall,
        "cross_lingual": cross_lingual,
        "error_analysis": error_analysis,
        "performance": performance,
        "interpretation": " ".join(interpretation_lines),
        "cross_lingual_observations": [
            "Compare monolingual vs cross-lingual groups using the table above.",
            "Use target_lang in queries to control cross-lingual direction.",
        ],
        "error_categories": [
            "Translation Failures",
            "Named Entity Mismatches",
            "Cross-Script Issues",
            "Code-Switching",
            "Semantic vs Lexical Wins",
        ],
        "key_findings": [
            "Best overall method is the one with highest P@10 and nDCG@10.",
            "Hybrid often balances recall and precision in mixed-language queries.",
            "Cross-lingual performance depends on translation and entity matching.",
            "Most common failure mode appears in error distribution.",
            "Use hybrid for production unless latency or resource limits dominate.",
        ],
    }

    output_json = SCRIPT_DIR / args.output_json
    output_md = SCRIPT_DIR / args.output_md

    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    build_markdown_report(report, output_md)

    print(f"Wrote report JSON: {output_json}")
    print(f"Wrote report Markdown: {output_md}")


if __name__ == "__main__":
    main()
