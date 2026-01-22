"""
Evaluation Script with Visualization for Report
Generates tables, charts, and CSV files showing ranking and evaluation metrics
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from src.module3_retrieval.retriever import Retriever
from src.module4_ranking.ranker import Ranker

# Set style for professional-looking charts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Test queries with ground truth
TEST_QUERIES = [
    {
        "id": 1,
        "query_en": "Dhaka air pollution",
        "query_bn": "ঢাকার বায়ু দূষণ",
        "relevant_urls": [
            "https://unb.com.bd/category/Environment/unhealthy-air-quality-persists-in-dhaka/175160",
            "https://unb.com.bd/category/Environment/unhealthy-air-quality-persists-in-dhaka/175354"
        ]
    },
    {
        "id": 2,
        "query_en": "Bangladesh economy",
        "query_bn": "বাংলাদেশের অর্থনীতি",
        "relevant_urls": [
            "https://www.prothomalo.com/business/analysis/1xs122t6hk",
            "https://samakal.com/bangladesh/article/329972/",
            "https://samakal.com/opinion/article/329849/"
        ]
    },
    {
        "id": 3,
        "query_en": "Cricket match score",
        "query_bn": "ক্রিকেট ম্যাচের স্কোর",
        "relevant_urls": [
            "https://www.prothomalo.com/ampstories/sports/cricket/oejdb47pha?offset=7",
            "https://www.jugantor.com/sports/1041854",
            "https://www.prothomalo.com/ampstories/sports/cricket/978vipxzx2?offset=4"
        ]
    },
]


def evaluate_single_query(query_text: str, relevant_urls: List[str], retriever: Retriever, ranker: Ranker):
    """Evaluate a single query and return detailed results."""
    
    # Search
    search_results = retriever.search(query_text, k=50)
    
    # Rank
    ranked_results = ranker.merge_and_rank(
        lexical_results=search_results['lexical'],
        semantic_results=search_results['semantic'],
        fuzzy_results=search_results['fuzzy'],
        alpha=0.6,
        fuzzy_weight=0.2
    )
    
    # Extract URLs
    retrieved_urls = [r['url'] for r in ranked_results['results'][:50]]
    
    # Calculate metrics
    metrics = ranker.calculate_metrics(retrieved_urls, set(relevant_urls))
    
    # Find positions of relevant docs
    relevant_positions = []
    for url in relevant_urls:
        for i, result in enumerate(ranked_results['results'], 1):
            if result['url'] == url:
                relevant_positions.append({
                    'url': url,
                    'position': i,
                    'score': result['final_score'],
                    'title': result['title']
                })
                break
    
    return {
        'metrics': metrics,
        'top_10': ranked_results['results'][:10],
        'relevant_positions': relevant_positions,
        'timing': search_results['timing']
    }


def generate_ranking_table(query_data: Dict, results: Dict, output_file: str):
    """Generate HTML/CSV table showing top 10 ranked results."""
    
    data = []
    for i, result in enumerate(results['top_10'], 1):
        is_relevant = result['url'] in query_data['relevant_urls']
        data.append({
            'Rank': i,
            'Title': result['title'][:80] + '...' if len(result['title']) > 80 else result['title'],
            'Language': result['lang'],
            'Lexical': f"{result['lexical_score']:.3f}",
            'Semantic': f"{result['semantic_score']:.3f}",
            'Fuzzy': f"{result['fuzzy_score']:.3f}",
            'Final Score': f"{result['final_score']:.3f}",
            'Relevant': '✓' if is_relevant else '✗'
        })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_file = output_file.replace('.html', '.csv')
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    # Save as HTML with styling
    html = df.to_html(index=False, classes='table table-striped')
    html_styled = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .relevant {{ color: green; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2>Query: {query_data['query_en']} / {query_data['query_bn']}</h2>
        {html}
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_styled)
    
    print(f"✓ Saved ranking table: {output_file}")
    print(f"✓ Saved CSV: {csv_file}")
    
    return df


def generate_metrics_chart(all_results: List[Dict], output_file: str):
    """Generate bar chart comparing metrics across queries."""
    
    queries = [f"Q{r['id']}" for r in all_results]
    metrics_names = ['P@10', 'R@50', 'MRR', 'nDCG@10']
    
    data = {
        'P@10': [r['metrics']['precision@10'] for r in all_results],
        'R@50': [r['metrics']['recall@50'] for r in all_results],
        'MRR': [r['metrics']['mrr'] for r in all_results],
        'nDCG@10': [r['metrics']['ndcg@10'] for r in all_results]
    }
    
    df = pd.DataFrame(data, index=queries)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title('Evaluation Metrics by Query', fontsize=16, fontweight='bold')
    ax.set_xlabel('Query ID', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(title='Metrics', loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics chart: {output_file}")
    
    return df


def generate_score_distribution(results: Dict, query_data: Dict, output_file: str):
    """Generate chart showing score breakdown for top 10 results."""
    
    top_10 = results['top_10']
    ranks = list(range(1, 11))
    lexical_scores = [r['lexical_score'] for r in top_10]
    semantic_scores = [r['semantic_score'] for r in top_10]
    fuzzy_scores = [r['fuzzy_score'] for r in top_10]
    final_scores = [r['final_score'] for r in top_10]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Stacked bar chart
    width = 0.6
    ax1.bar(ranks, lexical_scores, width, label='Lexical (20%)', color='#3498db')
    ax1.bar(ranks, semantic_scores, width, bottom=lexical_scores, label='Semantic (60%)', color='#e74c3c')
    bottom = [l + s for l, s in zip(lexical_scores, semantic_scores)]
    ax1.bar(ranks, fuzzy_scores, width, bottom=bottom, label='Fuzzy (20%)', color='#f39c12')
    
    ax1.set_xlabel('Rank', fontsize=12)
    ax1.set_ylabel('Normalized Score', fontsize=12)
    ax1.set_title('Score Composition (Stacked)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_xticks(ranks)
    ax1.grid(axis='y', alpha=0.3)
    
    # Line chart for final scores
    ax2.plot(ranks, final_scores, marker='o', linewidth=2, markersize=8, color='#2ecc71')
    ax2.fill_between(ranks, final_scores, alpha=0.3, color='#2ecc71')
    ax2.set_xlabel('Rank', fontsize=12)
    ax2.set_ylabel('Final Score', fontsize=12)
    ax2.set_title('Final Ranking Scores', fontsize=14, fontweight='bold')
    ax2.set_xticks(ranks)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, max(final_scores) * 1.1)
    
    plt.suptitle(f"Query: {query_data['query_en']}", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved score distribution: {output_file}")


def generate_summary_report(all_results: List[Dict], output_file: str):
    """Generate comprehensive summary report as HTML."""
    
    # Calculate average metrics
    avg_metrics = {
        'P@10': sum(r['metrics']['precision@10'] for r in all_results) / len(all_results),
        'R@50': sum(r['metrics']['recall@50'] for r in all_results) / len(all_results),
        'MRR': sum(r['metrics']['mrr'] for r in all_results) / len(all_results),
        'nDCG@10': sum(r['metrics']['ndcg@10'] for r in all_results) / len(all_results)
    }
    
    # Create detailed table
    detailed_data = []
    for result in all_results:
        detailed_data.append({
            'Query ID': result['id'],
            'Query': result['query'],
            'P@10': f"{result['metrics']['precision@10']:.3f}",
            'R@50': f"{result['metrics']['recall@50']:.3f}",
            'MRR': f"{result['metrics']['mrr']:.3f}",
            'nDCG@10': f"{result['metrics']['ndcg@10']:.3f}",
            'Time (ms)': f"{result['timing']['total']*1000:.1f}"
        })
    
    df = pd.DataFrame(detailed_data)
    
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .summary {{ background-color: #e8f8f5; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .metric {{ font-size: 18px; font-weight: bold; color: #27ae60; }}
        </style>
    </head>
    <body>
        <h1>CLIR System Evaluation Report</h1>
        
        <div class="summary">
            <h2>Average Performance Across All Queries</h2>
            <p><span class="metric">Precision@10:</span> {avg_metrics['P@10']:.3f} (How many top 10 results are relevant)</p>
            <p><span class="metric">Recall@50:</span> {avg_metrics['R@50']:.3f} (Coverage of relevant documents)</p>
            <p><span class="metric">MRR:</span> {avg_metrics['MRR']:.3f} (How quickly first relevant appears)</p>
            <p><span class="metric">nDCG@10:</span> {avg_metrics['nDCG@10']:.3f} (Ranking quality)</p>
        </div>
        
        <h2>Detailed Results by Query</h2>
        {df.to_html(index=False)}
        
        <h2>System Configuration</h2>
        <ul>
            <li><strong>Retrieval Methods:</strong> Lexical (BM25), Semantic (LaBSE), Fuzzy (Transliteration)</li>
            <li><strong>Fusion Weights:</strong> Semantic 60%, Lexical 20%, Fuzzy 20%</li>
            <li><strong>Dataset:</strong> 5,194 documents (Bangla + English)</li>
            <li><strong>Evaluation:</strong> {len(all_results)} test queries with manual ground truth</li>
        </ul>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Saved summary report: {output_file}")
    
    # Also save as CSV
    csv_file = output_file.replace('.html', '_detailed.csv')
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"✓ Saved detailed CSV: {csv_file}")


def main():
    print("="*70)
    print("CLIR SYSTEM EVALUATION WITH VISUALIZATION")
    print("="*70)
    print()
    
    # Initialize system
    print("Loading retriever and ranker...")
    retriever = Retriever()
    ranker = Ranker()
    print("✓ System ready\n")
    
    # Run evaluation
    all_results = []
    
    for query_data in TEST_QUERIES:
        query_id = query_data['id']
        query_en = query_data['query_en']
        query_bn = query_data['query_bn']
        
        print(f"Query {query_id}: {query_en} / {query_bn}")
        print("-" * 70)
        
        # Evaluate English query
        results = evaluate_single_query(query_en, query_data['relevant_urls'], retriever, ranker)
        
        print(f"  P@10:    {results['metrics']['precision@10']:.3f}")
        print(f"  R@50:    {results['metrics']['recall@50']:.3f}")
        print(f"  MRR:     {results['metrics']['mrr']:.3f}")
        print(f"  nDCG@10: {results['metrics']['ndcg@10']:.3f}")
        print(f"  Time:    {results['timing']['total']*1000:.1f} ms")
        print()
        
        # Generate visualizations
        ranking_table_file = f"evaluation_results/query_{query_id}_ranking.html"
        score_dist_file = f"evaluation_results/query_{query_id}_scores.png"
        
        generate_ranking_table(query_data, results, ranking_table_file)
        generate_score_distribution(results, query_data, score_dist_file)
        
        # Store for summary
        all_results.append({
            'id': query_id,
            'query': query_en,
            'metrics': results['metrics'],
            'timing': results['timing']
        })
        
        print()
    
    # Generate summary visualizations
    print("="*70)
    print("GENERATING SUMMARY VISUALIZATIONS")
    print("="*70)
    print()
    
    generate_metrics_chart(all_results, "evaluation_results/metrics_comparison.png")
    generate_summary_report(all_results, "evaluation_results/evaluation_summary.html")
    
    print()
    print("="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print()
    print("Generated files in evaluation_results/:")
    print("  - evaluation_summary.html (Main report)")
    print("  - metrics_comparison.png (Chart)")
    print("  - query_X_ranking.html/csv (Per-query results)")
    print("  - query_X_scores.png (Score visualizations)")
    print()
    print("Open evaluation_summary.html in your browser to see the full report!")


if __name__ == "__main__":
    import os
    os.makedirs("evaluation_results", exist_ok=True)
    main()
