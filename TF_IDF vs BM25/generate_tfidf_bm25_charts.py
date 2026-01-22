"""
Generate comparison charts for TF-IDF vs BM25.
Run this to create visualizations for your PDF report.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_term_frequency_saturation():
    """
    Visualize how TF-IDF and BM25 score term frequency differently.
    """
    tf_values = np.arange(1, 101, 1)
    
    # TF-IDF: Linear
    tfidf_scores = tf_values * 1.0  # Assuming IDF=1 for simplicity
    
    # BM25: Saturating (k1=1.5)
    k1 = 1.5
    bm25_scores = (tf_values * (k1 + 1)) / (tf_values + k1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(tf_values, tfidf_scores / tfidf_scores[0], label='TF-IDF (Linear)', linewidth=2, color='red')
    plt.plot(tf_values, bm25_scores / bm25_scores[0], label='BM25 (Saturating, k1=1.5)', linewidth=2, color='blue')
    
    plt.xlabel('Term Frequency (occurrences in document)', fontsize=12)
    plt.ylabel('Relative Score (normalized to TF=1)', fontsize=12)
    plt.title('Term Frequency Saturation: TF-IDF vs BM25', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    
    # Annotate key points
    plt.annotate('TF-IDF continues growing linearly', 
                 xy=(80, tfidf_scores[79]/tfidf_scores[0]), 
                 xytext=(50, 60),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 fontsize=10, color='red')
    
    plt.annotate('BM25 saturates around k1+1', 
                 xy=(80, bm25_scores[79]/bm25_scores[0]), 
                 xytext=(50, 5),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                 fontsize=10, color='blue')
    
    plt.tight_layout()
    plt.savefig('tfidf_vs_bm25_saturation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: tfidf_vs_bm25_saturation.png")
    plt.close()


def plot_metrics_comparison():
    """
    Bar chart comparing evaluation metrics.
    """
    metrics = ['P@10', 'Recall@50', 'MRR', 'nDCG@10']
    tfidf_values = [0.24, 0.96, 0.72, 0.68]
    bm25_values = [0.28, 0.92, 0.78, 0.73]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, tfidf_values, width, label='TF-IDF', color='#ff7f0e')
    bars2 = ax.bar(x + width/2, bm25_values, width, label='BM25', color='#2ca02c')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('TF-IDF vs BM25: Evaluation Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('tfidf_vs_bm25_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: tfidf_vs_bm25_metrics.png")
    plt.close()


def plot_document_length_effect():
    """
    Show how BM25 normalizes by document length.
    """
    doc_lengths = np.arange(100, 2001, 100)
    avgdl = 800  # Average document length
    
    # BM25 length penalty with different b values
    b_values = [0.0, 0.5, 0.75, 1.0]
    
    plt.figure(figsize=(10, 6))
    
    for b in b_values:
        length_penalty = 1 - b + b * (doc_lengths / avgdl)
        normalized_score = 1.0 / length_penalty
        plt.plot(doc_lengths, normalized_score, label=f'b={b}', linewidth=2)
    
    plt.xlabel('Document Length (words)', fontsize=12)
    plt.ylabel('Relative Score (normalized to avg length)', fontsize=12)
    plt.title('BM25 Document Length Normalization (k1=1.5)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axvline(avgdl, color='gray', linestyle='--', alpha=0.5, label=f'Avg Length ({avgdl} words)')
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Annotate
    plt.annotate('b=0: No penalty\n(all docs equal)', 
                 xy=(1800, 1.0), fontsize=10, color='blue')
    plt.annotate('b=0.75: Balanced\n(default)', 
                 xy=(1500, 0.7), fontsize=10, color='green', fontweight='bold')
    plt.annotate('b=1: Full penalty\n(exact normalization)', 
                 xy=(1800, 0.45), fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('bm25_length_normalization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: bm25_length_normalization.png")
    plt.close()


def plot_winner_summary():
    """
    Pie chart showing overall winner.
    """
    labels = ['BM25 Wins', 'TF-IDF Wins', 'Ties']
    sizes = [15, 4, 1]  # From 5 queries × 4 metrics = 20 comparisons
    colors = ['#2ca02c', '#ff7f0e', '#d3d3d3']
    explode = (0.1, 0, 0)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
    plt.title('Overall Winner: BM25 vs TF-IDF\n(5 queries × 4 metrics = 20 comparisons)',
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('tfidf_vs_bm25_winner.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: tfidf_vs_bm25_winner.png")
    plt.close()


def main():
    """
    Generate all comparison visualizations.
    """
    print("="*80)
    print("GENERATING TF-IDF vs BM25 VISUALIZATIONS FOR REPORT")
    print("="*80)
    
    plot_term_frequency_saturation()
    plot_metrics_comparison()
    plot_document_length_effect()
    plot_winner_summary()
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS GENERATED")
    print("="*80)
    print("\nInclude these images in your PDF report:")
    print("  1. tfidf_vs_bm25_saturation.png - Shows term frequency saturation")
    print("  2. tfidf_vs_bm25_metrics.png - Evaluation metrics comparison")
    print("  3. bm25_length_normalization.png - Document length handling")
    print("  4. tfidf_vs_bm25_winner.png - Overall winner summary")


if __name__ == "__main__":
    main()
