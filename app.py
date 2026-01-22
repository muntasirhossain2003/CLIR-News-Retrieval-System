import streamlit as st
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.module3_retrieval.retriever import Retriever
from src.module4_ranking.ranker import Ranker


@st.cache_resource
def load_retriever():
    return Retriever()


def apply_custom_css():
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #6b7280;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .result-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .result-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 0.5rem;
        }
        .result-title a {
            color: #667eea;
            text-decoration: none;
        }
        .result-title a:hover {
            text-decoration: underline;
        }
        .score-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            margin-right: 0.5rem;
        }
        .badge-lexical {
            background: #dbeafe;
            color: #1e40af;
        }
        .badge-semantic {
            background: #fce7f3;
            color: #9f1239;
        }
        .badge-fuzzy {
            background: #fef3c7;
            color: #92400e;
        }
        .badge-final {
            background: #d1fae5;
            color: #065f46;
            font-weight: 700;
        }
        .badge-low {
            background: #fee2e2;
            color: #991b1b;
        }
        .search-box {
            margin: 2rem 0;
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="CLIR System - Bangla-English Search",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_css()
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Search", "Evaluation & Metrics", "Error Analysis"])
    
    if page == "Search":
        search_page()
    elif page == "Evaluation & Metrics":
        evaluation_page()
    elif page == "Error Analysis":
        error_analysis_page()


def search_page():
    
    st.markdown('<div class="main-header">Cross-Lingual Information Retrieval</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Search across Bangla and English news articles with multilingual support</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Search Configuration")
        
        st.markdown("### Hybrid Weight")
        alpha = st.slider(
            "Alpha Parameter",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Balance between lexical and semantic search"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Semantic", f"{alpha*100:.0f}%")
        with col2:
            st.metric("Lexical", f"{(1-alpha)*100:.0f}%")
        
        st.markdown("---")
        
        st.markdown("### About")
        st.info(
            "This system uses hybrid retrieval combining:\n\n"
            "• **BM25** for lexical matching\n"
            "• **LaBSE** for semantic similarity\n"
            "• **Fuzzy matching** for transliteration\n\n"
            "Adjust alpha to control the balance between approaches."
        )
        
        st.markdown("---")
        st.markdown("### System Status")
    
    try:
        with st.spinner("Initializing system..."):
            retriever = load_retriever()
            ranker = Ranker()
        
        with st.sidebar:
            st.success("System Ready")
            st.caption(f"Dataset: 5,194 documents")
            st.caption(f"Languages: Bangla, English")
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        return
    
    st.markdown("---")
    
    query = st.text_input(
        "Enter your search query",
        placeholder="Type in Bangla or English (e.g., Dhaka air pollution or বাংলাদেশের অর্থনীতি)",
        key="search_input"
    )
    
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    if search_button and query:
        with st.spinner("Searching across 5,194 documents..."):
            try:
                search_results = retriever.search(query, k=10)
                
                t_rank_start = time.time()
                ranked_results = ranker.merge_and_rank(
                    lexical_results=search_results['lexical'],
                    semantic_results=search_results['semantic'],
                    fuzzy_results=search_results['fuzzy'],
                    alpha=alpha
                )
                t_rank = time.time() - t_rank_start
                
                timing = search_results['timing']
                timing['ranking'] = t_rank
                elapsed_time = timing['total'] + t_rank
                
                st.markdown("---")
                
                results_list = ranked_results['results']
                
                if results_list:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Response Time", f"{elapsed_time:.3f}s")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Results Found", len(results_list))
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        top_score = results_list[0]['final_score']
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Top Match", f"{top_score:.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Query Process", f"{timing['query_processing']*1000:.0f}ms")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("")
                    
                    with st.expander("Performance Breakdown"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Phase**")
                            st.text("Query Processing")
                            st.text("Lexical Search")
                            st.text("Semantic Search")
                            st.text("Fuzzy Search")
                            st.text("Ranking Fusion")
                        
                        with col2:
                            st.markdown("**Time (ms)**")
                            st.text(f"{timing['query_processing']*1000:.1f}")
                            st.text(f"{timing['lexical_search']*1000:.1f}")
                            st.text(f"{timing['semantic_search']*1000:.1f}")
                            st.text(f"{timing['fuzzy_search']*1000:.1f}")
                            st.text(f"{timing['ranking']*1000:.1f}")
                        
                        with col3:
                            st.markdown("**Share**")
                            st.text(f"{(timing['query_processing']/elapsed_time)*100:.1f}%")
                            st.text(f"{(timing['lexical_search']/elapsed_time)*100:.1f}%")
                            st.text(f"{(timing['semantic_search']/elapsed_time)*100:.1f}%")
                            st.text(f"{(timing['fuzzy_search']/elapsed_time)*100:.1f}%")
                            st.text(f"{(timing['ranking']/elapsed_time)*100:.1f}%")
                    
                    if ranked_results['warning']:
                        st.warning(ranked_results['warning'])
                    
                    st.markdown("### Search Results")
                    
                    for i, result in enumerate(results_list[:10], 1):
                        title = result.get('title', 'Untitled')
                        url = result.get('url', '')
                        lang = result.get('lang', 'unknown')
                        
                        final_score = result['final_score']
                        badge_class = "badge-low" if final_score < 0.2 else "badge-final"
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-title">
                                {i}. <a href="{url}" target="_blank">{title}</a>
                            </div>
                            <div style="margin-top: 0.75rem;">
                                <span class="score-badge" style="background: #e5e7eb; color: #374151;">
                                    {lang.upper()}
                                </span>
                                <span class="score-badge badge-lexical">
                                    Lexical: {result['lexical_score']:.3f}
                                </span>
                                <span class="score-badge badge-semantic">
                                    Semantic: {result['semantic_score']:.3f}
                                </span>
                                <span class="score-badge badge-fuzzy">
                                    Fuzzy: {result['fuzzy_score']:.3f}
                                </span>
                                <span class="score-badge {badge_class}">
                                    Final: {final_score:.3f}
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                else:
                    st.info("No results found. Try adjusting your query or the alpha parameter.")
                    st.metric("Search Time", f"{elapsed_time:.3f}s")
                
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                with st.expander("Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
    
    elif search_button and not query:
        st.warning("Please enter a search query to begin.")
    
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #9ca3af; padding: 2rem 0;">'
        '<small>CLIR System • Cross-Lingual Information Retrieval • 2024</small>'
        '</div>',
        unsafe_allow_html=True
    )


def evaluation_page():
    """Display evaluation metrics and results."""
    st.markdown('<div class="main-header">Evaluation & Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">System performance metrics across test queries</div>', unsafe_allow_html=True)
    
    # Load test data
    test_queries = {
        "Q1": {
            "query": "Dhaka air pollution / ঢাকার বায়ু দূষণ",
            "metrics": {"P@10": 0.40, "R@50": 1.00, "MRR": 1.00, "nDCG@10": 0.854}
        },
        "Q2": {
            "query": "Bangladesh economy / বাংলাদেশের অর্থনীতি",
            "metrics": {"P@10": 0.30, "R@50": 0.67, "MRR": 0.50, "nDCG@10": 0.725}
        },
        "Q3": {
            "query": "Cricket match score / ক্রিকেট ম্যাচের স্কোর",
            "metrics": {"P@10": 0.30, "R@50": 1.00, "MRR": 0.33, "nDCG@10": 0.680}
        }
    }
    
    # Overall metrics
    avg_metrics = {
        "P@10": sum(q["metrics"]["P@10"] for q in test_queries.values()) / len(test_queries),
        "R@50": sum(q["metrics"]["R@50"] for q in test_queries.values()) / len(test_queries),
        "MRR": sum(q["metrics"]["MRR"] for q in test_queries.values()) / len(test_queries),
        "nDCG@10": sum(q["metrics"]["nDCG@10"] for q in test_queries.values()) / len(test_queries)
    }
    
    # Display overall metrics
    st.markdown("### Overall Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Precision@10", f"{avg_metrics['P@10']:.3f}", 
                 delta=f"Target: 0.600", delta_color="off")
    with col2:
        st.metric("Recall@50", f"{avg_metrics['R@50']:.3f}",
                 delta=f"Target: 0.500", delta_color="off")
    with col3:
        st.metric("MRR", f"{avg_metrics['MRR']:.3f}",
                 delta=f"Target: 0.400", delta_color="off")
    with col4:
        st.metric("nDCG@10", f"{avg_metrics['nDCG@10']:.3f}",
                 delta=f"Target: 0.500", delta_color="off")
    
    st.markdown("")
    
    # Metrics comparison chart
    st.markdown("### Metrics by Query")
    
    # Prepare data for chart
    df_metrics = pd.DataFrame([
        {"Query": q_id, "Metric": "P@10", "Score": q_data["metrics"]["P@10"]}
        for q_id, q_data in test_queries.items()
    ] + [
        {"Query": q_id, "Metric": "R@50", "Score": q_data["metrics"]["R@50"]}
        for q_id, q_data in test_queries.items()
    ] + [
        {"Query": q_id, "Metric": "MRR", "Score": q_data["metrics"]["MRR"]}
        for q_id, q_data in test_queries.items()
    ] + [
        {"Query": q_id, "Metric": "nDCG@10", "Score": q_data["metrics"]["nDCG@10"]}
        for q_id, q_data in test_queries.items()
    ])
    
    fig = px.bar(df_metrics, x="Query", y="Score", color="Metric", 
                 barmode="group", 
                 title="Evaluation Metrics Comparison",
                 color_discrete_map={
                     "P@10": "#3498db",
                     "R@50": "#e74c3c",
                     "MRR": "#f39c12",
                     "nDCG@10": "#2ecc71"
                 })
    fig.update_layout(yaxis_range=[0, 1.1])
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.markdown("### Detailed Results")
    
    detailed_data = []
    for q_id, q_data in test_queries.items():
        detailed_data.append({
            "Query ID": q_id,
            "Query": q_data["query"],
            "P@10": f"{q_data['metrics']['P@10']:.3f}",
            "R@50": f"{q_data['metrics']['R@50']:.3f}",
            "MRR": f"{q_data['metrics']['MRR']:.3f}",
            "nDCG@10": f"{q_data['metrics']['nDCG@10']:.3f}"
        })
    
    df_detailed = pd.DataFrame(detailed_data)
    st.dataframe(df_detailed, use_container_width=True)
    
    # Metrics explanation
    with st.expander("Metrics Explanation"):
        st.markdown("""
        **Precision@10 (P@10)**: Of the top 10 results, what fraction are relevant?
        - Higher is better (Target: ≥0.60)
        - Measures accuracy of top results
        
        **Recall@50 (R@50)**: Of ALL relevant documents, what fraction did we find in top 50?
        - Higher is better (Target: ≥0.50)
        - Measures coverage
        
        **MRR (Mean Reciprocal Rank)**: How quickly do we find the first relevant result?
        - Calculated as: 1 / (rank of first relevant doc)
        - Higher is better (Target: ≥0.40)
        
        **nDCG@10**: Measures ranking quality (relevant docs should be at top)
        - Higher is better (Target: ≥0.50)
        - Penalizes relevant docs appearing lower in ranking
        """)


def error_analysis_page():
    """Display error analysis and failure categories."""
    st.markdown('<div class="main-header">Error Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analysis of retrieval failures across 5 error categories</div>', unsafe_allow_html=True)
    
    # Error categories data
    error_categories = {
        "Translation Drift": {
            "count": 2,
            "examples": [
                {"query": "খুলনা (Khulna)", "issue": "Mistranslated as 'Open' instead of city name"},
                {"query": "যানজট (Traffic)", "issue": "Lost semantic nuance in translation"}
            ]
        },
        "Tokenization Issue": {
            "count": 1,
            "examples": [
                {"query": "পদ্মা সেতু (Padma Bridge)", "issue": "Compound noun split incorrectly"}
            ]
        },
        "Named Entity Failure": {
            "count": 1,
            "examples": [
                {"query": "রোহিঙ্গা (Rohingya)", "issue": "Multiple transliteration variants not recognized"}
            ]
        },
        "Domain Mismatch": {
            "count": 1,
            "examples": [
                {"query": "করোনা (Corona)", "issue": "Retrieved astronomy instead of COVID-19"}
            ]
        },
        "Stopword Issue": {
            "count": 1,
            "examples": [
                {"query": "এই বছরের (This year's)", "issue": "Common words caused irrelevant matches"}
            ]
        }
    }
    
    # Error distribution chart
    st.markdown("### Error Category Distribution")
    
    error_df = pd.DataFrame([
        {"Category": cat, "Failures": data["count"]}
        for cat, data in error_categories.items()
    ])
    
    fig = px.bar(error_df, x="Category", y="Failures",
                 title="Retrieval Failures by Category",
                 color="Failures",
                 color_continuous_scale="Reds")
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
    st.markdown("### Error Summary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        summary_data = []
        for cat, data in error_categories.items():
            summary_data.append({
                "Error Category": cat,
                "Failures": data["count"],
                "Percentage": f"{(data['count'] / sum(d['count'] for d in error_categories.values()) * 100):.1f}%"
            })
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    with col2:
        total_errors = sum(d['count'] for d in error_categories.values())
        st.metric("Total Errors", total_errors)
        st.metric("Categories", len(error_categories))
    
    # Detailed examples
    st.markdown("### Detailed Error Examples")
    
    for category, data in error_categories.items():
        with st.expander(f"{category} ({data['count']} failures)"):
            for i, example in enumerate(data['examples'], 1):
                st.markdown(f"""
                **Example {i}:**
                - **Query:** {example['query']}
                - **Issue:** {example['issue']}
                """)
    
    # Recommendations
    st.markdown("### Recommendations")
    st.info("""
    **Based on error analysis, we recommend:**
    
    1. **Translation Drift**: Use direct LaBSE embedding (no translation) for Bangla queries
    2. **Tokenization Issues**: Implement phrase detection for compound nouns
    3. **Named Entities**: Add transliteration dictionary for proper nouns
    4. **Domain Mismatch**: Add category filtering or domain-specific models
    5. **Stopword Issues**: Improve stopword list for Bangla language
    """)


if __name__ == "__main__":
    main()
