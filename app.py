import streamlit as st
import time
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


if __name__ == "__main__":
    main()
