"""
CLIR System: Bangla-English Cross-Lingual Information Retrieval
Streamlit web interface for searching across Bangla and English news articles
"""

import streamlit as st
import time
import pandas as pd

from src.module3_retrieval.retriever import Retriever
from src.module4_ranking.ranker import Ranker


@st.cache_resource
def load_retriever():
    """Load retriever with cached models and indices."""
    return Retriever()


def main():
    """Main Streamlit app."""
    
    # Page configuration
    st.set_page_config(
        page_title="CLIR System",
        page_icon="🇧🇩",
        layout="wide"
    )
    
    # Title
    st.title("🇧🇩 CLIR System: Bangla-English Search")
    st.markdown("Cross-Lingual Information Retrieval across Bangla and English news articles")
    
    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    alpha = st.sidebar.slider(
        "Hybrid Weight (Alpha)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0.0 = Pure Lexical, 1.0 = Pure Semantic, 0.5 = Equal Mix"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **How it works:**
    - Enter a query in Bangla or English
    - System searches across both languages
    - Results are ranked using hybrid fusion
    
    **Alpha Parameter:**
    - Controls weighting between lexical (BM25) and semantic (LaBSE) search
    - Higher α = More semantic
    - Lower α = More keyword-based
    """)
    
    # Initialize components
    try:
        with st.spinner("Loading models and indices..."):
            retriever = load_retriever()
            ranker = Ranker()
        st.success("✅ System ready!")
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {e}")
        return
    
    # Search bar
    st.markdown("---")
    query = st.text_input(
        "🔍 Enter your search query (Bangla or English)",
        placeholder="e.g., Dhaka air pollution or বাংলাদেশের অর্থনীতি"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("Search", type="primary", use_container_width=True)
    with col2:
        if query:
            st.caption(f"Searching with α={alpha:.1f} (Semantic: {alpha*100:.0f}%, Lexical: {(1-alpha)*100:.0f}%)")
    
    # Search logic
    if search_button and query:
        with st.spinner("Searching..."):
            # Start timer
            start_time = time.time()
            
            try:
                # Retrieve results
                search_results = retriever.search(query, k=10)
                
                # Merge and rank
                ranked_results = ranker.merge_and_rank(
                    lexical_results=search_results['lexical'],
                    semantic_results=search_results['semantic'],
                    alpha=alpha
                )
                
                # Stop timer
                elapsed_time = time.time() - start_time
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Search Results")
                
                # Metrics bar
                results_list = ranked_results['results']
                if results_list:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Search Time", f"{elapsed_time:.3f} sec")
                    with col2:
                        st.metric("📝 Results Found", len(results_list))
                    with col3:
                        top_score = results_list[0]['final_score'] if results_list else 0.0
                        st.metric("⭐ Top Score", f"{top_score:.3f}")
                    
                    # Warning
                    if ranked_results['warning']:
                        st.warning(f"⚠️ {ranked_results['warning']}")
                    
                    st.markdown("---")
                    
                    # Display top 10 results
                    for i, result in enumerate(results_list[:10], 1):
                        with st.container():
                            # Title with link
                            title = result.get('title', 'No title')
                            url = result.get('url', '')
                            
                            if url:
                                st.markdown(f"### {i}. [{title}]({url})")
                            else:
                                st.markdown(f"### {i}. {title}")
                            
                            # Score badges
                            col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
                            
                            with col1:
                                lang_emoji = "🇧🇩" if result.get('lang') == 'bangla' else "🇬🇧"
                                st.caption(f"{lang_emoji} {result.get('lang', 'unknown').title()}")
                            
                            with col2:
                                st.caption(f"🔤 Lexical: {result['lexical_score']:.3f}")
                            
                            with col3:
                                st.caption(f"🧠 Semantic: {result['semantic_score']:.3f}")
                            
                            with col4:
                                final = result['final_score']
                                if final < 0.2:
                                    st.caption(f"⚡ **Final: {final:.3f}** 🔴 Low Confidence")
                                else:
                                    st.caption(f"⚡ **Final: {final:.3f}**")
                            
                            st.markdown("---")
                
                else:
                    st.info("No results found. Try a different query or adjust the alpha parameter.")
                    st.metric("⏱️ Search Time", f"{elapsed_time:.3f} sec")
                
            except Exception as e:
                st.error(f"❌ Search failed: {e}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
    
    elif search_button and not query:
        st.warning("⚠️ Please enter a search query.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <small>CLIR System | Cross-Lingual Information Retrieval | Bangla-English News Search</small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
