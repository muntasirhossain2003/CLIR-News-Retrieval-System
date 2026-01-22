import streamlit as st
import sys
import os
import time
import pandas as pd
from pathlib import Path

# Get project root (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "indexes"

# Add Module C and Module B to path
module_c_path = PROJECT_ROOT / "src" / "Module C — Retrieval Models"
module_b_path = (
    PROJECT_ROOT / "src" / "Module B — Query Processing & Cross-Lingual Handling"
)
sys.path.insert(0, str(module_c_path))
sys.path.insert(0, str(module_b_path))

from retrieval_pipeline import RetrievalPipeline

# Try to import language detection
try:
    from language_detection_normalization import detect_query_language

    LANG_DETECTION_AVAILABLE = True
except ImportError:
    LANG_DETECTION_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="CLIR News Retrieval",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .clir-info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #fff3e0 100%);
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        color: #333;
    }
    .result-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .result-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #333;
    }
    .result-title-bn {
        font-size: 1.3rem;
        font-weight: bold;
        color: #333;
        font-family: 'Noto Sans Bengali', sans-serif;
    }
    .result-meta {
        font-size: 0.9rem;
        color: #666;
    }
    .lang-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .lang-bn {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .lang-en {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    .score-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .score-high {
        background-color: #d4edda;
        color: #155724;
    }
    .score-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .score-low {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;600&display=swap" rel="stylesheet">
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.pipeline_loaded = False

if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "current_results" not in st.session_state:
    st.session_state.current_results = None

if "page_number" not in st.session_state:
    st.session_state.page_number = 0


@st.cache_resource
def load_pipeline():
    """Load retrieval pipeline with full model warming (cached)."""
    pipeline = RetrievalPipeline(index_dir=str(INDEX_DIR))

    # Load indexes
    status = pipeline.load_indexes()

    # Pre-warm ALL models to avoid delay on first query
    import time

    # Warm up query processing (Module B)
    try:
        import sys

        module_b_path = str(
            PROJECT_ROOT
            / "src"
            / "Module B — Query Processing & Cross-Lingual Handling"
        )
        if module_b_path not in sys.path:
            sys.path.insert(0, module_b_path)
        from query_pipeline import process_complete_query

        # Dummy query to initialize all NLP models
        process_complete_query("test query", target_lang="bn")
    except:
        pass

    # Warm up semantic model (loads sentence transformer)
    if pipeline.semantic_index:
        try:
            pipeline.semantic_index.search("warm up query", top_k=1, preprocess=False)
        except:
            pass

    # Warm up hybrid retriever
    try:
        pipeline.hybrid_retriever.search(
            "initialization", top_k=1, preprocess=False, cross_lingual=False
        )
    except:
        pass

    return pipeline, status


def format_score_badge(score, confidence="high"):
    """Format score as colored badge."""
    if score >= 0.7 or confidence == "high":
        css_class = "score-high"
    elif score >= 0.4 or confidence == "medium":
        css_class = "score-medium"
    else:
        css_class = "score-low"

    return f'<span class="score-badge {css_class}">{score:.3f}</span>'


def display_result(result, rank):
    """Display a single search result with language badge."""
    doc_id = result.get("doc_id", "Unknown")
    score = result.get("score", result.get("final_score", 0.0))
    confidence = result.get("confidence", "high")
    warnings = result.get("warnings", [])
    metadata = result.get("metadata", {})

    title = metadata.get("title", doc_id)
    source = metadata.get("source", "Unknown")
    language = metadata.get("language", "Unknown")
    url = metadata.get("url", "")

    # Determine language badge
    is_bangla = language.lower() in ["bangla", "bn"]
    lang_class = "lang-bn" if is_bangla else "lang-en"
    lang_text = "বাংলা" if is_bangla else "English"
    title_class = "result-title-bn" if is_bangla else "result-title"

    # Create view article link
    if url:
        view_link = f'<a href="{url}" target="_blank" style="color: #1f77b4; text-decoration: none; font-size: 0.85rem;">🔗 View Article</a>'
    else:
        view_link = f'<span style="color: #999; font-size: 0.85rem;">🆔 {doc_id}</span>'

    # Create result card
    st.markdown(
        f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex-grow: 1;">
                <span class="lang-badge {lang_class}">{lang_text}</span>
                <span style="color: #666; font-size: 0.85rem;">#{rank} • {source}</span>
                <div class="{title_class}" style="margin-top: 0.3rem;">{title}</div>
                <div class="result-meta" style="margin-top: 0.2rem;">
                    {view_link}
                </div>
            </div>
            <div style="text-align: right;">
                {format_score_badge(score, confidence)}
                <div style="font-size: 0.8rem; color: #999; margin-top: 0.2rem;">
                    {confidence.upper()}
                </div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Display warnings if any
    if warnings:
        st.markdown(
            f"<div style='margin-top: 0.5rem; font-size: 0.85rem; color: #856404;'>⚠️ {', '.join(warnings)}</div>",
            unsafe_allow_html=True,
        )

    # Display score breakdown for hybrid
    breakdown = result.get("scores_breakdown", result.get("scores", {}))
    if breakdown and len(breakdown) > 1:
        breakdown_text = " | ".join(
            [f"{k.upper()}: {v:.3f}" for k, v in breakdown.items() if v > 0]
        )
        if breakdown_text:
            st.markdown(
                f"<div style='margin-top: 0.3rem; font-size: 0.8rem; color: #666;'>📊 {breakdown_text}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


def paginate_results(results, page_size=10):
    """Paginate results."""
    total_results = len(results)
    total_pages = (total_results + page_size - 1) // page_size

    start_idx = st.session_state.page_number * page_size
    end_idx = min(start_idx + page_size, total_results)

    return results[start_idx:end_idx], total_results, total_pages


def check_indexes():
    """Check if required indexes exist."""
    semantic_path = INDEX_DIR / "semantic"

    if not INDEX_DIR.exists():
        return False, f"Index directory not found: {INDEX_DIR}"

    if not semantic_path.exists():
        return (
            False,
            f"Semantic index not found. Build indexes first:\n\n```bash\npython -m src.module1_data_acquisition.indexing.build_index --data data/metadata.csv\n```",
        )

    # Check if semantic index has required files
    required_files = ["embeddings.npy", "doc_ids.json", "metadata.json"]
    missing = [f for f in required_files if not (semantic_path / f).exists()]

    if missing:
        return False, f"Semantic index incomplete. Missing: {', '.join(missing)}"

    return True, "OK"


# Main UI
st.markdown(
    '<div class="main-header">🌐 CLIR News Retrieval System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align: center; color: #666;">Cross-Lingual Information Retrieval: Search in English or বাংলা, find results in BOTH languages</p>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # Check indexes first
    indexes_ok, message = check_indexes()
    if not indexes_ok:
        st.error("❌ Indexes Not Found")
        st.warning(message)
        st.info("Build indexes before starting frontend.")
        st.stop()

    # Load pipeline
    if not st.session_state.pipeline_loaded:
        progress_text = st.empty()
        progress_bar = st.progress(0)

        try:
            progress_text.text("🔄 Loading indexes...")
            progress_bar.progress(20)

            pipeline, status = load_pipeline()

            progress_text.text("🔥 Warming up NLP models...")
            progress_bar.progress(60)

            # Additional warming with progress
            progress_text.text("🚀 Initializing retrieval system...")
            progress_bar.progress(80)

            st.session_state.pipeline = pipeline
            st.session_state.pipeline_loaded = True

            progress_text.text("✅ Ready!")
            progress_bar.progress(100)

            time.sleep(0.5)  # Brief pause to show completion
            progress_text.empty()
            progress_bar.empty()

            # Show loaded indexes
            st.success("✅ Pipeline loaded and ready!")
            available = pipeline.get_available_methods()
            if available:
                st.info(f"Available: {', '.join(available)}")
            else:
                st.warning("No retrieval methods available")

        except Exception as e:
            progress_text.empty()
            progress_bar.empty()
            st.error(f"Failed to load pipeline: {e}")
            import traceback

            st.code(traceback.format_exc())
            st.stop()

    st.divider()

    # Retrieval settings
    st.subheader("🎯 Retrieval Settings")

    method = st.selectbox(
        "Method",
        ["hybrid", "semantic", "whoosh"],
        help="Hybrid (recommended) combines BM25 + Semantic for cross-lingual search. 'whoosh' is the lexical/BM25 index.",
    )

    top_k = st.slider("Results per query", 5, 50, 20, 5)

    # Cross-lingual search is automatic with hybrid
    st.info(
        "💡 **Hybrid mode** automatically searches in BOTH Bangla and English regardless of your query language."
    )

    st.divider()

    # Search tips
    st.subheader("💡 Search Tips")
    st.markdown(
        """
    - **English query**: `climate change`
    - **Bangla query**: `জলবায়ু পরিবর্তন`
    - Results will include BOTH languages
    """
    )

    st.divider()

    # Query history
    st.subheader("📜 Recent Queries")
    if st.session_state.search_history:
        for i, query in enumerate(reversed(st.session_state.search_history[-5:])):
            st.text(f"{i+1}. {query[:30]}...")
    else:
        st.text("No queries yet")


# Main content
tab1, tab2 = st.tabs(["🔎 Single Query", "📝 Batch Queries"])

# Tab 1: Single Query
with tab1:
    st.header("Single Query Search")

    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., climate change, জলবায়ু পরিবর্তন, COVID-19 Bangladesh",
        key="single_query",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🔄 Clear", use_container_width=True)

    if clear_button:
        st.session_state.current_results = None
        st.session_state.page_number = 0
        st.experimental_rerun()

    if search_button and query:
        # Add to history
        if query not in st.session_state.search_history:
            st.session_state.search_history.append(query)

        # Detect query language
        detected_lang = "en"
        for char in query:
            if "\u0980" <= char <= "\u09ff":
                detected_lang = "bn"
                break

        # Search
        with st.spinner(f"Searching with {method.upper()} (Cross-Lingual)..."):
            start_time = time.time()

            try:
                result = st.session_state.pipeline.search(
                    query,
                    method=method,
                    top_k=top_k,
                    target_lang=None,  # Let hybrid handle CLIR internally
                    preprocess=True,
                )

                elapsed = time.time() - start_time

                # Store results
                st.session_state.current_results = result
                st.session_state.current_results["detected_lang"] = detected_lang
                st.session_state.page_number = 0

                # Display timing
                st.success(f"✅ Search completed in {elapsed*1000:.0f}ms")

            except Exception as e:
                st.error(f"Search failed: {e}")
                import traceback

                st.code(traceback.format_exc())

    # Display results
    if st.session_state.current_results:
        result = st.session_state.current_results
        results = result.get("results", [])
        detected_lang = result.get("detected_lang", "en")

        if results:
            # Count results by language
            bn_count = sum(
                1
                for r in results
                if r.get("metadata", {}).get("language", "").lower() in ["bangla", "bn"]
            )
            en_count = len(results) - bn_count

            # Display CLIR info box
            lang_name = "Bangla (বাংলা)" if detected_lang == "bn" else "English"
            other_lang = "English" if detected_lang == "bn" else "Bangla (বাংলা)"
            st.markdown(
                f"""
            <div class="clir-info-box">
                <strong style="color: #1f77b4; font-size: 1.1rem;">🌐 Cross-Lingual Search Results</strong><br>
                <span style="color: #333;">Query language: <strong>{lang_name}</strong></span><br>
                <span style="color: #333;">Results: 🇧🇩 Bangla: <strong style="color: #2e7d32;">{bn_count}</strong> | 🇬🇧 English: <strong style="color: #1565c0;">{en_count}</strong></span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Display query info
            query_info = result.get("query", {})

            with st.expander("📋 Query Processing Details", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Original", query_info.get("original", "N/A"))
                col2.metric("Language", query_info.get("language", "N/A").upper())
                if query_info.get("translated"):
                    col3.metric("Translated", query_info.get("translated", "N/A"))

            st.divider()

            # Pagination
            page_results, total, total_pages = paginate_results(results, page_size=10)

            st.subheader(
                f"📄 Results (Page {st.session_state.page_number + 1}/{total_pages})"
            )
            st.text(f"Showing {len(page_results)} of {total} results")

            # Display results
            for i, res in enumerate(
                page_results, start=st.session_state.page_number * 10 + 1
            ):
                display_result(res, i)

            # Pagination controls
            st.divider()
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

            with col1:
                if st.button("⏮️ First", disabled=st.session_state.page_number == 0):
                    st.session_state.page_number = 0
                    st.experimental_rerun()

            with col2:
                if st.button("◀️ Prev", disabled=st.session_state.page_number == 0):
                    st.session_state.page_number -= 1
                    st.experimental_rerun()

            with col3:
                st.text(f"Page {st.session_state.page_number + 1} of {total_pages}")

            with col4:
                if st.button(
                    "Next ▶️", disabled=st.session_state.page_number >= total_pages - 1
                ):
                    st.session_state.page_number += 1
                    st.experimental_rerun()

            with col5:
                if st.button(
                    "Last ⏭️", disabled=st.session_state.page_number >= total_pages - 1
                ):
                    st.session_state.page_number = total_pages - 1
                    st.experimental_rerun()
        else:
            st.warning("No results found.")


# Tab 2: Batch Queries
with tab2:
    st.header("Batch Query Search")

    queries_text = st.text_area(
        "Enter queries (one per line)",
        placeholder="climate change\nCOVID-19 Bangladesh\nজলবায়ু পরিবর্তন\neconomy growth",
        height=200,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        batch_search_button = st.button(
            "🔍 Search All", type="primary", use_container_width=True
        )
    with col2:
        if st.button(
            "📥 Download Results",
            use_container_width=True,
            disabled=not st.session_state.get("batch_results"),
        ):
            if st.session_state.get("batch_results"):
                df = pd.DataFrame(st.session_state.batch_results)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV", csv, "search_results.csv", "text/csv"
                )

    if batch_search_button and queries_text:
        queries = [q.strip() for q in queries_text.split("\n") if q.strip()]

        if queries:
            st.info(f"Processing {len(queries)} queries...")

            batch_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, query in enumerate(queries):
                status_text.text(
                    f"Processing query {i+1}/{len(queries)}: {query[:50]}..."
                )

                try:
                    result = st.session_state.pipeline.search(
                        query, method=method, top_k=top_k, target_lang=None
                    )

                    # Store top 3 results for each query
                    for rank, res in enumerate(result.get("results", [])[:3], 1):
                        batch_results.append(
                            {
                                "query": query,
                                "rank": rank,
                                "doc_id": res.get("doc_id"),
                                "title": res.get("metadata", {}).get("title", ""),
                                "score": res.get("score"),
                                "confidence": res.get("confidence"),
                                "source": res.get("metadata", {}).get("source", ""),
                                "language": res.get("metadata", {}).get("language", ""),
                            }
                        )

                except Exception as e:
                    st.error(f"Failed on query '{query}': {e}")

                progress_bar.progress((i + 1) / len(queries))

            status_text.text("✅ Batch processing complete!")
            st.session_state.batch_results = batch_results

            # Display summary
            if batch_results:
                st.success(
                    f"Processed {len(queries)} queries, found {len(batch_results)} results"
                )

                # Display as table
                df = pd.DataFrame(batch_results)
                st.dataframe(df, use_container_width=True)


# Add Module D - Evaluation Section
st.divider()
st.markdown("## Evaluation & Metrics")

# Add evaluation module to path
try:
    module_d_path = PROJECT_ROOT / "src" / "Module D — Ranking, Scoring, & Evaluation"
    if str(module_d_path) not in sys.path:
        sys.path.insert(0, str(module_d_path))
    
    from evaluation_metrics import EvaluationMetrics
    from error_analysis import ErrorAnalyzer
    from relevance_labeling import RelevanceLabeler
    from ranking_scorer import RankingScorer
    
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False

if EVALUATION_AVAILABLE:
    eval_col1, eval_col2, eval_col3 = st.columns(3)
    
    with eval_col1:
        if st.button("📈 Run Evaluation", key="eval_btn"):
            st.info("🔄 Loading evaluation system...")
            
            try:
                import json
                
                # Load test queries
                test_queries_path = module_d_path / "test_queries.json"
                if test_queries_path.exists():
                    with open(test_queries_path, "r", encoding="utf-8") as f:
                        test_queries = json.load(f)
                    
                    st.success(f"✅ Loaded {len(test_queries)} test queries")
                    
                    # Show sample metrics
                    st.markdown("### Sample Evaluation Results")
                    
                    total_queries = len(test_queries)
                    total_relevant = sum(len(q.get("relevant_docs", [])) for q in test_queries.values())
                    total_retrieved = sum(len(q.get("retrieved_docs", [])) for q in test_queries.values())
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Test Queries", total_queries)
                    with metric_col2:
                        st.metric("Relevant Docs", total_relevant)
                    with metric_col3:
                        st.metric("Retrieved Docs", total_retrieved)
                    
                    # Calculate and display metrics
                    st.markdown("### Metrics Breakdown")
                    
                    metrics_data = []
                    for query_id, query_data in test_queries.items():
                        relevant = query_data.get("relevant_docs", [])
                        retrieved = query_data.get("retrieved_docs", [])
                        
                        p10 = EvaluationMetrics.precision_at_k(relevant, retrieved, 10)
                        r50 = EvaluationMetrics.recall_at_k(relevant, retrieved, 50)
                        ndcg = EvaluationMetrics.ndcg(relevant, retrieved, 10)
                        mrr = EvaluationMetrics.reciprocal_rank(relevant, retrieved)
                        
                        metrics_data.append({
                            "Query ID": query_id,
                            "P@10": f"{p10:.3f}",
                            "R@50": f"{r50:.3f}",
                            "nDCG@10": f"{ndcg:.3f}",
                            "MRR": f"{mrr:.3f}"
                        })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Batch evaluation
                    batch_queries = {
                        qid: {
                            "relevant": q.get("relevant_docs", []),
                            "retrieved": q.get("retrieved_docs", [])
                        }
                        for qid, q in test_queries.items()
                    }
                    
                    per_query, summary = EvaluationMetrics.evaluate_batch(batch_queries, k=10)
                    
                    st.markdown("### Overall Performance")
                    
                    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                    with perf_col1:
                        st.metric("Avg P@10", f"{summary['mean_precision_at_10']:.3f}", 
                                 "✓" if summary['mean_precision_at_10'] >= 0.6 else "✗")
                    with perf_col2:
                        st.metric("Avg R@50", f"{summary['mean_recall_at_50']:.3f}",
                                 "✓" if summary['mean_recall_at_50'] >= 0.5 else "✗")
                    with perf_col3:
                        st.metric("Avg nDCG@10", f"{summary['mean_ndcg_at_10']:.3f}",
                                 "✓" if summary['mean_ndcg_at_10'] >= 0.5 else "✗")
                    with perf_col4:
                        st.metric("Avg MRR", f"{summary['mean_reciprocal_rank']:.3f}",
                                 "✓" if summary['mean_reciprocal_rank'] >= 0.4 else "✗")
                    
                    st.success("✅ Evaluation Complete!")
                else:
                    st.warning("⚠️ Test queries file not found")
                    
            except Exception as e:
                st.error(f"❌ Evaluation error: {e}")
    
    with eval_col2:
        if st.button("🔍 Analyze Errors", key="error_btn"):
            st.info("🔄 Loading error analyzer...")
            
            try:
                analyzer = ErrorAnalyzer()
                
                # Add sample errors for demonstration
                analyzer.add_translation_failure(
                    query_id="demo_1",
                    query_text="example query",
                    query_language="english",
                    original_query="চেয়ার",
                    mistranslated_query="Chairman",
                    expected_docs=["doc_1"],
                    retrieved_docs=["doc_2"],
                    example="Translation error detected"
                )
                
                st.markdown("### Error Analysis Summary")
                
                summary = analyzer.summarize_errors()
                error_col1, error_col2 = st.columns(2)
                
                with error_col1:
                    st.metric("Total Errors", len(analyzer.error_cases))
                
                with error_col2:
                    st.metric("Error Types", len(summary))
                
                if summary:
                    st.markdown("### Errors by Type")
                    error_data = [{"Error Type": k, "Count": v} for k, v in summary.items()]
                    error_df = pd.DataFrame(error_data)
                    st.bar_chart(error_df.set_index("Error Type"))
                
                st.info("✅ Error analysis complete. Check Module D for detailed reports.")
                    
            except Exception as e:
                st.error(f"❌ Error analysis failed: {e}")
    
    with eval_col3:
        if st.button("🏷️ View Labels", key="labels_btn"):
            st.info("🔄 Loading relevance labels...")
            
            try:
                labeler = RelevanceLabeler()
                
                # Try to load existing labels
                labels_path = module_d_path / "labels.csv"
                if labels_path.exists():
                    labeler.load_from_csv(str(labels_path))
                    stats = labeler.get_statistics()
                    
                    st.markdown("### Labeling Statistics")
                    
                    label_col1, label_col2, label_col3, label_col4 = st.columns(4)
                    
                    with label_col1:
                        st.metric("Total Labels", stats['total_labels'])
                    with label_col2:
                        st.metric("Relevant", stats['relevant_count'])
                    with label_col3:
                        st.metric("Not Relevant", stats['not_relevant_count'])
                    with label_col4:
                        st.metric("Queries", stats['unique_queries'])
                    
                    st.markdown(f"**Relevant %**: {stats['relevant_percentage']:.1f}%")
                    st.markdown(f"**Avg Confidence**: {stats['average_confidence']:.2f}/3.0")
                else:
                    st.warning("No labels file found. Create one using Module D.")
                    st.info("Generate template: `python evaluate.py --create-template`")
                    
            except Exception as e:
                st.error(f"❌ Label loading failed: {e}")

else:
    st.warning("⚠️ Module D (Evaluation) not available. Install/configure required.")


# Footer
st.divider()
st.markdown(
    """
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>CLIR News Retrieval System | Supports English ↔ Bangla Cross-Lingual Search</small>
</div>
""",
    unsafe_allow_html=True,
)
