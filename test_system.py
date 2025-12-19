"""
Test script to verify all CLIR modules are working correctly.
Run this to ensure the system is properly installed and functional.
"""

import sys
from pathlib import Path

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text):
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}\n")


def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text):
    print(f"{RED}✗ {text}{RESET}")


def print_info(text):
    print(f"{YELLOW}ℹ {text}{RESET}")


def test_module_a():
    """Test Module A: Indexing"""
    print_header("Testing Module A: Indexing")

    try:
        # Test 1: Language Detector
        print("Test 1: Language Detector...")
        from src.module_a_indexing import LanguageDetector

        detector = LanguageDetector()

        result_en = detector.detect("This is an English sentence")
        result_bn = detector.detect("এটি একটি বাংলা বাক্য")

        assert result_en["language"] == "en", "English detection failed"
        assert result_bn["language"] == "bn", "Bangla detection failed"
        print_success("Language Detector working")

        # Test 2: Tokenizer
        print("\nTest 2: Tokenizer...")
        from src.module_a_indexing import MultilingualTokenizer

        tokenizer = MultilingualTokenizer()

        tokens_en = tokenizer.tokenize("Hello world", language="en")
        assert len(tokens_en) > 0, "Tokenization failed"
        print_success(f"Tokenizer working (tokens: {tokens_en})")

        # Test 3: NER Extractor
        print("\nTest 3: NER Extractor...")
        from src.module_a_indexing import NERExtractor

        ner = NERExtractor()

        entities = ner.extract_entities(
            "Sheikh Hasina visited New Delhi", language="en"
        )
        print_success(f"NER Extractor working (found {len(entities)} entities)")

        # Test 4: Inverted Index
        print("\nTest 4: Inverted Index...")
        from src.module_a_indexing import InvertedIndex

        index = InvertedIndex()

        index.add_document("doc1", ["hello", "world"], {"title": "Test"})
        posting = index.get_posting_list("hello")
        assert len(posting) > 0, "Index add/retrieve failed"
        print_success("Inverted Index working")

        # Test 5: Document Processor
        print("\nTest 5: Document Processor...")
        from src.module_a_indexing import DocumentProcessor

        processor = DocumentProcessor("data/raw", "test_output")
        print_success("Document Processor initialized")

        print(f"\n{GREEN}{'✓ Module A: All tests passed!'}{RESET}")
        return True

    except Exception as e:
        print_error(f"Module A failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_module_b():
    """Test Module B: Query Processing"""
    print_header("Testing Module B: Query Processing")

    try:
        # Test 1: Query Detector
        print("Test 1: Query Language Detector...")
        from src.module_b_query_processing import QueryLanguageDetector

        detector = QueryLanguageDetector()

        result = detector.detect_query_language("education policy")
        assert result["language"] in ["en", "bn"], "Query detection failed"
        print_success(f"Query Detector working (detected: {result['language']})")

        # Test 2: Query Normalizer
        print("\nTest 2: Query Normalizer...")
        from src.module_b_query_processing import QueryNormalizer

        normalizer = QueryNormalizer()

        normalized = normalizer.normalize("  HELLO   WORLD  ", language="en")
        assert normalized == "hello world", "Normalization failed"
        print_success(f"Query Normalizer working ('{normalized}')")

        # Test 3: Query Translator
        print("\nTest 3: Query Translator...")
        from src.module_b_query_processing import QueryTranslator

        translator = QueryTranslator()
        print_info("Loading translation models (this may take a moment)...")

        translated = translator.translate_to_bangla("education")
        print_success(f"Query Translator working (EN→BN: 'education' → '{translated}')")

        # Test 4: Query Expander
        print("\nTest 4: Query Expander...")
        from src.module_b_query_processing import QueryExpander

        expander = QueryExpander()

        expanded = expander.expand("education policy", language="en", max_synonyms=2)
        print_success(f"Query Expander working (expanded: '{expanded[:50]}...')")

        # Test 5: Named Entity Mapper
        print("\nTest 5: Named Entity Mapper...")
        from src.module_b_query_processing import NamedEntityMapper

        mapper = NamedEntityMapper()

        mapped = mapper.map_entity("Bangladesh", "en", "bn")
        print_success(f"NE Mapper working (Bangladesh → {mapped})")

        # Test 6: Query Pipeline
        print("\nTest 6: Query Pipeline...")
        from src.module_b_query_processing import QueryProcessor

        processor = QueryProcessor(
            enable_translation=False, enable_expansion=False  # Skip for faster test
        )

        result = processor.process_query("test query")
        assert "normalized_query" in result, "Pipeline failed"
        print_success("Query Pipeline working")

        print(f"\n{GREEN}{'✓ Module B: All tests passed!'}{RESET}")
        return True

    except Exception as e:
        print_error(f"Module B failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_module_c():
    """Test Module C: Retrieval"""
    print_header("Testing Module C: Retrieval")

    try:
        # Test 1: Lexical Retrieval
        print("Test 1: Lexical Retrieval...")
        from src.module_c_retrieval import LexicalRetriever

        documents = [
            {"doc_id": "doc1", "tokens": ["education", "policy", "new"]},
            {"doc_id": "doc2", "tokens": ["health", "minister", "visit"]},
        ]
        retriever = LexicalRetriever(documents)
        results = retriever.search_bm25(["education"], top_k=2)
        assert len(results) > 0, "BM25 search failed"
        print_success(f"Lexical Retrieval working (BM25 found {len(results)} results)")

        # Test 2: Fuzzy Matcher
        print("\nTest 2: Fuzzy Matcher...")
        from src.module_c_retrieval import FuzzyMatcher

        matcher = FuzzyMatcher()

        score = matcher.character_ngram_match("education", "educaton", n=3)
        print_success(f"Fuzzy Matcher working (similarity: {score:.2f})")

        # Test 3: Semantic Retrieval
        print("\nTest 3: Semantic Retrieval...")
        from src.module_c_retrieval import SemanticRetriever

        print_info("Loading embedding model (this may take a moment)...")

        retriever = SemanticRetriever(use_gpu=False)
        docs = [
            {"doc_id": "doc1", "content": "education policy announcement"},
            {"doc_id": "doc2", "content": "health minister visits hospital"},
        ]
        retriever.encode_documents(docs)
        results = retriever.search_semantic("teaching policy", top_k=2)
        assert len(results) > 0, "Semantic search failed"
        print_success(f"Semantic Retrieval working (found {len(results)} results)")

        # Test 4: Hybrid Ranker
        print("\nTest 4: Hybrid Ranker...")
        from src.module_c_retrieval import HybridRanker

        ranker = HybridRanker()

        scores = {
            "bm25": {"doc1": 0.8, "doc2": 0.5},
            "semantic": {"doc1": 0.7, "doc2": 0.9},
        }
        results = ranker.rank_documents(scores, method="weighted")
        assert len(results) > 0, "Hybrid ranking failed"
        print_success(f"Hybrid Ranker working (ranked {len(results)} documents)")

        # Test 5: Retrieval Engine
        print("\nTest 5: Retrieval Engine...")
        from src.module_c_retrieval import RetrievalEngine
        from src.module_a_indexing import InvertedIndex

        index = InvertedIndex()
        index.add_document("doc1", ["education", "policy"], {"title": "Test Doc"})

        docs = [
            {"doc_id": "doc1", "tokens": ["education", "policy"], "content": "test"}
        ]
        engine = RetrievalEngine(index, docs, enable_semantic=False)
        print_success("Retrieval Engine initialized")

        print(f"\n{GREEN}{'✓ Module C: All tests passed!'}{RESET}")
        return True

    except Exception as e:
        print_error(f"Module C failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_dependencies():
    """Test all required dependencies"""
    print_header("Testing Dependencies")

    dependencies = {
        "langdetect": "Language Detection",
        "spacy": "NLP Processing",
        "transformers": "Translation Models",
        "torch": "PyTorch (Deep Learning)",
        "nltk": "Query Expansion",
        "rank_bm25": "BM25 Retrieval",
        "sklearn": "Machine Learning",
        "numpy": "Numerical Computing",
        "sentence_transformers": "Semantic Embeddings",
    }

    all_good = True
    for package, description in dependencies.items():
        try:
            __import__(package)
            print_success(f"{package:25} - {description}")
        except ImportError:
            print_error(f"{package:25} - MISSING!")
            all_good = False

    # Check spaCy model
    print("\nChecking spaCy models...")
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        print_success("en_core_web_sm model installed")
    except:
        print_error(
            "en_core_web_sm model missing - run: python -m spacy download en_core_web_sm"
        )
        all_good = False

    # Check NLTK data
    print("\nChecking NLTK data...")
    try:
        import nltk
        from nltk.corpus import wordnet

        wordnet.synsets("test")
        print_success("NLTK WordNet data available")
    except:
        print_error("NLTK WordNet missing - run NLTK downloads")
        all_good = False

    return all_good


def main():
    """Run all tests"""
    print(f"\n{BLUE}{'*' * 60}{RESET}")
    print(f"{BLUE}{'CLIR System Test Suite':^60}{RESET}")
    print(f"{BLUE}{'*' * 60}{RESET}")

    results = {}

    # Test dependencies first
    results["dependencies"] = test_dependencies()

    # Test each module
    results["module_a"] = test_module_a()
    results["module_b"] = test_module_b()
    results["module_c"] = test_module_c()

    # Summary
    print_header("Test Summary")

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)

    for test_name, passed in results.items():
        status = f"{GREEN}PASSED{RESET}" if passed else f"{RED}FAILED{RESET}"
        print(f"{test_name.replace('_', ' ').title():20} : {status}")

    print(f"\n{'-' * 60}")
    if passed_tests == total_tests:
        print(f"{GREEN}✓ ALL TESTS PASSED ({passed_tests}/{total_tests}){RESET}")
        print(f"{GREEN}Your CLIR system is ready to use!{RESET}")
        return 0
    else:
        print(f"{RED}✗ SOME TESTS FAILED ({passed_tests}/{total_tests}){RESET}")
        print(f"{YELLOW}Please fix the errors above before proceeding.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
