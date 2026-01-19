"""
Module C - Model 2: Fuzzy + Transliteration Matching

This module implements string-level similarity matching for handling:
- Spelling variations ("colour" vs "color")
- Typos ("clmate" vs "climate")
- Cross-script matching (English ↔ বাংলা)
- Transliteration variants ("Dhaka" vs "ঢাকা")

WHY FUZZY MATCHING FOR CLIR?
-----------------------------
- Handles spelling variations that lexical models miss
- Useful for proper nouns with multiple spellings
- Helps with user typos in queries
- Bridges English and Bangla through transliteration

FUZZY MATCHING METHODS:
-----------------------
1. Levenshtein Distance (Edit Distance)
   - Counts minimum edits (insert/delete/replace) to transform one string to another
   - "cat" -> "car" = 1 edit (replace 't' with 'r')

2. Sequence Matcher (difflib)
   - Finds longest contiguous matching subsequences
   - Better for longer strings with similar structures

3. Character N-gram Jaccard Similarity
   - Compares sets of character n-grams
   - "climate" -> {"cli", "lim", "ima", "mat", "ate"}
   - Robust to word reordering

TRANSLITERATION:
----------------
- Converts script while preserving pronunciation
- "Dhaka" (English) ↔ "ঢাকা" (Bangla)
- Uses phonetic mapping between scripts

FAILURE CASES:
--------------
- SEMANTIC SIMILARITY: Fuzzy matching fails for meaning-based similarity
  - "car" and "automobile" have low string similarity but same meaning

- LONG TEXTS: Edit distance becomes unreliable for full documents
  - Best used for titles, keywords, entity names

- SCRIPT NORMALIZATION: Some characters may not have exact transliterations
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Set
import re

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Transliteration mappings (Bangla ↔ English phonetic)
BANGLA_TO_ENGLISH = {
    "অ": "o",
    "আ": "a",
    "ই": "i",
    "ঈ": "ee",
    "উ": "u",
    "ঊ": "oo",
    "এ": "e",
    "ঐ": "oi",
    "ও": "o",
    "ঔ": "ou",
    "ক": "k",
    "খ": "kh",
    "গ": "g",
    "ঘ": "gh",
    "ঙ": "ng",
    "চ": "ch",
    "ছ": "chh",
    "জ": "j",
    "ঝ": "jh",
    "ঞ": "n",
    "ট": "t",
    "ঠ": "th",
    "ড": "d",
    "ঢ": "dh",
    "ণ": "n",
    "ত": "t",
    "থ": "th",
    "দ": "d",
    "ধ": "dh",
    "ন": "n",
    "প": "p",
    "ফ": "ph",
    "ব": "b",
    "ভ": "bh",
    "ম": "m",
    "য": "j",
    "র": "r",
    "ল": "l",
    "শ": "sh",
    "ষ": "sh",
    "স": "s",
    "হ": "h",
    "ড়": "r",
    "ঢ়": "rh",
    "য়": "y",
    "ৎ": "t",
    "ং": "ng",
    "ঃ": "h",
    "ঁ": "n",
    "া": "a",
    "ি": "i",
    "ী": "ee",
    "ু": "u",
    "ূ": "oo",
    "ে": "e",
    "ৈ": "oi",
    "ো": "o",
    "ৌ": "ou",
    "্": "",
}

# Common English to Bangla place name mappings
COMMON_TRANSLITERATIONS = {
    # Places
    "dhaka": "ঢাকা",
    "bangladesh": "বাংলাদেশ",
    "chittagong": "চট্টগ্রাম",
    "sylhet": "সিলেট",
    "rajshahi": "রাজশাহী",
    "khulna": "খুলনা",
    "comilla": "কুমিল্লা",
    "cox's bazar": "কক্সবাজার",
    "coxs bazar": "কক্সবাজার",
    # Common terms
    "prime minister": "প্রধানমন্ত্রী",
    "government": "সরকার",
    "university": "বিশ্ববিদ্যালয়",
    "police": "পুলিশ",
    "cricket": "ক্রিকেট",
    "football": "ফুটবল",
    # People (examples)
    "sheikh hasina": "শেখ হাসিনা",
    "shakib": "শাকিব",
}

# Reverse mapping (Bangla -> English)
ENGLISH_TO_BANGLA = {v: k for k, v in COMMON_TRANSLITERATIONS.items()}


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.

    Uses dynamic programming for O(mn) time complexity.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Minimum number of edits to transform s1 to s2
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Calculate costs for each operation
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calculate normalized Levenshtein similarity in [0, 1].

    Formula: 1 - (edit_distance / max_length)

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score in [0, 1] (1 = identical)
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    max_len = max(len(s1), len(s2))
    distance = levenshtein_distance(s1.lower(), s2.lower())

    return 1.0 - (distance / max_len)


def sequence_matcher_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity using difflib's SequenceMatcher.

    Finds longest contiguous matching subsequences.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity ratio in [0, 1]
    """
    from difflib import SequenceMatcher

    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def ngram_jaccard_similarity(s1: str, s2: str, n: int = 3) -> float:
    """
    Calculate character n-gram Jaccard similarity.

    Jaccard = |intersection| / |union|

    Args:
        s1: First string
        s2: Second string
        n: N-gram size (default: 3 for trigrams)

    Returns:
        Jaccard similarity in [0, 1]
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    def get_ngrams(text: str, n: int) -> Set[str]:
        text = text.lower()
        # Pad string to handle short words
        text = f"{'$' * (n-1)}{text}{'$' * (n-1)}"
        return set(text[i : i + n] for i in range(len(text) - n + 1))

    ngrams1 = get_ngrams(s1, n)
    ngrams2 = get_ngrams(s2, n)

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    return intersection / union if union > 0 else 0.0


def fuzzy_match(query: str, target: str, method: str = "combined") -> float:
    """
    Calculate fuzzy match score between query and target.

    Args:
        query: Query string
        target: Target string to match against
        method: Matching method
            - "levenshtein": Edit distance based
            - "sequence": SequenceMatcher based
            - "ngram": Character n-gram Jaccard
            - "combined": Average of all methods (default)

    Returns:
        Similarity score in [0, 1]
    """
    if method == "levenshtein":
        return levenshtein_similarity(query, target)
    elif method == "sequence":
        return sequence_matcher_similarity(query, target)
    elif method == "ngram":
        return ngram_jaccard_similarity(query, target)
    elif method == "combined":
        # Weighted combination for robustness
        lev = levenshtein_similarity(query, target)
        seq = sequence_matcher_similarity(query, target)
        ngram = ngram_jaccard_similarity(query, target)
        return 0.4 * lev + 0.3 * seq + 0.3 * ngram
    else:
        logger.warning(f"Unknown fuzzy method: {method}, using combined")
        return fuzzy_match(query, target, "combined")


def transliterate_bangla_to_english(text: str) -> str:
    """
    Transliterate Bangla text to English phonetics.

    Args:
        text: Bangla text

    Returns:
        Phonetic English representation
    """
    # Check for common known mappings first
    text_lower = text.lower()
    if text_lower in ENGLISH_TO_BANGLA:
        return ENGLISH_TO_BANGLA[text_lower]

    # Character-by-character transliteration
    result = []
    for char in text:
        if char in BANGLA_TO_ENGLISH:
            result.append(BANGLA_TO_ENGLISH[char])
        elif char.isascii():
            result.append(char)
        elif char.isspace():
            result.append(" ")
        # Skip unknown characters

    return "".join(result)


def transliterate_english_to_bangla(text: str) -> str:
    """
    Transliterate English text to Bangla.

    Uses common mappings for known words.

    Args:
        text: English text

    Returns:
        Bangla representation (or original if no mapping found)
    """
    text_lower = text.lower().strip()

    # Check for exact matches
    if text_lower in COMMON_TRANSLITERATIONS:
        return COMMON_TRANSLITERATIONS[text_lower]

    # Check for partial matches (multi-word)
    for eng, ban in COMMON_TRANSLITERATIONS.items():
        if eng in text_lower:
            text_lower = text_lower.replace(eng, ban)

    return text_lower


def get_transliteration_variants(text: str, source_lang: str = None) -> List[str]:
    """
    Generate transliteration variants for cross-script matching.

    Args:
        text: Input text
        source_lang: Source language ('bn' or 'en') or None for auto-detect

    Returns:
        List of variant strings (including original)
    """
    variants = [text]

    # Auto-detect language if not specified
    if source_lang is None:
        # Check for Bangla characters
        has_bangla = any("\u0980" <= c <= "\u09ff" for c in text)
        source_lang = "bn" if has_bangla else "en"

    if source_lang == "bn":
        # Add English transliteration
        english_variant = transliterate_bangla_to_english(text)
        if english_variant and english_variant != text:
            variants.append(english_variant)
    else:
        # Add Bangla transliteration
        bangla_variant = transliterate_english_to_bangla(text)
        if bangla_variant and bangla_variant != text.lower():
            variants.append(bangla_variant)

    return variants


def retrieve_fuzzy(
    query: str,
    documents: List[Dict[str, Any]],
    text_field: str = "title",
    top_k: int = 10,
    min_score: float = 0.3,
    method: str = "combined",
    use_transliteration: bool = True,
) -> List[Dict[str, Any]]:
    """
    Retrieve documents using fuzzy matching.

    Best used for matching against short fields like titles or keywords.

    Args:
        query: Search query
        documents: List of document dicts
        text_field: Field to match against (title, keywords, etc.)
        top_k: Number of results to return
        min_score: Minimum similarity threshold
        method: Fuzzy matching method
        use_transliteration: Whether to include transliteration variants

    Returns:
        List of results with doc_id, score, and matched_text
    """
    if not query or not documents:
        return []

    start_time = time.time()

    # Generate query variants (including transliterations)
    query_variants = [query]
    if use_transliteration:
        query_variants = get_transliteration_variants(query)

    results = []

    for doc in documents:
        doc_id = doc.get("id", doc.get("doc_id", ""))
        target_text = doc.get(text_field, "")

        if not target_text:
            continue

        # Calculate best match score across query variants
        best_score = 0.0
        best_variant = query

        for variant in query_variants:
            score = fuzzy_match(variant, target_text, method=method)
            if score > best_score:
                best_score = score
                best_variant = variant

        if best_score >= min_score:
            results.append(
                {
                    "doc_id": doc_id,
                    "score": best_score,
                    "score_normalized": best_score,  # Already in [0, 1]
                    "matched_text": target_text,
                    "query_variant": best_variant,
                    "method": "fuzzy",
                }
            )

    # Sort by score and limit to top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_k]

    # Add ranks
    for i, r in enumerate(results, 1):
        r["rank"] = i

    elapsed = time.time() - start_time
    logger.debug(
        f"Fuzzy search completed in {elapsed*1000:.2f}ms ({len(results)} results)"
    )

    return results


def retrieve_fuzzy_per_term(
    query: str,
    documents: List[Dict[str, Any]],
    text_field: str = "content",
    top_k: int = 10,
    min_score: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Retrieve documents by matching individual query terms.

    Splits query into terms and finds documents with fuzzy-matching terms.

    Args:
        query: Search query (will be tokenized)
        documents: List of document dicts
        text_field: Field to search in
        top_k: Number of results
        min_score: Minimum per-term match threshold

    Returns:
        List of results with aggregated scores
    """
    if not query or not documents:
        return []

    # Tokenize query
    query_terms = query.lower().split()
    if not query_terms:
        return []

    doc_scores = {}

    for doc in documents:
        doc_id = doc.get("id", doc.get("doc_id", ""))
        text = doc.get(text_field, "").lower()

        if not text:
            continue

        # Tokenize document
        doc_terms = set(re.findall(r"\b[\w-]+\b", text))

        # Calculate match score for each query term
        term_scores = []
        for qterm in query_terms:
            best_match = 0.0
            for dterm in doc_terms:
                score = fuzzy_match(qterm, dterm, method="levenshtein")
                if score > best_match:
                    best_match = score
            term_scores.append(best_match)

        # Aggregate score: average of best matches per term
        if term_scores:
            avg_score = sum(term_scores) / len(term_scores)
            if avg_score >= min_score:
                doc_scores[doc_id] = avg_score

    # Sort and format results
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for rank, (doc_id, score) in enumerate(sorted_docs, 1):
        results.append(
            {
                "doc_id": doc_id,
                "score": score,
                "score_normalized": score,
                "rank": rank,
                "method": "fuzzy_terms",
            }
        )

    return results


class FuzzyMatcher:
    """
    Fuzzy Matching Index for document retrieval.

    Pre-processes documents for faster fuzzy matching.
    """

    def __init__(self, method: str = "combined", use_transliteration: bool = True):
        """
        Initialize fuzzy matcher.

        Args:
            method: Fuzzy matching method
            use_transliteration: Enable transliteration variants
        """
        self.method = method
        self.use_transliteration = use_transliteration
        self.documents = []
        self.doc_ids = []
        self.titles = []
        self.keywords = []
        self._is_built = False

    def build(
        self,
        documents: List[Dict[str, Any]],
        title_field: str = "title",
        keyword_field: str = "keywords",
    ) -> None:
        """
        Build fuzzy matching index.

        Args:
            documents: List of document dicts
            title_field: Field containing document title
            keyword_field: Field containing keywords (list or string)
        """
        self.documents = documents
        self.doc_ids = []
        self.titles = []
        self.keywords = []

        for doc in documents:
            doc_id = doc.get("id", doc.get("doc_id", str(len(self.doc_ids))))
            title = doc.get(title_field, "")
            keywords = doc.get(keyword_field, [])

            if isinstance(keywords, str):
                keywords = keywords.split(",")

            self.doc_ids.append(doc_id)
            self.titles.append(title.lower().strip())
            self.keywords.append([k.lower().strip() for k in keywords if k])

        self._is_built = True
        logger.info(f"Fuzzy matcher built with {len(documents)} documents")

    def search(
        self, query: str, top_k: int = 10, min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search using fuzzy matching.

        Matches against titles and keywords.
        """
        if not self._is_built:
            logger.error("Fuzzy matcher not built. Call build() first.")
            return []

        if not query:
            return []

        query_variants = [query.lower()]
        if self.use_transliteration:
            query_variants = [v.lower() for v in get_transliteration_variants(query)]

        scores = []

        for i, doc_id in enumerate(self.doc_ids):
            title = self.titles[i]
            keywords = self.keywords[i]

            best_score = 0.0
            matched_on = ""

            for variant in query_variants:
                # Match title
                title_score = fuzzy_match(variant, title, self.method)
                if title_score > best_score:
                    best_score = title_score
                    matched_on = "title"

                # Match keywords
                for kw in keywords:
                    kw_score = fuzzy_match(variant, kw, self.method)
                    if kw_score > best_score:
                        best_score = kw_score
                        matched_on = f"keyword:{kw}"

            if best_score >= min_score:
                scores.append(
                    {
                        "doc_id": doc_id,
                        "score": best_score,
                        "score_normalized": best_score,
                        "matched_on": matched_on,
                        "method": "fuzzy",
                    }
                )

        # Sort and rank
        scores.sort(key=lambda x: x["score"], reverse=True)
        results = scores[:top_k]

        for i, r in enumerate(results, 1):
            r["rank"] = i

        return results


# Command line interface
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Fuzzy + Transliteration Matching for CLIR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test fuzzy similarity between two strings
  python fuzzy_retrieval.py --compare "Dhaka" "ঢাকা"
  
  # Get transliteration variants
  python fuzzy_retrieval.py --transliterate "Bangladesh"
  python fuzzy_retrieval.py --transliterate "বাংলাদেশ"
  
  # Search documents (requires JSON file)
  python fuzzy_retrieval.py "climate" --data data/documents.json
        """,
    )

    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument(
        "--compare",
        "-c",
        nargs=2,
        metavar=("STR1", "STR2"),
        help="Compare two strings for similarity",
    )
    parser.add_argument("--transliterate", "-t", help="Get transliteration variants")
    parser.add_argument("--data", "-d", help="Path to documents JSON")
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of results")
    parser.add_argument(
        "--method",
        "-m",
        default="combined",
        choices=["levenshtein", "sequence", "ngram", "combined"],
        help="Fuzzy matching method",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.compare:
        s1, s2 = args.compare
        print(f"\nComparing: '{s1}' vs '{s2}'")
        print("-" * 40)
        print(f"Levenshtein:     {levenshtein_similarity(s1, s2):.4f}")
        print(f"SequenceMatcher: {sequence_matcher_similarity(s1, s2):.4f}")
        print(f"N-gram Jaccard:  {ngram_jaccard_similarity(s1, s2):.4f}")
        print(f"Combined:        {fuzzy_match(s1, s2, 'combined'):.4f}")

    elif args.transliterate:
        variants = get_transliteration_variants(args.transliterate)
        print(f"\nTransliteration variants for: '{args.transliterate}'")
        for v in variants:
            print(f"  → {v}")

    elif args.query and args.data:
        with open(args.data, "r", encoding="utf-8") as f:
            docs = json.load(f)

        results = retrieve_fuzzy(args.query, docs, top_k=args.top_k, method=args.method)

        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print(f"\nFuzzy Results for: {args.query}")
            print("=" * 50)
            for r in results:
                print(f"  [{r['rank']}] {r['doc_id']}: {r['score']:.4f}")
                if r.get("matched_text"):
                    print(f"      Matched: {r['matched_text'][:50]}...")
            print()

    else:
        parser.print_help()
