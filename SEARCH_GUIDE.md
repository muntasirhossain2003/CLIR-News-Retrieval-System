# CLIR Search Interface Guide

## Quick Start

### Command Line Search

```bash
# Search with English query
python search_interface.py "osman hadi"

# Search with Bangla query
python search_interface.py "ওসমান হাদী"

# Search with any query
python search_interface.py "climate change"
python search_interface.py "জলবায়ু পরিবর্তন"
```

### Interactive Mode

```bash
python search_interface.py
```

Then enter queries interactively:

```
Enter search query: osman hadi
Enter search query: ওসমান হাদী
Enter search query: climate change
```

Commands:

- `hybrid` - Toggle hybrid ranking on/off
- `fuzzy` - Toggle fuzzy matching on/off
- `quit` or `exit` - Exit program

## Features

### Retrieval Methods

**1. TF-IDF (Lexical Baseline)**

- Fast keyword matching
- Same-language only
- Best for: Keyword-heavy queries

**2. BM25 (Improved Lexical)**

- Better ranking than TF-IDF
- Term frequency saturation
- Same-language only
- Best for: Exact keyword matches

**3. Semantic Retrieval (Cross-Lingual)** ⚠️ _Requires PyTorch_

- Cross-lingual capability (English ↔ Bangla)
- Semantic similarity
- Best for: Conceptual queries, different languages

**4. Hybrid Ranking (Optional)**

- Combines BM25 + Semantic
- Configurable weights (default: 50-50)
- Best for: General-purpose CLIR

**5. Fuzzy Matching (Optional)**

- Spelling variation tolerance
- Character-level matching
- Best for: Typos and misspellings

## Output Format

For each query, you'll see:

```
================================================================================
METHOD NAME RESULTS
================================================================================

Query: 'your query here'
Found: X documents

1. doc_id
   Language: 🇬🇧 English (or 🇧🇩 Bangla)
   Score: X.XXXX
   Preview: First 100 characters of document...

2. doc_id
   ...
```

### Comparison Summary

```
================================================================================
COMPARISON SUMMARY
================================================================================

Method          Results    Top Score    Status
--------------------------------------------------------------------------------
TFIDF           10         0.5595       ✓ Success
BM25            10         11.1596      ✓ Success
SEMANTIC        10         0.8802       ✓ Success
HYBRID          10         1.0000       ✓ Success
```

### Failure Cases

When a method fails, you'll see:

```
❌ NO RESULTS FOUND (FAILURE CASE)
   Reason: TF-IDF (0.000s) cannot match this query
```

**Common Failure Scenarios:**

1. **TF-IDF/BM25 fail on cross-lingual queries**

   - English query on Bangla documents → No results
   - Bangla query on English documents → No results
   - **Why:** Lexical methods require exact token matching

2. **All methods fail**
   - Very rare/specific terms not in corpus
   - Query contains only stopwords

## Example Results

### ✅ Success Case: English Query "osman hadi"

**TF-IDF Results:**

```
1. en_dhaka_tribune_398601_0231ac57 (Score: 0.5595)
   Preview: Ducsu protests shooting of Osman Hadi

2. en_new_age_285155_e6465ef7 (Score: 0.5247)
   Preview: Theft at Osman Hadi's Jhalakathi home
```

**BM25 Results:** (Better ranking)

```
1. en_dhaka_tribune_398601_0231ac57 (Score: 11.1596)
   Preview: Ducsu protests shooting of Osman Hadi

2. en_daily_star_961d5934d35e4d0d (Score: 9.9024)
   Preview: Faisal sued over attempted murder of Osman Hadi
```

### ❌ Failure Case: Bangla Query "ওসমান হাদী"

**TF-IDF:** No results (cannot match cross-lingually)
**BM25:** No results (cannot match cross-lingually)
**Semantic:** Would find both English and Bangla documents! _(requires PyTorch)_

## Performance

On 5,170 documents corpus:

| Operation                 | Time                            |
| ------------------------- | ------------------------------- |
| Load documents            | ~1-2 seconds                    |
| Build TF-IDF index        | ~0.03 seconds                   |
| Build BM25 index          | ~0.02 seconds                   |
| Build Semantic embeddings | ~2-10 minutes (first time only) |
| TF-IDF query              | ~0.001 seconds                  |
| BM25 query                | ~0.002 seconds                  |
| Semantic query            | ~0.1 seconds                    |

## Troubleshooting

### PyTorch/Semantic Not Available

You'll see:

```
⚠️  Warning: Semantic retrieval not available: [WinError 1114]...
   Continuing with TF-IDF, BM25, and Fuzzy only...
```

**Solution:** The system still works with TF-IDF and BM25 for same-language queries. For cross-lingual capability, you would need to fix PyTorch installation.

### No Documents Loaded

```
ERROR: No documents loaded. Exiting.
```

**Solution:** Ensure `data/raw/` directory exists with document corpus.

## Query Tips

**For Best Results:**

1. **Same-language queries:** Use TF-IDF or BM25
   - Fast and accurate for keyword matching
2. **Cross-lingual queries:** Use Semantic (if available)

   - English query → finds both English and Bangla docs
   - Bangla query → finds both Bangla and English docs

3. **Keyword-heavy queries:** Use BM25
   - Better than TF-IDF for exact matches
4. **Conceptual queries:** Use Semantic

   - Understands synonyms and semantic similarity

5. **Queries with typos:** Enable Fuzzy matching
   - Tolerates spelling variations

## Example Queries to Try

**English:**

- "osman hadi"
- "climate change"
- "rohingya crisis"
- "digital bangladesh"
- "garment industry"

**Bangla:**

- "ওসমান হাদী"
- "জলবায়ু পরিবর্তন"
- "রোহিঙ্গা সংকট"
- "ডিজিটাল বাংলাদেশ"

**Cross-Lingual (Semantic only):**

- English query → retrieves Bangla documents
- Bangla query → retrieves English documents

## Technical Details

**Corpus:** 5,170 documents

- English: 2,581 documents (7 sources)
- Bangla: 2,589 documents (6 sources)

**Vocabulary:**

- TF-IDF: 7,263 unique terms
- BM25: 14,152 unique tokens

**Methods Available:**

- ✅ TF-IDF (always available)
- ✅ BM25 (always available)
- ⚠️ Semantic (requires PyTorch)
- ⚠️ Hybrid (requires Semantic)
- ✅ Fuzzy (always available, optional)

## Notes

- **First run:** Index building takes a few seconds
- **Semantic embeddings:** First-time encoding takes several minutes for full corpus
- **Subsequent runs:** Use cached embeddings (if implemented)
- **Scores:** Different ranges for different methods (normalized in hybrid)
