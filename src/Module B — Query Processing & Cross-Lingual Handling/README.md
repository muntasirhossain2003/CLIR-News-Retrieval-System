# Module B — Query Processing & Cross-Lingual Handling

## Part 1: Language Detection & Normalization

This module implements basic query preprocessing for the CLIR system.

### Features

✅ **Language Detection**

- Detects Bangla ('bn') or English ('en')
- Uses fastText model if available
- Unicode-range fallback for Bangla (U+0980 to U+09FF)

✅ **Query Normalization**

- Trims whitespace
- Unicode normalization (NFC form)
- Lowercase conversion (English only)
- Preserves stopwords, punctuation, and numbers

### API

#### `detect_query_language(query: str) -> str`

Detects query language.

**Returns:** `'bn'` or `'en'`

**Example:**

```python
from query_processor_part1 import detect_query_language

lang = detect_query_language("জলবায়ু পরিবর্তন")
print(lang)  # Output: 'bn'

lang = detect_query_language("climate change")
print(lang)  # Output: 'en'
```

#### `normalize_query(query: str, lang: str) -> str`

Normalizes query based on language.

**Example:**

```python
from query_processor_part1 import normalize_query

normalized = normalize_query("  Climate Change  ", "en")
print(normalized)  # Output: 'climate change'

normalized = normalize_query("  জলবায়ু পরিবর্তন  ", "bn")
print(normalized)  # Output: 'জলবায়ু পরিবর্তন'
```

#### `process_query_part1(query: str) -> dict`

Main entry point that combines detection and normalization.

**Returns:**

```python
{
    "original_query": str,    # Raw input
    "language": "bn" | "en",  # Detected language
    "normalized_query": str   # Processed query
}
```

**Example:**

```python
from query_processor_part1 import process_query_part1

result = process_query_part1("  Mobile Phone Technology  ")
print(result)
# Output:
# {
#     'original_query': '  Mobile Phone Technology  ',
#     'language': 'en',
#     'normalized_query': 'mobile phone technology'
# }
```

### Usage

```powershell
# Test the module
python "src/Module B — Query Processing & Cross-Lingual Handling/query_processor_part1.py"
```

### Test Output

```
======================================================================
Module B - Part 1: Query Processing Test
======================================================================

Original:   'Climate Change'
Language:   en
Normalized: 'climate change'
----------------------------------------------------------------------

Original:   '  Mobile Phone Technology  '
Language:   en
Normalized: 'mobile phone technology'
----------------------------------------------------------------------

Original:   'জলবায়ু পরিবর্তন'
Language:   bn
Normalized: 'জলবায়ু পরিবর্তন'
----------------------------------------------------------------------

Original:   'মোবাইল ফোন'
Language:   bn
Normalized: 'মোবাইল ফোন'
----------------------------------------------------------------------

Original:   ''
Language:   en
Normalized: ''
----------------------------------------------------------------------

Original:   'COVID-19 pandemic'
Language:   en
Normalized: 'covid-19 pandemic'
----------------------------------------------------------------------

Original:   'করোনাভাইরাস মহামারী'
Language:   bn
Normalized: 'করোনাভাইরাস মহামারী'
----------------------------------------------------------------------
```

### Design Decisions

**Why no stopword removal?**

- Stopwords are important for semantic search
- Phrases like "the impact of climate change" lose meaning without stopwords

**Why preserve punctuation?**

- Punctuation can be significant (e.g., "COVID-19", "U.S.A")
- Named entities often contain punctuation

**Why Unicode NFC normalization?**

- Ensures consistent character representation
- Prevents duplicate results due to different character encodings
- Example: é can be one character or e+accent; NFC makes it consistent

**Why lowercase only English?**

- Bangla script has no uppercase/lowercase distinction
- Lowercasing Bangla can corrupt the text

### Academic Explanation

**For Viva Defense:**

1. **Language Detection Strategy:**

   - Primary: fastText (trained on 176 languages, high accuracy)
   - Fallback: Unicode range check (robust, no dependencies)
   - Threshold: 30% Bangla characters → classify as Bangla

2. **Normalization Strategy:**

   - Minimal preprocessing to preserve query intent
   - Language-aware (different rules for Bangla vs English)
   - Keeps original semantic meaning intact

3. **Why Simple Functions?**
   - Easy to test and debug
   - Easy to explain in academic setting
   - No hidden complexity
   - Clear input/output contracts

### Next Steps

- **Part 2:** Query Translation (Bangla ↔ English)
- **Part 3:** Query Expansion & Named Entity Mapping

### Dependencies

- `unicodedata` (built-in)
- `os` (built-in)
- `fasttext` (optional, falls back to Unicode check)

### Integration with Search

```python
from query_processor_part1 import process_query_part1
from src.module1_data_acquisition.indexing import LexicalIndexer

# Process query
result = process_query_part1("মোবাইল ফোন")

# Use normalized query for search
indexer = LexicalIndexer(index_dir="indexes/whoosh")
indexer.open_index()
results = indexer.search(
    result["normalized_query"],
    language=result["language"]
)
```
