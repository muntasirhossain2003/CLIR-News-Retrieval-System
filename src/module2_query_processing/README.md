# Module 2: Query Processing

## Purpose

Handles cross-lingual query processing through automatic language detection and translation for Bangla-English CLIR system.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Query Processing                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  User Query (Bangla or English)                              │
│         │                                                    │
│         ↓                                                    │
│  Language Detector (langdetect)                              │
│         │                                                    │
│         ├──► Bangla Detected ──► Translate to English       │
│         │    (bn)                 (GoogleTranslator)         │
│         │                              │                     │
│         └──► English Detected ──► Translate to Bangla       │
│              (en)                     (GoogleTranslator)     │
│                                           │                  │
│                                           ↓                  │
│                      ┌────────────────────────────────┐     │
│                      │ Processed Query:               │     │
│                      │ - original_query               │     │
│                      │ - detected_language            │     │
│                      │ - translated_query             │     │
│                      └────────────────────────────────┘     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Components

### 1. **QueryProcessor Class**

- **Language Detection**: Uses `langdetect` library
  - Input: Query string
  - Output: 2-letter ISO code ('bn' or 'en')
  - Fallback: Defaults to 'en' if detection fails

- **Translation**: Uses `deep-translator` (Google Translate API)
  - Bangla → English: Uses 'bengali' as source language
  - English → Bangla: Uses 'english' as source language
  - Fallback: Returns original query if translation fails

### 2. **Query Processing Flow**

```python
def process_query(query: str) -> dict:
    1. Detect language (bn/en)
    2. Translate to opposite language
    3. Return {
        'original_query': str,
        'detected_language': str,
        'translated_query': str
    }
```

## Data Flow

```
User Input: "বাংলাদেশের অর্থনীতি"
      ↓
detect_language()
      ↓
Language: "bn" (Bangla)
      ↓
translate_query("বাংলাদেশের অর্থনীতি", "bn")
      ↓
GoogleTranslator(source='bengali', target='english')
      ↓
Translation: "Bangladesh's economy"
      ↓
Output: {
    'original_query': 'বাংলাদেশের অর্থনীতি',
    'detected_language': 'bn',
    'translated_query': "Bangladesh's economy"
}
```

## Usage

### From Python Code

```python
from src.module2_query_processing.query_processor import QueryProcessor

# Initialize processor
processor = QueryProcessor()

# Process Bangla query
result = processor.process_query("বাংলাদেশের অর্থনীতি")
print(f"Original: {result['original_query']}")
print(f"Language: {result['detected_language']}")
print(f"Translated: {result['translated_query']}")

# Process English query
result = processor.process_query("Bangladesh economy")
print(f"Original: {result['original_query']}")
print(f"Language: {result['detected_language']}")
print(f"Translated: {result['translated_query']}")
```

### Integration in Retrieval Pipeline

```python
# In app.py or retriever
query_processor = QueryProcessor()
retriever = Retriever()

# User enters query
user_query = "বাংলাদেশের অর্থনীতি"

# Process query
processed = query_processor.process_query(user_query)

# Search with both original and translated
results = retriever.search(
    original_query=processed['original_query'],
    translated_query=processed['translated_query'],
    language=processed['detected_language']
)
```

## Output Format

### Successful Processing

```python
{
    'original_query': 'বাংলাদেশের অর্থনীতি',
    'detected_language': 'bn',
    'translated_query': "Bangladesh's economy"
}
```

### Translation Failure

```python
{
    'original_query': 'some query',
    'detected_language': 'en',
    'translated_query': 'some query'  # Falls back to original
}
```

## Error Handling

1. **Language Detection Failure**
   - Fallback: Assumes English ('en')
   - Logs warning message
   - Continues processing

2. **Translation Failure**
   - Fallback: Uses original query as translated query
   - Logs error message
   - Continues processing

3. **Network Issues**
   - Google Translate API requires internet
   - Fails gracefully with original query
   - Does not crash the system

## Dependencies

```
langdetect==1.0.9
deep-translator==1.11.4
```

## Performance

- **Language Detection**: ~5-10 ms per query
- **Translation**: ~100-300 ms per query (depends on network)
- **Total Processing Time**: ~110-310 ms per query

## Limitations

1. **Internet Dependency**: Requires active connection for Google Translate
2. **Language Support**: Only Bangla ↔ English (can be extended)
3. **Detection Accuracy**: langdetect may misidentify short queries
4. **Translation Quality**: Depends on Google Translate API quality
   from src.module2_query_processing import QueryProcessor

# Initialize processor

processor = QueryProcessor()

# Process a query

result = processor.process_query("Dhaka air pollution")

print(result)

# {

# 'original_text': 'Dhaka air pollution',

# 'translated_text': 'ঢাকার বায়ু দূষণ',

# 'source_lang': 'en',

# 'target_lang': 'bn'

# }

````

### Test the Module

```bash
python src/module3_query_processing/query_processor.py
````

This will run test examples with both Bangla and English queries.

## Dependencies

- `langdetect`: Language detection
- `deep-translator`: Google Translate API

Install with:

```bash
pip install langdetect deep-translator
```

## How It Works

1. **Input**: User query in Bangla or English
2. **Detection**: Detect language using langdetect
3. **Translation**: Translate to opposite language using GoogleTranslator
4. **Output**: Dictionary with original, translated, and language info

## Error Handling

- If language detection fails → Assumes English
- If translation fails → Returns None for translated_text but keeps original
- Empty/invalid input → Returns appropriate error messages

## Example Queries

**English to Bangla:**

- "Dhaka air pollution" → "ঢাকার বায়ু দূষণ"
- "tax reform policies" → "কর সংস্কার নীতি"

**Bangla to English:**

- "বাংলাদেশের অর্থনীতি" → "Bangladesh's economy"
- "মুদ্রাস্ফীতি কমানো" → "reducing inflation"
