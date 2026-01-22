# Module 3: Query Processing

This module handles query translation and language detection for cross-lingual information retrieval.

## Files

- **`query_processor.py`**: Query processing with language detection and translation

## Features

- **Language Detection**: Automatically detects if query is in Bangla or English
- **Translation**: Translates queries to the opposite language using Google Translate
- **Error Handling**: Gracefully handles translation failures
- **Bilingual Support**: Works with both Bangla (bn) and English (en)

## Usage

### Basic Usage

```python
from src.module2_query_processing import QueryProcessor

# Initialize processor
processor = QueryProcessor()

# Process a query
result = processor.process_query("Dhaka air pollution")

print(result)
# {
#     'original_text': 'Dhaka air pollution',
#     'translated_text': 'ঢাকার বায়ু দূষণ',
#     'source_lang': 'en',
#     'target_lang': 'bn'
# }
```

### Test the Module

```bash
python src/module3_query_processing/query_processor.py
```

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
