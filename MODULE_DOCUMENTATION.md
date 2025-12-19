# CLIR System - Module Documentation

Complete documentation for all modules with explanations and simple examples.

---

## Table of Contents

- [Module A: Indexing](#module-a-indexing)
  - [1. language_detector.py](#1-language_detectorpy)
  - [2. tokenizer.py](#2-tokenizerpy)
  - [3. ner_extractor.py](#3-ner_extractorpy)
  - [4. inverted_index.py](#4-inverted_indexpy)
  - [5. document_processor.py](#5-document_processorpy)
- [Module B: Query Processing](#module-b-query-processing)
  - [6. query_detector.py](#6-query_detectorpy)
  - [7. query_normalizer.py](#7-query_normalizerpy)
  - [8. query_translator.py](#8-query_translatorpy)
  - [9. query_expander.py](#9-query_expanderpy)
  - [10. ne_mapper.py](#10-ne_mapperpy)
  - [11. query_pipeline.py](#11-query_pipelinepy)
- [Module C: Retrieval](#module-c-retrieval)
  - [12. lexical_retrieval.py](#12-lexical_retrievalpy)
  - [13. fuzzy_matcher.py](#13-fuzzy_matcherpy)
  - [14. semantic_retrieval.py](#14-semantic_retrievalpy)
  - [15. hybrid_ranker.py](#15-hybrid_rankerpy)
  - [16. retrieval_engine.py](#16-retrieval_enginepy)

---

# Module A: Indexing

Module A handles document preprocessing, language detection, tokenization, named entity extraction, and building the inverted index.

---

## 1. language_detector.py

### Purpose

Detects the language of text (English or Bangla) with confidence scores. Handles code-switching (mixed language text).

### Key Features

- Language detection for single text or batches
- Confidence scoring
- Mixed language detection
- Support for short text

### Main Class: `LanguageDetector`

#### Methods:

- `detect(text)` - Detect language of a single text
- `detect_batch(texts)` - Detect language for multiple texts
- `is_mixed_language(text)` - Check if text contains mixed languages

### Simple Example

```python
from src.module_a_indexing import LanguageDetector

# Initialize detector
detector = LanguageDetector()

# Detect English text
result = detector.detect("Bangladesh prime minister visits India")
print(result)
# Output: {'language': 'en', 'confidence': 0.99, 'is_mixed': False}

# Detect Bangla text
result = detector.detect("প্রধানমন্ত্রী ভারত সফর করেছেন")
print(result)
# Output: {'language': 'bn', 'confidence': 0.99, 'is_mixed': False}

# Detect mixed language
result = detector.detect("Sheikh Hasina শেখ হাসিনা visited India")
print(result)
# Output: {'language': 'en', 'confidence': 0.85, 'is_mixed': True}

# Batch detection
texts = [
    "Bangladesh wins the match",
    "বাংলাদেশ জয়ী হয়েছে",
    "Prime Minister প্রধানমন্ত্রী"
]
results = detector.detect_batch(texts)
for text, result in zip(texts, results):
    print(f"{text[:30]}: {result['language']} (conf: {result['confidence']:.2f})")
```

**Use Case:** Automatically identify document language before processing.

---

## 2. tokenizer.py

### Purpose

Tokenizes text into individual words/tokens for both English and Bangla. Uses spaCy for advanced tokenization.

### Key Features

- Multilingual tokenization (EN/BN)
- Position tracking (character offsets)
- Token count statistics
- Fallback for Bangla when spaCy model unavailable
- Stopword removal
- Lowercasing option

### Main Class: `MultilingualTokenizer`

#### Methods:

- `tokenize(text, language, lowercase, remove_stopwords)` - Tokenize text
- `tokenize_with_positions(text, language)` - Tokenize with character positions
- `get_token_count(text, language)` - Count tokens in text

### Simple Example

```python
from src.module_a_indexing import MultilingualTokenizer

# Initialize tokenizer
tokenizer = MultilingualTokenizer()

# Tokenize English text
text = "The Prime Minister announced new policies today."
tokens = tokenizer.tokenize(text, language='en', lowercase=True)
print(tokens)
# Output: ['prime', 'minister', 'announced', 'new', 'policies', 'today']

# Tokenize Bangla text
text = "প্রধানমন্ত্রী নতুন নীতি ঘোষণা করেছেন"
tokens = tokenizer.tokenize(text, language='bn')
print(tokens)
# Output: ['প্রধানমন্ত্রী', 'নতুন', 'নীতি', 'ঘোষণা', 'করেছেন']

# Get token positions
text = "Sheikh Hasina visits India"
result = tokenizer.tokenize_with_positions(text, language='en')
print(result)
# Output: {
#   'tokens': ['sheikh', 'hasina', 'visits', 'india'],
#   'positions': [(0, 6), (7, 13), (14, 20), (21, 26)]
# }

# Count tokens
count = tokenizer.get_token_count("This is a test sentence", language='en')
print(f"Token count: {count}")
# Output: Token count: 5
```

**Use Case:** Convert raw text into searchable tokens for indexing.

---

## 3. ner_extractor.py

### Purpose

Extracts Named Entities (persons, organizations, locations, dates) from text using spaCy's NER.

### Key Features

- Entity type recognition (PERSON, ORG, GPE, LOC, DATE)
- Batch processing
- Entity filtering by type
- Fallback for unsupported languages

### Main Class: `NERExtractor`

#### Methods:

- `extract_entities(text, language)` - Extract all entities from text
- `extract_entities_batch(texts, language)` - Extract from multiple texts
- `filter_by_type(entities, entity_type)` - Filter entities by type

### Simple Example

```python
from src.module_a_indexing import NERExtractor

# Initialize NER extractor
ner = NERExtractor()

# Extract entities from English text
text = "Sheikh Hasina met with Narendra Modi in New Delhi on December 15, 2023."
entities = ner.extract_entities(text, language='en')

print("All entities:")
for entity in entities:
    print(f"  {entity['text']}: {entity['label']}")
# Output:
#   Sheikh Hasina: PERSON
#   Narendra Modi: PERSON
#   New Delhi: GPE
#   December 15, 2023: DATE

# Filter by type
persons = ner.filter_by_type(entities, 'PERSON')
print("\nPersons only:")
for person in persons:
    print(f"  {person['text']}")
# Output:
#   Sheikh Hasina
#   Narendra Modi

locations = ner.filter_by_type(entities, 'GPE')
print("\nLocations only:")
for loc in locations:
    print(f"  {loc['text']}")
# Output:
#   New Delhi

# Batch processing
texts = [
    "Apple Inc. released new products in California.",
    "Microsoft Corporation announced updates in Seattle."
]
all_entities = ner.extract_entities_batch(texts, language='en')
for i, entities in enumerate(all_entities, 1):
    print(f"\nText {i} entities:")
    for entity in entities:
        print(f"  {entity['text']}: {entity['label']}")
```

**Use Case:** Identify important entities for enhanced search and entity-based retrieval.

---

## 4. inverted_index.py

### Purpose

Builds and manages an inverted index (term → list of documents containing that term). Core data structure for efficient retrieval.

### Key Features

- Term to document mapping
- Term frequency tracking
- Position tracking for terms
- Document frequency statistics
- Save/load functionality (pickle/JSON)
- Posting list retrieval

### Main Class: `InvertedIndex`

#### Methods:

- `add_document(doc_id, tokens, metadata)` - Add document to index
- `get_posting_list(term)` - Get all documents containing term
- `get_term_frequency(term, doc_id)` - Get term count in document
- `get_document_frequency(term)` - Get number of documents containing term
- `save_index(filepath)` - Save index to disk
- `load_index(filepath)` - Load index from disk

### Simple Example

```python
from src.module_a_indexing import InvertedIndex

# Initialize index
index = InvertedIndex()

# Add documents
index.add_document(
    doc_id='doc1',
    tokens=['prime', 'minister', 'announces', 'policy'],
    metadata={'title': 'PM Announces Policy', 'language': 'en'}
)

index.add_document(
    doc_id='doc2',
    tokens=['minister', 'visits', 'india'],
    metadata={'title': 'Minister Visits India', 'language': 'en'}
)

index.add_document(
    doc_id='doc3',
    tokens=['prime', 'minister', 'prime', 'policy'],
    metadata={'title': 'PM Policy Update', 'language': 'en'}
)

# Get posting list for term
posting = index.get_posting_list('minister')
print(f"Documents containing 'minister': {posting}")
# Output: [
#   {'doc_id': 'doc1', 'frequency': 1, 'positions': [1]},
#   {'doc_id': 'doc2', 'frequency': 1, 'positions': [0]},
#   {'doc_id': 'doc3', 'frequency': 1, 'positions': [1]}
# ]

# Get term frequency in specific document
freq = index.get_term_frequency('prime', 'doc3')
print(f"'prime' appears {freq} times in doc3")
# Output: 'prime' appears 2 times in doc3

# Get document frequency
df = index.get_document_frequency('minister')
print(f"'minister' appears in {df} documents")
# Output: 'minister' appears in 3 documents

# Get statistics
stats = index.get_statistics()
print(f"Total documents: {stats['total_documents']}")
print(f"Total unique terms: {stats['total_terms']}")

# Save index
index.save_index('my_index.pkl')

# Load index later
new_index = InvertedIndex()
new_index.load_index('my_index.pkl')
```

**Use Case:** Enable fast document lookup by search terms.

---

## 5. document_processor.py

### Purpose

Orchestrates the complete document processing pipeline: language detection → tokenization → NER → indexing.

### Key Features

- End-to-end document processing
- Batch processing with progress tracking
- Statistics generation
- Error handling and logging
- Metadata enhancement
- Multiple output formats (pickle, JSON)

### Main Class: `DocumentProcessor`

#### Methods:

- `process_single_document(doc_data)` - Process one document
- `process_all_documents(batch_size)` - Process all documents in data directory
- `generate_statistics()` - Generate processing statistics

### Simple Example

```python
from src.module_a_indexing import DocumentProcessor

# Initialize processor
processor = DocumentProcessor(
    data_dir='data/raw',           # Input directory
    output_dir='processed_data'     # Output directory
)

# Process all documents
print("Processing documents...")
stats = processor.process_all_documents(batch_size=100)

# View statistics
print(f"\nProcessing Complete!")
print(f"Total processed: {stats['processing']['processed']}")
print(f"Total errors: {stats['processing']['errors']}")
print(f"English docs: {stats['processing']['by_language']['en']}")
print(f"Bangla docs: {stats['processing']['by_language']['bn']}")
print(f"Total tokens: {stats['processing']['total_tokens']}")
print(f"Total entities: {stats['processing']['total_entities']}")
print(f"Avg tokens/doc: {stats['avg_tokens_per_doc']:.2f}")

# Output files created:
# - processed_data/inverted_index.pkl
# - processed_data/document_metadata.json
# - processed_data/processing_statistics.json

# Process a single document manually
doc_data = {
    'title': 'Test Article',
    'content': 'Sheikh Hasina announces new policy in Dhaka.',
    'date': '2023-12-15',
    'source': 'test_source'
}
result = processor.process_single_document(doc_data)
print(f"\nProcessed: {result['metadata']['title']}")
print(f"Language: {result['language']}")
print(f"Tokens: {result['tokens'][:10]}")
print(f"Entities: {result['entities']}")
```

**Use Case:** Transform raw crawled documents into searchable indexed data.

---

# Module B: Query Processing

Module B handles query language detection, normalization, translation, expansion, and entity mapping.

---

## 6. query_detector.py

### Purpose

Detects the language of search queries, even short ones with code-switching.

### Key Features

- Short query language detection
- Code-switching detection
- Dominant language identification
- Character-based fallback for very short queries

### Main Class: `QueryLanguageDetector`

#### Methods:

- `detect_query_language(query)` - Detect query language
- `is_code_switched(query)` - Check if query is code-switched
- `get_dominant_language(query)` - Get dominant language in mixed query

### Simple Example

```python
from src.module_b_query_processing import QueryLanguageDetector

# Initialize detector
detector = QueryLanguageDetector()

# Detect English query
query = "Sheikh Hasina"
result = detector.detect_query_language(query)
print(f"Query: {query}")
print(f"Language: {result['language']}, Confidence: {result['confidence']:.2f}")
print(f"Code-switched: {result['is_code_switched']}")
# Output: Language: en, Confidence: 0.95, Code-switched: False

# Detect Bangla query
query = "শেখ হাসিনা"
result = detector.detect_query_language(query)
print(f"\nQuery: {query}")
print(f"Language: {result['language']}")
# Output: Language: bn

# Detect code-switched query
query = "Sheikh Hasina শেখ হাসিনা"
result = detector.detect_query_language(query)
print(f"\nQuery: {query}")
print(f"Language: {result['language']}")
print(f"Code-switched: {result['is_code_switched']}")
# Output: Language: en, Code-switched: True

# Get dominant language
dominant = detector.get_dominant_language(query)
print(f"Dominant language: {dominant}")
# Output: Dominant language: en

# Very short query
query = "PM"
result = detector.detect_query_language(query)
print(f"\nQuery: {query}")
print(f"Language: {result['language']} (using fallback)")
```

**Use Case:** Determine query language to route to appropriate translation/processing.

---

## 7. query_normalizer.py

### Purpose

Normalizes queries by removing extra whitespace, standardizing punctuation, lowercasing (for English), and Unicode normalization.

### Key Features

- Language-specific normalization
- Whitespace cleanup
- Punctuation standardization
- Unicode normalization (NFC)
- Mixed language support

### Main Class: `QueryNormalizer`

#### Methods:

- `normalize(query, language)` - Normalize query
- `normalize_mixed(query)` - Normalize mixed language query

### Simple Example

```python
from src.module_b_query_processing import QueryNormalizer

# Initialize normalizer
normalizer = QueryNormalizer()

# Normalize English query
query = "  Sheikh   HASINA   visits   India!!!  "
normalized = normalizer.normalize(query, language='en')
print(f"Original:   '{query}'")
print(f"Normalized: '{normalized}'")
# Output: Normalized: 'sheikh hasina visits india'

# Normalize Bangla query
query = "  শেখ   হাসিনা   ভারত   সফর  "
normalized = normalizer.normalize(query, language='bn')
print(f"\nOriginal:   '{query}'")
print(f"Normalized: '{normalized}'")
# Output: Normalized: 'শেখ হাসিনা ভারত সফর'

# Normalize with special characters
query = "What is PM's new policy???"
normalized = normalizer.normalize(query, language='en')
print(f"\nOriginal:   '{query}'")
print(f"Normalized: '{normalized}'")
# Output: Normalized: 'what is pm new policy'

# Normalize mixed language
query = "Sheikh Hasina  শেখ হাসিনা   visits"
normalized = normalizer.normalize_mixed(query)
print(f"\nOriginal:   '{query}'")
print(f"Normalized: '{normalized}'")
# Output: Normalized: 'sheikh hasina শেখ হাসিনা visits'

# Handle multiple spaces and newlines
query = "prime\n\nminister\t\tvisits"
normalized = normalizer.normalize(query, language='en')
print(f"\nOriginal:   '{query}'")
print(f"Normalized: '{normalized}'")
# Output: Normalized: 'prime minister visits'
```

**Use Case:** Clean and standardize user queries before processing.

---

## 8. query_translator.py

### Purpose

Translates queries between English and Bangla using transformer-based neural machine translation (Helsinki-NLP OPUS-MT models).

### Key Features

- English ↔ Bangla translation
- Translation caching for performance
- Confidence scoring
- Batch translation support

### Main Class: `QueryTranslator`

#### Methods:

- `translate_to_bangla(english_query)` - Translate EN → BN
- `translate_to_english(bangla_query)` - Translate BN → EN
- `translate(query, source_lang, target_lang)` - Generic translation
- `get_cached_translation(query, direction)` - Get cached result

### Simple Example

```python
from src.module_b_query_processing import QueryTranslator

# Initialize translator
translator = QueryTranslator(cache_dir='cache/translations')

# Translate English to Bangla
query = "prime minister education policy"
translated = translator.translate_to_bangla(query)
print(f"English:  {query}")
print(f"Bangla:   {translated}")
# Output: Bangla: প্রধানমন্ত্রী শিক্ষা নীতি

# Translate Bangla to English
query = "প্রধানমন্ত্রী শিক্ষা নীতি"
translated = translator.translate_to_english(query)
print(f"\nBangla:   {query}")
print(f"English:  {translated}")
# Output: English: prime minister education policy

# Generic translation
translated = translator.translate(
    query="Sheikh Hasina visits India",
    source_lang='en',
    target_lang='bn'
)
print(f"\nTranslated: {translated}")

# Translation is cached - second call is instant
query = "prime minister education policy"
translated1 = translator.translate_to_bangla(query)  # Takes time
translated2 = translator.translate_to_bangla(query)  # Instant (cached)
print(f"\nCached translation: {translated2}")

# Multiple queries
queries = [
    "education policy",
    "health ministry",
    "economic growth"
]
for q in queries:
    bn = translator.translate_to_bangla(q)
    print(f"{q} → {bn}")
```

**Use Case:** Enable cross-lingual search - search in one language, retrieve documents in another.

---

## 9. query_expander.py

### Purpose

Expands queries with synonyms to improve recall. Uses WordNet for English and manual dictionary for Bangla.

### Key Features

- Synonym expansion using WordNet
- Bangla synonym dictionary
- Configurable expansion limit
- Duplicate removal

### Main Class: `QueryExpander`

#### Methods:

- `expand(query, language, max_synonyms)` - Expand query with synonyms
- `get_synonyms_english(word, max_synonyms)` - Get English synonyms
- `get_similar_words_bangla(word)` - Get Bangla synonyms

### Simple Example

```python
from src.module_b_query_processing import QueryExpander

# Initialize expander
expander = QueryExpander()

# Expand English query
query = "education policy"
expanded = expander.expand(query, language='en', max_synonyms=3)
print(f"Original: {query}")
print(f"Expanded: {expanded}")
# Output: Expanded: education policy instruction learning schooling guideline rule regulation

# Expand with different synonym limit
query = "government minister"
expanded = expander.expand(query, language='en', max_synonyms=2)
print(f"\nOriginal: {query}")
print(f"Expanded: {expanded}")
# Output: Expanded: government minister regime administration curate diplomatic

# Get synonyms for specific word
word = "policy"
synonyms = expander.get_synonyms_english(word, max_synonyms=5)
print(f"\nSynonyms of '{word}': {synonyms}")
# Output: Synonyms of 'policy': ['guideline', 'rule', 'regulation', 'procedure', 'plan']

# Expand Bangla query
query = "শিক্ষা নীতি"
expanded = expander.expand(query, language='bn')
print(f"\nOriginal: {query}")
print(f"Expanded: {expanded}")
# Output: Expanded: শিক্ষা নীতি শিক্ষণ পড়াশোনা নিয়ম আইন

# No expansion for unknown words
query = "xyzabc"
expanded = expander.expand(query, language='en')
print(f"\nOriginal: {query}")
print(f"Expanded: {expanded}")
# Output: Expanded: xyzabc (no synonyms found)
```

**Use Case:** Improve search recall by matching documents that use different but related terms.

---

## 10. ne_mapper.py

### Purpose

Maps named entities between English and Bangla (e.g., "Sheikh Hasina" ↔ "শেখ হাসিনা"). Enables cross-lingual entity matching.

### Key Features

- Bilingual entity dictionary
- Entity variation support
- Automatic dictionary building
- Entity type awareness

### Main Class: `NamedEntityMapper`

#### Methods:

- `map_entity(entity, source_lang, target_lang)` - Map entity across languages
- `get_entity_variations(entity)` - Get all variations of entity
- `build_entity_dictionary()` - Build custom entity mappings

### Simple Example

```python
from src.module_b_query_processing import NamedEntityMapper

# Initialize mapper
mapper = NamedEntityMapper()

# Map English to Bangla
entity = "Sheikh Hasina"
mapped = mapper.map_entity(entity, source_lang='en', target_lang='bn')
print(f"English: {entity}")
print(f"Bangla:  {mapped}")
# Output: Bangla: ['শেখ হাসিনা']

# Map Bangla to English
entity = "শেখ হাসিনা"
mapped = mapper.map_entity(entity, source_lang='bn', target_lang='en')
print(f"\nBangla:  {entity}")
print(f"English: {mapped}")
# Output: English: ['Sheikh Hasina']

# Get all variations
entity = "Dhaka"
variations = mapper.get_entity_variations(entity)
print(f"\nVariations of '{entity}': {variations}")
# Output: Variations of 'Dhaka': ['Dhaka', 'ঢাকা']

# Map multiple entities
entities = ["Bangladesh", "India", "Sheikh Hasina"]
for ent in entities:
    bn = mapper.map_entity(ent, 'en', 'bn')
    print(f"{ent} → {bn}")
# Output:
#   Bangladesh → ['বাংলাদেশ']
#   India → ['ভারত']
#   Sheikh Hasina → ['শেখ হাসিনা']

# Unknown entity (no mapping)
entity = "Random Person"
mapped = mapper.map_entity(entity, 'en', 'bn')
print(f"\n'{entity}' mapping: {mapped}")
# Output: 'Random Person' mapping: []

# Add custom mapping
mapper.entity_dict['John Doe'] = {
    'bn': ['জন ডো'],
    'type': 'PERSON'
}
mapped = mapper.map_entity('John Doe', 'en', 'bn')
print(f"John Doe → {mapped}")
# Output: John Doe → ['জন ডো']
```

**Use Case:** Match entity mentions across languages (searching "Sheikh Hasina" finds "শেখ হাসিনা").

---

## 11. query_pipeline.py

### Purpose

Orchestrates the complete query processing pipeline: detection → normalization → translation → expansion → entity mapping.

### Key Features

- End-to-end query processing
- Optional translation, expansion, entity mapping
- Timing tracking
- Both English and Bangla query outputs

### Main Class: `QueryProcessor`

#### Methods:

- `process_query(query)` - Process complete query pipeline
- Internal methods for each step

### Simple Example

```python
from src.module_b_query_processing import QueryProcessor

# Initialize processor with all features
processor = QueryProcessor(
    enable_translation=True,
    enable_expansion=True,
    enable_entity_mapping=True
)

# Process English query
query = "Sheikh Hasina education policy"
result = processor.process_query(query)

print("Query Processing Results:")
print(f"Original query:    {result['original_query']}")
print(f"Detected language: {result['language']}")
print(f"Normalized query:  {result['normalized_query']}")
print(f"English query:     {result['en_query']}")
print(f"Bangla query:      {result['bn_query']}")
print(f"Expanded (EN):     {result['expanded_en']}")
print(f"Expanded (BN):     {result['expanded_bn']}")
print(f"Entities:          {result['entities']}")
print(f"Processing time:   {result['processing_time_ms']:.2f}ms")

# Output:
# Original query:    Sheikh Hasina education policy
# Detected language: en
# Normalized query:  sheikh hasina education policy
# English query:     sheikh hasina education policy
# Bangla query:      শেখ হাসিনা শিক্ষা নীতি
# Expanded (EN):     sheikh hasina education policy instruction learning guideline
# Expanded (BN):     শেখ হাসিনা শিক্ষা নীতি শিক্ষণ নিয়ম
# Entities:          {'Sheikh Hasina': ['শেখ হাসিনা']}
# Processing time:   245.32ms

# Process Bangla query
query = "শেখ হাসিনা শিক্ষা"
result = processor.process_query(query)
print(f"\nOriginal: {result['original_query']}")
print(f"Language: {result['language']}")
print(f"English:  {result['en_query']}")
print(f"Bangla:   {result['bn_query']}")

# Minimal processing (no expansion/translation)
processor_minimal = QueryProcessor(
    enable_translation=False,
    enable_expansion=False,
    enable_entity_mapping=False
)
result = processor_minimal.process_query("education policy")
print(f"\nMinimal processing:")
print(f"Normalized: {result['normalized_query']}")
```

**Use Case:** Transform user queries into optimized search queries for both EN and BN document retrieval.

---

# Module C: Retrieval

Module C handles document retrieval using various algorithms: BM25, TF-IDF, semantic embeddings, fuzzy matching, and hybrid ranking.

---

## 12. lexical_retrieval.py

### Purpose

Implements lexical retrieval algorithms (BM25 and TF-IDF) for term-based document matching.

### Key Features

- BM25 ranking (industry standard)
- TF-IDF ranking
- Configurable parameters (k1, b)
- Document scoring

### Main Class: `LexicalRetriever`

#### Methods:

- `search_bm25(query_tokens, top_k)` - BM25 search
- `search_tfidf(query_tokens, top_k)` - TF-IDF search
- `get_scores(query_tokens, method)` - Get scores for all documents

### Simple Example

```python
from src.module_c_retrieval import LexicalRetriever

# Prepare documents
documents = [
    {
        'doc_id': 'doc1',
        'tokens': ['prime', 'minister', 'announces', 'education', 'policy'],
        'metadata': {'title': 'PM Announces Education Policy'}
    },
    {
        'doc_id': 'doc2',
        'tokens': ['minister', 'visits', 'school', 'education', 'program'],
        'metadata': {'title': 'Minister Visits School'}
    },
    {
        'doc_id': 'doc3',
        'tokens': ['new', 'health', 'policy', 'announced'],
        'metadata': {'title': 'Health Policy Announced'}
    }
]

# Initialize retriever
retriever = LexicalRetriever(documents, k1=1.5, b=0.75)

# BM25 search
query_tokens = ['education', 'policy']
results = retriever.search_bm25(query_tokens, top_k=3)

print("BM25 Results:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['metadata']['title']}")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Doc ID: {result['doc_id']}")
# Output:
#   1. PM Announces Education Policy (Score: 2.4567)
#   2. Minister Visits School (Score: 1.2345)
#   3. Health Policy Announced (Score: 0.8901)

# TF-IDF search
results = retriever.search_tfidf(query_tokens, top_k=3)
print("\nTF-IDF Results:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['metadata']['title']} (Score: {result['score']:.4f})")

# Get all scores (not just top-k)
all_scores = retriever.get_scores(query_tokens, method='bm25')
print("\nAll BM25 scores:")
for doc_id, score in all_scores.items():
    print(f"  {doc_id}: {score:.4f}")

# Different BM25 parameters
retriever_strict = LexicalRetriever(documents, k1=2.0, b=0.5)
results_strict = retriever_strict.search_bm25(query_tokens, top_k=2)
print("\nWith stricter parameters:")
for result in results_strict:
    print(f"  {result['metadata']['title']}: {result['score']:.4f}")
```

**Use Case:** Rank documents by term overlap with query (traditional search).

---

## 13. fuzzy_matcher.py

### Purpose

Implements fuzzy string matching to handle typos, spelling variations, and approximate matches.

### Key Features

- Levenshtein distance
- Character n-gram matching
- Jaccard similarity
- Transliteration matching
- Configurable similarity threshold

### Main Class: `FuzzyMatcher`

#### Methods:

- `fuzzy_match(query_tokens, document_tokens, threshold)` - Calculate fuzzy match score
- `character_ngram_match(str1, str2, n)` - N-gram similarity
- `transliteration_match(en_word, bn_word)` - Cross-lingual phonetic matching

### Simple Example

```python
from src.module_c_retrieval import FuzzyMatcher

# Initialize matcher
matcher = FuzzyMatcher()

# Exact match vs typo
query = ['educaton', 'policy']  # typo: educaton
document = ['education', 'policy', 'announced']

score = matcher.fuzzy_match(query, document, threshold=0.8)
print(f"Match score: {score:.4f}")
# Output: Match score: 0.9500 (high score despite typo)

# Different variations
query = ['helth', 'ministr']  # typos
document = ['health', 'minister', 'visits']
score = matcher.fuzzy_match(query, document)
print(f"Typo match score: {score:.4f}")

# Character n-gram matching
str1 = "education"
str2 = "educaton"
similarity = matcher.character_ngram_match(str1, str2, n=3)
print(f"\nN-gram similarity: {similarity:.4f}")
# Output: N-gram similarity: 0.8889

# Test multiple documents
documents = [
    ['education', 'policy', 'new'],
    ['educaton', 'polcy', 'announcement'],  # typos
    ['health', 'minister', 'visit']
]

query_tokens = ['education', 'policy']
print("\nFuzzy matching with query 'education policy':")
for i, doc in enumerate(documents, 1):
    score = matcher.fuzzy_match(query_tokens, doc)
    print(f"Doc {i}: {doc}")
    print(f"  Score: {score:.4f}")
# Output:
#   Doc 1: ['education', 'policy', 'new'] - Score: 1.0000
#   Doc 2: ['educaton', 'polcy', 'announcement'] - Score: 0.9200
#   Doc 3: ['health', 'minister', 'visit'] - Score: 0.0000

# Lower threshold for more lenient matching
score_strict = matcher.fuzzy_match(
    ['edcation'],  # typo
    ['education'],
    threshold=0.9
)
score_lenient = matcher.fuzzy_match(
    ['edcation'],
    ['education'],
    threshold=0.7
)
print(f"\nStrict (0.9): {score_strict:.4f}")
print(f"Lenient (0.7): {score_lenient:.4f}")
```

**Use Case:** Handle user typos and spelling variations in searches.

---

## 14. semantic_retrieval.py

### Purpose

Implements semantic retrieval using multilingual sentence embeddings. Finds documents with similar meaning, not just matching words.

### Key Features

- Multilingual sentence embeddings
- Semantic similarity computation
- Document encoding with caching
- GPU acceleration support
- Cosine similarity ranking

### Main Class: `SemanticRetriever`

#### Methods:

- `encode_query(query)` - Encode query to embedding vector
- `encode_documents(documents)` - Encode documents to embeddings
- `compute_similarity(query_embedding, doc_embeddings)` - Calculate similarities
- `search_semantic(query, top_k)` - Semantic search

### Simple Example

```python
from src.module_c_retrieval import SemanticRetriever

# Prepare documents
documents = [
    {
        'doc_id': 'doc1',
        'content': 'The prime minister announced new education policy',
        'metadata': {'title': 'Education Policy'}
    },
    {
        'doc_id': 'doc2',
        'content': 'Government introduces reforms in school system',
        'metadata': {'title': 'School Reforms'}
    },
    {
        'doc_id': 'doc3',
        'content': 'Health ministry releases vaccination schedule',
        'metadata': {'title': 'Vaccination News'}
    }
]

# Initialize retriever
retriever = SemanticRetriever(
    model_name='paraphrase-multilingual-mpnet-base-v2',
    cache_dir='cache/embeddings',
    use_gpu=False
)

# Encode documents (done once, cached)
print("Encoding documents...")
retriever.encode_documents(documents)

# Semantic search - finds similar meaning
query = "school teaching policy changes"
results = retriever.search_semantic(query, top_k=3)

print(f"\nQuery: {query}")
print("Semantic Results:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['metadata']['title']}")
    print(f"   Similarity: {result['score']:.4f}")
    print(f"   Content: {result['content'][:50]}...")
# Output:
#   1. Education Policy (Similarity: 0.7543)
#   2. School Reforms (Similarity: 0.7102)
#   3. Vaccination News (Similarity: 0.2341)

# Note: "School Reforms" ranks high even without exact word match
# because it's semantically similar to "school teaching policy"

# Cross-lingual semantic search
query_bangla = "শিক্ষা নীতি"  # "education policy" in Bangla
results = retriever.search_semantic(query_bangla, top_k=2)
print(f"\nBangla query: {query_bangla}")
print("Cross-lingual results:")
for result in results:
    print(f"  {result['metadata']['title']}: {result['score']:.4f}")

# Direct embedding comparison
query1 = "education policy"
query2 = "teaching guidelines"
query3 = "health vaccination"

emb1 = retriever.encode_query(query1)
emb2 = retriever.encode_query(query2)
emb3 = retriever.encode_query(query3)

# Compare similarities
from numpy import dot
from numpy.linalg import norm

sim_1_2 = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
sim_1_3 = dot(emb1, emb3) / (norm(emb1) * norm(emb3))

print(f"\nSimilarity between '{query1}' and '{query2}': {sim_1_2:.4f}")
print(f"Similarity between '{query1}' and '{query3}': {sim_1_3:.4f}")
# Output:
#   Similarity between 'education policy' and 'teaching guidelines': 0.7234
#   Similarity between 'education policy' and 'health vaccination': 0.3421
```

**Use Case:** Find conceptually similar documents even when words don't match exactly.

---

## 15. hybrid_ranker.py

### Purpose

Combines scores from multiple retrieval methods (BM25, semantic, fuzzy) using weighted fusion or reciprocal rank fusion.

### Key Features

- Multiple score fusion strategies
- Weighted combination
- Reciprocal rank fusion
- Score normalization
- Configurable weights

### Main Class: `HybridRanker`

#### Methods:

- `rank_documents(all_scores, method, weights)` - Rank using fusion
- `combine_scores(all_scores, weights)` - Weighted score combination
- `reciprocal_rank_fusion(all_scores, k)` - RRF combination

### Simple Example

```python
from src.module_c_retrieval import HybridRanker

# Scores from different methods
bm25_scores = {
    'doc1': 2.5,
    'doc2': 1.8,
    'doc3': 0.5
}

semantic_scores = {
    'doc1': 0.85,
    'doc2': 0.65,
    'doc3': 0.90
}

fuzzy_scores = {
    'doc1': 0.95,
    'doc2': 0.70,
    'doc3': 0.30
}

all_scores = {
    'bm25': bm25_scores,
    'semantic': semantic_scores,
    'fuzzy': fuzzy_scores
}

# Initialize ranker
ranker = HybridRanker(default_weights={
    'bm25': 0.3,
    'semantic': 0.5,
    'fuzzy': 0.2
})

# Weighted fusion
results = ranker.rank_documents(all_scores, method='weighted')
print("Weighted Fusion Results:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['doc_id']}")
    print(f"   Combined score: {result['score']:.4f}")
    print(f"   BM25: {result['individual_scores']['bm25']:.4f}")
    print(f"   Semantic: {result['individual_scores']['semantic']:.4f}")
    print(f"   Fuzzy: {result['individual_scores']['fuzzy']:.4f}")
# Output:
#   1. doc1 (Combined: 0.8850, BM25: 1.0, Semantic: 0.85, Fuzzy: 0.95)
#   2. doc3 (Combined: 0.7200, BM25: 0.2, Semantic: 0.90, Fuzzy: 0.30)
#   3. doc2 (Combined: 0.6750, BM25: 0.72, Semantic: 0.65, Fuzzy: 0.70)

# Reciprocal rank fusion
results_rrf = ranker.rank_documents(all_scores, method='rrf')
print("\nReciprocal Rank Fusion:")
for i, result in enumerate(results_rrf, 1):
    print(f"{i}. {result['doc_id']} (score: {result['score']:.4f})")

# Custom weights
custom_results = ranker.rank_documents(
    all_scores,
    method='weighted',
    weights={'bm25': 0.7, 'semantic': 0.2, 'fuzzy': 0.1}
)
print("\nCustom Weights (BM25 heavy):")
for result in custom_results:
    print(f"  {result['doc_id']}: {result['score']:.4f}")

# Only use subset of scores
partial_scores = {
    'bm25': bm25_scores,
    'semantic': semantic_scores
}
results_partial = ranker.rank_documents(
    partial_scores,
    method='weighted',
    weights={'bm25': 0.5, 'semantic': 0.5}
)
print("\nUsing only BM25 and Semantic:")
for result in results_partial:
    print(f"  {result['doc_id']}: {result['score']:.4f}")
```

**Use Case:** Combine strengths of different retrieval algorithms for better results.

---

## 16. retrieval_engine.py

### Purpose

Main retrieval orchestrator that integrates all retrieval methods and provides a unified search interface.

### Key Features

- Multiple retrieval methods (BM25, TF-IDF, semantic, fuzzy, hybrid)
- Cross-lingual retrieval
- Result enhancement with metadata
- Query processing integration
- Configurable retrieval strategies

### Main Class: `RetrievalEngine`

#### Methods:

- `retrieve(query, method, top_k)` - Main search method
- `retrieve_cross_lingual(query, top_k)` - Cross-lingual search
- Internal methods for each retrieval type

### Simple Example

```python
from src.module_c_retrieval import RetrievalEngine
from src.module_a_indexing import InvertedIndex

# Load inverted index
index = InvertedIndex()
index.load_index('processed_data/inverted_index.pkl')

# Prepare documents
documents = []
for doc_id, metadata in index.doc_metadata.items():
    doc = {
        'doc_id': doc_id,
        'tokens': metadata.get('tokens', []),
        'content': metadata.get('content', ''),
        'metadata': metadata
    }
    documents.append(doc)

# Initialize engine
engine = RetrievalEngine(
    inverted_index=index,
    documents=documents,
    enable_semantic=True,
    enable_fuzzy=True
)

# BM25 retrieval
query = "Sheikh Hasina education policy"
results = engine.retrieve(query, method='bm25', top_k=5)

print("BM25 Results:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['title']}")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Source: {result['source']}")
    print(f"   Snippet: {result['snippet'][:100]}...")
    print()

# Semantic retrieval
results_semantic = engine.retrieve(query, method='semantic', top_k=5)
print("\nSemantic Results:")
for i, result in enumerate(results_semantic, 1):
    print(f"{i}. {result['title']} (Score: {result['score']:.4f})")

# Hybrid retrieval (RECOMMENDED)
results_hybrid = engine.retrieve(query, method='hybrid', top_k=10)
print("\nHybrid Results (BM25 + Semantic + Fuzzy):")
for i, result in enumerate(results_hybrid, 1):
    print(f"{i}. {result['title']}")
    print(f"   Combined Score: {result['score']:.4f}")
    if 'component_scores' in result:
        print(f"   BM25: {result['component_scores'].get('bm25', 0):.4f}")
        print(f"   Semantic: {result['component_scores'].get('semantic', 0):.4f}")
        print(f"   Fuzzy: {result['component_scores'].get('fuzzy', 0):.4f}")
    print()

# Cross-lingual retrieval
query_en = "prime minister education"
results_cross = engine.retrieve_cross_lingual(query_en, top_k=10)

print("\nCross-lingual Results:")
print("English documents:")
for result in results_cross[:5]:
    if result.get('language') == 'en':
        print(f"  - {result['title']} ({result['score']:.4f})")

print("\nBangla documents:")
for result in results_cross[:5]:
    if result.get('language') == 'bn':
        print(f"  - {result['title']} ({result['score']:.4f})")

# Try different methods
methods = ['bm25', 'tfidf', 'semantic', 'fuzzy', 'hybrid']
query = "health ministry"
print(f"\nComparing methods for query: '{query}'")
for method in methods:
    results = engine.retrieve(query, method=method, top_k=3)
    print(f"\n{method.upper()}:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['title'][:50]} ({result['score']:.4f})")

# Get more results
results_large = engine.retrieve(query, method='hybrid', top_k=50)
print(f"\nRetrieved {len(results_large)} documents")
```

**Use Case:** Main entry point for all search operations in the CLIR system.

---

## Complete Pipeline Example

Here's how all modules work together:

```python
from src.module_a_indexing import DocumentProcessor, InvertedIndex
from src.module_b_query_processing import QueryProcessor
from src.module_c_retrieval import RetrievalEngine

# Step 1: Index documents (run once)
print("=" * 50)
print("STEP 1: INDEXING DOCUMENTS")
print("=" * 50)
processor = DocumentProcessor('data/raw', 'processed_data')
stats = processor.process_all_documents()
print(f"Indexed {stats['processing']['processed']} documents")
print(f"Languages: EN={stats['processing']['by_language']['en']}, "
      f"BN={stats['processing']['by_language']['bn']}")

# Step 2: Load index and prepare retrieval
print("\n" + "=" * 50)
print("STEP 2: LOADING INDEX")
print("=" * 50)
index = InvertedIndex()
index.load_index('processed_data/inverted_index.pkl')

documents = []
for doc_id, metadata in index.doc_metadata.items():
    documents.append({
        'doc_id': doc_id,
        'tokens': metadata.get('tokens', []),
        'content': metadata.get('content', ''),
        'metadata': metadata
    })
print(f"Loaded {len(documents)} documents")

# Step 3: Initialize query processor and retrieval engine
print("\n" + "=" * 50)
print("STEP 3: INITIALIZING SEARCH ENGINE")
print("=" * 50)
query_processor = QueryProcessor(
    enable_translation=True,
    enable_expansion=True,
    enable_entity_mapping=True
)
retrieval_engine = RetrievalEngine(
    inverted_index=index,
    documents=documents,
    enable_semantic=True,
    enable_fuzzy=True
)
print("Search engine ready!")

# Step 4: Process and search
print("\n" + "=" * 50)
print("STEP 4: SEARCHING")
print("=" * 50)

# English query
query = "Sheikh Hasina education policy"
print(f"Query: {query}")

# Process query
processed = query_processor.process_query(query)
print(f"\nProcessed Query:")
print(f"  Language: {processed['language']}")
print(f"  English:  {processed['en_query']}")
print(f"  Bangla:   {processed['bn_query']}")
print(f"  Expanded: {processed['expanded_en'][:60]}...")

# Retrieve English documents
print(f"\n--- English Documents ---")
en_results = retrieval_engine.retrieve(
    processed['en_query'],
    method='hybrid',
    top_k=5
)
for i, result in enumerate(en_results, 1):
    if result.get('language') == 'en':
        print(f"{i}. {result['title']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Snippet: {result['snippet'][:80]}...")

# Retrieve Bangla documents
print(f"\n--- Bangla Documents ---")
bn_results = retrieval_engine.retrieve(
    processed['bn_query'],
    method='hybrid',
    top_k=5
)
for i, result in enumerate(bn_results, 1):
    if result.get('language') == 'bn':
        print(f"{i}. {result['title']}")
        print(f"   Score: {result['score']:.4f}")

print("\n" + "=" * 50)
print("SEARCH COMPLETE!")
print("=" * 50)
```

---

## Summary

### Module A (Indexing)

1. **language_detector** - Identify document language
2. **tokenizer** - Split text into tokens
3. **ner_extractor** - Extract named entities
4. **inverted_index** - Build searchable index
5. **document_processor** - Orchestrate indexing

### Module B (Query Processing)

6. **query_detector** - Detect query language
7. **query_normalizer** - Clean and normalize
8. **query_translator** - Translate EN ↔ BN
9. **query_expander** - Add synonyms
10. **ne_mapper** - Map entities across languages
11. **query_pipeline** - Orchestrate query processing

### Module C (Retrieval)

12. **lexical_retrieval** - BM25/TF-IDF ranking
13. **fuzzy_matcher** - Handle typos
14. **semantic_retrieval** - Meaning-based search
15. **hybrid_ranker** - Combine multiple scores
16. **retrieval_engine** - Main search interface

Each module is independent and can be tested separately, but they work best when integrated together for a complete CLIR system.
