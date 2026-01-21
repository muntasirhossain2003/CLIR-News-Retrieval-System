"""
Complete Query Processing Pipeline

Integrates language detection, normalization, named entity extraction, and translation.
"""

import argparse
import json
from language_detection_normalization import process_query
from named_entity_extraction import process_query_with_entities
from query_translation import process_query_with_translation


def process_complete_query(user_query: str, target_lang: str = None) -> dict:
    # Step 1: Language detection & normalization
    result = process_query(user_query)

    # Step 2: Named entity extraction
    result = process_query_with_entities(result)

    # Step 3: Translation (optional)
    if target_lang is not None:
        result = process_query_with_translation(result, target_lang)

    return result


def main():
    """Command line interface for query processing."""
    parser = argparse.ArgumentParser(
        description="Process queries for CLIR system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query processing
  python query_pipeline.py "COVID-19 pandemic in Bangladesh"
  
  # With translation to Bangla
  python query_pipeline.py "climate change" --target bn
  
  # Bangla query with translation to English
  python query_pipeline.py "জলবায়ু পরিবর্তন" --target en
        """,
    )

    parser.add_argument("query", type=str, help="Query text to process")

    parser.add_argument(
        "--target",
        "-t",
        type=str,
        choices=["bn", "en"],
        default=None,
        help="Target language for translation (bn or en). If not specified, no translation is performed.",
    )

    parser.add_argument("--json", action="store_true", help="Output result as JSON")

    args = parser.parse_args()

    # Process query
    result = process_complete_query(args.query, target_lang=args.target)

    # Output
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"\nQuery: {result['original_query']}")
        print(f"Language: {result['language']}")
        print(f"Normalized: {result['normalized_query']}")
        print(f"Entities: {result.get('entities', [])}")
        if "translated_query" in result:
            print(f"Translation: {result['translated_query']}")
        print()


if __name__ == "__main__":
    main()
