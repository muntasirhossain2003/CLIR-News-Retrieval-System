"""
Error Analysis Module

Analyzes retrieval failures and identifies common error patterns:
- Translation failures
- Named entity mismatches
- Cross-script issues
- Code-switching problems
- Semantic vs. lexical wins
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ErrorCase:
    """Represents a single error case."""
    query_id: str
    query_text: str
    query_language: str
    error_type: str  # "translation", "ner_mismatch", "script", "code_switch", "semantic_win", "lexical_win"
    error_description: str
    expected_docs: List[str]
    retrieved_docs: List[str]
    example_doc: str = ""


class ErrorAnalyzer:
    """
    Analyzes retrieval failures and error patterns.
    """

    def __init__(self):
        self.error_cases: List[ErrorCase] = []

    def add_translation_failure(
        self,
        query_id: str,
        query_text: str,
        query_language: str,
        original_query: str,
        mistranslated_query: str,
        expected_docs: List[str],
        retrieved_docs: List[str],
        example: str = "",
    ):
        """
        Log translation failure.

        Example:
        - Query "চেয়ার" (chair) mistranslated to "Chairman"
        """
        error = ErrorCase(
            query_id=query_id,
            query_text=query_text,
            query_language=query_language,
            error_type="translation",
            error_description=f'Mistranslation: "{original_query}" → "{mistranslated_query}"',
            expected_docs=expected_docs,
            retrieved_docs=retrieved_docs,
            example_doc=example,
        )
        self.error_cases.append(error)

    def add_ner_mismatch(
        self,
        query_id: str,
        query_text: str,
        query_language: str,
        entity_in_query: str,
        entity_in_docs: str,
        expected_docs: List[str],
        retrieved_docs: List[str],
        example: str = "",
    ):
        """
        Log Named Entity Recognition mismatch.

        Example:
        - Query mentions "ঢাকা" (Dhaka) but documents use "Dhaka" in English
        """
        error = ErrorCase(
            query_id=query_id,
            query_text=query_text,
            query_language=query_language,
            error_type="ner_mismatch",
            error_description=f'NER Mismatch: "{entity_in_query}" (query) vs "{entity_in_docs}" (docs)',
            expected_docs=expected_docs,
            retrieved_docs=retrieved_docs,
            example_doc=example,
        )
        self.error_cases.append(error)

    def add_cross_script_issue(
        self,
        query_id: str,
        query_text: str,
        query_language: str,
        script_variant_1: str,
        script_variant_2: str,
        expected_docs: List[str],
        retrieved_docs: List[str],
        example: str = "",
    ):
        """
        Log cross-script ambiguity issue.

        Example:
        - "Bangladesh" could be transliterated as "বাংলাদেশ" or "Bangla Desh"
        """
        error = ErrorCase(
            query_id=query_id,
            query_text=query_text,
            query_language=query_language,
            error_type="script",
            error_description=f'Cross-script issue: "{script_variant_1}" vs "{script_variant_2}"',
            expected_docs=expected_docs,
            retrieved_docs=retrieved_docs,
            example_doc=example,
        )
        self.error_cases.append(error)

    def add_code_switching_issue(
        self,
        query_id: str,
        query_text: str,
        mixed_components: List[str],
        expected_docs: List[str],
        retrieved_docs: List[str],
        example: str = "",
    ):
        """
        Log code-switching issue (mixed language query).

        Example:
        - Query: "আমরা COVID-19 এর বিরুদ্ধে লড়াই করছি" (Bangla + English)
        """
        error = ErrorCase(
            query_id=query_id,
            query_text=query_text,
            query_language="code-mixed",
            error_type="code_switch",
            error_description=f"Code-switching detected in components: {mixed_components}",
            expected_docs=expected_docs,
            retrieved_docs=retrieved_docs,
            example_doc=example,
        )
        self.error_cases.append(error)

    def add_semantic_vs_lexical(
        self,
        query_id: str,
        query_text: str,
        query_language: str,
        winner: str,  # "semantic" or "lexical"
        lexical_results: List[str],
        semantic_results: List[str],
        example: str = "",
    ):
        """
        Log case where semantic retrieval outperforms lexical (or vice versa).

        Example:
        - Query "শিক্ষা" (education): BM25 returns 0 results, but embedding model retrieves "স্কুল" (school)
        """
        error = ErrorCase(
            query_id=query_id,
            query_text=query_text,
            query_language=query_language,
            error_type=f"{winner}_win",
            error_description=f"{winner.capitalize()} retrieval significantly outperformed",
            expected_docs=semantic_results if winner == "semantic" else lexical_results,
            retrieved_docs=lexical_results if winner == "semantic" else semantic_results,
            example_doc=example,
        )
        self.error_cases.append(error)

    def summarize_errors(self) -> Dict[str, int]:
        """
        Summarize error types and frequencies.

        Returns:
            Dict mapping error_type -> count
        """
        error_summary = {}

        for error in self.error_cases:
            error_summary[error.error_type] = error_summary.get(error.error_type, 0) + 1

        return error_summary

    def get_errors_by_type(self, error_type: str) -> List[ErrorCase]:
        """Get all errors of a specific type."""
        return [e for e in self.error_cases if e.error_type == error_type]

    def get_translation_failures(self) -> List[ErrorCase]:
        """Get all translation failure errors."""
        return self.get_errors_by_type("translation")

    def get_ner_mismatches(self) -> List[ErrorCase]:
        """Get all NER mismatch errors."""
        return self.get_errors_by_type("ner_mismatch")

    def get_script_issues(self) -> List[ErrorCase]:
        """Get all cross-script issues."""
        return self.get_errors_by_type("script")

    def get_code_switching_issues(self) -> List[ErrorCase]:
        """Get all code-switching issues."""
        return self.get_errors_by_type("code_switch")

    def format_error_report(self) -> str:
        """Format error analysis report."""
        output = []
        output.append("\n" + "=" * 80)
        output.append("ERROR ANALYSIS REPORT")
        output.append("=" * 80)

        summary = self.summarize_errors()

        if not summary:
            output.append("\n✓ No errors detected!\n")
            return "\n".join(output)

        output.append(f"\nTotal Errors: {len(self.error_cases)}\n")

        output.append("ERROR SUMMARY BY TYPE:")
        output.append("─" * 40)

        for error_type, count in sorted(summary.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.error_cases)) * 100
            output.append(f"  {error_type:.<20} {count:>3} ({percentage:>5.1f}%)")

        output.append("\n" + "─" * 80 + "\n")

        # Detailed error examples
        for error_type in sorted(summary.keys()):
            errors = self.get_errors_by_type(error_type)
            output.append(f"\n{error_type.upper()} ERRORS ({len(errors)} cases):")
            output.append("─" * 40)

            for i, error in enumerate(errors[:5], 1):  # Show first 5
                output.append(f"\n{i}. Query ID: {error.query_id}")
                output.append(f"   Query: {error.query_text}")
                output.append(f"   Language: {error.query_language}")
                output.append(f"   Issue: {error.error_description}")
                output.append(f"   Expected Docs: {error.expected_docs[:3]}")
                output.append(f"   Retrieved Docs: {error.retrieved_docs[:3]}")
                if error.example_doc:
                    output.append(f"   Example: {error.example_doc[:100]}...")

            if len(errors) > 5:
                output.append(f"\n   ... and {len(errors) - 5} more cases")

        return "\n".join(output)

    def format_error_summary_table(self) -> str:
        """Format error summary as table."""
        output = []
        output.append("\n📋 ERROR SUMMARY TABLE")
        output.append("─" * 60)
        output.append(
            f"{'Error Type':<20} {'Count':<10} {'Percentage':<15} {'Severity':<15}"
        )
        output.append("─" * 60)

        summary = self.summarize_errors()
        total = len(self.error_cases)

        for error_type, count in sorted(summary.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0

            # Severity based on frequency
            if percentage >= 30:
                severity = "🔴 Critical"
            elif percentage >= 15:
                severity = "🟠 High"
            elif percentage >= 5:
                severity = "🟡 Medium"
            else:
                severity = "🟢 Low"

            output.append(
                f"{error_type:<20} {count:<10} {percentage:>6.1f}%{'':<8} {severity:<15}"
            )

        output.append("─" * 60)
        output.append(f"{'TOTAL':<20} {total:<10} {'100.0%':<15}")

        return "\n".join(output)
