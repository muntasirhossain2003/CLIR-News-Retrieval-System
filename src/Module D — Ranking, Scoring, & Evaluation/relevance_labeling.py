"""
Relevance Labeling Module

Utilities for managing relevance labels and creating annotated datasets.
"""

import csv
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class RelevanceLabel:
    """Represents a relevance label for a document."""
    query_id: str
    query_text: str
    doc_id: str
    doc_title: str
    doc_url: str
    language: str
    relevant: bool  # True/False
    confidence: int = 1  # 0-3 scale for confidence in labeling
    annotator: str = "default"
    notes: str = ""


class RelevanceLabeler:
    """
    Manages relevance labeling for queries and documents.
    """

    def __init__(self):
        self.labels: List[RelevanceLabel] = []

    def add_label(
        self,
        query_id: str,
        query_text: str,
        doc_id: str,
        doc_title: str,
        doc_url: str,
        language: str,
        relevant: bool,
        confidence: int = 1,
        annotator: str = "default",
        notes: str = "",
    ):
        """
        Add a relevance label.

        Args:
            query_id: Unique query identifier
            query_text: Query text
            doc_id: Document ID
            doc_title: Document title
            doc_url: Document URL
            language: Document language
            relevant: Whether document is relevant to query
            confidence: Annotator confidence (0-3)
            annotator: Name of annotator
            notes: Additional notes about labeling
        """
        label = RelevanceLabel(
            query_id=query_id,
            query_text=query_text,
            doc_id=doc_id,
            doc_title=doc_title,
            doc_url=doc_url,
            language=language,
            relevant=relevant,
            confidence=confidence,
            annotator=annotator,
            notes=notes,
        )
        self.labels.append(label)

    def add_labels_batch(self, labels: List[RelevanceLabel]):
        """Add multiple labels at once."""
        self.labels.extend(labels)

    def get_relevant_docs_for_query(self, query_id: str) -> List[str]:
        """Get list of relevant doc_ids for a query."""
        return [
            label.doc_id
            for label in self.labels
            if label.query_id == query_id and label.relevant
        ]

    def get_all_docs_for_query(self, query_id: str) -> List[str]:
        """Get all doc_ids labeled for a query (relevant or not)."""
        return [label.doc_id for label in self.labels if label.query_id == query_id]

    def get_query_labels(self, query_id: str) -> List[RelevanceLabel]:
        """Get all labels for a specific query."""
        return [label for label in self.labels if label.query_id == query_id]

    def get_labels_by_annotator(self, annotator: str) -> List[RelevanceLabel]:
        """Get all labels from a specific annotator."""
        return [label for label in self.labels if label.annotator == annotator]

    def get_high_confidence_labels(self, min_confidence: int = 2) -> List[RelevanceLabel]:
        """Get labels with high annotator confidence."""
        return [label for label in self.labels if label.confidence >= min_confidence]

    def save_to_csv(self, filepath: str):
        """
        Save labels to CSV file.

        CSV Format:
        query_id, query_text, doc_id, doc_title, doc_url, language, relevant, confidence, annotator, notes
        """
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self.labels[0]).keys())
            writer.writeheader()

            for label in self.labels:
                writer.writerow(asdict(label))

        print(f"✓ Saved {len(self.labels)} labels to {filepath}")

    def load_from_csv(self, filepath: str):
        """Load labels from CSV file."""
        self.labels = []

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Convert string "True"/"False" to boolean
                relevant_str = row.get("relevant", "False").strip().lower()
                relevant = relevant_str in ["true", "yes", "1"]

                confidence = int(row.get("confidence", 1))

                label = RelevanceLabel(
                    query_id=row.get("query_id", ""),
                    query_text=row.get("query_text", ""),
                    doc_id=row.get("doc_id", ""),
                    doc_title=row.get("doc_title", ""),
                    doc_url=row.get("doc_url", ""),
                    language=row.get("language", ""),
                    relevant=relevant,
                    confidence=confidence,
                    annotator=row.get("annotator", "default"),
                    notes=row.get("notes", ""),
                )
                self.labels.append(label)

        print(f"✓ Loaded {len(self.labels)} labels from {filepath}")

    def save_to_json(self, filepath: str):
        """Save labels to JSON file."""
        json_data = [asdict(label) for label in self.labels]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved {len(self.labels)} labels to {filepath}")

    def load_from_json(self, filepath: str):
        """Load labels from JSON file."""
        self.labels = []

        with open(filepath, "r", encoding="utf-8") as f:
            json_data = json.load(f)

            for item in json_data:
                # Convert string "true"/"false" to boolean
                if isinstance(item.get("relevant"), str):
                    relevant = item["relevant"].lower() in ["true", "yes", "1"]
                else:
                    relevant = bool(item.get("relevant", False))

                label = RelevanceLabel(
                    query_id=item.get("query_id", ""),
                    query_text=item.get("query_text", ""),
                    doc_id=item.get("doc_id", ""),
                    doc_title=item.get("doc_title", ""),
                    doc_url=item.get("doc_url", ""),
                    language=item.get("language", ""),
                    relevant=relevant,
                    confidence=item.get("confidence", 1),
                    annotator=item.get("annotator", "default"),
                    notes=item.get("notes", ""),
                )
                self.labels.append(label)

        print(f"✓ Loaded {len(self.labels)} labels from {filepath}")

    def get_statistics(self) -> Dict[str, any]:
        """Get labeling statistics."""
        total_labels = len(self.labels)
        relevant_count = sum(1 for label in self.labels if label.relevant)
        not_relevant_count = total_labels - relevant_count

        unique_queries = len(set(label.query_id for label in self.labels))
        unique_docs = len(set(label.doc_id for label in self.labels))
        unique_annotators = len(set(label.annotator for label in self.labels))

        avg_confidence = (
            sum(label.confidence for label in self.labels) / total_labels
            if total_labels > 0
            else 0
        )

        return {
            "total_labels": total_labels,
            "relevant_count": relevant_count,
            "not_relevant_count": not_relevant_count,
            "relevant_percentage": (relevant_count / total_labels * 100) if total_labels > 0 else 0,
            "unique_queries": unique_queries,
            "unique_documents": unique_docs,
            "unique_annotators": unique_annotators,
            "average_confidence": avg_confidence,
        }

    def format_statistics(self) -> str:
        """Format statistics for display."""
        stats = self.get_statistics()

        output = []
        output.append("\n📊 LABELING STATISTICS")
        output.append("─" * 50)
        output.append(f"Total Labels: {stats['total_labels']}")
        output.append(f"Relevant Documents: {stats['relevant_count']} ({stats['relevant_percentage']:.1f}%)")
        output.append(f"Not Relevant Documents: {stats['not_relevant_count']}")
        output.append(f"Unique Queries: {stats['unique_queries']}")
        output.append(f"Unique Documents: {stats['unique_documents']}")
        output.append(f"Unique Annotators: {stats['unique_annotators']}")
        output.append(f"Average Confidence: {stats['average_confidence']:.2f}/3.0")

        return "\n".join(output)

    @staticmethod
    def create_sample_labeling_csv(output_path: str):
        """
        Create a sample CSV template for manual labeling.

        Users can download this, manually fill it, and then load it back.
        """
        headers = [
            "query_id",
            "query_text",
            "doc_id",
            "doc_title",
            "doc_url",
            "language",
            "relevant",
            "confidence",
            "annotator",
            "notes",
        ]

        sample_rows = [
            {
                "query_id": "q001",
                "query_text": "climate change Bangladesh",
                "doc_id": "doc_123",
                "doc_title": "Climate Crisis in Bangladesh",
                "doc_url": "https://example.com/article",
                "language": "english",
                "relevant": "yes",
                "confidence": "3",
                "annotator": "annotator_1",
                "notes": "Directly addresses the query",
            },
            {
                "query_id": "q001",
                "query_text": "climate change Bangladesh",
                "doc_id": "doc_456",
                "doc_title": "Sports News",
                "doc_url": "https://example.com/sports",
                "language": "english",
                "relevant": "no",
                "confidence": "3",
                "annotator": "annotator_1",
                "notes": "Unrelated to query",
            },
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(sample_rows)

        print(f"✓ Created sample labeling template: {output_path}")

    def inter_annotator_agreement(self) -> float:
        """
        Calculate inter-annotator agreement (simple version).

        For queries with multiple annotators, calculates percentage of agreement.
        """
        # Group labels by (query_id, doc_id)
        annotations_by_pair = {}

        for label in self.labels:
            pair = (label.query_id, label.doc_id)
            if pair not in annotations_by_pair:
                annotations_by_pair[pair] = []
            annotations_by_pair[pair].append(label.relevant)

        # Find pairs with multiple annotators
        multi_annotated = {
            pair: relevances
            for pair, relevances in annotations_by_pair.items()
            if len(relevances) > 1
        }

        if not multi_annotated:
            return 1.0  # No disagreement if only one annotator

        # Calculate agreement
        agreements = 0
        total = len(multi_annotated)

        for pair, relevances in multi_annotated.items():
            # All annotators agree if all have same value
            if all(rel == relevances[0] for rel in relevances):
                agreements += 1

        return agreements / total if total > 0 else 1.0
