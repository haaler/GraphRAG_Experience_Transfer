"""Retrieval metric used by the evaluation runner."""

from typing import List, Optional


def document_recall(retrieved_docs: List[str], expected_docs: List[str]) -> Optional[float]:
    """Fraction of expected documents that were actually retrieved.

    Returns None when no expected documents are given (for example,
    questions in the 'negative' category). Returning None lets the runner
    exclude those rows from the mean rather than skewing it with a 1.0.
    """
    if not expected_docs:
        return None

    hits = len(set(retrieved_docs) & set(expected_docs))
    return hits / len(expected_docs)
