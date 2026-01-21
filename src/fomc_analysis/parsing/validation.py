"""
Validation utilities for parsing pipeline.

This module provides functions to validate that AI-generated outputs
match the ground truth deterministic extractions.
"""

from __future__ import annotations

from difflib import SequenceMatcher


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute similarity between two text strings.

    Uses Python's difflib SequenceMatcher to compute a similarity ratio.

    Parameters
    ----------
    text1 : str
        First text to compare.
    text2 : str
        Second text to compare.

    Returns
    -------
    float
        Similarity score between 0 and 1 (1 = identical).
    """
    return SequenceMatcher(None, text1, text2).ratio()


def compute_coverage(segments_text: str, original_text: str) -> float:
    """
    Compute coverage: what fraction of original text is present in segments.

    Parameters
    ----------
    segments_text : str
        Concatenated text from segments.
    original_text : str
        Original full text.

    Returns
    -------
    float
        Coverage ratio (0-1).
    """
    if not original_text:
        return 1.0 if not segments_text else 0.0

    return len(segments_text) / len(original_text)
