"""
Parsing pipeline for FOMC press conference transcripts.

This package implements a two-stage parsing pipeline:
- Stage A: Deterministic PDF extraction to raw and clean text
- Stage B: Speaker segmentation (deterministic with optional AI repair)
"""

from .pdf_extractor import extract_pdf_to_text, clean_text
from .speaker_segmenter import segment_speakers, validate_segments
from .validation import compute_text_similarity

__all__ = [
    "extract_pdf_to_text",
    "clean_text",
    "segment_speakers",
    "validate_segments",
    "compute_text_similarity",
]
