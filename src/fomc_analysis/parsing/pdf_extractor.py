"""
Stage A: Deterministic PDF extraction.

This module provides deterministic extraction of text from FOMC press conference
PDFs using PyMuPDF (fitz). The pipeline is:
1. Raw PDF → raw_text (with page markers)
2. raw_text → clean_text (normalized whitespace, dehyphenation)

All operations are deterministic and reproducible.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


def extract_pdf_to_text(pdf_path: Path, include_page_markers: bool = True) -> str:
    """
    Extract raw plaintext from a PDF file using PyMuPDF.

    This is a deterministic extraction that preserves all text content
    from the PDF. Page markers can optionally be included to track
    which page each section of text comes from.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file to extract.
    include_page_markers : bool, default=True
        If True, insert `--- Page N ---` markers between pages.

    Returns
    -------
    str
        The raw extracted text with optional page markers.

    Notes
    -----
    This function is deterministic and will produce the same output
    for the same PDF file across different runs.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if include_page_markers:
            pages.append(f"--- Page {page_num} ---\n{text}")
        else:
            pages.append(text)

    doc.close()
    return "\n".join(pages)


def clean_text(raw_text: str) -> str:
    """
    Clean raw text by normalizing whitespace and removing hyphenation.

    This performs deterministic cleaning operations:
    1. Remove page markers (if present)
    2. Fix line-end hyphenation (e.g., "eco-\\nnomic" → "economic")
    3. Normalize multiple whitespace to single space
    4. Normalize line breaks

    Parameters
    ----------
    raw_text : str
        Raw text extracted from PDF.

    Returns
    -------
    str
        Cleaned text with normalized whitespace and dehyphenation.

    Notes
    -----
    This function is deterministic and reproducible.
    """
    # Remove page markers
    text = re.sub(r"^--- Page \d+ ---\n?", "", raw_text, flags=re.MULTILINE)

    # Fix line-end hyphenation: "word-\n" → "word"
    # Match hyphen followed by newline and lowercase letter
    text = re.sub(r"-\s*\n\s*([a-z])", r"\1", text)

    # Normalize multiple spaces to single space
    text = re.sub(r" +", " ", text)

    # Normalize multiple newlines to at most 2 (preserve paragraph breaks)
    text = re.sub(r"\n\n\n+", "\n\n", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def extract_page_number(text: str, position: int) -> Optional[int]:
    """
    Extract the page number for a given character position in raw text.

    Parameters
    ----------
    text : str
        Raw text with page markers (from extract_pdf_to_text).
    position : int
        Character position in the text.

    Returns
    -------
    Optional[int]
        Page number (1-indexed), or None if no page marker found.
    """
    # Find all page markers before this position
    markers = re.finditer(r"^--- Page (\d+) ---", text[:position], flags=re.MULTILINE)
    page_num = None
    for match in markers:
        page_num = int(match.group(1))
    return page_num
