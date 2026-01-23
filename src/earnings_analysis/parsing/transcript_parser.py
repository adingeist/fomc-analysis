"""
Basic transcript parsing utilities.

Handles cleaning and preprocessing of raw transcript text.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


class TranscriptParser:
    """
    Parse and clean earnings call transcripts.

    Handles various transcript formats and performs basic cleaning.
    """

    def __init__(self):
        pass

    def parse(self, raw_text: str) -> str:
        """
        Parse and clean raw transcript text.

        Parameters
        ----------
        raw_text : str
            Raw transcript text

        Returns
        -------
        str
            Cleaned transcript text
        """
        text = raw_text

        # Remove common transcript artifacts
        text = self._remove_artifacts(text)

        # Normalize whitespace
        text = self._normalize_whitespace(text)

        # Fix hyphenation at line breaks
        text = self._fix_hyphenation(text)

        return text

    def _remove_artifacts(self, text: str) -> str:
        """Remove common transcript artifacts."""
        # Remove page numbers
        text = re.sub(r"Page \d+ of \d+", "", text)

        # Remove timestamps (e.g., [00:15:32])
        text = re.sub(r"\[\d{2}:\d{2}:\d{2}\]", "", text)

        # Remove common headers/footers
        text = re.sub(r"Earnings Conference Call.*?\n", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Copyright.*?\n", "", text, flags=re.IGNORECASE)

        # Remove excessive disclaimer text
        text = re.sub(
            r"This transcript.*?forward-looking statements.*?\.",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)

        # Replace multiple newlines with double newline
        text = re.sub(r"\n\n+", "\n\n", text)

        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def _fix_hyphenation(self, text: str) -> str:
        """Fix words hyphenated across line breaks."""
        # Pattern: word- \n word
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

        return text

    def extract_sections(self, text: str) -> dict:
        """
        Extract major sections from transcript.

        Common sections:
        - Prepared Remarks (executives)
        - Q&A Session

        Returns
        -------
        dict
            Sections: {"prepared_remarks": str, "qa_session": str}
        """
        sections = {}

        # Try to find Q&A section
        qa_patterns = [
            r"Questions? and Answers?",
            r"Q&A",
            r"Question-and-Answer Session",
            r"Operator.*?question",
        ]

        qa_start_match = None
        for pattern in qa_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                qa_start_match = match
                break

        if qa_start_match:
            qa_start = qa_start_match.start()
            sections["prepared_remarks"] = text[:qa_start].strip()
            sections["qa_session"] = text[qa_start:].strip()
        else:
            # No clear Q&A section found
            sections["prepared_remarks"] = text
            sections["qa_session"] = ""

        return sections


def parse_transcript(
    file_path: Path,
    clean: bool = True,
    extract_sections: bool = False,
) -> str | dict:
    """
    Convenience function to parse a transcript file.

    Parameters
    ----------
    file_path : Path
        Path to transcript file
    clean : bool
        Whether to clean the text
    extract_sections : bool
        Whether to extract prepared remarks and Q&A sections

    Returns
    -------
    str or dict
        Cleaned transcript text or dict of sections
    """
    parser = TranscriptParser()

    raw_text = Path(file_path).read_text(encoding="utf-8", errors="ignore")

    if clean:
        text = parser.parse(raw_text)
    else:
        text = raw_text

    if extract_sections:
        return parser.extract_sections(text)

    return text
