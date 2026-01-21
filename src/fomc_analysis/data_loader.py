"""
data_loader
===========

Functions and classes for loading and parsing Federal Reserve press
conference transcripts.  This module provides a `Transcript`
dataclass that holds the contents of a transcript and helper
functions to convert PDF files to text, segment the text by
speaker, and extract the remarks made by Chair Jerome Powell.

The parsing logic is intentionally conservative: it looks for
speaker labels such as ``"CHAIR POWELL:"`` and collects all lines
following that label until a new speaker label appears.  If your
transcripts use a different format, you may need to adjust the
regular expressions.  See the examples in README.md for guidance.

Note that this module does not perform any phrase counting – that
logic lives in :mod:`feature_extraction`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Iterable

import fitz  # PyMuPDF


SPEAKER_PATTERN = re.compile(
    r"^(CHAIR\s+POWELL|MR\.|MS\.|CHAIR|GOV\.|VICE\s+CHAIR)[^:]*:"
)
POWELL_LABELS = [
    "CHAIR POWELL",
    "Chair Powell",
    "CHAIR POWELL",  # with non‑breaking space
    "CHAIR POWELL.",
]


@dataclass
class Transcript:
    """Represents a single press conference transcript.

    Attributes
    ----------
    file_path: Path
        The original path to the PDF or text file.
    date: Optional[str]
        A human‑readable date extracted from the filename or metadata.
    raw_text: str
        The full text of the transcript with no speaker segmentation.
    speaker_segments: List[Tuple[str, str]]
        A list of (speaker, utterance) tuples in the order they appear
        in the transcript.
    powell_text: str
        A concatenation of all utterances attributed to Chair Powell.
    """

    file_path: Path
    date: Optional[str] = None
    raw_text: str = ""
    speaker_segments: List[Tuple[str, str]] = field(default_factory=list)
    powell_text: str = ""

    @classmethod
    def from_file(cls, file_path: Path, date: Optional[str] = None) -> "Transcript":
        """Load a transcript from a PDF or plain‑text file.

        Parameters
        ----------
        file_path: Path
            The path to the PDF or text transcript.  PDF files are
            converted to text using PyMuPDF; plain text files are read
            as‑is.
        date: Optional[str]
            The date associated with the transcript.  If not supplied,
            the method will attempt to infer it from the filename (see
            :func:`_infer_date_from_filename`).

        Returns
        -------
        Transcript
            A populated Transcript instance.
        """
        if file_path.suffix.lower() == ".pdf":
            text = _extract_text_from_pdf(file_path)
        else:
            text = file_path.read_text(encoding="utf-8")

        if date is None:
            date = _infer_date_from_filename(file_path.name)

        speaker_segments = list(_segment_by_speaker(text))
        powell_text = "\n".join(
            utterance.strip()
            for speaker, utterance in speaker_segments
            if speaker.upper().startswith("CHAIR POWELL") or speaker.lower().startswith("chair powell")
        )
        return cls(
            file_path=file_path,
            date=date,
            raw_text=text,
            speaker_segments=speaker_segments,
            powell_text=powell_text,
        )


def _extract_text_from_pdf(file_path: Path) -> str:
    """Extract plain text from a PDF file using PyMuPDF.

    Parameters
    ----------
    file_path: Path
        The path to the PDF file.

    Returns
    -------
    str
        The concatenated text of all pages.

    Notes
    -----
    PyMuPDF may warn about missing fonts or unsupported characters.
    Those warnings do not typically impact the extracted text.  If
    certain pages are images or scanned, you may need to use OCR
    separately (not provided here).
    """
    doc = fitz.open(file_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def _infer_date_from_filename(filename: str) -> Optional[str]:
    """Attempt to infer a date string from a transcript filename.

    The function looks for patterns like YYYYMMDD or YYYY‑MM‑DD in the
    filename and returns them in a standard ISO format.  If no date is
    found, returns None.

    Examples
    --------
    >>> _infer_date_from_filename("FOMCpressconf20251210.pdf")
    '2025-12-10'
    >>> _infer_date_from_filename("pressconf_2024-05-01.txt")
    '2024-05-01'
    >>> _infer_date_from_filename("randomfile.txt")
    None
    """
    match = re.search(r"(20\d{2})(?:-|)?(0\d|1[0-2])(?:-|)?(0\d|[12]\d|3[01])", filename)
    if not match:
        return None
    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def _segment_by_speaker(text: str) -> Iterable[Tuple[str, str]]:
    """Split a transcript into (speaker, utterance) pairs.

    Parameters
    ----------
    text: str
        The raw transcript text.

    Yields
    ------
    Tuple[str, str]
        Pairs of speaker label and their spoken text.  The speaker
        label is stripped of trailing punctuation and whitespace.  The
        utterance is stripped of leading/trailing whitespace.

    Notes
    -----
    This function assumes that speaker labels are on their own line
    followed by a colon, e.g. ``"CHAIR POWELL:"`` or ``"MR. SMITH:"``.
    Lines that do not match the speaker pattern are appended to the
    previous speaker's utterance.  If the transcript does not follow
    this format, you may need to customise the regular expression in
    :data:`SPEAKER_PATTERN`.
    """
    current_speaker = "Unknown"
    buffer = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if SPEAKER_PATTERN.match(line):
            # yield previous segment
            if buffer:
                yield (current_speaker, " ".join(buffer).strip())
                buffer = []
            # new speaker
            speaker_label, _, remainder = line.partition(":")
            current_speaker = speaker_label.strip()
            if remainder:
                buffer.append(remainder.strip())
        else:
            buffer.append(line)
    if buffer:
        yield (current_speaker, " ".join(buffer).strip())


def load_transcripts(directory: str | Path) -> List[Transcript]:
    """Load all transcripts from a directory.

    Parameters
    ----------
    directory: str or Path
        The directory containing PDF or text transcripts.

    Returns
    -------
    List[Transcript]
        A list of Transcript objects sorted by date (ascending) when
        dates are available.  Files without a parsable date appear at
        the end.
    """
    directory = Path(directory)
    transcripts: List[Transcript] = []
    for file_path in sorted(directory.iterdir()):
        if file_path.suffix.lower() not in {".pdf", ".txt"}:
            continue
        transcript = Transcript.from_file(file_path)
        transcripts.append(transcript)
    # sort by date if available
    transcripts.sort(key=lambda t: t.date or "")
    return transcripts


def extract_powell_text(transcripts: List[Transcript]) -> List[str]:
    """Extract only Powell's remarks from a list of transcripts.

    Parameters
    ----------
    transcripts: List[Transcript]
        A list of Transcript objects.

    Returns
    -------
    List[str]
        A list of strings, each containing the concatenated Powell
        utterances from a transcript.  The order matches the order of
        the input list.
    """
    return [t.powell_text for t in transcripts]