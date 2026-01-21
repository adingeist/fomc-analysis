"""
Tests for PDF extraction and speaker segmentation.
"""

import pytest
from pathlib import Path

from fomc_analysis.parsing.pdf_extractor import clean_text, extract_page_number
from fomc_analysis.parsing.speaker_segmenter import (
    segment_speakers_deterministic,
    classify_speaker_role,
    validate_segments,
    SpeakerTurn,
)
from fomc_analysis.parsing.validation import compute_text_similarity


class TestPDFExtraction:
    """Tests for PDF extraction utilities."""

    def test_clean_text_removes_page_markers(self):
        """Test that page markers are removed."""
        raw = "--- Page 1 ---\nHello world\n--- Page 2 ---\nFoo bar"
        cleaned = clean_text(raw)
        assert "Page 1" not in cleaned
        assert "Page 2" not in cleaned
        assert "Hello world" in cleaned
        assert "Foo bar" in cleaned

    def test_clean_text_fixes_hyphenation(self):
        """Test that line-end hyphenation is fixed."""
        raw = "This is an eco-\nnomic policy"
        cleaned = clean_text(raw)
        assert "economic" in cleaned
        assert "eco-\nnomic" not in cleaned

    def test_clean_text_normalizes_whitespace(self):
        """Test that multiple spaces are normalized."""
        raw = "This  has    multiple   spaces"
        cleaned = clean_text(raw)
        assert "  " not in cleaned
        assert "This has multiple spaces" in cleaned


class TestSpeakerSegmentation:
    """Tests for speaker segmentation."""

    def test_classify_speaker_role_powell(self):
        """Test Powell classification."""
        assert classify_speaker_role("CHAIR POWELL") == "powell"
        assert classify_speaker_role("CHAIRMAN POWELL") == "powell"
        assert classify_speaker_role("Powell") == "powell"

    def test_classify_speaker_role_reporter(self):
        """Test reporter classification."""
        assert classify_speaker_role("MR. SMITH") == "reporter"
        assert classify_speaker_role("MS. JONES") == "reporter"

    def test_classify_speaker_role_moderator(self):
        """Test moderator classification."""
        assert classify_speaker_role("MODERATOR") == "moderator"
        assert classify_speaker_role("MR. ENGLISH") == "moderator"

    def test_segment_speakers_basic(self):
        """Test basic speaker segmentation."""
        text = """CHAIR POWELL: Good afternoon. Welcome to our press conference.

MR. SMITH: Thank you, Chair. My question is about inflation.

CHAIR POWELL: That's a great question. Let me address that."""

        segments = segment_speakers_deterministic(text)

        assert len(segments) == 3
        assert segments[0].speaker == "CHAIR POWELL"
        assert segments[0].role == "powell"
        assert "Good afternoon" in segments[0].text

        assert segments[1].speaker == "MR. SMITH"
        assert segments[1].role == "reporter"
        assert "inflation" in segments[1].text

        assert segments[2].speaker == "CHAIR POWELL"
        assert segments[2].role == "powell"
        assert "great question" in segments[2].text

    def test_segment_speakers_multiline(self):
        """Test segmentation with multi-line utterances."""
        text = """CHAIR POWELL: This is the first line.
This is the second line.
This is the third line.

MR. JONES: Short question."""

        segments = segment_speakers_deterministic(text)

        assert len(segments) == 2
        assert "first line" in segments[0].text
        assert "second line" in segments[0].text
        assert "third line" in segments[0].text

    def test_validate_segments_exact_match(self):
        """Test validation with exact match."""
        original = "Hello world foo bar"
        segments = [
            SpeakerTurn(speaker="A", role="other", text="Hello world"),
            SpeakerTurn(speaker="B", role="other", text="foo bar"),
        ]

        assert validate_segments(segments, original, similarity_threshold=0.9)

    def test_validate_segments_mismatch(self):
        """Test validation with text mismatch."""
        original = "Hello world foo bar"
        segments = [
            SpeakerTurn(speaker="A", role="other", text="Completely different"),
            SpeakerTurn(speaker="B", role="other", text="text content"),
        ]

        assert not validate_segments(segments, original, similarity_threshold=0.9)


class TestValidation:
    """Tests for validation utilities."""

    def test_compute_text_similarity_identical(self):
        """Test similarity of identical texts."""
        text = "The quick brown fox jumps over the lazy dog"
        similarity = compute_text_similarity(text, text)
        assert similarity == 1.0

    def test_compute_text_similarity_different(self):
        """Test similarity of different texts."""
        text1 = "Hello world"
        text2 = "Goodbye universe"
        similarity = compute_text_similarity(text1, text2)
        assert similarity < 0.5

    def test_compute_text_similarity_partial(self):
        """Test similarity of partially matching texts."""
        text1 = "The quick brown fox"
        text2 = "The quick brown dog"
        similarity = compute_text_similarity(text1, text2)
        assert 0.7 < similarity < 0.95
