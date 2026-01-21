"""
Tests for feature extraction and phrase matching.
"""

import pytest
from pathlib import Path

from fomc_analysis.featurizer import (
    match_phrase_in_text,
    count_phrase_mentions,
    extract_features_from_segments,
    FeatureConfig,
)
from fomc_analysis.parsing.speaker_segmenter import SpeakerTurn


class TestPhraseMatching:
    """Tests for phrase matching functionality."""

    def test_match_phrase_exact(self):
        """Test exact phrase matching."""
        text = "The Federal Reserve announced new policies today."
        matches = match_phrase_in_text("Federal Reserve", text, word_boundaries=True)
        assert len(matches) == 1

    def test_match_phrase_case_insensitive(self):
        """Test case-insensitive matching."""
        text = "The FEDERAL RESERVE announced new policies."
        matches = match_phrase_in_text(
            "federal reserve",
            text,
            case_sensitive=False,
            word_boundaries=True,
        )
        assert len(matches) == 1

    def test_match_phrase_word_boundaries(self):
        """Test that word boundaries are respected."""
        text = "We discussed the median value and medians in general."

        # Should match "median" as a word
        matches = match_phrase_in_text("median", text, word_boundaries=True)
        # "median" appears twice as a standalone word
        assert len(matches) == 2

    def test_match_phrase_no_word_boundaries(self):
        """Test matching without word boundaries."""
        text = "The firetruck arrived at the fire station."

        # Should NOT match "fire" in "firetruck" with word boundaries
        matches = match_phrase_in_text("fire", text, word_boundaries=True)
        assert len(matches) == 1  # Only "fire station"

        # Should match "fire" in both with no boundaries
        matches = match_phrase_in_text("fire", text, word_boundaries=False)
        assert len(matches) == 2

    def test_count_phrase_mentions_multiple_variants(self):
        """Test counting with multiple phrase variants."""
        text = "We discussed AI and artificial intelligence technologies."
        phrases = ["ai", "artificial intelligence"]

        count = count_phrase_mentions(
            text,
            phrases,
            case_sensitive=False,
            word_boundaries=True,
        )
        assert count == 2  # Both variants appear once


class TestFeatureExtraction:
    """Tests for feature extraction from segments."""

    def test_extract_features_powell_only(self):
        """Test feature extraction with powell_only mode."""
        segments = [
            SpeakerTurn(
                speaker="CHAIR POWELL",
                role="powell",
                text="We discussed the median projection today.",
            ),
            SpeakerTurn(
                speaker="MR. REPORTER",
                role="reporter",
                text="What about expectations for inflation?",
            ),
        ]

        contracts = {
            "Median": ["median", "medians"],
            "Expectation": ["expectation", "expectations"],
        }

        config = FeatureConfig(speaker_mode="powell_only")

        features = extract_features_from_segments(segments, contracts, config)

        # "median" appears in Powell's text
        assert features["Median_mentioned"] == 1
        assert features["Median_count"] >= 1

        # "expectations" appears only in reporter's text
        assert features["Expectation_mentioned"] == 0

    def test_extract_features_full_transcript(self):
        """Test feature extraction with full_transcript mode."""
        segments = [
            SpeakerTurn(
                speaker="CHAIR POWELL",
                role="powell",
                text="We discussed the median projection today.",
            ),
            SpeakerTurn(
                speaker="MR. REPORTER",
                role="reporter",
                text="What about expectations for inflation?",
            ),
        ]

        contracts = {
            "Median": ["median", "medians"],
            "Expectation": ["expectation", "expectations"],
        }

        config = FeatureConfig(speaker_mode="full_transcript")

        features = extract_features_from_segments(segments, contracts, config)

        # Both should be detected in full transcript
        assert features["Median_mentioned"] == 1
        assert features["Expectation_mentioned"] == 1

    def test_extract_features_contract_name_sanitization(self):
        """Test that contract names are sanitized for feature keys."""
        segments = [
            SpeakerTurn(
                speaker="CHAIR POWELL",
                role="powell",
                text="We discussed AI and artificial intelligence.",
            ),
        ]

        contracts = {
            "AI / Artificial Intelligence": ["ai", "artificial intelligence"],
        }

        config = FeatureConfig()

        features = extract_features_from_segments(segments, contracts, config)

        # Contract name should be sanitized (/ and spaces replaced)
        assert "AI___Artificial_Intelligence_mentioned" in features
        assert features["AI___Artificial_Intelligence_mentioned"] == 1
