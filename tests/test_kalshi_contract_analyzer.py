"""
Tests for Kalshi contract analyzer module.
"""

import pytest
from pathlib import Path

from fomc_analysis.kalshi_contract_analyzer import (
    parse_market_title,
    parse_market_metadata,
    contract_words_to_mapping,
    scan_transcript_for_words,
    ContractWord,
    fetch_mention_contracts,
)
from fomc_analysis.parsing.speaker_segmenter import SpeakerTurn, save_segments_jsonl


class TestMarketTitleParsing:
    """Test market title parsing logic."""

    def test_parse_simple_word(self):
        """Test parsing simple single-word titles."""
        assert parse_market_title("President") == "President"
        assert parse_market_title("Layoff") == "Layoff"
        assert parse_market_title("Median") == "Median"

    def test_parse_with_mention_suffix(self):
        """Test parsing titles with 'mention' suffix."""
        assert parse_market_title("Layoff mention") == "Layoff"
        assert parse_market_title("President mention") == "President"
        assert parse_market_title("Median mention") == "Median"

    def test_parse_multi_word_phrases(self):
        """Test parsing multi-word phrases."""
        assert parse_market_title("Good Afternoon") == "Good Afternoon"
        assert parse_market_title("Balance of Risk") == "Balance of Risk"
        assert parse_market_title("Tariff Inflation") == "Tariff Inflation"

    def test_parse_multi_word_with_mention(self):
        """Test parsing multi-word phrases with 'mention' suffix."""
        assert parse_market_title("Good Afternoon mention") == "Good Afternoon"
        assert parse_market_title("Balance of Risk mention") == "Balance of Risk"

    def test_parse_slash_phrases(self):
        """Test parsing phrases with slashes."""
        result = parse_market_title("AI / Artificial Intelligence")
        assert result == "AI / Artificial Intelligence"

    def test_parse_with_extra_whitespace(self):
        """Test parsing handles extra whitespace."""
        assert parse_market_title("  President  ") == "President"
        assert parse_market_title("  Layoff mention  ") == "Layoff"

    def test_parse_question_phrase(self):
        """Question-form markets should extract the inner phrase."""
        word, threshold = parse_market_metadata(
            "Will Powell say Good Afternoon at his Dec 2025 press conference?"
        )
        assert word == "Good Afternoon"
        assert threshold is None

    def test_parse_threshold_parenthetical(self):
        """Parenthetical thresholds should be removed from base phrase."""
        word, threshold = parse_market_metadata("Uncertainty (5+ Times)")
        assert word == "Uncertainty"
        assert threshold == 5

    def test_parse_market_metadata_threshold(self):
        """Thresholds should be detected from titles."""
        word, threshold = parse_market_metadata("Inflation 40+ times mention")
        assert word == "Inflation"
        assert threshold == 40

        word, threshold = parse_market_metadata("Layoff mention")
        assert word == "Layoff"
        assert threshold is None


class TestContractWord:
    """Test ContractWord dataclass."""

    def test_contract_word_creation(self):
        """Test creating ContractWord instance."""
        word = ContractWord(
            word="President",
            market_ticker="KXFEDMENTION-26JAN-PRS",
            market_title="President mention",
            variants=["president", "presidents", "president's"],
        )

        assert word.word == "President"
        assert word.market_ticker == "KXFEDMENTION-26JAN-PRS"
        assert len(word.variants) == 3

    def test_contract_word_to_dict(self):
        """Test converting ContractWord to dictionary."""
        word = ContractWord(
            word="Layoff",
            market_ticker="KXFEDMENTION-26JAN-LAY",
            market_title="Layoff mention",
            variants=["layoff", "layoffs", "layoff's"],
        )

        word_dict = word.to_dict()
        assert isinstance(word_dict, dict)
        assert word_dict["word"] == "Layoff"
        assert word_dict["variants"] == ["layoff", "layoffs", "layoff's"]


class TestContractMapping:
    """Tests for converting ContractWord objects to mapping entries."""

    def test_contract_words_to_mapping(self):
        contracts = [
            ContractWord(
                word="Layoff",
                market_ticker="KXFEDMENTION-26JAN-LAY",
                market_title="Layoff mention",
                variants=["Layoff", "Layoffs"],
            ),
            ContractWord(
                word="Inflation",
                market_ticker="KXFEDMENTION-26JAN-INF",
                market_title="Inflation 40+",
                variants=["Inflation"],
                threshold=40,
            ),
        ]

        mapping = contract_words_to_mapping(contracts)

        assert "Layoff" in mapping
        lay_entry = mapping["Layoff"]
        assert lay_entry["threshold"] == 1
        assert lay_entry["scope"] == "powell_only"
        assert set(lay_entry["synonyms"]) == {"layoff", "layoffs"}

        inf_entry = mapping["Inflation (40+)"]
        assert inf_entry["threshold"] == 40
        assert inf_entry["match_mode"] == "strict_literal"

    def test_mapping_includes_threshold_in_name(self):
        contracts = [
            ContractWord(
                word="Uncertainty",
                market_ticker="T1",
                market_title="Uncertainty mention",
                variants=["uncertainty"],
                threshold=1,
            ),
            ContractWord(
                word="Uncertainty",
                market_ticker="T2",
                market_title="Uncertainty (5+ times)",
                variants=["uncertainty"],
                threshold=5,
            ),
        ]
        mapping = contract_words_to_mapping(contracts)
        assert "Uncertainty" in mapping
        assert "Uncertainty (5+)" in mapping  # expecting plus sign
        assert mapping["Uncertainty"]["threshold"] == 1
        assert mapping["Uncertainty (5+)"]["threshold"] == 5


class TestScanTranscript:
    def test_scan_handles_dataclass_segments(self, tmp_path):
        segments = [
            SpeakerTurn(speaker="CHAIR POWELL", role="powell", text="Layoff trends are easing."),
            SpeakerTurn(speaker="Reporter", role="reporter", text="Layoff, layoff everywhere."),
        ]
        seg_file = tmp_path / "20250129.jsonl"
        save_segments_jsonl(segments, seg_file)

        contracts = [
            ContractWord(
                word="Layoff",
                market_ticker="KXFEDMENTION-FAKE",
                market_title="Layoff mention",
                variants=["layoff"],
            )
        ]

        powell_counts = scan_transcript_for_words(seg_file, contracts, scope="powell_only")
        assert powell_counts["Layoff"] == 1

        full_counts = scan_transcript_for_words(seg_file, contracts, scope="full_transcript")
        assert full_counts["Layoff"] == 3


class DummyKalshiClient:
    def __init__(self, markets):
        self._markets = markets

    def get_event(self, *args, **kwargs):
        return {"event": {"markets": self._markets}}

    def get_markets(self, *args, **kwargs):
        return self._markets


def test_fetch_contracts_merges_thresholds():
    markets = [
        {"title": "Uncertainty", "ticker": "T1", "event_ticker": "ev1", "close_time": "2025-01-01T00:00:00Z"},
        {"title": "Uncertainty (5+ Times)", "ticker": "T2", "event_ticker": "ev2", "close_time": "2026-01-01T00:00:00Z"},
    ]
    client = DummyKalshiClient(markets)
    results = fetch_mention_contracts(client)
    assert len(results) == 2
    unc_binary = next(r for r in results if r.threshold == 1)
    unc_thresh = next(r for r in results if r.threshold == 5)
    assert unc_binary.market_ticker == "T1"
    assert unc_thresh.market_ticker == "T2"
    assert any(
        entry["ticker"] == "T2" and entry["threshold"] == 5 and entry["close_date"] == "2026-01-01"
        for entry in unc_thresh.markets
    )
