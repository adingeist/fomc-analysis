"""
Tests for Kalshi contract analyzer module.
"""

import pytest
from pathlib import Path

from fomc_analysis.kalshi_contract_analyzer import (
    parse_market_title,
    ContractWord,
)


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
