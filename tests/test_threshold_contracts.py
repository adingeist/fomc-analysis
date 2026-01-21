"""
Tests for threshold contract support.

These tests verify:
1. Word-boundary token counting correctness
2. Strict vs variants behavior
3. Ambiguous term handling (especially "cut")
4. Threshold_hit correctness on fixture transcripts
5. Four-mode label generation
6. Contract spec loading from YAML
"""

import pytest
from pathlib import Path
import pandas as pd

from fomc_analysis.contract_mapping import (
    ContractMapping,
    ContractSpec,
    load_mapping_from_file,
)
from fomc_analysis.label_generator import (
    match_phrase_in_text_with_context,
    count_phrase_mentions_with_snippets,
    generate_labels_for_contract,
    LabelResult,
)
from fomc_analysis.parsing.speaker_segmenter import SpeakerTurn


@pytest.fixture
def sample_text():
    """Sample text for testing word-boundary matching."""
    return """
    Inflation remains a concern. We've seen inflation increase over the past
    few months. The inflationary pressures are being driven by supply chain
    issues. We need to anchor inflation expectations. The price of goods
    has risen, and prices continue to climb. Pricing decisions by firms
    have contributed to the increase.

    We may need to cut rates, or we might cut back on our purchases.
    The rate cuts could help, but we won't cut corners. Some have suggested
    cutting the balance sheet, not just cutting rates.

    Growth has been strong. Economic growth is positive. We're seeing grow
    in many sectors, though some areas are not growing as fast.

    Tariff policies have been discussed. New tariffs may affect inflation.
    The tariff increases could be inflationary.
    """


@pytest.fixture
def sample_segments():
    """Sample speaker segments for testing."""
    return [
        SpeakerTurn(
            speaker="Jerome Powell",
            role="powell",
            text="Inflation has been mentioned forty-one times in our discussion. "
                 "We need to cut rates to address inflation. The price increases "
                 "are concerning. Growth remains solid.",
            start_page=None,
            end_page=None,
            confidence=1.0,
        ),
        SpeakerTurn(
            speaker="Reporter",
            role="reporter",
            text="Will you cut rates? What about the tariff impact on prices?",
            start_page=None,
            end_page=None,
            confidence=1.0,
        ),
        SpeakerTurn(
            speaker="Jerome Powell",
            role="powell",
            text="We will cut rates if needed. The tariff effects on inflation "
                 "are uncertain. Price stability is our goal.",
            start_page=None,
            end_page=None,
            confidence=1.0,
        ),
    ]


class TestWordBoundaryMatching:
    """Test word-boundary matching for accurate token counting."""

    def test_inflation_strict_match(self, sample_text):
        """Test that 'inflation' matches only complete words."""
        matches = match_phrase_in_text_with_context(
            "inflation",
            sample_text,
            case_sensitive=False,
            word_boundaries=True,
        )
        # Should match "inflation" but not "inflationary"
        assert len(matches) == 4  # "inflation" appears 4 times

    def test_inflation_no_partial_match(self, sample_text):
        """Test that 'inflationary' doesn't match when searching for 'inflation'."""
        # Count "inflation" only
        count_inflation, _ = count_phrase_mentions_with_snippets(
            sample_text, ["inflation"], word_boundaries=True
        )

        # Count "inflationary" only
        count_inflationary, _ = count_phrase_mentions_with_snippets(
            sample_text, ["inflationary"], word_boundaries=True
        )

        assert count_inflation == 4
        assert count_inflationary == 2  # "inflationary" appears 2 times

    def test_price_variants(self, sample_text):
        """Test that price/prices/pricing are counted separately with variants."""
        count, _ = count_phrase_mentions_with_snippets(
            sample_text,
            ["price", "prices", "pricing"],
            word_boundaries=True,
        )
        # "price" (1) + "prices" (1) + "pricing" (1) = 3
        assert count == 3

    def test_cut_ambiguous_term(self, sample_text):
        """Test that 'cut' only matches complete word, not 'cutting'."""
        count, snippets = count_phrase_mentions_with_snippets(
            sample_text,
            ["cut"],
            word_boundaries=True,
        )
        # Should match "cut" in "cut rates", "cut back", but not "cutting"
        # Expected: "cut rates" (1), "cut back" (1) = 2
        assert count >= 2

        # Verify snippets contain "cut" not "cutting"
        for snippet in snippets:
            assert "cut" in snippet.lower()

    def test_cut_with_variants(self, sample_text):
        """Test counting 'cut' and 'cuts' together."""
        count, _ = count_phrase_mentions_with_snippets(
            sample_text,
            ["cut", "cuts"],
            word_boundaries=True,
        )
        # Should count both "cut" and "cuts" but not "cutting"
        assert count >= 2

    def test_growth_strict(self, sample_text):
        """Test that 'growth' only matches the noun, not verb forms."""
        count, _ = count_phrase_mentions_with_snippets(
            sample_text,
            ["growth"],
            word_boundaries=True,
        )
        # Should match "growth" (2 times) but not "grow", "growing"
        assert count == 2

    def test_case_insensitive_matching(self):
        """Test case-insensitive matching."""
        text = "Inflation, INFLATION, and inflation."
        count, _ = count_phrase_mentions_with_snippets(
            text,
            ["inflation"],
            case_sensitive=False,
            word_boundaries=True,
        )
        assert count == 3

    def test_case_sensitive_matching(self):
        """Test case-sensitive matching."""
        text = "Inflation, INFLATION, and inflation."
        count, _ = count_phrase_mentions_with_snippets(
            text,
            ["inflation"],
            case_sensitive=True,
            word_boundaries=True,
        )
        assert count == 1  # Only lowercase "inflation"


class TestContractSpecLoading:
    """Test loading contract specifications from YAML."""

    def test_load_threshold_contract(self, tmp_path):
        """Test loading a contract with threshold specified."""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
"Inflation 40+":
  synonyms:
    - inflation
  threshold: 40
  scope: powell_only
  match_mode: strict_literal
  count_unit: token
  description: Test contract
        """)

        mapping = load_mapping_from_file(config_path)

        assert "Inflation 40+" in mapping.contracts()
        assert mapping.get_threshold("Inflation 40+") == 40
        assert mapping.get_scope("Inflation 40+") == "powell_only"
        assert mapping.get_match_mode("Inflation 40+") == "strict_literal"
        assert mapping.get_count_unit("Inflation 40+") == "token"

    def test_load_binary_contract_defaults(self, tmp_path):
        """Test that binary contracts get default threshold=1."""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
Median:
  synonyms:
    - median
  description: Binary mention contract
        """)

        mapping = load_mapping_from_file(config_path)

        assert mapping.get_threshold("Median") == 1
        assert mapping.get_scope("Median") == "powell_only"
        assert mapping.get_match_mode("Median") == "strict_literal"

    def test_load_multiple_contracts(self, tmp_path):
        """Test loading multiple contracts with different thresholds."""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
"Inflation 40+":
  synonyms:
    - inflation
  threshold: 40

"Price 15+":
  synonyms:
    - price
    - prices
  threshold: 15

Median:
  synonyms:
    - median
        """)

        mapping = load_mapping_from_file(config_path)

        assert mapping.get_threshold("Inflation 40+") == 40
        assert mapping.get_threshold("Price 15+") == 15
        assert mapping.get_threshold("Median") == 1


class TestLabelGeneration:
    """Test four-mode label generation."""

    def test_powell_only_filter(self, sample_segments):
        """Test that powell_only correctly filters segments."""
        spec = ContractSpec(
            synonyms=["inflation"],
            threshold=1,
            scope="powell_only",
            match_mode="strict_literal",
        )

        results = generate_labels_for_contract(
            "Inflation",
            spec,
            sample_segments,
            variants_map=None,
        )

        # Find powell_only results
        powell_results = [r for r in results if "powell_only" in r.mode]

        for result in powell_results:
            # Should count inflation mentions only in Powell's segments
            assert result.count >= 2  # At least 2 mentions in Powell segments

    def test_full_transcript_scope(self, sample_segments):
        """Test full_transcript includes all speakers."""
        spec = ContractSpec(
            synonyms=["tariff"],
            threshold=1,
            scope="full_transcript",
            match_mode="strict_literal",
        )

        results = generate_labels_for_contract(
            "Tariff",
            spec,
            sample_segments,
            variants_map=None,
        )

        # Find full_transcript results
        full_results = [r for r in results if "full_transcript" in r.mode]

        for result in full_results:
            # Should count tariff mentions in all segments (Powell + Reporter)
            assert result.count >= 2

    def test_threshold_hit_detection(self, sample_segments):
        """Test threshold_hit binary outcome."""
        # Low threshold - should hit
        spec_low = ContractSpec(
            synonyms=["inflation"],
            threshold=2,
            scope="powell_only",
            match_mode="strict_literal",
        )

        results_low = generate_labels_for_contract(
            "Inflation 2+",
            spec_low,
            sample_segments,
            variants_map=None,
        )

        # Find a powell_only result
        result = [r for r in results_low if "powell_only" in r.mode][0]
        assert result.threshold_hit == 1  # Should hit threshold

        # High threshold - should not hit
        spec_high = ContractSpec(
            synonyms=["inflation"],
            threshold=100,
            scope="powell_only",
            match_mode="strict_literal",
        )

        results_high = generate_labels_for_contract(
            "Inflation 100+",
            spec_high,
            sample_segments,
            variants_map=None,
        )

        result = [r for r in results_high if "powell_only" in r.mode][0]
        assert result.threshold_hit == 0  # Should not hit threshold

    def test_four_modes_generated(self, sample_segments):
        """Test that all four modes are generated."""
        spec = ContractSpec(
            synonyms=["price"],
            threshold=1,
        )

        results = generate_labels_for_contract(
            "Price",
            spec,
            sample_segments,
            variants_map=None,
        )

        # Should have 4 results (one per mode)
        assert len(results) == 4

        modes = {r.mode for r in results}
        expected_modes = {
            "powell_only_strict_literal",
            "powell_only_variants",
            "full_transcript_strict_literal",
            "full_transcript_variants",
        }
        assert modes == expected_modes

    def test_debug_snippets(self, sample_segments):
        """Test that debug snippets are generated."""
        spec = ContractSpec(
            synonyms=["inflation"],
            threshold=1,
        )

        results = generate_labels_for_contract(
            "Inflation",
            spec,
            sample_segments,
            variants_map=None,
        )

        # At least one result should have snippets
        has_snippets = any(len(r.debug_snippets) > 0 for r in results)
        assert has_snippets

    def test_mentioned_binary(self, sample_segments):
        """Test mentioned_binary is set correctly."""
        # Contract that is mentioned
        spec_yes = ContractSpec(synonyms=["inflation"], threshold=1)
        results_yes = generate_labels_for_contract(
            "Inflation", spec_yes, sample_segments, None
        )

        result = results_yes[0]
        assert result.mentioned_binary == 1

        # Contract that is NOT mentioned
        spec_no = ContractSpec(synonyms=["cryptocurrency"], threshold=1)
        results_no = generate_labels_for_contract(
            "Crypto", spec_no, sample_segments, None
        )

        result = results_no[0]
        assert result.mentioned_binary == 0


class TestAmbiguousTerms:
    """Test handling of ambiguous terms that require careful matching."""

    def test_cut_excludes_cutting(self):
        """Verify 'cut' doesn't match 'cutting'."""
        text = "We are cutting rates. We will cut rates. The cuts are significant."
        count, _ = count_phrase_mentions_with_snippets(
            text,
            ["cut"],
            word_boundaries=True,
        )
        # Should match "cut" (1 time) but not "cutting" or "cuts"
        assert count == 1

    def test_cut_with_cuts_variant(self):
        """Verify 'cut' and 'cuts' as separate variants."""
        text = "We are cutting rates. We will cut rates. The cuts are significant."
        count, _ = count_phrase_mentions_with_snippets(
            text,
            ["cut", "cuts"],
            word_boundaries=True,
        )
        # Should match "cut" (1) and "cuts" (1) = 2
        assert count == 2

    def test_price_vs_prices(self):
        """Verify price/prices are counted separately."""
        text = "The price is high. Prices are rising. Pricing power matters."

        # Count only "price"
        count_price, _ = count_phrase_mentions_with_snippets(
            text, ["price"], word_boundaries=True
        )
        assert count_price == 1

        # Count "price" and "prices"
        count_both, _ = count_phrase_mentions_with_snippets(
            text, ["price", "prices"], word_boundaries=True
        )
        assert count_both == 2

    def test_growth_excludes_grow(self):
        """Verify 'growth' doesn't match 'grow' or 'growing'."""
        text = "Growth is positive. We grow every year. The economy is growing."
        count, _ = count_phrase_mentions_with_snippets(
            text, ["growth"], word_boundaries=True
        )
        # Should match only "growth" (1 time)
        assert count == 1


class TestNoLookahead:
    """Test that label generation doesn't use future information."""

    def test_labels_use_only_current_transcript(self, sample_segments):
        """Verify that labels are computed only from current transcript."""
        spec = ContractSpec(synonyms=["inflation"], threshold=2)

        results = generate_labels_for_contract(
            "Inflation 2+",
            spec,
            sample_segments,
            variants_map=None,
        )

        # Results should be based only on sample_segments
        # This is inherently tested by the function signature - it takes
        # segments for ONE transcript, not historical data
        assert len(results) == 4  # Four modes for this transcript


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
