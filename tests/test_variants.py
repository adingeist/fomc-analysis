"""
Tests for phrase variant generation and caching.
"""

import pytest
import tempfile
from pathlib import Path

from fomc_analysis.variants.generator import (
    compute_cache_key,
    load_variants,
)


class TestVariantCaching:
    """Tests for variant caching mechanism."""

    def test_cache_key_deterministic(self):
        """Test that cache key is deterministic."""
        contract = "Test Contract"
        phrases = ["test", "example"]

        key1 = compute_cache_key(contract, phrases)
        key2 = compute_cache_key(contract, phrases)

        assert key1 == key2

    def test_cache_key_order_independent(self):
        """Test that cache key is independent of phrase order."""
        contract = "Test Contract"
        phrases1 = ["alpha", "beta", "gamma"]
        phrases2 = ["gamma", "alpha", "beta"]

        key1 = compute_cache_key(contract, phrases1)
        key2 = compute_cache_key(contract, phrases2)

        # Keys should be the same (phrases are sorted internally)
        assert key1 == key2

    def test_cache_key_changes_with_input(self):
        """Test that cache key changes when inputs change."""
        contract = "Test Contract"
        phrases1 = ["test", "example"]
        phrases2 = ["different", "phrases"]

        key1 = compute_cache_key(contract, phrases1)
        key2 = compute_cache_key(contract, phrases2)

        assert key1 != key2

    def test_cache_key_changes_with_model(self):
        """Test that cache key changes when model changes."""
        contract = "Test Contract"
        phrases = ["test"]

        key1 = compute_cache_key(contract, phrases, model="gpt-4o-mini")
        key2 = compute_cache_key(contract, phrases, model="gpt-4o")

        assert key1 != key2

    def test_cache_key_changes_with_prompt_version(self):
        """Test that cache key changes when prompt version changes."""
        contract = "Test Contract"
        phrases = ["test"]

        key1 = compute_cache_key(contract, phrases, prompt_version="v1")
        key2 = compute_cache_key(contract, phrases, prompt_version="v2")

        assert key1 != key2


# Note: We don't test the actual OpenAI API calls in unit tests
# Those would require API keys and incur costs
# They should be tested separately in integration tests
