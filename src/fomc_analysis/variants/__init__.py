"""
Phrase variant generation using OpenAI API.

This package provides functionality to generate synonym and paraphrase
variants for contract phrases using OpenAI's language models. Results
are cached to disk to avoid redundant API calls.
"""

from .generator import generate_variants, load_variants, VariantResult

__all__ = ["generate_variants", "load_variants", "VariantResult"]
