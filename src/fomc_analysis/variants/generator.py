"""
OpenAI-powered phrase variant generation with caching.

This module generates phrase variants (synonyms, plurals, etc.) for
contract phrases using OpenAI's API. Results are cached to disk to avoid
redundant API calls.

The cache key is computed as: hash(prompt_version + model + contract + base_phrases)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from openai import OpenAI


# Prompt version - increment when prompt changes
PROMPT_VERSION = "v1"


# Prompt template for variant generation
VARIANT_GENERATION_PROMPT = """You are generating phrase variants for a Kalshi mention contract.

The contract tracks whether specific words or phrases are mentioned in FOMC press conference transcripts.

Contract Resolution Rules:
- **INCLUDED**: Plural and possessive forms (e.g., "immigrant" → "immigrants", "immigrant's")
- **INCLUDED**: Compound words including hyphenated forms (e.g., "AI" → "AI-powered", "AI technology")
- **INCLUDED**: Homonyms (same spelling, different meaning - all count)
- **INCLUDED**: Homographs (same spelling, different pronunciation - all count)
- **EXCLUDED**: Grammatical and tense inflections (e.g., "immigrant" does NOT include "immigration")
- **EXCLUDED**: Synonyms (e.g., "AI" does NOT include "machine learning")
- **EXCLUDED**: Homophones (different spelling, same sound - do NOT count)
- **IMPORTANT**: Only mutate the provided base phrases. Never prepend or append unrelated context from the contract title.
- **IMPORTANT**: Ignore threshold text (e.g., "(5+ times)"). The base phrase already excludes these counts.

Your task:
Given a contract name and base phrases, generate a comprehensive list of variants that would count as mentions according to the rules above.

Contract: {contract_name}
Base phrases: {base_phrases}

Generate variants including:
1. Plural forms
2. Possessive forms
3. Compound forms (hyphenated and space-separated)
4. Case variations (if applicable)

Return ONLY a JSON array of variant strings. Each variant should be lowercase.
Do NOT include synonyms or related terms.
Do NOT include tense inflections (e.g., no verb conjugations).

Example output format:
["variant1", "variant2", "variant3"]
"""


@dataclass
class VariantResult:
    """Result of variant generation for a contract."""
    contract: str
    base_phrases: List[str]
    variants: List[str]
    metadata: dict
    cache_key: str
    generated_at: str


def compute_cache_key(
    contract: str,
    base_phrases: List[str],
    model: str = "gpt-4o-mini",
    prompt_version: str = PROMPT_VERSION,
) -> str:
    """
    Compute cache key for variant generation.

    Parameters
    ----------
    contract : str
        Contract name.
    base_phrases : List[str]
        Base phrases for the contract.
    model : str
        OpenAI model used.
    prompt_version : str
        Version of the prompt template.

    Returns
    -------
    str
        SHA256 hash of the inputs.
    """
    # Deterministic ordering
    sorted_phrases = sorted(base_phrases)

    # Create hash input
    hash_input = json.dumps({
        "prompt_version": prompt_version,
        "model": model,
        "contract": contract,
        "base_phrases": sorted_phrases,
    }, sort_keys=True)

    return hashlib.sha256(hash_input.encode()).hexdigest()


def generate_variants_with_openai(
    contract: str,
    base_phrases: List[str],
    openai_client: OpenAI,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> Optional[List[str]]:
    """
    Generate phrase variants using OpenAI API.

    Parameters
    ----------
    contract : str
        Contract name.
    base_phrases : List[str]
        Base phrases to generate variants from.
    openai_client : OpenAI
        Configured OpenAI client.
    model : str, default="gpt-4o-mini"
        OpenAI model to use.
    max_retries : int, default=3
        Number of retry attempts on failure.

    Returns
    -------
    Optional[List[str]]
        List of variant phrases, or None if generation failed.
    """
    prompt = VARIANT_GENERATION_PROMPT.format(
        contract_name=contract,
        base_phrases=", ".join(f'"{p}"' for p in base_phrases),
    )

    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise language assistant that generates phrase variants. Output only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # Deterministic
            )

            content = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            import re
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"^```\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

            # Parse JSON
            variants = json.loads(content)

            if not isinstance(variants, list):
                raise ValueError("Response is not a list")

            # Normalize to lowercase and dedupe
            normalized = sorted(set(v.lower().strip() for v in variants if v))

            return normalized

        except Exception as e:
            if attempt == max_retries - 1:
                return None

            # Exponential backoff
            time.sleep(2 ** attempt)

    return None


def generate_variants(
    contract: str,
    base_phrases: List[str],
    openai_client: OpenAI,
    cache_dir: Path = Path("data/variants"),
    model: str = "gpt-4o-mini",
    force_regenerate: bool = False,
    extra_metadata: Optional[dict] = None,
) -> VariantResult:
    """
    Generate phrase variants with caching.

    This function checks the cache first. If a cached result exists and
    force_regenerate=False, it returns the cached result. Otherwise, it
    calls the OpenAI API to generate new variants.

    Parameters
    ----------
    contract : str
        Contract name.
    base_phrases : List[str]
        Base phrases for the contract.
    openai_client : OpenAI
        Configured OpenAI client.
    cache_dir : Path, default="data/variants"
        Directory to store cached variants.
    model : str, default="gpt-4o-mini"
        OpenAI model to use.
    force_regenerate : bool, default=False
        If True, ignore cache and regenerate variants.

    Returns
    -------
    VariantResult
        Variant generation result with metadata.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Compute cache key
    cache_key = compute_cache_key(contract, base_phrases, model)

    # Generate safe filename (use contract name + cache key prefix)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in contract)
    cache_file = cache_dir / f"{safe_name}_{cache_key[:12]}.json"

    # Check cache
    if cache_file.exists() and not force_regenerate:
        data = json.loads(cache_file.read_text())
        return VariantResult(**data)

    # Generate variants
    variants = generate_variants_with_openai(
        contract, base_phrases, openai_client, model
    )

    if variants is None:
        # Fall back to just base phrases if generation failed
        variants = sorted(set(p.lower() for p in base_phrases))

    # Create result
    metadata = {
        "model": model,
        "prompt_version": PROMPT_VERSION,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    result = VariantResult(
        contract=contract,
        base_phrases=sorted(set(p.lower() for p in base_phrases)),
        variants=variants,
        metadata=metadata,
        cache_key=cache_key,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    # Save to cache
    cache_file.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False))

    return result


def load_variants(cache_file: Path) -> VariantResult:
    """
    Load cached variants from file.

    Parameters
    ----------
    cache_file : Path
        Path to cached variant JSON file.

    Returns
    -------
    VariantResult
        Loaded variant result.
    """
    data = json.loads(cache_file.read_text())
    return VariantResult(**data)


def load_all_variants(cache_dir: Path = Path("data/variants")) -> dict[str, VariantResult]:
    """
    Load all cached variants from directory.

    Parameters
    ----------
    cache_dir : Path
        Directory containing cached variant files.

    Returns
    -------
    dict[str, VariantResult]
        Dictionary mapping contract names to their variants.
    """
    cache_dir = Path(cache_dir)
    results = {}

    if not cache_dir.exists():
        return results

    for file_path in cache_dir.glob("*.json"):
        variant_result = load_variants(file_path)
        results[variant_result.contract] = variant_result

    return results
