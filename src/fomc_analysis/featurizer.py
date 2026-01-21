"""
Featurization module for FOMC transcript analysis.

This module extracts features from parsed transcripts with support for
different resolution modes:
- powell_only vs full_transcript (speaker filtering)
- strict_literal vs variants (phrase matching)

Features are computed at the transcript level and can be used for
modeling mention probabilities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd

from .parsing.speaker_segmenter import SpeakerTurn, load_segments_jsonl
from .variants.generator import VariantResult, load_all_variants


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    speaker_mode: str = "powell_only"  # "powell_only" or "full_transcript"
    phrase_mode: str = "strict"  # "strict" or "variants"
    case_sensitive: bool = False
    word_boundaries: bool = True  # Require word boundaries for matching


def match_phrase_in_text(
    phrase: str,
    text: str,
    case_sensitive: bool = False,
    word_boundaries: bool = True,
) -> List[Tuple[int, int]]:
    """
    Find all occurrences of a phrase in text.

    Parameters
    ----------
    phrase : str
        Phrase to search for.
    text : str
        Text to search in.
    case_sensitive : bool, default=False
        Whether to perform case-sensitive matching.
    word_boundaries : bool, default=True
        Whether to require word boundaries around matches.

    Returns
    -------
    List[Tuple[int, int]]
        List of (start, end) positions of matches.
    """
    if not phrase:
        return []

    # Prepare text and phrase
    search_text = text if case_sensitive else text.lower()
    search_phrase = phrase if case_sensitive else phrase.lower()

    # Build regex pattern
    if word_boundaries:
        # Escape special regex characters
        escaped = re.escape(search_phrase)
        pattern = r"\b" + escaped + r"\b"
    else:
        pattern = re.escape(search_phrase)

    # Find all matches
    matches = []
    for match in re.finditer(pattern, search_text):
        matches.append((match.start(), match.end()))

    return matches


def count_phrase_mentions(
    text: str,
    phrases: List[str],
    case_sensitive: bool = False,
    word_boundaries: bool = True,
) -> int:
    """
    Count how many times any of the phrases appear in text.

    Parameters
    ----------
    text : str
        Text to search in.
    phrases : List[str]
        List of phrase variants to search for.
    case_sensitive : bool, default=False
        Whether to perform case-sensitive matching.
    word_boundaries : bool, default=True
        Whether to require word boundaries.

    Returns
    -------
    int
        Total count of all phrase occurrences.
    """
    total_count = 0
    for phrase in phrases:
        matches = match_phrase_in_text(phrase, text, case_sensitive, word_boundaries)
        total_count += len(matches)
    return total_count


def extract_features_from_segments(
    segments: List[SpeakerTurn],
    contracts: Dict[str, List[str]],
    config: FeatureConfig,
) -> Dict[str, any]:
    """
    Extract features from speaker segments for all contracts.

    Parameters
    ----------
    segments : List[SpeakerTurn]
        Speaker turns from a transcript.
    contracts : Dict[str, List[str]]
        Mapping from contract names to phrase lists.
    config : FeatureConfig
        Feature extraction configuration.

    Returns
    -------
    Dict[str, any]
        Feature dictionary with keys like:
        - {contract}_mentioned: bool (1 if mentioned, 0 otherwise)
        - {contract}_count: int (number of mentions)
        - {contract}_snippets: List[str] (optional debug snippets)
    """
    # Filter segments by speaker mode
    if config.speaker_mode == "powell_only":
        relevant_segments = [s for s in segments if s.role == "powell"]
    else:  # full_transcript
        relevant_segments = segments

    # Concatenate all relevant text
    full_text = " ".join(seg.text for seg in relevant_segments)

    # Extract features for each contract
    features = {}

    for contract, phrases in contracts.items():
        # Count mentions
        count = count_phrase_mentions(
            full_text,
            phrases,
            case_sensitive=config.case_sensitive,
            word_boundaries=config.word_boundaries,
        )

        # Binary mention indicator
        mentioned = 1 if count > 0 else 0

        # Store features
        safe_name = contract.replace("/", "_").replace(" ", "_")
        features[f"{safe_name}_mentioned"] = mentioned
        features[f"{safe_name}_count"] = count

    return features


def build_feature_matrix(
    segments_dir: Path,
    contracts: Dict[str, List[str]],
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Build a feature matrix from all transcripts.

    Parameters
    ----------
    segments_dir : Path
        Directory containing segment JSONL files.
    contracts : Dict[str, List[str]]
        Mapping from contract names to phrase lists.
    config : FeatureConfig
        Feature extraction configuration.

    Returns
    -------
    pd.DataFrame
        Feature matrix with rows = transcript dates, columns = features.
    """
    segments_dir = Path(segments_dir)
    rows = []
    dates = []

    # Process each segment file
    for segment_file in sorted(segments_dir.glob("*.jsonl")):
        # Extract date from filename (e.g., "20250115.jsonl")
        date_str = segment_file.stem

        # Load segments
        segments = load_segments_jsonl(segment_file)

        # Extract features
        features = extract_features_from_segments(segments, contracts, config)
        features["date"] = date_str

        rows.append(features)
        dates.append(date_str)

    # Create DataFrame
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df = df.set_index("date")

    return df


def load_contract_phrases(
    mapping_file: Path,
    variants_dir: Optional[Path] = None,
    use_variants: bool = False,
) -> Dict[str, List[str]]:
    """
    Load contract phrase mappings.

    Parameters
    ----------
    mapping_file : Path
        Path to contract mapping YAML file.
    variants_dir : Optional[Path]
        Directory containing variant cache files.
    use_variants : bool, default=False
        If True, use AI-generated variants instead of base phrases.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from contract names to phrase lists.
    """
    import yaml

    # Load base mapping
    data = yaml.safe_load(mapping_file.read_text())

    contracts = {}

    if use_variants and variants_dir is not None:
        # Load variants from cache
        variant_results = load_all_variants(variants_dir)

        for contract, entry in data.items():
            if contract in variant_results:
                # Use cached variants
                contracts[contract] = variant_results[contract].variants
            else:
                # Fall back to base synonyms
                contracts[contract] = [s.lower() for s in entry.get("synonyms", [])]
    else:
        # Use base synonyms only
        for contract, entry in data.items():
            contracts[contract] = [s.lower() for s in entry.get("synonyms", [])]

    return contracts
