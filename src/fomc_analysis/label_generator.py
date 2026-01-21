"""
Label Generation Module
========================

This module generates resolution-aligned labels for threshold contracts
across all four modes:
1. powell_only + strict_literal
2. powell_only + variants
3. full_transcript + strict_literal
4. full_transcript + variants

For each mode, it computes:
- count: Total mentions
- mentioned_binary: 1 if count >= 1, else 0
- threshold_hit: 1 if count >= threshold, else 0
- debug_snippets: Context snippets showing matches
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd

from .parsing.speaker_segmenter import SpeakerTurn, load_segments_jsonl
from .contract_mapping import ContractMapping, ContractSpec
from .variants.generator import load_all_variants


@dataclass
class LabelResult:
    """Result of label generation for a single contract and mode."""

    contract: str
    mode: str  # e.g., "powell_only_strict_literal"
    count: int
    mentioned_binary: int  # 1 if count >= 1, else 0
    threshold_hit: int  # 1 if count >= threshold, else 0
    threshold: int
    debug_snippets: List[str]  # Context snippets showing matches


def match_phrase_in_text_with_context(
    phrase: str,
    text: str,
    case_sensitive: bool = False,
    word_boundaries: bool = True,
    context_chars: int = 50,
) -> List[Tuple[int, int, str]]:
    """
    Find all occurrences of a phrase in text with surrounding context.

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
    context_chars : int, default=50
        Number of characters to include before and after match.

    Returns
    -------
    List[Tuple[int, int, str]]
        List of (start, end, context_snippet) tuples.
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

    # Find all matches with context
    matches = []
    for match in re.finditer(pattern, search_text):
        start, end = match.start(), match.end()

        # Extract context
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)
        context = text[context_start:context_end]

        # Clean context
        context = context.replace("\n", " ").strip()

        matches.append((start, end, context))

    return matches


def count_phrase_mentions_with_snippets(
    text: str,
    phrases: List[str],
    case_sensitive: bool = False,
    word_boundaries: bool = True,
    max_snippets: int = 5,
) -> Tuple[int, List[str]]:
    """
    Count phrase mentions and return debug snippets.

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
    max_snippets : int, default=5
        Maximum number of debug snippets to return.

    Returns
    -------
    Tuple[int, List[str]]
        (total_count, debug_snippets)
    """
    total_count = 0
    all_snippets = []

    for phrase in phrases:
        matches = match_phrase_in_text_with_context(
            phrase, text, case_sensitive, word_boundaries
        )
        total_count += len(matches)

        # Collect snippets
        for _, _, context in matches[:max_snippets]:
            all_snippets.append(f"'{phrase}': ...{context}...")

    return total_count, all_snippets[:max_snippets]


def generate_labels_for_contract(
    contract: str,
    spec: ContractSpec,
    segments: List[SpeakerTurn],
    variants_map: Optional[Dict[str, List[str]]] = None,
) -> List[LabelResult]:
    """
    Generate labels for a single contract across all four modes.

    Parameters
    ----------
    contract : str
        Contract name.
    spec : ContractSpec
        Contract specification with threshold, scope, match_mode.
    segments : List[SpeakerTurn]
        Speaker segments from transcript.
    variants_map : Optional[Dict[str, List[str]]]
        Mapping from contract to AI-generated variants.

    Returns
    -------
    List[LabelResult]
        Four LabelResult objects, one for each mode.
    """
    results = []

    # Define all four modes
    modes = [
        ("powell_only", "strict_literal"),
        ("powell_only", "variants"),
        ("full_transcript", "strict_literal"),
        ("full_transcript", "variants"),
    ]

    for scope, match_mode in modes:
        # Filter segments by scope
        if scope == "powell_only":
            relevant_segments = [s for s in segments if s.role == "powell"]
        else:  # full_transcript
            relevant_segments = segments

        # Concatenate text
        full_text = " ".join(seg.text for seg in relevant_segments)

        # Get phrases based on match_mode
        if match_mode == "variants" and variants_map and contract in variants_map:
            phrases = variants_map[contract]
        else:
            # Use strict literal synonyms from spec
            phrases = spec.synonyms

        # Count mentions with snippets
        count, snippets = count_phrase_mentions_with_snippets(
            full_text,
            phrases,
            case_sensitive=False,
            word_boundaries=True,  # Always use word boundaries for accuracy
        )

        # Compute binary outcomes
        mentioned_binary = 1 if count >= 1 else 0
        threshold_hit = 1 if count >= spec.threshold else 0

        # Store result
        mode_name = f"{scope}_{match_mode}"
        results.append(
            LabelResult(
                contract=contract,
                mode=mode_name,
                count=count,
                mentioned_binary=mentioned_binary,
                threshold_hit=threshold_hit,
                threshold=spec.threshold,
                debug_snippets=snippets,
            )
        )

    return results


def generate_labels_for_transcript(
    segments: List[SpeakerTurn],
    mapping: ContractMapping,
    variants_dir: Optional[Path] = None,
) -> List[LabelResult]:
    """
    Generate labels for all contracts in a transcript.

    Parameters
    ----------
    segments : List[SpeakerTurn]
        Speaker segments from transcript.
    mapping : ContractMapping
        Contract mapping with specifications.
    variants_dir : Optional[Path]
        Directory containing variant cache files.

    Returns
    -------
    List[LabelResult]
        All label results for all contracts and modes.
    """
    # Load variants if available
    variants_map = {}
    if variants_dir is not None:
        from .variants.generator import load_all_variants

        variant_results = load_all_variants(variants_dir)
        variants_map = {
            contract: result.variants for contract, result in variant_results.items()
        }

    all_results = []

    for contract in mapping.contracts():
        spec = mapping.get_spec(contract)
        if spec is None:
            # Create default spec if not found
            spec = ContractSpec(synonyms=mapping.phrases_for(contract))

        results = generate_labels_for_contract(contract, spec, segments, variants_map)
        all_results.extend(results)

    return all_results


def labels_to_dataframe(labels: List[LabelResult]) -> pd.DataFrame:
    """
    Convert label results to a pandas DataFrame.

    Parameters
    ----------
    labels : List[LabelResult]
        Label results from generate_labels_for_transcript.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: contract, mode, count, mentioned_binary,
        threshold_hit, threshold, debug_snippets
    """
    rows = []
    for label in labels:
        rows.append(
            {
                "contract": label.contract,
                "mode": label.mode,
                "count": label.count,
                "mentioned_binary": label.mentioned_binary,
                "threshold_hit": label.threshold_hit,
                "threshold": label.threshold,
                "debug_snippets": "; ".join(label.debug_snippets[:3]),  # First 3
            }
        )
    return pd.DataFrame(rows)


def generate_label_matrix(
    segments_dir: Path,
    mapping: ContractMapping,
    variants_dir: Optional[Path] = None,
    mode: str = "powell_only_strict_literal",
) -> pd.DataFrame:
    """
    Generate a label matrix for all transcripts and contracts.

    Parameters
    ----------
    segments_dir : Path
        Directory containing segment JSONL files.
    mapping : ContractMapping
        Contract mapping with specifications.
    variants_dir : Optional[Path]
        Directory containing variant cache files.
    mode : str
        Which mode to use: "powell_only_strict_literal", "powell_only_variants",
        "full_transcript_strict_literal", or "full_transcript_variants".

    Returns
    -------
    pd.DataFrame
        Matrix with rows=dates, columns=contracts, values=threshold_hit (0/1).
    """
    segments_dir = Path(segments_dir)

    all_data = []

    for segment_file in sorted(segments_dir.glob("*.jsonl")):
        date_str = segment_file.stem
        segments = load_segments_jsonl(segment_file)

        # Generate labels for this transcript
        labels = generate_labels_for_transcript(segments, mapping, variants_dir)

        # Filter to the specified mode
        mode_labels = [l for l in labels if l.mode == mode]

        # Create row for this date
        row = {"date": date_str}
        for label in mode_labels:
            row[label.contract] = label.threshold_hit

        all_data.append(row)

    df = pd.DataFrame(all_data)
    if "date" in df.columns:
        df = df.set_index("date")

    return df
