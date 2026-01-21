"""
Kalshi Contract Analysis for FOMC Mention Markets
==================================================

This module fetches Kalshi "mention" contracts (e.g., KXFEDMENTION series),
extracts the tracked words, generates variants using OpenAI, and analyzes
historical FOMC transcripts to build statistical data on mention frequencies.

Usage:
    1. Fetch contracts from Kalshi API
    2. Parse market titles to extract tracked words
    3. Generate word variants using OpenAI (plurals, possessives, compounds)
    4. Scan FOMC transcripts for matches
    5. Build statistical analysis of historical mention frequencies
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from openai import OpenAI

from .kalshi_api import KalshiClient
from .variants.generator import generate_variants
from .parsing.speaker_segmenter import load_segments_jsonl


@dataclass
class ContractWord:
    """A word tracked by a Kalshi mention contract."""
    word: str
    market_ticker: str
    market_title: str
    variants: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MentionAnalysis:
    """Analysis of word mentions in FOMC transcripts."""
    word: str
    variants: List[str]
    total_transcripts: int
    transcripts_with_mention: int
    mention_frequency: float  # Proportion of transcripts with mention
    total_mentions: int
    avg_mentions_per_transcript: float
    max_mentions_in_transcript: int
    mention_counts_distribution: Dict[int, int]  # count -> number of transcripts

    def to_dict(self) -> dict:
        return asdict(self)


def parse_market_title(title: str) -> Optional[str]:
    """
    Extract the tracked word from a Kalshi market title.

    Examples:
        "President" -> "President"
        "Layoff mention" -> "Layoff"
        "AI / Artificial Intelligence mention" -> "AI / Artificial Intelligence"
        "Good Afternoon mention" -> "Good Afternoon"

    Parameters
    ----------
    title : str
        Market title from Kalshi.

    Returns
    -------
    Optional[str]
        Extracted word/phrase, or None if parsing fails.
    """
    # Remove common suffixes
    title = title.strip()

    # Pattern 1: Remove " mention" suffix
    if title.lower().endswith(" mention"):
        return title[:-8].strip()

    # Pattern 2: Direct word (single or multi-word)
    return title


def fetch_mention_contracts(
    kalshi_client: KalshiClient,
    series_ticker: str = "KXFEDMENTION",
    event_ticker: Optional[str] = None,
) -> List[ContractWord]:
    """
    Fetch mention contracts from Kalshi API.

    Parameters
    ----------
    kalshi_client : KalshiClient
        Configured Kalshi API client.
    series_ticker : str, default="KXFEDMENTION"
        Series ticker for mention contracts.
    event_ticker : Optional[str]
        Specific event ticker (e.g., "kxfedmention-26jan").
        If None, fetches all markets in the series.

    Returns
    -------
    List[ContractWord]
        List of contract words (without variants yet).
    """
    contracts = []

    if event_ticker:
        # Fetch specific event
        event_data = kalshi_client.get_event(event_ticker, with_nested_markets=True)
        markets = event_data.get("event", {}).get("markets", [])
    else:
        # Fetch all markets in series
        markets = kalshi_client.get_markets(series_ticker=series_ticker)

    for market in markets:
        title = market.get("title", "")
        ticker = market.get("ticker", "")

        word = parse_market_title(title)
        if word:
            contracts.append(ContractWord(
                word=word,
                market_ticker=ticker,
                market_title=title,
                variants=[],  # Will be filled later
            ))

    return contracts


def generate_word_variants(
    contract_words: List[ContractWord],
    openai_client: OpenAI,
    cache_dir: Path = Path("data/kalshi_variants"),
) -> List[ContractWord]:
    """
    Generate word variants using OpenAI for each contract word.

    Parameters
    ----------
    contract_words : List[ContractWord]
        List of contract words.
    openai_client : OpenAI
        Configured OpenAI client.
    cache_dir : Path
        Directory to cache variant results.

    Returns
    -------
    List[ContractWord]
        Contract words with variants populated.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    for contract in contract_words:
        # Generate variants using existing generator
        result = generate_variants(
            contract=contract.word,
            base_phrases=[contract.word],
            openai_client=openai_client,
            cache_dir=cache_dir,
            model="gpt-4o-mini",
            force_regenerate=False,
        )

        contract.variants = result.variants

    return contract_words


def scan_transcript_for_words(
    segments_file: Path,
    contract_words: List[ContractWord],
    scope: str = "powell_only",
) -> Dict[str, int]:
    """
    Scan a single transcript for word mentions.

    Parameters
    ----------
    segments_file : Path
        Path to JSONL file with speaker segments.
    contract_words : List[ContractWord]
        List of contract words with variants.
    scope : str, default="powell_only"
        Search scope: "powell_only" or "full_transcript".

    Returns
    -------
    Dict[str, int]
        Dictionary mapping word -> mention count.
    """
    segments = load_segments_jsonl(segments_file)

    # Build text based on scope
    if scope == "powell_only":
        text_parts = [
            seg["text"] for seg in segments
            if seg.get("role", "").lower() in ["chair", "chairman"]
        ]
    else:
        text_parts = [seg["text"] for seg in segments]

    text = " ".join(text_parts).lower()

    # Count mentions for each word
    mention_counts = {}

    for contract in contract_words:
        count = 0
        for variant in contract.variants:
            variant_lower = variant.lower()

            # Use word boundary matching to avoid false positives
            # e.g., "layoff" should not match "playoff"
            pattern = r'\b' + re.escape(variant_lower) + r'\b'
            matches = re.findall(pattern, text)
            count += len(matches)

        mention_counts[contract.word] = count

    return mention_counts


def analyze_historical_mentions(
    contract_words: List[ContractWord],
    segments_dir: Path = Path("data/segments"),
    scope: str = "powell_only",
) -> List[MentionAnalysis]:
    """
    Analyze historical mention frequencies across all FOMC transcripts.

    Parameters
    ----------
    contract_words : List[ContractWord]
        List of contract words with variants.
    segments_dir : Path
        Directory containing segmented transcripts (JSONL files).
    scope : str, default="powell_only"
        Search scope: "powell_only" or "full_transcript".

    Returns
    -------
    List[MentionAnalysis]
        Statistical analysis for each word.
    """
    segments_files = sorted(segments_dir.glob("*.jsonl"))

    if not segments_files:
        raise ValueError(f"No segment files found in {segments_dir}")

    # Initialize tracking dictionaries
    word_to_counts = {contract.word: [] for contract in contract_words}

    # Scan all transcripts
    for segments_file in segments_files:
        try:
            mention_counts = scan_transcript_for_words(
                segments_file, contract_words, scope
            )

            for word, count in mention_counts.items():
                word_to_counts[word].append(count)
        except Exception as e:
            print(f"Warning: Failed to process {segments_file}: {e}")
            continue

    # Build analysis for each word
    analyses = []

    for contract in contract_words:
        counts = word_to_counts[contract.word]

        if not counts:
            continue

        total_transcripts = len(counts)
        transcripts_with_mention = sum(1 for c in counts if c > 0)
        mention_frequency = transcripts_with_mention / total_transcripts
        total_mentions = sum(counts)
        avg_mentions = total_mentions / total_transcripts
        max_mentions = max(counts)

        # Build distribution: count -> number of transcripts
        distribution = {}
        for count in counts:
            distribution[count] = distribution.get(count, 0) + 1

        analyses.append(MentionAnalysis(
            word=contract.word,
            variants=contract.variants,
            total_transcripts=total_transcripts,
            transcripts_with_mention=transcripts_with_mention,
            mention_frequency=mention_frequency,
            total_mentions=total_mentions,
            avg_mentions_per_transcript=avg_mentions,
            max_mentions_in_transcript=max_mentions,
            mention_counts_distribution=distribution,
        ))

    return analyses


def save_analysis_results(
    contract_words: List[ContractWord],
    analyses: List[MentionAnalysis],
    output_dir: Path = Path("data/kalshi_analysis"),
):
    """
    Save analysis results to JSON and CSV files.

    Parameters
    ----------
    contract_words : List[ContractWord]
        List of contract words with variants.
    analyses : List[MentionAnalysis]
        Statistical analysis results.
    output_dir : Path
        Directory to save results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save contract words with variants
    contracts_file = output_dir / "contract_words.json"
    with open(contracts_file, "w") as f:
        json.dump(
            [c.to_dict() for c in contract_words],
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Save detailed analysis
    analysis_file = output_dir / "mention_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(
            [a.to_dict() for a in analyses],
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Save summary CSV
    summary_data = []
    for analysis in analyses:
        summary_data.append({
            "Word": analysis.word,
            "Variants Count": len(analysis.variants),
            "Total Transcripts": analysis.total_transcripts,
            "Transcripts with Mention": analysis.transcripts_with_mention,
            "Mention Frequency": f"{analysis.mention_frequency:.2%}",
            "Total Mentions": analysis.total_mentions,
            "Avg Mentions/Transcript": f"{analysis.avg_mentions_per_transcript:.2f}",
            "Max Mentions": analysis.max_mentions_in_transcript,
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("Mention Frequency", ascending=False)

    summary_file = output_dir / "mention_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"\n✓ Analysis results saved to {output_dir}/")
    print(f"  - contract_words.json: Contract words with variants")
    print(f"  - mention_analysis.json: Detailed statistical analysis")
    print(f"  - mention_summary.csv: Summary table")


def run_kalshi_analysis(
    kalshi_client: KalshiClient,
    openai_client: OpenAI,
    series_ticker: str = "KXFEDMENTION",
    event_ticker: Optional[str] = None,
    segments_dir: Path = Path("data/segments"),
    output_dir: Path = Path("data/kalshi_analysis"),
    scope: str = "powell_only",
):
    """
    Run complete Kalshi contract analysis pipeline.

    Parameters
    ----------
    kalshi_client : KalshiClient
        Configured Kalshi API client.
    openai_client : OpenAI
        Configured OpenAI client.
    series_ticker : str, default="KXFEDMENTION"
        Kalshi series ticker.
    event_ticker : Optional[str]
        Specific event ticker (e.g., "kxfedmention-26jan").
    segments_dir : Path
        Directory with segmented transcripts.
    output_dir : Path
        Directory to save results.
    scope : str, default="powell_only"
        Search scope: "powell_only" or "full_transcript".
    """
    print(f"=== Kalshi Contract Analysis ===\n")
    print(f"Series: {series_ticker}")
    if event_ticker:
        print(f"Event: {event_ticker}")
    print(f"Scope: {scope}\n")

    # Step 1: Fetch contracts
    print("Step 1: Fetching contracts from Kalshi API...")
    contract_words = fetch_mention_contracts(
        kalshi_client, series_ticker, event_ticker
    )
    print(f"✓ Fetched {len(contract_words)} contract words\n")

    # Step 2: Generate variants
    print("Step 2: Generating word variants with OpenAI...")
    contract_words = generate_word_variants(contract_words, openai_client)
    print(f"✓ Generated variants for {len(contract_words)} words\n")

    # Step 3: Analyze historical mentions
    print("Step 3: Analyzing historical mentions in FOMC transcripts...")
    analyses = analyze_historical_mentions(
        contract_words, segments_dir, scope
    )
    print(f"✓ Analyzed {len(analyses)} words across transcripts\n")

    # Step 4: Save results
    print("Step 4: Saving analysis results...")
    save_analysis_results(contract_words, analyses, output_dir)
    print("\n=== Analysis Complete ===")
