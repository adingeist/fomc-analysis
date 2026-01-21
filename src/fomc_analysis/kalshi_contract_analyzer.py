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

import asyncio
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from openai import OpenAI

from .kalshi_client_factory import KalshiClientProtocol
from .variants.generator import generate_variants
from .parsing.speaker_segmenter import load_segments_jsonl


@dataclass
class ContractWord:
    """A word tracked by a Kalshi mention contract."""
    word: str
    market_ticker: str
    market_title: str
    variants: List[str]
    threshold: Optional[int] = None
    markets: List[Dict[str, Any]] = field(default_factory=list)

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

    market_status : Optional[str]
        Optional status filter ("open", "resolved", etc.).

    Returns
    -------
    Optional[str]
        Extracted word/phrase, or None if parsing fails.
    """
    word, _ = parse_market_metadata(title)
    return word


PAREN_THRESHOLD_PATTERN = re.compile(
    r"\((?P<threshold>\d+)\s*\+\s*(?:times|mentions)?\)",
    re.IGNORECASE,
)
THRESHOLD_PATTERN = re.compile(
    r"(?P<threshold>\d+)\s*\+\s*(?:times|mentions)?\)?$",
    re.IGNORECASE,
)
QUOTE_PATTERN = re.compile(r"[\"'“”‘’]([^\"'“”‘’]+)[\"'“”‘’]")
QUESTION_FOCUS_PATTERNS = [
    re.compile(r"will\s+(?:chair\s+)?powell\s+(?:say|mention|use|reference)\s+(?P<phrase>.+)", re.IGNORECASE),
]
TRAILING_CONTEXT_CUES = [
    " at his ",
    " at the ",
    " at ",
    " during ",
    " before ",
    " in his ",
    " in the ",
    " in ",
    " on ",
    " for ",
    " by ",
]


def parse_market_metadata(title: str) -> tuple[Optional[str], Optional[int]]:
    """Return cleaned contract phrase and inferred threshold from a title."""
    if not title:
        return None, None

    cleaned = title.strip()
    threshold = None

    match = PAREN_THRESHOLD_PATTERN.search(cleaned)
    if match:
        threshold = int(match.group("threshold"))
        cleaned = (cleaned[: match.start()] + cleaned[match.end():]).strip()

    lower = cleaned.lower()
    if lower.endswith(" mention"):
        cleaned = cleaned[: -len(" mention")].strip()
    elif lower.endswith(" mentions"):
        cleaned = cleaned[: -len(" mentions")].strip()

    match = THRESHOLD_PATTERN.search(cleaned)
    if match:
        threshold = int(match.group("threshold"))
        cleaned = cleaned[: match.start()].rstrip().rstrip("(").strip()

    cleaned = extract_base_phrase(cleaned)

    return cleaned or None, threshold


def strip_contextual_suffix(phrase: str) -> str:
    lowered = phrase.lower()
    for cue in TRAILING_CONTEXT_CUES:
        idx = lowered.find(cue)
        if idx != -1:
            return phrase[:idx]
    return phrase


def normalize_phrase_case(phrase: str) -> str:
    stripped = phrase.strip()
    if not stripped:
        return phrase
    if stripped.isupper() and len(stripped) > 3:
        return stripped.title()
    if stripped.islower():
        return stripped.title()
    return stripped


def extract_base_phrase(text: str) -> str:
    if not text:
        return text

    stripped = text.strip().strip("?")

    quote_match = QUOTE_PATTERN.search(stripped)
    if quote_match:
        candidate = quote_match.group(1)
        return normalize_phrase_case(candidate)

    for pattern in QUESTION_FOCUS_PATTERNS:
        match = pattern.search(stripped)
        if match:
            phrase = match.group("phrase")
            phrase = strip_contextual_suffix(phrase)
            phrase = re.sub(r"\([^)]*\)", "", phrase)
            phrase = re.split(r"\bor\b", phrase, 1)[0]
            phrase = phrase.strip()
            phrase = phrase.strip(" \"'.,")
            return normalize_phrase_case(phrase)

    stripped = re.sub(r"\([^)]*\)", "", stripped)
    stripped = stripped.strip()
    return normalize_phrase_case(stripped)


def _segment_attr(segment, attr: str):
    if isinstance(segment, dict):
        return segment.get(attr)
    return getattr(segment, attr, None)


def fetch_mention_contracts(
    kalshi_client: KalshiClientProtocol,
    series_ticker: str = "KXFEDMENTION",
    event_ticker: Optional[str] = None,
    market_status: Optional[str] = None,
) -> List[ContractWord]:
    """
    Fetch mention contracts from Kalshi API.

    Parameters
    ----------
    kalshi_client : KalshiClientProtocol
        Configured Kalshi API client (legacy REST or SDK adapter).
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
    contracts: Dict[str, ContractWord] = {}

    if event_ticker:
        # Fetch specific event
        event_data = kalshi_client.get_event(event_ticker, with_nested_markets=True)
        markets = event_data.get("event", {}).get("markets", [])
    else:
        # Fetch all markets in series
        markets = kalshi_client.get_markets(
            series_ticker=series_ticker,
        )

    if market_status:
        desired_status = market_status.lower()

        def _status_matches(market: Dict[str, Any]) -> bool:
            status = str(market.get("status", "")).lower()
            return status == desired_status

        filtered = [m for m in markets if _status_matches(m)]
        if not filtered:
            print(
                f"Warning: No markets returned with status '{market_status}'."
            )
        else:
            markets = filtered

    for market in markets:
        title = market.get("title", "")
        ticker = market.get("ticker", "")

        word, threshold = parse_market_metadata(title)
        if not word:
            continue

        threshold_value = threshold if threshold is not None else 1

        market_record = {
            "ticker": ticker,
            "title": title,
            "threshold": threshold_value,
            "event_ticker": market.get("event_ticker"),
            "close_time": market.get("close_time"),
            "expiration_time": market.get("expiration_time"),
            "close_date": None,
            "expiration_date": None,
        }
        from datetime import datetime
        for ts_key, date_key in [("close_time", "close_date"), ("expiration_time", "expiration_date")]:
            ts = market_record[ts_key]
            if ts:
                try:
                    market_record[date_key] = datetime.fromisoformat(ts.replace("Z", "+00:00")).date().isoformat()
                except ValueError:
                    market_record[date_key] = ts[:10]

        key = f"{word.lower()}__{threshold_value}"
        existing = contracts.get(key)
        if existing:
            if threshold_value is not None and (
                existing.threshold is None or existing.threshold < threshold_value
            ):
                existing.threshold = threshold_value
                existing.market_ticker = ticker
                existing.market_title = title
            existing.markets.append(market_record)
            continue

        contracts[key] = ContractWord(
            word=word,
            market_ticker=ticker,
            market_title=title,
            variants=[],  # Will be filled later
            threshold=threshold_value,
            markets=[market_record],
        )

    return list(contracts.values())


VARIANT_BATCH_SIZE = 10


def generate_word_variants(
    contract_words: List[ContractWord],
    openai_client: OpenAI,
    cache_dir: Path = Path("data/kalshi_variants"),
) -> List[ContractWord]:
    """
    Generate word variants using OpenAI for each contract word.

    Requests are dispatched concurrently (10 at a time) using asyncio to
    accelerate batched generation. Progress prints are streamed to stdout.
    """
    if not contract_words:
        return contract_words

    cache_dir.mkdir(parents=True, exist_ok=True)

    async def runner():
        semaphore = asyncio.Semaphore(VARIANT_BATCH_SIZE)
        print_lock = asyncio.Lock()
        total = len(contract_words)
        completed = 0

        async def process(contract: ContractWord):
            nonlocal completed
            async with semaphore:
                try:
                    threshold_value = contract.threshold if contract.threshold is not None else 1
                    market_records = [
                        {
                            "ticker": entry.get("ticker"),
                            "title": entry.get("title"),
                            "threshold": entry.get("threshold") if entry.get("threshold") is not None else 1,
                        }
                        for entry in contract.markets
                    ] or [{
                        "ticker": contract.market_ticker,
                        "title": contract.market_title,
                        "threshold": threshold_value,
                    }]

                    result = await asyncio.to_thread(
                        generate_variants,
                        contract.word,
                        [contract.word.lower()],
                        openai_client,
                        cache_dir,
                        "gpt-4o-mini",
                        False,
                        {
                            "threshold": threshold_value,
                            "markets": market_records,
                        },
                    )
                    contract.variants = result.variants
                except Exception as exc:  # pragma: no cover - network/IO errors
                    contract.variants = [contract.word.lower()]
                    async with print_lock:
                        print(f"Variant generation failed for {contract.word}: {exc}")
                finally:
                    async with print_lock:
                        completed += 1
                        print(f"[variants] {completed}/{total}: {contract.word}", flush=True)

        tasks = [asyncio.create_task(process(contract)) for contract in contract_words]
        await asyncio.gather(*tasks)

    asyncio.run(runner())

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
    text_parts: List[str] = []
    for seg in segments:
        text = _segment_attr(seg, "text")
        if not text:
            continue
        role = (_segment_attr(seg, "role") or "").lower()
        if scope == "powell_only" and role != "powell":
            continue
        text_parts.append(text)

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
    summary_columns = [
        "Word",
        "Variants Count",
        "Total Transcripts",
        "Transcripts with Mention",
        "Mention Frequency",
        "Total Mentions",
        "Avg Mentions/Transcript",
        "Max Mentions",
    ]
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

    summary_df = pd.DataFrame(summary_data, columns=summary_columns)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("Mention Frequency", ascending=False)

    summary_file = output_dir / "mention_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"\n✓ Analysis results saved to {output_dir}/")
    print(f"  - contract_words.json: Contract words with variants")
    print(f"  - mention_analysis.json: Detailed statistical analysis")
    print(f"  - mention_summary.csv: Summary table")


def contract_words_to_mapping(
    contract_words: List[ContractWord],
    default_scope: str = "powell_only",
) -> Dict[str, Dict[str, object]]:
    """Convert fetched contract words into contract mapping entries."""
    mapping: Dict[str, Dict[str, object]] = {}

    def display_name(contract: ContractWord) -> str:
        if contract.threshold and contract.threshold > 1:
            return f"{contract.word} ({contract.threshold}+)"
        return contract.word

    for contract in sorted(contract_words, key=lambda c: c.word.lower()):
        base = contract.word.lower()
        variants = sorted({base, *(variant.lower() for variant in contract.variants or [])})
        match_mode = "variants" if len(variants) > 1 else "strict_literal"

        mapping[display_name(contract)] = {
            "synonyms": variants,
            "threshold": contract.threshold or 1,
            "scope": default_scope,
            "match_mode": match_mode,
            "description": (
                f"Auto-generated from Kalshi market '{contract.market_title}' "
                f"({contract.market_ticker})"
            ),
        }

    return mapping


def run_kalshi_analysis(
    kalshi_client: KalshiClientProtocol,
    openai_client: OpenAI,
    series_ticker: str = "KXFEDMENTION",
    event_ticker: Optional[str] = None,
    segments_dir: Path = Path("data/segments"),
    output_dir: Path = Path("data/kalshi_analysis"),
    scope: str = "powell_only",
    market_status: Optional[str] = None,
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
    market_status : Optional[str]
        Optional Kalshi market status filter passed to `get_markets`.
    """
    print(f"=== Kalshi Contract Analysis ===\n")
    print(f"Series: {series_ticker}")
    if event_ticker:
        print(f"Event: {event_ticker}")
    print(f"Scope: {scope}\n")

    # Step 1: Fetch contracts
    print("Step 1: Fetching contracts from Kalshi API...")
    contract_words = fetch_mention_contracts(
        kalshi_client,
        series_ticker,
        event_ticker,
        market_status=market_status,
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
