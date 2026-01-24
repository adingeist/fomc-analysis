"""
Kalshi Contract Analysis for Earnings Call Mention Markets.

Adapted from fomc_analysis.kalshi_contract_analyzer for earnings calls.

This module:
1. Fetches Kalshi earnings mention contracts (e.g., KXEARNINGSMENTIONMETA)
2. Extracts tracked words and thresholds
3. Analyzes earnings call transcripts for word frequencies
4. Builds statistical data on historical mention frequencies
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from fomc_analysis.kalshi_client_factory import KalshiClientProtocol, KalshiSdkAdapter


@dataclass
class EarningsContractWord:
    """A word tracked by a Kalshi earnings mention contract."""
    word: str
    ticker: str  # Company stock ticker (e.g., "META", "TSLA")
    market_ticker: str  # Kalshi market ticker
    market_title: str
    threshold: int = 1  # Minimum mentions required
    markets: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EarningsMentionAnalysis:
    """Analysis of word mentions in earnings call transcripts."""
    word: str
    ticker: str  # Company ticker
    total_calls: int
    calls_with_mention: int
    mention_frequency: float  # Proportion of calls with mention
    total_mentions: int
    avg_mentions_per_call: float
    max_mentions_in_call: int
    mention_counts_distribution: Dict[int, int]  # count -> number of calls

    def to_dict(self) -> dict:
        return asdict(self)


class EarningsContractAnalyzer:
    """
    Analyze Kalshi earnings call mention contracts.

    Parameters
    ----------
    kalshi_client : KalshiClientProtocol
        Kalshi API client
    ticker : str
        Company stock ticker (e.g., "META", "COIN")
    """

    def __init__(
        self,
        kalshi_client: KalshiClientProtocol,
        ticker: str,
    ):
        self.client = kalshi_client
        self.ticker = ticker.upper()

    async def fetch_contracts(
        self,
        market_status: Optional[str] = None,
    ) -> List[EarningsContractWord]:
        """
        Fetch earnings mention contracts for this ticker from Kalshi.

        Parameters
        ----------
        market_status : Optional[str]
            Filter by market status: "open", "closed", "resolved", "all"

        Returns
        -------
        List[EarningsContractWord]
            List of contract words
        """
        # Construct series ticker
        series_ticker = f"KXEARNINGSMENTION{self.ticker}"

        print(f"Fetching contracts for series: {series_ticker}")

        try:
            # Use async method if available (KalshiSdkAdapter)
            if isinstance(self.client, KalshiSdkAdapter):
                markets = await self.client.get_markets_async(series_ticker=series_ticker)
            else:
                # Fallback to sync for other client types
                markets = self.client.get_markets(series_ticker=series_ticker)
        except Exception as e:
            print(f"Error fetching markets: {e}")
            return []

        if not markets:
            print(f"No markets found for {series_ticker}")
            return []

        # Parse markets to extract words
        contract_words = {}  # word -> ContractWord

        for market in markets:
            ticker = market.get("ticker", "")
            title = market.get("title", "")
            status = market.get("status", "").lower()

            # Filter by status if requested
            if market_status and market_status.lower() != "all":
                requested_status = market_status.lower()
                if requested_status == "resolved":
                    if status not in {"resolved", "settled"}:
                        continue
                elif status != requested_status:
                    continue

            # Extract word and threshold from title
            # Expected formats:
            # - "Will [CEO NAME] say '[WORD]' at least [N] times?"
            # - "Will [CEO NAME] mention '[WORD]'?"

            word, threshold = self._parse_market_title(title)

            if not word:
                continue

            # Create or update ContractWord
            if word not in contract_words:
                contract_words[word] = EarningsContractWord(
                    word=word,
                    ticker=self.ticker,
                    market_ticker=ticker,
                    market_title=title,
                    threshold=threshold,
                    markets=[],
                )

            # Add this market to the contract
            contract_words[word].markets.append(market)

        return list(contract_words.values())

    def _parse_market_title(self, title: str) -> tuple[Optional[str], int]:
        """
        Parse market title to extract tracked word and threshold.

        Parameters
        ----------
        title : str
            Market title from Kalshi

        Returns
        -------
        tuple[Optional[str], int]
            (word, threshold) or (None, 1) if parsing fails
        """
        # Pattern 1: "say '[WORD]' at least [N] times"
        match = re.search(r"say ['\"](.+?)['\"] at least (\d+) times?", title, re.IGNORECASE)
        if match:
            word = match.group(1).lower()
            threshold = int(match.group(2))
            return word, threshold

        # Pattern 2: "mention '[WORD]' at least [N] times"
        match = re.search(r"mention ['\"](.+?)['\"] at least (\d+) times?", title, re.IGNORECASE)
        if match:
            word = match.group(1).lower()
            threshold = int(match.group(2))
            return word, threshold

        # Pattern 3: "say '[WORD]'" (no threshold specified)
        match = re.search(r"(?:say|mention) ['\"](.+?)['\"]", title, re.IGNORECASE)
        if match:
            word = match.group(1).lower()
            return word, 1

        return None, 1

    def analyze_transcripts(
        self,
        contract_words: List[EarningsContractWord],
        segments_dir: Path,
        speaker_mode: str = "executives_only",
    ) -> List[EarningsMentionAnalysis]:
        """
        Analyze earnings call transcripts for word mentions.

        Parameters
        ----------
        contract_words : List[EarningsContractWord]
            Contracts to analyze
        segments_dir : Path
            Directory with segmented transcripts (JSONL files)
        speaker_mode : str
            Which speakers to include: "ceo_only", "cfo_only", "executives_only", "full_transcript"

        Returns
        -------
        List[EarningsMentionAnalysis]
            Analysis results for each word
        """
        # Load all transcript segments for this ticker
        segment_files = sorted(Path(segments_dir).glob(f"{self.ticker}_*.jsonl"))

        if not segment_files:
            print(f"No transcript segments found for {self.ticker} in {segments_dir}")
            return []

        analyses = []

        for contract_word in contract_words:
            word = contract_word.word
            threshold = contract_word.threshold

            # Count mentions across all earnings calls
            call_counts = []  # List of mention counts per call

            for segment_file in segment_files:
                # Load segments
                segments = self._load_segments(segment_file)

                # Filter by speaker mode
                filtered_segments = self._filter_segments(segments, speaker_mode)

                # Combine text
                combined_text = " ".join(seg["text"] for seg in filtered_segments)

                # Count mentions (case-insensitive, word boundaries)
                pattern = r"\b" + re.escape(word) + r"\b"
                mentions = len(re.findall(pattern, combined_text, re.IGNORECASE))

                call_counts.append(mentions)

            # Calculate statistics
            total_calls = len(call_counts)
            calls_with_mention = sum(1 for count in call_counts if count >= threshold)
            total_mentions = sum(call_counts)

            avg_mentions = total_mentions / total_calls if total_calls > 0 else 0
            max_mentions = max(call_counts) if call_counts else 0
            mention_frequency = calls_with_mention / total_calls if total_calls > 0 else 0

            # Distribution
            distribution = {}
            for count in call_counts:
                distribution[count] = distribution.get(count, 0) + 1

            analysis = EarningsMentionAnalysis(
                word=word,
                ticker=self.ticker,
                total_calls=total_calls,
                calls_with_mention=calls_with_mention,
                mention_frequency=mention_frequency,
                total_mentions=total_mentions,
                avg_mentions_per_call=avg_mentions,
                max_mentions_in_call=max_mentions,
                mention_counts_distribution=distribution,
            )

            analyses.append(analysis)

        return analyses

    def _load_segments(self, segment_file: Path) -> List[Dict]:
        """Load segments from JSONL file."""
        segments = []
        with open(segment_file, "r") as f:
            for line in f:
                segments.append(json.loads(line))
        return segments

    def _filter_segments(self, segments: List[Dict], speaker_mode: str) -> List[Dict]:
        """Filter segments by speaker role."""
        if speaker_mode == "ceo_only":
            return [seg for seg in segments if seg.get("role") == "ceo"]
        elif speaker_mode == "cfo_only":
            return [seg for seg in segments if seg.get("role") == "cfo"]
        elif speaker_mode == "executives_only":
            return [seg for seg in segments if seg.get("role") in ("ceo", "cfo", "executive")]
        elif speaker_mode == "full_transcript":
            return segments
        else:
            raise ValueError(f"Unknown speaker_mode: {speaker_mode}")

    async def export_contract_mapping(
        self,
        output_file: Path,
        market_status: str = "all",
    ):
        """
        Export contract mapping to YAML format (compatible with FOMC framework).

        Parameters
        ----------
        output_file : Path
            Output YAML file path
        market_status : str
            Filter markets by status
        """
        import yaml

        contracts = await self.fetch_contracts(market_status=market_status)

        # Convert to YAML format
        mapping = {}

        for contract in contracts:
            # Contract name: "{word} ({threshold}+)" or just "{word}"
            if contract.threshold > 1:
                contract_name = f"{contract.word.title()} ({contract.threshold}+)"
            else:
                contract_name = contract.word.title()

            mapping[contract_name] = {
                "synonyms": [contract.word],
                "threshold": contract.threshold,
                "scope": "executives_only",  # Default
                "match_mode": "strict_literal",
                "description": f"{self.ticker} CEO/CFO mentions '{contract.word}' at least {contract.threshold} time(s)",
            }

        # Save to YAML
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            yaml.dump(mapping, f, default_flow_style=False, sort_keys=False)

        print(f"Exported {len(mapping)} contracts to {output_file}")


async def batch_fetch_contracts(
    kalshi_client: KalshiClientProtocol,
    tickers: List[str],
    market_status: Optional[str] = None,
) -> Dict[str, List[EarningsContractWord]]:
    """
    Fetch contracts for multiple tickers in parallel.

    Parameters
    ----------
    kalshi_client : KalshiClientProtocol
        Kalshi API client
    tickers : List[str]
        List of company stock tickers
    market_status : Optional[str]
        Filter by market status

    Returns
    -------
    Dict[str, List[EarningsContractWord]]
        Mapping of ticker -> list of contract words
    """
    # Create analyzers for each ticker
    analyzers = [
        EarningsContractAnalyzer(kalshi_client, ticker)
        for ticker in tickers
    ]

    # Fetch contracts in parallel using asyncio.gather
    results = await asyncio.gather(
        *[analyzer.fetch_contracts(market_status=market_status) for analyzer in analyzers],
        return_exceptions=True,
    )

    # Build result mapping
    contracts_by_ticker = {}
    for ticker, result in zip(tickers, results):
        if isinstance(result, Exception):
            print(f"Error fetching contracts for {ticker}: {result}")
            contracts_by_ticker[ticker] = []
        else:
            contracts_by_ticker[ticker] = result

    return contracts_by_ticker


async def analyze_earnings_kalshi_contracts(
    kalshi_client: KalshiClientProtocol,
    ticker: str,
    segments_dir: Path,
    output_dir: Path,
    market_status: str = "all",
    speaker_mode: str = "executives_only",
):
    """
    Convenience function to analyze earnings Kalshi contracts.

    Parameters
    ----------
    kalshi_client : KalshiClientProtocol
        Kalshi API client
    ticker : str
        Company stock ticker
    segments_dir : Path
        Directory with transcript segments
    output_dir : Path
        Output directory for results
    market_status : str
        Filter markets by status
    speaker_mode : str
        Speaker filtering mode
    """
    analyzer = EarningsContractAnalyzer(kalshi_client, ticker)

    # Fetch contracts
    print(f"\nFetching Kalshi contracts for {ticker}...")
    contracts = await analyzer.fetch_contracts(market_status=market_status)
    print(f"Found {len(contracts)} contract words")

    # Analyze transcripts
    print(f"\nAnalyzing {ticker} earnings call transcripts...")
    analyses = analyzer.analyze_transcripts(
        contracts,
        segments_dir=segments_dir,
        speaker_mode=speaker_mode,
    )

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save contract words
    contracts_file = output_dir / "contract_words.json"
    with open(contracts_file, "w") as f:
        json.dump([c.to_dict() for c in contracts], f, indent=2)
    print(f"\nSaved contract words to {contracts_file}")

    # Save analysis
    analysis_file = output_dir / "mention_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump([a.to_dict() for a in analyses], f, indent=2)
    print(f"Saved mention analysis to {analysis_file}")

    # Save summary CSV
    summary_df = pd.DataFrame([
        {
            "word": a.word,
            "ticker": a.ticker,
            "total_calls": a.total_calls,
            "mention_frequency": f"{a.mention_frequency:.1%}",
            "avg_mentions": f"{a.avg_mentions_per_call:.1f}",
            "max_mentions": a.max_mentions_in_call,
        }
        for a in analyses
    ])

    summary_file = output_dir / "mention_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary to {summary_file}")

    return contracts, analyses
