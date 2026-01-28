"""
Build synthetic historical outcomes from pre-Kalshi earnings data.

The FOMC framework had years of Federal Reserve transcripts predating Kalshi
contracts.  This module applies the same approach to earnings calls: scan
historical press releases or full transcripts for the words that Kalshi now
tracks, producing synthetic YES/NO outcomes for quarters that predate live
contracts.

This dramatically increases the training set for the Beta-Binomial model
(e.g. 7+ quarters instead of 2) and gives much tighter probability estimates.

Two source quality levels:

  "press_release" — SEC EDGAR 8-K exhibits (Item 2.02 / 9.01).
      Formal financial documents.  If a word appears here it was certainly
      discussed on the call, but absence is inconclusive because the Q&A
      discussion is not included.
      → found = YES, not-found = NaN (unknown) by default.

  "full_transcript" — complete earnings call transcript (sourced manually
      or from a paid API).  Definitive: found = YES, not-found = NO.
      → found = YES, not-found = NO.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd


@dataclass
class TextSource:
    """A single text document (press release or transcript) for one date."""

    ticker: str
    date: str  # YYYY-MM-DD
    text: str
    source_type: str  # "press_release" or "full_transcript"
    file_path: Optional[str] = None


@dataclass
class SyntheticOutcome:
    """Outcome for one word on one date, derived from text scanning."""

    word: str
    date: str
    mentioned: Optional[bool]  # True=YES, False=NO, None=unknown
    source_type: str
    match_count: int = 0  # how many times the word/variants appeared


def _word_variants(word: str) -> List[str]:
    """Split a Kalshi word spec into matchable variants.

    "VR / Virtual Reality" → ["vr", "virtual reality"]
    "Generative AI / Gen AI / Gen-AI" → ["generative ai", "gen ai", "gen-ai"]
    """
    return [v.strip().lower() for v in word.split("/") if v.strip()]


def scan_text_for_words(
    text: str,
    words: List[str],
) -> Dict[str, SyntheticOutcome]:
    """Scan a text document for a list of Kalshi-tracked words.

    Returns a dict mapping word → SyntheticOutcome with match details.
    """
    text_lower = text.lower()
    results = {}

    for word in words:
        variants = _word_variants(word)
        total_matches = 0
        found = False

        for variant in variants:
            if not variant:
                continue
            pattern = r"\b" + re.escape(variant) + r"\b"
            matches = re.findall(pattern, text_lower)
            total_matches += len(matches)
            if matches:
                found = True

        results[word] = SyntheticOutcome(
            word=word,
            date="",  # filled by caller
            mentioned=found if found else None,  # None = not found (ambiguous for PR)
            source_type="",  # filled by caller
            match_count=total_matches,
        )

    return results


def build_synthetic_history(
    text_sources: List[TextSource],
    words: List[str],
    absent_means_no: bool = False,
) -> pd.DataFrame:
    """Build a synthetic outcomes DataFrame from historical text sources.

    Parameters
    ----------
    text_sources : List[TextSource]
        Historical press releases or transcripts, sorted chronologically.
    words : List[str]
        Kalshi-tracked words to scan for.
    absent_means_no : bool
        If True, words not found in the text are marked as NO (0).
        If False (default), they are marked as NaN (unknown).
        Set True for full transcripts, False for press releases.

    Returns
    -------
    pd.DataFrame
        Index = date (datetime), columns = words (lowercase),
        values = 1 (YES), 0 (NO), or NaN (unknown).
    """
    rows = []

    for source in text_sources:
        scan = scan_text_for_words(source.text, words)
        row = {"date": source.date}

        for word in words:
            key = word.lower()
            result = scan.get(word)
            if result and result.mentioned:
                row[key] = 1
            elif absent_means_no or (source.source_type == "full_transcript"):
                row[key] = 0
            else:
                row[key] = np.nan  # unknown for press releases

            # Update metadata
            if result:
                result.date = source.date
                result.source_type = source.source_type

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    return df


def load_press_releases(
    ticker: str,
    pr_dir: Path = Path("data/earnings/press_releases"),
) -> List[TextSource]:
    """Load all press release text files for a ticker."""
    ticker_dir = pr_dir / ticker
    if not ticker_dir.exists():
        return []

    sources = []
    for filepath in sorted(ticker_dir.glob(f"{ticker}_*.txt")):
        date = filepath.stem.split("_", 1)[1]  # META_2025-01-29 → 2025-01-29
        text = filepath.read_text(encoding="utf-8", errors="ignore")
        sources.append(TextSource(
            ticker=ticker,
            date=date,
            text=text,
            source_type="press_release",
            file_path=str(filepath),
        ))

    return sources


def load_transcripts(
    ticker: str,
    transcripts_dir: Path = Path("data/earnings/transcripts"),
) -> List[TextSource]:
    """Load all full transcript text files for a ticker."""
    ticker_dir = transcripts_dir / ticker
    if not ticker_dir.exists():
        return []

    sources = []
    for filepath in sorted(ticker_dir.glob(f"{ticker}_*.txt")):
        date = filepath.stem.split("_", 1)[1]
        text = filepath.read_text(encoding="utf-8", errors="ignore")
        sources.append(TextSource(
            ticker=ticker,
            date=date,
            text=text,
            source_type="full_transcript",
            file_path=str(filepath),
        ))

    # Also check for .json files
    for filepath in sorted(ticker_dir.glob(f"{ticker}_*.json")):
        date = filepath.stem.split("_", 1)[1]
        with open(filepath) as f:
            data = json.load(f)
        text = data.get("text", data.get("transcript", ""))
        sources.append(TextSource(
            ticker=ticker,
            date=date,
            text=text,
            source_type="full_transcript",
            file_path=str(filepath),
        ))

    return sources


def merge_synthetic_and_real(
    synthetic: pd.DataFrame,
    real_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Merge synthetic history with real Kalshi outcomes.

    Where both exist for the same date, real outcomes take precedence.
    Synthetic data fills in older dates that predate Kalshi contracts.

    Parameters
    ----------
    synthetic : pd.DataFrame
        Synthetic outcomes from text scanning. May contain NaN.
    real_outcomes : pd.DataFrame
        Real outcomes from settled Kalshi contracts (0 or 1).

    Returns
    -------
    pd.DataFrame
        Combined outcomes, chronologically sorted.
    """
    # Normalize column names to lowercase
    synthetic.columns = [c.lower() for c in synthetic.columns]
    real_outcomes.columns = [c.lower() for c in real_outcomes.columns]

    # Get all dates and words
    all_dates = sorted(set(synthetic.index) | set(real_outcomes.index))
    all_words = sorted(set(synthetic.columns) | set(real_outcomes.columns))

    # Build combined DataFrame
    combined = pd.DataFrame(index=all_dates, columns=all_words, dtype=float)
    combined.index = pd.to_datetime(combined.index)

    # Fill with synthetic first
    for date in synthetic.index:
        for word in synthetic.columns:
            val = synthetic.loc[date, word]
            if pd.notna(val):
                combined.loc[date, word] = val

    # Override with real outcomes (these are authoritative)
    for date in real_outcomes.index:
        for word in real_outcomes.columns:
            val = real_outcomes.loc[date, word]
            if pd.notna(val):
                combined.loc[date, word] = val

    combined = combined.sort_index()
    return combined


def get_words_for_ticker(
    ticker: str,
    ground_truth_dir: Path = Path("data/ground_truth"),
) -> List[str]:
    """Get all tracked words for a ticker from the ground truth dataset."""
    gt_path = ground_truth_dir / "ground_truth.json"
    if not gt_path.exists():
        return []

    with open(gt_path) as f:
        data = json.load(f)

    words = set()
    for c in data["contracts"]:
        if c["company_ticker"] == ticker.upper():
            words.add(c["word"])

    return sorted(words)


def build_expanded_training_data(
    ticker: str,
    ground_truth_dir: Path = Path("data/ground_truth"),
    pr_dir: Path = Path("data/earnings/press_releases"),
    transcripts_dir: Path = Path("data/earnings/transcripts"),
    absent_in_pr_means_no: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Build an expanded training dataset for a ticker by combining:
    1. Synthetic outcomes from press releases (going back further in time)
    2. Synthetic outcomes from full transcripts (if available)
    3. Real outcomes from settled Kalshi contracts

    Parameters
    ----------
    ticker : str
        Company ticker symbol.
    ground_truth_dir : Path
        Directory containing ground_truth.json.
    pr_dir : Path
        Directory containing press release text files.
    transcripts_dir : Path
        Directory containing full transcript text files.
    absent_in_pr_means_no : bool
        If True, words not found in press releases are marked NO.
        If False, they are marked NaN (conservative — only confirmed YES counts).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict]
        (features, outcomes, diagnostics)
        - features: same as outcomes (for Beta-Binomial compatibility)
        - outcomes: combined synthetic + real outcomes
        - diagnostics: metadata about data sources
    """
    words = get_words_for_ticker(ticker, ground_truth_dir)
    if not words:
        return pd.DataFrame(), pd.DataFrame(), {"error": "No tracked words found"}

    diagnostics = {
        "ticker": ticker,
        "tracked_words": len(words),
        "press_releases": 0,
        "full_transcripts": 0,
        "real_kalshi_dates": 0,
        "total_dates": 0,
        "synthetic_dates": 0,
        "data_sources": [],
    }

    # Load text sources
    pr_sources = load_press_releases(ticker, pr_dir)
    tr_sources = load_transcripts(ticker, transcripts_dir)
    diagnostics["press_releases"] = len(pr_sources)
    diagnostics["full_transcripts"] = len(tr_sources)

    # Build synthetic outcomes from each source type
    synthetic_parts = []

    if pr_sources:
        pr_outcomes = build_synthetic_history(
            pr_sources, words, absent_means_no=absent_in_pr_means_no
        )
        synthetic_parts.append(pr_outcomes)
        diagnostics["data_sources"].append(
            f"{len(pr_sources)} press releases ({pr_sources[0].date} to {pr_sources[-1].date})"
        )

    if tr_sources:
        tr_outcomes = build_synthetic_history(
            tr_sources, words, absent_means_no=True  # full transcripts: absent = NO
        )
        synthetic_parts.append(tr_outcomes)
        diagnostics["data_sources"].append(
            f"{len(tr_sources)} full transcripts ({tr_sources[0].date} to {tr_sources[-1].date})"
        )

    # Combine synthetic sources (transcripts override press releases for same date)
    if synthetic_parts:
        synthetic = synthetic_parts[0]
        for part in synthetic_parts[1:]:
            synthetic = merge_synthetic_and_real(synthetic, part)
    else:
        synthetic = pd.DataFrame()

    # Load real Kalshi outcomes
    from .ground_truth import load_ground_truth, build_backtest_dataframes

    gt_path = ground_truth_dir / "ground_truth.json"
    if gt_path.exists():
        dataset = load_ground_truth(ground_truth_dir)
        _, real_outcomes, _ = build_backtest_dataframes(dataset, ticker)
        diagnostics["real_kalshi_dates"] = len(real_outcomes)
        diagnostics["data_sources"].append(
            f"{len(real_outcomes)} Kalshi settled dates"
        )
    else:
        real_outcomes = pd.DataFrame()

    # Merge synthetic + real (real takes precedence)
    if not synthetic.empty and not real_outcomes.empty:
        combined = merge_synthetic_and_real(synthetic, real_outcomes)
    elif not synthetic.empty:
        combined = synthetic
    elif not real_outcomes.empty:
        combined = real_outcomes
    else:
        return pd.DataFrame(), pd.DataFrame(), {"error": "No data found"}

    diagnostics["total_dates"] = len(combined)
    diagnostics["synthetic_dates"] = diagnostics["total_dates"] - diagnostics["real_kalshi_dates"]

    # For the Beta-Binomial model, features = outcomes
    features = combined.copy()
    outcomes = combined.copy()

    return features, outcomes, diagnostics


def print_expanded_data_summary(
    ticker: str,
    outcomes: pd.DataFrame,
    diagnostics: dict,
):
    """Print a summary of the expanded training data."""
    print(f"\n  Expanded training data for {ticker}:")
    print(f"    Total event dates:  {diagnostics['total_dates']}")
    print(f"      Synthetic:        {diagnostics['synthetic_dates']}")
    print(f"      Real Kalshi:      {diagnostics['real_kalshi_dates']}")
    print(f"    Tracked words:      {diagnostics['tracked_words']}")

    for src in diagnostics.get("data_sources", []):
        print(f"    Source: {src}")

    if not outcomes.empty:
        # Coverage: percentage of non-NaN values
        total_cells = outcomes.size
        known_cells = outcomes.notna().sum().sum()
        yes_cells = (outcomes == 1).sum().sum()
        no_cells = (outcomes == 0).sum().sum()
        nan_cells = outcomes.isna().sum().sum()

        print(f"    Coverage: {known_cells}/{total_cells} "
              f"({known_cells/total_cells:.0%} known)")
        print(f"    YES: {yes_cells}, NO: {no_cells}, Unknown: {nan_cells}")

        # Per-date summary
        print(f"\n    Date breakdown:")
        for date in outcomes.index:
            row = outcomes.loc[date]
            n_yes = (row == 1).sum()
            n_no = (row == 0).sum()
            n_nan = row.isna().sum()
            total = len(row)
            print(f"      {date.date()}: {n_yes} YES, {n_no} NO, "
                  f"{n_nan} unknown ({n_yes + n_no}/{total} known)")
