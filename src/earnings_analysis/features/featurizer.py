"""
Featurizer for earnings call transcripts.

Extracts features from segmented transcripts:
- Keyword frequencies (sentiment, guidance, competition, etc.)
- Speaker-specific features (CEO vs CFO)
- Section-specific features (prepared remarks vs Q&A)
- Comparison to historical baselines

Similar to FOMC featurizer but adapted for earnings calls.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import yaml

from ..parsing import SpeakerSegment


class EarningsFeaturizer:
    """
    Extract features from earnings call transcripts.

    Parameters
    ----------
    keywords_config : Path or dict
        Path to keywords YAML or dict of keywords
    speaker_mode : str
        Which speakers to include: "executives_only", "ceo_only", "cfo_only", "full_transcript"
    phrase_mode : str
        How to match phrases: "strict_literal" or "variants"
    """

    def __init__(
        self,
        keywords_config: Path | dict,
        speaker_mode: str = "executives_only",
        phrase_mode: str = "variants",
    ):
        self.speaker_mode = speaker_mode
        self.phrase_mode = phrase_mode

        # Load keywords
        if isinstance(keywords_config, dict):
            self.keywords = keywords_config
        else:
            with open(keywords_config, "r") as f:
                self.keywords = yaml.safe_load(f)

    def featurize(
        self,
        segments: List[SpeakerSegment],
        ticker: str,
        call_date: str,
    ) -> pd.Series:
        """
        Extract features from transcript segments.

        Parameters
        ----------
        segments : List[SpeakerSegment]
            Speaker segments from transcript
        ticker : str
            Stock ticker
        call_date : str
            Earnings call date (YYYY-MM-DD)

        Returns
        -------
        pd.Series
            Feature vector with index names
        """
        # Filter segments by speaker mode
        filtered_segments = self._filter_segments(segments)

        # Combine text
        combined_text = " ".join(seg.text for seg in filtered_segments)
        combined_text_lower = combined_text.lower()

        # Extract features
        features = {
            "ticker": ticker,
            "call_date": call_date,
            "total_words": len(combined_text.split()),
            "total_characters": len(combined_text),
        }

        # Extract keyword features for each category
        for category_name, category_keywords in self.keywords.items():
            category_features = self._extract_category_features(
                combined_text_lower,
                category_keywords,
                category_name,
            )
            features.update(category_features)

        # Speaker-specific features
        speaker_features = self._extract_speaker_features(segments)
        features.update(speaker_features)

        return pd.Series(features)

    def _filter_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Filter segments based on speaker mode."""
        if self.speaker_mode == "ceo_only":
            return [seg for seg in segments if seg.role == "ceo"]
        elif self.speaker_mode == "cfo_only":
            return [seg for seg in segments if seg.role == "cfo"]
        elif self.speaker_mode == "executives_only":
            return [seg for seg in segments if seg.role in ("ceo", "cfo", "executive")]
        elif self.speaker_mode == "full_transcript":
            return segments
        else:
            raise ValueError(f"Unknown speaker_mode: {self.speaker_mode}")

    def _extract_category_features(
        self,
        text: str,
        keywords: List[str] | dict,
        category_name: str,
    ) -> Dict:
        """Extract features for a keyword category."""
        features = {}

        # Handle both list and dict formats
        if isinstance(keywords, list):
            keyword_list = keywords
        elif isinstance(keywords, dict):
            # Dict format with variants
            keyword_list = []
            for base_phrase, variants in keywords.items():
                keyword_list.append(base_phrase)
                if self.phrase_mode == "variants" and variants:
                    keyword_list.extend(variants)
        else:
            keyword_list = []

        # Count mentions
        total_count = 0
        mentioned = 0

        for keyword in keyword_list:
            # Use word boundary matching
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
            count = len(re.findall(pattern, text))

            total_count += count
            if count > 0:
                mentioned = 1

        features[f"{category_name}_count"] = total_count
        features[f"{category_name}_mentioned"] = mentioned

        return features

    def _extract_speaker_features(self, segments: List[SpeakerSegment]) -> Dict:
        """Extract speaker-specific features."""
        features = {}

        # Count words by speaker role
        for role in ["ceo", "cfo", "executive", "analyst"]:
            role_segments = [seg for seg in segments if seg.role == role]
            role_text = " ".join(seg.text for seg in role_segments)
            role_word_count = len(role_text.split())

            features[f"{role}_word_count"] = role_word_count
            features[f"{role}_segments"] = len(role_segments)

        # CEO to CFO ratio (for balance analysis)
        ceo_words = features.get("ceo_word_count", 0)
        cfo_words = features.get("cfo_word_count", 0)

        if cfo_words > 0:
            features["ceo_cfo_ratio"] = ceo_words / cfo_words
        else:
            features["ceo_cfo_ratio"] = 0.0

        # Q&A participation (analyst questions count)
        features["qa_questions"] = features.get("analyst_segments", 0)

        return features

    def featurize_multiple(
        self,
        segments_dir: Path,
        ticker: str,
    ) -> pd.DataFrame:
        """
        Featurize multiple transcripts for a ticker.

        Parameters
        ----------
        segments_dir : Path
            Directory containing segment files (JSONL format)
        ticker : str
            Stock ticker

        Returns
        -------
        pd.DataFrame
            Features dataframe with one row per earnings call
        """
        import json

        segment_files = sorted(Path(segments_dir).glob(f"{ticker}_*.jsonl"))

        if not segment_files:
            print(f"No segment files found for {ticker} in {segments_dir}")
            return pd.DataFrame()

        features_list = []

        for segment_file in segment_files:
            # Extract call date from filename
            # Expected format: TICKER_YYYY-MM-DD.jsonl
            filename = segment_file.stem
            parts = filename.split("_")
            if len(parts) >= 2:
                call_date = parts[1]
            else:
                call_date = "unknown"

            # Load segments
            segments = []
            with open(segment_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    segment = SpeakerSegment(**data)
                    segments.append(segment)

            # Featurize
            features = self.featurize(segments, ticker, call_date)
            features_list.append(features)

        df = pd.DataFrame(features_list)
        df = df.set_index("call_date")
        df = df.sort_index()

        return df


def featurize_earnings_calls(
    segments_dir: Path,
    ticker: str,
    keywords_config: Path | dict,
    speaker_mode: str = "executives_only",
    phrase_mode: str = "variants",
) -> pd.DataFrame:
    """
    Convenience function to featurize earnings calls.

    Parameters
    ----------
    segments_dir : Path
        Directory containing segment files
    ticker : str
        Stock ticker
    keywords_config : Path or dict
        Keywords configuration
    speaker_mode : str
        Speaker filtering mode
    phrase_mode : str
        Phrase matching mode

    Returns
    -------
    pd.DataFrame
        Features dataframe
    """
    featurizer = EarningsFeaturizer(
        keywords_config=keywords_config,
        speaker_mode=speaker_mode,
        phrase_mode=phrase_mode,
    )

    return featurizer.featurize_multiple(segments_dir, ticker)
