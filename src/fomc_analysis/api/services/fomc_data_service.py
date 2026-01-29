"""FOMC data service for transcripts and word frequencies."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from fomc_analysis.data_loader import Transcript, load_transcripts
from fomc_analysis.parsing.speaker_segmenter import load_segments_jsonl, SpeakerTurn
from fomc_analysis.featurizer import (
    build_feature_matrix,
    load_contract_phrases,
    count_phrase_mentions,
    FeatureConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class TranscriptInfo:
    """Summary information about a transcript."""

    meeting_date: str
    total_segments: int
    powell_segments: int
    word_count: int
    available: bool = True


class FOMCDataService:
    """Service for accessing FOMC transcript data and word frequencies."""

    def __init__(
        self,
        transcripts_dir: Path = Path("data/transcripts"),
        segments_dir: Path = Path("data/segments"),
        contract_mapping_file: Path = Path("configs/contract_mapping.yaml"),
        word_freq_file: Path = Path("results/backtest_v3/word_frequency_timeseries.csv"),
    ):
        self.transcripts_dir = Path(transcripts_dir)
        self.segments_dir = Path(segments_dir)
        self.contract_mapping_file = Path(contract_mapping_file)
        self.word_freq_file = Path(word_freq_file)

        self._transcripts: Dict[str, Transcript] = {}
        self._segments: Dict[str, List[SpeakerTurn]] = {}
        self._contract_phrases: Dict[str, List[str]] = {}
        self._word_frequencies: Optional[pd.DataFrame] = None

    def _load_contract_phrases(self) -> Dict[str, List[str]]:
        """Load contract phrase mappings from YAML config."""
        if self._contract_phrases:
            return self._contract_phrases

        if not self.contract_mapping_file.exists():
            logger.warning("Contract mapping file not found: %s", self.contract_mapping_file)
            return {}

        try:
            data = yaml.safe_load(self.contract_mapping_file.read_text())
            for contract, entry in data.items():
                synonyms = entry.get("synonyms", [])
                self._contract_phrases[contract] = [s.lower() for s in synonyms]
            logger.info("Loaded %d contract phrases", len(self._contract_phrases))
        except Exception as e:
            logger.error("Failed to load contract mapping: %s", e)

        return self._contract_phrases

    def _load_segments(self, meeting_date: str) -> List[SpeakerTurn]:
        """Load speaker segments for a specific meeting."""
        if meeting_date in self._segments:
            return self._segments[meeting_date]

        segment_file = self.segments_dir / f"{meeting_date.replace('-', '')}.jsonl"
        if not segment_file.exists():
            logger.debug("Segment file not found: %s", segment_file)
            return []

        try:
            segments = load_segments_jsonl(segment_file)
            self._segments[meeting_date] = segments
            return segments
        except Exception as e:
            logger.error("Failed to load segments for %s: %s", meeting_date, e)
            return []

    def get_available_transcripts(self) -> List[TranscriptInfo]:
        """Get list of available transcripts with summary info."""
        transcripts = []

        for segment_file in sorted(self.segments_dir.glob("*.jsonl")):
            date_str = segment_file.stem
            meeting_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

            try:
                segments = load_segments_jsonl(segment_file)
                powell_segments = [s for s in segments if getattr(s, "role", "") == "powell"]
                powell_word_count = sum(
                    len(getattr(s, "text", "").split()) for s in powell_segments
                )

                transcripts.append(TranscriptInfo(
                    meeting_date=meeting_date,
                    total_segments=len(segments),
                    powell_segments=len(powell_segments),
                    word_count=powell_word_count,
                    available=True,
                ))
            except Exception as e:
                logger.warning("Failed to read %s: %s", segment_file, e)
                transcripts.append(TranscriptInfo(
                    meeting_date=meeting_date,
                    total_segments=0,
                    powell_segments=0,
                    word_count=0,
                    available=False,
                ))

        transcripts.sort(key=lambda t: t.meeting_date, reverse=True)
        return transcripts

    def get_transcript(self, meeting_date: str) -> Optional[Dict[str, Any]]:
        """Get full transcript for a specific meeting date."""
        segments = self._load_segments(meeting_date)
        if not segments:
            return None

        powell_segments = [s for s in segments if getattr(s, "role", "") == "powell"]
        powell_word_count = sum(
            len(getattr(s, "text", "").split()) for s in powell_segments
        )

        return {
            "meeting_date": meeting_date,
            "segments": [
                {
                    "segment_idx": i,
                    "speaker": getattr(s, "speaker", "Unknown"),
                    "role": getattr(s, "role", "unknown"),
                    "text": getattr(s, "text", ""),
                }
                for i, s in enumerate(segments)
            ],
            "total_segments": len(segments),
            "powell_word_count": powell_word_count,
        }

    def get_word_frequencies(
        self,
        words: Optional[List[str]] = None,
        meeting_dates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get word frequency data across meetings."""
        if self._word_frequencies is None:
            self._word_frequencies = self._build_word_frequencies()

        df = self._word_frequencies
        if df.empty:
            return {
                "words": [],
                "meeting_dates": [],
                "total_meetings": 0,
            }

        if words:
            available_words = [w for w in words if w in df.columns]
            if available_words:
                df = df[available_words]

        if meeting_dates:
            available_dates = [d for d in meeting_dates if d in df.index]
            if available_dates:
                df = df.loc[available_dates]

        result_words = []
        for word in df.columns:
            series = df[word]
            frequencies = [
                {
                    "meeting_date": str(date),
                    "word": word,
                    "count": int(count) if pd.notna(count) else 0,
                    "mentioned": bool(count > 0) if pd.notna(count) else False,
                }
                for date, count in series.items()
            ]
            total_mentions = int(series.sum()) if series.notna().any() else 0
            mention_rate = float(
                (series > 0).sum() / len(series)
            ) if len(series) > 0 else 0.0

            result_words.append({
                "word": word,
                "frequencies": frequencies,
                "total_mentions": total_mentions,
                "mention_rate": mention_rate,
            })

        return {
            "words": result_words,
            "meeting_dates": [str(d) for d in df.index.tolist()],
            "total_meetings": len(df),
        }

    def _build_word_frequencies(self) -> pd.DataFrame:
        """Build word frequency matrix from segments and contract mapping."""
        if self.word_freq_file.exists():
            try:
                df = pd.read_csv(self.word_freq_file, index_col=0)
                logger.info("Loaded word frequencies from %s", self.word_freq_file)
                return df
            except Exception as e:
                logger.warning("Failed to load word freq file: %s", e)

        if not self.segments_dir.exists():
            logger.warning("Segments directory not found: %s", self.segments_dir)
            return pd.DataFrame()

        contract_phrases = self._load_contract_phrases()
        if not contract_phrases:
            return pd.DataFrame()

        config = FeatureConfig(speaker_mode="powell_only", phrase_mode="strict")

        try:
            df = build_feature_matrix(self.segments_dir, contract_phrases, config)
            mention_cols = [c for c in df.columns if c.endswith("_mentioned")]
            count_cols = [c for c in df.columns if c.endswith("_count")]

            if count_cols:
                rename_map = {c: c.replace("_count", "") for c in count_cols}
                df = df[count_cols].rename(columns=rename_map)
            elif mention_cols:
                rename_map = {c: c.replace("_mentioned", "") for c in mention_cols}
                df = df[mention_cols].rename(columns=rename_map)

            return df
        except Exception as e:
            logger.error("Failed to build word frequencies: %s", e)
            return pd.DataFrame()

    def get_backtest_results(self) -> Optional[Dict[str, Any]]:
        """Load backtest results from disk."""
        backtest_file = Path("results/backtest_v3/backtest_results.json")

        if not backtest_file.exists():
            backtest_file = Path("data/backtest_results/fomc/backtest_results.json")

        if not backtest_file.exists():
            logger.warning("No FOMC backtest results found")
            return None

        try:
            return json.loads(backtest_file.read_text())
        except Exception as e:
            logger.error("Failed to load backtest results: %s", e)
            return None
