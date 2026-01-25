"""
End-to-end transcript processing pipeline.

This module provides:
1. Fetching raw transcripts from various sources
2. Parsing and cleaning transcript text
3. Speaker segmentation with role identification
4. Output to JSONL format for model training
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from earnings_analysis.fetchers.transcript_fetcher import (
    TranscriptFetcher,
    TranscriptMetadata,
)
from earnings_analysis.parsing.speaker_segmenter import (
    EarningsSpeakerSegmenter,
    SpeakerSegment,
)


@dataclass
class ProcessedTranscript:
    """A fully processed earnings call transcript."""
    ticker: str
    call_date: str  # YYYY-MM-DD
    fiscal_quarter: str
    source: str
    segments: List[SpeakerSegment]
    word_counts: Dict[str, int]  # Word -> count (executives only)
    metadata: Dict[str, Any]


class TranscriptProcessor:
    """
    End-to-end transcript processing pipeline.

    This processor:
    1. Fetches raw transcripts (from cache or SEC EDGAR)
    2. Cleans and normalizes text
    3. Segments by speaker
    4. Counts word mentions
    5. Outputs JSONL files

    Parameters
    ----------
    output_dir : Path
        Directory for processed transcripts
    cache_dir : Path
        Directory for caching raw transcripts
    use_ai_segmentation : bool
        Whether to use OpenAI for segmentation
    """

    def __init__(
        self,
        output_dir: Path,
        cache_dir: Optional[Path] = None,
        use_ai_segmentation: bool = False,
        openai_api_key: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/transcript_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.fetcher = TranscriptFetcher(
            output_dir=self.cache_dir,
            rate_limit_delay=1.0,
        )

        self.segmenter = EarningsSpeakerSegmenter(
            use_ai=use_ai_segmentation,
            openai_api_key=openai_api_key,
        )

    def process_ticker(
        self,
        ticker: str,
        num_quarters: int = 8,
        words_to_track: Optional[List[str]] = None,
        force_refetch: bool = False,
    ) -> List[ProcessedTranscript]:
        """
        Process transcripts for a ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        num_quarters : int
            Number of quarters to process
        words_to_track : Optional[List[str]]
            Words to count (for word_counts field)
        force_refetch : bool
            Whether to refetch transcripts even if cached

        Returns
        -------
        List[ProcessedTranscript]
            Processed transcripts
        """
        ticker = ticker.upper()
        words_to_track = words_to_track or []

        print(f"\nProcessing transcripts for {ticker}...")

        # Check for cached processed transcripts
        cached_file = self.output_dir / ticker / "processed_transcripts.json"
        if not force_refetch and cached_file.exists():
            print(f"Loading cached processed transcripts for {ticker}")
            return self._load_processed_transcripts(cached_file)

        # Fetch raw transcripts
        metadata_list = self._fetch_or_load_raw(ticker, num_quarters, force_refetch)

        if not metadata_list:
            print(f"No transcripts found for {ticker}")
            return []

        # Process each transcript
        processed = []
        for metadata in metadata_list:
            try:
                transcript = self._process_single_transcript(
                    metadata, words_to_track
                )
                if transcript:
                    processed.append(transcript)
            except Exception as e:
                print(f"Error processing transcript for {metadata.fiscal_quarter}: {e}")
                continue

        # Save processed transcripts
        if processed:
            self._save_processed_transcripts(processed, cached_file)

        return processed

    def _fetch_or_load_raw(
        self,
        ticker: str,
        num_quarters: int,
        force_refetch: bool,
    ) -> List[TranscriptMetadata]:
        """Fetch raw transcripts or load from cache."""
        # Check for cached metadata
        metadata_file = self.cache_dir / ticker / "transcript_metadata.json"

        if not force_refetch and metadata_file.exists():
            print(f"Loading cached raw transcripts for {ticker}")
            return self._load_metadata(metadata_file)

        # Fetch from source
        print(f"Fetching raw transcripts for {ticker}...")
        metadata_list = self.fetcher.fetch_ticker(
            ticker=ticker,
            num_quarters=num_quarters,
            source="auto",
        )

        # Save metadata
        if metadata_list:
            self.fetcher.save_metadata(metadata_list, metadata_file)

        return metadata_list

    def _process_single_transcript(
        self,
        metadata: TranscriptMetadata,
        words_to_track: List[str],
    ) -> Optional[ProcessedTranscript]:
        """Process a single transcript."""
        if not metadata.has_transcript or not metadata.file_path:
            return None

        # Load raw text
        transcript_path = Path(metadata.file_path)
        if not transcript_path.exists():
            return None

        raw_text = transcript_path.read_text()

        # Clean text
        cleaned_text = self._clean_transcript_text(raw_text)

        if not cleaned_text or len(cleaned_text) < 500:
            print(f"Transcript too short for {metadata.fiscal_quarter}, skipping")
            return None

        # Segment by speaker
        segments = self.segmenter.segment(cleaned_text, metadata.ticker)

        if not segments:
            # Fallback: treat entire text as single executive segment
            segments = [
                SpeakerSegment(
                    speaker="Unknown Executive",
                    role="executive",
                    text=cleaned_text,
                    confidence=0.5,
                )
            ]

        # Count words (executives only)
        word_counts = self._count_words(segments, words_to_track)

        # Create processed transcript
        call_date = metadata.call_date.strftime("%Y-%m-%d")

        return ProcessedTranscript(
            ticker=metadata.ticker,
            call_date=call_date,
            fiscal_quarter=metadata.fiscal_quarter,
            source=metadata.source,
            segments=segments,
            word_counts=word_counts,
            metadata={
                "eps_actual": metadata.eps_actual,
                "eps_estimate": metadata.eps_estimate,
                "revenue_actual": metadata.revenue_actual,
                "revenue_estimate": metadata.revenue_estimate,
                "url": metadata.url,
            },
        )

    def _clean_transcript_text(self, raw_text: str) -> str:
        """Clean raw transcript text."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", raw_text)

        # Remove common boilerplate patterns
        boilerplate_patterns = [
            r"LEGAL DISCLAIMER.*?(?=\n\n)",
            r"FORWARD[ -]LOOKING STATEMENTS.*?(?=\n\n)",
            r"©.*?\d{4}",
            r"All rights reserved\.?",
        ]

        for pattern in boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

        # Remove HTML artifacts if any
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)

        return text.strip()

    def _count_words(
        self,
        segments: List[SpeakerSegment],
        words_to_track: List[str],
    ) -> Dict[str, int]:
        """Count word mentions by executives."""
        # Filter to executive segments
        exec_segments = [
            s for s in segments
            if s.role in ("ceo", "cfo", "executive")
        ]

        # Combine text
        combined_text = " ".join(s.text for s in exec_segments).lower()

        # Count each word
        counts = {}
        for word in words_to_track:
            # Handle multi-word patterns (e.g., "VR / Virtual Reality")
            variants = [w.strip() for w in word.split("/")]

            total_count = 0
            for variant in variants:
                # Word boundary matching
                pattern = r"\b" + re.escape(variant.lower()) + r"\b"
                total_count += len(re.findall(pattern, combined_text))

            counts[word.lower()] = total_count

        return counts

    def save_segments_jsonl(
        self,
        processed: ProcessedTranscript,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save transcript segments to JSONL format.

        Parameters
        ----------
        processed : ProcessedTranscript
            Processed transcript
        output_dir : Optional[Path]
            Output directory (default: self.output_dir/ticker/segments)

        Returns
        -------
        Path
            Path to saved JSONL file
        """
        output_dir = output_dir or (self.output_dir / processed.ticker / "segments")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{processed.ticker}_{processed.call_date}.jsonl"

        with open(output_file, "w") as f:
            for i, segment in enumerate(processed.segments):
                record = {
                    "speaker": segment.speaker,
                    "role": segment.role,
                    "text": segment.text,
                    "segment_idx": i,
                    "confidence": segment.confidence,
                }
                if segment.company:
                    record["company"] = segment.company
                f.write(json.dumps(record) + "\n")

        return output_file

    def export_to_training_format(
        self,
        processed_transcripts: List[ProcessedTranscript],
        words_to_track: List[str],
        output_dir: Optional[Path] = None,
    ) -> tuple[Path, Path]:
        """
        Export processed transcripts to training format.

        Creates:
        1. features.csv - Word counts per call (Index=call_date, Columns=words)
        2. segments/*.jsonl - Transcript segments per call

        Parameters
        ----------
        processed_transcripts : List[ProcessedTranscript]
            Processed transcripts
        words_to_track : List[str]
            Words to include in features
        output_dir : Optional[Path]
            Output directory

        Returns
        -------
        tuple[Path, Path]
            (features_csv_path, segments_dir_path)
        """
        if not processed_transcripts:
            raise ValueError("No processed transcripts to export")

        ticker = processed_transcripts[0].ticker
        output_dir = output_dir or (self.output_dir / ticker / "training")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build features DataFrame
        features_data = {}
        for transcript in processed_transcripts:
            call_date = transcript.call_date

            # Get word counts (recount if needed)
            counts = transcript.word_counts
            if not counts and words_to_track:
                counts = self._count_words(transcript.segments, words_to_track)

            features_data[call_date] = {
                word.lower(): counts.get(word.lower(), 0)
                for word in words_to_track
            }

        import pandas as pd
        features_df = pd.DataFrame.from_dict(features_data, orient="index")
        features_df.index = pd.to_datetime(features_df.index)
        features_df = features_df.sort_index()

        features_path = output_dir / "features.csv"
        features_df.to_csv(features_path)
        print(f"Saved features to {features_path}")

        # Save segments
        segments_dir = output_dir / "segments"
        for transcript in processed_transcripts:
            self.save_segments_jsonl(transcript, segments_dir)

        print(f"Saved segments to {segments_dir}")

        return features_path, segments_dir

    def _load_metadata(self, path: Path) -> List[TranscriptMetadata]:
        """Load transcript metadata from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        metadata_list = []
        for item in data:
            # Convert call_date back to datetime
            if isinstance(item.get("call_date"), str):
                item["call_date"] = datetime.fromisoformat(item["call_date"])
            metadata_list.append(TranscriptMetadata(**item))

        return metadata_list

    def _save_processed_transcripts(
        self,
        processed: List[ProcessedTranscript],
        path: Path,
    ) -> None:
        """Save processed transcripts to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for transcript in processed:
            data.append({
                "ticker": transcript.ticker,
                "call_date": transcript.call_date,
                "fiscal_quarter": transcript.fiscal_quarter,
                "source": transcript.source,
                "segments": [asdict(s) for s in transcript.segments],
                "word_counts": transcript.word_counts,
                "metadata": transcript.metadata,
            })

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_processed_transcripts(self, path: Path) -> List[ProcessedTranscript]:
        """Load processed transcripts from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        processed = []
        for item in data:
            segments = [
                SpeakerSegment(**s)
                for s in item.get("segments", [])
            ]
            processed.append(ProcessedTranscript(
                ticker=item["ticker"],
                call_date=item["call_date"],
                fiscal_quarter=item["fiscal_quarter"],
                source=item["source"],
                segments=segments,
                word_counts=item.get("word_counts", {}),
                metadata=item.get("metadata", {}),
            ))

        return processed


def process_earnings_transcripts(
    ticker: str,
    num_quarters: int = 8,
    words_to_track: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    use_ai_segmentation: bool = False,
) -> List[ProcessedTranscript]:
    """
    Convenience function to process earnings transcripts.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    num_quarters : int
        Number of quarters to process
    words_to_track : Optional[List[str]]
        Words to count
    output_dir : Optional[Path]
        Output directory
    use_ai_segmentation : bool
        Whether to use AI for segmentation

    Returns
    -------
    List[ProcessedTranscript]
        Processed transcripts
    """
    processor = TranscriptProcessor(
        output_dir=output_dir or Path("data/processed_transcripts"),
        use_ai_segmentation=use_ai_segmentation,
    )

    return processor.process_ticker(
        ticker=ticker,
        num_quarters=num_quarters,
        words_to_track=words_to_track,
    )
