"""
Fetch real earnings call data from SEC EDGAR.

SEC EDGAR provides:
- 8-K filings with Item 2.02 (Results of Operations and Financial Condition)
- Press releases attached as exhibits to 8-K filings
- Filing dates that correspond to earnings announcement dates

Full earnings call transcripts are NOT typically filed with the SEC.
This module fetches what IS available: earnings dates, press releases,
and any conference call transcripts that are filed as exhibits.

For actual word-level transcript analysis, transcripts must be sourced from:
- Company investor relations pages
- Financial Modeling Prep API (free tier available)
- Paid services (Seeking Alpha Premium, Refinitiv, FactSet)
- Manual collection

This module also provides utilities for ingesting manually-sourced transcripts
into the pipeline format required by the backtester.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ..parsing.speaker_segmenter import SpeakerSegment, EarningsSpeakerSegmenter
from ..parsing.transcript_parser import TranscriptParser


SEC_EDGAR_BASE = "https://efts.sec.gov"
SEC_DATA_BASE = "https://data.sec.gov"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# SEC requires a valid User-Agent header with contact info
SEC_USER_AGENT = (
    "EarningsAnalyzer/1.0 (earnings-analysis-research; "
    "contact: research@example.com)"
)


@dataclass
class EarningsFilingInfo:
    """Metadata about an earnings-related SEC filing."""

    ticker: str
    cik: str
    company_name: str
    filing_date: str  # YYYY-MM-DD
    form_type: str  # 8-K, 8-K/A
    accession_number: str
    filing_url: str
    items_reported: List[str]  # e.g. ["2.02", "7.01", "9.01"]
    has_press_release: bool = False
    press_release_url: Optional[str] = None
    has_transcript: bool = False
    transcript_url: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TranscriptRecord:
    """A processed transcript ready for the analysis pipeline."""

    ticker: str
    call_date: str  # YYYY-MM-DD
    quarter: str  # e.g. "Q4 2025"
    source: str  # "sec_edgar", "manual", "fmp_api"
    raw_text: str
    segments: List[SpeakerSegment]
    file_path: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # SpeakerSegment -> dict
        d["segments"] = [asdict(s) for s in self.segments]
        return d


class SECEdgarFetcher:
    """
    Fetch earnings-related data from SEC EDGAR.

    Parameters
    ----------
    rate_limit : float
        Seconds between requests (SEC requires <=10 req/sec, we default to 0.15).
    output_dir : Path
        Directory to save downloaded filings.
    """

    def __init__(
        self,
        rate_limit: float = 0.15,
        output_dir: Path = Path("data/earnings/sec_filings"),
    ):
        self.rate_limit = rate_limit
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        })
        self._last_request_time = 0.0

    def _throttle(self):
        """Rate-limit requests to SEC EDGAR."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str) -> requests.Response:
        """Make a throttled GET request."""
        self._throttle()
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp

    def get_cik(self, ticker: str) -> Optional[str]:
        """Look up CIK (Central Index Key) for a stock ticker."""
        url = f"{SEC_DATA_BASE}/submissions/CIK-lookup.json"
        try:
            # Use the company tickers JSON endpoint
            resp = self._get("https://www.sec.gov/files/company_tickers.json")
            data = resp.json()

            for entry in data.values():
                if entry["ticker"].upper() == ticker.upper():
                    return str(entry["cik_str"]).zfill(10)
        except Exception:
            pass

        # Fallback: try the EFTS full-text search
        try:
            url = f"{SEC_EDGAR_BASE}/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2024-01-01&forms=8-K"
            resp = self._get(url)
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            if hits:
                cik = hits[0].get("_source", {}).get("entity_id", "")
                if cik:
                    return str(cik).zfill(10)
        except Exception:
            pass

        return None

    def fetch_earnings_filings(
        self,
        ticker: str,
        num_quarters: int = 8,
    ) -> List[EarningsFilingInfo]:
        """
        Fetch recent earnings-related 8-K filings for a ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        num_quarters : int
            Number of recent quarters to search.

        Returns
        -------
        List[EarningsFilingInfo]
            Earnings filing information.
        """
        cik = self.get_cik(ticker)
        if not cik:
            print(f"Could not find CIK for {ticker}")
            return []

        print(f"Fetching 8-K filings for {ticker} (CIK: {cik})...")

        # Fetch company submissions
        url = f"{SEC_DATA_BASE}/submissions/CIK{cik}.json"
        try:
            resp = self._get(url)
            data = resp.json()
        except Exception as e:
            print(f"Error fetching submissions: {e}")
            return []

        company_name = data.get("name", ticker)
        filings_data = data.get("filings", {}).get("recent", {})

        forms = filings_data.get("form", [])
        dates = filings_data.get("filingDate", [])
        accessions = filings_data.get("accessionNumber", [])
        primary_docs = filings_data.get("primaryDocument", [])
        items_list = filings_data.get("items", [])

        earnings_filings = []

        for i, form in enumerate(forms):
            if len(earnings_filings) >= num_quarters:
                break

            # Only look at 8-K and 8-K/A filings
            if form not in ("8-K", "8-K/A"):
                continue

            # Check if this is an earnings 8-K (Item 2.02)
            items_str = items_list[i] if i < len(items_list) else ""
            items = [item.strip() for item in items_str.split(",") if item.strip()]

            # Item 2.02 = Results of Operations and Financial Condition
            # Item 7.01 = Regulation FD Disclosure
            # Item 9.01 = Financial Statements and Exhibits
            has_earnings_item = any(
                item in ("2.02", "7.01") for item in items
            )
            if not has_earnings_item:
                continue

            filing_date = dates[i] if i < len(dates) else ""
            accession = accessions[i] if i < len(accessions) else ""
            primary_doc = primary_docs[i] if i < len(primary_docs) else ""

            # Build filing URL
            accession_clean = accession.replace("-", "")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik.lstrip('0')}/{accession_clean}/{primary_doc}"
            )

            info = EarningsFilingInfo(
                ticker=ticker,
                cik=cik,
                company_name=company_name,
                filing_date=filing_date,
                form_type=form,
                accession_number=accession,
                filing_url=filing_url,
                items_reported=items,
            )

            # Try to find press release or transcript in exhibits
            self._check_exhibits(info, cik, accession_clean)

            earnings_filings.append(info)
            print(
                f"  Found: {form} on {filing_date} "
                f"(Items: {', '.join(items)}) "
                f"{'[has PR]' if info.has_press_release else ''}"
                f"{'[has transcript]' if info.has_transcript else ''}"
            )

        return earnings_filings

    def _check_exhibits(
        self,
        info: EarningsFilingInfo,
        cik: str,
        accession_clean: str,
    ):
        """Check filing exhibits for press releases or transcripts."""
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik.lstrip('0')}/{accession_clean}/index.json"
        )

        try:
            resp = self._get(index_url)
            index_data = resp.json()
        except Exception:
            return

        directory = index_data.get("directory", {})
        items = directory.get("item", [])

        for item in items:
            name = item.get("name", "").lower()
            doc_type = item.get("type", "").lower()

            # Press releases are usually EX-99.1
            if "ex-99" in name or "exhibit" in name:
                exhibit_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik.lstrip('0')}/{accession_clean}/{item['name']}"
                )

                # Check if it's a press release or transcript
                if any(ext in name for ext in [".htm", ".html", ".txt"]):
                    info.has_press_release = True
                    info.press_release_url = exhibit_url

                    # Some companies file full transcripts as exhibits
                    if "transcript" in name or "conference" in name:
                        info.has_transcript = True
                        info.transcript_url = exhibit_url

    def download_press_release(self, filing: EarningsFilingInfo) -> Optional[str]:
        """Download and extract text from a press release exhibit."""
        if not filing.press_release_url:
            return None

        try:
            resp = self._get(filing.press_release_url)
            content_type = resp.headers.get("content-type", "")

            if "html" in content_type:
                soup = BeautifulSoup(resp.text, "html.parser")
                # Remove script and style elements
                for tag in soup(["script", "style"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
            else:
                text = resp.text

            return text

        except Exception as e:
            print(f"Error downloading press release: {e}")
            return None

    def get_earnings_dates(
        self,
        ticker: str,
        num_quarters: int = 8,
    ) -> List[str]:
        """
        Get historical earnings announcement dates from SEC filings.

        Returns list of YYYY-MM-DD date strings.
        """
        filings = self.fetch_earnings_filings(ticker, num_quarters)
        return [f.filing_date for f in filings if f.filing_date]


class ManualTranscriptIngester:
    """
    Ingest manually-sourced earnings call transcripts into the pipeline.

    Supports:
    - Plain text files (.txt)
    - JSON files with structured speaker data
    - Raw copy-paste from transcript services

    Expected directory structure:
        data/earnings/transcripts/
            META/
                META_2025-01-29.txt    # Plain text
                META_2025-04-30.json   # Structured JSON
            TSLA/
                TSLA_2025-01-29.txt
    """

    def __init__(
        self,
        transcripts_dir: Path = Path("data/earnings/transcripts"),
        segments_dir: Path = Path("data/earnings/segments"),
    ):
        self.transcripts_dir = Path(transcripts_dir)
        self.segments_dir = Path(segments_dir)
        self.parser = TranscriptParser()
        self.segmenter = EarningsSpeakerSegmenter(use_ai=False)

    def ingest_transcript(
        self,
        ticker: str,
        call_date: str,
        text: str,
        quarter: Optional[str] = None,
        source: str = "manual",
    ) -> TranscriptRecord:
        """
        Process a single transcript and save it in pipeline format.

        Parameters
        ----------
        ticker : str
            Stock ticker.
        call_date : str
            Earnings call date (YYYY-MM-DD).
        text : str
            Raw transcript text.
        quarter : str, optional
            Fiscal quarter (e.g. "Q4 2025").
        source : str
            Source of the transcript.

        Returns
        -------
        TranscriptRecord
            Processed transcript.
        """
        # Clean the text
        cleaned = self.parser.parse(text)

        # Segment by speaker
        segments = self.segmenter.segment(cleaned, ticker=ticker)

        # Infer quarter if not provided
        if not quarter:
            quarter = self._infer_quarter(call_date)

        record = TranscriptRecord(
            ticker=ticker,
            call_date=call_date,
            quarter=quarter,
            source=source,
            raw_text=cleaned,
            segments=segments,
        )

        # Save segments as JSONL
        segments_path = self.segments_dir / ticker
        segments_path.mkdir(parents=True, exist_ok=True)
        output_file = segments_path / f"{ticker}_{call_date}.jsonl"

        with open(output_file, "w") as f:
            for seg in segments:
                f.write(json.dumps(asdict(seg)) + "\n")

        record.file_path = str(output_file)
        print(f"Saved {len(segments)} segments to {output_file}")

        return record

    def ingest_directory(self, ticker: str) -> List[TranscriptRecord]:
        """
        Ingest all transcripts for a ticker from the transcripts directory.

        Looks for files matching: {ticker}_{date}.txt or {ticker}_{date}.json
        """
        ticker_dir = self.transcripts_dir / ticker
        if not ticker_dir.exists():
            print(f"No transcript directory found at {ticker_dir}")
            return []

        records = []
        files = sorted(ticker_dir.glob(f"{ticker}_*.txt")) + sorted(
            ticker_dir.glob(f"{ticker}_*.json")
        )

        for filepath in files:
            # Extract date from filename
            stem = filepath.stem  # e.g. META_2025-01-29
            parts = stem.split("_", 1)
            if len(parts) < 2:
                continue
            call_date = parts[1]

            print(f"\nProcessing {filepath.name}...")

            if filepath.suffix == ".json":
                # Structured JSON format
                with open(filepath) as f:
                    data = json.load(f)
                text = data.get("text", data.get("transcript", ""))
                quarter = data.get("quarter")
                source = data.get("source", "manual")
            else:
                text = filepath.read_text(encoding="utf-8", errors="ignore")
                quarter = None
                source = "manual"

            record = self.ingest_transcript(
                ticker=ticker,
                call_date=call_date,
                text=text,
                quarter=quarter,
                source=source,
            )
            records.append(record)

        print(f"\nIngested {len(records)} transcripts for {ticker}")
        return records

    def count_word_mentions(
        self,
        ticker: str,
        words: List[str],
        speaker_mode: str = "executives_only",
    ) -> pd.DataFrame:
        """
        Count word mentions across all ingested transcripts for a ticker.

        Returns a DataFrame with index=call_date, columns=words, values=1/0
        (whether the word was mentioned by the specified speakers).
        """
        import pandas as pd

        segments_dir = self.segments_dir / ticker
        if not segments_dir.exists():
            return pd.DataFrame()

        segment_files = sorted(segments_dir.glob(f"{ticker}_*.jsonl"))
        if not segment_files:
            return pd.DataFrame()

        rows = []
        for filepath in segment_files:
            # Extract date
            stem = filepath.stem
            parts = stem.split("_", 1)
            call_date = parts[1] if len(parts) >= 2 else "unknown"

            # Load segments
            segments = []
            with open(filepath) as f:
                for line in f:
                    segments.append(json.loads(line))

            # Filter by speaker mode
            if speaker_mode == "executives_only":
                segments = [
                    s for s in segments
                    if s.get("role") in ("ceo", "cfo", "executive")
                ]
            elif speaker_mode == "ceo_only":
                segments = [s for s in segments if s.get("role") == "ceo"]

            # Combine text
            combined = " ".join(s.get("text", "") for s in segments).lower()

            # Count mentions for each word
            row = {"call_date": call_date}
            for word in words:
                # Handle slash-separated variants (e.g. "VR / Virtual Reality")
                variants = [v.strip().lower() for v in word.split("/")]
                mentioned = 0
                for variant in variants:
                    if variant and re.search(
                        r"\b" + re.escape(variant) + r"\b",
                        combined,
                    ):
                        mentioned = 1
                        break
                row[word.lower()] = mentioned

            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index("call_date").sort_index()

        return df

    def _infer_quarter(self, call_date: str) -> str:
        """Infer fiscal quarter from earnings call date."""
        try:
            dt = datetime.strptime(call_date, "%Y-%m-%d")
        except ValueError:
            return "Unknown"

        month = dt.month
        year = dt.year

        # Earnings calls typically report on the previous quarter
        if month <= 2:
            return f"Q4 {year - 1}"
        elif month <= 5:
            return f"Q1 {year}"
        elif month <= 8:
            return f"Q2 {year}"
        else:
            return f"Q3 {year}"


# Import here to avoid circular imports
import pandas as pd


def fetch_earnings_dates_from_sec(
    ticker: str,
    num_quarters: int = 8,
) -> List[str]:
    """Convenience: fetch earnings dates from SEC EDGAR."""
    fetcher = SECEdgarFetcher()
    return fetcher.get_earnings_dates(ticker, num_quarters)
