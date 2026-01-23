"""
Fetch earnings call transcripts from multiple sources.

Supports:
- SEC EDGAR (8-K filings)
- Alpha Vantage API
- Web scraping (Seeking Alpha, Motley Fool) - when legally permitted
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


@dataclass
class TranscriptMetadata:
    """Metadata for an earnings call transcript."""
    ticker: str
    company_name: str
    fiscal_quarter: str  # e.g., "Q4 2024"
    fiscal_year: int
    call_date: datetime
    source: str  # "sec_edgar", "alpha_vantage", "seeking_alpha"
    url: str
    file_path: Optional[str] = None
    has_transcript: bool = False
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None


class TranscriptFetcher:
    """
    Fetch earnings call transcripts from multiple sources.

    Parameters
    ----------
    output_dir : Path
        Directory to save transcripts
    alpha_vantage_key : Optional[str]
        Alpha Vantage API key (if using that source)
    rate_limit_delay : float
        Delay between requests (seconds) to respect rate limits
    """

    def __init__(
        self,
        output_dir: Path,
        alpha_vantage_key: Optional[str] = None,
        rate_limit_delay: float = 1.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.alpha_vantage_key = alpha_vantage_key
        self.rate_limit_delay = rate_limit_delay

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; EarningsAnalyzer/1.0)"
        })

    def fetch_ticker(
        self,
        ticker: str,
        num_quarters: int = 8,
        source: str = "auto",
    ) -> List[TranscriptMetadata]:
        """
        Fetch transcripts for a ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g., "COIN", "GOOGL")
        num_quarters : int
            Number of recent quarters to fetch
        source : str
            Data source: "auto", "sec", "alpha_vantage", "seeking_alpha"

        Returns
        -------
        List[TranscriptMetadata]
            List of transcript metadata
        """
        ticker = ticker.upper()

        if source == "auto":
            # Try sources in order of preference
            sources = ["sec", "alpha_vantage", "seeking_alpha"]
        else:
            sources = [source]

        metadata_list = []

        for src in sources:
            try:
                if src == "sec":
                    metadata_list = self._fetch_from_sec_edgar(ticker, num_quarters)
                elif src == "alpha_vantage":
                    if not self.alpha_vantage_key:
                        print(f"Skipping Alpha Vantage (no API key)")
                        continue
                    metadata_list = self._fetch_from_alpha_vantage(ticker, num_quarters)
                elif src == "seeking_alpha":
                    metadata_list = self._fetch_from_seeking_alpha(ticker, num_quarters)

                if metadata_list:
                    print(f"Successfully fetched {len(metadata_list)} transcripts from {src}")
                    break
            except Exception as e:
                print(f"Failed to fetch from {src}: {e}")
                continue

        return metadata_list

    def _fetch_from_sec_edgar(
        self,
        ticker: str,
        num_quarters: int,
    ) -> List[TranscriptMetadata]:
        """
        Fetch transcripts from SEC EDGAR.

        Note: Earnings call transcripts are often in 8-K filings (Item 7.01).
        Some companies also file them as exhibits to 10-Q/10-K.
        """
        print(f"Fetching transcripts for {ticker} from SEC EDGAR...")

        # Get CIK (Central Index Key) for the ticker
        cik = self._get_cik_from_ticker(ticker)
        if not cik:
            print(f"Could not find CIK for {ticker}")
            return []

        # Fetch recent 8-K filings
        filings = self._get_sec_filings(cik, form_type="8-K", count=num_quarters * 2)

        metadata_list = []

        for filing in filings:
            if len(metadata_list) >= num_quarters:
                break

            # Check if this 8-K contains earnings info
            if not self._is_earnings_8k(filing):
                continue

            # Download the filing
            filing_url = filing.get("filing_url")
            filing_date = filing.get("filing_date")

            if not filing_url or not filing_date:
                continue

            # Try to extract transcript from filing
            transcript_text = self._extract_transcript_from_8k(filing_url)

            if not transcript_text:
                continue

            # Save transcript
            file_path = self.output_dir / ticker / f"{ticker}_{filing_date}_8K.txt"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(transcript_text)

            # Infer quarter from filing date
            quarter, year = self._infer_quarter_from_date(filing_date)

            metadata = TranscriptMetadata(
                ticker=ticker,
                company_name=filing.get("company_name", ticker),
                fiscal_quarter=quarter,
                fiscal_year=year,
                call_date=datetime.fromisoformat(filing_date),
                source="sec_edgar",
                url=filing_url,
                file_path=str(file_path),
                has_transcript=True,
            )

            metadata_list.append(metadata)
            time.sleep(self.rate_limit_delay)

        return metadata_list

    def _fetch_from_alpha_vantage(
        self,
        ticker: str,
        num_quarters: int,
    ) -> List[TranscriptMetadata]:
        """
        Fetch transcripts from Alpha Vantage API.

        Note: Alpha Vantage provides earnings data but not full transcripts.
        We use this primarily for earnings dates and fundamentals.
        """
        print(f"Fetching earnings data for {ticker} from Alpha Vantage...")

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "EARNINGS",
            "symbol": ticker,
            "apikey": self.alpha_vantage_key,
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        if "quarterlyEarnings" not in data:
            print(f"No earnings data found for {ticker}")
            return []

        metadata_list = []

        for earnings in data["quarterlyEarnings"][:num_quarters]:
            fiscal_date_ending = earnings.get("fiscalDateEnding")
            reported_date = earnings.get("reportedDate")

            if not reported_date:
                continue

            # Parse quarter
            quarter_match = re.search(r"(\d{4})-(\d{2})", fiscal_date_ending or "")
            if quarter_match:
                year = int(quarter_match.group(1))
                month = int(quarter_match.group(2))
                quarter = f"Q{(month - 1) // 3 + 1} {year}"
            else:
                continue

            metadata = TranscriptMetadata(
                ticker=ticker,
                company_name=data.get("symbol", ticker),
                fiscal_quarter=quarter,
                fiscal_year=year,
                call_date=datetime.fromisoformat(reported_date),
                source="alpha_vantage",
                url=f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}",
                has_transcript=False,  # Alpha Vantage doesn't provide transcripts
                eps_actual=float(earnings.get("reportedEPS", 0) or 0),
                eps_estimate=float(earnings.get("estimatedEPS", 0) or 0),
            )

            metadata_list.append(metadata)

        return metadata_list

    def _fetch_from_seeking_alpha(
        self,
        ticker: str,
        num_quarters: int,
    ) -> List[TranscriptMetadata]:
        """
        Fetch transcripts from Seeking Alpha.

        Note: This is a placeholder. Actual implementation would require
        either API access or careful web scraping within ToS.
        """
        print(f"Seeking Alpha fetching not yet implemented for {ticker}")
        return []

    def _get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """Get CIK (Central Index Key) from ticker symbol."""
        # SEC provides a JSON mapping
        url = "https://www.sec.gov/files/company_tickers.json"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()

            for entry in data.values():
                if entry["ticker"].upper() == ticker.upper():
                    # CIK must be 10 digits, zero-padded
                    return str(entry["cik_str"]).zfill(10)
        except Exception as e:
            print(f"Error fetching CIK: {e}")

        return None

    def _get_sec_filings(
        self,
        cik: str,
        form_type: str = "8-K",
        count: int = 20,
    ) -> List[Dict]:
        """Get recent SEC filings for a CIK."""
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()

            filings_data = data.get("filings", {}).get("recent", {})

            filings = []
            forms = filings_data.get("form", [])
            filing_dates = filings_data.get("filingDate", [])
            accession_numbers = filings_data.get("accessionNumber", [])

            for i, form in enumerate(forms):
                if form == form_type and len(filings) < count:
                    accession = accession_numbers[i].replace("-", "")
                    filing_url = f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik}&accession_number={accession_numbers[i]}&xbrl_type=v"

                    filings.append({
                        "form_type": form,
                        "filing_date": filing_dates[i],
                        "accession_number": accession_numbers[i],
                        "filing_url": filing_url,
                        "company_name": data.get("name", ""),
                    })

            return filings
        except Exception as e:
            print(f"Error fetching SEC filings: {e}")
            return []

    def _is_earnings_8k(self, filing: Dict) -> bool:
        """
        Check if an 8-K filing is related to earnings.

        Earnings 8-Ks typically have Item 2.02 (Results of Operations)
        and/or Item 7.01 (Regulation FD Disclosure).
        """
        # This would require parsing the actual filing
        # For now, we assume all 8-Ks might contain earnings info
        return True

    def _extract_transcript_from_8k(self, filing_url: str) -> Optional[str]:
        """
        Extract earnings call transcript from 8-K filing.

        Note: This is a simplified version. Real implementation would need
        to parse the HTML/XBRL structure more carefully.
        """
        try:
            response = self.session.get(filing_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Try to find text content
            # This is very simplified - real 8-Ks have complex structure
            text_content = soup.get_text()

            # Look for earnings call indicators
            if any(keyword in text_content.lower() for keyword in [
                "earnings call",
                "conference call",
                "q&a session",
                "prepared remarks"
            ]):
                return text_content

            return None
        except Exception as e:
            print(f"Error extracting transcript: {e}")
            return None

    def _infer_quarter_from_date(self, date_str: str) -> tuple[str, int]:
        """Infer fiscal quarter from filing date."""
        date = datetime.fromisoformat(date_str)
        year = date.year
        month = date.month

        # Approximate quarter (actual fiscal quarters may differ)
        quarter_num = ((month - 1) // 3 + 1)

        # Adjust for typical earnings reporting (usually 1-2 months after quarter end)
        if month <= 2:
            quarter_num = 4
            year -= 1
        elif month <= 5:
            quarter_num = 1
        elif month <= 8:
            quarter_num = 2
        else:
            quarter_num = 3

        return f"Q{quarter_num} {year}", year

    def save_metadata(self, metadata_list: List[TranscriptMetadata], output_file: Path):
        """Save transcript metadata to JSON."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(
                [asdict(m) for m in metadata_list],
                f,
                indent=2,
                default=str,
            )


def fetch_earnings_transcripts(
    ticker: str,
    output_dir: Path,
    num_quarters: int = 8,
    source: str = "auto",
    alpha_vantage_key: Optional[str] = None,
) -> List[TranscriptMetadata]:
    """
    Convenience function to fetch earnings transcripts.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    output_dir : Path
        Directory to save transcripts
    num_quarters : int
        Number of recent quarters to fetch
    source : str
        Data source: "auto", "sec", "alpha_vantage", "seeking_alpha"
    alpha_vantage_key : Optional[str]
        Alpha Vantage API key

    Returns
    -------
    List[TranscriptMetadata]
        List of fetched transcripts
    """
    fetcher = TranscriptFetcher(
        output_dir=output_dir,
        alpha_vantage_key=alpha_vantage_key,
    )

    return fetcher.fetch_ticker(ticker, num_quarters, source)
