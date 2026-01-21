"""
Fetch FOMC press conference transcript PDFs from the Federal Reserve website.

Strategy:
- For recent years (2021+): Scrape https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- For historical years (≤2020): Scrape https://www.federalreserve.gov/monetarypolicy/fomchistoricalYYYY.htm
- Collect all meeting pages whose URL contains "fomcpresconfYYYYMMDD.htm"
- Extract the date from each meeting page URL
- Construct PDF URL directly: https://www.federalreserve.gov/mediacenter/files/FOMCpresconfYYYYMMDD.pdf
- Write an index CSV: date, meeting_page_url, pdf_url, local_path
"""

from __future__ import annotations

import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

FED_BASE = "https://www.federalreserve.gov"
YEAR_PAGE_TMPL = (
    "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
)
CURRENT_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

# Years 2021+ are on the current calendar page, not separate historical pages
HISTORICAL_CUTOFF_YEAR = 2020

UA = "powell-transcript-fetcher/1.0 (+https://www.federalreserve.gov)"


DATE_RE = re.compile(r"fomcpresconf(\d{8})\.htm$", re.IGNORECASE)


@dataclass(frozen=True)
class PressConfItem:
    date_yyyymmdd: str
    meeting_page_url: str
    pdf_url: str


def http_get(url: str, timeout: int = 30) -> requests.Response:
    resp = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
    resp.raise_for_status()
    return resp


def parse_current_calendar_for_meeting_pages(years_needed: set[int]) -> dict[int, list[str]]:
    """
    Parse the current FOMC calendar page and return meeting pages by year.
    Used for recent years (2021+) that don't have separate historical pages.
    """
    html = http_get(CURRENT_CALENDAR_URL).text
    soup = BeautifulSoup(html, "html.parser")

    # Group meeting pages by year
    pages_by_year: dict[int, set[str]] = {y: set() for y in years_needed}

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "fomcpresconf" in href.lower() and href.lower().endswith(".htm"):
            full_url = urljoin(FED_BASE, href)
            # Extract year from URL (e.g., fomcpresconf20250129.htm -> 2025)
            date = extract_date_from_meeting_url(full_url)
            if date and len(date) == 8:
                year = int(date[:4])
                if year in pages_by_year:
                    pages_by_year[year].add(full_url)

    return {y: sorted(pages) for y, pages in pages_by_year.items()}


def parse_year_for_meeting_pages(year: int) -> list[str]:
    """Return all press-conference meeting page URLs for a given year."""
    if year > HISTORICAL_CUTOFF_YEAR:
        # Recent years are on the current calendar page
        all_pages = parse_current_calendar_for_meeting_pages({year})
        return all_pages.get(year, [])
    else:
        # Older years have dedicated historical pages
        url = YEAR_PAGE_TMPL.format(year=year)
        html = http_get(url).text
        soup = BeautifulSoup(html, "html.parser")

        meeting_pages: set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "fomcpresconf" in href.lower() and href.lower().endswith(".htm"):
                meeting_pages.add(urljoin(FED_BASE, href))
        return sorted(meeting_pages)


def extract_date_from_meeting_url(meeting_page_url: str) -> Optional[str]:
    m = DATE_RE.search(meeting_page_url)
    return m.group(1) if m else None


def construct_pressconf_pdf_url(date_yyyymmdd: str) -> str:
    """Construct the Press Conference Transcript PDF URL from the date."""
    return f"https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{date_yyyymmdd}.pdf"


def download_pdf(
    pdf_url: str, out_path: Path, overwrite: bool, timeout: int = 60
) -> None:
    if out_path.exists() and not overwrite:
        return

    resp = http_get(pdf_url, timeout=timeout)
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "pdf" not in ctype and not pdf_url.lower().endswith(".pdf"):
        raise RuntimeError(f"Unexpected content-type for {pdf_url}: {ctype}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)

    if out_path.stat().st_size < 1024:  # sanity check
        raise RuntimeError(
            f"Downloaded file too small: {out_path} ({out_path.stat().st_size} bytes)"
        )


def build_items(years: Iterable[int]) -> list[PressConfItem]:
    items: list[PressConfItem] = []
    for y in years:
        meeting_pages = parse_year_for_meeting_pages(y)
        for mp in meeting_pages:
            date = extract_date_from_meeting_url(mp)
            if not date:
                continue
            pdf = construct_pressconf_pdf_url(date)
            items.append(
                PressConfItem(date_yyyymmdd=date, meeting_page_url=mp, pdf_url=pdf)
            )
    # unique by date
    uniq: dict[str, PressConfItem] = {}
    for it in items:
        uniq[it.date_yyyymmdd] = it
    return [uniq[k] for k in sorted(uniq.keys())]


def fetch_transcripts(
    start_year: int,
    end_year: int,
    out_dir: Path,
    index_csv: Optional[Path] = None,
    workers: int = 8,
    sleep: float = 0.0,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Fetch FOMC press conference transcript PDFs by scraping Fed historical year pages.

    Args:
        start_year: First year to fetch
        end_year: Last year to fetch (inclusive)
        out_dir: Directory to save PDFs
        index_csv: Optional path to write index CSV. If None, defaults to out_dir parent / "pressconf_index.csv"
        workers: Number of concurrent download workers
        sleep: Sleep between year-page fetches (seconds)
        overwrite: Overwrite existing PDFs
        dry_run: Only print what would be downloaded
    """
    years = list(range(start_year, end_year + 1))

    # Collect meeting pages
    all_items: list[PressConfItem] = []

    # Separate recent years from historical years for efficiency
    recent_years = [y for y in years if y > HISTORICAL_CUTOFF_YEAR]
    historical_years = [y for y in years if y <= HISTORICAL_CUTOFF_YEAR]

    # Fetch recent years from current calendar (single request for all)
    if recent_years:
        print(f"  - fetching current calendar for years {min(recent_years)}-{max(recent_years)}")
        try:
            pages_by_year = parse_current_calendar_for_meeting_pages(set(recent_years))
            for y in recent_years:
                meeting_pages = pages_by_year.get(y, [])
                print(f"    year {y}: {len(meeting_pages)} meetings")
                for mp in meeting_pages:
                    date = extract_date_from_meeting_url(mp)
                    if not date:
                        continue
                    try:
                        pdf = construct_pressconf_pdf_url(date)
                        all_items.append(PressConfItem(date, mp, pdf))
                    except Exception as e:
                        print(f"    ! failed {mp}: {e}")
        except Exception as e:
            print(f"    ! failed to parse current calendar: {e}")

    # Fetch historical years (separate request per year)
    for y in historical_years:
        print(f"  - year {y}")
        try:
            meeting_pages = parse_year_for_meeting_pages(y)
        except Exception as e:
            print(f"    ! failed to parse year {y}: {e}")
            continue

        for mp in meeting_pages:
            date = extract_date_from_meeting_url(mp)
            if not date:
                continue
            try:
                pdf = construct_pressconf_pdf_url(date)
                all_items.append(PressConfItem(date, mp, pdf))
            except Exception as e:
                print(f"    ! failed {mp}: {e}")

        if sleep > 0:
            time.sleep(sleep)

    # de-dupe by date
    uniq = {}
    for it in all_items:
        uniq[it.date_yyyymmdd] = it
    items = [uniq[k] for k in sorted(uniq.keys())]

    print(f"Found {len(items)} press conference transcripts.")

    # Write index CSV
    if index_csv is None:
        index_csv = out_dir.parent / "pressconf_index.csv"
    
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    with index_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date_yyyymmdd", "meeting_page_url", "pdf_url", "local_path"])
        for it in items:
            local = out_dir / f"FOMCpresconf{it.date_yyyymmdd}.pdf"
            w.writerow([it.date_yyyymmdd, it.meeting_page_url, it.pdf_url, str(local)])

    if dry_run:
        for it in items:
            print(f"{it.date_yyyymmdd}  {it.pdf_url}")
        print(f"(dry-run) wrote index: {index_csv}")
        return

    # Download concurrently
    print(f"Downloading to {out_dir} with {workers} workers ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for it in items:
            local = out_dir / f"FOMCpresconf{it.date_yyyymmdd}.pdf"
            futures[ex.submit(download_pdf, it.pdf_url, local, overwrite)] = (it, local)

        done = 0
        for fut in as_completed(futures):
            it, local = futures[fut]
            try:
                fut.result()
                done += 1
                if done % 10 == 0 or done == len(items):
                    print(f"  downloaded {done}/{len(items)}")
            except Exception as e:
                print(f"  ! failed {it.date_yyyymmdd}: {e}")
