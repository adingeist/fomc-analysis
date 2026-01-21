#!/usr/bin/env python3
"""
Fetch all FOMC press conference transcript PDFs for a year range.

Strategy (robust):
- Scrape https://www.federalreserve.gov/monetarypolicy/fomchistoricalYYYY.htm
  and collect all meeting pages whose URL contains "fomcpresconfYYYYMMDD.htm".
- For each meeting page, find the "Press Conference Transcript (PDF)" link and download it.
- Write an index CSV: date, meeting_page_url, pdf_url, local_path.

Why not hardcode https://www.federalreserve.gov/mediacenter/files/FOMCpresconfYYYYMMDD.pdf ?
Because older years sometimes differ in filename case (e.g. 'fomcpresconf...pdf') and other quirks.
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

import click
import requests
from bs4 import BeautifulSoup

FED_BASE = "https://www.federalreserve.gov"
YEAR_PAGE_TMPL = (
    "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
)

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


def parse_year_for_meeting_pages(year: int) -> list[str]:
    """Return all press-conference meeting page URLs for a given year."""
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


def find_pressconf_pdf(meeting_page_url: str) -> str:
    """Given a meeting pressconf page, locate the Press Conference Transcript PDF URL."""
    html = http_get(meeting_page_url).text
    soup = BeautifulSoup(html, "html.parser")

    # 1) Prefer the exact link text
    for a in soup.find_all("a", href=True):
        text = " ".join(a.get_text(" ", strip=True).split()).lower()
        href = a["href"]
        if "press conference transcript" in text and href.lower().endswith(".pdf"):
            return urljoin(FED_BASE, href)

    # 2) Fallback: any PDF link that looks like presconf transcript
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf") and "presconf" in href.lower():
            return urljoin(FED_BASE, href)

    raise RuntimeError(
        f"Could not find press conference transcript PDF on {meeting_page_url}"
    )


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
            pdf = find_pressconf_pdf(mp)
            items.append(
                PressConfItem(date_yyyymmdd=date, meeting_page_url=mp, pdf_url=pdf)
            )
    # unique by date
    uniq: dict[str, PressConfItem] = {}
    for it in items:
        uniq[it.date_yyyymmdd] = it
    return [uniq[k] for k in sorted(uniq.keys())]


@click.command()
@click.option("--start-year", type=int, default=2011, show_default=True)
@click.option("--end-year", type=int, default=time.gmtime().tm_year, show_default=True)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("data/transcripts_pdf"),
    show_default=True,
)
@click.option(
    "--index-csv",
    type=click.Path(path_type=Path),
    default=Path("data/pressconf_index.csv"),
    show_default=True,
)
@click.option(
    "--workers",
    type=int,
    default=8,
    show_default=True,
    help="Concurrent download workers",
)
@click.option(
    "--sleep",
    type=float,
    default=0.0,
    show_default=True,
    help="Sleep between year-page fetches (seconds)",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing PDFs")
@click.option("--dry-run", is_flag=True, help="Only print what would be downloaded")
def main(
    start_year: int,
    end_year: int,
    out_dir: Path,
    index_csv: Path,
    workers: int,
    sleep: float,
    overwrite: bool,
    dry_run: bool,
):
    """
    Fetch FOMC press conference transcript PDFs by scraping Fed historical year pages.
    """
    years = list(range(start_year, end_year + 1))

    click.echo(f"Collecting meeting pages for years {start_year}..{end_year} ...")
    all_items: list[PressConfItem] = []

    # Slight politeness: optional sleep between year fetches
    for y in years:
        click.echo(f"  - year {y}")
        try:
            meeting_pages = parse_year_for_meeting_pages(y)
        except Exception as e:
            click.echo(f"    ! failed to parse year {y}: {e}")
            continue

        for mp in meeting_pages:
            date = extract_date_from_meeting_url(mp)
            if not date:
                continue
            try:
                pdf = find_pressconf_pdf(mp)
                all_items.append(PressConfItem(date, mp, pdf))
            except Exception as e:
                click.echo(f"    ! failed {mp}: {e}")

        if sleep > 0:
            time.sleep(sleep)

    # de-dupe by date
    uniq = {}
    for it in all_items:
        uniq[it.date_yyyymmdd] = it
    items = [uniq[k] for k in sorted(uniq.keys())]

    click.echo(f"Found {len(items)} press conference transcripts.")

    # Write index CSV
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    with index_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date_yyyymmdd", "meeting_page_url", "pdf_url", "local_path"])
        for it in items:
            local = out_dir / f"FOMCpresconf{it.date_yyyymmdd}.pdf"
            w.writerow([it.date_yyyymmdd, it.meeting_page_url, it.pdf_url, str(local)])

    if dry_run:
        for it in items:
            click.echo(f"{it.date_yyyymmdd}  {it.pdf_url}")
        click.echo(f"(dry-run) wrote index: {index_csv}")
        return

    # Download concurrently
    click.echo(f"Downloading to {out_dir} with {workers} workers ...")
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
                    click.echo(f"  downloaded {done}/{len(items)}")
            except Exception as e:
                click.echo(f"  ! failed {it.date_yyyymmdd}: {e}")

    click.echo(f"Done. Index CSV: {index_csv}")


if __name__ == "__main__":
    main()
