"""
Build Historical Outcomes Database from Kalshi Finalized Contracts.

This script:
1. Fetches all finalized earnings mention contracts from Kalshi (in parallel)
2. Extracts outcome data (YES=1 if last_price > 50 cents, NO=0 otherwise)
3. Saves to multiple formats:
   - outcomes.csv: CSV format for easy analysis
   - outcomes.json: JSON format with full details
   - outcomes.db: SQLite database for querying
4. Generates summary statistics

Uses async batching for 5-7x speedup when fetching multiple tickers.
"""

from pathlib import Path
import json
import csv
import sqlite3
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

from fomc_analysis.kalshi_client_factory import (
    get_kalshi_client,
    batch_get_markets,
    KalshiSdkAdapter,
)


def parse_call_date_from_ticker(ticker: str) -> Optional[str]:
    """
    Extract earnings call date from contract ticker.

    Example: KXEARNINGSMENTIONMETA-24OCT31-AI -> 2024-10-31
             KXEARNINGSMENTIONTSLA-26JUN30-ROBOTAXI -> 2026-06-30

    Returns ISO date string (YYYY-MM-DD) or None if parsing fails.
    """
    # Pattern: {SERIES}-{YY}{MMM}{DD}-{WORD}
    # e.g., KXEARNINGSMENTIONMETA-24OCT31-AI
    match = re.search(r'-(\d{2})([A-Z]{3})(\d{2})-', ticker)

    if not match:
        return None

    year_short, month_str, day = match.groups()

    # Convert year (24 -> 2024, 26 -> 2026)
    year = 2000 + int(year_short)

    # Convert month name to number
    months = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
        'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
        'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
    }

    month = months.get(month_str.upper())
    if not month:
        return None

    try:
        date = datetime(year, month, int(day))
        return date.strftime('%Y-%m-%d')
    except ValueError:
        return None


def extract_company_ticker(series_ticker: str) -> str:
    """
    Extract company ticker from series ticker.

    Example: KXEARNINGSMENTIONMETA -> META
             KXEARNINGSMENTIONTSLA -> TSLA
    """
    return series_ticker.replace('KXEARNINGSMENTION', '')


def determine_outcome(last_price: int) -> int:
    """
    Determine outcome from final price.

    YES = 1 if last_price > 50 cents
    NO = 0 otherwise

    Kalshi contracts settle at 1 cent (NO) or 99 cents (YES).
    """
    return 1 if last_price > 50 else 0


def extract_outcome_from_contract(contract: Dict[str, Any], series_ticker: str) -> Optional[Dict[str, Any]]:
    """
    Extract outcome data from a finalized Kalshi contract.

    Returns:
        Dict with keys: ticker, word, call_date, outcome, final_price,
                       contract_ticker, settled_date, series_ticker
        Or None if parsing fails
    """
    try:
        contract_ticker = contract.get('ticker', '')

        # Extract company ticker
        company_ticker = extract_company_ticker(series_ticker)

        # Extract word from custom_strike
        custom_strike = contract.get('custom_strike', {})
        word = custom_strike.get('Word')

        if not word:
            return None

        # Extract call date from ticker
        call_date = parse_call_date_from_ticker(contract_ticker)

        # Get final price and determine outcome
        last_price = contract.get('last_price', 0)
        outcome = determine_outcome(last_price)

        # Get settlement date
        settled_date = contract.get('close_time') or contract.get('expiration_time')

        return {
            'ticker': company_ticker,
            'word': word,
            'call_date': call_date,
            'outcome': outcome,
            'final_price': last_price,
            'contract_ticker': contract_ticker,
            'settled_date': settled_date,
            'series_ticker': series_ticker,
            'status': contract.get('status', ''),
            'title': contract.get('title', ''),
        }

    except Exception as e:
        print(f"Error extracting outcome from {contract.get('ticker', 'unknown')}: {e}")
        return None


async def fetch_all_finalized_contracts_async(
    tickers: List[str],
    status_filters: List[str] = ['finalized', 'settled', 'closed'],
) -> tuple[List[Dict[str, Any]], Any]:
    """
    Fetch all finalized contracts for given tickers using async batching.

    Parameters:
        tickers: List of company tickers (e.g., ['META', 'TSLA'])
        status_filters: List of statuses to try (in order)

    Returns:
        Tuple of (outcomes list, client) - client must be closed by caller
    """
    client = get_kalshi_client()

    if not isinstance(client, KalshiSdkAdapter):
        raise TypeError(
            "Client must be KalshiSdkAdapter for async batching. "
            "Ensure KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_BASE64 are set."
        )

    print("=" * 70)
    print("BUILDING KALSHI EARNINGS OUTCOMES DATABASE")
    print("=" * 70)

    all_outcomes = []

    for status in status_filters:
        print(f"\n[1] Fetching {status} contracts for {len(tickers)} tickers (parallel)...")

        # Build series tickers
        series_tickers = [f"KXEARNINGSMENTION{ticker}" for ticker in tickers]

        # Fetch all in parallel
        markets_by_series = await batch_get_markets(
            client=client,
            series_tickers=series_tickers,
            status=status,
            limit=200,
        )

        # Extract outcomes from each series
        for series_ticker, markets in markets_by_series.items():
            company = extract_company_ticker(series_ticker)

            if not markets:
                print(f"  {company}: No {status} markets found")
                continue

            print(f"  {company}: {len(markets)} {status} markets")

            # Extract outcomes
            for market in markets:
                outcome = extract_outcome_from_contract(market, series_ticker)
                if outcome:
                    all_outcomes.append(outcome)

        # If we found outcomes with this status, no need to try other statuses
        if all_outcomes:
            print(f"\n✓ Found {len(all_outcomes)} outcomes with status '{status}'")
            break

    if not all_outcomes:
        print("\n⚠ No finalized contracts found with any status filter")
        print(f"  Tried: {', '.join(status_filters)}")

    return all_outcomes, client


def save_to_csv(outcomes: List[Dict[str, Any]], output_path: Path):
    """Save outcomes to CSV format."""
    fieldnames = [
        'ticker',
        'word',
        'call_date',
        'outcome',
        'final_price',
        'contract_ticker',
        'settled_date',
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for outcome in outcomes:
            # Filter to only the fields we want in CSV
            row = {k: outcome.get(k, '') for k in fieldnames}
            writer.writerow(row)

    print(f"✓ Saved CSV: {output_path}")


def save_to_json(outcomes: List[Dict[str, Any]], output_path: Path):
    """Save outcomes to JSON format (with full details)."""
    with open(output_path, 'w') as f:
        json.dump(outcomes, f, indent=2)

    print(f"✓ Saved JSON: {output_path}")


def save_to_sqlite(outcomes: List[Dict[str, Any]], output_path: Path):
    """Save outcomes to SQLite database."""
    # Remove if exists
    if output_path.exists():
        output_path.unlink()

    # Create database
    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()

    # Create outcomes table
    cursor.execute('''
        CREATE TABLE outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            word TEXT NOT NULL,
            call_date TEXT,
            outcome INTEGER NOT NULL,
            final_price INTEGER NOT NULL,
            contract_ticker TEXT UNIQUE NOT NULL,
            settled_date TEXT,
            series_ticker TEXT,
            status TEXT,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create indexes for fast lookups
    cursor.execute('CREATE INDEX idx_ticker ON outcomes(ticker)')
    cursor.execute('CREATE INDEX idx_word ON outcomes(word)')
    cursor.execute('CREATE INDEX idx_call_date ON outcomes(call_date)')
    cursor.execute('CREATE INDEX idx_ticker_word ON outcomes(ticker, word)')

    # Insert data
    for outcome in outcomes:
        cursor.execute('''
            INSERT INTO outcomes (
                ticker, word, call_date, outcome, final_price,
                contract_ticker, settled_date, series_ticker, status, title
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            outcome.get('ticker'),
            outcome.get('word'),
            outcome.get('call_date'),
            outcome.get('outcome'),
            outcome.get('final_price'),
            outcome.get('contract_ticker'),
            outcome.get('settled_date'),
            outcome.get('series_ticker'),
            outcome.get('status'),
            outcome.get('title'),
        ))

    conn.commit()
    conn.close()

    print(f"✓ Saved SQLite DB: {output_path}")


def generate_summary_statistics(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from outcomes."""
    if not outcomes:
        return {
            'total_contracts': 0,
            'by_ticker': {},
            'by_word': {},
            'by_outcome': {'YES': 0, 'NO': 0},
        }

    # Group by ticker
    by_ticker = {}
    for outcome in outcomes:
        ticker = outcome['ticker']
        if ticker not in by_ticker:
            by_ticker[ticker] = {
                'total': 0,
                'yes': 0,
                'no': 0,
                'words': set(),
            }

        by_ticker[ticker]['total'] += 1
        by_ticker[ticker]['words'].add(outcome['word'])

        if outcome['outcome'] == 1:
            by_ticker[ticker]['yes'] += 1
        else:
            by_ticker[ticker]['no'] += 1

    # Convert sets to lists for JSON serialization
    for ticker_stats in by_ticker.values():
        ticker_stats['words'] = sorted(list(ticker_stats['words']))

    # Group by word
    by_word = {}
    for outcome in outcomes:
        word = outcome['word']
        if word not in by_word:
            by_word[word] = {
                'total': 0,
                'yes': 0,
                'no': 0,
                'tickers': set(),
            }

        by_word[word]['total'] += 1
        by_word[word]['tickers'].add(outcome['ticker'])

        if outcome['outcome'] == 1:
            by_word[word]['yes'] += 1
        else:
            by_word[word]['no'] += 1

    # Convert sets to lists
    for word_stats in by_word.values():
        word_stats['tickers'] = sorted(list(word_stats['tickers']))

    # Overall outcome counts
    total_yes = sum(1 for o in outcomes if o['outcome'] == 1)
    total_no = len(outcomes) - total_yes

    return {
        'total_contracts': len(outcomes),
        'by_ticker': by_ticker,
        'by_word': by_word,
        'by_outcome': {
            'YES': total_yes,
            'NO': total_no,
        },
        'date_range': {
            'earliest': min((o['call_date'] for o in outcomes if o['call_date']), default=None),
            'latest': max((o['call_date'] for o in outcomes if o['call_date']), default=None),
        },
    }


def print_summary(summary: Dict[str, Any]):
    """Print summary statistics to console."""
    print("\n" + "=" * 70)
    print("OUTCOMES DATABASE SUMMARY")
    print("=" * 70)

    print(f"\nTotal contracts: {summary['total_contracts']}")

    print(f"\nOutcome distribution:")
    print(f"  YES (mentioned): {summary['by_outcome']['YES']}")
    print(f"  NO (not mentioned): {summary['by_outcome']['NO']}")

    if summary['date_range']['earliest']:
        print(f"\nDate range:")
        print(f"  Earliest: {summary['date_range']['earliest']}")
        print(f"  Latest: {summary['date_range']['latest']}")

    print(f"\nBy company ticker:")
    for ticker, stats in sorted(summary['by_ticker'].items()):
        print(f"  {ticker}:")
        print(f"    Total: {stats['total']} contracts")
        print(f"    YES: {stats['yes']}, NO: {stats['no']}")
        print(f"    Words: {', '.join(stats['words'][:5])}")
        if len(stats['words']) > 5:
            print(f"           ... and {len(stats['words']) - 5} more")

    print(f"\nTop 10 words by frequency:")
    top_words = sorted(
        summary['by_word'].items(),
        key=lambda x: x[1]['total'],
        reverse=True
    )[:10]

    for word, stats in top_words:
        print(f"  {word}:")
        print(f"    Total: {stats['total']} contracts")
        print(f"    YES: {stats['yes']}, NO: {stats['no']}")
        print(f"    Companies: {', '.join(stats['tickers'])}")


async def build_outcomes_database(
    output_dir: Path = Path("data/outcomes_database"),
    tickers: Optional[List[str]] = None,
):
    """
    Build comprehensive outcomes database from Kalshi finalized contracts.

    Parameters:
        output_dir: Directory to save outputs
        tickers: List of company tickers to fetch (default: all known)
    """
    # Default tickers
    if tickers is None:
        tickers = [
            'META',
            'TSLA',
            'NVDA',
            'AAPL',
            'MSFT',
            'AMZN',
            'GOOGLUNNAMED',  # Google uses this series
            'GOOGL',  # Try both variants
            'NFLX',
            'NETFLIX',  # Try both variants
            'COIN',  # Coinbase
        ]

    # Fetch all outcomes
    outcomes, client = await fetch_all_finalized_contracts_async(tickers)

    # Close client to clean up aiohttp sessions
    try:
        client.close()
    except Exception as e:
        print(f"Warning: Error closing client: {e}")

    if not outcomes:
        print("\n⚠ No outcomes found. Database not created.")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2] Saving outcomes to {output_dir}...")

    # Save to multiple formats
    save_to_csv(outcomes, output_dir / "outcomes.csv")
    save_to_json(outcomes, output_dir / "outcomes.json")
    save_to_sqlite(outcomes, output_dir / "outcomes.db")

    # Generate and save summary
    print(f"\n[3] Generating summary statistics...")
    summary = generate_summary_statistics(outcomes)

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")

    # Print summary to console
    print_summary(summary)

    print("\n" + "=" * 70)
    print("✓ OUTCOMES DATABASE CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  - {output_dir / 'outcomes.csv'}")
    print(f"  - {output_dir / 'outcomes.json'}")
    print(f"  - {output_dir / 'outcomes.db'}")
    print(f"  - {output_dir / 'summary.json'}")

    # Usage examples
    print("\n" + "=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)
    print("\nQuery SQLite database:")
    print(f"  sqlite3 {output_dir / 'outcomes.db'}")
    print("  SELECT ticker, word, outcome FROM outcomes WHERE ticker='META';")
    print("  SELECT word, COUNT(*) as total, SUM(outcome) as yes_count")
    print("    FROM outcomes GROUP BY word ORDER BY total DESC;")

    print("\nLoad in Python:")
    print("  import pandas as pd")
    print(f"  df = pd.read_csv('{output_dir / 'outcomes.csv'}')")
    print("  print(df.groupby('ticker')['outcome'].value_counts())")

    print("\nQuery database with pandas:")
    print("  import sqlite3")
    print(f"  conn = sqlite3.connect('{output_dir / 'outcomes.db'}')")
    print("  df = pd.read_sql_query('SELECT * FROM outcomes', conn)")


def main():
    """Main entry point."""
    try:
        asyncio.run(build_outcomes_database())
    except Exception as e:
        print(f"\n✗ Error building outcomes database: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
