"""
Query insights from the outcomes database.

This script provides pre-built queries for common analysis tasks:
- Finding predictable words
- Analyzing ticker-specific patterns
- Identifying high-variance words
- Comparing outcomes across companies
"""

import sqlite3
import argparse
from pathlib import Path
import pandas as pd


def get_connection(db_path: Path = Path("data/outcomes_database/outcomes.db")):
    """Get database connection."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Run 'python scripts/build_outcomes_database.py' first"
        )
    return sqlite3.connect(db_path)


def query_predictable_words(conn, min_occurrences: int = 3, threshold: float = 0.9):
    """
    Find words that are highly predictable (>90% or <10% mention rate).

    Parameters:
        conn: Database connection
        min_occurrences: Minimum number of times word must appear
        threshold: Threshold for "predictable" (0.9 = 90%)

    Returns:
        DataFrame with predictable words
    """
    query = f"""
    SELECT
        word,
        COUNT(*) as total,
        SUM(outcome) as yes_count,
        COUNT(*) - SUM(outcome) as no_count,
        ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as yes_pct,
        GROUP_CONCAT(DISTINCT ticker) as tickers,
        CASE
            WHEN SUM(outcome) >= {threshold} * COUNT(*) THEN 'HIGH_YES'
            WHEN SUM(outcome) <= {1-threshold} * COUNT(*) THEN 'HIGH_NO'
        END as pattern
    FROM outcomes
    GROUP BY word
    HAVING COUNT(*) >= {min_occurrences}
        AND (SUM(outcome) >= {threshold} * COUNT(*)
             OR SUM(outcome) <= {1-threshold} * COUNT(*))
    ORDER BY total DESC, yes_pct DESC
    """

    df = pd.read_sql_query(query, conn)
    return df


def query_ticker_patterns(conn, ticker: str):
    """
    Analyze word mention patterns for a specific ticker.

    Parameters:
        conn: Database connection
        ticker: Company ticker (e.g., 'META', 'TSLA')

    Returns:
        DataFrame with ticker-specific patterns
    """
    query = f"""
    SELECT
        word,
        COUNT(*) as occurrences,
        SUM(outcome) as mentioned,
        COUNT(*) - SUM(outcome) as not_mentioned,
        ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as mention_pct,
        MIN(call_date) as first_date,
        MAX(call_date) as last_date
    FROM outcomes
    WHERE ticker = '{ticker}'
    GROUP BY word
    ORDER BY occurrences DESC, mention_pct DESC
    """

    df = pd.read_sql_query(query, conn)
    return df


def query_high_variance_words(conn, min_occurrences: int = 3):
    """
    Find words with high variance (sometimes mentioned, sometimes not).

    These are the most interesting for prediction models as they're not deterministic.

    Parameters:
        conn: Database connection
        min_occurrences: Minimum occurrences

    Returns:
        DataFrame with high-variance words
    """
    query = f"""
    SELECT
        word,
        COUNT(*) as total,
        SUM(outcome) as yes_count,
        COUNT(*) - SUM(outcome) as no_count,
        ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as yes_pct,
        GROUP_CONCAT(DISTINCT ticker) as tickers,
        -- Variance is highest around 50%
        ABS(50.0 - 100.0 * SUM(outcome) / COUNT(*)) as distance_from_50
    FROM outcomes
    GROUP BY word
    HAVING COUNT(*) >= {min_occurrences}
        AND SUM(outcome) > 0  -- Mentioned at least once
        AND SUM(outcome) < COUNT(*)  -- Not mentioned at least once
    ORDER BY distance_from_50 ASC, total DESC
    LIMIT 30
    """

    df = pd.read_sql_query(query, conn)
    return df


def query_cross_ticker_comparison(conn, word: str):
    """
    Compare how a word is mentioned across different tickers.

    Parameters:
        conn: Database connection
        word: Word to compare

    Returns:
        DataFrame with per-ticker stats
    """
    query = f"""
    SELECT
        ticker,
        COUNT(*) as total,
        SUM(outcome) as yes,
        COUNT(*) - SUM(outcome) as no,
        ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as yes_pct
    FROM outcomes
    WHERE word = '{word}'
    GROUP BY ticker
    ORDER BY total DESC
    """

    df = pd.read_sql_query(query, conn)
    return df


def query_temporal_trends(conn, ticker: str = None):
    """
    Analyze mention rates over time.

    Parameters:
        conn: Database connection
        ticker: Optional ticker to filter by

    Returns:
        DataFrame with temporal trends
    """
    where_clause = f"WHERE ticker = '{ticker}'" if ticker else ""

    query = f"""
    SELECT
        call_date,
        ticker,
        COUNT(*) as total_contracts,
        SUM(outcome) as mentions,
        COUNT(*) - SUM(outcome) as no_mentions,
        ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as mention_pct
    FROM outcomes
    {where_clause}
    GROUP BY call_date, ticker
    ORDER BY call_date, ticker
    """

    df = pd.read_sql_query(query, conn)
    return df


def query_rare_events(conn, max_occurrences: int = 2):
    """
    Find rare words that have only appeared a few times.

    These might be one-off events or emerging trends.

    Parameters:
        conn: Database connection
        max_occurrences: Maximum number of occurrences

    Returns:
        DataFrame with rare words
    """
    query = f"""
    SELECT
        word,
        COUNT(*) as occurrences,
        SUM(outcome) as mentioned,
        GROUP_CONCAT(ticker) as tickers,
        GROUP_CONCAT(call_date) as dates
    FROM outcomes
    GROUP BY word
    HAVING COUNT(*) <= {max_occurrences}
    ORDER BY SUM(outcome) DESC, occurrences DESC
    """

    df = pd.read_sql_query(query, conn)
    return df


def print_report(title: str, df: pd.DataFrame):
    """Print formatted report."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if len(df) == 0:
        print("No results found.")
    else:
        print(df.to_string(index=False))

    print(f"\nTotal rows: {len(df)}")


def main():
    """Main CLI."""
    parser = argparse.ArgumentParser(description="Query outcomes database insights")
    parser.add_argument(
        "--query",
        choices=[
            "predictable",
            "high-variance",
            "ticker-patterns",
            "cross-ticker",
            "temporal",
            "rare",
        ],
        default="predictable",
        help="Type of query to run",
    )
    parser.add_argument("--ticker", help="Ticker for ticker-specific queries")
    parser.add_argument("--word", help="Word for cross-ticker comparison")
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=3,
        help="Minimum occurrences for word analysis",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Threshold for predictable words (default: 0.9)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/outcomes_database/outcomes.db"),
        help="Path to database file",
    )

    args = parser.parse_args()

    try:
        conn = get_connection(args.db)

        if args.query == "predictable":
            df = query_predictable_words(conn, args.min_occurrences, args.threshold)
            print_report("PREDICTABLE WORDS", df)

        elif args.query == "high-variance":
            df = query_high_variance_words(conn, args.min_occurrences)
            print_report("HIGH-VARIANCE WORDS (Best for Prediction)", df)

        elif args.query == "ticker-patterns":
            if not args.ticker:
                parser.error("--ticker required for ticker-patterns query")
            df = query_ticker_patterns(conn, args.ticker)
            print_report(f"WORD PATTERNS FOR {args.ticker}", df)

        elif args.query == "cross-ticker":
            if not args.word:
                parser.error("--word required for cross-ticker query")
            df = query_cross_ticker_comparison(conn, args.word)
            print_report(f"CROSS-TICKER COMPARISON: {args.word}", df)

        elif args.query == "temporal":
            df = query_temporal_trends(conn, args.ticker)
            title = f"TEMPORAL TRENDS FOR {args.ticker}" if args.ticker else "TEMPORAL TRENDS (ALL TICKERS)"
            print_report(title, df)

        elif args.query == "rare":
            df = query_rare_events(conn, args.min_occurrences)
            print_report(f"RARE EVENTS (≤{args.min_occurrences} occurrences)", df)

        conn.close()

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
