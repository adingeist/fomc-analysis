"""
Verify and explore the outcomes database.

This script demonstrates how to query and analyze the outcomes database
created by build_outcomes_database.py.
"""

import sqlite3
import json
from pathlib import Path
import pandas as pd


def verify_database(db_path: Path = Path("data/outcomes_database/outcomes.db")):
    """Verify and explore the outcomes database."""

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("\nRun 'python scripts/build_outcomes_database.py' first")
        return

    print("=" * 70)
    print("OUTCOMES DATABASE VERIFICATION")
    print("=" * 70)

    # Connect to database
    conn = sqlite3.connect(db_path)

    print(f"\n✓ Database: {db_path}")
    print(f"  Size: {db_path.stat().st_size / 1024:.1f} KB")

    # 1. Basic counts
    print("\n" + "-" * 70)
    print("1. BASIC STATISTICS")
    print("-" * 70)

    query = """
    SELECT
        COUNT(*) as total_contracts,
        SUM(outcome) as yes_count,
        COUNT(*) - SUM(outcome) as no_count,
        ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as yes_pct
    FROM outcomes
    """
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))

    # 2. By ticker
    print("\n" + "-" * 70)
    print("2. OUTCOMES BY TICKER")
    print("-" * 70)

    query = """
    SELECT
        ticker,
        COUNT(*) as total,
        SUM(outcome) as yes,
        COUNT(*) - SUM(outcome) as no,
        ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as yes_pct,
        COUNT(DISTINCT word) as unique_words
    FROM outcomes
    GROUP BY ticker
    ORDER BY total DESC
    """
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))

    # 3. Most common words
    print("\n" + "-" * 70)
    print("3. TOP 15 WORDS BY FREQUENCY")
    print("-" * 70)

    query = """
    SELECT
        word,
        COUNT(*) as total,
        SUM(outcome) as yes,
        COUNT(*) - SUM(outcome) as no,
        ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as yes_pct,
        GROUP_CONCAT(DISTINCT ticker) as tickers
    FROM outcomes
    GROUP BY word
    ORDER BY total DESC
    LIMIT 15
    """
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))

    # 4. Perfect prediction words (100% YES or 100% NO)
    print("\n" + "-" * 70)
    print("4. PERFECT PREDICTORS (100% YES or 100% NO, min 3 occurrences)")
    print("-" * 70)

    query = """
    SELECT
        word,
        COUNT(*) as occurrences,
        SUM(outcome) as yes_count,
        CASE
            WHEN SUM(outcome) = COUNT(*) THEN 'ALWAYS YES'
            WHEN SUM(outcome) = 0 THEN 'ALWAYS NO'
        END as pattern,
        GROUP_CONCAT(DISTINCT ticker) as tickers
    FROM outcomes
    GROUP BY word
    HAVING (SUM(outcome) = COUNT(*) OR SUM(outcome) = 0) AND COUNT(*) >= 3
    ORDER BY occurrences DESC
    """
    df = pd.read_sql_query(query, conn)
    if len(df) > 0:
        print(df.to_string(index=False))
    else:
        print("No perfect predictors found")

    # 5. Date range
    print("\n" + "-" * 70)
    print("5. TEMPORAL COVERAGE")
    print("-" * 70)

    query = """
    SELECT
        MIN(call_date) as earliest_call,
        MAX(call_date) as latest_call,
        COUNT(DISTINCT call_date) as unique_dates,
        COUNT(DISTINCT ticker) as unique_tickers
    FROM outcomes
    WHERE call_date IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))

    # 6. Sample records
    print("\n" + "-" * 70)
    print("6. SAMPLE RECORDS (5 YES, 5 NO)")
    print("-" * 70)

    print("\nYES outcomes (mentioned):")
    query = """
    SELECT ticker, word, call_date, final_price
    FROM outcomes
    WHERE outcome = 1
    ORDER BY RANDOM()
    LIMIT 5
    """
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))

    print("\nNO outcomes (not mentioned):")
    query = """
    SELECT ticker, word, call_date, final_price
    FROM outcomes
    WHERE outcome = 0
    ORDER BY RANDOM()
    LIMIT 5
    """
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))

    # 7. Example queries for users
    print("\n" + "-" * 70)
    print("7. EXAMPLE QUERIES")
    print("-" * 70)

    print("\nFind all outcomes for a specific ticker (e.g., META):")
    print("  SELECT * FROM outcomes WHERE ticker='META' LIMIT 10;")

    print("\nFind all outcomes for a specific word (e.g., 'AI'):")
    print("  SELECT ticker, call_date, outcome FROM outcomes WHERE word LIKE '%AI%';")

    print("\nFind words mentioned in multiple companies:")
    print("""  SELECT word, COUNT(DISTINCT ticker) as num_companies
     FROM outcomes
     GROUP BY word
     HAVING num_companies > 1
     ORDER BY num_companies DESC;""")

    print("\nAnalyze mention rate by quarter:")
    print("""  SELECT
       strftime('%Y-Q', call_date) as quarter,
       COUNT(*) as total,
       SUM(outcome) as mentions,
       ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as mention_pct
     FROM outcomes
     GROUP BY quarter
     ORDER BY quarter;""")

    # Close connection
    conn.close()

    print("\n" + "=" * 70)
    print("✓ VERIFICATION COMPLETE")
    print("=" * 70)


def load_with_pandas():
    """Demonstrate loading with pandas."""
    print("\n" + "=" * 70)
    print("LOADING WITH PANDAS")
    print("=" * 70)

    csv_path = Path("data/outcomes_database/outcomes.csv")
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nOutcome distribution by ticker:")
    print(df.groupby(['ticker', 'outcome']).size().unstack(fill_value=0))

    print("\nMost common words (top 10):")
    print(df['word'].value_counts().head(10))


def main():
    """Main entry point."""
    verify_database()
    load_with_pandas()

    print("\n" + "=" * 70)
    print("DATABASE FILES")
    print("=" * 70)

    db_dir = Path("data/outcomes_database")
    if db_dir.exists():
        for file in sorted(db_dir.glob("*")):
            size_kb = file.stat().st_size / 1024
            print(f"  {file.name:20s} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
