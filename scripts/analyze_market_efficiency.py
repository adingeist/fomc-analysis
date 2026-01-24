"""
Analyze Market Efficiency from Outcomes Database.

This script analyzes how efficiently Kalshi markets priced earnings mention contracts by:
1. Identifying words where outcomes were highly predictable (potential for mispricing)
2. Measuring consistency of outcomes (easier for market to price correctly)
3. Finding patterns that might indicate market inefficiencies
4. Estimating theoretical edge if market was at 50/50 (uninformed prior)

Note: This analysis uses historical outcomes only. To measure actual market
prices and profitability, we would need historical candlestick data from Kalshi.
However, this analysis identifies which words/tickers had exploitable patterns.
"""

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
import json


class MarketEfficiencyAnalyzer:
    """Analyze market efficiency from outcomes database."""

    def __init__(self, db_path: Path = Path("data/outcomes_database/outcomes.db")):
        """Initialize analyzer."""
        if not db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {db_path}\n"
                "Run 'python scripts/build_outcomes_database.py' first"
            )

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def close(self):
        """Close database connection."""
        self.conn.close()

    def calculate_information_entropy(self) -> pd.DataFrame:
        """
        Calculate information entropy for each word.

        Low entropy = predictable = easier to price correctly
        High entropy = unpredictable = harder to price

        Entropy = -p*log(p) - (1-p)*log(1-p)
        where p = probability of YES

        Returns:
            DataFrame with entropy metrics
        """
        query = """
        WITH word_probs AS (
            SELECT
                word,
                COUNT(*) as n,
                CAST(SUM(outcome) AS REAL) / COUNT(*) as p_yes,
                GROUP_CONCAT(DISTINCT ticker) as tickers
            FROM outcomes
            GROUP BY word
            HAVING COUNT(*) >= 3
        )
        SELECT
            word,
            n as occurrences,
            ROUND(p_yes * 100, 1) as yes_pct,
            -- Information entropy in bits
            ROUND(
                CASE
                    WHEN p_yes = 0 OR p_yes = 1 THEN 0
                    ELSE -(p_yes * LOG(p_yes) / LOG(2) +
                           (1-p_yes) * LOG(1-p_yes) / LOG(2))
                END,
                3
            ) as entropy_bits,
            -- Predictability score (inverse of entropy)
            ROUND(
                1 - CASE
                    WHEN p_yes = 0 OR p_yes = 1 THEN 0
                    ELSE -(p_yes * LOG(p_yes) / LOG(2) +
                           (1-p_yes) * LOG(1-p_yes) / LOG(2))
                END,
                3
            ) as predictability,
            tickers
        FROM word_probs
        ORDER BY entropy_bits ASC, occurrences DESC
        """

        df = pd.read_sql_query(query, self.conn)
        return df

    def identify_streaks(self) -> pd.DataFrame:
        """
        Identify words with consistent streaks (always YES or always NO).

        Long streaks suggest high predictability and potential for edge
        if market doesn't price them at extremes.

        Returns:
            DataFrame with streak information
        """
        query = """
        SELECT
            word,
            COUNT(*) as occurrences,
            SUM(outcome) as yes_count,
            ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as yes_pct,
            GROUP_CONCAT(DISTINCT ticker) as tickers,
            CASE
                WHEN SUM(outcome) = COUNT(*) THEN COUNT(*)  -- All YES
                WHEN SUM(outcome) = 0 THEN COUNT(*)  -- All NO
                ELSE 0
            END as streak_length,
            CASE
                WHEN SUM(outcome) = COUNT(*) THEN 'YES'
                WHEN SUM(outcome) = 0 THEN 'NO'
                ELSE 'MIXED'
            END as streak_type
        FROM outcomes
        GROUP BY word
        HAVING COUNT(*) >= 3 AND streak_length > 0
        ORDER BY streak_length DESC, occurrences DESC
        """

        df = pd.read_sql_query(query, self.conn)
        return df

    def calculate_theoretical_edge(self, assumed_market_price: float = 50.0) -> pd.DataFrame:
        """
        Calculate theoretical edge if market consistently priced at a given level.

        This shows how much profit could have been made if:
        - Market always priced at 50 cents (uninformed prior)
        - We knew the historical base rates

        Parameters:
            assumed_market_price: Assumed market price in cents (default 50)

        Returns:
            DataFrame with theoretical P&L
        """
        query = f"""
        WITH word_stats AS (
            SELECT
                word,
                COUNT(*) as occurrences,
                SUM(outcome) as yes_count,
                ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as yes_pct,
                GROUP_CONCAT(DISTINCT ticker) as tickers
            FROM outcomes
            GROUP BY word
            HAVING COUNT(*) >= 3
        )
        SELECT
            word,
            occurrences,
            yes_pct,
            tickers,
            -- If we bought YES at {assumed_market_price} cents each time
            CASE
                WHEN yes_pct >= 60 THEN
                    -- Buy YES: profit = (yes_count * 99) - (occurrences * {assumed_market_price})
                    ROUND((yes_count * 99.0 - occurrences * {assumed_market_price}) / 100.0, 2)
                WHEN yes_pct <= 40 THEN
                    -- Buy NO: profit = ((occurrences - yes_count) * 99) - (occurrences * {assumed_market_price})
                    ROUND(((occurrences - yes_count) * 99.0 - occurrences * {assumed_market_price}) / 100.0, 2)
                ELSE 0
            END as theoretical_profit_dollars,
            -- Profit per occurrence
            CASE
                WHEN yes_pct >= 60 THEN
                    ROUND((yes_count * 99.0 - occurrences * {assumed_market_price}) / (occurrences * 100.0), 2)
                WHEN yes_pct <= 40 THEN
                    ROUND(((occurrences - yes_count) * 99.0 - occurrences * {assumed_market_price}) / (occurrences * 100.0), 2)
                ELSE 0
            END as profit_per_contract,
            CASE
                WHEN yes_pct >= 60 THEN 'BUY_YES'
                WHEN yes_pct <= 40 THEN 'BUY_NO'
                ELSE 'AVOID'
            END as strategy
        FROM word_stats
        WHERE yes_pct >= 60 OR yes_pct <= 40
        ORDER BY ABS(theoretical_profit_dollars) DESC
        """

        df = pd.read_sql_query(query, self.conn)
        return df

    def compare_tickers_efficiency(self) -> pd.DataFrame:
        """
        Compare predictability across different tickers.

        Some companies may have more predictable language patterns,
        making their markets easier to trade profitably.

        Returns:
            DataFrame with per-ticker efficiency metrics
        """
        query = """
        WITH ticker_stats AS (
            SELECT
                ticker,
                COUNT(*) as total_contracts,
                COUNT(DISTINCT word) as unique_words,
                -- Proportion of perfect predictions (0% or 100%)
                SUM(CASE
                    WHEN outcome = 1 THEN 1
                    WHEN outcome = 0 THEN 1
                    ELSE 0
                END) * 1.0 / COUNT(*) as outcome_rate,
                -- Average outcome (higher = more mentions)
                ROUND(AVG(outcome) * 100, 1) as avg_mention_pct
            FROM outcomes
            GROUP BY ticker
        ),
        perfect_words AS (
            SELECT
                ticker,
                COUNT(DISTINCT word) as num_perfect_words
            FROM (
                SELECT
                    ticker,
                    word,
                    COUNT(*) as n,
                    SUM(outcome) as yes
                FROM outcomes
                GROUP BY ticker, word
                HAVING n >= 2 AND (yes = 0 OR yes = n)
            )
            GROUP BY ticker
        )
        SELECT
            t.ticker,
            t.total_contracts,
            t.unique_words,
            COALESCE(p.num_perfect_words, 0) as perfect_words,
            ROUND(COALESCE(p.num_perfect_words, 0) * 100.0 / t.unique_words, 1) as perfect_pct,
            t.avg_mention_pct,
            -- Predictability score: higher = more tradeable
            ROUND(COALESCE(p.num_perfect_words, 0) * 1.0 / t.total_contracts * 100, 1) as predictability_score
        FROM ticker_stats t
        LEFT JOIN perfect_words p ON t.ticker = p.ticker
        ORDER BY predictability_score DESC
        """

        df = pd.read_sql_query(query, self.conn)
        return df

    def find_exploitable_patterns(self) -> pd.DataFrame:
        """
        Find the most exploitable patterns in the data.

        A pattern is exploitable if:
        1. It's consistent (low variance)
        2. It's not at 50/50 (has directional bias)
        3. It occurs frequently enough to be confident

        Returns:
            DataFrame with exploitability scores
        """
        query = """
        WITH word_metrics AS (
            SELECT
                word,
                COUNT(*) as n,
                SUM(outcome) as yes,
                ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as yes_pct,
                GROUP_CONCAT(DISTINCT ticker) as tickers,
                -- Distance from 50% (higher = more biased)
                ABS(50 - ROUND(100.0 * SUM(outcome) / COUNT(*), 1)) as bias,
                -- Variance (lower = more consistent)
                ROUND(
                    SQRT((SUM(outcome) * 1.0 / COUNT(*)) *
                         (1 - SUM(outcome) * 1.0 / COUNT(*)))
                * 100, 1) as std_dev
            FROM outcomes
            GROUP BY word
            HAVING COUNT(*) >= 3
        )
        SELECT
            word,
            n as occurrences,
            yes_pct,
            bias as edge_magnitude,
            std_dev as uncertainty,
            tickers,
            -- Exploitability score: bias / uncertainty * log(n)
            ROUND(
                CASE
                    WHEN std_dev > 0 THEN
                        (bias / std_dev) * LOG(n) / 10
                    ELSE 0
                END,
                2
            ) as exploitability_score,
            CASE
                WHEN yes_pct >= 70 THEN 'STRONG_YES'
                WHEN yes_pct >= 60 THEN 'LEAN_YES'
                WHEN yes_pct <= 30 THEN 'STRONG_NO'
                WHEN yes_pct <= 40 THEN 'LEAN_NO'
                ELSE 'NEUTRAL'
            END as classification
        FROM word_metrics
        WHERE std_dev > 0  -- Exclude perfect predictors
        ORDER BY exploitability_score DESC
        LIMIT 30
        """

        df = pd.read_sql_query(query, self.conn)
        return df

    def generate_report(self, output_path: Path = Path("data/market_efficiency_report.json")):
        """
        Generate comprehensive market efficiency report.

        Returns:
            Dict with all analysis results
        """
        report = {
            "metadata": {
                "generated_at": pd.Timestamp.now().isoformat(),
                "database": str(self.db_path),
                "note": "This analysis uses outcomes only. Historical prices would be needed for actual profitability analysis.",
            },
            "entropy_analysis": self.calculate_information_entropy().to_dict(orient="records"),
            "perfect_streaks": self.identify_streaks().to_dict(orient="records"),
            "theoretical_edge": self.calculate_theoretical_edge().to_dict(orient="records"),
            "ticker_efficiency": self.compare_tickers_efficiency().to_dict(orient="records"),
            "exploitable_patterns": self.find_exploitable_patterns().to_dict(orient="records"),
        }

        # Calculate summary statistics
        entropy_df = pd.DataFrame(report["entropy_analysis"])
        streaks_df = pd.DataFrame(report["perfect_streaks"])
        edge_df = pd.DataFrame(report["theoretical_edge"])

        report["summary"] = {
            "most_predictable_words": entropy_df.nsmallest(10, "entropy_bits")[["word", "entropy_bits", "yes_pct"]].to_dict(orient="records"),
            "longest_streaks": streaks_df.nlargest(10, "streak_length")[["word", "streak_length", "streak_type"]].to_dict(orient="records"),
            "highest_theoretical_profit": edge_df.nlargest(10, "theoretical_profit_dollars")[
                ["word", "theoretical_profit_dollars", "strategy", "yes_pct"]
            ].to_dict(orient="records"),
        }

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"✓ Saved market efficiency report: {output_path}")

        return report


def print_summary(analyzer: MarketEfficiencyAnalyzer):
    """Print summary to console."""
    print("=" * 80)
    print("MARKET EFFICIENCY ANALYSIS")
    print("=" * 80)

    # 1. Most predictable (low entropy)
    print("\n1. MOST PREDICTABLE WORDS (Low Entropy)")
    print("-" * 80)
    df = analyzer.calculate_information_entropy().head(15)
    print(df[["word", "occurrences", "yes_pct", "entropy_bits", "predictability"]].to_string(index=False))

    # 2. Perfect streaks
    print("\n\n2. PERFECT PREDICTION STREAKS")
    print("-" * 80)
    df = analyzer.identify_streaks().head(15)
    print(df[["word", "occurrences", "streak_type", "streak_length", "tickers"]].to_string(index=False))

    # 3. Theoretical edge
    print("\n\n3. THEORETICAL PROFIT (if market at 50 cents)")
    print("-" * 80)
    df = analyzer.calculate_theoretical_edge().head(15)
    print(df[["word", "occurrences", "yes_pct", "strategy", "theoretical_profit_dollars", "profit_per_contract"]].to_string(index=False))

    total_profit = df["theoretical_profit_dollars"].sum()
    print(f"\nTotal theoretical profit (top 15 words): ${total_profit:.2f}")
    print("Note: Assumes market always at 50 cents, which is unrealistic")

    # 4. Ticker efficiency
    print("\n\n4. TICKER PREDICTABILITY COMPARISON")
    print("-" * 80)
    df = analyzer.compare_tickers_efficiency()
    print(df.to_string(index=False))

    # 5. Most exploitable
    print("\n\n5. MOST EXPLOITABLE PATTERNS")
    print("-" * 80)
    df = analyzer.find_exploitable_patterns().head(15)
    print(df[["word", "yes_pct", "edge_magnitude", "uncertainty", "exploitability_score", "classification"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    # Count perfect predictors
    streaks = analyzer.identify_streaks()
    print(f"\n• {len(streaks)} words with perfect prediction streaks")
    print(f"• {len(streaks[streaks['streak_type'] == 'YES'])} always mentioned")
    print(f"• {len(streaks[streaks['streak_type'] == 'NO'])} never mentioned")

    # Entropy stats
    entropy = analyzer.calculate_information_entropy()
    low_entropy = entropy[entropy["entropy_bits"] < 0.5]
    print(f"\n• {len(low_entropy)} words with very low entropy (< 0.5 bits)")
    print(f"  These are highly predictable and easy to price correctly")

    # Ticker comparison
    tickers = analyzer.compare_tickers_efficiency()
    best_ticker = tickers.iloc[0]
    print(f"\n• Most predictable ticker: {best_ticker['ticker']}")
    print(f"  {best_ticker['perfect_pct']:.1f}% of words are perfectly predictable")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    print("Analyzing market efficiency from outcomes database...\n")

    analyzer = MarketEfficiencyAnalyzer()

    try:
        # Print summary
        print_summary(analyzer)

        # Generate report
        print("\nGenerating comprehensive report...")
        report = analyzer.generate_report()

        print(f"\nReport includes:")
        print(f"  - {len(report['entropy_analysis'])} words analyzed for entropy")
        print(f"  - {len(report['perfect_streaks'])} perfect prediction streaks")
        print(f"  - {len(report['theoretical_edge'])} words with theoretical edge")
        print(f"  - {len(report['ticker_efficiency'])} tickers compared")
        print(f"  - {len(report['exploitable_patterns'])} exploitable patterns identified")

    finally:
        analyzer.close()

    return 0


if __name__ == "__main__":
    exit(main())
