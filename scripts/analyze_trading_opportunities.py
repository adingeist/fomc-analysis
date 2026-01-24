"""
Analyze Trading Opportunities from Historical Outcomes.

This script analyzes the outcomes database to identify:
1. Predictable words with exploitable patterns
2. Market inefficiencies where edge likely exists
3. Optimal ticker/word combinations for trading
4. Risk-adjusted opportunity scoring

Uses actual Kalshi settlement data to find profitable patterns.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import json


class TradingOpportunityAnalyzer:
    """Analyze outcomes database for trading opportunities."""

    def __init__(self, db_path: Path = Path("data/outcomes_database/outcomes.db")):
        """Initialize analyzer with database connection."""
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

    def find_high_confidence_words(
        self, min_occurrences: int = 3, confidence_threshold: float = 0.85
    ) -> pd.DataFrame:
        """
        Find words with high confidence (>85% or <15% mention rate).

        These represent near-certain outcomes where edge exists if market misprices.

        Parameters:
            min_occurrences: Minimum times word must appear
            confidence_threshold: Threshold for "high confidence" (0.85 = 85%)

        Returns:
            DataFrame with high-confidence words and expected profit
        """
        query = f"""
        SELECT
            word,
            COUNT(*) as occurrences,
            SUM(outcome) as yes_count,
            ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as yes_pct,
            GROUP_CONCAT(DISTINCT ticker) as tickers,
            CASE
                WHEN SUM(outcome) >= {confidence_threshold} * COUNT(*)
                    THEN 'BUY_YES'
                WHEN SUM(outcome) <= {1-confidence_threshold} * COUNT(*)
                    THEN 'BUY_NO'
            END as trade_direction,
            -- Expected profit if market is at 50/50
            CASE
                WHEN SUM(outcome) >= {confidence_threshold} * COUNT(*)
                    THEN ROUND((100.0 * SUM(outcome) / COUNT(*)) - 50, 1)
                WHEN SUM(outcome) <= {1-confidence_threshold} * COUNT(*)
                    THEN ROUND(50 - (100.0 * SUM(outcome) / COUNT(*)), 1)
                ELSE 0
            END as expected_edge_pct
        FROM outcomes
        GROUP BY word
        HAVING COUNT(*) >= {min_occurrences}
            AND (SUM(outcome) >= {confidence_threshold} * COUNT(*)
                 OR SUM(outcome) <= {1-confidence_threshold} * COUNT(*))
        ORDER BY occurrences DESC, ABS(50 - yes_pct) DESC
        """

        df = pd.read_sql_query(query, self.conn)
        return df

    def find_ticker_specific_patterns(self, ticker: str) -> pd.DataFrame:
        """
        Find words that are highly predictable for a specific ticker.

        Parameters:
            ticker: Company ticker (e.g., 'META', 'TSLA')

        Returns:
            DataFrame with ticker-specific opportunities
        """
        query = f"""
        SELECT
            word,
            COUNT(*) as occurrences,
            SUM(outcome) as yes_count,
            ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as yes_pct,
            CASE
                WHEN SUM(outcome) = COUNT(*) THEN 'ALWAYS_YES'
                WHEN SUM(outcome) = 0 THEN 'ALWAYS_NO'
                WHEN SUM(outcome) >= 0.8 * COUNT(*) THEN 'USUALLY_YES'
                WHEN SUM(outcome) <= 0.2 * COUNT(*) THEN 'USUALLY_NO'
                ELSE 'VARIABLE'
            END as pattern,
            CASE
                WHEN SUM(outcome) = COUNT(*) THEN 99  -- Always YES, buy at <99
                WHEN SUM(outcome) = 0 THEN 1  -- Always NO, buy NO at <99 (or avoid YES)
                WHEN SUM(outcome) >= 0.8 * COUNT(*)
                    THEN ROUND(100.0 * SUM(outcome) / COUNT(*), 0)
                WHEN SUM(outcome) <= 0.2 * COUNT(*)
                    THEN ROUND(100.0 * (COUNT(*) - SUM(outcome)) / COUNT(*), 0)
                ELSE 50
            END as fair_value_cents
        FROM outcomes
        WHERE ticker = '{ticker}'
        GROUP BY word
        HAVING COUNT(*) >= 2
        ORDER BY
            CASE pattern
                WHEN 'ALWAYS_YES' THEN 1
                WHEN 'ALWAYS_NO' THEN 2
                WHEN 'USUALLY_YES' THEN 3
                WHEN 'USUALLY_NO' THEN 4
                ELSE 5
            END,
            occurrences DESC
        """

        df = pd.read_sql_query(query, self.conn)
        return df

    def calculate_word_volatility(self, min_occurrences: int = 3) -> pd.DataFrame:
        """
        Calculate volatility (variance) for each word.

        High volatility = unpredictable = hard to trade
        Low volatility = predictable = easier to trade

        Returns:
            DataFrame with volatility metrics
        """
        query = f"""
        SELECT
            word,
            COUNT(*) as occurrences,
            ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as yes_pct,
            -- Standard deviation of binary outcomes
            ROUND(SQRT(
                (SUM(outcome) * 1.0 / COUNT(*)) *
                (1 - SUM(outcome) * 1.0 / COUNT(*))
            ) * 100, 1) as volatility_pct,
            GROUP_CONCAT(DISTINCT ticker) as tickers,
            CASE
                WHEN SQRT((SUM(outcome) * 1.0 / COUNT(*)) * (1 - SUM(outcome) * 1.0 / COUNT(*))) < 0.3
                    THEN 'LOW_VOLATILITY'
                WHEN SQRT((SUM(outcome) * 1.0 / COUNT(*)) * (1 - SUM(outcome) * 1.0 / COUNT(*))) < 0.45
                    THEN 'MEDIUM_VOLATILITY'
                ELSE 'HIGH_VOLATILITY'
            END as risk_category
        FROM outcomes
        GROUP BY word
        HAVING COUNT(*) >= {min_occurrences}
        ORDER BY volatility_pct ASC, occurrences DESC
        """

        df = pd.read_sql_query(query, self.conn)
        return df

    def identify_cross_ticker_opportunities(self) -> pd.DataFrame:
        """
        Find words that behave differently across tickers.

        These represent opportunities to trade the same word differently
        depending on the company.

        Returns:
            DataFrame with cross-ticker patterns
        """
        query = """
        WITH word_ticker_stats AS (
            SELECT
                word,
                ticker,
                COUNT(*) as occurrences,
                ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as yes_pct
            FROM outcomes
            GROUP BY word, ticker
            HAVING COUNT(*) >= 2
        ),
        word_variance AS (
            SELECT
                word,
                COUNT(DISTINCT ticker) as num_tickers,
                MAX(yes_pct) - MIN(yes_pct) as pct_spread,
                GROUP_CONCAT(ticker || ':' || yes_pct || '%') as ticker_breakdown
            FROM word_ticker_stats
            GROUP BY word
            HAVING COUNT(DISTINCT ticker) >= 2
        )
        SELECT *
        FROM word_variance
        WHERE pct_spread >= 30  -- At least 30% difference between tickers
        ORDER BY pct_spread DESC, num_tickers DESC
        """

        df = pd.read_sql_query(query, self.conn)
        return df

    def score_trading_opportunities(self) -> pd.DataFrame:
        """
        Score all words by trading opportunity quality.

        Scoring factors:
        - Predictability (low variance = good)
        - Sample size (more data = more confident)
        - Edge magnitude (far from 50% = more profit potential)

        Returns:
            DataFrame with scored opportunities
        """
        query = """
        WITH word_stats AS (
            SELECT
                word,
                COUNT(*) as occurrences,
                SUM(outcome) as yes_count,
                ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as yes_pct,
                GROUP_CONCAT(DISTINCT ticker) as tickers,
                -- Variance (lower = better)
                ROUND(SQRT(
                    (SUM(outcome) * 1.0 / COUNT(*)) *
                    (1 - SUM(outcome) * 1.0 / COUNT(*))
                ) * 100, 1) as volatility,
                -- Distance from 50% (higher = more edge)
                ABS(50 - ROUND(100.0 * SUM(outcome) / COUNT(*), 1)) as edge_magnitude
            FROM outcomes
            GROUP BY word
            HAVING COUNT(*) >= 3
        )
        SELECT
            word,
            occurrences,
            yes_pct,
            volatility,
            edge_magnitude,
            tickers,
            -- Composite score: edge / volatility * log(occurrences)
            ROUND(
                (edge_magnitude / NULLIF(volatility, 0)) *
                (1 + LOG(occurrences)) / 10,
                2
            ) as opportunity_score,
            CASE
                WHEN yes_pct >= 70 THEN 'BUY_YES'
                WHEN yes_pct <= 30 THEN 'BUY_NO'
                ELSE 'AVOID'
            END as suggested_trade
        FROM word_stats
        WHERE volatility > 0  -- Exclude perfect predictors from scoring
        ORDER BY opportunity_score DESC
        LIMIT 50
        """

        df = pd.read_sql_query(query, self.conn)
        return df

    def generate_report(self, output_path: Path = Path("data/trading_opportunities_report.json")):
        """
        Generate comprehensive trading opportunities report.

        Returns:
            Dict with all analysis results
        """
        report = {
            "metadata": {
                "generated_at": pd.Timestamp.now().isoformat(),
                "database": str(self.db_path),
            },
            "high_confidence_words": self.find_high_confidence_words().to_dict(orient="records"),
            "volatility_analysis": self.calculate_word_volatility().to_dict(orient="records"),
            "cross_ticker_opportunities": self.identify_cross_ticker_opportunities().to_dict(orient="records"),
            "top_opportunities": self.score_trading_opportunities().to_dict(orient="records"),
        }

        # Add per-ticker analysis
        tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM outcomes", self.conn)["ticker"].tolist()
        report["ticker_specific"] = {}

        for ticker in tickers:
            report["ticker_specific"][ticker] = self.find_ticker_specific_patterns(ticker).to_dict(orient="records")

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"✓ Saved trading opportunities report: {output_path}")

        return report


def print_summary(analyzer: TradingOpportunityAnalyzer):
    """Print summary of trading opportunities to console."""
    print("=" * 80)
    print("TRADING OPPORTUNITIES ANALYSIS")
    print("=" * 80)

    # 1. High confidence words
    print("\n1. HIGH CONFIDENCE OPPORTUNITIES (>85% predictable)")
    print("-" * 80)
    df = analyzer.find_high_confidence_words()
    if len(df) > 0:
        print(df[["word", "occurrences", "yes_pct", "trade_direction", "tickers"]].head(15).to_string(index=False))
    else:
        print("No high-confidence opportunities found")

    # 2. Low volatility (easy to trade)
    print("\n\n2. LOW VOLATILITY WORDS (Most Predictable)")
    print("-" * 80)
    df = analyzer.calculate_word_volatility()
    low_vol = df[df["risk_category"] == "LOW_VOLATILITY"].head(15)
    if len(low_vol) > 0:
        print(low_vol[["word", "occurrences", "yes_pct", "volatility_pct", "tickers"]].to_string(index=False))
    else:
        print("No low-volatility words found")

    # 3. Cross-ticker opportunities
    print("\n\n3. CROSS-TICKER OPPORTUNITIES (Different behavior by company)")
    print("-" * 80)
    df = analyzer.identify_cross_ticker_opportunities()
    if len(df) > 0:
        print(df.head(10).to_string(index=False))
    else:
        print("No significant cross-ticker differences found")

    # 4. Top scored opportunities
    print("\n\n4. TOP TRADING OPPORTUNITIES (Risk-Adjusted Score)")
    print("-" * 80)
    df = analyzer.score_trading_opportunities()
    if len(df) > 0:
        print(df[["word", "yes_pct", "volatility", "edge_magnitude", "opportunity_score", "suggested_trade"]].head(20).to_string(index=False))
    else:
        print("No scored opportunities found")

    # 5. Example ticker-specific
    print("\n\n5. EXAMPLE: META-SPECIFIC PATTERNS")
    print("-" * 80)
    df = analyzer.find_ticker_specific_patterns("META")
    if len(df) > 0:
        print(df[["word", "occurrences", "yes_pct", "pattern", "fair_value_cents"]].head(15).to_string(index=False))

    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 80)


def main():
    """Main entry point."""
    print("Analyzing trading opportunities from outcomes database...\n")

    analyzer = TradingOpportunityAnalyzer()

    try:
        # Print summary to console
        print_summary(analyzer)

        # Generate full report
        print("\nGenerating comprehensive report...")
        report = analyzer.generate_report()

        print(f"\nReport includes:")
        print(f"  - {len(report['high_confidence_words'])} high-confidence words")
        print(f"  - {len(report['volatility_analysis'])} volatility-analyzed words")
        print(f"  - {len(report['cross_ticker_opportunities'])} cross-ticker opportunities")
        print(f"  - {len(report['top_opportunities'])} top-scored opportunities")
        print(f"  - {len(report['ticker_specific'])} ticker-specific analyses")

        print("\n" + "=" * 80)
        print("ACTIONABLE INSIGHTS")
        print("=" * 80)

        # Print top 5 actionable insights
        top_ops = pd.DataFrame(report["top_opportunities"]).head(5)
        if len(top_ops) > 0:
            print("\nTop 5 Words to Trade:")
            for idx, row in top_ops.iterrows():
                print(f"\n{idx + 1}. {row['word']}")
                print(f"   Suggestion: {row['suggested_trade']}")
                print(f"   Historical Rate: {row['yes_pct']}%")
                print(f"   Edge: {row['edge_magnitude']}%")
                print(f"   Risk: {row['volatility']}% volatility")
                print(f"   Score: {row['opportunity_score']}")
                print(f"   Tickers: {row['tickers']}")

    finally:
        analyzer.close()

    return 0


if __name__ == "__main__":
    exit(main())
