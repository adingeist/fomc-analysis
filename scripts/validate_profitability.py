"""
Validate Profitability of Trading Strategy.

Compares theoretical opportunities (from outcomes analysis) against actual
historical market prices to determine:
1. Which opportunities actually existed (market mispricing)
2. What actual returns would have been achieved
3. Whether theoretical edge translated to real edge

Uses entry rules from ACTIONABLE_INSIGHTS.md to calculate P&L.
"""

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json


class ProfitabilityValidator:
    """Validate profitability using historical prices and outcomes."""

    def __init__(
        self,
        outcomes_db: Path = Path("data/outcomes_database/outcomes.db"),
        prices_dir: Path = Path("data/historical_prices"),
    ):
        """Initialize validator."""
        if not outcomes_db.exists():
            raise FileNotFoundError(f"Outcomes database not found: {outcomes_db}")

        if not prices_dir.exists():
            raise FileNotFoundError(
                f"Prices directory not found: {prices_dir}\n"
                "Run 'python scripts/fetch_historical_prices.py' first"
            )

        # Load outcomes
        conn = sqlite3.connect(outcomes_db)
        self.outcomes_df = pd.read_sql_query("SELECT * FROM outcomes", conn)
        conn.close()

        # Load price summary
        price_summary_path = prices_dir / "price_summary.csv"
        if price_summary_path.exists():
            self.prices_df = pd.read_csv(price_summary_path)
        else:
            raise FileNotFoundError(
                f"Price summary not found: {price_summary_path}\n"
                "Run 'python scripts/fetch_historical_prices.py' first"
            )

        self.prices_dir = prices_dir

        print(f"Loaded {len(self.outcomes_df)} outcomes")
        print(f"Loaded {len(self.prices_df)} price histories")

    def calculate_perfect_predictor_performance(self) -> pd.DataFrame:
        """
        Calculate performance on perfect predictors (100% YES or NO).

        Entry rules (from ACTIONABLE_INSIGHTS.md):
        - For 100% YES words: Buy YES at < 85 cents
        - For 100% NO words: Buy NO at < 85 cents (YES price < 15 cents)

        Returns:
            DataFrame with per-word performance
        """
        # Identify perfect predictors
        word_stats = self.outcomes_df.groupby('word').agg({
            'outcome': ['count', 'sum', 'mean']
        }).reset_index()
        word_stats.columns = ['word', 'count', 'yes_sum', 'yes_rate']

        # Perfect predictors: 100% YES or 100% NO, min 3 occurrences
        perfect_yes = word_stats[(word_stats['yes_rate'] == 1.0) & (word_stats['count'] >= 3)]
        perfect_no = word_stats[(word_stats['yes_rate'] == 0.0) & (word_stats['count'] >= 3)]

        print(f"\nPerfect predictors:")
        print(f"  YES: {len(perfect_yes)} words")
        print(f"  NO: {len(perfect_no)} words")

        results = []

        # Analyze each perfect predictor
        for word in perfect_yes['word']:
            perf = self._analyze_word_performance(
                word,
                expected_outcome='YES',
                entry_threshold=85,  # Buy YES if < 85 cents
            )
            if perf:
                results.append(perf)

        for word in perfect_no['word']:
            perf = self._analyze_word_performance(
                word,
                expected_outcome='NO',
                entry_threshold=15,  # Buy YES if < 15 cents (bet NO)
            )
            if perf:
                results.append(perf)

        return pd.DataFrame(results)

    def calculate_high_confidence_performance(self) -> pd.DataFrame:
        """
        Calculate performance on high-confidence words (>75% bias).

        Entry rules:
        - For >75% YES: Buy YES at < 60 cents
        - For <25% YES: Buy NO at < 75 cents (YES < 25 cents)

        Returns:
            DataFrame with per-word performance
        """
        # Calculate word stats
        word_stats = self.outcomes_df.groupby('word').agg({
            'outcome': ['count', 'sum', 'mean']
        }).reset_index()
        word_stats.columns = ['word', 'count', 'yes_sum', 'yes_rate']

        # High confidence: 75-99% or 1-25%, min 3 occurrences
        high_yes = word_stats[
            (word_stats['yes_rate'] >= 0.75) &
            (word_stats['yes_rate'] < 1.0) &
            (word_stats['count'] >= 3)
        ]
        high_no = word_stats[
            (word_stats['yes_rate'] <= 0.25) &
            (word_stats['yes_rate'] > 0.0) &
            (word_stats['count'] >= 3)
        ]

        print(f"\nHigh-confidence words:")
        print(f"  YES bias: {len(high_yes)} words")
        print(f"  NO bias: {len(high_no)} words")

        results = []

        for word in high_yes['word']:
            perf = self._analyze_word_performance(
                word,
                expected_outcome='YES',
                entry_threshold=60,  # More conservative for non-perfect
            )
            if perf:
                results.append(perf)

        for word in high_no['word']:
            perf = self._analyze_word_performance(
                word,
                expected_outcome='NO',
                entry_threshold=40,  # Buy YES < 40 cents (bet NO at > 60)
            )
            if perf:
                results.append(perf)

        return pd.DataFrame(results)

    def _analyze_word_performance(
        self,
        word: str,
        expected_outcome: str,
        entry_threshold: int,
    ) -> Dict:
        """
        Analyze performance for a specific word.

        Parameters:
            word: Word to analyze
            expected_outcome: 'YES' or 'NO'
            entry_threshold: Max price to enter (cents)

        Returns:
            Dict with performance metrics
        """
        # Get all contracts for this word
        word_contracts = self.outcomes_df[self.outcomes_df['word'] == word]

        if len(word_contracts) == 0:
            return None

        # Merge with price data
        merged = word_contracts.merge(
            self.prices_df,
            on='contract_ticker',
            how='inner',
            suffixes=('', '_price')
        )

        if len(merged) == 0:
            return None

        # Calculate trades
        trades = []
        for _, row in merged.iterrows():
            contract_ticker = row['contract_ticker']
            outcome = row['outcome']
            avg_price = row['avg_price']
            first_price = row['first_price']
            min_price = row['min_price']
            final_price = row['final_price']

            # Determine entry price (use average as proxy for "could have entered")
            entry_price = avg_price

            # Would we have entered?
            if expected_outcome == 'YES':
                # Buy YES
                entered = entry_price < entry_threshold
                if entered:
                    # P&L: (final - entry) if YES, else -entry
                    if outcome == 1:  # YES
                        pnl = (final_price - entry_price) / 100.0
                    else:  # NO (we lose)
                        pnl = -entry_price / 100.0
                    trades.append({
                        'contract': contract_ticker,
                        'entry': entry_price,
                        'outcome': outcome,
                        'pnl': pnl,
                        'won': outcome == 1,
                    })
            else:  # expected_outcome == 'NO'
                # Buy NO (which means YES price should be low)
                entered = entry_price < entry_threshold
                if entered:
                    # If we're buying NO, and actual outcome is NO (0), we win
                    if outcome == 0:  # NO (we win)
                        pnl = (99 - (100 - entry_price)) / 100.0  # Win on NO side
                        pnl = (100 - entry_price - 1) / 100.0  # Simplified
                    else:  # YES (we lose)
                        pnl = -(100 - entry_price) / 100.0
                    trades.append({
                        'contract': contract_ticker,
                        'entry': entry_price,
                        'outcome': outcome,
                        'pnl': pnl,
                        'won': outcome == 0,
                    })

        if len(trades) == 0:
            return {
                'word': word,
                'expected_outcome': expected_outcome,
                'num_contracts': len(merged),
                'num_trades': 0,
                'trades_entered': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl_per_trade': 0.0,
                'avg_entry_price': 0.0,
                'roi_pct': 0.0,
            }

        # Calculate metrics
        trades_df = pd.DataFrame(trades)
        num_wins = trades_df['won'].sum()
        total_pnl = trades_df['pnl'].sum()
        total_risked = len(trades) * (entry_threshold / 100.0)  # Approximate

        return {
            'word': word,
            'expected_outcome': expected_outcome,
            'num_contracts': len(merged),
            'num_trades': len(trades),
            'trades_entered': len(trades),
            'win_rate': num_wins / len(trades) if len(trades) > 0 else 0.0,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / len(trades) if len(trades) > 0 else 0.0,
            'avg_entry_price': trades_df['entry'].mean(),
            'roi_pct': (total_pnl / total_risked * 100) if total_risked > 0 else 0.0,
        }

    def generate_report(self, output_path: Path = Path("data/profitability_validation_report.json")):
        """Generate comprehensive profitability validation report."""
        print("=" * 80)
        print("PROFITABILITY VALIDATION")
        print("=" * 80)

        # Analyze perfect predictors
        print("\n[1] Analyzing perfect predictors...")
        perfect_results = self.calculate_perfect_predictor_performance()

        # Analyze high-confidence
        print("\n[2] Analyzing high-confidence words...")
        high_conf_results = self.calculate_high_confidence_performance()

        # Calculate overall metrics
        report = {
            "metadata": {
                "generated_at": pd.Timestamp.now().isoformat(),
                "num_outcomes": len(self.outcomes_df),
                "num_prices": len(self.prices_df),
            },
            "perfect_predictors": perfect_results.to_dict(orient='records') if not perfect_results.empty else [],
            "high_confidence": high_conf_results.to_dict(orient='records') if not high_conf_results.empty else [],
        }

        # Summary statistics
        if not perfect_results.empty:
            report["summary"] = {
                "perfect_predictors": {
                    "total_words": len(perfect_results),
                    "total_trades": int(perfect_results['num_trades'].sum()),
                    "avg_win_rate": float(perfect_results['win_rate'].mean()),
                    "total_pnl": float(perfect_results['total_pnl'].sum()),
                    "avg_roi_pct": float(perfect_results['roi_pct'].mean()),
                    "best_word": perfect_results.nlargest(1, 'total_pnl')['word'].iloc[0] if len(perfect_results) > 0 else None,
                    "worst_word": perfect_results.nsmallest(1, 'total_pnl')['word'].iloc[0] if len(perfect_results) > 0 else None,
                }
            }

        if not high_conf_results.empty:
            if "summary" not in report:
                report["summary"] = {}
            report["summary"]["high_confidence"] = {
                "total_words": len(high_conf_results),
                "total_trades": int(high_conf_results['num_trades'].sum()),
                "avg_win_rate": float(high_conf_results['win_rate'].mean()),
                "total_pnl": float(high_conf_results['total_pnl'].sum()),
                "avg_roi_pct": float(high_conf_results['roi_pct'].mean()),
            }

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Saved report: {output_path}")

        return report

    def print_summary(self):
        """Print summary to console."""
        # Generate report
        report = self.generate_report()

        print("\n" + "=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)

        if "perfect_predictors" in report and report["perfect_predictors"]:
            print("\nPERFECT PREDICTORS:")
            df = pd.DataFrame(report["perfect_predictors"])
            print(df[['word', 'num_trades', 'win_rate', 'total_pnl', 'avg_pnl_per_trade', 'roi_pct']].to_string(index=False))

            summary = report["summary"]["perfect_predictors"]
            print(f"\nTotals:")
            print(f"  Total trades: {summary['total_trades']}")
            print(f"  Avg win rate: {summary['avg_win_rate']:.1%}")
            print(f"  Total P&L: ${summary['total_pnl']:.2f}")
            print(f"  Avg ROI: {summary['avg_roi_pct']:.1f}%")

        if "high_confidence" in report and report["high_confidence"]:
            print("\n\nHIGH-CONFIDENCE WORDS:")
            df = pd.DataFrame(report["high_confidence"])
            if len(df) > 0:
                print(df[['word', 'num_trades', 'win_rate', 'total_pnl', 'avg_pnl_per_trade', 'roi_pct']].head(10).to_string(index=False))

                summary = report["summary"]["high_confidence"]
                print(f"\nTotals:")
                print(f"  Total trades: {summary['total_trades']}")
                print(f"  Avg win rate: {summary['avg_win_rate']:.1%}")
                print(f"  Total P&L: ${summary['total_pnl']:.2f}")
                print(f"  Avg ROI: {summary['avg_roi_pct']:.1f}%")

        print("\n" + "=" * 80)


def main():
    """Main entry point."""
    validator = ProfitabilityValidator()
    validator.print_summary()

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("\nPositive P&L → Real edge existed (market was inefficient)")
    print("Negative P&L → No edge (market was efficient or we had wrong direction)")
    print("Win rate < expected → Historical base rate not predictive")
    print("\nNote: Uses average price as entry proxy. Actual results may vary.")

    return 0


if __name__ == "__main__":
    exit(main())
