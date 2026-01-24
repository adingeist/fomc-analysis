"""
Fetch Historical Market Prices from Kalshi API.

This script fetches historical candlestick data for all contracts in the outcomes
database to determine:
1. What prices markets traded at before settlement
2. Whether theoretical opportunities actually existed
3. Entry/exit price points for profitability analysis

Uses the Kalshi API's market history endpoint to get daily close prices.
"""

import sqlite3
from pathlib import Path
import pandas as pd
import asyncio
from typing import List, Dict, Any
import json
from datetime import datetime, timedelta

from fomc_analysis.kalshi_client_factory import get_kalshi_client, KalshiSdkAdapter


class HistoricalPriceFetcher:
    """Fetch historical market prices for contracts in outcomes database."""

    def __init__(
        self,
        db_path: Path = Path("data/outcomes_database/outcomes.db"),
        output_dir: Path = Path("data/historical_prices"),
    ):
        """Initialize fetcher."""
        if not db_path.exists():
            raise FileNotFoundError(
                f"Outcomes database not found: {db_path}\n"
                "Run 'python scripts/build_outcomes_database.py' first"
            )

        self.db_path = db_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load outcomes
        conn = sqlite3.connect(db_path)
        self.outcomes_df = pd.read_sql_query("SELECT * FROM outcomes", conn)
        conn.close()

        print(f"Loaded {len(self.outcomes_df)} contracts from outcomes database")

    def fetch_price_history(self, contract_ticker: str, settled_date: str = None) -> pd.DataFrame:
        """
        Fetch historical price data for a single contract.

        Parameters:
            contract_ticker: Full Kalshi contract ticker
            settled_date: Settlement date (ISO format) - used to set end date

        Returns:
            DataFrame with date, close_price columns
        """
        client = get_kalshi_client()

        try:
            # Parse settled date if available
            if settled_date:
                try:
                    end_date = datetime.fromisoformat(settled_date.replace('Z', '+00:00'))
                    end_date_str = end_date.strftime('%Y-%m-%d')
                except:
                    end_date_str = None
            else:
                end_date_str = None

            # Fetch history (client method handles date parsing)
            history_df = client.get_market_history(
                ticker=contract_ticker,
                start_date=None,  # Get all available history
                end_date=end_date_str,
            )

            if history_df.empty:
                print(f"  ⚠ No price history for {contract_ticker}")
                return pd.DataFrame()

            # Rename column to close_price for clarity
            if contract_ticker in history_df.columns:
                history_df = history_df.rename(columns={contract_ticker: 'close_price'})
                history_df = history_df.reset_index()
                history_df.columns = ['date', 'close_price']

            return history_df

        except Exception as e:
            print(f"  ✗ Error fetching {contract_ticker}: {e}")
            return pd.DataFrame()

        finally:
            if hasattr(client, 'close'):
                try:
                    client.close()
                except:
                    pass

    def fetch_all_prices(self, max_contracts: int = None, skip_existing: bool = True):
        """
        Fetch historical prices for all contracts in outcomes database.

        Parameters:
            max_contracts: Limit number of contracts to fetch (for testing)
            skip_existing: Skip contracts that already have price data

        Returns:
            Dict mapping contract_ticker -> price DataFrame
        """
        print("=" * 80)
        print("FETCHING HISTORICAL MARKET PRICES")
        print("=" * 80)

        contracts = self.outcomes_df['contract_ticker'].unique()

        if max_contracts:
            contracts = contracts[:max_contracts]
            print(f"\n⚠ Limited to first {max_contracts} contracts for testing")

        print(f"\nFetching prices for {len(contracts)} contracts...")
        print("This may take a while (API rate limits apply)\n")

        all_prices = {}
        success_count = 0
        fail_count = 0
        skip_count = 0

        for idx, contract_ticker in enumerate(contracts, 1):
            # Check if already fetched
            price_file = self.output_dir / f"{contract_ticker}.csv"
            if skip_existing and price_file.exists():
                print(f"[{idx}/{len(contracts)}] ⊘ Skipping {contract_ticker} (already exists)")
                skip_count += 1
                continue

            print(f"[{idx}/{len(contracts)}] Fetching {contract_ticker}...")

            # Get settled date for this contract
            contract_data = self.outcomes_df[
                self.outcomes_df['contract_ticker'] == contract_ticker
            ].iloc[0]
            settled_date = contract_data.get('settled_date')

            # Fetch price history
            price_df = self.fetch_price_history(contract_ticker, settled_date)

            if not price_df.empty:
                # Save to CSV
                price_df.to_csv(price_file, index=False)
                all_prices[contract_ticker] = price_df
                success_count += 1

                # Print summary
                min_price = price_df['close_price'].min()
                max_price = price_df['close_price'].max()
                last_price = price_df['close_price'].iloc[-1]
                print(f"  ✓ Got {len(price_df)} days, range: ${min_price:.2f}-${max_price:.2f}, final: ${last_price:.2f}")

                # Save progress periodically
                if success_count % 10 == 0:
                    self._save_summary(all_prices, partial=True)

            else:
                fail_count += 1

            # Rate limiting: sleep 100ms between requests
            if idx < len(contracts):
                asyncio.run(asyncio.sleep(0.1))

        print("\n" + "=" * 80)
        print("FETCH COMPLETE")
        print("=" * 80)
        print(f"Success: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"Skipped: {skip_count}")
        print(f"Total: {len(contracts)}")

        # Save final summary
        self._save_summary(all_prices)

        return all_prices

    def _save_summary(self, all_prices: Dict[str, pd.DataFrame], partial: bool = False):
        """Save summary of fetched prices."""
        summary = []

        for contract_ticker, price_df in all_prices.items():
            if price_df.empty:
                continue

            # Get outcome data
            outcome_data = self.outcomes_df[
                self.outcomes_df['contract_ticker'] == contract_ticker
            ].iloc[0]

            summary.append({
                'contract_ticker': contract_ticker,
                'ticker': outcome_data['ticker'],
                'word': outcome_data['word'],
                'outcome': int(outcome_data['outcome']),
                'final_price': int(outcome_data['final_price']),
                'num_price_points': len(price_df),
                'min_price': float(price_df['close_price'].min()),
                'max_price': float(price_df['close_price'].max()),
                'first_price': float(price_df['close_price'].iloc[0]),
                'last_price': float(price_df['close_price'].iloc[-1]),
                'avg_price': float(price_df['close_price'].mean()),
                'first_date': str(price_df['date'].iloc[0]),
                'last_date': str(price_df['date'].iloc[-1]),
            })

        summary_df = pd.DataFrame(summary)

        # Save to CSV
        suffix = "_partial" if partial else ""
        summary_path = self.output_dir / f"price_summary{suffix}.csv"
        summary_df.to_csv(summary_path, index=False)

        if not partial:
            # Also save as JSON
            json_path = self.output_dir / "price_summary.json"
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"\n✓ Saved summary: {summary_path}")
            print(f"✓ Saved JSON: {json_path}")

    def load_existing_prices(self) -> Dict[str, pd.DataFrame]:
        """Load previously fetched prices from disk."""
        all_prices = {}

        price_files = list(self.output_dir.glob("KXEARNINGSMENTION*.csv"))
        print(f"Loading {len(price_files)} existing price files...")

        for price_file in price_files:
            contract_ticker = price_file.stem
            price_df = pd.read_csv(price_file)
            all_prices[contract_ticker] = price_df

        print(f"✓ Loaded {len(all_prices)} price histories")
        return all_prices


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch historical prices for Kalshi contracts")
    parser.add_argument(
        "--max-contracts",
        type=int,
        help="Limit number of contracts to fetch (for testing)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-fetch contracts that already have data",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Only load existing prices, don't fetch new ones",
    )

    args = parser.parse_args()

    fetcher = HistoricalPriceFetcher()

    if args.load_only:
        # Just load existing
        all_prices = fetcher.load_existing_prices()
        fetcher._save_summary(all_prices)
    else:
        # Fetch prices
        all_prices = fetcher.fetch_all_prices(
            max_contracts=args.max_contracts,
            skip_existing=not args.no_skip,
        )

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Run profitability validation:")
    print("   python scripts/validate_profitability.py")
    print("\n2. Analyze price patterns:")
    print("   - Check data/historical_prices/price_summary.csv")
    print("   - Compare avg_price vs final_price")
    print("   - Identify mispriced contracts")

    return 0


if __name__ == "__main__":
    exit(main())
