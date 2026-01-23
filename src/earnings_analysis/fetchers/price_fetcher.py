"""
Fetch stock price data for earnings analysis.

Calculates price movements after earnings calls for backtesting.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import yfinance as yf


class PriceFetcher:
    """
    Fetch stock price data and calculate post-earnings movements.

    Parameters
    ----------
    horizons : List[int]
        Days after earnings to measure price movement (e.g., [1, 5, 10])
    """

    def __init__(self, horizons: List[int] = None):
        self.horizons = horizons or [1, 5, 10, 20]

    def fetch_outcomes(
        self,
        ticker: str,
        earnings_dates: List[datetime],
    ) -> pd.DataFrame:
        """
        Fetch price outcomes for earnings dates.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        earnings_dates : List[datetime]
            List of earnings call dates

        Returns
        -------
        pd.DataFrame
            Columns: earnings_date, close_before, close_0, close_1, close_5, close_10,
                     return_1d, return_5d, return_10d, direction_1d, direction_5d, etc.
        """
        if not earnings_dates:
            return pd.DataFrame()

        # Fetch price data with buffer
        start_date = min(earnings_dates) - timedelta(days=10)
        end_date = max(earnings_dates) + timedelta(days=max(self.horizons) + 10)

        print(f"Fetching price data for {ticker} from {start_date.date()} to {end_date.date()}...")

        try:
            stock = yf.Ticker(ticker)
            prices = stock.history(start=start_date, end=end_date)
        except Exception as e:
            print(f"Error fetching price data: {e}")
            return pd.DataFrame()

        if prices.empty:
            print(f"No price data found for {ticker}")
            return pd.DataFrame()

        outcomes = []

        for earnings_date in earnings_dates:
            outcome = self._calculate_outcome(prices, earnings_date)
            if outcome:
                outcomes.append(outcome)

        if not outcomes:
            return pd.DataFrame()

        df = pd.DataFrame(outcomes)
        df["ticker"] = ticker
        return df

    def _calculate_outcome(
        self,
        prices: pd.DataFrame,
        earnings_date: datetime,
    ) -> Optional[Dict]:
        """Calculate price outcome for a single earnings date."""
        earnings_date = pd.Timestamp(earnings_date).normalize()

        # Find the close price on earnings day (or nearest trading day before)
        available_dates_before = prices.index[prices.index <= earnings_date]
        if len(available_dates_before) == 0:
            return None

        close_before_date = available_dates_before[-1]
        close_before = float(prices.loc[close_before_date, "Close"])

        # Find close on earnings day (or next trading day)
        available_dates_after = prices.index[prices.index >= earnings_date]
        if len(available_dates_after) == 0:
            return None

        close_0_date = available_dates_after[0]
        close_0 = float(prices.loc[close_0_date, "Close"])

        outcome = {
            "earnings_date": earnings_date,
            "close_before": close_before,
            "close_0": close_0,
            "return_0d": (close_0 - close_before) / close_before,
            "direction_0d": 1 if close_0 > close_before else 0,
        }

        # Calculate returns for each horizon
        for horizon in self.horizons:
            target_date = close_0_date + timedelta(days=horizon)

            # Find next available trading day at or after target_date
            future_dates = prices.index[prices.index >= target_date]

            if len(future_dates) > 0:
                close_date = future_dates[0]
                close_price = float(prices.loc[close_date, "Close"])

                ret = (close_price - close_0) / close_0
                direction = 1 if close_price > close_0 else 0

                outcome[f"close_{horizon}d"] = close_price
                outcome[f"return_{horizon}d"] = ret
                outcome[f"direction_{horizon}d"] = direction
            else:
                outcome[f"close_{horizon}d"] = None
                outcome[f"return_{horizon}d"] = None
                outcome[f"direction_{horizon}d"] = None

        return outcome

    def fetch_intraday_outcomes(
        self,
        ticker: str,
        earnings_dates: List[datetime],
    ) -> pd.DataFrame:
        """
        Fetch intraday price movements (open to close on earnings day).

        Useful for same-day trading strategies.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        earnings_dates : List[datetime]
            List of earnings call dates

        Returns
        -------
        pd.DataFrame
            Columns: earnings_date, open, high, low, close, intraday_return
        """
        if not earnings_dates:
            return pd.DataFrame()

        outcomes = []

        for earnings_date in earnings_dates:
            try:
                # Fetch data for the specific day
                stock = yf.Ticker(ticker)
                data = stock.history(
                    start=earnings_date,
                    end=earnings_date + timedelta(days=1),
                    interval="1d"
                )

                if data.empty:
                    continue

                row = data.iloc[0]
                open_price = float(row["Open"])
                close_price = float(row["Close"])

                outcomes.append({
                    "earnings_date": earnings_date,
                    "open": open_price,
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": close_price,
                    "volume": int(row["Volume"]),
                    "intraday_return": (close_price - open_price) / open_price,
                    "intraday_direction": 1 if close_price > open_price else 0,
                })

            except Exception as e:
                print(f"Error fetching intraday data for {earnings_date}: {e}")
                continue

        if not outcomes:
            return pd.DataFrame()

        df = pd.DataFrame(outcomes)
        df["ticker"] = ticker
        return df


def fetch_price_outcomes(
    ticker: str,
    earnings_dates: List[datetime],
    horizons: List[int] = None,
    include_intraday: bool = False,
) -> pd.DataFrame:
    """
    Convenience function to fetch price outcomes.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    earnings_dates : List[datetime]
        List of earnings call dates
    horizons : List[int]
        Days after earnings to measure (default: [1, 5, 10, 20])
    include_intraday : bool
        Whether to include intraday data

    Returns
    -------
    pd.DataFrame
        Price outcomes dataframe
    """
    fetcher = PriceFetcher(horizons=horizons)
    outcomes = fetcher.fetch_outcomes(ticker, earnings_dates)

    if include_intraday and not outcomes.empty:
        intraday = fetcher.fetch_intraday_outcomes(ticker, earnings_dates)
        if not intraday.empty:
            # Merge on earnings_date
            outcomes = outcomes.merge(
                intraday[["earnings_date", "open", "high", "low", "volume", "intraday_return", "intraday_direction"]],
                on="earnings_date",
                how="left",
            )

    return outcomes
