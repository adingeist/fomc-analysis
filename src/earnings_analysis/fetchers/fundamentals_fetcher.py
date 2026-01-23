"""
Fetch earnings fundamentals data (EPS, revenue, estimates).

Used for analyzing earnings surprises and guidance changes.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict

import pandas as pd
import yfinance as yf


class FundamentalsFetcher:
    """
    Fetch earnings fundamentals data.

    This includes:
    - Actual EPS and revenue
    - Estimated EPS and revenue (consensus)
    - Earnings surprise (beat/miss)
    """

    def __init__(self):
        pass

    def fetch_earnings_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetch historical earnings data for a ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol

        Returns
        -------
        pd.DataFrame
            Columns: date, eps_actual, eps_estimate, revenue_actual, revenue_estimate,
                     eps_surprise, eps_surprise_pct, revenue_surprise, revenue_surprise_pct
        """
        print(f"Fetching earnings data for {ticker}...")

        try:
            stock = yf.Ticker(ticker)

            # Get earnings history
            earnings = stock.earnings_dates

            if earnings is None or earnings.empty:
                print(f"No earnings data found for {ticker}")
                return pd.DataFrame()

            # Reset index to get date as column
            earnings = earnings.reset_index()

            # Rename columns for consistency
            column_mapping = {
                "Earnings Date": "date",
                "EPS Estimate": "eps_estimate",
                "Reported EPS": "eps_actual",
                "Surprise(%)": "eps_surprise_pct",
            }

            earnings = earnings.rename(columns=column_mapping)

            # Calculate raw surprise
            if "eps_actual" in earnings.columns and "eps_estimate" in earnings.columns:
                earnings["eps_surprise"] = earnings["eps_actual"] - earnings["eps_estimate"]

                # Calculate surprise percentage if not already present
                if "eps_surprise_pct" not in earnings.columns:
                    earnings["eps_surprise_pct"] = (
                        earnings["eps_surprise"] / earnings["eps_estimate"].abs() * 100
                    )

            # Add ticker
            earnings["ticker"] = ticker

            # Sort by date descending
            earnings = earnings.sort_values("date", ascending=False)

            return earnings

        except Exception as e:
            print(f"Error fetching earnings data: {e}")
            return pd.DataFrame()

    def fetch_revenue_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetch historical revenue data.

        Note: yfinance doesn't provide revenue estimates directly.
        This is a placeholder for future integration with premium data sources.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol

        Returns
        -------
        pd.DataFrame
            Revenue data
        """
        try:
            stock = yf.Ticker(ticker)

            # Get quarterly financials
            financials = stock.quarterly_financials

            if financials is None or financials.empty:
                return pd.DataFrame()

            # Extract total revenue
            if "Total Revenue" in financials.index:
                revenue = financials.loc["Total Revenue"]
                revenue_df = pd.DataFrame({
                    "date": revenue.index,
                    "revenue_actual": revenue.values,
                    "ticker": ticker,
                })
                return revenue_df

            return pd.DataFrame()

        except Exception as e:
            print(f"Error fetching revenue data: {e}")
            return pd.DataFrame()

    def calculate_earnings_surprise_binary(
        self,
        earnings_df: pd.DataFrame,
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        Add binary columns for beat/miss classification.

        Parameters
        ----------
        earnings_df : pd.DataFrame
            Earnings data with eps_surprise or eps_surprise_pct
        threshold : float
            Threshold for beat classification (default: 0.0)

        Returns
        -------
        pd.DataFrame
            Earnings data with beat/miss columns
        """
        df = earnings_df.copy()

        if "eps_surprise" in df.columns:
            df["eps_beat"] = (df["eps_surprise"] > threshold).astype(int)
            df["eps_miss"] = (df["eps_surprise"] < -threshold).astype(int)
            df["eps_inline"] = (
                (df["eps_surprise"] >= -threshold) & (df["eps_surprise"] <= threshold)
            ).astype(int)

        if "eps_surprise_pct" in df.columns:
            df["eps_beat_pct"] = (df["eps_surprise_pct"] > threshold).astype(int)
            df["eps_miss_pct"] = (df["eps_surprise_pct"] < -threshold).astype(int)

        return df


def fetch_earnings_data(ticker: str) -> pd.DataFrame:
    """
    Convenience function to fetch earnings data.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol

    Returns
    -------
    pd.DataFrame
        Earnings data
    """
    fetcher = FundamentalsFetcher()
    earnings = fetcher.fetch_earnings_data(ticker)

    if not earnings.empty:
        earnings = fetcher.calculate_earnings_surprise_binary(earnings)

    return earnings
