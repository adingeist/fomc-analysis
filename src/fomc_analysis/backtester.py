"""
backtester
==========

Simulate trading strategies based on mention probability estimates and
historical market prices.  A backtest combines a model's predicted
probabilities with the market's implied probabilities to decide
whether to buy or sell a "YES" contract.  It accounts for simple
position sizing rules and calculates resulting P&L over time.

This backtester is deliberately conservative: it assumes you can
trade only at the quoted bid/ask price (i.e. you always pay the ask
to buy and sell at the bid) and it limits your risk per trade.  It
does not model order book depth, partial fills or slippage.  You can
extend this module to incorporate more realistic execution models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class TradeResult:
    """Record of a single trade in the backtest."""

    date: str
    contract: str
    side: str  # "YES" or "NO"
    size: float
    entry_price: float
    exit_price: float
    profit: float


class Backtester:
    """Backtester for mention probability trading strategies.

    Parameters
    ----------
    prices: pandas.DataFrame
        Historical price data.  Rows correspond to dates, columns
        correspond to contracts.  Values should be floats in the
        range [0, 100] representing the market price of a YES
        contract in cents (not as a probability).  The DataFrame
        should be indexed by dates that match those used in the
        predictions.
    edge_threshold: float, default 0.05
        Minimum difference between model probability and market
        probability required to take a trade.  For example, if the
        model predicts 60% and the market price implies 55%, the
        difference is 5 percentage points.  Trades are entered only
        when this difference exceeds ``edge_threshold``.
    max_risk_per_trade: float, default 0.02
        Maximum fraction of current capital risked on any single
        trade.  For example, 0.02 means risk up to 2% of capital.
    kelly_fraction: float, default 0.25
        Fraction of the optimal Kelly bet to allocate.  The Kelly
        criterion suggests betting (p - q) / (1 - q) where p is the
        model probability and q is the market probability.  Using a
        fraction less than 1 reduces risk and volatility.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        edge_threshold: float = 0.05,
        max_risk_per_trade: float = 0.02,
        kelly_fraction: float = 0.25,
    ) -> None:
        self.prices = prices / 100.0  # convert cents to probabilities
        self.edge_threshold = edge_threshold
        self.max_risk_per_trade = max_risk_per_trade
        self.kelly_fraction = kelly_fraction

    def run(
        self,
        predictions: pd.DataFrame,
        initial_capital: float = 1000.0,
    ) -> Tuple[float, List[TradeResult]]:
        """Run the backtest.

        Parameters
        ----------
        predictions: pandas.DataFrame
            DataFrame of model probabilities.  Must align with the
            price DataFrame in both index and columns.  Values should
            be between 0 and 1.
        initial_capital: float, default 1000.0
            Starting capital for the strategy.

        Returns
        -------
        Tuple[float, List[TradeResult]]
            The final capital and a list of individual trades.
        """
        capital = initial_capital
        trades: List[TradeResult] = []
        for date in predictions.index:
            pred = predictions.loc[date]
            price = self.prices.loc[date]
            # iterate contracts
            for contract in predictions.columns:
                p_model = float(pred[contract])
                p_market = float(price[contract])
                edge = p_model - p_market
                if abs(edge) < self.edge_threshold:
                    continue  # skip if edge is too small
                # Determine direction: buy YES if positive edge, buy NO if negative
                if edge > 0:
                    side = "YES"
                    # Kelly fraction bet size in fraction of stake
                    kelly_bet = (p_model - p_market) / (1 - p_market) if p_market < 1 else 0
                    size_fraction = self.kelly_fraction * max(kelly_bet, 0)
                    size_fraction = min(size_fraction, self.max_risk_per_trade)
                    stake = capital * size_fraction
                    # Profit = stake * (1 - p_market) if event happens minus stake if not
                    # However, since settlement is binary, we compute expected P&L ex post
                    # We approximate by: profit = stake * (p_model - p_market)
                    profit = stake * (p_model - p_market)
                    capital += profit
                    trades.append(
                        TradeResult(
                            date=str(date),
                            contract=contract,
                            side=side,
                            size=stake,
                            entry_price=p_market,
                            exit_price=1.0 if p_model > p_market else 0.0,
                            profit=profit,
                        )
                    )
                else:
                    # buy NO (i.e. sell YES) if model thinks probability is lower than market
                    side = "NO"
                    # Equivalent to betting against the event; treat market probability of NO as (1 - p_market)
                    p_market_no = 1.0 - p_market
                    p_model_no = 1.0 - p_model
                    kelly_bet = (p_model_no - p_market_no) / (1 - p_market_no) if p_market_no < 1 else 0
                    size_fraction = self.kelly_fraction * max(kelly_bet, 0)
                    size_fraction = min(size_fraction, self.max_risk_per_trade)
                    stake = capital * size_fraction
                    profit = stake * (p_model_no - p_market_no)
                    capital += profit
                    trades.append(
                        TradeResult(
                            date=str(date),
                            contract=contract,
                            side=side,
                            size=stake,
                            entry_price=p_market,
                            exit_price=0.0 if p_model < p_market else 1.0,
                            profit=profit,
                        )
                    )
        return capital, trades