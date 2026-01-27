"""
Execution strategy module for Kalshi prediction market trading.

Implements maker vs taker execution simulation and spread-aware trade
filtering based on microstructure findings from Becker (2025):
- Makers earn +1.12% systematic excess return over takers
- Spread capture is structural (direction-independent)
- Limit orders at mid-price capture most of this advantage

Execution modes:
- TAKER: Market orders at ask/bid (worst price, guaranteed fill)
- MAKER: Limit orders at mid-price (better price, uncertain fill)
- HYBRID: Try maker first, escalate to taker if unfilled
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class ExecutionMode(Enum):
    """Order execution strategy."""
    TAKER = "taker"      # Market order: immediate fill at worst price
    MAKER = "maker"      # Limit order: fill at mid, probabilistic
    HYBRID = "hybrid"    # Limit then market escalation


# Empirical maker excess return from Becker (2025)
_MAKER_EXCESS_RETURN = 0.0112  # +1.12%


@dataclass
class ExecutionResult:
    """Result of simulated order execution."""
    execution_price: float    # Actual fill price (0-1)
    filled: bool              # Whether order was filled
    is_maker: bool            # True if filled as maker (limit order)
    spread_cost: float        # Cost of spread crossing (0 for makers)
    slippage: float           # Price impact beyond spread


@dataclass
class SpreadFilter:
    """
    Filter trades based on bid-ask spread.

    Only trade when edge exceeds the cost of crossing the spread,
    ensuring positive expected value after execution costs.

    Parameters
    ----------
    min_net_edge : float
        Minimum edge remaining after spread cost (default: 0.03).
        Must be positive to trade.
    max_spread_cents : int
        Maximum allowed spread in cents (default: 15).
        Wider spreads indicate illiquid markets.
    """

    min_net_edge: float = 0.03
    max_spread_cents: int = 15

    def should_trade(
        self,
        edge: float,
        bid_cents: int,
        ask_cents: int,
    ) -> bool:
        """
        Determine if edge is sufficient to overcome spread costs.

        Parameters
        ----------
        edge : float
            Raw edge (model_prob - market_price), absolute value.
        bid_cents : int
            Best bid price in cents.
        ask_cents : int
            Best ask price in cents.

        Returns
        -------
        bool
            True if trade is worthwhile after spread costs.
        """
        spread = ask_cents - bid_cents

        # Reject illiquid markets
        if spread > self.max_spread_cents:
            return False

        # Half-spread cost for a taker (crossing from mid to bid/ask)
        half_spread_cost = (spread / 2) / 100.0

        # Net edge after execution cost
        net_edge = abs(edge) - half_spread_cost

        return net_edge >= self.min_net_edge

    def net_edge_after_spread(
        self,
        edge: float,
        bid_cents: int,
        ask_cents: int,
    ) -> float:
        """Compute edge remaining after spread crossing cost."""
        spread = ask_cents - bid_cents
        half_spread_cost = (spread / 2) / 100.0
        return abs(edge) - half_spread_cost


class ExecutionSimulator:
    """
    Simulate order execution with maker/taker dynamics.

    Models the trade-off between execution certainty (taker) and
    execution quality (maker). Based on empirical data showing
    +1.12% maker excess return.

    Parameters
    ----------
    mode : ExecutionMode
        Order placement strategy (default: HYBRID).
    base_fill_probability : float
        Probability of a limit order filling at mid-price (default: 0.65).
        Empirically, ~65% of limit orders at mid fill within a session.
    urgency_fill_decay : float
        How much urgency reduces maker fill probability (default: 0.3).
        High urgency (close to expiry) reduces patience for limit fills.
    taker_slippage : float
        Additional slippage for market orders beyond the spread (default: 0.005).
    maker_improvement : float
        Price improvement for limit orders vs mid (default: 0.002).
    """

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.HYBRID,
        base_fill_probability: float = 0.65,
        urgency_fill_decay: float = 0.3,
        taker_slippage: float = 0.005,
        maker_improvement: float = 0.002,
    ):
        self.mode = mode
        self.base_fill_probability = base_fill_probability
        self.urgency_fill_decay = urgency_fill_decay
        self.taker_slippage = taker_slippage
        self.maker_improvement = maker_improvement

    def simulate_execution(
        self,
        side: str,
        bid_cents: int,
        ask_cents: int,
        urgency: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> ExecutionResult:
        """
        Simulate order execution and return fill details.

        Parameters
        ----------
        side : str
            "YES" or "NO" — which side we're buying.
        bid_cents : int
            Best bid in cents.
        ask_cents : int
            Best ask in cents.
        urgency : float
            How urgently we need to fill (0=patient, 1=immediate).
            Higher urgency reduces maker fill probability.
        rng : numpy Generator, optional
            Random number generator for fill simulation.

        Returns
        -------
        ExecutionResult
            Simulated execution details.
        """
        if rng is None:
            rng = np.random.default_rng()

        spread = ask_cents - bid_cents
        mid = (bid_cents + ask_cents) / 2.0

        if self.mode == ExecutionMode.TAKER:
            return self._execute_taker(side, bid_cents, ask_cents, mid)

        elif self.mode == ExecutionMode.MAKER:
            return self._execute_maker(side, bid_cents, ask_cents, mid, urgency, rng)

        else:  # HYBRID
            return self._execute_hybrid(side, bid_cents, ask_cents, mid, urgency, rng)

    def _execute_taker(
        self,
        side: str,
        bid_cents: int,
        ask_cents: int,
        mid: float,
    ) -> ExecutionResult:
        """Market order execution: cross the spread."""
        if side == "YES":
            # Buy YES at ask price
            fill_price = ask_cents / 100.0 + self.taker_slippage
        else:
            # Buy NO = sell YES at bid, so our cost is (100 - bid) / 100
            fill_price = (100 - bid_cents) / 100.0 + self.taker_slippage

        fill_price = np.clip(fill_price, 0.01, 0.99)
        spread_cost = (ask_cents - bid_cents) / 2 / 100.0

        return ExecutionResult(
            execution_price=float(fill_price),
            filled=True,
            is_maker=False,
            spread_cost=spread_cost,
            slippage=self.taker_slippage,
        )

    def _execute_maker(
        self,
        side: str,
        bid_cents: int,
        ask_cents: int,
        mid: float,
        urgency: float,
        rng: np.random.Generator,
    ) -> ExecutionResult:
        """Limit order at mid-price: better price, uncertain fill."""
        # Fill probability decreases with urgency
        fill_prob = self.base_fill_probability * (1 - urgency * self.urgency_fill_decay)
        filled = rng.random() < fill_prob

        if not filled:
            return ExecutionResult(
                execution_price=0.0,
                filled=False,
                is_maker=True,
                spread_cost=0.0,
                slippage=0.0,
            )

        # Fill at mid-price with slight improvement
        if side == "YES":
            fill_price = mid / 100.0 - self.maker_improvement
        else:
            fill_price = (100 - mid) / 100.0 - self.maker_improvement

        fill_price = np.clip(fill_price, 0.01, 0.99)

        return ExecutionResult(
            execution_price=float(fill_price),
            filled=True,
            is_maker=True,
            spread_cost=0.0,
            slippage=-self.maker_improvement,  # Negative = price improvement
        )

    def _execute_hybrid(
        self,
        side: str,
        bid_cents: int,
        ask_cents: int,
        mid: float,
        urgency: float,
        rng: np.random.Generator,
    ) -> ExecutionResult:
        """Try maker first, fall back to taker if unfilled."""
        maker_result = self._execute_maker(
            side, bid_cents, ask_cents, mid, urgency, rng
        )

        if maker_result.filled:
            return maker_result

        # Fall back to taker execution
        return self._execute_taker(side, bid_cents, ask_cents, mid)

    def expected_execution_price(
        self,
        side: str,
        bid_cents: int,
        ask_cents: int,
        urgency: float = 0.5,
    ) -> float:
        """
        Compute expected execution price across execution modes.

        Useful for backtesting when we want deterministic results
        without random fill simulation.

        Parameters
        ----------
        side : str
            "YES" or "NO".
        bid_cents : int
            Best bid.
        ask_cents : int
            Best ask.
        urgency : float
            Urgency level (0-1).

        Returns
        -------
        float
            Expected fill price (probability-weighted).
        """
        mid = (bid_cents + ask_cents) / 2.0

        if self.mode == ExecutionMode.TAKER:
            if side == "YES":
                return float(np.clip(ask_cents / 100.0 + self.taker_slippage, 0.01, 0.99))
            else:
                return float(np.clip((100 - bid_cents) / 100.0 + self.taker_slippage, 0.01, 0.99))

        fill_prob = self.base_fill_probability * (1 - urgency * self.urgency_fill_decay)

        if side == "YES":
            maker_price = mid / 100.0 - self.maker_improvement
            taker_price = ask_cents / 100.0 + self.taker_slippage
        else:
            maker_price = (100 - mid) / 100.0 - self.maker_improvement
            taker_price = (100 - bid_cents) / 100.0 + self.taker_slippage

        maker_price = np.clip(maker_price, 0.01, 0.99)
        taker_price = np.clip(taker_price, 0.01, 0.99)

        if self.mode == ExecutionMode.MAKER:
            # Expected price = fill_prob * maker_price + (1 - fill_prob) * no_fill
            # If not filled, we don't trade, so just return maker_price
            # (the backtest should check fill probability separately)
            return float(maker_price)

        # HYBRID: weighted average
        expected = fill_prob * maker_price + (1 - fill_prob) * taker_price
        return float(np.clip(expected, 0.01, 0.99))

    def adjust_backtest_entry_price(
        self,
        raw_entry_price: float,
        side: str,
        market_price_cents: Optional[int] = None,
    ) -> float:
        """
        Adjust backtester entry price based on execution mode.

        For backtesting without full order book data, this provides a
        deterministic price adjustment that accounts for maker/taker dynamics.

        Parameters
        ----------
        raw_entry_price : float
            Original entry price (0-1) from backtester.
        side : str
            "YES" or "NO".
        market_price_cents : int, optional
            Market price for spread estimation. If not provided, uses
            a default spread assumption.

        Returns
        -------
        float
            Adjusted entry price incorporating execution dynamics.
        """
        if self.mode == ExecutionMode.TAKER:
            # Taker pays more (worse price)
            return float(np.clip(raw_entry_price + self.taker_slippage, 0.01, 0.99))

        elif self.mode == ExecutionMode.MAKER:
            # Maker gets price improvement
            return float(np.clip(raw_entry_price - self.maker_improvement, 0.01, 0.99))

        else:
            # Hybrid: probability-weighted
            fill_prob = self.base_fill_probability * 0.85  # Default urgency 0.5
            maker_price = raw_entry_price - self.maker_improvement
            taker_price = raw_entry_price + self.taker_slippage
            adjusted = fill_prob * maker_price + (1 - fill_prob) * taker_price
            return float(np.clip(adjusted, 0.01, 0.99))
