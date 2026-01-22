#!/usr/bin/env python3
"""Grid-search backtester settings for robust parameter selection."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from fomc_analysis.backtester_v3 import (
    TimeHorizonBacktester,
    fetch_historical_prices_at_horizons,
    fetch_kalshi_contract_outcomes,
)
from fomc_analysis.kalshi_client_factory import get_kalshi_client
from fomc_analysis.models import BetaBinomialModel


@dataclass
class SweepResult:
    params: Dict[str, Any]
    total_pnl: float
    roi: float
    sharpe: float
    trades: int
    win_rate: float


def parse_float_grid(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_grid(raw: str) -> List[int]:
    return [int(float(item.strip())) for item in raw.split(",") if item.strip()]


def parse_str_grid(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def generate_parameter_grid(grid_axes: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid_axes.keys())
    values = [grid_axes[key] for key in keys]
    combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos]


def evaluate_combo(
    outcomes: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: List[int],
    base_kwargs: Dict[str, Any],
    combo: Dict[str, Any],
    model_params: Dict[str, Any],
    initial_capital: float,
) -> SweepResult:
    backtester_kwargs = base_kwargs.copy()
    backtester_kwargs.update(combo)
    backtester = TimeHorizonBacktester(
        outcomes=outcomes,
        historical_prices=prices,
        horizons=horizons,
        **backtester_kwargs,
    )
    result = backtester.run(
        model_class=BetaBinomialModel,
        model_params=model_params,
        initial_capital=initial_capital,
    )
    overall = result.overall_metrics
    return SweepResult(
        params=combo,
        total_pnl=overall["total_pnl"],
        roi=overall["roi"],
        sharpe=overall["sharpe"],
        trades=overall["total_trades"],
        win_rate=overall["win_rate"],
    )


def load_kalshi_data(contract_words: Path, horizons: List[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    contract_entries = json.loads(contract_words.read_text())
    kalshi_client = get_kalshi_client()
    try:
        outcomes = fetch_kalshi_contract_outcomes(contract_entries, kalshi_client)
        if outcomes.empty:
            raise RuntimeError("No resolved markets in contract_words file")
        prices = fetch_historical_prices_at_horizons(
            tickers=outcomes["ticker"].unique().tolist(),
            meeting_dates=outcomes["meeting_date"].unique(),
            horizons=horizons,
            kalshi_client=kalshi_client,
        )
    finally:
        if kalshi_client and hasattr(kalshi_client, "close"):
            kalshi_client.close()
    return outcomes, prices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parameter sweep for backtest v3")
    parser.add_argument("--contract-words", type=Path, default=Path("data/kalshi_analysis/contract_words.json"))
    parser.add_argument("--horizons", type=str, default="7,14,30")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta-prior", type=float, default=1.0)
    parser.add_argument("--half-life", type=int, default=4)
    parser.add_argument("--prior-strength", type=float, default=4.0)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--initial-capital", type=float, default=10000.0)

    parser.add_argument("--edge-threshold", type=float, default=0.12)
    parser.add_argument("--position-size-pct", type=float, default=0.05)
    parser.add_argument("--fee-rate", type=float, default=0.07)
    parser.add_argument("--min-yes-prob", type=float, default=0.65)
    parser.add_argument("--max-no-prob", type=float, default=0.35)
    parser.add_argument("--yes-edge-threshold", type=float, default=0.20)
    parser.add_argument("--no-edge-threshold", type=float, default=0.08)
    parser.add_argument("--yes-position-size", type=float, default=0.04)
    parser.add_argument("--no-position-size", type=float, default=0.03)
    parser.add_argument("--max-position-size", type=float, default=1500.0)
    parser.add_argument("--train-window-size", type=int, default=12)
    parser.add_argument("--test-start-date", type=str, default="2022-01-01")

    parser.add_argument("--slippage-grid", type=str, default="0.01,0.02")
    parser.add_argument("--transaction-cost-grid", type=str, default="0.01,0.02")
    parser.add_argument("--max-position-grid", type=str, default="1000,1500")
    parser.add_argument("--train-window-grid", type=str, default="9,12,15")
    parser.add_argument("--test-start-grid", type=str, default="2021-01-01,2022-01-01")
    parser.add_argument("--yes-size-grid", type=str, default="0.03,0.04")
    parser.add_argument("--no-size-grid", type=str, default="0.02,0.03")
    parser.add_argument("--yes-edge-grid", type=str, default="0.18,0.20,0.22")
    parser.add_argument("--no-edge-grid", type=str, default="0.08,0.10")

    parser.add_argument("--sort-by", choices=["roi", "sharpe", "total_pnl"], default="roi")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/backtest_v3/grid_search.csv"))
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    print("Downloading Kalshi data once ...")
    outcomes, prices = load_kalshi_data(args.contract_words, horizons)

    base_kwargs = {
        "edge_threshold": args.edge_threshold,
        "position_size_pct": args.position_size_pct,
        "fee_rate": args.fee_rate,
        "min_yes_probability": args.min_yes_prob,
        "max_no_probability": args.max_no_prob,
        "yes_edge_threshold": args.yes_edge_threshold,
        "no_edge_threshold": args.no_edge_threshold,
        "yes_position_size_pct": args.yes_position_size,
        "no_position_size_pct": args.no_position_size,
        "max_position_size": args.max_position_size,
        "train_window_size": args.train_window_size,
        "test_start_date": args.test_start_date,
        "transaction_cost_rate": parse_float_grid(args.transaction_cost_grid)[0],
        "slippage": parse_float_grid(args.slippage_grid)[0],
    }

    grid_axes = {
        "slippage": parse_float_grid(args.slippage_grid),
        "transaction_cost_rate": parse_float_grid(args.transaction_cost_grid),
        "max_position_size": parse_float_grid(args.max_position_grid),
        "train_window_size": parse_int_grid(args.train_window_grid),
        "test_start_date": parse_str_grid(args.test_start_grid),
        "yes_position_size_pct": parse_float_grid(args.yes_size_grid),
        "no_position_size_pct": parse_float_grid(args.no_size_grid),
        "yes_edge_threshold": parse_float_grid(args.yes_edge_grid),
        "no_edge_threshold": parse_float_grid(args.no_edge_grid),
    }

    combos = generate_parameter_grid(grid_axes)
    print(f"Evaluating {len(combos)} combinations...")

    model_params = {
        "alpha_prior": args.alpha,
        "beta_prior": args.beta_prior,
        "half_life": args.half_life,
        "prior_strength": args.prior_strength,
        "min_history": args.min_history,
    }

    results: List[SweepResult] = []
    for idx, combo in enumerate(combos, start=1):
        print(f"[{idx}/{len(combos)}] combo={combo}")
        result = evaluate_combo(
            outcomes=outcomes,
            prices=prices,
            horizons=horizons,
            base_kwargs=base_kwargs,
            combo=combo,
            model_params=model_params,
            initial_capital=args.initial_capital,
        )
        results.append(result)

    records = []
    for entry in results:
        row = {
            "total_pnl": entry.total_pnl,
            "roi": entry.roi,
            "sharpe": entry.sharpe,
            "trades": entry.trades,
            "win_rate": entry.win_rate,
        }
        row.update(entry.params)
        records.append(row)

    df = pd.DataFrame(records)
    df = df.sort_values(by=args.sort_by, ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved full grid results to {args.output}")

    print("\nTop combinations:")
    print(df.head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
