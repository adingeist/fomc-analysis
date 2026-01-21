"""
Command‑line entry points for the fomc_analysis package.

This module exposes a CLI with several subcommands to parse
transcripts, count contract mentions, estimate mention probabilities
with different models, and backtest trading strategies.  Use
`python -m fomc_analysis.main <subcommand> --help` to see
available options for each subcommand.

Example usage:

```
python -m fomc_analysis.main count \
    --transcripts-dir data/transcripts \
    --contract-mapping configs/contract_mapping.yaml \
    --output counts.csv

python -m fomc_analysis.main estimate \
    --counts-file counts.csv \
    --model ewma \
    --alpha 0.5 \
    --output estimates.csv

python -m fomc_analysis.main backtest \
    --price-file data/prices/kxfedmention-26jan-day.csv \
    --predictions estimates.csv \
    --edge-threshold 0.05 \
    --initial-capital 1000 \
    --output backtest_results.json
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .data_loader import load_transcripts
from .contract_mapping import load_mapping_from_file
from .feature_extraction import (
    count_mentions,
    compute_binary_events,
    ewma_probabilities,
    beta_binomial_estimator,
)
from .model import EwmaModel, BetaBinomialModel, LogisticRegressionModel
from .backtester import Backtester


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FOMC press conference analysis toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # count subcommand
    count_parser = subparsers.add_parser("count", help="Count contract mentions in transcripts")
    count_parser.add_argument("--transcripts-dir", required=True, help="Directory with parsed transcripts (PDF or text)")
    count_parser.add_argument("--contract-mapping", required=True, help="Path to contract mapping YAML/JSON file")
    count_parser.add_argument("--output", required=True, help="CSV file to write mention counts")

    # estimate subcommand
    est_parser = subparsers.add_parser("estimate", help="Estimate mention probabilities")
    est_parser.add_argument("--counts-file", required=True, help="CSV of mention counts (output of count)")
    est_parser.add_argument("--model", choices=["ewma", "beta", "logistic"], default="ewma", help="Estimation model")
    est_parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for EWMA model")
    est_parser.add_argument("--alpha-prior", type=float, default=1.0, help="Alpha prior for Beta–Binomial model")
    est_parser.add_argument("--beta-prior", type=float, default=1.0, help="Beta prior for Beta–Binomial model")
    est_parser.add_argument("--half-life", type=int, default=None, help="Half-life for exponential decay (Beta–Binomial)")
    est_parser.add_argument("--output", required=True, help="CSV file to write probability estimates")

    # backtest subcommand
    back_parser = subparsers.add_parser("backtest", help="Backtest trading strategy")
    back_parser.add_argument("--price-file", required=True, help="CSV file of historical prices (YES prices in cents)")
    back_parser.add_argument("--predictions", required=True, help="CSV file of model probabilities")
    back_parser.add_argument("--edge-threshold", type=float, default=0.05, help="Minimum edge to take a trade")
    back_parser.add_argument("--initial-capital", type=float, default=1000.0, help="Starting capital")
    back_parser.add_argument("--output", required=True, help="JSON file to write backtest results")

    return parser.parse_args()


def cmd_count(args: argparse.Namespace) -> None:
    transcripts = load_transcripts(args.transcripts_dir)
    mapping = load_mapping_from_file(args.contract_mapping)
    counts = count_mentions(transcripts, mapping)
    counts.to_csv(args.output)
    print(f"Wrote mention counts to {args.output}")


def cmd_estimate(args: argparse.Namespace) -> None:
    counts = pd.read_csv(args.counts_file, index_col=0)
    events = compute_binary_events(counts)
    if args.model == "ewma":
        model = EwmaModel(alpha=args.alpha)
        model.fit(events)
        probs = model.predict_proba(events)
    elif args.model == "beta":
        model = BetaBinomialModel(
            alpha_prior=args.alpha_prior,
            beta_prior=args.beta_prior,
            half_life=args.half_life,
        )
        model.fit(events)
        probs = model.predict_proba(events)
    else:  # logistic
        model = LogisticRegressionModel()
        model.fit(events)
        probs = model.predict_proba(events)
    probs.to_csv(args.output)
    print(f"Wrote probability estimates to {args.output}")


def cmd_backtest(args: argparse.Namespace) -> None:
    prices = pd.read_csv(args.price_file, index_col=0, parse_dates=True)
    predictions = pd.read_csv(args.predictions, index_col=0)
    # align indices
    common_index = predictions.index.intersection(prices.index)
    prices = prices.loc[common_index]
    predictions = predictions.loc[common_index]
    backtester = Backtester(prices, edge_threshold=args.edge_threshold)
    final_capital, trades = backtester.run(predictions, initial_capital=args.initial_capital)
    # serialise results
    results = {
        "final_capital": final_capital,
        "trades": [trade.__dict__ for trade in trades],
    }
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"Backtest complete.  Final capital: {final_capital:.2f}.  Results saved to {args.output}")


def main() -> None:
    args = parse_args()
    if args.command == "count":
        cmd_count(args)
    elif args.command == "estimate":
        cmd_estimate(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()