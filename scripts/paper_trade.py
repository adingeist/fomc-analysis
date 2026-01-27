#!/usr/bin/env python3
"""
Paper trading harness for Kalshi earnings mention contracts.

This script manages the complete paper trading lifecycle:

1. SNAPSHOT: Before earnings call - capture active contracts and make predictions
2. PREDICT:  Generate predictions for upcoming earnings calls using historical data
3. SETTLE:   After earnings call - check settled outcomes and score predictions
4. REPORT:   Generate performance report across all paper trades

A paper trade cycle for one earnings call:
    # Before the call: capture prices and make predictions
    python scripts/paper_trade.py snapshot --ticker META
    python scripts/paper_trade.py predict --ticker META

    # After the call settles: record outcomes
    python scripts/paper_trade.py settle --ticker META

    # View cumulative performance
    python scripts/paper_trade.py report

Usage:
    python scripts/paper_trade.py snapshot --ticker META
    python scripts/paper_trade.py predict --ticker META
    python scripts/paper_trade.py settle --ticker META
    python scripts/paper_trade.py report
    python scripts/paper_trade.py report --ticker META
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fomc_analysis.kalshi_client_factory import get_kalshi_client
from earnings_analysis.ground_truth import (
    fetch_ground_truth,
    load_ground_truth,
    build_backtest_dataframes,
    _extract_company_from_series,
    _extract_event_date_from_ticker,
    EARNINGS_SERIES_TICKERS,
)
from earnings_analysis.models.beta_binomial import BetaBinomialEarningsModel


PAPER_TRADE_DIR = Path("data/paper_trades")


def _load_journal() -> List[Dict]:
    """Load the paper trading journal."""
    journal_path = PAPER_TRADE_DIR / "journal.json"
    if journal_path.exists():
        with open(journal_path) as f:
            return json.load(f)
    return []


def _save_journal(journal: List[Dict]):
    """Save the paper trading journal."""
    PAPER_TRADE_DIR.mkdir(parents=True, exist_ok=True)
    journal_path = PAPER_TRADE_DIR / "journal.json"
    with open(journal_path, "w") as f:
        json.dump(journal, f, indent=2, default=str)


def cmd_snapshot(args):
    """
    Capture a snapshot of active contracts before an earnings call.

    Records current market prices for all active contracts for a ticker.
    """
    print(f"\nCapturing pre-earnings snapshot for {args.ticker}...")

    client = get_kalshi_client()
    series_ticker = f"KXEARNINGSMENTION{args.ticker.upper()}"

    markets = client.get_markets(series_ticker=series_ticker, limit=200)
    if not markets:
        print(f"No markets found for {series_ticker}")
        return

    # Filter to active contracts
    active = [m for m in markets if m.get("status", "").lower() in ("active", "open")]
    if not active:
        print(f"No active contracts found for {series_ticker}")
        print(f"Found {len(markets)} markets total (may all be settled)")
        return

    snapshot = {
        "ticker": args.ticker.upper(),
        "series_ticker": series_ticker,
        "snapshot_time": datetime.now().isoformat(),
        "num_contracts": len(active),
        "contracts": [],
    }

    print(f"\n  Active contracts ({len(active)}):")
    for market in active:
        custom_strike = market.get("custom_strike") or {}
        word = custom_strike.get("Word") or market.get("yes_sub_title", "")
        last_price = (market.get("last_price", 0) or 0) / 100.0
        yes_bid = (market.get("yes_bid", 0) or 0) / 100.0
        yes_ask = (market.get("yes_ask", 0) or 0) / 100.0
        volume = market.get("volume", 0)

        contract_data = {
            "market_ticker": market.get("ticker", ""),
            "word": word,
            "last_price": last_price,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "mid_price": (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else last_price,
            "volume": volume,
            "expiration_time": market.get("expiration_time", ""),
        }
        snapshot["contracts"].append(contract_data)
        print(
            f"    {word:30s}  price={last_price:.2f}  "
            f"bid={yes_bid:.2f}  ask={yes_ask:.2f}  vol={volume}"
        )

    # Save snapshot
    snapshot_dir = PAPER_TRADE_DIR / "snapshots" / args.ticker.upper()
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = snapshot_dir / f"snapshot_{timestamp}.json"
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"\n  Snapshot saved to {snapshot_path}")
    print(f"\n  Next step: python scripts/paper_trade.py predict --ticker {args.ticker}")


def cmd_predict(args):
    """
    Generate predictions for upcoming earnings call.

    Uses historical ground truth (from settled contracts) to train the
    Beta-Binomial model and predict which words will be mentioned.
    """
    ticker = args.ticker.upper()
    print(f"\nGenerating predictions for {ticker}...")

    # Load the latest snapshot
    snapshot_dir = PAPER_TRADE_DIR / "snapshots" / ticker
    if not snapshot_dir.exists():
        print(f"No snapshot found. Run 'snapshot' first.")
        return

    snapshot_files = sorted(snapshot_dir.glob("snapshot_*.json"))
    if not snapshot_files:
        print(f"No snapshot files found in {snapshot_dir}")
        return

    with open(snapshot_files[-1]) as f:
        snapshot = json.load(f)

    print(f"  Using snapshot from {snapshot['snapshot_time']}")

    # Load ground truth for training
    ground_truth_dir = Path("data/ground_truth")
    if not (ground_truth_dir / "ground_truth.json").exists():
        print("  No ground truth data found. Fetching from Kalshi...")
        dataset = fetch_ground_truth()
        if dataset.contracts:
            from earnings_analysis.ground_truth import save_ground_truth
            save_ground_truth(dataset, ground_truth_dir)
        else:
            print("  No settled contracts found. Cannot make predictions without history.")
            return
    else:
        dataset = load_ground_truth(ground_truth_dir)

    # Build training data
    features, outcomes, _ = build_backtest_dataframes(dataset, ticker)

    if outcomes.empty:
        print(f"  No historical outcomes found for {ticker}")
        print(f"  Need at least some settled contracts to make predictions")
        return

    print(f"  Training data: {len(outcomes)} historical events, {len(outcomes.columns)} words")

    # Make predictions for each active contract
    predictions = []
    model_params = {"alpha_prior": 1.0, "beta_prior": 1.0, "half_life": args.half_life}

    for contract in snapshot["contracts"]:
        word = contract["word"].lower()
        market_price = contract["mid_price"]

        # Check if we have history for this word
        if word in outcomes.columns:
            # Train on all historical data
            model = BetaBinomialEarningsModel(**model_params)
            model.fit(features, outcomes[word])
            pred = model.predict()

            predicted_prob = float(pred.iloc[0]["probability"])
            lower = float(pred.iloc[0]["lower_bound"])
            upper = float(pred.iloc[0]["upper_bound"])
            uncertainty = float(pred.iloc[0]["uncertainty"])

            history_count = int(outcomes[word].sum())
            history_total = len(outcomes[word])
        else:
            # No history: use uninformative prior
            predicted_prob = 0.5
            lower = 0.025
            upper = 0.975
            uncertainty = 0.289  # std of uniform(0,1)
            history_count = 0
            history_total = 0

        edge = predicted_prob - market_price

        # Determine recommended action
        if edge > args.edge_threshold and predicted_prob > 0.65:
            action = "BUY YES"
        elif edge < -args.edge_threshold and predicted_prob < 0.35:
            action = "BUY NO"
        else:
            action = "PASS"

        prediction = {
            "word": contract["word"],
            "market_ticker": contract["market_ticker"],
            "market_price": market_price,
            "predicted_probability": predicted_prob,
            "confidence_lower": lower,
            "confidence_upper": upper,
            "uncertainty": uncertainty,
            "edge": edge,
            "recommended_action": action,
            "history": f"{history_count}/{history_total} mentions",
        }
        predictions.append(prediction)

    # Display predictions
    print(f"\n  Predictions:")
    print(f"  {'Word':<30s} {'Market':>6s} {'Model':>6s} {'Edge':>7s} {'Action':<10s} {'History'}")
    print(f"  {'-'*90}")

    for p in sorted(predictions, key=lambda x: abs(x["edge"]), reverse=True):
        print(
            f"  {p['word']:<30s} "
            f"{p['market_price']:>6.2f} "
            f"{p['predicted_probability']:>6.2f} "
            f"{p['edge']:>+7.3f} "
            f"{p['recommended_action']:<10s} "
            f"{p['history']}"
        )

    # Record to journal
    journal = _load_journal()
    entry = {
        "type": "prediction",
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "snapshot_time": snapshot["snapshot_time"],
        "parameters": {
            "half_life": args.half_life,
            "edge_threshold": args.edge_threshold,
        },
        "predictions": predictions,
        "status": "pending",  # will be updated by 'settle'
    }
    journal.append(entry)
    _save_journal(journal)

    # Save predictions file
    pred_dir = PAPER_TRADE_DIR / "predictions" / ticker
    pred_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = pred_dir / f"predictions_{timestamp}.json"
    with open(pred_path, "w") as f:
        json.dump(entry, f, indent=2)

    print(f"\n  Predictions saved to {pred_path}")
    print(f"  Journal updated ({len(journal)} total entries)")

    # Summary of recommended trades
    trades = [p for p in predictions if p["recommended_action"] != "PASS"]
    if trades:
        print(f"\n  Recommended paper trades ({len(trades)}):")
        for t in trades:
            print(f"    {t['recommended_action']:8s} {t['word']} @ {t['market_price']:.2f} (edge: {t['edge']:+.3f})")
    else:
        print(f"\n  No trades recommended (edge threshold: {args.edge_threshold})")

    print(f"\n  After the earnings call settles:")
    print(f"    python scripts/paper_trade.py settle --ticker {ticker}")


def cmd_settle(args):
    """
    Check settled outcomes and score predictions.

    Looks up finalized contracts to determine actual outcomes,
    then calculates simulated P&L.
    """
    ticker = args.ticker.upper()
    print(f"\nChecking settlements for {ticker}...")

    # Fetch current contract states
    client = get_kalshi_client()
    series_ticker = f"KXEARNINGSMENTION{ticker}"

    markets = client.get_markets(series_ticker=series_ticker, limit=200)
    if not markets:
        print(f"No markets found for {series_ticker}")
        return

    # Find settled contracts
    settled = {}
    for market in markets:
        status = market.get("status", "").lower()
        if status in ("settled", "finalized", "closed"):
            custom_strike = market.get("custom_strike") or {}
            word = custom_strike.get("Word") or market.get("yes_sub_title", "")
            if word:
                last_price = (market.get("last_price", 0) or 0) / 100.0
                result_field = market.get("result", "")
                if result_field:
                    outcome = 1 if result_field.lower() in ("yes", "true", "1") else 0
                else:
                    outcome = 1 if last_price > 0.5 else 0
                settled[word.lower()] = {
                    "outcome": outcome,
                    "settlement_price": last_price,
                    "market_ticker": market.get("ticker", ""),
                }

    if not settled:
        print(f"No settled contracts found yet for {ticker}")
        print("Contracts may not have settled. Check again later.")
        return

    print(f"  Found {len(settled)} settled contracts")

    # Load pending predictions from journal
    journal = _load_journal()
    pending = [
        e for e in journal
        if e.get("type") == "prediction"
        and e.get("ticker") == ticker
        and e.get("status") == "pending"
    ]

    if not pending:
        print(f"  No pending predictions found for {ticker}")
        print(f"  Run 'predict' before 'settle'")
        return

    # Score the most recent pending prediction
    entry = pending[-1]
    predictions = entry["predictions"]

    print(f"\n  Scoring predictions from {entry['timestamp']}:")
    print(f"  {'Word':<30s} {'Predicted':>8s} {'Market':>6s} {'Actual':>6s} {'Edge':>7s} {'Action':<10s} {'P&L':>8s}")
    print(f"  {'-'*100}")

    total_pnl = 0.0
    scored = 0
    correct = 0
    trades_pnl = []

    position_size = args.position_size  # dollars per trade

    for pred in predictions:
        word_lower = pred["word"].lower()
        if word_lower not in settled:
            continue

        actual = settled[word_lower]
        outcome = actual["outcome"]
        action = pred["recommended_action"]

        # Calculate P&L for recommended trades
        pnl = 0.0
        if action == "BUY YES":
            entry_price = pred["market_price"]
            if outcome == 1:
                # Won: payout is $1 per contract, cost was entry_price
                pnl = position_size * (1 - entry_price) / entry_price * 0.93  # 7% fee
            else:
                pnl = -position_size
            pnl -= position_size * 0.01  # transaction cost
            trades_pnl.append(pnl)
        elif action == "BUY NO":
            entry_price = 1 - pred["market_price"]
            if outcome == 0:
                pnl = position_size * pred["market_price"] / (1 - pred["market_price"]) * 0.93
            else:
                pnl = -position_size
            pnl -= position_size * 0.01
            trades_pnl.append(pnl)

        total_pnl += pnl

        # Score prediction accuracy
        pred_yes = pred["predicted_probability"] > 0.5
        actual_yes = outcome == 1
        is_correct = pred_yes == actual_yes
        if is_correct:
            correct += 1
        scored += 1

        pnl_str = f"${pnl:+.2f}" if action != "PASS" else "-"
        print(
            f"  {pred['word']:<30s} "
            f"{pred['predicted_probability']:>8.2f} "
            f"{pred['market_price']:>6.2f} "
            f"{'YES' if outcome == 1 else 'NO':>6s} "
            f"{pred['edge']:>+7.3f} "
            f"{action:<10s} "
            f"{pnl_str:>8s}"
        )

    # Update journal entry
    entry["status"] = "settled"
    entry["settlement"] = {
        "settled_time": datetime.now().isoformat(),
        "scored": scored,
        "correct": correct,
        "accuracy": correct / scored if scored > 0 else 0,
        "total_pnl": total_pnl,
        "num_trades": len(trades_pnl),
        "position_size": position_size,
    }
    _save_journal(journal)

    print(f"\n  Summary:")
    print(f"    Predictions scored: {scored}")
    print(f"    Correct:            {correct}/{scored} ({correct/scored:.1%})" if scored else "")
    print(f"    Trades executed:    {len(trades_pnl)}")
    if trades_pnl:
        print(f"    Total P&L:          ${total_pnl:+.2f}")
        print(f"    Avg P&L per trade:  ${np.mean(trades_pnl):+.2f}")
        print(f"    Win rate:           {sum(1 for p in trades_pnl if p > 0)}/{len(trades_pnl)}")

    print(f"\n  Journal updated. Run 'report' to see cumulative performance.")


def cmd_report(args):
    """Generate cumulative paper trading performance report."""
    journal = _load_journal()

    if not journal:
        print("No paper trades recorded yet.")
        print("Start with: python scripts/paper_trade.py snapshot --ticker META")
        return

    # Filter by ticker if specified
    if args.ticker:
        journal = [e for e in journal if e.get("ticker") == args.ticker.upper()]

    settled = [e for e in journal if e.get("status") == "settled"]
    pending = [e for e in journal if e.get("status") == "pending"]

    print(f"\n{'='*70}")
    print(f"PAPER TRADING REPORT")
    print(f"{'='*70}")

    print(f"\n  Total entries:  {len(journal)}")
    print(f"  Settled:        {len(settled)}")
    print(f"  Pending:        {len(pending)}")

    if not settled:
        print(f"\n  No settled predictions yet.")
        if pending:
            print(f"  {len(pending)} predictions awaiting settlement:")
            for p in pending:
                print(f"    {p['ticker']} ({p['timestamp']})")
        return

    # Aggregate settled results
    total_scored = sum(e["settlement"]["scored"] for e in settled)
    total_correct = sum(e["settlement"]["correct"] for e in settled)
    total_trades = sum(e["settlement"]["num_trades"] for e in settled)
    total_pnl = sum(e["settlement"]["total_pnl"] for e in settled)

    print(f"\n  Cumulative Results:")
    print(f"    Predictions scored: {total_scored}")
    print(f"    Correct:            {total_correct}/{total_scored} ({total_correct/total_scored:.1%})" if total_scored else "")
    print(f"    Total trades:       {total_trades}")
    print(f"    Total P&L:          ${total_pnl:+.2f}")

    # Per-ticker breakdown
    tickers = sorted(set(e["ticker"] for e in settled))
    if len(tickers) > 1:
        print(f"\n  Per-ticker breakdown:")
        for t in tickers:
            t_settled = [e for e in settled if e["ticker"] == t]
            t_pnl = sum(e["settlement"]["total_pnl"] for e in t_settled)
            t_trades = sum(e["settlement"]["num_trades"] for e in t_settled)
            t_correct = sum(e["settlement"]["correct"] for e in t_settled)
            t_scored = sum(e["settlement"]["scored"] for e in t_settled)
            print(
                f"    {t}: {t_trades} trades, ${t_pnl:+.2f} P&L, "
                f"{t_correct}/{t_scored} correct"
            )

    # Per-cycle breakdown
    print(f"\n  Per-cycle breakdown:")
    for entry in settled:
        s = entry["settlement"]
        print(
            f"    {entry['ticker']} ({entry['timestamp'][:10]}): "
            f"{s['num_trades']} trades, ${s['total_pnl']:+.2f} P&L, "
            f"{s['correct']}/{s['scored']} correct"
        )

    # Save report
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_entries": len(journal),
        "settled_entries": len(settled),
        "pending_entries": len(pending),
        "cumulative": {
            "predictions_scored": total_scored,
            "correct": total_correct,
            "accuracy": total_correct / total_scored if total_scored > 0 else 0,
            "total_trades": total_trades,
            "total_pnl": total_pnl,
        },
        "per_ticker": {
            t: {
                "trades": sum(e["settlement"]["num_trades"] for e in settled if e["ticker"] == t),
                "pnl": sum(e["settlement"]["total_pnl"] for e in settled if e["ticker"] == t),
            }
            for t in tickers
        },
    }

    report_path = PAPER_TRADE_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Paper trading harness for Kalshi earnings mention contracts"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # snapshot
    snap = subparsers.add_parser("snapshot", help="Capture pre-earnings contract prices")
    snap.add_argument("--ticker", type=str, required=True)

    # predict
    pred = subparsers.add_parser("predict", help="Generate predictions")
    pred.add_argument("--ticker", type=str, required=True)
    pred.add_argument("--half-life", type=float, default=8.0)
    pred.add_argument("--edge-threshold", type=float, default=0.10)

    # settle
    sett = subparsers.add_parser("settle", help="Score settled predictions")
    sett.add_argument("--ticker", type=str, required=True)
    sett.add_argument("--position-size", type=float, default=100.0, help="Simulated $ per trade")

    # report
    rep = subparsers.add_parser("report", help="Generate performance report")
    rep.add_argument("--ticker", type=str, default=None, help="Filter by ticker")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\nCommands:")
        print("  snapshot  Capture active contract prices before earnings")
        print("  predict   Generate model predictions")
        print("  settle    Score predictions against settled outcomes")
        print("  report    View cumulative paper trading performance")
        return

    print("=" * 70)
    print("KALSHI EARNINGS PAPER TRADING")
    print("=" * 70)

    if args.command == "snapshot":
        cmd_snapshot(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "settle":
        cmd_settle(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
