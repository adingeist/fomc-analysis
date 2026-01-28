#!/usr/bin/env python3
"""
Unified pipeline for Kalshi earnings mention prediction.

One script to run the entire workflow:

    python scripts/pipeline.py train          # Build data + backtest
    python scripts/pipeline.py paper META     # Paper trade one ticker
    python scripts/pipeline.py status         # Show current state

Training pipeline (runs steps 1-3 automatically):
  1. Fetch ground truth from Kalshi settled contracts
  2. Download press releases from SEC EDGAR
  3. Run expanded backtest with synthetic history

Paper trading (interactive, per earnings cycle):
  1. Snapshot active contract prices
  2. Generate predictions
  3. (after earnings settles) Score and record P&L
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


DATA_DIR = Path("data")
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
PR_DIR = DATA_DIR / "earnings" / "press_releases"
BACKTEST_DIR = DATA_DIR / "expanded_backtest"
PAPER_DIR = DATA_DIR / "paper_trades"


def run_python(script_args: list[str], description: str) -> bool:
    """Run a python script and return True on success."""
    cmd = [sys.executable] + script_args
    print(f"\n  Running: {' '.join(script_args[:3])}...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  FAILED: {description}")
        return False
    return True


def cmd_train(args):
    """Build training data and run backtest."""
    from earnings_analysis.config import get_config
    cfg = get_config()
    tickers = args.tickers.split(",") if args.tickers else cfg.tickers

    print("=" * 70)
    print("STEP 1/3: Build ground truth from Kalshi settled contracts")
    print("=" * 70)

    if (GROUND_TRUTH_DIR / "ground_truth.json").exists() and not args.refresh:
        with open(GROUND_TRUTH_DIR / "summary.json") as f:
            summary = json.load(f)
        print(f"  Using cached ground truth: {summary['total_contracts']} contracts")
        print(f"  (pass --refresh to re-fetch from API)")
    else:
        from earnings_analysis.ground_truth import fetch_ground_truth, save_ground_truth
        dataset = fetch_ground_truth()
        if not dataset.contracts:
            print("  No settled contracts found. Check API credentials.")
            return
        save_ground_truth(dataset, GROUND_TRUTH_DIR)

    print(f"\n{'=' * 70}")
    print("STEP 2/3: Download press releases from SEC EDGAR")
    print("=" * 70)

    from earnings_analysis.fetchers.sec_edgar_transcripts import SECEdgarFetcher
    fetcher = SECEdgarFetcher(output_dir=DATA_DIR / "earnings" / "sec_filings")

    for ticker in tickers:
        ticker_pr_dir = PR_DIR / ticker
        existing = len(list(ticker_pr_dir.glob(f"{ticker}_*.txt"))) if ticker_pr_dir.exists() else 0

        if existing >= 6 and not args.refresh:
            print(f"  {ticker}: {existing} press releases cached (skip)")
            continue

        print(f"\n  Fetching {ticker}...")
        filings = fetcher.fetch_earnings_filings(ticker, num_quarters=8)
        saved = 0
        for filing in filings:
            if filing.has_press_release:
                text = fetcher.download_press_release(filing)
                if text and len(text) > 500:
                    ticker_pr_dir.mkdir(parents=True, exist_ok=True)
                    pr_file = ticker_pr_dir / f"{ticker}_{filing.filing_date}.txt"
                    if not pr_file.exists() or args.refresh:
                        pr_file.write_text(text)
                        saved += 1
        print(f"  {ticker}: {saved} new press releases saved")

    print(f"\n{'=' * 70}")
    print("STEP 3/3: Run expanded backtest")
    print("=" * 70)

    from earnings_analysis.ground_truth import load_ground_truth
    from earnings_analysis.synthetic_history import (
        build_expanded_training_data,
        print_expanded_data_summary,
    )
    from earnings_analysis.models.beta_binomial import BetaBinomialEarningsModel
    from earnings_analysis.kalshi.backtester import (
        EarningsKalshiBacktester,
        save_earnings_backtest_result,
    )

    edge_threshold = args.edge_threshold or cfg.edge_threshold
    half_life = args.half_life or cfg.half_life

    all_results = []

    for ticker in tickers:
        pr_count = len(list((PR_DIR / ticker).glob(f"{ticker}_*.txt"))) if (PR_DIR / ticker).exists() else 0
        if pr_count == 0:
            print(f"\n  {ticker}: no press releases, skipping")
            continue

        features, outcomes, diagnostics = build_expanded_training_data(
            ticker=ticker,
            ground_truth_dir=GROUND_TRUTH_DIR,
            pr_dir=PR_DIR,
            absent_in_pr_means_no=True,
        )

        if outcomes.empty:
            print(f"\n  {ticker}: no outcome data, skipping")
            continue

        print_expanded_data_summary(ticker, outcomes, diagnostics)

        if len(outcomes) < cfg.min_train_window + 1:
            print(f"  {ticker}: not enough dates for walk-forward ({len(outcomes)})")
            continue

        backtester = EarningsKalshiBacktester(
            features=features,
            outcomes=outcomes,
            model_class=BetaBinomialEarningsModel,
            model_params={"alpha_prior": 1.0, "beta_prior": 1.0, "half_life": half_life},
            edge_threshold=edge_threshold,
            min_train_window=cfg.min_train_window,
            require_variation=False,
        )

        result = backtester.run(ticker=ticker, initial_capital=cfg.initial_capital, market_prices=None)
        all_results.append(result)

        m = result.metrics
        print(f"    Accuracy: {m.accuracy:.1%}  Brier: {m.brier_score:.4f}  "
              f"Trades: {m.total_trades}  Win: {m.win_rate:.0%}")

        BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        save_earnings_backtest_result(result, BACKTEST_DIR / ticker)

    # Summary
    if all_results:
        total_preds = sum(r.metrics.total_predictions for r in all_results)
        total_trades = sum(r.metrics.total_trades for r in all_results)
        avg_brier = sum(r.metrics.brier_score for r in all_results) / len(all_results)
        avg_accuracy = sum(r.metrics.accuracy for r in all_results) / len(all_results)

        print(f"\n{'=' * 70}")
        print(f"TRAINING COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Companies:        {len(all_results)}")
        print(f"  Total predictions: {total_preds}")
        print(f"  Total trades:     {total_trades}")
        print(f"  Avg accuracy:     {avg_accuracy:.1%}")
        print(f"  Avg Brier score:  {avg_brier:.4f}")
        print(f"\n  Results saved to {BACKTEST_DIR}")
        print(f"\n  Next: python scripts/pipeline.py paper META")
        print(f"        (before the next META earnings call)")
    else:
        print("\n  No backtests completed.")


def cmd_paper(args):
    """Paper trade for a specific ticker."""
    from earnings_analysis.config import get_config
    cfg = get_config()

    ticker = args.ticker.upper()
    action = args.action  # snapshot, predict, settle, or auto

    if action == "auto":
        # Auto-detect: snapshot+predict if no pending, settle if pending
        journal = _load_paper_journal()
        pending = [e for e in journal if e.get("ticker") == ticker and e.get("status") == "pending"]
        if pending:
            action = "settle"
            print(f"Found pending predictions for {ticker}. Running settle...")
        else:
            action = "snapshot+predict"
            print(f"No pending predictions. Running snapshot + predict...")

    if action in ("snapshot", "snapshot+predict"):
        from fomc_analysis.kalshi_client_factory import get_kalshi_client
        from earnings_analysis.ground_truth import (
            fetch_ground_truth, save_ground_truth, load_ground_truth,
            EARNINGS_SERIES_TICKERS,
        )
        from earnings_analysis.models.beta_binomial import BetaBinomialEarningsModel

        client = get_kalshi_client()
        series_ticker = f"KXEARNINGSMENTION{ticker}"

        markets = client.get_markets(series_ticker=series_ticker, limit=200)
        active = [m for m in (markets or []) if m.get("status", "").lower() in ("active", "open")]

        if not active:
            print(f"No active contracts for {ticker}.")
            print(f"This ticker may not have upcoming earnings, or contracts haven't been listed yet.")
            return

        print(f"\nActive contracts for {ticker}: {len(active)}")

        # Snapshot
        snapshot = {
            "ticker": ticker,
            "snapshot_time": datetime.now().isoformat(),
            "contracts": [],
        }

        for market in active:
            cs = market.get("custom_strike") or {}
            word = cs.get("Word") or market.get("yes_sub_title", "")
            last_price = (market.get("last_price", 0) or 0) / 100.0
            yes_bid = (market.get("yes_bid", 0) or 0) / 100.0
            yes_ask = (market.get("yes_ask", 0) or 0) / 100.0

            snapshot["contracts"].append({
                "market_ticker": market.get("ticker", ""),
                "word": word,
                "last_price": last_price,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "mid_price": (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else last_price,
                "volume": market.get("volume", 0),
            })

        snap_dir = PAPER_DIR / "snapshots" / ticker
        snap_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_path = snap_dir / f"snapshot_{ts}.json"
        snap_path.write_text(json.dumps(snapshot, indent=2))
        print(f"Snapshot saved: {snap_path}")

        if action == "snapshot":
            print(f"\nNext: python scripts/pipeline.py paper {ticker} predict")
            return

        # Predict
        if not (GROUND_TRUTH_DIR / "ground_truth.json").exists():
            print("No ground truth. Run: python scripts/pipeline.py train")
            return

        # Use expanded training data if available
        from earnings_analysis.synthetic_history import build_expanded_training_data

        features, outcomes, diag = build_expanded_training_data(
            ticker=ticker,
            ground_truth_dir=GROUND_TRUTH_DIR,
            pr_dir=PR_DIR,
            absent_in_pr_means_no=True,
        )

        training_dates = len(outcomes) if not outcomes.empty else 0
        print(f"Training on {training_dates} historical events")

        edge_threshold = cfg.edge_threshold
        predictions = []

        for contract in snapshot["contracts"]:
            word = contract["word"].lower()
            market_price = contract["mid_price"]

            if not outcomes.empty and word in outcomes.columns:
                word_outcomes = outcomes[word].dropna()
                if len(word_outcomes) > 0:
                    model = BetaBinomialEarningsModel(
                        alpha_prior=1.0, beta_prior=1.0, half_life=cfg.half_life
                    )
                    model.fit(features.loc[word_outcomes.index], word_outcomes)
                    pred = model.predict()
                    prob = float(pred.iloc[0]["probability"])
                    history = f"{int(word_outcomes.sum())}/{len(word_outcomes)}"
                else:
                    prob = 0.5
                    history = "0/0"
            else:
                prob = 0.5
                history = "0/0"

            edge = prob - market_price
            if edge > edge_threshold and prob > 0.65:
                act = "BUY YES"
            elif edge < -edge_threshold and prob < 0.35:
                act = "BUY NO"
            else:
                act = "PASS"

            predictions.append({
                "word": contract["word"],
                "market_ticker": contract["market_ticker"],
                "market_price": market_price,
                "predicted_probability": prob,
                "edge": edge,
                "recommended_action": act,
                "history": history,
            })

        # Display
        print(f"\n{'Word':<30s} {'Mkt':>5s} {'Model':>5s} {'Edge':>6s} {'Action':<9s} {'Hist'}")
        print("-" * 75)
        for p in sorted(predictions, key=lambda x: abs(x["edge"]), reverse=True):
            print(
                f"{p['word']:<30s} {p['market_price']:>5.2f} "
                f"{p['predicted_probability']:>5.2f} {p['edge']:>+6.3f} "
                f"{p['recommended_action']:<9s} {p['history']}"
            )

        trades = [p for p in predictions if p["recommended_action"] != "PASS"]
        print(f"\nRecommended trades: {len(trades)}")
        for t in trades:
            print(f"  {t['recommended_action']:8s} {t['word']} @ {t['market_price']:.2f}")

        # Save to journal
        journal = _load_paper_journal()
        journal.append({
            "type": "prediction",
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
            "status": "pending",
        })
        _save_paper_journal(journal)
        print(f"\nPredictions recorded. After earnings settles:")
        print(f"  python scripts/pipeline.py paper {ticker} settle")

    elif action == "settle":
        from fomc_analysis.kalshi_client_factory import get_kalshi_client
        import numpy as np

        client = get_kalshi_client()
        series_ticker = f"KXEARNINGSMENTION{ticker}"
        markets = client.get_markets(series_ticker=series_ticker, limit=200)

        settled = {}
        for market in (markets or []):
            status = market.get("status", "").lower()
            if status in ("settled", "finalized", "closed"):
                cs = market.get("custom_strike") or {}
                word = cs.get("Word") or market.get("yes_sub_title", "")
                if word:
                    last_price = (market.get("last_price", 0) or 0) / 100.0
                    result_field = market.get("result", "")
                    outcome = 1 if (result_field and result_field.lower() in ("yes", "true", "1")) else (1 if last_price > 0.5 else 0)
                    settled[word.lower()] = outcome

        if not settled:
            print(f"No settled contracts yet for {ticker}. Check again later.")
            return

        journal = _load_paper_journal()
        pending = [e for e in journal if e.get("ticker") == ticker and e.get("status") == "pending"]
        if not pending:
            print(f"No pending predictions for {ticker}.")
            return

        entry = pending[-1]
        scored = 0
        correct = 0

        for pred in entry["predictions"]:
            word_lower = pred["word"].lower()
            if word_lower in settled:
                outcome = settled[word_lower]
                pred_yes = pred["predicted_probability"] > 0.5
                if pred_yes == (outcome == 1):
                    correct += 1
                scored += 1

        entry["status"] = "settled"
        entry["settlement"] = {
            "scored": scored,
            "correct": correct,
            "accuracy": correct / scored if scored > 0 else 0,
        }
        _save_paper_journal(journal)

        print(f"\nSettled: {correct}/{scored} correct ({correct/scored:.0%})" if scored else "No matches")
        print(f"\nView full report: python scripts/pipeline.py status")


def cmd_status(args):
    """Show current pipeline state."""
    print("=" * 70)
    print("PIPELINE STATUS")
    print("=" * 70)

    # Ground truth
    gt_path = GROUND_TRUTH_DIR / "summary.json"
    if gt_path.exists():
        with open(gt_path) as f:
            s = json.load(f)
        print(f"\n  Ground truth:     {s['total_contracts']} contracts, "
              f"{s['num_companies']} companies, {s['num_event_dates']} dates")
    else:
        print(f"\n  Ground truth:     NOT BUILT")
        print(f"                    Run: python scripts/pipeline.py train")

    # Press releases
    pr_counts = {}
    if PR_DIR.exists():
        for d in sorted(PR_DIR.iterdir()):
            if d.is_dir():
                count = len(list(d.glob("*.txt")))
                if count > 0:
                    pr_counts[d.name] = count
    if pr_counts:
        total = sum(pr_counts.values())
        tickers_str = ", ".join(f"{t}({c})" for t, c in pr_counts.items())
        print(f"  Press releases:   {total} total — {tickers_str}")
    else:
        print(f"  Press releases:   NONE")

    # Backtest results
    if BACKTEST_DIR.exists():
        bt_dirs = [d for d in BACKTEST_DIR.iterdir() if d.is_dir()]
        if bt_dirs:
            print(f"  Backtests:        {len(bt_dirs)} companies completed")
            for d in sorted(bt_dirs):
                report_file = d / "backtest_metrics.json"
                if report_file.exists():
                    with open(report_file) as f:
                        m = json.load(f)
                    print(f"    {d.name}: acc={m.get('accuracy', 0):.0%} "
                          f"brier={m.get('brier_score', 0):.3f} "
                          f"trades={m.get('total_trades', 0)}")
        else:
            print(f"  Backtests:        NONE")
    else:
        print(f"  Backtests:        NOT RUN")

    # Paper trades
    journal = _load_paper_journal()
    if journal:
        pending = [e for e in journal if e.get("status") == "pending"]
        settled = [e for e in journal if e.get("status") == "settled"]
        print(f"  Paper trades:     {len(journal)} entries ({len(settled)} settled, {len(pending)} pending)")
        for e in pending:
            trades = [p for p in e.get("predictions", []) if p.get("recommended_action") != "PASS"]
            print(f"    PENDING: {e['ticker']} ({e['timestamp'][:10]}) — {len(trades)} recommended trades")
        for e in settled:
            s = e.get("settlement", {})
            print(f"    SETTLED: {e['ticker']} ({e['timestamp'][:10]}) — "
                  f"{s.get('correct', 0)}/{s.get('scored', 0)} correct")
    else:
        print(f"  Paper trades:     NONE")

    # Next steps
    print(f"\n  Recommended next steps:")
    if not gt_path.exists():
        print(f"    1. python scripts/pipeline.py train")
    elif not pr_counts:
        print(f"    1. python scripts/pipeline.py train")
    elif not BACKTEST_DIR.exists():
        print(f"    1. python scripts/pipeline.py train")
    else:
        pending = [e for e in journal if e.get("status") == "pending"]
        if pending:
            ticker = pending[-1]["ticker"]
            print(f"    1. python scripts/pipeline.py paper {ticker} settle")
        else:
            print(f"    1. python scripts/pipeline.py paper META  (before next earnings)")


def _load_paper_journal() -> list:
    path = PAPER_DIR / "journal.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def _save_paper_journal(journal: list):
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    path = PAPER_DIR / "journal.json"
    with open(path, "w") as f:
        json.dump(journal, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="Kalshi earnings prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/pipeline.py train                  # Build data + backtest
  python scripts/pipeline.py paper META             # Snapshot + predict
  python scripts/pipeline.py paper META settle      # Score after settlement
  python scripts/pipeline.py status                 # Show current state
  python scripts/pipeline.py train --refresh        # Re-fetch everything
        """,
    )
    subparsers = parser.add_subparsers(dest="command")

    # train
    t = subparsers.add_parser("train", help="Build training data and run backtest")
    t.add_argument("--tickers", type=str, help="Comma-separated tickers (default: all)")
    t.add_argument("--refresh", action="store_true", help="Re-fetch all data from APIs")
    t.add_argument("--edge-threshold", type=float, default=None)
    t.add_argument("--half-life", type=float, default=None)

    # paper
    p = subparsers.add_parser("paper", help="Paper trade a ticker")
    p.add_argument("ticker", type=str, help="Stock ticker (e.g. META)")
    p.add_argument("action", nargs="?", default="auto",
                   choices=["auto", "snapshot", "snapshot+predict", "predict", "settle"],
                   help="Action (default: auto-detect)")

    # status
    subparsers.add_parser("status", help="Show pipeline status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "train":
        cmd_train(args)
    elif args.command == "paper":
        cmd_paper(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
