#!/usr/bin/env python
"""
Verify Earnings Call Kalshi Framework - End-to-End Test

This script demonstrates the complete earnings Kalshi framework workflow:
1. Create mock Kalshi contracts
2. Generate sample earnings call transcript segments
3. Featurize based on contract words
4. Make predictions using Beta-Binomial model
5. Run backtest

This verifies the framework works without requiring:
- Real Kalshi API credentials
- Real earnings call transcripts
- Actual Kalshi market prices
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Core imports
from earnings_analysis.kalshi import (
    EarningsContractWord,
    EarningsMentionAnalysis,
    EarningsContractAnalyzer,
)
from earnings_analysis.kalshi.backtester import (
    EarningsKalshiBacktester,
    save_earnings_backtest_result,
    create_microstructure_backtester,
)
from earnings_analysis.models import BetaBinomialEarningsModel


def create_mock_transcript_segments(
    ticker: str,
    call_dates: list,
    contract_words: list,
    output_dir: Path,
):
    """
    Create mock earnings call transcript segments.

    For each call date, creates a JSONL file with segments containing:
    - Speaker role (CEO, CFO, Analyst)
    - Text with varying word mentions
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Templates for different speakers
    ceo_templates = [
        "I'm excited to share our strong {word} performance this quarter.",
        "Our {word} strategy is delivering excellent results.",
        "We continue to focus on {word} and innovation.",
        "Looking ahead, {word} remains a key priority.",
        "The team has done exceptional work on {word}.",
    ]

    cfo_templates = [
        "Our {word} metrics show solid growth.",
        "We're seeing positive trends in {word}.",
        "From a financial perspective, {word} is performing well.",
        "This contributed significantly to our margins.",
    ]

    analyst_templates = [
        "Can you comment on {word}?",
        "What's your outlook for {word}?",
        "How does {word} fit into your strategy?",
    ]

    for i, call_date in enumerate(call_dates):
        segments = []

        # Each call has 20-30 segments
        n_segments = np.random.randint(20, 31)

        for seg_idx in range(n_segments):
            # Determine speaker
            role_choice = np.random.choice(
                ["ceo", "cfo", "analyst"],
                p=[0.4, 0.3, 0.3]
            )

            if role_choice == "ceo":
                speaker = "CEO"
                role = "ceo"
                templates = ceo_templates
            elif role_choice == "cfo":
                speaker = "CFO"
                role = "cfo"
                templates = cfo_templates
            else:
                speaker = "Analyst"
                role = "analyst"
                templates = analyst_templates

            # Generate text with random word mentions
            text_parts = []
            n_sentences = np.random.randint(2, 5)

            for _ in range(n_sentences):
                # Pick a random contract word
                word = np.random.choice(contract_words)
                template = np.random.choice(templates)

                # Randomly decide whether to actually include the word
                if np.random.random() < 0.4:  # 40% chance to mention
                    sentence = template.format(word=word)
                else:
                    # Replace with generic content
                    sentence = template.format(word="growth").replace("growth", "our operations")

                text_parts.append(sentence)

            segment = {
                "speaker": speaker,
                "role": role,
                "text": " ".join(text_parts),
                "segment_idx": seg_idx,
            }

            segments.append(segment)

        # Save to JSONL
        output_file = output_dir / f"{ticker}_{call_date}.jsonl"
        with open(output_file, "w") as f:
            for seg in segments:
                f.write(json.dumps(seg) + "\n")

        print(f"  Created {output_file.name} with {len(segments)} segments")


def create_mock_contracts(ticker: str) -> list:
    """Create mock Kalshi earnings mention contracts."""

    # Common words tracked in earnings calls
    words = ["AI", "cloud", "revenue", "margin", "innovation"]

    contracts = []
    for word in words:
        contract = EarningsContractWord(
            word=word.lower(),
            ticker=ticker,
            market_ticker=f"KXEARNINGS{ticker}{word.upper()}",
            market_title=f"Will {ticker} CEO say '{word}' at least 3 times?",
            threshold=3,
            markets=[],
        )
        contracts.append(contract)

    return contracts


def analyze_mock_transcripts(
    ticker: str,
    contracts: list,
    segments_dir: Path,
) -> list:
    """Analyze mock transcripts to count word mentions."""

    analyses = []

    # Get all transcript files
    segment_files = sorted(segments_dir.glob(f"{ticker}_*.jsonl"))

    print(f"\nAnalyzing {len(segment_files)} earnings calls...")

    for contract in contracts:
        word = contract.word
        threshold = contract.threshold

        call_counts = []

        for segment_file in segment_files:
            # Load segments
            segments = []
            with open(segment_file, "r") as f:
                for line in f:
                    segments.append(json.loads(line))

            # Filter to executives only
            exec_segments = [
                seg for seg in segments
                if seg.get("role") in ("ceo", "cfo")
            ]

            # Combine text
            combined_text = " ".join(seg["text"] for seg in exec_segments)

            # Count mentions (case-insensitive, word boundaries)
            import re
            pattern = r"\b" + re.escape(word) + r"\b"
            mentions = len(re.findall(pattern, combined_text, re.IGNORECASE))

            call_counts.append(mentions)

        # Calculate statistics
        total_calls = len(call_counts)
        calls_with_mention = sum(1 for count in call_counts if count >= threshold)
        total_mentions = sum(call_counts)

        avg_mentions = total_mentions / total_calls if total_calls > 0 else 0
        max_mentions = max(call_counts) if call_counts else 0
        mention_frequency = calls_with_mention / total_calls if total_calls > 0 else 0

        # Distribution
        distribution = {}
        for count in call_counts:
            distribution[count] = distribution.get(count, 0) + 1

        analysis = EarningsMentionAnalysis(
            word=word,
            ticker=ticker,
            total_calls=total_calls,
            calls_with_mention=calls_with_mention,
            mention_frequency=mention_frequency,
            total_mentions=total_mentions,
            avg_mentions_per_call=avg_mentions,
            max_mentions_in_call=max_mentions,
            mention_counts_distribution=distribution,
        )

        analyses.append(analysis)

    return analyses


def create_features_and_outcomes(
    call_dates: list,
    analyses: list,
    segments_dir: Path,
    ticker: str,
) -> tuple:
    """
    Create features and outcomes dataframes for backtesting.

    Features: word count features
    Outcomes: binary (1 if word mentioned >= threshold, 0 otherwise)
    """

    features_data = []
    outcomes_data = []

    for call_date in call_dates:
        feature_row = {}
        outcome_row = {}

        # Load segments for this call
        segment_file = segments_dir / f"{ticker}_{call_date}.jsonl"
        segments = []
        with open(segment_file, "r") as f:
            for line in f:
                segments.append(json.loads(line))

        # Filter to executives
        exec_segments = [
            seg for seg in segments
            if seg.get("role") in ("ceo", "cfo")
        ]
        combined_text = " ".join(seg["text"] for seg in exec_segments)

        # Count each word
        import re
        for analysis in analyses:
            word = analysis.word
            pattern = r"\b" + re.escape(word) + r"\b"
            count = len(re.findall(pattern, combined_text, re.IGNORECASE))

            feature_row[word] = count

            # Outcome: 1 if count >= 3 (our threshold), 0 otherwise
            outcome_row[word] = 1 if count >= 3 else 0

        features_data.append(feature_row)
        outcomes_data.append(outcome_row)

    # Create dataframes
    features_df = pd.DataFrame(features_data, index=call_dates)
    outcomes_df = pd.DataFrame(outcomes_data, index=call_dates)

    return features_df, outcomes_df


def run_backtest(
    ticker: str,
    features_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    output_dir: Path,
):
    """Run backtest using Beta-Binomial model (basic + microstructure)."""

    model_params = {
        "alpha_prior": 1.0,
        "beta_prior": 1.0,
        "half_life": 8.0,  # Weight recent calls more
    }

    # ── Basic backtest (no microstructure) ──
    print(f"\n{'='*60}")
    print("RUNNING BASIC BACKTEST (no microstructure)")
    print('='*60)

    backtester = EarningsKalshiBacktester(
        features=features_df,
        outcomes=outcomes_df,
        model_class=BetaBinomialEarningsModel,
        model_params=model_params,
        edge_threshold=0.12,
        position_size_pct=0.03,
        fee_rate=0.07,
        min_train_window=4,
    )

    result = backtester.run(
        ticker=ticker,
        initial_capital=10000.0,
        market_prices=None,
    )

    print(f"\nBasic Backtest Results for {ticker}:")
    print(f"  Predictions: {result.metrics.total_predictions}")
    print(f"  Accuracy: {result.metrics.accuracy:.1%}")
    print(f"  Brier Score: {result.metrics.brier_score:.3f}")
    print(f"\nTrading Results:")
    print(f"  Total Trades: {result.metrics.total_trades}")
    print(f"  Win Rate: {result.metrics.win_rate:.1%}")
    print(f"  Total P&L: ${result.metrics.total_pnl:,.2f}")
    print(f"  ROI: {result.metrics.roi:.1%}")
    print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"  Microstructure: {result.metadata.get('microstructure', {})}")

    save_earnings_backtest_result(result, output_dir / "basic")

    # ── Microstructure-enhanced backtest ──
    print(f"\n{'='*60}")
    print("RUNNING MICROSTRUCTURE-ENHANCED BACKTEST")
    print('='*60)

    micro_backtester = create_microstructure_backtester(
        features=features_df,
        outcomes=outcomes_df,
        model_class=BetaBinomialEarningsModel,
        model_params=model_params,
        execution_mode="hybrid",
        spread_filter_min_net_edge=0.03,
        edge_threshold=0.12,
        position_size_pct=0.03,
        fee_rate=0.07,
        min_train_window=4,
    )

    micro_result = micro_backtester.run(
        ticker=ticker,
        initial_capital=10000.0,
        market_prices=None,
    )

    print(f"\nMicrostructure Backtest Results for {ticker}:")
    print(f"  Predictions: {micro_result.metrics.total_predictions}")
    print(f"  Accuracy: {micro_result.metrics.accuracy:.1%}")
    print(f"  Brier Score: {micro_result.metrics.brier_score:.3f}")
    print(f"\nTrading Results:")
    print(f"  Total Trades: {micro_result.metrics.total_trades}")
    print(f"  Win Rate: {micro_result.metrics.win_rate:.1%}")
    print(f"  Total P&L: ${micro_result.metrics.total_pnl:,.2f}")
    print(f"  ROI: {micro_result.metrics.roi:.1%}")
    print(f"  Sharpe Ratio: {micro_result.metrics.sharpe_ratio:.2f}")
    print(f"  Microstructure: {micro_result.metadata.get('microstructure', {})}")

    save_earnings_backtest_result(micro_result, output_dir / "microstructure")

    # ── Comparison ──
    print(f"\n{'='*60}")
    print("COMPARISON: Basic vs Microstructure-Enhanced")
    print('='*60)
    print(f"  {'Metric':<25} {'Basic':>12} {'Micro':>12} {'Delta':>12}")
    print(f"  {'-'*61}")
    print(f"  {'Total Trades':<25} {result.metrics.total_trades:>12} {micro_result.metrics.total_trades:>12} {micro_result.metrics.total_trades - result.metrics.total_trades:>+12}")
    print(f"  {'Win Rate':<25} {result.metrics.win_rate:>11.1%} {micro_result.metrics.win_rate:>11.1%} {micro_result.metrics.win_rate - result.metrics.win_rate:>+11.1%}")
    print(f"  {'Total P&L':<25} ${result.metrics.total_pnl:>10,.2f} ${micro_result.metrics.total_pnl:>10,.2f} ${micro_result.metrics.total_pnl - result.metrics.total_pnl:>+10,.2f}")
    print(f"  {'ROI':<25} {result.metrics.roi:>11.1%} {micro_result.metrics.roi:>11.1%} {micro_result.metrics.roi - result.metrics.roi:>+11.1%}")
    print(f"  {'Sharpe Ratio':<25} {result.metrics.sharpe_ratio:>12.2f} {micro_result.metrics.sharpe_ratio:>12.2f} {micro_result.metrics.sharpe_ratio - result.metrics.sharpe_ratio:>+12.2f}")

    return micro_result


def main():
    """Run complete verification workflow."""

    print("="*60)
    print("EARNINGS CALL KALSHI FRAMEWORK VERIFICATION")
    print("="*60)

    # Configuration
    TICKER = "META"
    N_CALLS = 12  # 3 years of quarterly calls
    OUTPUT_DIR = Path("data/verification")

    print(f"\nConfiguration:")
    print(f"  Ticker: {TICKER}")
    print(f"  Number of Calls: {N_CALLS}")
    print(f"  Output Directory: {OUTPUT_DIR}")

    # Step 1: Create mock contracts
    print(f"\n{'='*60}")
    print("STEP 1: CREATE MOCK KALSHI CONTRACTS")
    print('='*60)

    contracts = create_mock_contracts(TICKER)
    print(f"\nCreated {len(contracts)} mock contracts:")
    for contract in contracts:
        print(f"  - {contract.word.upper()}: {contract.market_title}")

    # Step 2: Generate mock transcript segments
    print(f"\n{'='*60}")
    print("STEP 2: GENERATE MOCK TRANSCRIPT SEGMENTS")
    print('='*60)

    # Create quarterly call dates
    base_date = datetime(2021, 1, 1)
    call_dates = [
        (base_date + timedelta(days=90*i)).strftime("%Y-%m-%d")
        for i in range(N_CALLS)
    ]

    segments_dir = OUTPUT_DIR / "segments"
    contract_words = [c.word for c in contracts]

    print(f"\nGenerating {N_CALLS} earnings call transcripts...")
    create_mock_transcript_segments(
        ticker=TICKER,
        call_dates=call_dates,
        contract_words=contract_words,
        output_dir=segments_dir,
    )

    # Step 3: Analyze transcripts
    print(f"\n{'='*60}")
    print("STEP 3: ANALYZE TRANSCRIPTS FOR WORD MENTIONS")
    print('='*60)

    analyses = analyze_mock_transcripts(
        ticker=TICKER,
        contracts=contracts,
        segments_dir=segments_dir,
    )

    print(f"\nMention Analysis Results:")
    for analysis in analyses:
        print(f"\n  {analysis.word.upper()}:")
        print(f"    Total Calls: {analysis.total_calls}")
        print(f"    Calls with Mention (≥3): {analysis.calls_with_mention}")
        print(f"    Mention Frequency: {analysis.mention_frequency:.1%}")
        print(f"    Avg Mentions/Call: {analysis.avg_mentions_per_call:.1f}")
        print(f"    Max Mentions: {analysis.max_mentions_in_call}")

    # Step 4: Create features and outcomes
    print(f"\n{'='*60}")
    print("STEP 4: CREATE FEATURES AND OUTCOMES")
    print('='*60)

    features_df, outcomes_df = create_features_and_outcomes(
        call_dates=call_dates,
        analyses=analyses,
        segments_dir=segments_dir,
        ticker=TICKER,
    )

    print(f"\nFeatures DataFrame:")
    print(features_df.head())
    print(f"\nOutcomes DataFrame:")
    print(outcomes_df.head())

    # Step 5: Run backtest
    backtest_output = OUTPUT_DIR / "backtest"
    result = run_backtest(
        ticker=TICKER,
        features_df=features_df,
        outcomes_df=outcomes_df,
        output_dir=backtest_output,
    )

    # Summary
    print(f"\n{'='*60}")
    print("VERIFICATION COMPLETE!")
    print('='*60)
    print(f"\nAll framework components working correctly:")
    print(f"  - Mock Kalshi contracts created")
    print(f"  - Transcript segments generated")
    print(f"  - Word mentions analyzed")
    print(f"  - Features and outcomes extracted")
    print(f"  - Beta-Binomial model predictions made")
    print(f"  - Basic backtest completed")
    print(f"  - Microstructure-enhanced backtest completed")
    print(f"    - Calibration curve (gamma={result.metadata['microstructure']['calibration_gamma']})")
    print(f"    - Execution simulator ({result.metadata['microstructure']['execution_mode']})")
    print(f"    - Spread filter (min_net_edge={result.metadata['microstructure']['spread_filter_min_net_edge']})")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nNext Steps:")
    print(f"  1. Fetch actual earnings call transcripts")
    print(f"  2. Run real backtest with historical Kalshi data")
    print(f"  3. Tune microstructure parameters with real trade data")


if __name__ == "__main__":
    main()
