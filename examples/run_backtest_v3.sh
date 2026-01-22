#!/bin/bash
#
# Example script for running Time-Horizon Backtest v3
#
# This script demonstrates the complete workflow:
# 1. Fetch FOMC transcripts
# 2. Parse into speaker segments
# 3. Analyze Kalshi contracts
# 4. Run backtest
#
# Prerequisites:
# - Set environment variables in .env file:
#   * KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_BASE64 (or legacy credentials)
#   * OPENAI_API_KEY
#

set -e  # Exit on error

echo "=================================================="
echo "FOMC Analysis - Time-Horizon Backtest v3"
echo "=================================================="

# Configuration
START_YEAR=${START_YEAR:-2020}
END_YEAR=${END_YEAR:-2025}
SERIES_TICKER=${SERIES_TICKER:-KXFEDMENTION}
INITIAL_CAPITAL=${INITIAL_CAPITAL:-10000}
EDGE_THRESHOLD=${EDGE_THRESHOLD:-0.10}
POSITION_SIZE=${POSITION_SIZE:-0.05}
TRAIN_WINDOW=${TRAIN_WINDOW:-12}
TEST_START_DATE=${TEST_START_DATE:-2022-01-01}
FEE_RATE=${FEE_RATE:-0.07}
TRANSACTION_COST=${TRANSACTION_COST:-0.01}
SLIPPAGE=${SLIPPAGE:-0.01}
MAX_POSITION_SIZE=${MAX_POSITION_SIZE:-1500}
YES_EDGE_THRESHOLD=${YES_EDGE_THRESHOLD:-0.12}
NO_EDGE_THRESHOLD=${NO_EDGE_THRESHOLD:-0.08}
YES_POSITION_SIZE=${YES_POSITION_SIZE:-0.04}
NO_POSITION_SIZE=${NO_POSITION_SIZE:-0.03}

echo ""
echo "Configuration:"
echo "  Year range: $START_YEAR - $END_YEAR"
echo "  Series ticker: $SERIES_TICKER"
echo "  Initial capital: \$$INITIAL_CAPITAL"
echo "  Edge threshold: ${EDGE_THRESHOLD} ($(echo "$EDGE_THRESHOLD * 100" | bc)%)"
echo "  Position size: ${POSITION_SIZE} ($(echo "$POSITION_SIZE * 100" | bc)%)"
echo "  Train window: ${TRAIN_WINDOW} meetings"
echo "  Test start date: ${TEST_START_DATE}"
echo "  Fee rate: ${FEE_RATE}"
echo "  Transaction cost: ${TRANSACTION_COST}"
echo "  Slippage: ${SLIPPAGE}"
echo "  Max position size: ${MAX_POSITION_SIZE}"
echo "  YES edge threshold / size: ${YES_EDGE_THRESHOLD} / ${YES_POSITION_SIZE}"
echo "  NO edge threshold / size: ${NO_EDGE_THRESHOLD} / ${NO_POSITION_SIZE}"
echo ""

# Step 1: Fetch transcripts (if not already done)
if [ ! -d "data/raw_pdf" ] || [ -z "$(ls -A data/raw_pdf 2>/dev/null)" ]; then
    echo "Step 1: Fetching FOMC transcripts..."
    fomc-analysis fetch-transcripts \
        --start-year $START_YEAR \
        --end-year $END_YEAR \
        --out-dir data/raw_pdf
    echo "✓ Transcripts fetched"
else
    echo "Step 1: Skipping fetch (transcripts already exist)"
fi

# Step 2: Parse transcripts (if not already done)
if [ ! -d "data/segments" ] || [ -z "$(ls -A data/segments 2>/dev/null)" ]; then
    echo ""
    echo "Step 2: Parsing transcripts..."
    fomc-analysis parse \
        --input-dir data/raw_pdf \
        --mode deterministic \
        --segments-dir data/segments
    echo "✓ Transcripts parsed"
else
    echo "Step 2: Skipping parse (segments already exist)"
fi

# Step 3: Analyze Kalshi contracts
echo ""
echo "Step 3: Analyzing Kalshi contracts..."
fomc-analysis analyze-kalshi-contracts \
    --series-ticker $SERIES_TICKER \
    --segments-dir data/segments \
    --output-dir data/kalshi_analysis \
    --scope powell_only \
    --market-status resolved

echo "✓ Kalshi contracts analyzed"

# Step 4: Run backtest
echo ""
echo "Step 4: Running time-horizon backtest..."
fomc-analysis backtest-v3 \
    --contract-words data/kalshi_analysis/contract_words.json \
    --segments-dir data/segments \
    --model beta \
    --alpha 1.0 \
    --beta-prior 1.0 \
    --half-life 4 \
    --horizons "7,14,30" \
    --edge-threshold $EDGE_THRESHOLD \
    --yes-edge-threshold $YES_EDGE_THRESHOLD \
    --no-edge-threshold $NO_EDGE_THRESHOLD \
    --position-size-pct $POSITION_SIZE \
    --yes-position-size-pct $YES_POSITION_SIZE \
    --no-position-size-pct $NO_POSITION_SIZE \
    --max-position-size $MAX_POSITION_SIZE \
    --initial-capital $INITIAL_CAPITAL \
    --fee-rate $FEE_RATE \
    --transaction-cost $TRANSACTION_COST \
    --slippage $SLIPPAGE \
    --train-window-size $TRAIN_WINDOW \
    --test-start-date $TEST_START_DATE \
    --output results/backtest_v3

echo ""
echo "=================================================="
echo "✓ Backtest complete!"
echo "=================================================="
echo ""
echo "Results saved to: results/backtest_v3/"
echo ""
echo "Files created:"
echo "  - backtest_results.json (complete results)"
echo "  - predictions.csv (all predictions)"
echo "  - trades.csv (executed trades)"
echo "  - horizon_metrics.csv (performance by time horizon)"
echo ""
echo "View results summary:"
echo "  cat results/backtest_v3/horizon_metrics.csv"
echo ""
