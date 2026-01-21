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

echo ""
echo "Configuration:"
echo "  Year range: $START_YEAR - $END_YEAR"
echo "  Series ticker: $SERIES_TICKER"
echo "  Initial capital: \$$INITIAL_CAPITAL"
echo "  Edge threshold: ${EDGE_THRESHOLD} ($(echo "$EDGE_THRESHOLD * 100" | bc)%)"
echo "  Position size: ${POSITION_SIZE} ($(echo "$POSITION_SIZE * 100" | bc)%)"
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
    --position-size-pct $POSITION_SIZE \
    --initial-capital $INITIAL_CAPITAL \
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
