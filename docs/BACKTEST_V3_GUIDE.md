# Time-Horizon Backtest v3 Guide

## Overview

The new Time-Horizon Backtest (v3) addresses the limitations of the previous backtest by:

1. **Using Actual Kalshi Outcomes**: Fetches real contract resolution data (100% YES or 0% NO) from Kalshi API
2. **Multi-Horizon Predictions**: Makes predictions at 7, 14, and 30 days before each FOMC meeting
3. **Accurate Performance Tracking**: Measures prediction accuracy and profitability for each time horizon
4. **Realistic Trading Simulation**: Includes Kalshi's 7% fee on profits and proper position sizing

## Key Improvements Over v2

| Feature | Backtest v2 | Backtest v3 |
|---------|------------|-------------|
| Outcome source | Transcript features only | Actual Kalshi contract resolutions |
| Prediction timing | One prediction per meeting | Multiple predictions (7, 14, 30 days before) |
| Accuracy tracking | Overall only | Per time-horizon breakdown |
| Market prices | Optional, limited | Historical prices at each prediction time |
| Performance metrics | Basic | Comprehensive (accuracy, Brier score, ROI per horizon) |

## Prerequisites

1. **Kalshi API Credentials**: You need either:
   - Legacy: `KALSHI_API_KEY` and `KALSHI_API_SECRET`
   - Or RSA: `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_BASE64`

2. **OpenAI API Key**: For generating phrase variants
   - `OPENAI_API_KEY`

3. **Data Requirements**:
   - FOMC transcript PDFs (parsed into segments)
   - Contract words JSON file with Kalshi market data

## Step-by-Step Workflow

### Step 1: Fetch and Parse Transcripts

```bash
# Fetch FOMC transcripts
fomc-analysis fetch-transcripts --start-year 2020 --end-year 2025

# Parse transcripts into speaker segments
fomc-analysis parse \
  --input-dir data/raw_pdf \
  --mode deterministic
```

### Step 2: Analyze Kalshi Contracts

This step fetches Kalshi mention contracts and builds historical statistics:

```bash
# Analyze Kalshi contracts and export to JSON
fomc-analysis analyze-kalshi-contracts \
  --series-ticker KXFEDMENTION \
  --segments-dir data/segments \
  --output-dir data/kalshi_analysis \
  --scope powell_only \
  --market-status resolved
```

This creates:
- `data/kalshi_analysis/contract_words.json` - Contract definitions with resolved outcomes
- `data/kalshi_analysis/statistics.json` - Historical mention statistics

### Step 3: Run the Time-Horizon Backtest

```bash
# Run backtest with Beta-Binomial model
fomc-analysis backtest-v3 \
  --contract-words data/kalshi_analysis/contract_words.json \
  --segments-dir data/segments \
  --model beta \
  --alpha 1.0 \
  --beta-prior 1.0 \
  --half-life 4 \
  --horizons "7,14,30" \
  --edge-threshold 0.10 \
  --position-size-pct 0.05 \
  --initial-capital 10000.0 \
  --output results/backtest_v3
```

### Step 4: Analyze Results

The backtest creates several output files:

```
results/backtest_v3/
├── backtest_results.json    # Complete results with all data
├── predictions.csv          # All predictions made
├── trades.csv              # All trades executed
└── horizon_metrics.csv     # Performance by time horizon
```

## Understanding the Output

### Overall Metrics

```
Overall Performance:
  Total predictions: 450         # Predictions made across all horizons
  Total trades: 127             # Trades executed (where edge > threshold)
  Win rate: 58.3%              # Percentage of profitable trades
  Total P&L: $3,245.12         # Total profit/loss
  ROI: 32.5%                   # Return on initial capital
  Sharpe ratio: 1.42           # Risk-adjusted return
  Final capital: $13,245.12    # Ending capital
```

### Per-Horizon Metrics

```
7 days before meeting:
  Predictions: 150             # Number of predictions at this horizon
  Accuracy: 62.7%             # Percentage of correct predictions
  Trades: 42                  # Trades executed at this horizon
  Win rate: 61.9%            # Percentage of profitable trades
  Total P&L: $1,234.56       # P&L for this horizon
  Avg P&L/trade: $29.39      # Average profit per trade
  ROI: 24.2%                 # Return on investment
  Brier score: 0.185         # Calibration metric (lower is better)
```

**Key Insights**:
- **Accuracy**: How often the model predicted the correct outcome (>50% prob = YES)
- **Brier Score**: Measures calibration (0 = perfect, 0.25 = random guessing)
- **ROI**: Return on invested capital for this specific horizon
- **Win Rate**: Percentage of trades that were profitable

## Model Selection

### Beta-Binomial Model (Recommended)

```bash
--model beta \
--alpha 1.0 \          # Beta prior alpha (1.0 = uniform prior)
--beta-prior 1.0 \     # Beta prior beta
--half-life 4          # Recent events weighted more (4 meetings)
```

**Best for**: Small sample sizes, natural uncertainty quantification

### EWMA Model

```bash
--model ewma \
--alpha 0.5            # Smoothing factor (higher = more recent weight)
```

**Best for**: Simple baseline, faster computation

## Parameter Tuning

### Edge Threshold

Controls how large the model-market disagreement must be to trade:

```bash
--edge-threshold 0.05   # Trade if model differs by 5% (aggressive)
--edge-threshold 0.10   # Trade if model differs by 10% (balanced)
--edge-threshold 0.15   # Trade if model differs by 15% (conservative)
```

**Lower threshold** = More trades, potentially lower edge per trade
**Higher threshold** = Fewer trades, higher edge per trade

### Position Size

Controls risk per trade:

```bash
--position-size-pct 0.02   # 2% of capital per trade (conservative)
--position-size-pct 0.05   # 5% of capital per trade (balanced)
--position-size-pct 0.10   # 10% of capital per trade (aggressive)
```

### Prediction Horizons

Customize when predictions are made:

```bash
--horizons "7,14,30"        # Default: 1 week, 2 weeks, 1 month before
--horizons "1,3,7,14"       # More frequent: 1, 3, 7, 14 days before
--horizons "14,30"          # Longer term only
```

## Interpreting Results

### Is the Strategy Profitable?

Look at these key metrics:

1. **Overall ROI > 0**: Strategy makes money
2. **Sharpe Ratio > 1.0**: Good risk-adjusted returns
3. **Win Rate > 55%**: Above breakeven after fees
4. **Brier Score < 0.20**: Well-calibrated predictions

### Which Horizon Works Best?

Compare the per-horizon metrics:

- **Best Accuracy**: Which horizon predicts outcomes most accurately?
- **Best ROI**: Which horizon generates the most profit?
- **Best Sharpe**: Which horizon has the best risk-adjusted returns?

Often you'll find:
- **7 days before**: Higher accuracy (more recent data), but lower market edges
- **30 days before**: Lower accuracy, but potentially larger market edges

## Common Issues

### No Resolved Contracts Found

**Error**: `No resolved contract outcomes found`

**Solution**:
```bash
# Make sure you're analyzing resolved markets
fomc-analysis analyze-kalshi-contracts \
  --market-status resolved   # ← Important!
```

### No Historical Prices

**Warning**: `No historical prices found. Trades will be skipped.`

**Cause**: Kalshi API may not have historical price data for older markets

**Impact**: Backtest can still measure prediction accuracy, but won't simulate trades

### Too Few Trades

If you get very few trades:

1. **Lower edge threshold**: Try `--edge-threshold 0.05`
2. **Check price data**: Ensure historical prices are available
3. **More training data**: Ensure you have enough historical meetings

## Example Analysis Workflow

```bash
# 1. Fetch data
fomc-analysis fetch-transcripts --start-year 2020 --end-year 2025
fomc-analysis parse --input-dir data/raw_pdf --mode deterministic

# 2. Analyze Kalshi contracts
fomc-analysis analyze-kalshi-contracts \
  --series-ticker KXFEDMENTION \
  --market-status resolved \
  --output-dir data/kalshi_analysis

# 3. Run backtest with different configurations
# Conservative
fomc-analysis backtest-v3 \
  --contract-words data/kalshi_analysis/contract_words.json \
  --edge-threshold 0.15 \
  --position-size-pct 0.02 \
  --output results/conservative

# Balanced
fomc-analysis backtest-v3 \
  --contract-words data/kalshi_analysis/contract_words.json \
  --edge-threshold 0.10 \
  --position-size-pct 0.05 \
  --output results/balanced

# Aggressive
fomc-analysis backtest-v3 \
  --contract-words data/kalshi_analysis/contract_words.json \
  --edge-threshold 0.05 \
  --position-size-pct 0.10 \
  --output results/aggressive

# 4. Compare results
python -c "
import pandas as pd
import json
from pathlib import Path

for strategy in ['conservative', 'balanced', 'aggressive']:
    result = json.loads(Path(f'results/{strategy}/backtest_results.json').read_text())
    print(f'\n{strategy.upper()}:')
    print(f'  ROI: {result['overall_metrics']['roi']*100:.1f}%')
    print(f'  Sharpe: {result['overall_metrics']['sharpe']:.2f}')
    print(f'  Win Rate: {result['overall_metrics']['win_rate']*100:.1f}%')
"
```

## Next Steps

1. **Analyze trade patterns**: Look at `trades.csv` to see which contracts are most profitable
2. **Optimize parameters**: Try different edge thresholds and position sizes
3. **Test different models**: Compare Beta-Binomial vs EWMA
4. **Add features**: Consider incorporating sentiment, economic indicators, etc.

## Advanced: Custom Analysis

```python
import pandas as pd
import json
from pathlib import Path

# Load results
results = json.loads(Path('results/backtest_v3/backtest_results.json').read_text())

# Analyze predictions
predictions_df = pd.DataFrame(results['predictions'])

# Which contracts are most predictable?
contract_accuracy = predictions_df.groupby('contract').agg({
    'correct': 'mean',
    'predicted_probability': 'count'
}).rename(columns={'predicted_probability': 'count'})

print("Most Predictable Contracts:")
print(contract_accuracy.sort_values('correct', ascending=False).head(10))

# Analyze trades
trades_df = pd.DataFrame(results['trades'])

# Which contracts are most profitable?
contract_pnl = trades_df.groupby('contract').agg({
    'pnl': ['sum', 'mean', 'count']
})

print("\nMost Profitable Contracts:")
print(contract_pnl.sort_values(('pnl', 'sum'), ascending=False).head(10))
```

## Support

For issues or questions:
1. Check that all API credentials are set correctly
2. Ensure you have resolved Kalshi markets in your contract_words.json
3. Verify that segments have been parsed correctly
4. Review the error messages - they usually indicate the specific issue
