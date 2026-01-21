# End-to-End Backtest Guide

## Overview

The `run_e2e_backtest.py` script provides a complete, automated workflow for:

1. **Fetching** FOMC press conference transcripts
2. **Parsing** transcripts into speaker segments
3. **Analyzing** Kalshi contracts and generating contract mappings
4. **Running** the Time-Horizon Backtest v3
5. **Visualizing** word frequencies for traded contracts

## Quick Start

### Prerequisites

1. **Install the package**:
```bash
pip install -e .
```

2. **Set up API credentials** in `.env` file:
```bash
# Kalshi API (choose one auth method)
KALSHI_API_KEY_ID=your_key_id
KALSHI_PRIVATE_KEY_BASE64=your_base64_key

# Or legacy:
# KALSHI_API_KEY=your_key
# KALSHI_API_SECRET=your_secret

# OpenAI API
OPENAI_API_KEY=your_openai_key
```

### Run the Complete Pipeline

```bash
# Clean data and run full pipeline
python run_e2e_backtest.py --clean

# Use existing data if available
python run_e2e_backtest.py
```

## Command-Line Options

### Basic Options

- `--clean`: Remove all data and start fresh
- `--start-year YEAR`: Start year for transcripts (default: 2020)
- `--end-year YEAR`: End year for transcripts (default: 2025)
- `--series-ticker TICKER`: Kalshi series (default: KXFEDMENTION)

### Trading Parameters

- `--edge-threshold FLOAT`: Minimum edge to trade (default: 0.10 = 10%)
- `--position-size FLOAT`: Position size as % of capital (default: 0.05 = 5%)
- `--initial-capital FLOAT`: Starting capital (default: 10000)
- `--horizons DAYS`: Prediction horizons (default: "7,14,30")

### Skip Options (for faster re-runs)

- `--skip-fetch`: Skip transcript fetch
- `--skip-parse`: Skip parsing
- `--skip-analyze`: Skip Kalshi analysis

## Usage Examples

### Example 1: Full Clean Run

```bash
# Start from scratch with all data
python run_e2e_backtest.py --clean \
  --start-year 2020 \
  --end-year 2025
```

### Example 2: Quick Re-run with Existing Data

```bash
# Re-run backtest with different parameters
# Skip fetch/parse/analyze to save time
python run_e2e_backtest.py \
  --skip-fetch \
  --skip-parse \
  --skip-analyze \
  --edge-threshold 0.15 \
  --position-size 0.02
```

### Example 3: Conservative Strategy

```bash
python run_e2e_backtest.py \
  --edge-threshold 0.15 \
  --position-size 0.02 \
  --initial-capital 5000
```

### Example 4: Aggressive Strategy

```bash
python run_e2e_backtest.py \
  --edge-threshold 0.05 \
  --position-size 0.10 \
  --initial-capital 20000
```

### Example 5: Custom Time Horizons

```bash
# Test 1, 3, 7, 14 days before meetings
python run_e2e_backtest.py \
  --horizons "1,3,7,14"
```

## Output Files

The script generates:

```
results/backtest_v3/
├── backtest_results.json      # Complete results with all data
├── predictions.csv            # All predictions made
├── trades.csv                # All trades executed
├── horizon_metrics.csv       # Performance by time horizon
├── word_frequencies.pdf      # Visualization (vector)
└── word_frequencies.png      # Visualization (raster)
```

## Word Frequency Visualization

The script automatically generates a line chart showing:

- **X-axis**: FOMC meeting dates
- **Y-axis**: Mention count (Powell's speech only)
- **Lines**: Each traded contract
- **Style**: Professional publication-ready charts

### Example Chart Features

- Multiple contracts on the same plot
- Markers at each meeting date
- Grid for easy reading
- Legend with contract names
- Proper date formatting
- High DPI (300) for publication quality

## Workflow Details

### Step 1: Fetch Transcripts

Downloads PDF transcripts from Federal Reserve website for the specified year range.

**Output**: `data/raw_pdf/*.pdf`

### Step 2: Parse Transcripts

Parses PDFs into speaker-segmented JSONL files using deterministic mode (regex-based).

**Output**: `data/segments/*.jsonl`

### Step 3: Analyze Kalshi Contracts

Fetches resolved Kalshi contracts, extracts word lists, and builds contract mappings.

**Output**:
- `data/kalshi_analysis/contract_words.json`
- `data/kalshi_analysis/statistics.json`
- `data/kalshi_analysis/generated_contract_mapping.yaml`

### Step 4: Run Backtest

Executes Time-Horizon Backtest v3 with specified parameters.

**Output**: `results/backtest_v3/*`

### Step 5: Generate Visualizations

Creates word frequency charts for all traded contracts.

**Output**:
- `results/backtest_v3/word_frequencies.pdf`
- `results/backtest_v3/word_frequencies.png`

## Performance Metrics

The script displays:

### Overall Performance

```
Total Trades:     127
Win Rate:         58.3%
Total P&L:        $3,245.12
ROI:              32.5%
Sharpe Ratio:     1.42
Final Capital:    $13,245.12
```

### Per-Horizon Performance

```
7 days before meeting:
  Predictions:      150
  Accuracy:         62.7%
  Trades:           42
  Win Rate:         61.9%
  Total P&L:        $1,234.56
  ROI:              24.2%
  Brier Score:      0.185
```

### Word Frequency Statistics

```
Contract         Mean   Median  Std Dev  Min  Max  Total Mentions
-----------------------------------------------------------------
Inflation 40+    42.3   41.0    8.5      28   58   423
Unemployment     2.7    2.0     1.9      0    7    27
Volatility       1.4    1.0     1.2      0    4    14
```

## Troubleshooting

### Error: No module named 'fomc_analysis'

**Solution**: Install the package:
```bash
pip install -e .
```

### Error: Kalshi API credentials required

**Solution**: Create `.env` file with Kalshi credentials (see Prerequisites)

### Error: No resolved contracts found

**Solution**:
- Check that `--series-ticker` is correct (default: KXFEDMENTION)
- Verify Kalshi API credentials are valid
- Ensure there are resolved markets available

### Warning: No historical prices found

**Cause**: Kalshi may not have historical price data for older markets

**Impact**: Backtest can still measure prediction accuracy, but won't simulate trades

### Error: No trades executed

**Possible causes**:
1. **Edge threshold too high**: Lower `--edge-threshold`
2. **No price data**: Historical prices not available
3. **Not enough training data**: Need at least 5 historical meetings

**Solutions**:
```bash
# Try lower edge threshold
python run_e2e_backtest.py --edge-threshold 0.05

# Check if you have enough meetings
ls data/segments/*.jsonl | wc -l  # Should be > 5
```

## Advanced Usage

### Testing Visualization Only

If you just want to test the visualization with mock data:

```bash
python test_visualization.py
```

This creates sample data and generates a test chart without needing API credentials.

### Analyzing Results Programmatically

```python
import json
import pandas as pd
from pathlib import Path

# Load results
results = json.loads(Path('results/backtest_v3/backtest_results.json').read_text())

# Analyze trades
trades_df = pd.DataFrame(results['trades'])

# Most profitable contracts
contract_pnl = trades_df.groupby('contract')['pnl'].agg(['sum', 'mean', 'count'])
print(contract_pnl.sort_values('sum', ascending=False))

# Best time horizon
horizon_perf = trades_df.groupby('days_before_meeting')['pnl'].agg(['sum', 'mean'])
print(horizon_perf)
```

### Comparing Strategies

Run multiple backtests with different parameters:

```bash
# Conservative
python run_e2e_backtest.py --skip-fetch --skip-parse --skip-analyze \
  --edge-threshold 0.15 --position-size 0.02 \
  --output results/conservative

# Balanced
python run_e2e_backtest.py --skip-fetch --skip-parse --skip-analyze \
  --edge-threshold 0.10 --position-size 0.05 \
  --output results/balanced

# Aggressive
python run_e2e_backtest.py --skip-fetch --skip-parse --skip-analyze \
  --edge-threshold 0.05 --position-size 0.10 \
  --output results/aggressive
```

Then compare:

```python
import json
from pathlib import Path

for strategy in ['conservative', 'balanced', 'aggressive']:
    result = json.loads(Path(f'results/{strategy}/backtest_results.json').read_text())
    metrics = result['overall_metrics']
    print(f"\n{strategy.upper()}:")
    print(f"  ROI: {metrics['roi']*100:.1f}%")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")
    print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
```

## Time and Resource Requirements

### Approximate Runtime

- **Fetch transcripts**: 2-5 minutes (depends on internet speed)
- **Parse transcripts**: 1-3 minutes (deterministic mode)
- **Analyze Kalshi**: 5-10 minutes (API calls + variant generation)
- **Run backtest**: 1-2 minutes
- **Generate visualizations**: < 1 minute

**Total (clean run)**: ~10-20 minutes

**With skip flags**: ~2-4 minutes

### Disk Space

- **Raw PDFs**: ~50 MB
- **Segments**: ~10 MB
- **Results**: ~5 MB

**Total**: ~65 MB

## Best Practices

1. **Start with a clean run**: Use `--clean` to ensure reproducibility
2. **Use skip flags for reruns**: Much faster when testing parameters
3. **Test visualization separately**: Use `test_visualization.py` first
4. **Check data quality**: Verify segments look correct before running backtest
5. **Compare strategies**: Run multiple configurations to find optimal parameters
6. **Archive results**: Save results with descriptive names for comparison

## Next Steps

After running the e2e backtest:

1. **Analyze the visualization**: Look for patterns in word frequencies
2. **Review horizon metrics**: Determine which time window works best
3. **Examine trades.csv**: Identify most/least profitable contracts
4. **Optimize parameters**: Test different edge thresholds and position sizes
5. **Consider live deployment**: If profitable, test on paper trading

## Support

For issues:
1. Check error messages carefully
2. Verify API credentials in `.env`
3. Ensure Python >= 3.11
4. Try `--clean` to reset data
5. Run `test_visualization.py` to isolate issues

## Example Output

```
================================================================================
FOMC ANALYSIS - END-TO-END BACKTEST WITH VISUALIZATION
================================================================================

Configuration:
  Years:            2020 - 2025
  Series:           KXFEDMENTION
  Edge Threshold:   10%
  Position Size:    5%
  Initial Capital:  $10,000.00
  Horizons:         7,14,30 days
  Clean Data:       True

================================================================================
STEP 1: Fetching FOMC Transcripts
================================================================================
[...]
✓ STEP 1: Fetching FOMC Transcripts completed successfully

================================================================================
STEP 2: Parsing Transcripts
================================================================================
[...]
✓ STEP 2: Parsing Transcripts completed successfully

================================================================================
STEP 3: Analyzing Kalshi Contracts
================================================================================
[...]
✓ STEP 3: Analyzing Kalshi Contracts completed successfully

================================================================================
STEP 4: Running Backtest v3
================================================================================
[...]
✓ STEP 4: Running Backtest v3 completed successfully

================================================================================
GENERATING VISUALIZATION
================================================================================

✓ Found 15 traded contracts
Loading segment data for frequency analysis...
Counting contract mentions...
✓ Chart saved to: results/backtest_v3/word_frequencies.pdf
✓ Chart saved to: results/backtest_v3/word_frequencies.png

================================================================================
WORD FREQUENCY STATISTICS
================================================================================

Contract         Mean   Median  Std Dev  Min  Max  Total Mentions
Inflation 40+    42.30  41.00   8.50     28   58   423
Unemployment     2.70   2.00    1.90     0    7    27
[...]

================================================================================
FINAL BACKTEST SUMMARY
================================================================================

Overall Performance:
  Total Trades:     127
  Win Rate:         58.3%
  Total P&L:        $3,245.12
  ROI:              32.5%
  Sharpe Ratio:     1.42
  Final Capital:    $13,245.12

[...]

================================================================================
✓ END-TO-END PIPELINE COMPLETED SUCCESSFULLY!
================================================================================

Results saved to: results/backtest_v3/
  - backtest_results.json
  - predictions.csv
  - trades.csv
  - horizon_metrics.csv
  - word_frequencies.pdf
  - word_frequencies.png
```
