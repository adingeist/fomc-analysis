# Earnings Kalshi Framework - Quick Start Guide

This guide walks you through running your first earnings call Kalshi backtest, from setup to results.

---

## Prerequisites

Before you start, you need:

1. **Kalshi API Credentials**
   - Sign up at [kalshi.com](https://kalshi.com)
   - Generate API credentials (API Key ID + Private Key)
   - Save credentials securely

2. **Earnings Call Transcripts**
   - Historical earnings call transcripts for your target ticker(s)
   - Can be from SEC EDGAR, Alpha Vantage, or company investor relations

3. **Python Environment**
   - Python 3.11+
   - Virtual environment recommended

---

## Step 1: Installation

### Clone and Install

```bash
# Clone repository
git clone <repo-url>
cd fomc-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .
```

### Verify Installation

```bash
python -c "from earnings_analysis.kalshi import EarningsContractAnalyzer; print('✓ Install successful!')"
```

---

## Step 2: Configure API Credentials

### Create `.env` File

```bash
cp .env.example .env
```

### Add Your Credentials

Edit `.env`:

```bash
# Your Kalshi API credentials
KALSHI_API_KEY_ID=your_actual_key_id_here
KALSHI_PRIVATE_KEY_BASE64=your_actual_base64_private_key_here
```

**Get Base64 Private Key:**
```bash
# If you have private_key.pem
cat private_key.pem | base64 -w 0
```

### Test API Connection

```bash
python scripts/explore_kalshi_earnings_contracts.py
```

**Expected Output:**
```
============================================================
Exploring Kalshi Earnings Mention Contracts
============================================================

[1] Searching for earnings-related series...
  ✓ Found series: KXEARNINGSMENTIONMETA
  ✓ Found series: KXEARNINGSMENTIONTSLA
  ...
```

---

## Step 3: Explore Available Contracts

### See What Tickers Are Available

Run the exploration script to discover:
- Which companies have Kalshi earnings mention contracts
- What words each contract tracks
- Contract thresholds and status

```bash
python scripts/explore_kalshi_earnings_contracts.py
```

**Sample Output:**
```
Contract: KXEARNINGSMENTIONMETA-24Q4-AI-03
  Title: Will META CEO say 'AI' at least 3 times?
  Status: open
  Word: ai
  Threshold: 3

Contract: KXEARNINGSMENTIONTSLA-24Q4-DELIVERY-05
  Title: Will TSLA CEO say 'delivery' at least 5 times?
  Status: open
  Word: delivery
  Threshold: 5
```

### Pick Your Ticker

Choose a ticker with good contract coverage:
- **META** - Usually has AI, metaverse, revenue, etc.
- **TSLA** - Usually has delivery, production, FSD, etc.
- **NVDA** - Usually has AI, datacenter, gaming, etc.

For this guide, we'll use **META**.

---

## Step 4: Prepare Transcript Data

### Option A: Use Mock Data (Testing)

```bash
# Run the verification script to generate mock data
python examples/verify_earnings_framework.py
```

This creates synthetic transcripts in `data/verification/segments/`.

### Option B: Real Transcripts (Production)

1. **Download Transcripts**
   - Get transcripts from your chosen source
   - Should include speaker information (CEO, CFO, etc.)

2. **Convert to JSONL Format**

   Each transcript should be a JSONL file: `{TICKER}_{DATE}.jsonl`

   Example: `META_2024-01-31.jsonl`
   ```jsonl
   {"speaker": "Mark Zuckerberg", "role": "ceo", "text": "...", "segment_idx": 0}
   {"speaker": "Susan Li", "role": "cfo", "text": "...", "segment_idx": 1}
   {"speaker": "Analyst", "role": "analyst", "text": "...", "segment_idx": 2}
   ```

3. **Save to Directory**
   ```
   data/earnings/segments/
   ├── META_2023-01-31.jsonl
   ├── META_2023-04-30.jsonl
   ├── META_2023-07-31.jsonl
   └── ...
   ```

**Helper:** Use `src/earnings_analysis/parsing/speaker_segmenter.py` to auto-segment raw transcripts.

---

## Step 5: Run Your First Backtest

### Quick Test (Mock Data)

```bash
python examples/verify_earnings_framework.py
```

### Real Backtest (Python Script)

Create `run_meta_backtest.py`:

```python
import asyncio
from pathlib import Path
import pandas as pd

from earnings_analysis.kalshi import EarningsContractAnalyzer
from earnings_analysis.kalshi.backtester import (
    EarningsKalshiBacktester,
    save_earnings_backtest_result,
)
from earnings_analysis.models import BetaBinomialEarningsModel
from fomc_analysis.kalshi_client_factory import get_kalshi_client


async def main():
    # Configuration
    TICKER = "META"
    SEGMENTS_DIR = Path("data/earnings/segments")
    OUTPUT_DIR = Path("data/earnings/backtests/meta")

    # Step 1: Fetch Kalshi contracts
    print("Fetching Kalshi contracts...")
    client = get_kalshi_client()
    analyzer = EarningsContractAnalyzer(client, TICKER)

    contracts = await analyzer.fetch_contracts(market_status="resolved")
    print(f"Found {len(contracts)} contracts")

    for contract in contracts:
        print(f"  - {contract.word}: {contract.threshold}+ mentions")

    # Step 2: Analyze historical transcripts
    print("\nAnalyzing transcripts...")
    analyses = analyzer.analyze_transcripts(
        contracts,
        segments_dir=SEGMENTS_DIR,
        speaker_mode="executives_only",
    )

    # Step 3: Build features and outcomes
    print("\nBuilding features...")
    # Load transcript files
    segment_files = sorted(SEGMENTS_DIR.glob(f"{TICKER}_*.jsonl"))
    call_dates = [f.stem.split('_')[1] for f in segment_files]

    # Create features (word counts) and outcomes (1 if >= threshold)
    features_data = []
    outcomes_data = []

    import json
    import re

    for segment_file in segment_files:
        # Load segments
        segments = []
        with open(segment_file, 'r') as f:
            for line in f:
                segments.append(json.loads(line))

        # Filter to executives only
        exec_segments = [
            seg for seg in segments
            if seg.get('role') in ('ceo', 'cfo', 'executive')
        ]

        combined_text = " ".join(seg['text'] for seg in exec_segments)

        # Count each word
        feature_row = {}
        outcome_row = {}

        for contract in contracts:
            word = contract.word
            threshold = contract.threshold

            pattern = r'\b' + re.escape(word) + r'\b'
            count = len(re.findall(pattern, combined_text, re.IGNORECASE))

            feature_row[word] = count
            outcome_row[word] = 1 if count >= threshold else 0

        features_data.append(feature_row)
        outcomes_data.append(outcome_row)

    features_df = pd.DataFrame(features_data, index=call_dates)
    outcomes_df = pd.DataFrame(outcomes_data, index=call_dates)

    print(f"Features shape: {features_df.shape}")
    print(f"Outcomes shape: {outcomes_df.shape}")

    # Step 4: Run backtest
    print("\nRunning backtest...")
    backtester = EarningsKalshiBacktester(
        features=features_df,
        outcomes=outcomes_df,
        model_class=BetaBinomialEarningsModel,
        model_params={
            'alpha_prior': 1.0,
            'beta_prior': 1.0,
            'half_life': 8.0,  # Weight last 8 calls more heavily
        },
        edge_threshold=0.12,
        position_size_pct=0.03,
        fee_rate=0.07,
        min_train_window=4,
    )

    result = backtester.run(
        ticker=TICKER,
        initial_capital=10000.0,
        market_prices=None,  # TODO: Add real market prices
    )

    # Step 5: Print results
    print("\n" + "="*60)
    print(f"BACKTEST RESULTS - {TICKER}")
    print("="*60)
    print(f"\nPredictions:")
    print(f"  Total: {result.metrics.total_predictions}")
    print(f"  Accuracy: {result.metrics.accuracy:.1%}")
    print(f"  Brier Score: {result.metrics.brier_score:.3f}")

    print(f"\nTrading:")
    print(f"  Trades: {result.metrics.total_trades}")
    print(f"  Win Rate: {result.metrics.win_rate:.1%}")
    print(f"  Total P&L: ${result.metrics.total_pnl:,.2f}")
    print(f"  ROI: {result.metrics.roi:.1%}")
    print(f"  Sharpe: {result.metrics.sharpe_ratio:.2f}")

    # Step 6: Save results
    save_earnings_backtest_result(result, OUTPUT_DIR)
    print(f"\n✓ Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Run It

```bash
python run_meta_backtest.py
```

---

## Step 6: Interpret Results

### Output Files

After running, you'll find:

```
data/earnings/backtests/meta/
├── backtest_results.json    # Full results
├── predictions.csv          # All predictions
└── trades.csv              # All trades
```

### Key Metrics

**Prediction Quality:**
- **Accuracy** - % of correct predictions (baseline: 50%)
- **Brier Score** - Calibration quality (lower is better, 0.25 = random)

**Trading Performance:**
- **Win Rate** - % of profitable trades
- **Total P&L** - Profit/loss in dollars
- **ROI** - Return on investment (%)
- **Sharpe Ratio** - Risk-adjusted returns (>1 is good)

### Example Good Results

```
Predictions:
  Total: 48
  Accuracy: 68.8%  ✓ Better than random
  Brier Score: 0.185  ✓ Well-calibrated

Trading:
  Trades: 24
  Win Rate: 58.3%  ✓ Edge over market
  Total P&L: $1,247.50  ✓ Profitable
  ROI: 12.5%  ✓ Good return
  Sharpe: 1.45  ✓ Solid risk-adjusted return
```

### Example Bad Results

```
Predictions:
  Total: 48
  Accuracy: 52.1%  ⚠️ Barely better than random
  Brier Score: 0.248  ⚠️ Poor calibration

Trading:
  Trades: 18
  Win Rate: 44.4%  ❌ Losing more than winning
  Total P&L: -$327.18  ❌ Lost money
  ROI: -3.3%  ❌ Negative return
  Sharpe: -0.42  ❌ Bad risk-adjusted return
```

---

## Step 7: Improve Results

### If Results Are Poor

1. **Check Data Quality**
   - Are transcripts complete?
   - Is speaker segmentation accurate?
   - Are word counts correct?

2. **Tune Model Parameters**
   ```python
   # Try different half-life values
   'half_life': 6.0   # Weight last 6 calls
   'half_life': 12.0  # Weight last 12 calls

   # Try informative priors
   'alpha_prior': 2.0  # Expect ~67% YES rate
   'beta_prior': 1.0
   ```

3. **Adjust Trading Thresholds**
   ```python
   edge_threshold=0.15,  # Higher = fewer but better trades
   position_size_pct=0.02,  # Smaller = less risk
   ```

4. **Add Market Prices**
   - Collect real historical Kalshi prices
   - Compare model edge vs actual market efficiency

### If Results Look Good

1. **Validate on New Data**
   - Run on different ticker
   - Test on recent quarters not in training

2. **Paper Trade**
   - Make predictions for upcoming earnings calls
   - Track accuracy before risking real money

3. **Build Production Pipeline**
   - Automate transcript fetching
   - Auto-generate predictions
   - Set up alerts for high-edge opportunities

---

## Common Issues

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'earnings_analysis'`

**Solution:**
```bash
# Make sure you installed in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/fomc-analysis/src"
```

### Kalshi API Errors

**Problem:** `Error fetching markets: 401 Unauthorized`

**Solution:**
- Check `.env` file exists and has correct credentials
- Verify API key is active on Kalshi dashboard
- Test with `explore_kalshi_earnings_contracts.py`

### No Contracts Found

**Problem:** `Found 0 contract words`

**Solution:**
- Verify ticker has Kalshi contracts (try META, TSLA, NVDA first)
- Check `market_status` parameter (try "all" instead of "open")
- Confirm series ticker format: `KXEARNINGSMENTION{TICKER}`

### No Transcripts Found

**Problem:** `No transcript segments found for META`

**Solution:**
- Check `SEGMENTS_DIR` path is correct
- Verify files are named: `{TICKER}_{DATE}.jsonl`
- Files must be in JSONL format (one JSON object per line)

---

## Next Steps

### Beginner
1. Run verification script with mock data
2. Test Kalshi API connection
3. Get 1-2 historical transcripts for one ticker
4. Run first backtest

### Intermediate
1. Collect full historical data (2+ years)
2. Build market price database
3. Compare multiple tickers
4. Tune model parameters

### Advanced
1. Build automated pipeline
2. Implement live predictions
3. Add position sizing optimization
4. Build monitoring dashboard

---

## Support

- **Documentation:** `docs/` directory
- **Examples:** `examples/` directory
- **Issues:** GitHub issues
- **FOMC Framework:** Similar patterns, check `README.md`

---

## Appendix: File Formats

### Transcript JSONL Format

```jsonl
{"speaker": "Mark Zuckerberg", "role": "ceo", "text": "Thanks for joining us today. I'm excited to share our Q4 results.", "segment_idx": 0}
{"speaker": "Susan Li", "role": "cfo", "text": "Revenue this quarter was $40.1 billion, up 25% year over year.", "segment_idx": 1}
{"speaker": "Analyst", "role": "analyst", "text": "Can you provide more detail on AI investments?", "segment_idx": 2}
```

### Features CSV Format

```csv
,ai,cloud,revenue,margin
2023-01-31,12,5,8,3
2023-04-30,15,7,6,4
2023-07-31,18,9,7,5
```

### Outcomes CSV Format

```csv
,ai,cloud,revenue,margin
2023-01-31,1,1,1,0
2023-04-30,1,1,1,1
2023-07-31,1,1,1,1
```

Values: `1` = mentioned ≥ threshold, `0` = mentioned < threshold
