# Earnings Call + Kalshi Framework

## Overview

This framework analyzes earnings call transcripts to **predict Kalshi mention contract outcomes**, not stock prices. It follows the same proven approach as the FOMC analysis toolkit but adapted for company earnings calls.

### What This Framework Does

✅ **Analyzes earnings call transcripts** for word/phrase frequencies
✅ **Fetches Kalshi earnings mention contracts** (e.g., "Will META CEO say 'AI' 10+ times?")
✅ **Predicts contract outcomes** using historical transcript data
✅ **Trades Kalshi YES/NO contracts** based on edge over market price
✅ **Backtests strategies** using actual Kalshi resolved contract outcomes

### What This Framework Does NOT Do

❌ Trade stocks based on sentiment
❌ Predict stock price movements
❌ Analyze earnings surprises (EPS beats/misses)

## Kalshi Earnings Mention Contracts

Kalshi offers earnings mention contracts similar to their FOMC contracts. These are event contracts that resolve based on whether specific words are mentioned during earnings calls.

**Example Contracts:**
- "Will META CEO Mark Zuckerberg say 'AI' at least 10 times during the Q4 2024 earnings call?" → YES/NO
- "Will TSLA CEO Elon Musk mention 'Full Self-Driving' during Q1 2025 earnings?" → YES/NO
- "Will COIN executives say 'crypto' at least 20 times?" → YES/NO

**Series Format:** `KXEARNINGSMENTION[TICKER]`
- `KXEARNINGSMENTIONMETA` - Meta (Facebook) earnings mentions
- `KXEARNINGSMENTIONTSLA` - Tesla earnings mentions
- `KXEARNINGSMENTIONNVDA` - NVIDIA earnings mentions
- `KXEARNINGSMENTIONAMZN` - Amazon earnings mentions
- `KXEARNINGSMENTIONAAPL` - Apple earnings mentions

**Reference:**
- [Kalshi Earnings Mention Contracts](https://kalshi.com/markets/kxearningsmentionmeta/earnings-mention)

## Architecture

```
earnings_analysis/
├── fetchers/              # Transcript and data fetching
│   ├── transcript_fetcher.py    # SEC EDGAR, Alpha Vantage
│   └── fundamentals_fetcher.py  # Earnings dates
│
├── parsing/               # Transcript processing
│   ├── transcript_parser.py     # Clean raw text
│   └── speaker_segmenter.py     # CEO/CFO/analyst classification
│
├── features/              # Feature extraction
│   ├── featurizer.py            # Word frequency features
│   └── keyword_extractor.py     # Sentiment scoring
│
├── models/                # Prediction models
│   ├── base.py                  # Base interface
│   └── beta_binomial.py         # Bayesian model (adapted from FOMC)
│
├── kalshi/                # Kalshi integration (NEW)
│   ├── contract_analyzer.py     # Fetch/analyze Kalshi contracts
│   └── backtester.py            # Backtest with Kalshi outcomes
│
└── cli/                   # Command-line interface
    └── earnings_cli.py          # CLI commands
```

## Workflow: FOMC → Earnings Adaptation

The framework adapts the FOMC approach for earnings calls:

| **FOMC Framework** | **Earnings Framework** |
|-------------------|----------------------|
| Fed press conferences | Earnings call transcripts |
| Powell's remarks | CEO/CFO remarks |
| KXFEDMENTION series | KXEARNINGSMENTION[TICKER] series |
| Single entity (Fed) | Multiple companies (META, TSLA, etc.) |
| 8 meetings/year | 4 calls/year per company |
| "inflation" mentions | Company-specific keywords ("AI", "crypto", etc.) |

## Quick Start

### 1. Fetch Earnings Data

```bash
# Get earnings dates using yfinance
earnings fetch-transcripts --ticker META --num-quarters 8
```

### 2. Fetch/Parse Transcripts

```bash
# Parse earnings call transcripts
earnings parse \
  --ticker META \
  --input-dir data/earnings/transcripts/META \
  --output-dir data/earnings/segments/META
```

### 3. Analyze Kalshi Contracts

```bash
# Fetch Kalshi contracts and analyze historical transcripts
earnings analyze-kalshi-contracts \
  --ticker META \
  --segments-dir data/earnings/segments/META \
  --output-dir data/earnings/kalshi_analysis/META \
  --market-status all
```

This will:
- Fetch all Kalshi contracts for META (series: `KXEARNINGSMENTIONMETA`)
- Extract tracked words and thresholds from contract titles
- Analyze historical transcripts for those word frequencies
- Generate statistical summaries

### 4. Extract Features

```bash
# Featurize transcripts based on Kalshi contract words
earnings featurize \
  --ticker META \
  --segments-dir data/earnings/segments/META \
  --keywords-config data/earnings/kalshi_analysis/META/contract_mapping.yaml \
  --speaker-mode executives_only
```

### 5. Backtest Kalshi Contracts

```bash
# Backtest trading strategy on Kalshi contracts
earnings backtest-kalshi \
  --ticker META \
  --features-file data/earnings/features/META_features.parquet \
  --contract-words-file data/earnings/kalshi_analysis/META/contract_words.json \
  --model beta \
  --edge-threshold 0.12 \
  --initial-capital 10000
```

## Key Components

### Kalshi Contract Analyzer

Fetches and analyzes Kalshi earnings mention contracts:

```python
from fomc_analysis.kalshi_client_factory import get_kalshi_client
from earnings_analysis.kalshi import EarningsContractAnalyzer

client = get_kalshi_client()
analyzer = EarningsContractAnalyzer(client, ticker="META")

# Fetch contracts
contracts = await analyzer.fetch_contracts(market_status="all")

# Analyze transcripts
analyses = analyzer.analyze_transcripts(
    contracts,
    segments_dir="data/earnings/segments/META",
    speaker_mode="executives_only"
)
```

### Kalshi Backtester

Backtests trading strategies using actual Kalshi outcomes:

```python
from earnings_analysis.kalshi import EarningsKalshiBacktester
from earnings_analysis.models import BetaBinomialEarningsModel

backtester = EarningsKalshiBacktester(
    features=features_df,        # Word frequency features
    outcomes=outcomes_df,        # Actual Kalshi contract outcomes (0 or 1)
    model_class=BetaBinomialEarningsModel,
    edge_threshold=0.12,         # Same as FOMC
    position_size_pct=0.03,      # 3% per trade
    fee_rate=0.07,               # Kalshi's 7% fee on profits
)

result = backtester.run(ticker="META", initial_capital=10000)

# Results
print(f"ROI: {result.metrics.roi:.2%}")
print(f"Win Rate: {result.metrics.win_rate:.2%}")
print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
```

## Data Requirements

### Transcripts
- **Source**: SEC EDGAR 8-K filings, Alpha Vantage, or web scraping
- **Format**: JSONL with speaker segments
- **Speaker Roles**: CEO, CFO, executive, analyst, operator

### Kalshi Contracts
- **Source**: Kalshi API (requires credentials)
- **Format**: JSON with market metadata
- **Key Fields**:
  - `ticker`: Kalshi market ticker
  - `title`: Contract description (contains word + threshold)
  - `status`: open, closed, resolved, settled
  - `result`: "yes" or "no" (for resolved contracts)

### Features
- **Word Counts**: Number of times each contract word is mentioned
- **Thresholds**: Binary features (1 if count >= threshold, 0 otherwise)
- **Speaker-Specific**: CEO vs CFO word counts
- **Historical**: Rolling averages, deltas from previous quarters

### Outcomes
- **Source**: Resolved Kalshi contracts
- **Format**: Binary (1 = YES/mentioned, 0 = NO/not mentioned)
- **Resolution**: Based on official transcript (Kalshi verifies)

## Prediction Models

### Beta-Binomial Model (Recommended)

Adapted from FOMC framework:

```python
from earnings_analysis.models import BetaBinomialEarningsModel

model = BetaBinomialEarningsModel(
    alpha_prior=1.0,      # Uniform prior
    beta_prior=1.0,
    half_life=4,          # Weight recent calls more heavily
)

model.fit(historical_features, historical_outcomes)
predictions = model.predict(next_call_features)
```

**Advantages:**
- Works with small datasets (few historical earnings calls)
- Provides uncertainty estimates
- Proven on FOMC contracts
- Natural for binary outcomes (YES/NO)

## Backtesting

### Walk-Forward Validation

Ensures no lookahead bias (same as FOMC):

```
For each earnings call t:
  1. Train model on calls 1, 2, ..., t-1
  2. Make prediction for call t
  3. Trade if edge > threshold
  4. Observe actual Kalshi outcome
  5. Calculate P&L
  6. Update capital
```

### Trading Logic

```python
# Calculate edge
edge = predicted_probability - kalshi_market_price

# Trade YES if strong upside edge
if edge > yes_edge_threshold and predicted_prob > min_yes_probability:
    trade_side = "YES"
    entry_price = market_price + slippage

# Trade NO if strong downside edge
elif edge < -no_edge_threshold and predicted_prob < max_no_probability:
    trade_side = "NO"
    entry_price = (1 - market_price) + slippage

# Position sizing
position_size = capital * position_size_pct

# P&L calculation (same as FOMC)
if side == "YES":
    if outcome == 1:  # YES wins
        gross_pnl = position_size * (1 - entry_price) / entry_price
        fees = gross_pnl * 0.07  # 7% Kalshi fee
        net_pnl = gross_pnl - fees
    else:  # NO wins
        net_pnl = -position_size
```

### Metrics

- **Accuracy**: Prediction correctness (50% threshold)
- **Win Rate**: Percentage of profitable trades
- **ROI**: Return on initial capital
- **Sharpe Ratio**: Risk-adjusted return
- **Brier Score**: Calibration metric

## Example: META Earnings Analysis

### Scenario

You want to trade Kalshi contracts on META's Q4 2024 earnings call.

**Available Contracts:**
- "Will Mark Zuckerberg say 'AI' at least 10 times?" (Market price: 65%)
- "Will executives mention 'metaverse' at least 3 times?" (Market price: 40%)

### Steps

1. **Fetch Historical Transcripts**
   ```bash
   earnings fetch-transcripts --ticker META --num-quarters 12
   ```

2. **Parse Transcripts**
   ```bash
   earnings parse --ticker META --input-dir data/transcripts/META
   ```

3. **Analyze Kalshi Contracts**
   ```bash
   earnings analyze-kalshi-contracts \
     --ticker META \
     --segments-dir data/segments/META
   ```

   Output:
   ```
   Word: ai
   Historical calls: 12
   Calls with mention (>=10 times): 9 (75%)
   Avg mentions: 14.3
   Max mentions: 22
   ```

4. **Make Prediction**

   Model predicts: 80% probability of "AI" being mentioned 10+ times

5. **Trade Decision**

   ```
   Predicted: 80%
   Market: 65%
   Edge: +15%

   Action: BUY YES at 65¢
   Position: $300 (3% of $10k capital)
   ```

6. **Outcome**

   Actual result: Mark Zuckerberg says "AI" 18 times → **YES wins**

   ```
   Gross P&L: $300 * (1 - 0.65) / 0.65 = $161.54
   Kalshi Fee (7%): $11.31
   Net P&L: $150.23
   ROI: 50.1%
   ```

## Comparison to Stock Trading Approach

The original implementation (now deprecated) tried to predict stock prices. Here's why Kalshi contracts are better:

| **Aspect** | **Stock Trading** | **Kalshi Contracts** |
|-----------|------------------|---------------------|
| **Prediction Target** | Price movement (continuous) | Word mention (binary) |
| **Signal Quality** | Weak (many confounding factors) | Strong (direct transcript analysis) |
| **Market Efficiency** | Very efficient | Less efficient (newer market) |
| **Transaction Costs** | Moderate | Higher (7% fee on wins) |
| **Leverage** | Margin required | Built into contracts |
| **Backtestability** | Complex (slippage, gaps) | Simple (binary outcomes) |
| **Data Availability** | Easy (yfinance) | Requires Kalshi API |

**Verdict:** Kalshi contracts are a better fit because:
1. Transcript analysis directly predicts the outcome
2. Less market efficiency → more alpha potential
3. Binary outcomes easier to model
4. Proven approach (see FOMC framework results)

## Integration with FOMC Framework

This framework **reuses** FOMC components:

### Shared Components
- ✅ `fomc_analysis.kalshi_client_factory` - Kalshi API client
- ✅ `fomc_analysis.models.BetaBinomialModel` - Adapted for earnings
- ✅ `fomc_analysis.backtester_v3` - Walk-forward logic (pattern reused)
- ✅ `fomc_analysis.variants.generator` - Keyword variant generation

### Earnings-Specific Components
- 🆕 `earnings_analysis.kalshi.contract_analyzer` - Fetch earnings contracts
- 🆕 `earnings_analysis.kalshi.backtester` - Earnings-specific backtester
- 🆕 `earnings_analysis.parsing.speaker_segmenter` - CEO/CFO/analyst classification
- 🆕 CLI commands for earnings workflow

## Next Steps

1. **Test with Real Data**: Fetch actual Kalshi earnings contracts and historical outcomes
2. **Backtest Historical Calls**: Run walk-forward validation on 2023-2024 earnings seasons
3. **Parameter Optimization**: Grid search for optimal edge thresholds
4. **Multi-Ticker Analysis**: Compare performance across different companies
5. **Live Trading**: Deploy predictions for upcoming earnings calls

## Known Limitations

1. **Kalshi API Access**: Requires valid Kalshi credentials
2. **Transcript Availability**: Not all companies file transcripts with SEC
3. **Contract Coverage**: Kalshi doesn't cover all tickers
4. **Historical Data**: Limited history (Kalshi launched 2021)
5. **Market Liquidity**: Some contracts have low volume

## Resources

- **Kalshi Earnings Contracts**: https://kalshi.com/markets/kxearningsmentionmeta/earnings-mention
- **FOMC Framework**: See main README.md
- **Kalshi API Docs**: https://kalshi.com/docs
- **SEC EDGAR**: https://www.sec.gov/edgar

## Support

This framework follows the same patterns as the FOMC toolkit. For questions:
1. Review FOMC framework documentation
2. Check Kalshi contract analyzer implementation
3. See `examples/earnings_example.py` for usage patterns

---

**Last Updated**: 2026-01-24
**Framework Version**: 0.1.0 (Kalshi-focused)
