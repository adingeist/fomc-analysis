# Earnings Call Analysis Framework

## Overview

A comprehensive framework for analyzing earnings call transcripts to predict stock price movements and identify profitable trading opportunities. Built using the same proven architecture as the FOMC analysis toolkit.

## Quick Start

```bash
# 1. Install dependencies
uv pip install -e .

# 2. Fetch earnings data for a ticker
earnings fetch-transcripts --ticker COIN --num-quarters 8

# 3. Fetch price outcomes
earnings fetch-prices \
  --ticker COIN \
  --metadata-file data/earnings/transcripts/COIN/metadata.json

# 4. Parse transcripts (if available)
earnings parse \
  --ticker COIN \
  --input-dir data/earnings/transcripts/COIN

# 5. Extract features
earnings featurize \
  --ticker COIN \
  --segments-dir data/earnings/segments/COIN \
  --keywords-config configs/earnings/default_keywords.yaml

# 6. Run backtest
earnings backtest \
  --ticker COIN \
  --features-file data/earnings/features/COIN_features.parquet \
  --outcomes-file data/earnings/outcomes/COIN_outcomes.csv \
  --model beta \
  --initial-capital 10000
```

## Key Features

### Data Collection
- **Multi-source transcripts**: SEC EDGAR, Alpha Vantage, web scraping
- **Price data**: Stock prices via yfinance
- **Fundamentals**: EPS, revenue, estimates

### Analysis
- **Speaker segmentation**: Automatic CEO/CFO/analyst identification
- **Sentiment analysis**: Positive/negative keyword detection
- **Feature extraction**: 10+ keyword categories, speaker patterns
- **Comparison**: Historical baseline analysis

### Modeling
- **Beta-Binomial Model**: Bayesian approach with uncertainty quantification
- **Sentiment Model**: ML-based logistic regression
- **Custom models**: Easy to extend

### Backtesting
- **Walk-forward validation**: No lookahead bias
- **Realistic execution**: Transaction costs, slippage
- **Comprehensive metrics**: Win rate, Sharpe ratio, ROI

## Architecture

```
earnings_analysis/
├── fetchers/          # Data acquisition (transcripts, prices, fundamentals)
├── parsing/           # Transcript parsing and speaker segmentation
├── features/          # Feature extraction and sentiment analysis
├── models/            # Prediction models (Beta-Binomial, Sentiment)
├── backtester.py      # Walk-forward backtesting engine
└── cli/               # Command-line interface
```

## Example: Full Pipeline

```python
from earnings_analysis import *

# Fetch data
metadata = fetch_earnings_transcripts("COIN", num_quarters=8)
outcomes = fetch_price_outcomes("COIN", earnings_dates)

# Parse and segment
segments = segment_earnings_transcript(transcript_text, "COIN")

# Extract features
features = featurize_earnings_calls(
    segments_dir="data/segments/COIN",
    ticker="COIN",
    keywords_config="configs/earnings/default_keywords.yaml"
)

# Backtest
backtester = EarningsBacktester(
    features=features,
    outcomes=outcomes,
    model_class=BetaBinomialEarningsModel,
    edge_threshold=0.1,
)
result = backtester.run(initial_capital=10000)

# Results
print(f"ROI: {result.metrics['roi']:.2%}")
print(f"Win Rate: {result.metrics['win_rate']:.2%}")
print(f"Sharpe: {result.metrics['sharpe']:.2f}")
```

## Prediction Targets

The framework can predict various outcomes:

1. **Price Direction** (primary): Stock up/down 1 day after earnings
2. **Price Return**: % change 1, 5, 10 days after earnings
3. **Earnings Surprise**: Beat/miss vs estimates (future)
4. **Guidance Change**: Raised/lowered guidance (future)

## Keywords Configuration

Default categories (see `configs/earnings/default_keywords.yaml`):

- **Sentiment**: Positive (strong, growth, momentum) / Negative (headwind, challenge, pressure)
- **Guidance**: Raise, lower, expect, outlook
- **Competition**: Market share, pricing, competitors
- **Growth**: Revenue, margin, profit, customers
- **Efficiency**: Cost, productivity, automation
- **Innovation**: AI, technology, platform
- **Macro**: Inflation, economy, recession
- **Regulation**: Regulatory, compliance, legal

## Models

### Beta-Binomial (Bayesian)
- Simple, interpretable baseline
- Works with small datasets
- Provides uncertainty estimates
- Exponential weighting for recency

### Sentiment-Based (ML)
- Logistic regression on features
- Uses sentiment scores and keyword counts
- Requires more historical data
- Can capture feature interactions

## Data Sources

### Free
- **SEC EDGAR**: Official 8-K filings (transcripts in exhibits)
- **yfinance**: Stock prices and basic fundamentals
- **Alpha Vantage**: Earnings dates and estimates (API key required)

### Paid (Future)
- **Seeking Alpha API**: Best transcript coverage
- **Polygon.io**: Real-time prices and options data
- **Finnhub**: Earnings calendar and estimates

## Configuration

Create `.env` file:

```bash
# OpenAI (for AI-enhanced segmentation)
OPENAI_API_KEY=sk-...

# Alpha Vantage (for earnings data)
ALPHA_VANTAGE_API_KEY=...
```

## Comparison to FOMC Framework

| Feature | FOMC | Earnings |
|---------|------|----------|
| **Frequency** | 8/year | 4/year per company |
| **Speakers** | Powell, reporters | CEO, CFO, analysts |
| **Outcome** | Kalshi YES/NO | Price up/down |
| **Scope** | Single entity | Hundreds of tickers |
| **Contracts** | Prediction markets | Stock positions |

## Documentation

See `docs/EARNINGS_FRAMEWORK.md` for comprehensive documentation covering:
- Detailed architecture
- Advanced usage patterns
- Custom model development
- Multi-ticker analysis
- Troubleshooting guide

## Examples

- `examples/earnings_example.py`: Complete workflow demonstration
- CLI commands: Run `earnings --help` for all commands

## Roadmap

- [ ] Options strategy backtester (straddles, calls, puts)
- [ ] Intraday price prediction (same-day trading)
- [ ] Guidance prediction model
- [ ] Multi-ticker portfolio analysis
- [ ] Live prediction dashboard integration
- [ ] Advanced web scraping for more data sources

## Contributing

The framework is designed to be extensible:
- Add new data sources in `fetchers/`
- Create custom models by extending `EarningsModel`
- Add feature extractors in `features/`
- Enhance parsing in `parsing/`

## License

MIT License (same as fomc-analysis)
