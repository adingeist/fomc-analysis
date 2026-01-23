# Earnings Call Analysis Framework

## Overview

The Earnings Call Analysis Framework is a comprehensive toolkit for analyzing earnings call transcripts to predict stock price movements and identify profitable trading opportunities. It follows the same architecture as the FOMC analysis framework but is adapted for company earnings calls.

## Key Features

- **Multi-Source Transcript Fetching**: SEC EDGAR, Alpha Vantage, and web sources
- **Speaker Segmentation**: Automatic identification of CEO, CFO, executives, and analysts
- **Feature Extraction**: Sentiment analysis, keyword frequencies, speaker patterns
- **Price Outcome Analysis**: Track stock price movements after earnings calls
- **Backtesting Framework**: Walk-forward validation with realistic execution
- **Multiple Models**: Beta-Binomial (Bayesian) and Sentiment-Based (ML)
- **CLI Interface**: Easy-to-use command-line tools

## Architecture

```
earnings_analysis/
├── fetchers/           # Data acquisition
│   ├── transcript_fetcher.py    # Earnings call transcripts
│   ├── price_fetcher.py         # Stock price data
│   └── fundamentals_fetcher.py  # EPS, revenue, estimates
│
├── parsing/            # Text processing
│   ├── transcript_parser.py     # Clean and parse transcripts
│   └── speaker_segmenter.py     # Identify speakers (CEO/CFO/analysts)
│
├── features/           # Feature extraction
│   ├── featurizer.py            # Extract keyword features
│   └── keyword_extractor.py     # Sentiment scoring
│
├── models/             # Prediction models
│   ├── base.py                  # Base model interface
│   ├── beta_binomial.py         # Bayesian model
│   └── sentiment_model.py       # ML-based model
│
├── backtester.py       # Backtesting engine
├── cli/                # Command-line interface
└── config.py           # Configuration management
```

## Installation

The earnings framework is part of the fomc-analysis package:

```bash
# Install package with dependencies
uv pip install -e .

# Verify installation
earnings --help
```

## Quick Start

### 1. Fetch Earnings Data

```bash
# Fetch earnings dates and fundamentals
earnings fetch-transcripts --ticker COIN --num-quarters 8

# Fetch stock price outcomes
earnings fetch-prices \
  --ticker COIN \
  --metadata-file data/earnings/transcripts/COIN/metadata.json \
  --horizons "1,5,10"
```

### 2. Parse Transcripts

```bash
# Parse and segment transcripts by speaker
earnings parse \
  --ticker COIN \
  --input-dir data/earnings/transcripts/COIN \
  --output-dir data/earnings/segments/COIN
```

### 3. Extract Features

```bash
# Extract sentiment and keyword features
earnings featurize \
  --ticker COIN \
  --segments-dir data/earnings/segments/COIN \
  --keywords-config configs/earnings/default_keywords.yaml \
  --speaker-mode executives_only
```

### 4. Run Backtest

```bash
# Backtest trading strategy
earnings backtest \
  --ticker COIN \
  --features-file data/earnings/features/COIN_features.parquet \
  --outcomes-file data/earnings/outcomes/COIN_outcomes.csv \
  --model beta \
  --edge-threshold 0.1 \
  --initial-capital 10000
```

## Detailed Workflow

### Data Collection

#### Transcript Sources

1. **SEC EDGAR** (Free, official)
   - Source: 8-K filings (Item 2.02 or 7.01)
   - Pros: Official, free, comprehensive
   - Cons: Not all companies file transcripts, parsing required

2. **Alpha Vantage** (API)
   - Source: Earnings API
   - Pros: Easy to use, structured data
   - Cons: No full transcripts, limited to fundamentals

3. **Web Scraping** (Advanced)
   - Source: Seeking Alpha, Motley Fool
   - Pros: Comprehensive transcript coverage
   - Cons: Legal considerations, rate limiting

#### Price Data

Uses `yfinance` to fetch:
- Daily OHLC prices
- Intraday price movements
- Volume data

Calculates outcomes:
- `return_1d`: 1-day price return
- `return_5d`: 5-day price return
- `direction_1d`: Binary (1 if up, 0 if down)

### Speaker Segmentation

Automatically identifies and classifies speakers:

**Executive Roles**:
- `ceo`: Chief Executive Officer
- `cfo`: Chief Financial Officer
- `executive`: Other C-suite (CTO, COO, CMO, etc.)

**Other Roles**:
- `analyst`: Sell-side analysts asking questions
- `operator`: Conference call operator
- `other`: Unclassified speakers

**Segmentation Modes**:
1. **Deterministic** (default): Regex-based pattern matching
2. **AI-Enhanced**: Uses OpenAI to improve segmentation

Example segment:
```json
{
  "speaker": "Brian Armstrong",
  "role": "ceo",
  "text": "Thank you for joining us today. We had a strong quarter...",
  "confidence": 1.0,
  "company": null
}
```

### Feature Extraction

#### Keyword Categories

Default keywords (see `configs/earnings/default_keywords.yaml`):

- **Sentiment Positive**: strong, growth, momentum, exceeded, confident
- **Sentiment Negative**: headwind, challenge, pressure, decline, concern
- **Guidance**: raise, lower, expect, outlook, forecast
- **Competition**: compete, market share, pricing
- **Growth Metrics**: revenue, margin, profit, customer, user
- **Efficiency**: cost, efficiency, productivity
- **Innovation**: AI, technology, innovation, platform
- **Macro**: inflation, economy, recession, interest rate
- **Regulation**: regulatory, compliance, legal

#### Speaker Modes

- `ceo_only`: Extract features only from CEO remarks
- `cfo_only`: Extract features only from CFO remarks
- `executives_only`: CEO + CFO + other executives
- `full_transcript`: All speakers including analysts

#### Phrase Modes

- `strict_literal`: Match exact phrases only
- `variants`: Include plurals, possessives, compounds

#### Output Features

Per keyword category:
- `{category}_count`: Total mentions
- `{category}_mentioned`: Binary (1 if mentioned)

Per speaker role:
- `{role}_word_count`: Total words spoken
- `{role}_segments`: Number of turns

Derived features:
- `ceo_cfo_ratio`: CEO words / CFO words
- `qa_questions`: Number of analyst questions

### Prediction Models

#### 1. Beta-Binomial Model (Bayesian)

**Algorithm**:
```python
Prior: Beta(α, β)
Likelihood: Binomial(n, p)
Posterior: Beta(α + successes, β + failures)
Prediction: E[p] = α_post / (α_post + β_post)
```

**Parameters**:
- `alpha_prior`: Alpha parameter (default: 1.0 = uniform prior)
- `beta_prior`: Beta parameter (default: 1.0 = uniform prior)
- `half_life`: Exponential weighting for recency (optional)

**Use Cases**:
- Simple baseline model
- Works with small datasets
- Provides uncertainty estimates

#### 2. Sentiment-Based Model (ML)

**Algorithm**: Logistic Regression

**Features**:
- Sentiment scores (positive, negative)
- Keyword counts (guidance, competition, etc.)
- Speaker word counts (CEO, CFO)

**Use Cases**:
- More sophisticated predictions
- Requires more historical data
- Can capture feature interactions

### Backtesting

#### Walk-Forward Framework

Ensures no lookahead bias:

```
For each earnings call t:
  1. Train model on calls 1...(t-1)
  2. Make prediction for call t
  3. Observe actual outcome
  4. Update capital based on P&L
```

#### Trading Logic

```python
# Calculate edge
edge = abs(predicted_probability - 0.5)

# Trade if edge > threshold
if edge >= edge_threshold:
    if predicted_probability > 0.5:
        side = "LONG"  # Buy stock
    else:
        side = "SHORT"  # Short stock

    position_size = capital * position_size_pct

    # Simulate outcome
    if side == "LONG":
        pnl = position_size * actual_return
    else:
        pnl = position_size * (-actual_return)

    # Apply transaction costs
    pnl -= position_size * transaction_cost_rate

    capital += pnl
```

#### Metrics

- **Total Trades**: Number of trades executed
- **Win Rate**: Percentage of profitable trades
- **Total P&L**: Net profit/loss
- **ROI**: Return on initial capital
- **Sharpe Ratio**: Risk-adjusted return

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# OpenAI (for AI segmentation)
OPENAI_API_KEY=sk-...

# Alpha Vantage (for earnings data)
ALPHA_VANTAGE_API_KEY=...

# Database (optional)
DATABASE_URL=sqlite:///data/fomc_analysis.db
```

### Keywords Configuration

Edit `configs/earnings/default_keywords.yaml`:

```yaml
# Add custom keyword categories
my_category:
  keyword1:
    - keyword1
    - variant1
    - variant2
  keyword2:
    - keyword2
```

## Advanced Usage

### Custom Models

Implement the `EarningsModel` interface:

```python
from earnings_analysis.models import EarningsModel
import pandas as pd

class MyCustomModel(EarningsModel):
    def fit(self, features: pd.DataFrame, outcomes: pd.Series):
        # Train your model
        pass

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        # Return predictions with columns:
        # probability, lower_bound, upper_bound, uncertainty
        pass

    def save(self, filepath: str):
        pass

    def load(self, filepath: str):
        pass
```

### Multi-Ticker Analysis

```python
from earnings_analysis.backtester import EarningsBacktester

tickers = ["COIN", "GOOGL", "AAPL", "TSLA"]

for ticker in tickers:
    # Load data
    features = pd.read_parquet(f"data/earnings/features/{ticker}_features.parquet")
    outcomes = pd.read_csv(f"data/earnings/outcomes/{ticker}_outcomes.csv")

    # Run backtest
    backtester = EarningsBacktester(features, outcomes, ...)
    result = backtester.run()

    # Save results
    save_backtest_result(result, f"results/earnings/{ticker}")
```

### Sector Analysis

```python
# Compare performance across sectors
tech_tickers = ["GOOGL", "AAPL", "MSFT", "META"]
finance_tickers = ["JPM", "BAC", "GS", "C"]

# Aggregate results
tech_results = aggregate_backtest_results(tech_tickers)
finance_results = aggregate_backtest_results(finance_tickers)

# Compare metrics
compare_sector_performance(tech_results, finance_results)
```

## Data Pipelines

### End-to-End Pipeline

```python
from pathlib import Path
from earnings_analysis import *

def run_pipeline(ticker: str):
    """Complete earnings analysis pipeline."""

    # 1. Fetch transcripts
    metadata = fetch_earnings_transcripts(ticker, num_quarters=12)

    # 2. Fetch prices
    earnings_dates = [m.call_date for m in metadata]
    outcomes = fetch_price_outcomes(ticker, earnings_dates)

    # 3. Parse transcripts
    for m in metadata:
        if m.has_transcript:
            text = parse_transcript(m.file_path)
            segments = segment_earnings_transcript(text, ticker)
            save_segments(segments, f"data/segments/{ticker}/{m.call_date}.jsonl")

    # 4. Featurize
    features = featurize_earnings_calls(
        segments_dir=f"data/segments/{ticker}",
        ticker=ticker,
        keywords_config="configs/earnings/default_keywords.yaml"
    )

    # 5. Backtest
    backtester = EarningsBacktester(features, outcomes, BetaBinomialEarningsModel)
    result = backtester.run()

    # 6. Save results
    save_backtest_result(result, f"results/earnings/{ticker}")

    return result

# Run for multiple tickers
tickers = ["COIN", "GOOGL", "AAPL"]
results = {ticker: run_pipeline(ticker) for ticker in tickers}
```

## Comparison to FOMC Framework

| Aspect | FOMC | Earnings |
|--------|------|----------|
| **Frequency** | 8 meetings/year | 4 quarters/year per company |
| **Speakers** | Powell vs reporters | CEO/CFO vs analysts |
| **Outcome** | Kalshi contract (YES/NO) | Price movement (up/down, %) |
| **Scope** | Single entity (Fed) | Hundreds of companies |
| **Data Source** | Fed website (free) | Mixed (SEC, APIs, web) |
| **Models** | Beta-Binomial, EWMA | Beta-Binomial, Sentiment ML |
| **Trading** | Prediction markets | Stock/options positions |

## Best Practices

### Data Quality

1. **Validate Transcripts**: Check for completeness, missing sections
2. **Handle Missing Data**: Earnings dates without transcripts
3. **Price Adjustments**: Account for splits, dividends

### Feature Engineering

1. **Domain Knowledge**: Use industry-specific keywords
2. **Temporal Features**: Compare to previous quarters
3. **Cross-Sectional**: Compare to peer companies
4. **Sentiment Calibration**: Adjust for company-specific tone

### Model Selection

1. **Start Simple**: Beta-Binomial baseline
2. **Iterate**: Add features, try different models
3. **Validate**: Out-of-sample testing
4. **Monitor**: Track live performance

### Risk Management

1. **Position Sizing**: Don't over-allocate (e.g., max 5% per trade)
2. **Diversification**: Multiple tickers, sectors
3. **Transaction Costs**: Account for slippage, fees
4. **Liquidity**: Check volume, bid-ask spread

## Troubleshooting

### Common Issues

**No transcripts found**:
- Check ticker symbol is correct
- Try different data sources
- Some companies don't file transcripts

**Segmentation fails**:
- Transcript format not recognized
- Try AI-enhanced segmentation
- Manually inspect transcript structure

**Low prediction accuracy**:
- Need more historical data
- Try different features/keywords
- Adjust model parameters

**Poor backtest performance**:
- Check for overfitting
- Validate data quality
- Review trading logic

## Resources

- **SEC EDGAR**: https://www.sec.gov/edgar
- **Alpha Vantage**: https://www.alphavantage.co/
- **yfinance Docs**: https://pypi.org/project/yfinance/
- **FOMC Framework**: See `README.md`

## Contributing

To extend the framework:

1. **Add New Data Sources**: Implement in `fetchers/`
2. **Custom Models**: Extend `EarningsModel` base class
3. **New Features**: Add to `features/featurizer.py`
4. **Improved Parsing**: Enhance `parsing/speaker_segmenter.py`

## License

MIT License (same as fomc-analysis package)
