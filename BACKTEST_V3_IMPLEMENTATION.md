# Time-Horizon Backtest v3 - Implementation Summary

## Overview

This document summarizes the implementation of the improved backtest system (v3) for the FOMC analysis toolkit. The new system addresses critical limitations of the previous backtester and provides a more realistic evaluation framework for the quant model.

## Problem Statement

The original backtest (v2) had several limitations:

1. **Used transcript features only**: Relied on whether words were mentioned in transcripts, not actual Kalshi contract outcomes
2. **Single prediction timing**: Made one prediction per meeting without testing different time horizons
3. **Limited accuracy tracking**: No breakdown of performance by prediction timing
4. **Incomplete market integration**: Didn't properly fetch and use actual contract resolution data

## Solution: Time-Horizon Backtest v3

### Key Improvements

#### 1. Actual Kalshi Contract Outcomes

**Implementation**: `fetch_kalshi_contract_outcomes()` in `backtester_v3.py`

- Fetches real contract resolution data from Kalshi API
- Returns binary outcomes: 1 (YES/100%) or 0 (NO/0%)
- Only processes resolved markets for accurate backtesting
- Handles multiple contract formats (with/without thresholds)

```python
def fetch_kalshi_contract_outcomes(
    contract_words: List[Dict],
    kalshi_client,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    - meeting_date: When the FOMC meeting occurred
    - contract: Contract name
    - ticker: Kalshi market ticker
    - outcome: 1 if YES, 0 if NO
    - close_price: Final market price
    """
```

#### 2. Multi-Horizon Predictions

**Implementation**: `TimeHorizonBacktester` class

The backtester makes predictions at configurable time horizons (default: 7, 14, 30 days) before each meeting:

- **7 days before**: Tests model with most recent data
- **14 days before**: Mid-range prediction window
- **30 days before**: Early prediction, potentially larger market edges

For each meeting:
1. Train model on all previous meetings
2. Make predictions at each horizon
3. Fetch market prices at those specific dates
4. Compare predictions with actual outcomes
5. Execute simulated trades if edge > threshold

#### 3. Historical Price Fetching

**Implementation**: `fetch_historical_prices_at_horizons()`

- Fetches Kalshi market prices at specific dates before meetings
- Handles missing data gracefully
- Normalizes prices to 0-1 range
- Finds closest available price if exact date unavailable

#### 4. Comprehensive Metrics

**Per-Horizon Metrics** (`HorizonMetrics` dataclass):
- Total predictions made
- Correct predictions (accuracy)
- Total trades executed
- Win rate
- Total P&L
- Average P&L per trade
- ROI (return on investment)
- Sharpe ratio
- Brier score (calibration metric)

**Overall Metrics**:
- Aggregate performance across all horizons
- Final capital and ROI
- Overall win rate and Sharpe ratio

#### 5. Realistic Trading Simulation

**Trading Logic**:
```python
# Calculate edge
edge = predicted_probability - market_price

# Only trade if edge exceeds threshold
if abs(edge) >= edge_threshold:
    # Determine direction
    side = "YES" if edge > 0 else "NO"

    # Position size as % of capital
    position_size = capital * position_size_pct

    # Calculate P&L including 7% Kalshi fees
    if won:
        gross_pnl = position_size * payoff
        fees = gross_pnl * 0.07
        pnl = gross_pnl - fees
    else:
        pnl = -position_size

    # Update capital
    capital += pnl
```

## File Structure

### New Files Created

1. **`src/fomc_analysis/backtester_v3.py`** (600+ lines)
   - Core backtesting engine
   - Kalshi API integration functions
   - Metrics computation
   - Result serialization

2. **`docs/BACKTEST_V3_GUIDE.md`**
   - Comprehensive user guide
   - Step-by-step workflow
   - Parameter tuning guide
   - Troubleshooting section

3. **`examples/run_backtest_v3.sh`**
   - Automated workflow script
   - Configurable via environment variables
   - Complete end-to-end example

4. **`examples/backtest_v3_analysis.py`**
   - Advanced result analysis
   - Contract-level breakdowns
   - Edge calibration analysis
   - Programmatic access examples

### Modified Files

1. **`src/fomc_analysis/cli.py`**
   - Added `backtest-v3` command
   - Integrated with existing CLI structure
   - Comprehensive help text

2. **`README.md`**
   - Added prominent section on new backtest
   - Quick start example
   - Link to detailed documentation

## Data Flow

```
┌─────────────────────┐
│ Kalshi API          │
│ - Resolved markets  │
│ - Price history     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│ fetch_kalshi_contract_outcomes()                │
│ - Extract contract outcomes (YES/NO)            │
│ - Get meeting dates                             │
└──────────┬──────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│ fetch_historical_prices_at_horizons()           │
│ - Get prices at 7, 14, 30 days before          │
└──────────┬──────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│ TimeHorizonBacktester.run()                     │
│ ┌─────────────────────────────────────────────┐ │
│ │ For each meeting:                           │ │
│ │   1. Train on previous meetings             │ │
│ │   2. For each horizon (7, 14, 30):          │ │
│ │      - Make prediction                      │ │
│ │      - Get market price                     │ │
│ │      - Calculate edge                       │ │
│ │      - Execute trade if edge > threshold    │ │
│ │      - Compare with actual outcome          │ │
│ └─────────────────────────────────────────────┘ │
└──────────┬──────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│ Results                                         │
│ - predictions.csv (all predictions)             │
│ - trades.csv (executed trades)                  │
│ - horizon_metrics.csv (per-horizon stats)       │
│ - backtest_results.json (complete results)      │
└─────────────────────────────────────────────────┘
```

## Usage Example

### Command Line

```bash
# Run backtest with Beta-Binomial model
fomc-analysis backtest-v3 \
  --contract-words data/kalshi_analysis/contract_words.json \
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

### Programmatic

```python
from fomc_analysis.backtester_v3 import (
    TimeHorizonBacktester,
    fetch_kalshi_contract_outcomes,
    fetch_historical_prices_at_horizons,
)
from fomc_analysis.models import BetaBinomialModel
from fomc_analysis.kalshi_client_factory import get_kalshi_client

# Load Kalshi data
client = get_kalshi_client()
outcomes = fetch_kalshi_contract_outcomes(contract_data, client)
prices = fetch_historical_prices_at_horizons(tickers, dates, [7, 14, 30], client)

# Run backtest
backtester = TimeHorizonBacktester(
    outcomes=outcomes,
    historical_prices=prices,
    horizons=[7, 14, 30],
    edge_threshold=0.10,
)

result = backtester.run(
    model_class=BetaBinomialModel,
    model_params={"alpha_prior": 1.0, "beta_prior": 1.0, "half_life": 4},
    initial_capital=10000.0,
)

# Analyze results
print(f"ROI: {result.overall_metrics['roi']*100:.1f}%")
print(f"Sharpe: {result.overall_metrics['sharpe']:.2f}")

for horizon, metrics in result.horizon_metrics.items():
    print(f"{horizon}d: {metrics.accuracy*100:.1f}% accuracy, "
          f"{metrics.win_rate*100:.1f}% win rate")
```

## Output Format

### Prediction Snapshot

```python
@dataclass
class PredictionSnapshot:
    meeting_date: str              # "2024-01-31"
    contract: str                  # "Inflation 40+"
    prediction_date: str           # "2024-01-24" (7 days before)
    days_before_meeting: int       # 7
    predicted_probability: float   # 0.65
    confidence_lower: float        # 0.55
    confidence_upper: float        # 0.75
    actual_outcome: int           # 1 (YES)
    market_price: float           # 0.55
    edge: float                   # 0.10
    correct: bool                 # True
```

### Trade Record

```python
@dataclass
class Trade:
    meeting_date: str              # "2024-01-31"
    contract: str                  # "Inflation 40+"
    prediction_date: str           # "2024-01-24"
    days_before_meeting: int       # 7
    side: str                     # "YES"
    position_size: float          # 500.00 ($500)
    entry_price: float            # 0.55
    predicted_probability: float   # 0.65
    edge: float                   # 0.10
    actual_outcome: int           # 1 (YES)
    pnl: float                    # 380.91 (after fees)
    roi: float                    # 0.76 (76% return)
```

### Horizon Metrics

```python
@dataclass
class HorizonMetrics:
    horizon_days: int              # 7
    total_predictions: int         # 45
    correct_predictions: int       # 28
    accuracy: float               # 0.622
    total_trades: int             # 18
    winning_trades: int           # 11
    win_rate: float               # 0.611
    total_pnl: float              # 1245.67
    avg_pnl_per_trade: float      # 69.20
    roi: float                    # 0.138 (13.8%)
    sharpe_ratio: float           # 1.42
    brier_score: float            # 0.185
```

## Key Design Decisions

### 1. Walk-Forward Validation

**Decision**: Train only on meetings that occurred before the prediction date

**Rationale**: Ensures no lookahead bias. Each prediction uses only information that would have been available at that time.

### 2. Multiple Horizons

**Decision**: Test at 7, 14, and 30 days before each meeting

**Rationale**:
- Different horizons have different tradeoffs
- Early predictions (30d) may have larger edges but lower accuracy
- Late predictions (7d) more accurate but smaller edges
- Allows users to optimize strategy for their risk tolerance

### 3. Separate Accuracy and Trading Metrics

**Decision**: Track prediction accuracy independently from trading performance

**Rationale**:
- A model can be accurate but unprofitable (if market is efficient)
- A model can be less accurate but profitable (if edges are large)
- Separating metrics provides clearer insights

### 4. Brier Score for Calibration

**Decision**: Include Brier score alongside accuracy

**Rationale**:
- Brier score measures probability calibration, not just binary accuracy
- A well-calibrated model predicts probabilities that match true frequencies
- Important for assessing if model's confidence levels are meaningful

### 5. Realistic Fee Structure

**Decision**: Apply 7% fee on profits only (Kalshi's actual fee structure)

**Rationale**:
- Matches real-world trading costs
- Provides conservative profitability estimates
- Fee-adjusted returns are the only meaningful metric

## Testing Recommendations

### Minimum Data Requirements

- At least 10-15 resolved FOMC meetings
- Historical price data for contracts
- Segments parsed for all meeting dates

### Parameter Sensitivity Testing

Test different configurations:

```bash
# Conservative
--edge-threshold 0.15 --position-size-pct 0.02

# Balanced
--edge-threshold 0.10 --position-size-pct 0.05

# Aggressive
--edge-threshold 0.05 --position-size-pct 0.10
```

### Model Comparison

Compare Beta-Binomial vs EWMA:

```bash
# Beta-Binomial
--model beta --alpha 1.0 --beta-prior 1.0 --half-life 4

# EWMA
--model ewma --alpha 0.5
```

## Performance Expectations

Based on the design, expect:

- **Accuracy**: 55-70% (above random guessing)
- **Win Rate**: 50-60% (after fees)
- **Brier Score**: 0.15-0.25 (0.25 = random)
- **ROI**: Depends heavily on edge_threshold and market efficiency
- **Sharpe Ratio**: > 1.0 indicates good risk-adjusted returns

## Limitations and Future Work

### Current Limitations

1. **Historical data dependency**: Requires resolved Kalshi markets
2. **No feature engineering**: Uses only historical binary outcomes
3. **Simple models**: Beta-Binomial and EWMA are baselines
4. **No dynamic position sizing**: Fixed percentage of capital

### Future Enhancements

1. **Feature-based models**: Incorporate transcript sentiment, economic indicators
2. **Kelly criterion**: Optimal position sizing based on edge and confidence
3. **Multi-contract portfolios**: Correlation-aware position sizing
4. **Live deployment**: Real-time predictions and order execution
5. **Ensemble models**: Combine multiple prediction approaches

## Conclusion

The Time-Horizon Backtest v3 provides a robust framework for evaluating quant models on FOMC mention contracts. By using actual Kalshi outcomes, testing multiple prediction horizons, and applying realistic trading costs, it gives an honest assessment of strategy profitability.

The modular design allows easy extension to new models, different time horizons, and alternative trading strategies. The comprehensive output enables deep analysis of what works and what doesn't.

## Files Modified/Created Summary

**New Files** (4):
- `src/fomc_analysis/backtester_v3.py`
- `docs/BACKTEST_V3_GUIDE.md`
- `examples/run_backtest_v3.sh`
- `examples/backtest_v3_analysis.py`

**Modified Files** (2):
- `src/fomc_analysis/cli.py`
- `README.md`

**Total Lines Added**: ~1500 lines of code and documentation

## Dependencies

No new dependencies required. Uses existing packages:
- `pandas` for data handling
- `numpy` for numerical operations
- Existing Kalshi API clients
- Existing model classes (`BetaBinomialModel`, `EWMAModel`)
