# Earnings Call Mentions Pipeline - Analysis Report

**Generated:** 2026-01-26
**Branch:** claude/test-earnings-pipeline-RLOq6

## Executive Summary

I ran the earnings call mentions pipeline end-to-end, analyzing 424 historical finalized Kalshi contracts across 7 companies (META, TSLA, NVDA, AAPL, MSFT, AMZN, NFLX) and 42 currently active contracts. The analysis revealed that **the market is largely efficient** for these contracts, with only **1 potentially profitable trade** identified.

### Key Findings

| Metric | Value |
|--------|-------|
| Historical Contracts Analyzed | 424 |
| Active Contracts Analyzed | 42 |
| Model Accuracy (Backtest) | 84.7% |
| Brier Score | 0.1211 |
| Actionable Trades | 1 |

## 1. Data Overview

### Historical Data (Finalized Contracts)

| Company | Finalized | YES Rate |
|---------|-----------|----------|
| META | 56 | 57.1% |
| TSLA | 61 | 60.7% |
| NVDA | 75 | 60.0% |
| AAPL | 57 | 68.4% |
| MSFT | 58 | 63.8% |
| AMZN | 61 | 59.0% |
| NFLX | 56 | 64.3% |

### Active Contracts (Current Market)

| Company | Active Contracts |
|---------|-----------------|
| META | 14 |
| TSLA | 17 |
| MSFT | 11 |

## 2. Model Performance

### Original Model (Beta-Binomial with Category Priors)

The original model used category-level priors (e.g., "product" words have 67% mention rate) when word-specific data was limited. This led to:

- **Over-aggressive predictions** that often contradicted market prices
- **Many false signals** (30+ trades suggested)
- **Market mispricings were illusory** - the model was wrong, not the market

### Improved Model (Market-Adjusted)

I created a new `MarketAdjustedModel` that:

1. **Shrinks predictions towards market prices** when data is limited
2. **Requires minimum sample sizes** before generating trade signals
3. **Uses confidence-adjusted edge** calculations
4. **Applies Kelly criterion** for position sizing

Results after improvement:

| Metric | Before | After |
|--------|--------|-------|
| Accuracy | 52.9% | 84.7% |
| Brier Score | 0.2211 | 0.1211 |
| False Signals | Many | 1 |

## 3. Market Efficiency Analysis

### Why Most "Opportunities" Aren't Real

The initial analysis suggested many mispricings (e.g., DOGE at 8% vs model's 63%). Investigation revealed these were **model failures, not market failures**:

1. **Limited word-specific data**: Most words had only 1-3 historical samples
2. **Category priors were misleading**: Using "external events are mentioned 56% of the time" doesn't apply to specific sensitive topics like DOGE
3. **Markets have more information**: Traders factor in news, company strategy, and sentiment - not just historical frequency

### Market Prices Reflect Informed Predictions

- Words priced at **<15%** (DOGE, TikTok, Cybersecurity): Topics companies are likely avoiding
- Words priced at **>90%** (Robotaxi, FSD, Teams): Core business terms that will definitely be mentioned

## 4. Predictions for Upcoming Markets

### High Confidence Predictions (3+ historical samples)

| Company | Word | Market | Model | Signal |
|---------|------|--------|-------|--------|
| TSLA | DOGE | $0.08 | $0.26 | BUY YES |
| TSLA | Cybertruck | $0.64 | $0.40 | Hold |
| TSLA | Robotaxi | $0.99 | $0.80 | Hold |
| TSLA | Gigafactory | $0.58 | $0.40 | Hold |
| MSFT | Teams | $0.96 | $0.80 | Hold |
| MSFT | OpenAI | $0.95 | $0.80 | Hold |
| MSFT | LinkedIn | $0.97 | $0.80 | Hold |

### Single Recommended Trade

**TSLA - DOGE / Department of Government Efficiency**

| Metric | Value |
|--------|-------|
| Market Price | $0.08 (8%) |
| Model Price | $0.26 (26%) |
| Edge | +18% |
| Adjusted Edge | +13% |
| Historical Pattern | [0, 0, 1] - mentioned 1/3 times |
| Confidence | 75% (3 samples) |
| Recommended Position | 3.4% of capital |
| Kelly Fraction | 7.2% |

**Reasoning**: The historical data shows DOGE was mentioned 1 out of 3 times (33%), but the market prices it at only 8%. This suggests potential underpricing. However, **significant risk exists** as the market may be pricing in forward-looking information (e.g., Tesla may be deliberately avoiding political topics).

## 5. Improvements Implemented

### New MarketAdjustedModel

Created `/src/earnings_analysis/models/market_adjusted_model.py` with:

```python
class MarketAdjustedModel(EarningsModel):
    """
    Key improvements:
    1. Shrinkage towards market prices when data is limited
    2. Confidence-adjusted edge calculations
    3. Kelly criterion position sizing
    4. Minimum sample requirements before trading
    """
```

### Key Parameters

- `shrinkage_samples=3`: At 3 samples, model weight equals market weight
- `min_samples_to_trade=2`: Won't generate signals with <2 samples
- `min_adjusted_edge=0.10`: Requires 10%+ confidence-adjusted edge

## 6. Risk Warnings

### Why the Recommended Trade May Still Fail

1. **DOGE is politically charged**: Tesla may deliberately avoid this topic
2. **Market knows more**: Informed traders may have news/sentiment we don't see
3. **Small sample size**: 3 samples isn't statistically significant
4. **Regime change**: Past patterns may not predict future behavior

### Position Sizing Recommendation

- **Maximum position**: 3-5% of capital per trade
- **Stop loss**: Set at 50% of position value
- **Expected value**: Positive but uncertain

## 7. Data Files Generated

| File | Description |
|------|-------------|
| `data/historical_finalized_contracts.csv` | 424 historical outcomes |
| `data/active_contracts.csv` | 42 current market prices |
| `data/final_predictions.csv` | All predictions |
| `data/improved_model_recommendations.csv` | Filtered recommendations |
| `data/backtest_results_detailed.csv` | Backtest details |

## 8. Conclusion

The earnings call mentions pipeline is **functional but limited by data availability**. The market appears largely efficient, with most apparent mispricings being model errors rather than true opportunities.

### Profitability Assessment

| Scenario | Expected Return |
|----------|----------------|
| Optimistic (model is right) | +$26 per $100 bet on DOGE |
| Pessimistic (market is right) | -$100 per $100 bet |
| Expected Value (model confidence) | +$7 per $100 bet |

### Recommendations

1. **Consider the DOGE trade** with small position (3% of capital max)
2. **Do not trade** other contracts - market prices are reasonable
3. **Collect more data** - accuracy will improve with more historical samples
4. **Add sentiment analysis** - would help detect regime changes

## 9. Code Changes

Added new model: `src/earnings_analysis/models/market_adjusted_model.py`

- Market-adjusted shrinkage for limited data
- Confidence-based position sizing
- Kelly criterion integration
- Proper uncertainty quantification
