# Microstructure Edge Integration Plan

**Source:** [Jon-Becker/prediction-market-analysis](https://github.com/Jon-Becker/prediction-market-analysis)
**Paper:** "The Microstructure of Wealth Transfer in Prediction Markets"
**Dataset:** 72.1M trades, $18.26B volume on Kalshi (Jun 2021 - Nov 2025)
**Created:** 2026-01-26

---

## Executive Summary

The Becker research demonstrates that Kalshi markets exhibit systematic, exploitable inefficiencies driven by behavioral biases. By integrating these microstructure findings into our earnings prediction framework, we can add edge in three ways:

1. **Better predictions** - Calibrate our model outputs against empirical win rates at each price level
2. **Better execution** - Time orders, use limit orders, and exploit YES/NO asymmetry
3. **Better sizing** - Adjust position sizes based on where mispricing concentrates (longshots, categories, time-of-day)

---

## Key Findings From the Research

### The Numbers

| Finding | Value | Implication |
|---------|-------|-------------|
| Taker excess return | **-1.12%** | Retail systematically loses |
| Maker excess return | **+1.12%** | Liquidity providers systematically win |
| 5c contract actual win rate | **4.18%** (implied: 5%) | Longshots overpriced |
| 95c contract actual win rate | **95.83%** (implied: 95%) | Favorites slightly underpriced |
| Finance category maker-taker gap | **0.17 pp** | Most efficient category |
| Entertainment category gap | **4.79 pp** | Least efficient |
| Maker buying YES | **+0.77%** excess | Edge is structural |
| Maker buying NO | **+1.25%** excess | NO side has more edge |

### Core Insight

Wealth flows from aggressive order initiators (takers) to passive liquidity providers (makers). This transfer is driven by:
- **Optimism bias**: Takers overpay for YES contracts, especially at low prices
- **Longshot bias**: Extreme low-probability events are systematically overpriced
- **Structural advantage**: Makers capture spread regardless of information

---

## Integration Plan

### Phase 1: Calibration-Adjusted Predictions

**Goal:** Correct our model's probability outputs using empirical calibration data from 72M trades.

**What we learn from the research:** Market prices at each cent level deviate systematically from true probabilities. A contract priced at 5c wins only 4.18% of the time, not 5%. This means:
- When we compare our model's prediction to market price, we should compare to the *calibrated* win rate, not the raw price
- Edge calculations should use `true_probability - calibrated_market_probability`, not `model_probability - market_price`

**Implementation:**

#### 1a. Build Calibration Lookup Table

Create a module that maps market price (1-99 cents) to empirically-observed win rates.

```
src/earnings_analysis/microstructure/
├── __init__.py
├── calibration.py        # Price-to-win-rate calibration
├── execution.py          # Order placement optimization
└── position_sizing.py    # Microstructure-aware sizing
```

`calibration.py` should:
- Store the empirical calibration curve from the research data (win rate at each price 1-99)
- Provide `calibrated_probability(market_price_cents: int) -> float` function
- Compute `true_edge(model_prob: float, market_price_cents: int) -> float` using calibrated probabilities
- Flag "easy money" zones where calibration deviation exceeds threshold

#### 1b. Integrate Into Edge Calculation

Currently in our backtester, edge is computed as:
```python
edge = model_probability - (market_price / 100)
```

Replace with:
```python
from earnings_analysis.microstructure.calibration import calibrated_edge
edge = calibrated_edge(model_probability, market_price_cents)
```

This corrects for the market's systematic miscalibration, giving us a more accurate measure of when we truly have an edge versus when the market is simply mispriced at that price level.

#### 1c. Backtest the Calibration Adjustment

- Run existing backtest WITH and WITHOUT calibration adjustment
- Compare: accuracy, Sharpe ratio, P&L
- Validate that calibration data (from all Kalshi categories) applies to earnings mention contracts specifically

**Files to modify:**
- `src/earnings_analysis/kalshi/backtester.py` - Edge calculation
- `src/earnings_analysis/kalshi/enhanced_backtester.py` - Edge calculation
- `src/earnings_analysis/models/market_adjusted_model.py` - Market probability interpretation

---

### Phase 2: YES/NO Asymmetric Trading

**Goal:** Exploit the systematic overpricing of YES contracts by biasing toward NO positions.

**What we learn:** Takers disproportionately buy YES. At any given price level, NO contracts outperform YES by up to 64 percentage points at extreme prices. Makers buying NO earn +1.25% vs +0.77% buying YES.

**Implementation:**

#### 2a. Asymmetric Edge Thresholds

We already have asymmetric thresholds (`yes_edge_threshold=0.22`, `no_edge_threshold=0.08`), which is directionally correct. But the research provides data to set these precisely:

- Compute the YES mispricing curve: at each price level, how much does the YES probability exceed the calibrated win rate?
- Use this to set per-price-level thresholds rather than a single global number
- At 5-15 cents: YES overpriced by 5-18%, so lower our YES threshold here
- At 85-95 cents: NO overpriced (YES underpriced), so lower our NO threshold here

#### 2b. Directional Bias Score

Add a "directional bias" feature that signals when our model's prediction and the microstructure data BOTH favor the same direction:

```python
def directional_bias(model_prob, market_price_cents):
    """Score from -1 (strong NO bias) to +1 (strong YES bias)."""
    calibrated = calibrated_probability(market_price_cents)
    model_edge = model_prob - calibrated
    structural_edge = calibrated - (market_price_cents / 100)  # Market miscalibration

    # When model AND structure agree, edge is strongest
    if model_edge > 0 and structural_edge > 0:
        return min(1.0, model_edge + structural_edge)  # Double edge
    elif model_edge < 0 and structural_edge < 0:
        return max(-1.0, model_edge + structural_edge)
    else:
        return model_edge  # Conflicting signals: trust model only
```

#### 2c. Adjust Position Sizing by Direction

- When taking a YES position against structural bias (most YES buys): reduce size by 25%
- When taking a NO position with structural bias: increase size by 25%
- This tilts our portfolio toward structurally advantaged trades

**Files to modify:**
- `src/earnings_analysis/kalshi/backtester.py` - Trade direction logic
- `src/earnings_analysis/kalshi/enhanced_backtester.py` - Kelly sizing
- New: `src/earnings_analysis/microstructure/execution.py`

---

### Phase 3: Market Making Execution Strategy

**Goal:** Capture the +1.12% maker excess return by placing limit orders instead of market orders.

**What we learn:** The maker-taker gap is real and persistent. Makers earn +1.12% excess returns purely through structural positioning. Our current backtester assumes market orders (taker execution).

**Implementation:**

#### 3a. Limit Order Simulation in Backtester

Add execution mode to backtester:
- `execution_mode="taker"` (current): Immediate execution at ask/bid price
- `execution_mode="maker"` (new): Place limit order at mid-price, simulate fill probability
- `execution_mode="hybrid"` (new): Use limit order first, escalate to market order if not filled within time window

```python
class ExecutionSimulator:
    def __init__(self, mode="hybrid", patience_seconds=300):
        self.mode = mode
        self.patience = patience_seconds

    def simulate_execution(self, side, bid, ask, urgency):
        """Return expected execution price."""
        spread = ask - bid
        mid = (bid + ask) / 2

        if self.mode == "taker":
            return ask if side == "YES" else (100 - bid)
        elif self.mode == "maker":
            return mid  # Assume fill at mid (optimistic)
        elif self.mode == "hybrid":
            # Place at mid, escalate based on urgency
            fill_prob = self._fill_probability(spread, urgency)
            return mid * fill_prob + ask * (1 - fill_prob)
```

#### 3b. Spread-Aware Trade Selection

Only trade when bid-ask spread is narrow enough that our edge exceeds the spread cost:
```python
def is_tradeable(edge, bid, ask):
    spread_cost = (ask - bid) / 2  # Half-spread cost for a taker
    return edge > spread_cost + MIN_NET_EDGE
```

#### 3c. Order Timing

The research includes time-of-day analysis. If retail activity peaks during specific hours, we should:
- Place limit orders during low-activity periods (better fills, tighter spreads)
- Trade against retail flow during high-activity periods (exploit mispricing)

**Files to create:**
- `src/earnings_analysis/microstructure/execution.py` - Execution simulation
- `src/earnings_analysis/microstructure/spread_model.py` - Spread dynamics

**Files to modify:**
- `src/earnings_analysis/kalshi/backtester.py` - Add execution simulation
- `src/earnings_analysis/kalshi/enhanced_backtester.py` - Add execution simulation

---

### Phase 4: Historical Trade Data Pipeline

**Goal:** Replicate the Becker data pipeline for our specific contracts to validate findings and build proprietary microstructure features.

**What we learn:** Their DuckDB + Parquet pipeline efficiently handles 72M trades. We can use the same approach to build our own trade-level dataset for earnings mention contracts specifically.

**Implementation:**

#### 4a. Trade History Backfill

Build a targeted backfill for KXEARNINGSMENTION* tickers:

```python
class EarningsTradeBackfiller:
    """Fetch complete trade history for earnings mention contracts."""

    def __init__(self, client, storage_dir="data/trades"):
        self.client = client
        self.storage = ParquetStorage(storage_dir)

    def backfill_ticker(self, ticker: str):
        """Fetch all trades for a ticker (e.g., 'KXEARNINGSMENTIONMETA')."""
        trades = self.client.get_trades(ticker=ticker)
        self.storage.append(ticker, trades)

    def backfill_all_earnings(self):
        """Backfill all earnings mention tickers."""
        for company in ["META", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "NFLX"]:
            self.backfill_ticker(f"KXEARNINGSMENTION{company}")
```

#### 4b. Earnings-Specific Microstructure Analysis

Run the Becker analyses on our earnings subset:
- **Calibration curve** for earnings mention contracts specifically (may differ from aggregate)
- **Maker-taker gap** for finance/earnings category (they found only 0.17 pp - smallest category)
- **Time-of-day patterns** around earnings release dates
- **Spread dynamics** as earnings date approaches

Key question: Does the 0.17 pp finance gap apply to earnings mention contracts, or are those more like entertainment contracts (4.79 pp gap) since they're word-based gambles?

#### 4c. DuckDB Integration

Follow their approach: store trade data in Parquet, query with DuckDB for fast analytical queries:

```python
import duckdb

def analyze_trades(ticker: str):
    conn = duckdb.connect()
    return conn.execute(f"""
        SELECT
            price_level,
            COUNT(*) as trade_count,
            AVG(CASE WHEN outcome = 'yes' THEN 1 ELSE 0 END) as actual_win_rate,
            price_level / 100.0 as implied_probability,
            actual_win_rate - implied_probability as mispricing
        FROM read_parquet('data/trades/{ticker}/*.parquet')
        WHERE status = 'finalized'
        GROUP BY price_level
        ORDER BY price_level
    """).fetchdf()
```

**Files to create:**
- `src/earnings_analysis/microstructure/trade_backfiller.py`
- `src/earnings_analysis/microstructure/trade_analyzer.py`
- `scripts/backfill_earnings_trades.py`
- `scripts/analyze_earnings_microstructure.py`

**Dependencies to add:**
- `duckdb` - Already used by the research repo, fast SQL on Parquet
- `pyarrow` - Already in our deps for Parquet I/O

---

### Phase 5: Statistical Rigor

**Goal:** Apply proper hypothesis testing to validate our edge claims, following the Becker methodology.

**What we learn:** The research uses Mann-Whitney U tests, z-tests, Welch's t-tests, and correlation analysis with effect sizes (Cohen's d). Our current backtest reports raw metrics without statistical significance.

**Implementation:**

#### 5a. Significance Testing for Edge

```python
from scipy import stats

def test_edge_significance(trade_returns: list[float]) -> dict:
    """Test if our trading edge is statistically significant."""
    n = len(trade_returns)
    mean_return = np.mean(trade_returns)

    # One-sample t-test: H0 = mean return is 0
    t_stat, p_value = stats.ttest_1samp(trade_returns, 0)

    # Effect size (Cohen's d)
    cohens_d = mean_return / np.std(trade_returns)

    # Confidence interval
    ci_low, ci_high = stats.t.interval(0.95, n-1, loc=mean_return, scale=stats.sem(trade_returns))

    return {
        "n_trades": n,
        "mean_return": mean_return,
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "ci_95": (ci_low, ci_high),
        "significant": p_value < 0.05,
    }
```

#### 5b. Calibration Testing

Apply the Becker calibration methodology to our predictions:

```python
def calibration_test(predictions: list[float], outcomes: list[int], n_bins=10):
    """Test if our model is well-calibrated using binned calibration."""
    bins = np.linspace(0, 1, n_bins + 1)
    calibration = []

    for i in range(n_bins):
        mask = (predictions >= bins[i]) & (predictions < bins[i+1])
        if mask.sum() > 0:
            predicted = predictions[mask].mean()
            actual = outcomes[mask].mean()
            calibration.append({
                "bin_center": (bins[i] + bins[i+1]) / 2,
                "predicted": predicted,
                "actual": actual,
                "deviation": actual - predicted,
                "n": mask.sum(),
            })

    # Hosmer-Lemeshow test
    # Brier score decomposition (reliability + resolution + uncertainty)
    return calibration
```

#### 5c. Multiple Comparison Correction

When testing edge across many word-ticker combinations, apply Bonferroni or FDR correction:

```python
from statsmodels.stats.multitest import multipletests

# After testing edge for each word:
p_values = [test_result["p_value"] for test_result in all_tests]
reject, corrected_p, _, _ = multipletests(p_values, method="fdr_bh")
```

**Files to create:**
- `src/earnings_analysis/microstructure/statistical_tests.py`

**Files to modify:**
- `src/earnings_analysis/kalshi/backtester.py` - Add significance reporting
- `src/earnings_analysis/kalshi/enhanced_backtester.py` - Add significance reporting

---

### Phase 6: Regime-Aware Strategy

**Goal:** Adapt our strategy based on market regime, following the temporal analysis from the research.

**What we learn:** The maker-taker dynamic reversed around Q4 2024 when professional market makers entered. Before that, unsophisticated makers lost money. The market is becoming more efficient over time.

**Implementation:**

#### 6a. Market Efficiency Monitoring

Track efficiency metrics over time for our contracts:
- Bid-ask spread trends (narrowing = more efficient)
- Volume trends (increasing = more attention)
- Calibration deviation (shrinking = better pricing)

```python
class EfficiencyMonitor:
    def compute_efficiency_score(self, ticker: str, lookback_days=90) -> float:
        """Score from 0 (inefficient) to 1 (efficient)."""
        spread = self.avg_spread(ticker, lookback_days)
        volume = self.avg_volume(ticker, lookback_days)
        calibration = self.calibration_deviation(ticker, lookback_days)

        # Efficient markets have: tight spreads, high volume, low calibration error
        score = (
            0.4 * (1 - min(spread / 10, 1)) +  # Spread component
            0.3 * min(volume / 1000, 1) +        # Volume component
            0.3 * (1 - min(calibration / 0.1, 1)) # Calibration component
        )
        return score
```

#### 6b. Adaptive Edge Thresholds

Increase edge threshold as markets become more efficient:

```python
def adaptive_threshold(base_threshold: float, efficiency_score: float) -> float:
    """Higher efficiency → higher threshold needed."""
    # In efficient markets (score > 0.7), raise threshold
    # In inefficient markets (score < 0.3), lower threshold
    adjustment = (efficiency_score - 0.5) * 0.10  # +-5% adjustment range
    return base_threshold + adjustment
```

#### 6c. New Contract Detection

As Kalshi adds new earnings tickers, early markets tend to be less efficient. Prioritize:
- Newly listed tickers (less market maker attention)
- Low-volume contracts (less competition)
- Contracts with wide spreads (more room for edge)

**Files to create:**
- `src/earnings_analysis/microstructure/regime.py` - Efficiency monitoring
- `scripts/monitor_market_efficiency.py` - Periodic efficiency reports

---

## Implementation Priority & Sequencing

```
Phase 1: Calibration-Adjusted Predictions     [HIGH IMPACT, LOW EFFORT]
  └── Immediate edge improvement by correcting model outputs
  └── Depends on: extracting calibration data from research

Phase 2: YES/NO Asymmetric Trading            [HIGH IMPACT, LOW EFFORT]
  └── We already have asymmetric thresholds; refine with data
  └── Depends on: Phase 1 calibration data

Phase 3: Market Making Execution              [HIGH IMPACT, MEDIUM EFFORT]
  └── Captures structural edge (+1.12%) on every trade
  └── Depends on: Kalshi API order types, spread data

Phase 4: Historical Trade Data Pipeline        [MEDIUM IMPACT, MEDIUM EFFORT]
  └── Enables earnings-specific microstructure analysis
  └── Depends on: Kalshi trade API access, storage infra

Phase 5: Statistical Rigor                     [MEDIUM IMPACT, LOW EFFORT]
  └── Validates whether our edge is real or noise
  └── Depends on: sufficient trade history from backtests

Phase 6: Regime-Aware Strategy                 [LOW-MEDIUM IMPACT, HIGH EFFORT]
  └── Adapts to evolving market structure
  └── Depends on: Phase 4 data, longer time horizon
```

### Recommended Execution Order

1. **Phase 1 + Phase 2** (parallel) - Calibration and asymmetric trading
2. **Phase 5** - Statistical rigor (validate phases 1-2 actually help)
3. **Phase 3** - Market making execution
4. **Phase 4** - Trade data pipeline
5. **Phase 6** - Regime monitoring

---

## Risk Assessment

### What Could Go Wrong

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Aggregate calibration data doesn't apply to earnings contracts | Medium | High | Phase 4 builds earnings-specific calibration |
| Finance category is already too efficient (0.17 pp gap) | Medium | High | Focus on newly listed tickers, wider-spread contracts |
| Market adapts to our strategy | Low | Medium | Phase 6 monitors efficiency, adjusts thresholds |
| Kalshi changes fee structure | Low | High | Parameterize fees, re-run backtests |
| Sample size too small for statistical significance | High | Medium | Pool across tickers, extend time horizon |

### Key Unknowns to Resolve

1. **Are earnings mention contracts more like "Finance" (efficient) or "Entertainment" (inefficient)?**
   - They're CFTC-regulated financial instruments tracking company behavior
   - But the contract structure (word mentions) feels more like entertainment betting
   - Phase 4 will answer this definitively

2. **Does our model add edge beyond the structural maker advantage?**
   - The +1.12% maker return exists for ANY maker, regardless of information
   - Our model should add ADDITIONAL edge through superior probability estimates
   - Phase 5 will decompose our returns into structural vs informational components

3. **How does efficiency change around earnings dates?**
   - Markets likely become more efficient as earnings approach (more attention)
   - But they may be wildly inefficient for newly listed contracts
   - Phase 4 temporal analysis will reveal the pattern

---

## Dependencies & New Packages

```toml
# Add to pyproject.toml
[project.optional-dependencies]
microstructure = [
    "duckdb>=1.0",          # Fast analytical queries on Parquet
    "statsmodels>=0.14",    # Multiple comparison correction
]
```

Note: `scipy`, `pandas`, `pyarrow` are already in our dependencies.

---

## Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Backtest Sharpe ratio | ~0.8 | >1.2 | Walk-forward backtest |
| Edge per trade | ~5% raw | ~3% after calibration (more honest) | Calibration-adjusted P&L |
| Win rate | ~58% | >62% | Asymmetric threshold tuning |
| Statistical significance | Unknown | p < 0.05 | Phase 5 t-tests |
| Execution cost | Assumed 0 | Modeled | Phase 3 spread simulation |
| False edge rate | Unknown | <10% | FDR-corrected testing |

---

## Summary

The Becker research provides three categories of actionable intelligence:

1. **Calibration data** (Phase 1-2): Correct our probability estimates and trade direction using empirical win rates from 72M trades. This is the lowest-hanging fruit.

2. **Execution strategy** (Phase 3): Capture the persistent +1.12% maker excess return by using limit orders. This is free edge that requires no predictive skill.

3. **Market structure understanding** (Phase 4-6): Build earnings-specific microstructure data to validate whether aggregate findings apply to our niche, monitor regime changes, and adapt over time.

The key insight: even if our *prediction model* adds zero informational edge, the *execution strategy* (be a maker, sell longshots, exploit YES bias) adds structural edge. Combining both informational AND structural edge should compound returns.
