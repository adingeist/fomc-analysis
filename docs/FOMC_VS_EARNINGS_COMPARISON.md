# FOMC vs Earnings Framework Comparison

This document compares the FOMC and Earnings Kalshi frameworks to show what's reused, what's adapted, and what's new.

---

## High-Level Comparison

| Aspect | FOMC Framework | Earnings Framework |
|--------|----------------|-------------------|
| **Event Type** | FOMC press conferences | Earnings calls |
| **Frequency** | ~8 per year | ~4 per year per company |
| **Speaker** | Fed Chair (Jerome Powell) | CEO, CFO, executives |
| **Kalshi Markets** | `FEDSPEAK{WORD}` series | `KXEARNINGSMENTION{TICKER}` series |
| **Prediction Target** | Will Powell say X? | Will CEO/CFO say X? |
| **Data Source** | Federal Reserve transcripts | SEC filings, IR sites, APIs |
| **Model** | Beta-Binomial Bayesian | Beta-Binomial Bayesian ✓ |
| **Backtest Engine** | `backtester_v3.py` | Adapted version ✓ |

---

## Workflow Comparison

### FOMC Workflow

```
1. Fetch Kalshi Contracts
   ↓
   Input: "FEDSPEAK" series on Kalshi
   Output: Words like "inflation", "rates", "employment"

2. Fetch FOMC Transcripts
   ↓
   Input: Federal Reserve website
   Output: Press conference transcripts (Powell only)

3. Extract Phrases
   ↓
   Input: Transcripts + Contract words
   Output: Phrase counts per conference

4. Featurize
   ↓
   Input: Phrase counts
   Output: Feature matrix (dates × phrases)

5. Model Training
   ↓
   Model: Beta-Binomial (Powell's historical mention rates)
   Output: Probability of mention for next conference

6. Backtest
   ↓
   Walk-forward validation
   Output: P&L, accuracy, Sharpe ratio
```

### Earnings Workflow

```
1. Fetch Kalshi Contracts
   ↓
   Input: "KXEARNINGSMENTION{TICKER}" series on Kalshi
   Output: Words like "AI", "cloud", "margin"

2. Fetch Earnings Transcripts
   ↓
   Input: SEC EDGAR / Alpha Vantage / Company IR
   Output: Earnings call transcripts (CEO, CFO, analysts)

3. Segment by Speaker
   ↓
   Input: Raw transcripts
   Output: Segments labeled by role (CEO, CFO, analyst)

4. Extract Word Counts
   ↓
   Input: Segments + Contract words
   Output: Word counts per call (filtered by speaker)

5. Featurize
   ↓
   Input: Word counts
   Output: Feature matrix (call dates × words)

6. Model Training
   ↓
   Model: Beta-Binomial (company's historical mention rates)
   Output: Probability of mention for next call

7. Backtest
   ↓
   Walk-forward validation
   Output: P&L, accuracy, Sharpe ratio
```

---

## Code Reuse Analysis

### ✅ Directly Reused (100% shared)

| Component | Location | Notes |
|-----------|----------|-------|
| **Kalshi Client** | `fomc_analysis/kalshi_client_factory.py` | Same API for both |
| **Kalshi SDK** | `fomc_analysis/kalshi_sdk.py` | Same underlying SDK |
| **Beta-Binomial Math** | `scipy.stats.beta` | Same Bayesian approach |
| **Walk-Forward Logic** | Backtest loop structure | Same no-lookahead principle |
| **P&L Calculation** | YES/NO contract math | Same Kalshi contract mechanics |

### 🔧 Adapted (Modified from FOMC)

| Component | FOMC Version | Earnings Version | Key Differences |
|-----------|--------------|------------------|-----------------|
| **Contract Analyzer** | `kalshi_contract_analyzer.py` | `earnings_analysis/kalshi/contract_analyzer.py` | • Series ticker format<br>• Multiple companies vs single Fed<br>• Threshold parsing |
| **Backtester** | `backtester_v3.py` | `earnings_analysis/kalshi/backtester.py` | • Ticker parameter added<br>• Same core logic<br>• Same trading rules |
| **Beta-Binomial Model** | `beta_binomial_model.py` | `earnings_analysis/models/beta_binomial.py` | • Same algorithm<br>• Earnings-specific interface<br>• Same priors/posteriors |
| **Phrase Counting** | Simple word matching | `earnings_analysis/features/featurizer.py` | • Speaker filtering<br>• Role-based segmentation<br>• More complex preprocessing |

### 🆕 New (Earnings-Specific)

| Component | Location | Purpose |
|-----------|----------|---------|
| **Speaker Segmenter** | `earnings_analysis/parsing/speaker_segmenter.py` | Identify CEO, CFO, analysts in transcripts |
| **Transcript Parser** | `earnings_analysis/parsing/transcript_parser.py` | Parse earnings call text formats |
| **Transcript Fetcher** | `earnings_analysis/fetchers/transcript_fetcher.py` | Get transcripts from SEC/APIs |
| **Multi-Ticker Support** | `earnings_analysis/kalshi/contract_analyzer.py` | Handle different companies |
| **Role Filtering** | `featurizer.py` | Count words only from executives |

---

## Key Architectural Differences

### 1. Single Speaker vs Multiple Speakers

**FOMC:**
- Single speaker (Fed Chair)
- No speaker identification needed
- Simple text extraction

**Earnings:**
- Multiple speakers (CEO, CFO, analysts, others)
- Speaker identification required
- Role-based filtering critical

**Code Impact:**
```python
# FOMC: Simple
text = get_full_transcript(date)
count = count_word(text, "inflation")

# Earnings: Complex
segments = parse_transcript(call_file)
exec_segments = filter_by_role(segments, ["ceo", "cfo"])
combined_text = join_segments(exec_segments)
count = count_word(combined_text, "ai")
```

### 2. Single Entity vs Multiple Companies

**FOMC:**
- One time series (Fed Chair's patterns)
- All data in same format
- Consistent speaker style

**Earnings:**
- Multiple time series (one per company)
- Different transcript formats
- Varying speaker styles

**Code Impact:**
```python
# FOMC: Single series
analyzer = ContractAnalyzer(client)
contracts = analyzer.fetch_contracts()  # All FEDSPEAK

# Earnings: Ticker-specific
analyzer = EarningsContractAnalyzer(client, "META")
contracts = analyzer.fetch_contracts()  # Just META

analyzer2 = EarningsContractAnalyzer(client, "TSLA")
contracts2 = analyzer2.fetch_contracts()  # Just TSLA
```

### 3. Data Source Complexity

**FOMC:**
- Single source (Federal Reserve website)
- Consistent format
- Free and public
- Easy to scrape

**Earnings:**
- Multiple sources (SEC, APIs, IR sites)
- Inconsistent formats
- Some require payment/API keys
- Harder to scrape

**Code Impact:**
```python
# FOMC: Simple
transcript = fetch_fed_transcript(date)

# Earnings: Complex
if source == "sec":
    transcript = fetch_sec_filing(ticker, date)
elif source == "alphavantage":
    transcript = fetch_from_api(ticker, date)
elif source == "manual":
    transcript = load_local_file(ticker, date)
else:
    raise ValueError(f"Unknown source: {source}")
```

---

## File Structure Comparison

### FOMC Framework

```
fomc-analysis/
├── src/fomc_analysis/
│   ├── kalshi_client_factory.py  ← Shared
│   ├── kalshi_contract_analyzer.py
│   ├── beta_binomial_model.py
│   ├── backtester_v3.py
│   └── featurizer.py
└── data/
    └── fomc_transcripts/
```

### Earnings Framework

```
fomc-analysis/
├── src/
│   ├── fomc_analysis/  ← Original
│   │   └── kalshi_client_factory.py  ← Shared
│   └── earnings_analysis/  ← New
│       ├── kalshi/
│       │   ├── contract_analyzer.py  ← Adapted
│       │   └── backtester.py  ← Adapted
│       ├── models/
│       │   └── beta_binomial.py  ← Adapted
│       ├── features/
│       │   └── featurizer.py  ← New
│       ├── parsing/  ← New
│       │   ├── speaker_segmenter.py
│       │   └── transcript_parser.py
│       └── fetchers/  ← New
│           └── transcript_fetcher.py
└── data/
    └── earnings/
        └── segments/  ← New
```

---

## Model Comparison

### Beta-Binomial Model (Same Core)

Both use identical Bayesian approach:

```python
# Prior
alpha_prior = 1.0  # Uniform prior
beta_prior = 1.0

# Update with data
alpha_post = alpha_prior + sum(successes)
beta_post = beta_prior + sum(failures)

# Prediction
probability = alpha_post / (alpha_post + beta_post)
```

### Feature Differences

**FOMC Features:**
- Simple word/phrase counts
- No speaker filtering needed
- Historical Fed Chair patterns

**Earnings Features:**
- Word counts from executives only
- Filtered by speaker role
- Company-specific patterns
- Can vary by CEO/CFO changes

### Prediction Differences

**FOMC:**
- Predict: "Will Powell say 'inflation' ≥3 times?"
- Based on: Powell's historical mention rates
- Context: Monetary policy, economic conditions

**Earnings:**
- Predict: "Will META CEO say 'AI' ≥3 times?"
- Based on: META's historical mention rates
- Context: Business performance, strategy

---

## Backtest Comparison

### Core Logic (Identical)

Both use same walk-forward validation:

```python
for i, current_date in enumerate(call_dates):
    if i < min_train_window:
        continue

    # Train on ALL previous events
    train_features = features[:i]
    train_outcomes = outcomes[:i]

    # Predict ONLY current event
    model.fit(train_features, train_outcomes)
    prediction = model.predict(features[i])

    # Calculate edge vs market
    edge = prediction - market_price

    # Trade if edge exceeds threshold
    if abs(edge) > edge_threshold:
        execute_trade(...)
```

### Parameter Differences

| Parameter | FOMC Default | Earnings Default | Reasoning |
|-----------|--------------|------------------|-----------|
| `min_train_window` | 5 conferences | 4 calls | Fewer earnings calls available |
| `half_life` | 6 conferences | 8 calls | Companies more consistent than Fed policy |
| `edge_threshold` | 0.12 | 0.12 | Same ✓ |
| `position_size_pct` | 0.03 | 0.03 | Same ✓ |
| `fee_rate` | 0.07 | 0.07 | Same ✓ (Kalshi fee) |

---

## Testing Comparison

### FOMC Verification

```bash
# Run FOMC backtest
python run_e2e_backtest.py
```

Output:
- Fed Chair word mentions
- FEDSPEAK contract predictions
- Historical P&L

### Earnings Verification

```bash
# Run Earnings backtest
python examples/verify_earnings_framework.py
```

Output:
- CEO/CFO word mentions
- Earnings mention contract predictions
- Historical P&L per ticker

---

## Summary: What's Reused vs New

### 🟢 100% Reused (No Changes)
- Kalshi API client
- Beta-Binomial mathematics
- Walk-forward backtest logic
- P&L calculation
- Trading rules (edge thresholds, position sizing, fees)

### 🟡 75% Reused (Minor Adaptations)
- Contract analyzer (added ticker parameter)
- Backtester (added ticker tracking)
- Beta-Binomial model (same math, different interface)

### 🔵 50% Reused (Significant Adaptations)
- Featurizer (added speaker filtering)
- Phrase counting (added role-based logic)

### 🔴 0% Reused (Completely New)
- Speaker segmentation
- Transcript parsing
- Multi-ticker support
- Role filtering
- Earnings-specific data fetchers

---

## Code Reuse Percentage

**Overall Reuse:** ~60%

Breakdown:
- **Core Kalshi Logic:** 100% reused
- **Backtesting Engine:** 90% reused
- **Statistical Model:** 95% reused
- **Data Processing:** 40% reused (speaker segmentation is new)
- **Data Fetching:** 20% reused (different sources)

---

## Lessons from Adaptation

### ✅ What Worked Well

1. **Kalshi Client Abstraction**
   - Single client works for both FEDSPEAK and EARNINGS series
   - No changes needed to API layer

2. **Beta-Binomial Model**
   - Math is universal (binary outcomes)
   - Easy to adapt to different event types

3. **Backtest Framework**
   - Walk-forward logic is event-agnostic
   - Trading rules transfer perfectly

### ⚠️ What Required Rework

1. **Speaker Identification**
   - FOMC: trivial (always Powell)
   - Earnings: complex (CEO vs CFO vs analyst)
   - Solution: Built speaker segmentation module

2. **Data Sources**
   - FOMC: single reliable source
   - Earnings: multiple inconsistent sources
   - Solution: Flexible fetcher architecture

3. **Multi-Entity Support**
   - FOMC: one time series
   - Earnings: time series per company
   - Solution: Ticker-parameterized classes

---

## Recommendations for Future Event Types

Based on FOMC → Earnings adaptation, if building a new event type (e.g., Fed Minutes, Congressional Hearings):

### Reuse These:
- ✅ Kalshi client factory
- ✅ Beta-Binomial model (for binary prediction)
- ✅ Backtest walk-forward logic
- ✅ Trading rules and P&L calculation

### Adapt These:
- 🔧 Contract analyzer (new series ticker format)
- 🔧 Featurizer (event-specific text processing)
- 🔧 Data fetchers (event-specific sources)

### Build New:
- 🆕 Event-specific preprocessing (if speakers, documents, etc.)
- 🆕 Domain-specific features (if needed)
- 🆕 Data quality validation (format-specific)

---

## Conclusion

The Earnings framework successfully reuses ~60% of the FOMC codebase:

- **Core Kalshi & Trading Logic:** 100% reused ✅
- **Statistical Models:** 95% reused ✅
- **Backtest Engine:** 90% reused ✅
- **Data Processing:** 40% reused (speaker complexity)
- **Data Fetching:** 20% reused (source diversity)

The adaptation validates the original FOMC framework's design:
- Modular architecture enables reuse
- Event-agnostic backtesting works across domains
- Kalshi integration abstracts away market-specific details

**Key Takeaway:** The core prediction and trading infrastructure is highly reusable. Event-specific complexity lives in data preprocessing, which is isolated from the core logic.
