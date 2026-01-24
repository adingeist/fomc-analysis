# Claude Agent Setup Guide

**For:** Future Claude Code agents working on this project
**Updated:** 2026-01-24

This document explains how to set up and work with the FOMC/Earnings Kalshi prediction framework. Read this first before making changes.

---

## Quick Start

### Environment Setup

The user has already configured Kalshi API credentials as **environment variables** (not .env file):

```bash
# These are already set - DO NOT create a .env file
printenv KALSHI_API_KEY_ID        # Exists ✅
printenv KALSHI_PRIVATE_KEY_BASE64  # Exists ✅
```

**Important:** The code reads from environment variables automatically via `settings.kalshi_api_key_id` and `settings.kalshi_private_key_base64`. Do not assume you need a `.env` file.

### Virtual Environment

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install package in editable mode
uv pip install -e .
```

### Verify Installation

```bash
# Test imports
python -c "
from earnings_analysis.kalshi import EarningsContractAnalyzer
from earnings_analysis.kalshi.backtester import EarningsKalshiBacktester
print('✅ All imports working')
"

# Test Kalshi API connection
python scripts/explore_kalshi_earnings_contracts.py
```

---

## Project Structure

### Key Directories

```
fomc-analysis/
├── src/
│   ├── fomc_analysis/          # Original FOMC framework (reused)
│   │   ├── kalshi_client_factory.py  # ← Shared Kalshi client
│   │   ├── kalshi_sdk.py      # ← SDK wrapper (handles base64 quotes!)
│   │   └── kalshi_api.py      # Legacy API client
│   │
│   └── earnings_analysis/      # New Earnings framework
│       ├── __init__.py         # Main exports
│       ├── kalshi/
│       │   ├── contract_analyzer.py  # Fetch & analyze contracts
│       │   └── backtester.py  # Backtesting engine
│       ├── models/
│       │   ├── beta_binomial.py  # Prediction model
│       │   └── base.py        # Model interface
│       ├── parsing/
│       │   ├── speaker_segmenter.py  # CEO/CFO identification
│       │   └── transcript_parser.py  # Parse transcripts
│       └── features/
│           └── featurizer.py  # Word counting & features
│
├── scripts/
│   └── explore_kalshi_earnings_contracts.py  # ← Explore real contracts
│
├── examples/
│   └── verify_earnings_framework.py  # ← Complete working example
│
├── docs/
│   ├── EARNINGS_VERIFICATION_REPORT.md  # Detailed verification
│   ├── REAL_KALSHI_CONTRACTS_DISCOVERY.md  # Live API findings
│   ├── EARNINGS_QUICKSTART.md  # User guide
│   └── FOMC_VS_EARNINGS_COMPARISON.md  # Code reuse analysis
│
├── COMPLETE_VERIFICATION.md  # ← Read this for full context
└── CLAUDE.md  # ← This file
```

### Important Files

| File | Purpose | Notes |
|------|---------|-------|
| `src/fomc_analysis/kalshi_client_factory.py` | Kalshi API client | Shared between FOMC & Earnings |
| `src/fomc_analysis/kalshi_sdk.py` | SDK wrapper | **Strips quotes from base64!** |
| `src/earnings_analysis/kalshi/contract_analyzer.py` | Fetch contracts | Ticker-specific |
| `src/earnings_analysis/kalshi/backtester.py` | Backtest engine | Walk-forward validation |
| `examples/verify_earnings_framework.py` | Full demo | Use as reference |

---

## Kalshi API Integration

### How Credentials Work

1. **Environment variables** (set by user):
   ```bash
   KALSHI_API_KEY_ID="656d559f-0ffd-4738-a9d7-38f1e0efd5a9"
   KALSHI_PRIVATE_KEY_BASE64="LS0tLS1CRUd..."  # NOTE: Has quotes!
   ```

2. **Code reads via settings** (`src/fomc_analysis/config.py`):
   ```python
   from .config import settings
   api_key = settings.kalshi_api_key_id
   private_key = settings.kalshi_private_key_base64
   ```

3. **Quote stripping** (CRITICAL - see `kalshi_sdk.py:27-32`):
   ```python
   def _decode_private_key(private_key_base64: str) -> bytes:
       # Strip quotes if present (env vars may have them)
       cleaned = private_key_base64.strip().strip('"').strip("'")
       return base64.b64decode(cleaned, validate=True)
   ```

### Client Usage

```python
# Get client (synchronous wrapper around async SDK)
from fomc_analysis.kalshi_client_factory import get_kalshi_client

client = get_kalshi_client()

# Fetch contracts (synchronous - no await!)
markets = client.get_markets(
    series_ticker="KXEARNINGSMENTIONMETA",
    status="active"
)

# Client is synchronous - it uses internal event loop
# Do NOT use 'await' on client methods
```

### Common Pitfall: Async/Sync Confusion

```python
# ❌ WRONG - client methods are synchronous
markets = await client.get_markets(...)

# ✅ CORRECT - no await
markets = client.get_markets(...)
```

The `KalshiSdkAdapter` wraps async SDK calls in a sync interface using `run_until_complete()`. Scripts that use the client should be **synchronous**, not async.

---

## Kalshi Contract Structure

### Real Contract Format

Contracts are **BINARY** (mentioned vs not mentioned), NOT threshold-based!

```json
{
  "ticker": "KXEARNINGSMENTIONMETA-26JUN30-VR",
  "title": "What will Meta Platforms, Inc. say during their next earnings call?",
  "custom_strike": {
    "Word": "VR / Virtual Reality"  // ← Extract word from here!
  },
  "yes_sub_title": "VR / Virtual Reality",
  "status": "active",  // or "finalized"
  "last_price": 37,  // cents (0.37)
  "yes_bid": 33,
  "yes_ask": 37,
  "rules_primary": "If VR / Virtual Reality is said by any Meta Platforms, Inc. representative...",
  "expiration_time": "2026-06-30T14:00:00Z"
}
```

### Key Fields

- **Word:** `custom_strike.Word` (e.g., "VR / Virtual Reality", "TikTok")
- **Threshold:** Always 1 (binary: mentioned at all)
- **Prices:** In cents (divide by 100 for dollars)
- **Status:** `active`, `settled`, `finalized`, `closed`
- **Outcome:** For finalized contracts, `last_price` is 1 (YES) or 99 (NO)

### Series Ticker Format

```
KXEARNINGSMENTION{TICKER}
```

Examples:
- `KXEARNINGSMENTIONMETA`
- `KXEARNINGSMENTIONTSLA`
- `KXEARNINGSMENTIONNVDA`

---

## Running Tests and Examples

### 1. Explore Real Kalshi Contracts

```bash
python scripts/explore_kalshi_earnings_contracts.py
```

**Output:**
- Lists all available contracts for each company
- Shows current market prices
- Saves to `data/kalshi_earnings_contracts_summary.json`

### 2. Run Complete Verification (Mock Data)

```bash
python examples/verify_earnings_framework.py
```

**What it does:**
1. Creates mock Kalshi contracts
2. Generates fake earnings call transcripts
3. Analyzes word mentions
4. Builds features & outcomes
5. Trains Beta-Binomial model
6. Runs backtest
7. Saves results to `data/verification/`

**Use this to:**
- Verify code changes don't break the pipeline
- Understand the complete workflow
- See expected data formats

### 3. Run Unit Tests

```bash
pytest tests/
```

---

## Code Reuse: FOMC → Earnings

**Reuse:** ~60% of code shared

| Component | Reuse % | Location |
|-----------|---------|----------|
| Kalshi Client | 100% | `fomc_analysis/kalshi_client_factory.py` |
| Beta-Binomial Math | 95% | `scipy.stats.beta` (same algorithm) |
| Backtest Engine | 90% | Adapted `backtester_v3.py` |
| Contract Analyzer | 75% | Added ticker parameter |
| Featurizer | 40% | Added speaker filtering |
| Data Fetchers | 20% | Different sources |

**What's New in Earnings:**
- Multi-ticker support (FOMC = single Fed)
- Speaker segmentation (CEO, CFO, analysts)
- Multi-word phrase handling ("VR / Virtual Reality")
- Binary detection (not count-based)

**See:** `docs/FOMC_VS_EARNINGS_COMPARISON.md` for detailed comparison

---

## Common Tasks

### Add a New Ticker

```python
# 1. Check if contracts exist
client = get_kalshi_client()
markets = client.get_markets(series_ticker="KXEARNINGSMENTIONAAPL")

# 2. If exists, create analyzer
from earnings_analysis.kalshi import EarningsContractAnalyzer
analyzer = EarningsContractAnalyzer(client, "AAPL")
contracts = analyzer.fetch_contracts(market_status="active")

# 3. Get transcripts (manual for now)
# 4. Run backtest using examples/verify_earnings_framework.py as template
```

### Update Model Parameters

```python
# In backtester initialization
backtester = EarningsKalshiBacktester(
    features=features_df,
    outcomes=outcomes_df,
    model_class=BetaBinomialEarningsModel,
    model_params={
        'alpha_prior': 1.0,  # Uniform prior
        'beta_prior': 1.0,
        'half_life': 8.0,  # Weight last N calls more (tune this!)
    },
    edge_threshold=0.12,  # Min edge to trade (tune this!)
    position_size_pct=0.03,  # 3% per trade
)
```

### Fetch Historical Outcomes

```python
# Get finalized contracts (for validation)
client = get_kalshi_client()

finalized = client.get_markets(
    series_ticker="KXEARNINGSMENTIONMETA",
    status="finalized"
)

for contract in finalized:
    word = contract['custom_strike']['Word']
    outcome = 1 if contract['last_price'] > 50 else 0  # >$0.50 = YES
    date = contract['ticker'].split('-')[1]  # Extract date
    print(f"{date}: {word} = {'YES' if outcome else 'NO'}")
```

---

## Data Formats

### Transcript Segments (JSONL)

```jsonl
{"speaker": "CEO", "role": "ceo", "text": "Our AI revenue grew 150%.", "segment_idx": 0}
{"speaker": "CFO", "role": "cfo", "text": "Margins improved to 42%.", "segment_idx": 1}
{"speaker": "Analyst", "role": "analyst", "text": "Question about cloud?", "segment_idx": 2}
```

**Required fields:**
- `speaker`: Name or role
- `role`: `ceo`, `cfo`, `executive`, `analyst`, `operator`
- `text`: Transcript segment
- `segment_idx`: Sequential index

### Features DataFrame

```python
# Index = call dates, Columns = word names, Values = counts
features_df = pd.DataFrame({
    'ai': [5, 3, 12, 8],
    'cloud': [2, 4, 6, 3],
    'revenue': [10, 8, 9, 11],
}, index=['2024-01-31', '2024-04-30', '2024-07-31', '2024-10-31'])
```

### Outcomes DataFrame

```python
# Index = call dates, Columns = word names, Values = 0/1
outcomes_df = pd.DataFrame({
    'ai': [1, 1, 1, 1],  # Always mentioned
    'cloud': [0, 1, 1, 1],  # Not mentioned in Q1
    'revenue': [1, 1, 1, 1],  # Always mentioned
}, index=['2024-01-31', '2024-04-30', '2024-07-31', '2024-10-31'])
```

---

## Debugging Tips

### Kalshi API Not Working?

```bash
# 1. Check env vars are set
printenv | grep KALSHI

# 2. Check for quote wrapping
printenv KALSHI_PRIVATE_KEY_BASE64 | head -c 10
# Should NOT start with a quote character

# 3. Test client directly
python -c "
from fomc_analysis.kalshi_client_factory import get_kalshi_client
client = get_kalshi_client()
markets = client.get_markets(series_ticker='KXEARNINGSMENTIONMETA', limit=1)
print(f'✅ Got {len(markets)} markets')
"
```

### Imports Not Working?

```bash
# 1. Check virtual environment is activated
which python  # Should show .venv/bin/python

# 2. Reinstall in editable mode
pip install -e .

# 3. Check PYTHONPATH (usually not needed with editable install)
echo $PYTHONPATH
```

### Word Detection Issues?

```python
# Test word matching logic
from earnings_analysis.features.featurizer import count_word_mentions

text = "We're excited about VR and Virtual Reality products."
count = count_word_mentions(text, "VR / Virtual Reality")
print(f"Mentions: {count}")  # Should be 2 (VR + Virtual Reality)
```

---

## Important Caveats

### 1. Binary Contracts Only

Current Kalshi contracts are **binary** (mentioned vs not):
- ✅ "Will META mention 'AI'?" (YES/NO)
- ❌ "Will META say 'AI' 5+ times?" (not available)

Framework can handle counts, but contracts don't require it.

### 2. Speaker Filtering

Kalshi rules: "If said by **any representative** (including operator)"

Our framework filters to executives only (CEO, CFO). This is **intentional** - we want to predict what execs say, not operators/analysts.

**Trade-off:**
- More conservative predictions
- May miss some edge cases where operator says word
- But cleaner signal

### 3. Multi-Word Phrases

Some contracts track phrases:
- "VR / Virtual Reality" (either counts)
- "FSD / Full Self Driving"
- "Supply Chain"

Handle with:
```python
# Split on '/' and check each variant
variants = word.split('/')
for variant in variants:
    if variant.strip().lower() in text.lower():
        return True
```

### 4. Market Efficiency

NVDA finalized contracts showed:
- **100% accuracy** for extreme probabilities (>90% or <10%)
- Market is very good at obvious cases
- **Edge likely exists** in mid-range (30-70%)

Don't fight the market on high-confidence predictions.

---

## Historical Context

### Original Implementation (FOMC)

Built for Federal Reserve press conferences:
- Single speaker (Fed Chair Powell)
- Single time series
- Phrase-based contracts ("inflation", "employment", etc.)

**See:** `README.md` for FOMC framework details

### Earnings Adaptation (Jan 2024)

Adapted FOMC framework for earnings calls:
- Multiple speakers (CEO, CFO, etc.)
- Multiple tickers (META, TSLA, NVDA, etc.)
- Word-based contracts ("AI", "Robotaxi", etc.)

**See:** `EARNINGS_FRAMEWORK_SUMMARY.md` for adaptation details

### Verification (Jan 24, 2026)

Comprehensive end-to-end testing:
- ✅ Mock data test successful
- ✅ Live Kalshi API integration verified
- ✅ 410 real contracts discovered
- ✅ Historical outcomes validated

**See:** `COMPLETE_VERIFICATION.md` for full results

---

## Next Steps for Development

### Immediate Priorities

1. **Fetch Historical Outcomes**
   ```python
   # Get all finalized contracts across tickers
   # Build ground truth dataset
   # Validate word detection accuracy
   ```

2. **Get Real Transcripts**
   ```python
   # Source: SEC EDGAR, Alpha Vantage, or manual
   # Format as JSONL with speaker roles
   # Validate speaker segmentation
   ```

3. **Run Real Backtest**
   ```python
   # Use historical transcripts + finalized outcomes
   # Train Beta-Binomial model
   # Calculate actual performance metrics
   ```

### Medium-Term Goals

1. **Parameter Tuning**
   - Optimize `half_life` (recency weighting)
   - Optimize `edge_threshold` (trade selectivity)
   - Optimize `position_size_pct` (risk management)

2. **Feature Engineering**
   - Add product lifecycle features
   - Add earnings momentum features
   - Add external signals (news, social media)

3. **Multi-Ticker Strategy**
   - Portfolio optimization
   - Correlation analysis
   - Risk-adjusted position sizing

### Long-Term Vision

1. **Automated Pipeline**
   - Auto-fetch transcripts on earnings day
   - Auto-generate predictions
   - Auto-execute trades (if validated)

2. **Live Monitoring**
   - Track prediction accuracy
   - Monitor market prices
   - Alert on high-edge opportunities

3. **Continuous Improvement**
   - Retrain models quarterly
   - A/B test model variants
   - Adapt to market evolution

---

## Reference Documentation

### Must-Read Files

1. **COMPLETE_VERIFICATION.md** - Full verification report
2. **docs/REAL_KALSHI_CONTRACTS_DISCOVERY.md** - Live API findings
3. **docs/EARNINGS_QUICKSTART.md** - User guide
4. **examples/verify_earnings_framework.py** - Working example

### Deep Dives

1. **docs/FOMC_VS_EARNINGS_COMPARISON.md** - Code reuse analysis
2. **docs/EARNINGS_VERIFICATION_REPORT.md** - Detailed code verification
3. **EARNINGS_FRAMEWORK_SUMMARY.md** - Framework adaptation notes

### Legacy (FOMC Framework)

1. **README.md** - Original FOMC framework
2. **E2E_BACKTEST_GUIDE.md** - FOMC backtest guide
3. **BACKTEST_V3_IMPLEMENTATION.md** - Backtest v3 design

---

## Final Notes for Agents

### When Working on This Project

1. **Always activate virtual environment first**
   ```bash
   source .venv/bin/activate
   ```

2. **Environment variables exist** - don't create .env
   ```bash
   printenv | grep KALSHI  # Verify before coding
   ```

3. **Client is synchronous** - don't use await
   ```python
   markets = client.get_markets(...)  # No await!
   ```

4. **Contracts are binary** - threshold always 1
   ```python
   outcome = 1 if mentioned else 0  # Not count-based!
   ```

5. **Test changes with verification script**
   ```bash
   python examples/verify_earnings_framework.py
   ```

### When Modifying Code

1. **Read relevant docs first** (see Reference Documentation above)
2. **Test locally before committing**
3. **Update this file if you change setup process**
4. **Keep examples/ working** (used for testing)

### When Stuck

1. Check `COMPLETE_VERIFICATION.md` for context
2. Run `examples/verify_earnings_framework.py` to see working example
3. Test Kalshi API with `scripts/explore_kalshi_earnings_contracts.py`
4. Review `docs/EARNINGS_QUICKSTART.md` for troubleshooting

---

**Last Updated:** 2026-01-24
**Agent:** Claude Sonnet 4.5
**Branch:** claude/verify-kalshi-framework-jJOzi
