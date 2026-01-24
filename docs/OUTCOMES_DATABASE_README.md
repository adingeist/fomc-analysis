# Kalshi Earnings Outcomes Database

## Overview

This directory contains historical ground truth data from finalized Kalshi earnings mention contracts. This database serves as the foundation for:

- **Validating** word detection accuracy
- **Running** real backtests against actual market outcomes
- **Tuning** model parameters
- **Measuring** actual edge vs market prices

## Database Files

| File | Format | Description | Size |
|------|--------|-------------|------|
| `outcomes.csv` | CSV | Easy-to-analyze tabular format | ~43 KB |
| `outcomes.json` | JSON | Full details with metadata | ~178 KB |
| `outcomes.db` | SQLite | Database for complex queries | ~196 KB |
| `summary.json` | JSON | Aggregated statistics | ~45 KB |

## Data Schema

### outcomes.csv

```csv
ticker,word,call_date,outcome,final_price,contract_ticker,settled_date
META,AI,2025-10-29,1,99,KXEARNINGSMENTIONMETA-25OCT29-AI,2025-10-29T23:32:00Z
TSLA,Robotaxi,2025-07-21,1,95,KXEARNINGSMENTIONTSLA-25JUL21-ROBO,2025-07-22T14:00:00Z
NVDA,Gaming,2025-08-26,1,99,KXEARNINGSMENTIONNVDA-25AUG26-GAME,2025-08-27T14:00:00Z
```

**Fields:**
- `ticker`: Company stock ticker (META, TSLA, NVDA, etc.)
- `word`: Word or phrase tracked by contract
- `call_date`: Earnings call date (YYYY-MM-DD)
- `outcome`: 1 = mentioned (YES), 0 = not mentioned (NO)
- `final_price`: Settlement price in cents (99 = YES, 1 = NO)
- `contract_ticker`: Full Kalshi contract ticker
- `settled_date`: When contract was settled (ISO 8601)

### outcomes.db (SQLite)

Table: `outcomes`

```sql
CREATE TABLE outcomes (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    word TEXT NOT NULL,
    call_date TEXT,
    outcome INTEGER NOT NULL,
    final_price INTEGER NOT NULL,
    contract_ticker TEXT UNIQUE NOT NULL,
    settled_date TEXT,
    series_ticker TEXT,
    status TEXT,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast lookups
CREATE INDEX idx_ticker ON outcomes(ticker);
CREATE INDEX idx_word ON outcomes(word);
CREATE INDEX idx_call_date ON outcomes(call_date);
CREATE INDEX idx_ticker_word ON outcomes(ticker, word);
```

## Summary Statistics

Based on initial fetch (2026-01-24):

### Overall
- **Total contracts**: 477
- **YES outcomes**: 288 (60.4%)
- **NO outcomes**: 189 (39.6%)
- **Date range**: 2025-07-16 to 2026-06-30
- **Unique tickers**: 8 (META, TSLA, NVDA, AAPL, MSFT, AMZN, GOOGL, NFLX)

### By Ticker

| Ticker | Total | YES | NO | YES % | Unique Words |
|--------|-------|-----|----|----|--------------|
| NVDA   | 75    | 45  | 30 | 60.0% | 62 |
| AMZN   | 61    | 36  | 25 | 59.0% | 47 |
| TSLA   | 61    | 37  | 24 | 60.7% | 45 |
| MSFT   | 58    | 37  | 21 | 63.8% | 36 |
| AAPL   | 57    | 39  | 18 | 68.4% | 47 |
| META   | 56    | 32  | 24 | 57.1% | 41 |
| NFLX   | 56    | 36  | 20 | 64.3% | 38 |
| GOOGL  | 53    | 26  | 27 | 49.1% | 34 |

### Perfect Predictors

Words with 100% YES or 100% NO outcomes (min 3 occurrences):

**100% YES (always mentioned):**
- Gaming (8 occurrences)
- Cloud (6)
- Autonomous (4)
- Robotics (4)
- AI / Artificial Intelligence (3)
- Azure (3)
- WhatsApp (3)
- Threads (3)

**100% NO (never mentioned):**
- Shutdown / Shut Down (6 occurrences)
- Trump (5)
- Antitrust (4)
- Cybersecurity (3)
- Subscriber (3)

### Most Common Words

| Word | Occurrences | YES | NO | Tickers |
|------|-------------|-----|----|----|
| China | 9 | 7 | 2 | META, TSLA, NVDA, AAPL, AMZN |
| Gaming | 8 | 8 | 0 | NVDA, MSFT, NFLX |
| Cloud | 6 | 6 | 0 | META, NVDA, MSFT, GOOGL |
| Shutdown / Shut Down | 6 | 0 | 6 | META, TSLA, AAPL, MSFT, AMZN, GOOGL |
| Tariff | 6 | 4 | 2 | TSLA, NVDA, AAPL, AMZN, NFLX |

## Building the Database

### Requirements

- Kalshi API credentials (environment variables)
- Virtual environment with dependencies installed

### Run Builder

```bash
# Activate virtual environment
source .venv/bin/activate

# Build database (fetches from Kalshi API)
python scripts/build_outcomes_database.py

# Verify database
python scripts/verify_outcomes_database.py
```

### Performance

- Uses **async batching** for parallel API requests
- **5-7x speedup** vs sequential fetching
- Fetches 8 tickers in ~10-15 seconds

### Output

```
data/outcomes_database/
├── outcomes.csv      # CSV format
├── outcomes.json     # JSON format
├── outcomes.db       # SQLite database
└── summary.json      # Statistics
```

## Usage Examples

### Python (Pandas)

```python
import pandas as pd

# Load CSV
df = pd.read_csv('data/outcomes_database/outcomes.csv')

# Analyze META outcomes
meta_df = df[df['ticker'] == 'META']
print(meta_df.groupby('outcome').size())

# Find perfect predictors
perfect_yes = df.groupby('word').filter(
    lambda x: len(x) >= 3 and x['outcome'].sum() == len(x)
)

# Mention rate by ticker
print(df.groupby('ticker')['outcome'].mean())
```

### Python (SQLite)

```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('data/outcomes_database/outcomes.db')

# Query with SQL
query = """
SELECT ticker, word, COUNT(*) as total, SUM(outcome) as mentions
FROM outcomes
GROUP BY ticker, word
HAVING total >= 3
ORDER BY mentions DESC
"""
df = pd.read_sql_query(query, conn)
```

### SQL Queries

```sql
-- Find all META outcomes
SELECT * FROM outcomes WHERE ticker='META';

-- Most mentioned words across all companies
SELECT word, SUM(outcome) as total_mentions
FROM outcomes
GROUP BY word
ORDER BY total_mentions DESC
LIMIT 10;

-- Words mentioned in multiple companies
SELECT word, COUNT(DISTINCT ticker) as num_companies
FROM outcomes
GROUP BY word
HAVING num_companies > 1
ORDER BY num_companies DESC;

-- Mention rate by quarter
SELECT
    strftime('%Y-Q', call_date) as quarter,
    COUNT(*) as total,
    SUM(outcome) as mentions,
    ROUND(100.0 * SUM(outcome) / COUNT(*), 1) as mention_pct
FROM outcomes
GROUP BY quarter
ORDER BY quarter;

-- Perfect predictors (100% YES or NO)
SELECT
    word,
    COUNT(*) as occurrences,
    SUM(outcome) as yes_count,
    CASE
        WHEN SUM(outcome) = COUNT(*) THEN 'ALWAYS YES'
        WHEN SUM(outcome) = 0 THEN 'ALWAYS NO'
    END as pattern
FROM outcomes
GROUP BY word
HAVING (SUM(outcome) = COUNT(*) OR SUM(outcome) = 0) AND COUNT(*) >= 3
ORDER BY occurrences DESC;
```

## Use Cases

### 1. Validate Word Detection

Compare model predictions against actual outcomes:

```python
# Load outcomes
outcomes_df = pd.read_csv('data/outcomes_database/outcomes.csv')

# Load model predictions
predictions_df = pd.read_csv('data/predictions.csv')

# Merge and compare
merged = outcomes_df.merge(
    predictions_df,
    on=['ticker', 'word', 'call_date']
)

# Calculate accuracy
accuracy = (merged['outcome'] == merged['predicted']).mean()
print(f"Accuracy: {accuracy:.1%}")
```

### 2. Run Backtest

Use outcomes as ground truth for walk-forward validation:

```python
from earnings_analysis.kalshi.backtester import EarningsKalshiBacktester

# Load outcomes
outcomes_df = pd.read_csv('data/outcomes_database/outcomes.csv')

# Convert to pivot table (dates x words)
outcomes_pivot = outcomes_df.pivot_table(
    index='call_date',
    columns='word',
    values='outcome'
)

# Run backtest
backtester = EarningsKalshiBacktester(
    features=features_df,
    outcomes=outcomes_pivot,
    model_class=BetaBinomialEarningsModel,
    edge_threshold=0.12,
)

results = backtester.run_backtest()
```

### 3. Tune Model Parameters

Optimize parameters using historical outcomes:

```python
# Test different half-life values
half_lives = [4.0, 6.0, 8.0, 10.0, 12.0]

results = []
for hl in half_lives:
    backtest = run_backtest(half_life=hl, outcomes=outcomes_df)
    results.append({
        'half_life': hl,
        'sharpe': backtest['sharpe_ratio'],
        'total_return': backtest['total_return'],
    })

best = max(results, key=lambda x: x['sharpe'])
print(f"Best half_life: {best['half_life']}")
```

### 4. Measure Market Efficiency

Compare model edge to actual profitability:

```python
# Load outcomes and historical prices
outcomes = load_outcomes()
prices = load_market_prices()  # From Kalshi API

# Calculate what we would have made
for _, row in outcomes.iterrows():
    ticker = row['contract_ticker']
    outcome = row['outcome']
    entry_price = prices[ticker]['entry']  # When we would have bought

    # Calculate P&L
    if outcome == 1:  # YES
        pnl = 99 - entry_price  # Contract settled at 99 cents
    else:  # NO
        pnl = -entry_price  # Lost entry price

    print(f"{ticker}: {pnl} cents")
```

## Refreshing the Database

The database contains finalized contracts as of the build date. To get newer outcomes:

```bash
# Re-run builder to fetch latest
python scripts/build_outcomes_database.py

# Check for new contracts
python scripts/verify_outcomes_database.py
```

**Note**: The builder automatically fetches all contracts with status `settled` (finalized) across all known tickers.

## Data Quality

### Validation

- All outcomes verified against Kalshi API
- Settlement prices validated (1 or 99 cents)
- Dates parsed and validated from contract tickers
- Duplicate contracts prevented (unique constraint on `contract_ticker`)

### Known Limitations

1. **Call dates** parsed from contract tickers (may not exactly match actual call time)
2. **Some contracts** have `call_date=None` if ticker format is non-standard
3. **Multi-word phrases** stored as-is (e.g., "VR / Virtual Reality")
4. **Only executives** considered in current framework (operators/analysts excluded)

### Data Integrity Checks

Run verification script to check database health:

```bash
python scripts/verify_outcomes_database.py
```

Checks performed:
- ✓ All outcomes are 0 or 1
- ✓ All final prices are valid (1-99 cents)
- ✓ No duplicate contract tickers
- ✓ Date ranges are reasonable
- ✓ All tickers are valid

## Contributing

To add new tickers or update the database:

1. Edit `build_outcomes_database.py`
2. Add ticker to `tickers` list
3. Re-run builder
4. Verify with verification script
5. Commit updated scripts (not data files)

## License

This database contains public market data from Kalshi. Use in accordance with Kalshi's Terms of Service.

## References

- [Kalshi API Documentation](https://trading-api.readme.io/reference/getmarkets)
- [FOMC Analysis Framework](../README.md)
- [Earnings Framework Guide](./EARNINGS_QUICKSTART.md)
- [Backtest Implementation](../BACKTEST_V3_IMPLEMENTATION.md)

---

**Last Updated**: 2026-01-24
**Database Version**: v1.0
**Total Contracts**: 477
**Date Range**: 2025-07-16 to 2026-06-30
