# FOMC Press Conference Analytics

A **reproducible, backtestable** toolkit for analyzing Federal Reserve Chair Jerome Powell's press conference transcripts and estimating mispricing in Kalshi "mention" event contracts.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Pipeline Stages](#pipeline-stages)
6. [Resolution Modes](#resolution-modes)
7. [AI Usage and Validation](#ai-usage-and-validation)
8. [Backtesting](#backtesting)
9. [Adding New Contracts](#adding-new-contracts)
10. [Data Artifacts](#data-artifacts)
11. [API Reference](#api-reference)
12. [Testing](#testing)
13. [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a **systematic, reproducible pipeline** for:

1. **Fetching** FOMC press conference PDFs from the Federal Reserve website
2. **Parsing** transcripts into speaker-segmented turns (deterministic + optional AI)
3. **Generating** phrase variants using OpenAI (with caching)
4. **Featurizing** transcripts to count contract mentions
5. **Training** baseline probability models with uncertainty estimates
6. **Backtesting** strategies using walk-forward validation (no lookahead)
7. **Reporting** mispricing opportunities
8. **Analyzing Kalshi contracts** - Fetch live contracts, generate variants, build historical statistics

**Main Objective:**
Given a Kalshi mention contract and market price at time `t`, produce:
- Probability estimate `p*` that contract resolves YES
- Confidence/uncertainty estimate
- Mispricing/edge estimate vs market implied probability
- Backtest showing whether this strategy has positive expected value

**We do not place real trades.** Paper/backtest only.

### ✨ NEW: Time-Horizon Backtest v3

The improved backtest system (v3) provides realistic performance evaluation:

- ✅ **Actual Kalshi Outcomes**: Uses real contract resolution data (100% YES / 0% NO)
- ✅ **Multi-Horizon Predictions**: Tests predictions at 7, 14, and 30 days before meetings
- ✅ **Accuracy Tracking**: Measures prediction accuracy for each time horizon
- ✅ **Realistic Trading**: Includes Kalshi's 7% fee on profits and proper position sizing
- ✅ **Comprehensive Metrics**: ROI, Sharpe ratio, Brier score, win rate per horizon

**Quick Start**:
```bash
# Run the complete workflow
bash examples/run_backtest_v3.sh

# Or run manually
fomc-analysis backtest-v3 \
  --contract-words data/kalshi_analysis/contract_words.json \
  --model beta \
  --horizons "7,14,30" \
  --output results/backtest_v3
```

📖 **See [docs/BACKTEST_V3_GUIDE.md](docs/BACKTEST_V3_GUIDE.md) for detailed documentation**

---

## Architecture

### Pipeline Stages

```
1. Ingestion       → Fetch PDFs from Fed website
2. Parsing         → PDF → raw text → clean text → speaker segments
3. Variant Generation → OpenAI-powered phrase variant expansion
4. Featurization   → Count mentions per contract per transcript
5. Modeling        → Train probability models (EWMA, Beta-Binomial)
6. Backtesting     → Walk-forward simulation with realistic execution
7. Reporting       → Mispricing analysis
```

### Two-Stage Parsing Pipeline

**Stage A: Deterministic Extraction**
- Use PyMuPDF to extract raw text from PDF
- Store: `data/raw_text/<date>.txt` (with page markers)
- Clean: normalize whitespace, fix hyphenation → `data/clean_text/<date>.txt`
- **Fully deterministic and reproducible**

**Stage B: Speaker Segmentation**
- Regex-based speaker turn detection (ALL CAPS patterns)
- Automatically classifies speakers into roles: `"powell"`, `"reporter"`, `"moderator"`, `"other"`
- Output: `data/segments/<date>.jsonl` with fields: `speaker`, `role`, `text`, `confidence`

### Variant Generation (Cached)

- Uses OpenAI API to generate phrase variants (plurals, possessives, compounds)
- Cache key: `hash(prompt_version + model + contract + base_phrases)`
- Stored in: `data/variants/<contract_slug>.json`
- **Deterministic ordering and deduping**

### No-Lookahead Guarantee

All models use **walk-forward validation**:
- At time `t`, model can ONLY use information available before `t`
- Market prices from `t-1` or earlier
- Training on events `[:t-1]`
- Prediction for event at time `t`

---

## Installation

### Prerequisites

- **Python 3.12+**
- **[`uv`](https://github.com/astral-sh/uv)** – fast Python package manager

Install `uv`:
```bash
pip install uv
```

### Clone and Install

```bash
git clone <repo-url> fomc-analysis
cd fomc-analysis
uv sync --extra dev
```

This creates a virtual environment and installs all dependencies.

### Environment Variables

Create a `.env` file:
```bash
cp .env.example .env
```

Add your OpenAI API key (required for variant generation and AI parsing):
```
OPENAI_API_KEY=sk-...
```

For Kalshi access, set either the legacy REST pair or the RSA SDK pair (preferred).
The API moved to the elections host, so the `KALSHI_BASE_URL` now defaults to
`https://api.elections.kalshi.com/trade-api/v2`. Override it only if Kalshi issues
another migration notice.

```
# Option A: Legacy REST (if your account still supports it)
KALSHI_API_KEY=...
KALSHI_API_SECRET=...

# Option B: RSA SDK (recommended)
KALSHI_API_KEY_ID=...
KALSHI_PRIVATE_KEY_BASE64=...
# Optional override
# KALSHI_BASE_URL=https://api.elections.kalshi.com/trade-api/v2
```

---

## Quick Start

Run the complete pipeline end-to-end:

### 1. Fetch Transcripts

```bash
uv run fomc fetch-transcripts --start-year 2020 --end-year 2025 --out-dir data/raw_pdf
```

### 2. Parse Transcripts

```bash
uv run fomc parse \
  --input-dir data/raw_pdf \
  --segments-dir data/segments
```

This uses deterministic parsing to extract speaker segments into JSONL format with proper role assignment (`"powell"`, `"reporter"`, `"moderator"`, `"other"`).

### 3. Analyze Historical Kalshi Contract Universe

Fetch every mention contract Kalshi has ever listed (open or resolved) and build
frequency stats across all transcripts. This step ensures we are training on the
exact keywords that have traded in the past.

```bash
uv run fomc analyze-kalshi-contracts \
  --series-ticker KXFEDMENTION \
  --segments-dir data/segments \
  --output-dir data/kalshi_analysis \
  --scope powell_only \
  --market-status resolved
```

Outputs:
- `data/kalshi_analysis/contract_words.json` – canonical word list + Kalshi tickers
- `data/kalshi_analysis/mention_summary.csv` – quant-level mention stats (see [Quant Analysis](docs/QUANT_ANALYSIS.md))
- `data/kalshi_analysis/mention_analysis.json` – full distribution per keyword

> **Auth Note:** The CLI now auto-detects either legacy REST creds
> (`KALSHI_API_KEY`/`KALSHI_API_SECRET`) or SDK creds
> (`KALSHI_API_KEY_ID`/`KALSHI_PRIVATE_KEY_BASE64`). Set whichever pair you have in `.env` before running.

### 4. Export Kalshi Contract Mapping (auto-sync keywords)

Automatically convert the fetched Kalshi markets into a YAML mapping that the
rest of the pipeline can consume:

```bash
uv run fomc export-kalshi-contracts \
  --series-ticker KXFEDMENTION \
  --market-status resolved \
  --output configs/generated_contract_mapping.yaml
```

This command de-duplicates words across events, optionally generates OpenAI
variants, and writes a mapping keyed by Kalshi contract names with thresholds
derived from the market title. Use this generated file in the next stages to
train on every traded keyword.

### 5. Generate Phrase Variants

```bash
uv run fomc build-variants \
  --contracts configs/generated_contract_mapping.yaml \
  --output-dir data/variants
```

### 6. Featurize Transcripts

```bash
uv run fomc featurize \
  --segments-dir data/segments \
  --contracts configs/generated_contract_mapping.yaml \
  --variants-dir data/variants \
  --speaker-mode powell_only \
  --phrase-mode variants \
  --output data/features.parquet
```

### 7. Train Model

```bash
uv run fomc train \
  --features data/features.parquet \
  --model beta \
  --alpha 1.0 \
  --beta-prior 1.0 \
  --half-life 4 \
  --output models/beta_model.json
```

### 8. Download Kalshi Prices

```bash
uv run fomc download-prices \
  --contract-words data/kalshi_analysis/contract_words.json \
  --output data/kalshi_analysis/prices.parquet
```

This fetches historical YES prices for each contract/meeting pair (using
`KALSHI_API_KEY`/`KALSHI_API_SECRET`) and pivots them to the wide format the
backtester expects. Use `--date-format iso` if your feature matrix index uses
`YYYY-MM-DD` instead of `YYYYMMDD`.

### 9. Backtest

```bash
uv run fomc backtest \
  --features data/features.parquet \
  --model beta \
  --edge-threshold 0.05 \
  --initial-capital 1000 \
  --prices data/kalshi_analysis/prices.parquet \
  --output results/backtest/
```

`--prices` expects a wide CSV/Parquet with index = meeting date (YYYY-MM-DD or
YYYYMMDD) and one column per contract (probability in 0-1 scale). Use
`uv run fomc download-prices` to build this dataset automatically from
`data/kalshi_analysis/contract_words.json`. If you omit `--prices`, the
backtester will still produce model metrics but will not simulate trades.

### 10. Generate Report

```bash
uv run fomc report \
  --results results/backtest/ \
  --output results/mispricing_table.csv
```

---

## Threshold Contracts Workflow

This section demonstrates the complete workflow for analyzing **count-threshold contracts** (e.g., "Inflation 40+ times", "Price 15+").

### Prerequisites

1. Transcripts have been fetched and parsed (steps 1-2 above)
2. Contract mapping includes threshold specifications (see [Adding New Contracts](#adding-new-contracts))

### Complete Workflow

**1. Generate Labels for All Four Modes**

```python
from pathlib import Path
from fomc_analysis.contract_mapping import load_mapping_from_file
from fomc_analysis.label_generator import (
    generate_labels_for_transcript,
    labels_to_dataframe,
    generate_label_matrix,
)
from fomc_analysis.parsing.speaker_segmenter import load_segments_jsonl

# Load contract specifications
mapping = load_mapping_from_file("configs/contract_mapping.yaml")

# Load segments for a transcript
segments = load_segments_jsonl("data/segments/20250115.jsonl")

# Generate labels for all four modes
labels = generate_labels_for_transcript(
    segments=segments,
    mapping=mapping,
    variants_dir=Path("data/variants"),
)

# Convert to DataFrame
df = labels_to_dataframe(labels)
print(df[["contract", "mode", "count", "threshold_hit", "debug_snippets"]])
```

**Output example:**
```
           contract                          mode  count  threshold_hit                debug_snippets
0    Inflation 40+  powell_only_strict_literal     42              1  'inflation': ...inflation expec...
1    Inflation 40+       powell_only_variants     44              1  'inflation': ...inflation expec...
2    Inflation 40+  full_transcript_strict...     45              1  'inflation': ...inflation expec...
3    Inflation 40+  full_transcript_variants     47              1  'inflation': ...inflation expec...
```

**2. Build Label Matrix for a Specific Mode**

```python
# Generate binary threshold_hit matrix for backtesting
events = generate_label_matrix(
    segments_dir=Path("data/segments"),
    mapping=mapping,
    variants_dir=Path("data/variants"),
    mode="powell_only_strict_literal",  # Choose one of four modes
)

# events is a DataFrame with rows=dates, columns=contracts, values=threshold_hit (0/1)
print(events.head())
```

**3. Train Model on Threshold Outcomes**

```python
from fomc_analysis.models import BetaBinomialModel

# Train on threshold_hit binary outcomes
model = BetaBinomialModel(alpha_prior=1.0, beta_prior=1.0, half_life=4)
model.fit(events)

# Predict P(count >= threshold) for each contract
predictions = model.predict()
print(predictions[["contract", "probability", "lower_bound", "upper_bound"]])
```

**4. Generate Mispricing Table**

```python
from fomc_analysis.mispricing import (
    compute_mispricing,
    create_mispricing_table,
    print_mispricing_report,
    load_market_prices_from_csv,
)

# Load Kalshi market prices (you need to provide this)
market_prices = load_market_prices_from_csv("data/kalshi_prices_20250115.csv")

# Compute mispricing
mispricing_results = compute_mispricing(
    model_predictions=predictions,
    market_prices=market_prices,
    edge_threshold=0.05,
)

# Create mispricing table
table = create_mispricing_table(mispricing_results, sort_by="abs_edge")

# Print formatted report
print_mispricing_report(table, top_n=5, timestamp="2025-01-15T10:00:00")
```

**Example output:**
```
================================================================================
MISPRICING REPORT
Timestamp: 2025-01-15T10:00:00
================================================================================

Total contracts analyzed: 18
YES recommendations: 5
NO recommendations: 3
PASS (no edge): 10

--------------------------------------------------------------------------------
TOP 5 YES OPPORTUNITIES (Buy YES - Model believes higher prob)
--------------------------------------------------------------------------------
        contract  model_prob  market_prob   edge  confidence
   Inflation 40+        0.85         0.65   0.20        0.82
      Price 15+        0.72         0.42   0.30        0.75
  Unemployment 8+        0.68         0.58   0.10        0.88
      Growth 8+        0.55         0.48   0.07        0.79

--------------------------------------------------------------------------------
TOP 5 NO OPPORTUNITIES (Buy NO - Model believes lower prob)
--------------------------------------------------------------------------------
        contract  model_prob  market_prob  edge_magnitude  confidence
      Tariff 5+        0.25         0.55            0.30        0.81
         Cut 7+        0.40         0.65            0.25        0.77
```

**5. Run Walk-Forward Backtest**

```python
from fomc_analysis.backtester_v2 import WalkForwardBacktester

# Create backtester with threshold_hit events
backtester = WalkForwardBacktester(
    events=events,  # threshold_hit matrix from step 2
    prices=market_prices,
    edge_threshold=0.05,
    position_size_pct=0.02,
    fee_rate=0.07,
    min_train_window=5,
)

# Run backtest
result = backtester.run(
    model_class=BetaBinomialModel,
    model_params={"alpha_prior": 1.0, "beta_prior": 1.0, "half_life": 4},
    initial_capital=1000.0,
)

# Print metrics
print("ROI:", result.metrics["roi"])
print("Sharpe:", result.metrics["sharpe"])
print("Win Rate:", result.metrics["win_rate"])
print("Brier Score:", result.metrics["brier_score"])
```

**6. Analyze Results by Contract**

```python
import pandas as pd

trades_df = pd.DataFrame([asdict(t) for t in result.trades])

# Group by contract
contract_performance = trades_df.groupby("contract").agg({
    "pnl": ["count", "sum", "mean"],
    "edge": "mean",
}).round(3)

print(contract_performance)
```

### Mode Comparison Analysis

Compare performance across all four resolution modes:

```python
# Generate events for each mode
modes = [
    "powell_only_strict_literal",
    "powell_only_variants",
    "full_transcript_strict_literal",
    "full_transcript_variants",
]

results_by_mode = {}

for mode in modes:
    # Generate events
    events = generate_label_matrix(
        segments_dir=Path("data/segments"),
        mapping=mapping,
        variants_dir=Path("data/variants"),
        mode=mode,
    )

    # Run backtest
    backtester = WalkForwardBacktester(events, prices, edge_threshold=0.05)
    result = backtester.run(BetaBinomialModel, initial_capital=1000.0)

    results_by_mode[mode] = result.metrics

# Compare
comparison = pd.DataFrame(results_by_mode).T
print(comparison[["roi", "sharpe", "win_rate", "brier_score"]])
```

This helps identify which resolution mode (speaker scope + match mode) best aligns with actual Kalshi contract outcomes.

---

## Pipeline Stages

### 1. Fetch Transcripts

**Command:**
```bash
fomc fetch-transcripts --start-year <year> --end-year <year> --out-dir <dir>
```

**What it does:**
- Scrapes Fed historical year pages
- Finds press conference meeting pages
- Downloads PDF transcripts
- Creates index CSV with metadata

**Output:**
- `data/raw_pdf/<date>.pdf`
- `data/pressconf_index.csv`

---

### 2. Parse Transcripts

**Command:**
```bash
fomc parse \
  --input-dir <pdf-dir> \
  --segments-dir <output-dir>
```

Uses deterministic regex-based parsing to extract speaker segments with role classification.

**Output:**
- `data/raw_text/<date>.txt` – Raw extracted text with page markers
- `data/clean_text/<date>.txt` – Cleaned, normalized text
- `data/segments/<date>.jsonl` – Speaker turns with metadata

**Segment Format (JSONL):**
```json
{"speaker": "CHAIR POWELL", "role": "powell", "text": "...", "confidence": 1.0}
{"speaker": "MR. SMITH", "role": "reporter", "text": "...", "confidence": 1.0}
```

---

### 3. Build Variants

**Command:**
```bash
fomc build-variants \
  --contracts <mapping.yaml> \
  --output-dir <variants-dir> \
  --force
```

**What it does:**
- Reads contract mapping (base phrases)
- Calls OpenAI API to generate variants (plurals, possessives, compounds)
- Caches results with hash-based key
- Skips contracts that already have cached variants (unless `--force`)

**Variant Generation Rules:**
- ✅ Included: Plural forms, possessive forms, compound words (hyphenated)
- ❌ Excluded: Synonyms, tense inflections, homophones

**Output:**
- `data/variants/<contract_slug>_<hash>.json`

**Cache Key:**
```
SHA256(prompt_version + model + contract + sorted_base_phrases)
```

---

### 4. Featurize

**Command:**
```bash
fomc featurize \
  --segments-dir <segments> \
  --contracts <mapping.yaml> \
  --variants-dir <variants> \
  --speaker-mode <powell_only|full_transcript> \
  --phrase-mode <strict|variants> \
  --output <features.parquet>
```

**Parameters:**

- `--speaker-mode`:
  - `powell_only`: Only count mentions in Chair Powell's remarks
  - `full_transcript`: Count mentions from all speakers (reporters included)

- `--phrase-mode`:
  - `strict`: Use only base phrases from mapping file
  - `variants`: Use AI-generated variants

**Output:**
- `data/features.parquet` – Parquet file with columns:
  - Index: transcript date
  - Columns: `<contract>_mentioned` (binary), `<contract>_count` (integer)

---

### 5. Train Model

**Command:**
```bash
fomc train \
  --features <features.parquet> \
  --model <ewma|beta> \
  --alpha <float> \
  --output <model.json>
```

**Models:**

**EWMA (Exponentially Weighted Moving Average):**
- Simple recency-weighted baseline
- `--alpha`: Smoothing parameter (0-1, higher = more weight on recent)
- Uncertainty via bootstrap

**Beta-Binomial (Bayesian):**
- Beta prior updated with observed events
- `--alpha-prior`, `--beta-prior`: Prior hyperparameters
- `--half-life`: Exponential decay (optional, in number of events)
- Uncertainty via posterior credible intervals

**Output:**
- `models/<model>.json` – Trained model with parameters and training data

**Predictions include:**
- `probability`: Point estimate
- `lower_bound`: Lower confidence/credible bound
- `upper_bound`: Upper confidence/credible bound
- `uncertainty`: Standard deviation or interval width

---

### 6. Backtest

**Command:**
```bash
fomc backtest \
  --features <features.parquet> \
  --prices <prices.csv> \
  --model <ewma|beta> \
  --edge-threshold <float> \
  --initial-capital <float> \
  --output <results-dir>
```

**Walk-Forward Backtesting:**

For each event at time `t`:
1. Train model on events `[:t-1]`
2. Predict probability for event at `t`
3. Compare with market price (from `t-1`)
4. If `|model_prob - market_prob| > edge_threshold`, trade
5. Observe outcome at `t`
6. Update capital with P&L (including 7% fees)

**Outputs:**
- `results/backtest/backtest_results.json` – Full trade log
- `results/backtest/equity_curve.csv` – Capital over time

**Metrics:**
- `roi`: Return on investment
- `sharpe`: Sharpe ratio (annualized)
- `sortino`: Sortino ratio
- `max_drawdown`: Maximum drawdown
- `win_rate`: Fraction of profitable trades
- `brier_score`: Calibration metric (lower = better)

---

### 7. Report

**Command:**
```bash
fomc report \
  --results <backtest-dir> \
  --output <mispricing_table.csv>
```

**What it does:**
- Aggregates backtest trades by contract
- Computes win rate, total P&L, average edge per contract
- Ranks contracts by profitability

**Output:**
- `results/mispricing_table.csv`:
  - Columns: `contract`, `trades`, `win_rate`, `total_pnl`, `avg_edge`

---

## Resolution Modes

Kalshi mention contracts have specific resolution criteria. This toolkit supports multiple resolution modes and contract types:

### Contract Types

**Binary Mention Contracts (threshold = 1):**
- Resolve YES if the term is mentioned at least once
- Examples: "Good Afternoon", "AI / Artificial Intelligence", "Crypto / Bitcoin"
- Default behavior if no threshold is specified

**Count-Threshold Contracts (threshold ≥ N):**
- Resolve YES only if the term appears ≥ N times
- Examples: "Inflation 40+ times", "Price 15+", "Tariff 5+", "Cut 7+"
- Requires explicit `threshold` configuration in contract spec

### 1. Speaker Mode (Scope)

**`powell_only`** (recommended for most contracts):
- Count mentions ONLY in Chair Powell's remarks
- Excludes reporters, moderators, other speakers
- Most Kalshi contracts resolve on Powell's words only

**`full_transcript`**:
- Count mentions from ALL speakers
- Use only if contract explicitly includes reporter questions
- Example: "Tariff 5+" contract with full transcript scope

### 2. Phrase Mode (Match Mode)

**`strict_literal`**:
- Use only base phrases from `contract_mapping.yaml`
- Conservative matching with word boundaries
- Avoids partial matches (e.g., "inflation" won't match "inflationary")

**`variants`**:
- Use AI-generated phrase variants
- Includes plurals, possessives, compound forms
- More comprehensive but requires variant generation
- Example: "price" → ["price", "prices", "pricing"]

### 3. Count Unit

**`token`** (default):
- Count individual word occurrences
- Each separate mention counts toward threshold
- Example: "inflation" mentioned 40 times → count = 40

**`phrase`**:
- Count complete phrase occurrences
- For multi-word phrases like "balance of risks"

### Four-Mode Label Generation

For each contract, the system can compute labels under all four combinations:
1. `powell_only` + `strict_literal`
2. `powell_only` + `variants`
3. `full_transcript` + `strict_literal`
4. `full_transcript` + `variants`

This enables analysis of which resolution mode most closely aligns with actual Kalshi outcomes.

### Alignment with Kalshi Rules

See `src/fomc_analysis/prompts/MENTIONS_CONTRACT_RULES.md` for full Kalshi resolution rules.

**Included:**
- Plural and possessive forms (e.g., "immigrant" → "immigrants", "immigrant's")
- Compound words (e.g., "AI" → "AI-powered", "AI technology")
- Homonyms and homographs

**Excluded:**
- Synonyms (e.g., "AI" ≠ "machine learning")
- Tense inflections (e.g., "immigrant" ≠ "immigration")
- Homophones (e.g., "write" ≠ "right")

---

## AI Usage and Validation

This toolkit uses OpenAI's API for phrase variant generation:

### Phrase Variant Generation (Required for `variants` mode)

**When:** Running `build-variants` command

**Prompt:** Template in `src/fomc_analysis/variants/generator.py`

**Caching:**
- Cache key: `SHA256(prompt_version + model + contract + base_phrases)`
- Stored in `data/variants/`
- Regenerate only when:
  - Base phrases change
  - Prompt version changes
  - Model changes
  - `--force` flag is used

**Output:**
- JSON file with `contract`, `base_phrases`, `variants`, `metadata`, `cache_key`

---

## Backtesting

### No-Lookahead Guarantee

**Critical Rule:** At time `t`, model can ONLY use information available before `t`.

**Implementation:**
1. Events are sorted chronologically
2. Training window: `events[:t]` (strictly before current date)
3. Market price: from `t-1` or earlier
4. Prediction: for event at time `t`
5. Outcome: observed at time `t`

**Walk-Forward Loop:**
```python
for t in range(min_train_window, len(events)):
    train_events = events[:t]  # Before t
    model.fit(train_events)
    prediction = model.predict()
    market_price = prices[t-1]  # Before t
    if |prediction - market_price| > threshold:
        trade()
    outcome = events[t]  # Observe outcome
    update_capital(outcome)
```

### Realistic Execution

- **Bid/Ask:** Trade at market price (no favorable fills)
- **Fees:** 7% on profits (Kalshi standard)
- **Position Sizing:** Fixed fraction of capital (default 2%)
- **No Leverage:** Cannot trade more than available capital

### Metrics

**Brier Score:**
- Measures calibration: `mean((prob - outcome)^2)`
- Lower is better (perfectly calibrated = 0)

**Sharpe Ratio:**
- Risk-adjusted return: `mean(returns) / std(returns) * sqrt(n_events_per_year)`
- Annualized assuming ~8 FOMC events per year

**Max Drawdown:**
- Largest peak-to-trough decline in equity

---

## Adding New Contracts

### 1. Update Contract Mapping

Edit `configs/contract_mapping.yaml`:

**Binary Mention Contract (threshold = 1, default):**
```yaml
"New Contract Name":
  synonyms:
    - new phrase
    - another variant
  threshold: 1  # Optional, defaults to 1
  scope: powell_only  # Optional, defaults to "powell_only"
  match_mode: strict_literal  # Optional, defaults to "strict_literal"
  description: Human-readable description of contract
```

**Count-Threshold Contract (e.g., "Inflation 40+ times"):**
```yaml
"Inflation 40+":
  synonyms:
    - inflation
  threshold: 40  # Resolves YES only if count >= 40
  scope: powell_only
  match_mode: strict_literal
  count_unit: token  # Optional, defaults to "token"
  description: Powell mentions "inflation" at least 40 times
```

**Multi-phrase Variant Contract:**
```yaml
"Price 15+":
  synonyms:
    - price
    - prices
    - pricing
  threshold: 15
  scope: powell_only
  match_mode: variants  # Use AI-generated variants
  description: Powell mentions price/prices/pricing at least 15 times
```

**Full Transcript Contract:**
```yaml
"Tariff 5+":
  synonyms:
    - tariff
    - tariffs
  threshold: 5
  scope: full_transcript  # Count all speakers, not just Powell
  match_mode: variants
  description: Tariff/tariffs mentioned at least 5 times in full transcript
```

### Contract Configuration Fields

- **`synonyms`** (required): List of lowercase phrase variants to match
- **`threshold`** (optional, default=1): Minimum count for contract to resolve YES
- **`scope`** (optional, default="powell_only"): Speaker filter
  - `powell_only`: Count only Powell's remarks
  - `full_transcript`: Count all speakers
- **`match_mode`** (optional, default="strict_literal"): How to match phrases
  - `strict_literal`: Use only base synonyms with word boundaries
  - `variants`: Use AI-generated variants
- **`count_unit`** (optional, default="token"): What to count
  - `token`: Individual word occurrences
  - `phrase`: Complete phrase occurrences
- **`description`** (optional): Human-readable description

### Ambiguous Terms and Word Boundaries

The system uses **word-boundary matching** to avoid false positives:

- ✅ "inflation" matches "inflation" (exact word)
- ❌ "inflation" does NOT match "inflationary" (different word)
- ✅ "cut" matches "cut" but NOT "cutting" or "cutback"
- ✅ "price" + "prices" as variants counts both separately

**Best practices for ambiguous terms:**
- For "cut" contracts: Use `["cut", "cuts"]` as synonyms to avoid "cutting"
- For "price" contracts: Use `["price", "prices", "pricing"]` as variants
- For "growth" contracts: Use `["growth"]` only (not "grow", "growing")

### 2. Generate Variants

```bash
uv run fomc build-variants \
  --contracts configs/contract_mapping.yaml \
  --output-dir data/variants
```

### 3. Re-featurize

```bash
uv run fomc featurize \
  --segments-dir data/segments \
  --contracts configs/contract_mapping.yaml \
  --variants-dir data/variants \
  --speaker-mode powell_only \
  --phrase-mode variants \
  --output data/features_updated.parquet
```

### 4. Retrain and Backtest

```bash
uv run fomc train --features data/features_updated.parquet ...
uv run fomc backtest --features data/features_updated.parquet ...
```

---

## Data Artifacts

All intermediate artifacts are cached for reproducibility:

```
data/
├── raw_pdf/              # Original PDFs from Fed website
├── raw_text/             # Raw extracted text (with page markers)
├── clean_text/           # Cleaned text (normalized whitespace)
├── segments/             # Speaker-segmented JSONL files
├── variants/             # Cached AI-generated phrase variants
├── features.parquet      # Feature matrix (binary events + counts)
├── prices.csv            # Historical Kalshi market prices
└── pressconf_index.csv   # Index of fetched transcripts

models/
└── <model>.json          # Trained models with parameters

results/
├── backtest/
│   ├── backtest_results.json  # Full trade log
│   └── equity_curve.csv       # Equity over time
└── mispricing_table.csv       # Contract-level profitability
```

### Versioning and Hashing

- **Variants:** Cache key includes prompt version, model, base phrases
- **Features:** Deterministic given segments and contract mapping
- **Models:** Include training data hash in metadata (TODO)

---

## API Reference

### CLI Commands

```bash
fomc fetch-transcripts [OPTIONS]
fomc parse [OPTIONS]
fomc build-variants [OPTIONS]
fomc featurize [OPTIONS]
fomc train [OPTIONS]
fomc backtest [OPTIONS]
fomc report [OPTIONS]
fomc analyze-kalshi-contracts [OPTIONS]
fomc export-kalshi-contracts [OPTIONS]   # NEW: Auto-build contract mapping from Kalshi series
```

Run `fomc <command> --help` for detailed options.

**New in this release:** `analyze-kalshi-contracts` command fetches Kalshi mention contracts, generates word variants using OpenAI, and builds statistical analysis of historical mention frequencies. See [docs/KALSHI_ANALYSIS.md](docs/KALSHI_ANALYSIS.md) for details.

### Python API

**Parsing:**
```python
from fomc_analysis.parsing import extract_pdf_to_text, clean_text, segment_speakers

raw = extract_pdf_to_text(pdf_path)
cleaned = clean_text(raw)
segments = segment_speakers(cleaned, use_ai=False)
```

**Featurization:**
```python
from fomc_analysis.featurizer import build_feature_matrix, FeatureConfig

config = FeatureConfig(speaker_mode="powell_only", phrase_mode="strict")
features = build_feature_matrix(segments_dir, contracts, config)
```

**Modeling:**
```python
from fomc_analysis.models import BetaBinomialModel

model = BetaBinomialModel(alpha_prior=1.0, beta_prior=1.0, half_life=4)
model.fit(events)
predictions = model.predict()
```

**Backtesting:**
```python
from fomc_analysis.backtester_v2 import WalkForwardBacktester

bt = WalkForwardBacktester(events, prices, edge_threshold=0.05)
result = bt.run(model_class=BetaBinomialModel, initial_capital=1000)
```

---

## Testing

Run all tests:
```bash
uv run pytest
```

Run specific test file:
```bash
uv run pytest tests/test_parsing.py
```

Run with coverage:
```bash
uv run pytest --cov=fomc_analysis --cov-report=html
```

### Test Coverage

- ✅ PDF extraction and text cleaning
- ✅ Speaker segmentation (deterministic + validation)
- ✅ Phrase matching (strict + word boundaries)
- ✅ Feature extraction (powell_only vs full)
- ✅ Model training and predictions
- ✅ Backtester no-lookahead guarantees
- ✅ Variant caching stability

---

## Troubleshooting

### OpenAI API Errors

**Error:** `OPENAI_API_KEY not found`
**Fix:** Add `OPENAI_API_KEY=sk-...` to `.env` file

**Error:** Rate limit exceeded
**Fix:** Add retry logic or reduce concurrency (variant generation batches)

### Parsing Issues

**Problem:** Segmentation misses some speakers
**Solution:** The deterministic parser uses regex patterns for common speaker labels. Check the segment files to verify speaker roles are correctly assigned (`"powell"`, `"reporter"`, etc.).

### Backtesting

**Problem:** No trades executed
**Fix:** Lower `--edge-threshold` or check that prices align with features

**Problem:** Negative Sharpe ratio
**Fix:** Model may not have edge. Try different parameters or longer training window.

### Dependencies

**Problem:** `uv sync` fails
**Fix:** Ensure Python 3.12+ is installed. Try `uv sync --no-cache`

### Kalshi API Migration

**Problem:** `API has been moved to https://api.elections.kalshi.com/`
**Fix:** Upgrade to this version of the toolkit (default base URL already updated) or set `KALSHI_BASE_URL=https://api.elections.kalshi.com/trade-api/v2` in `.env`. Make sure you're using RSA credentials (`KALSHI_API_KEY_ID` + `KALSHI_PRIVATE_KEY_BASE64`).

---

## Kalshi SDK Authentication (API Keys)

Kalshi's Python SDK requires signing requests with your RSA private key. This
project includes a helper that reads a private key from a file created from
`KALSHI_PRIVATE_KEY_BASE64`, then initializes the SDK client with your key ID.

Set the following environment variables:

```bash
export KALSHI_API_KEY_ID="your_key_id"
export KALSHI_PRIVATE_KEY_BASE64="base64_encoded_private_key"
```

Then run the connectivity check:

```bash
python -m fomc_analysis.kalshi_sdk
```

If authentication succeeds, it prints the current exchange status.

---

## Contributions

This is a research/educational toolkit. Contributions welcome:
- Bug fixes
- New models (e.g., logistic regression, gradient boosting)
- Additional resolution rule support
- Improved validation metrics

---

## License

MIT License. See `LICENSE` file.

---

## Disclaimer

This toolkit is for **educational and research purposes only**. It does not provide financial advice. Always understand the rules of the market you are trading, ensure that your code replicates the resolution criteria precisely, and backtest thoroughly before risking capital.

The Federal Reserve and Kalshi are not affiliated with this project.

---

## Acknowledgments

- Federal Reserve for providing public press conference transcripts
- Kalshi for market data (if using)
- OpenAI for language models used in variant generation

---

**Happy Hacking! 🚀**

For questions or issues, please open a GitHub issue.
