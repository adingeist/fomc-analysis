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

**Main Objective:**
Given a Kalshi mention contract and market price at time `t`, produce:
- Probability estimate `p*` that contract resolves YES
- Confidence/uncertainty estimate
- Mispricing/edge estimate vs market implied probability
- Backtest showing whether this strategy has positive expected value

**We do not place real trades.** Paper/backtest only.

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
- **Deterministic mode**: Regex-based speaker turn detection (ALL CAPS patterns)
- **AI mode** (optional): OpenAI-powered segmentation with **validation**
  - If AI output fails validation (text similarity < 95%), fall back to deterministic
  - Validation ensures concatenated segments match cleaned text
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
  --mode deterministic \
  --segments-dir data/segments
```

For AI-assisted parsing (requires OpenAI API key):
```bash
uv run fomc parse \
  --input-dir data/raw_pdf \
  --mode ai \
  --segments-dir data/segments
```

### 3. Generate Phrase Variants

```bash
uv run fomc build-variants \
  --contracts configs/contract_mapping.yaml \
  --output-dir data/variants
```

### 4. Featurize Transcripts

```bash
uv run fomc featurize \
  --segments-dir data/segments \
  --contracts configs/contract_mapping.yaml \
  --variants-dir data/variants \
  --speaker-mode powell_only \
  --phrase-mode variants \
  --output data/features.parquet
```

### 5. Train Model

```bash
uv run fomc train \
  --features data/features.parquet \
  --model beta \
  --alpha 1.0 \
  --beta-prior 1.0 \
  --half-life 4 \
  --output models/beta_model.json
```

### 6. Backtest

```bash
uv run fomc backtest \
  --features data/features.parquet \
  --prices data/prices.csv \
  --model beta \
  --edge-threshold 0.05 \
  --initial-capital 1000 \
  --output results/backtest/
```

### 7. Generate Report

```bash
uv run fomc report \
  --results results/backtest/ \
  --output results/mispricing_table.csv
```

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
  --mode <deterministic|ai> \
  --segments-dir <output-dir>
```

**Modes:**
- `deterministic`: Regex-based speaker segmentation (fast, no API costs)
- `ai`: OpenAI-assisted segmentation with validation fallback

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

Kalshi mention contracts have specific resolution criteria. This toolkit supports two critical toggles:

### 1. Speaker Mode

**`powell_only`** (recommended for most contracts):
- Count mentions ONLY in Chair Powell's remarks
- Excludes reporters, moderators, other speakers
- Most Kalshi contracts resolve on Powell's words only

**`full_transcript`**:
- Count mentions from ALL speakers
- Use only if contract explicitly includes reporter questions

### 2. Phrase Mode

**`strict`**:
- Use only base phrases from `contract_mapping.yaml`
- Conservative matching

**`variants`**:
- Use AI-generated phrase variants
- Includes plurals, possessives, compound forms
- More comprehensive but requires variant generation

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

This toolkit uses OpenAI's API in two places:

### 1. Speaker Segmentation (Optional)

**When:** Using `--mode ai` in `parse` command

**Prompt:** Asks GPT to segment transcript into speaker turns

**Validation:**
- Concatenate AI-generated segments
- Compare with deterministic cleaned text
- Check similarity (must be ≥ 95%)
- Check coverage (must be ≥ 90%)
- **If validation fails:** fall back to deterministic segmentation

**Why use AI mode?**
- Can handle unusual transcript formatting
- Better at identifying ambiguous speaker labels
- Always validated against ground truth

### 2. Phrase Variant Generation (Required for `variants` mode)

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

```yaml
"New Contract Name":
  synonyms:
    - new phrase
    - another variant
  description: Human-readable description of contract
```

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
```

Run `fomc <command> --help` for detailed options.

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
**Solution:** Use `--mode ai` for AI-assisted segmentation

**Problem:** AI segmentation fails validation
**Fix:** System automatically falls back to deterministic mode. Check logs for warnings.

### Backtesting

**Problem:** No trades executed
**Fix:** Lower `--edge-threshold` or check that prices align with features

**Problem:** Negative Sharpe ratio
**Fix:** Model may not have edge. Try different parameters or longer training window.

### Dependencies

**Problem:** `uv sync` fails
**Fix:** Ensure Python 3.12+ is installed. Try `uv sync --no-cache`

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
