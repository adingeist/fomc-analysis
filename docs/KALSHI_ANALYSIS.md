# Kalshi Contract Analysis

This document explains how to use the Kalshi contract analysis feature to analyze FOMC mention contracts and build statistical data on historical word frequencies.

## Overview

The Kalshi contract analysis pipeline:

1. **Fetches contracts** from the Kalshi API (e.g., KXFEDMENTION series)
2. **Extracts tracked words** from market titles
3. **Generates word variants** using OpenAI (plurals, possessives, compounds)
4. **Scans FOMC transcripts** for matches using word-boundary matching
5. **Builds statistical analysis** of historical mention frequencies

## Setup

### Prerequisites

1. **Kalshi API credentials**: You need an API key and secret from Kalshi
2. **OpenAI API key**: For generating word variants
3. **FOMC transcripts**: Must be fetched and parsed first

### Environment Variables

Add the following to your `.env` file:

```bash
# Kalshi API (legacy auth)
KALSHI_API_KEY=your_api_key_here
KALSHI_API_SECRET=your_api_secret_here

# OpenAI API
OPENAI_API_KEY=your_openai_key_here
```

## Usage

### Step 1: Fetch and Parse FOMC Transcripts

Before running the analysis, you need to have parsed FOMC transcripts:

```bash
# Fetch transcripts
fomc fetch-transcripts --start-year 2011 --end-year 2025

# Parse transcripts
fomc parse \
  --input-dir data/raw_pdf \
  --output-dir data/segments \
  --mode deterministic
```

### Step 2: Run Kalshi Contract Analysis

```bash
fomc analyze-kalshi-contracts \
  --series-ticker KXFEDMENTION \
  --segments-dir data/segments \
  --output-dir data/kalshi_analysis \
  --scope powell_only
```

#### Options

- `--series-ticker`: Kalshi series ticker (default: `KXFEDMENTION`)
- `--event-ticker`: Specific event (e.g., `kxfedmention-26jan`). If not provided, analyzes all markets in series
- `--segments-dir`: Directory with parsed transcripts (default: `data/segments`)
- `--output-dir`: Output directory (default: `data/kalshi_analysis`)
- `--scope`: Search scope - `powell_only` (default) or `full_transcript`

### Step 3: Review Results

The analysis creates three output files in `data/kalshi_analysis/`:

1. **`contract_words.json`**: List of contract words with generated variants
   ```json
   [
     {
       "word": "Layoff",
       "market_ticker": "KXFEDMENTION-26JAN-LAY",
       "market_title": "Layoff mention",
       "variants": ["layoff", "layoffs", "layoff's"]
     }
   ]
   ```

2. **`mention_analysis.json`**: Detailed statistical analysis for each word
   ```json
   [
     {
       "word": "Layoff",
       "variants": ["layoff", "layoffs", "layoff's"],
       "total_transcripts": 50,
       "transcripts_with_mention": 12,
       "mention_frequency": 0.24,
       "total_mentions": 34,
       "avg_mentions_per_transcript": 0.68,
       "max_mentions_in_transcript": 5,
       "mention_counts_distribution": {
         "0": 38,
         "1": 7,
         "2": 3,
         "5": 2
       }
     }
   ]
   ```

3. **`mention_summary.csv`**: Summary table sorted by mention frequency
   ```
   Word,Variants Count,Total Transcripts,Transcripts with Mention,Mention Frequency,Total Mentions,Avg Mentions/Transcript,Max Mentions
   Inflation,3,50,45,90.00%,2450,49.00,73
   Median,2,50,35,70.00%,58,1.16,4
   Layoff,3,50,12,24.00%,34,0.68,5
   ```

## How It Works

### 1. Contract Fetching

The analyzer uses the Kalshi API to fetch market data:

```python
# Fetch all markets in KXFEDMENTION series
markets = kalshi_client.get_markets(series_ticker="KXFEDMENTION")

# Or fetch specific event
event_data = kalshi_client.get_event("kxfedmention-26jan", with_nested_markets=True)
```

### 2. Word Extraction

Market titles are parsed to extract the tracked word:

- `"President"` → `"President"`
- `"Layoff mention"` → `"Layoff"`
- `"AI / Artificial Intelligence"` → `"AI / Artificial Intelligence"`
- `"Good Afternoon mention"` → `"Good Afternoon"`

### 3. Variant Generation

Uses OpenAI to generate variants following Kalshi resolution rules:

**Included:**
- Plurals: `layoff` → `layoffs`
- Possessives: `layoff` → `layoff's`
- Compounds: `AI` → `AI-powered`, `AI technology`
- Case variations: `president`, `President`

**Excluded:**
- Synonyms: `AI` ≠ `machine learning`
- Tense inflections: `layoff` ≠ `laying off`
- Homophones: different spelling, same sound

### 4. Transcript Scanning

Scans transcripts using word-boundary matching to avoid false positives:

```python
# Filter for Powell's statements only
if scope == "powell_only":
    text_parts = [
        seg["text"] for seg in segments
        if seg.get("role", "").lower() == "powell"
    ]

# Uses regex word boundaries
pattern = r'\b' + re.escape(variant_lower) + r'\b'

# This matches:
"layoff" in "discussing layoffs today"  ✓

# But not:
"layoff" in "playoff season"  ✗
```

### 5. Statistical Analysis

Computes comprehensive statistics:

- **Mention frequency**: Proportion of transcripts mentioning the word
- **Total mentions**: Sum across all transcripts
- **Average mentions**: Mean per transcript
- **Distribution**: Histogram of mention counts

## Example: Analyzing January 2026 Fed Mention Contract

```bash
# Analyze the specific January 2026 event
fomc analyze-kalshi-contracts \
  --event-ticker kxfedmention-26jan \
  --scope powell_only \
  --output-dir results/jan2026_analysis
```

This will:
1. Fetch all markets in the `kxfedmention-26jan` event
2. Extract words like "President", "Layoff", "Median", etc.
3. Generate variants for each word
4. Scan all historical FOMC transcripts
5. Output statistical likelihood data

## Use Cases

### 1. Historical Probability Estimation

Use the analysis to estimate the probability that Powell will mention specific words:

```python
import json

# Load analysis
with open("data/kalshi_analysis/mention_analysis.json") as f:
    analyses = json.load(f)

# Get historical probability for "Layoff"
layoff_data = next(a for a in analyses if a["word"] == "Layoff")
print(f"Historical probability: {layoff_data['mention_frequency']:.1%}")
print(f"Average mentions when mentioned: {layoff_data['avg_mentions_per_transcript']:.2f}")
```

### 2. Compare with Market Prices

Fetch market prices and compare with historical frequencies:

```python
from fomc_analysis.kalshi_api import KalshiClient

client = KalshiClient()

# Get current price for Layoff mention
history = client.get_market_history("KXFEDMENTION-26JAN-LAY")
current_price = history.iloc[-1]["price"]

# Compare with historical frequency
print(f"Market price: {current_price:.0f}¢")
print(f"Historical frequency: {layoff_data['mention_frequency'] * 100:.0f}%")
print(f"Edge: {(layoff_data['mention_frequency'] * 100 - current_price):.1f}¢")
```

### 3. Identify Mispricing

Find words where the market price differs significantly from historical frequency:

```python
import pandas as pd

# Load summary
df = pd.read_csv("data/kalshi_analysis/mention_summary.csv")

# Sort by highest historical frequency
top_words = df.nlargest(10, "Mention Frequency")
print(top_words[["Word", "Mention Frequency", "Avg Mentions/Transcript"]])
```

## API Reference

### `KalshiClient` Extensions

New methods added to `KalshiClient`:

```python
# Get series information
series = client.get_series("KXFEDMENTION")

# Get event with markets
event = client.get_event("kxfedmention-26jan", with_nested_markets=True)

# Get markets with filtering
markets = client.get_markets(
    series_ticker="KXFEDMENTION",
    status="open",
    limit=200
)
```

### `kalshi_contract_analyzer` Module

Main functions:

```python
from fomc_analysis.kalshi_contract_analyzer import (
    fetch_mention_contracts,
    generate_word_variants,
    scan_transcript_for_words,
    analyze_historical_mentions,
    run_kalshi_analysis,
)

# Fetch contracts
contracts = fetch_mention_contracts(kalshi_client, "KXFEDMENTION")

# Generate variants
contracts = generate_word_variants(contracts, openai_client)

# Scan single transcript
mentions = scan_transcript_for_words(
    Path("data/segments/2024-12-18.jsonl"),
    contracts,
    scope="powell_only"
)

# Analyze all transcripts
analyses = analyze_historical_mentions(
    contracts,
    Path("data/segments"),
    scope="powell_only"
)
```

## Troubleshooting

### "No segment files found"

Make sure you've parsed the transcripts first:

```bash
fomc parse --input-dir data/raw_pdf --output-dir data/segments
```

### "Kalshi API credentials are required"

Set your credentials in `.env`:

```bash
KALSHI_API_KEY=your_key
KALSHI_API_SECRET=your_secret
```

### "OPENAI_API_KEY not found"

Add your OpenAI API key to `.env`:

```bash
OPENAI_API_KEY=your_key
```

### Rate Limiting

If you hit Kalshi or OpenAI rate limits:

1. **Kalshi**: The client doesn't implement rate limiting yet. Wait a few seconds between requests.
2. **OpenAI**: Variant generation uses caching, so subsequent runs are free. The generator also includes retry logic with exponential backoff.

## Advanced Usage

### Custom Variant Generation

You can manually control variant generation:

```python
from openai import OpenAI
from fomc_analysis.variants.generator import generate_variants

client = OpenAI()

# Generate variants for custom phrase
result = generate_variants(
    contract="Custom Term",
    base_phrases=["custom", "custom term"],
    openai_client=client,
    cache_dir=Path("data/custom_variants"),
    force_regenerate=True  # Bypass cache
)

print(result.variants)
```

### Full Transcript Scope

To search the entire transcript (not just Powell's remarks):

```bash
fomc analyze-kalshi-contracts \
  --scope full_transcript \
  --output-dir data/kalshi_analysis_full
```

This is useful for contracts that track mentions by anyone in the press conference.

## Future Enhancements

Potential improvements:

1. **Real-time monitoring**: Watch for new Kalshi contracts and auto-analyze
2. **Historical correlation**: Track which words tend to co-occur
3. **Time series analysis**: Detect trends in word usage over time
4. **Automated trading signals**: Compare prices with historical frequencies to identify edges
5. **Multi-series support**: Analyze other Kalshi series (unemployment, inflation, etc.)

## References

- [Kalshi API Documentation](https://docs.kalshi.com)
- [Kalshi Fed Mention Markets](https://kalshi.com/markets/kxfedmention)
- [OpenAI API Documentation](https://platform.openai.com/docs)
