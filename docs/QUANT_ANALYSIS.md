# Quant-Level Analysis of FOMC Mention Contracts

This report summarizes how often Jerome Powell has hit each Kalshi mention contract
from January 2020 through December 2025 (49 press conferences). The numbers come
from `data/features.parquet`, which is generated after running:

```bash
uv run fomc fetch-transcripts
uv run fomc parse --input-dir data/raw_pdf --segments-dir data/segments
uv run fomc export-kalshi-contracts --market-status resolved --output configs/generated_contract_mapping.yaml
uv run fomc build-variants --contracts configs/generated_contract_mapping.yaml
uv run fomc featurize --contracts configs/generated_contract_mapping.yaml --variants-dir data/variants --output data/features.parquet
```

The tables below split contracts into **threshold contracts** (e.g., "Inflation 40+")
and **binary mention contracts** (e.g., "Layoff"). Metrics:

- **Hit Rate** – Share of meetings that resolved YES historically.
- **Avg Mentions (hit)** – Average raw mention count conditional on the contract hitting.
- **Max Mentions** – Peak count observed in the sample.

## Threshold Contracts

| Contract | Hit Rate | Avg Mentions (hit) | Max Mentions |
| --- | --- | --- | --- |
| Price 15+ | 100.0% | 18.51 | 39 |
| Unemployment 8+ | 100.0% | 13.41 | 40 |
| Inflation 40+ |  98.0% | 53.75 | 103 |
| Inflation 50+ |  98.0% | 55.46 | 108 |
| Cut 7+ |  79.6% | 4.41 | 17 |
| Tariff 5+ |  18.4% | 16.44 | 35 |
| Growth 8+ |  14.3% | 1.00 | 1 |

## Binary Mention Contracts

| Contract | Hit Rate | Avg Mentions (hit) | Max Mentions |
| --- | --- | --- | --- |
| Expectation |  95.9% | 7.30 | 21 |
| Good Afternoon |  95.9% | 1.00 | 1 |
| Pandemic |  87.8% | 7.67 | 25 |
| Anchor |  85.7% | 2.60 | 7 |
| Median |  63.3% | 5.16 | 14 |
| Projection |  63.3% | 5.97 | 13 |
| Balance of Risk |  49.0% | 2.75 | 6 |
| Layoff |  26.5% | 2.08 | 5 |
| Tax |  22.4% | 1.73 | 4 |
| Tariff Inflation |  18.4% | 16.78 | 36 |
| AI / Artificial Intelligence |  12.2% | 4.33 | 8 |
| Crypto / Bitcoin |   6.1% | 2.33 | 5 |

## Takeaways

- The core macro barometers (Inflation, Unemployment, Price) hit their thresholds in
  virtually every meeting during 2020-2025, so Kalshi contracts on those terms are
  mostly about sizing the payout when they **fail**.
- Binary contracts show a much wider dispersion: "Good Afternoon" is almost
  deterministic, while "Crypto / Bitcoin" has appeared in only ~6% of meetings.
- Tariff-related words are rare but, when they appear, they tend to spike in
  clusters (high max counts but low hit rates), which aligns with the historical trade context.

To recreate the tables, load `data/features.parquet` into a notebook or script
and aggregate `_mentioned` / `_count` columns exactly as shown in the snippet in
this document. Any new contract added via `export-kalshi-contracts` will
automatically be included the next time you re-run `fomc featurize` and refresh
the aggregation.
