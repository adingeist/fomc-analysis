# Earnings Call Mentions – Future Work

We plan to reuse the new analytics database layer for a future pipeline that
scores quarterly earnings transcripts (or other corporate events) against
Kalshi-style mention markets. This document captures the current thoughts so we
can spin the workstream up quickly when ready.

## Dataset Thoughts

- **dataset_slug**: plan to namespace earnings artifacts (e.g.,
  `earnings_mentions_v1`) so they stay isolated from FOMC runs.
- **dataset_type**: use `earnings` to distinguish dashboards/queries.
- **Temporal fields**: include `earnings_call_date`, fiscal `quarter`, and
  `fiscal_year` columns on predictions/trades. Seasonality analysis will likely
  compare like-for-like quarters (Q1 vs Q1, etc.).
- **Metadata**: store `ticker`, `sector`, and `exchange` details in
  `dataset_runs.hyperparameters` so dashboards can filter by industry.

## Schema Extensions

The current SQLAlchemy models already support multiple dataset types, but we may
add columns when earnings-specific data arrives:

- `predictions.quarter`, `predictions.fiscal_year`
- `predictions.company_ticker`, `predictions.company_name`
- Optional `trades.revenue_surprise`/`eps_surprise` to link alpha to fundamentals

Add these via a new Alembic migration once the data feeds are confirmed.

## Tooling Hooks

1. Build an ingestion CLI (e.g., `scripts/load_earnings_backtest.py`) that wraps
   `load_backtest_artifacts` and injects the new metadata fields.
2. Extend the feature generator to understand corporate filings (10-Q, S-1, etc.)
   or vendor transcripts.
3. Update dashboards/notebooks to accept a `dataset_type` parameter so the same
   queries power both FOMC and earnings views.

## Open Questions

- Seasonality adjustments: do we normalize by historical mention frequency per
  quarter or per sector?
- Price data: are we comparing against Kalshi only, or also retail prediction
  markets/CFDs? This affects ingestion scripts.
- Contract mappings: we may need separate prompt templates and resolution rules
  (e.g., CEO vs CFO statements).

These notes should unblock future contributors who want to stand up the earnings
pipeline without rediscovering the design decisions made for the FOMC system.
