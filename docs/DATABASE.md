# Analytics Database

This repository now ships with a portable SQLAlchemy + Alembic database layer so the
upcoming dashboard (and future datasets like **Earnings Call Mentions**) can read
historical runs without re-computing heavy pipelines.

## Connection & Configuration

- Default URL: `sqlite:///data/fomc_analysis.db`
- Override: set `DATABASE_URL` (or add to `.env`). Any SQLAlchemy-compatible URL
  works, so you can point at Postgres for staging/production.
- The helper `fomc_analysis.db.session.resolve_database_url()` enforces the
  default and auto-creates the SQLite parent folder.

## Schema Overview

| Table | Purpose |
| --- | --- |
| `dataset_runs` | Metadata for every artifact export. Columns include `run_id`, `dataset_slug`, `dataset_type`, timestamps, hyperparameters JSON, and source file hashes. Unique constraint on `(dataset_slug, run_timestamp)` keeps loads idempotent. |
| `overall_metrics` | One-to-one summary metrics (ROI, Sharpe, win rate, capital) per run. |
| `horizon_metrics` | Metrics per prediction horizon (e.g., 7/14/30 days). |
| `predictions` | Individual prediction snapshots for both historical backtests (`prediction_kind="backtest"`) and live signals (`prediction_kind="live"`). Stores probabilities, confidence bounds, Kalshi tickers, prices, and realized outcomes. |
| `trades` | Executed trades from backtests, including sizing, entry prices, and PnL. |
| `grid_search_results` | Parameter sweep results with both the metrics and the tested hyperparameters per row. |

The schema is defined via Alembic migrations (see `alembic/versions/20240201_000001_initial_schema.py`).

## Running Migrations

```bash
export UV_CACHE_DIR=.uv_cache  # optional local cache
uv run alembic upgrade head
```

- Alembic reads `DATABASE_URL`. If unset, the SQLite file is created under `data/`.
- Use `alembic downgrade base` to reset when testing locally.

## Loading Artifacts

All ingestion CLIs live under `scripts/` and defer to reusable helpers inside
`fomc_analysis.db.ingestion` (used by the tests as well).

### Backtest v3 bundle

```bash
uv run python scripts/load_backtest_results.py \
  --backtest-json results/backtest_v3/backtest_results.json \
  --predictions-csv results/backtest_v3/predictions.csv \
  --trades-csv results/backtest_v3/trades.csv \
  --dataset-slug fomc_backtest_v3
```

- Upserts a `dataset_run` using `run_id` or `(dataset_slug, run_timestamp)`.
- If predictions/trades CSVs are omitted, the loader falls back to the JSON
  payload. CSVs are preferred because they hold Kalshi tickers and richer annotations.

### Grid search sweeps

```bash
uv run python scripts/load_grid_search.py \
  --grid-search-csv results/backtest_v3/grid_search.csv \
  --dataset-slug fomc_backtest_v3_grid_search
```

### Upcoming (live) predictions

```bash
uv run python scripts/load_upcoming_predictions.py \
  --predictions-csv results/upcoming_predictions/predictions.csv \
  --dataset-slug fomc_upcoming_predictions
```

Each CLI accepts `--database-url`, `--run-id`, and timestamp overrides for full control.

## Testing & Validation

`pytest tests/test_db_ingestion.py` loads fixture artifacts into a temporary
SQLite database and asserts:

- Prediction/trade counts equal their source CSV rows
- Horizon metrics persist per run
- Grid search and live predictions land in separate dataset slugs
- Re-running ingestion with the same `run_id` keeps the dataset stable (no duplicate rows)

## Extending to Earnings Call Mentions

1. Define a new `dataset_slug`/`dataset_type` (e.g., `earnings_backtest_v1`).
2. Reuse `load_backtest_artifacts` with the earnings JSON/CSV exports, or write a
   thin wrapper script that adapts to the new folder structure.
3. Include any extra hyperparameters inside the `metadata` dict you pass to the
   loader. They will be stored in `dataset_runs.hyperparameters` for dashboards.
4. If the dataset surfaces additional per-row metrics, extend the ORM model with
   new columns (plus an Alembic migration) – the ingest helpers already delete
   prior child rows before writing, so schema changes remain safe.

By centralizing everything in SQLite/Postgres, dashboards and notebooks can query
past runs instantly via SQL instead of re-running heavy pipelines.
