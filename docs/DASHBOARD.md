# Streamlit Dashboard

A lightweight Streamlit app surfaces the analytics database so you can explore
backtests, grid searches, and live predictions without re-running heavy scripts.

## Prerequisites

1. Install dependencies (includes Streamlit):

   ```bash
   uv sync --extra dev
   ```

2. Run migrations and load the artifacts you want to visualize:

   ```bash
   uv run alembic upgrade head
   uv run python scripts/load_backtest_results.py --predictions-csv results/backtest_v3/predictions.csv --trades-csv results/backtest_v3/trades.csv
   uv run python scripts/load_grid_search.py
   uv run python scripts/load_upcoming_predictions.py
   ```

3. Ensure `DATABASE_URL` points to the SQLite/Postgres instance that contains
   your data (optional if you're using the default SQLite location).

## Run the App

```bash
uv run streamlit run dashboard/app.py
```

- Use the sidebar to choose the dataset type (e.g., `fomc`) and select a run.
- The main view shows metadata, overall metrics, and horizon metrics.
- Tabs expose detailed tables for predictions, trades, and grid-search results.

## Customizing

- The dashboard reads data via `DashboardRepository` in
  `src/fomc_analysis/dashboard/queries.py`. Extend or reuse these helpers to add
  new panels.
- Add new tabs/visualizations under `dashboard/app.py` (e.g., word-frequency
  charts, upcoming prediction leaderboards).
- For future datasets (earnings, etc.), ingest them with unique dataset types
  and they will automatically appear in the dataset type selector.

## Deployment Notes

- For internal sharing, deploy with `streamlit run ...` behind your VPN or use
  Streamlit Community Cloud.
- Set `DATABASE_URL` in the deployment environment and ensure the process has
  read access to the DB file/instance.
