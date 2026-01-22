"""Reusable query helpers for the analytics dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
from sqlalchemy import Select, func, select

from fomc_analysis.db import models
from fomc_analysis.db.session import get_session_factory, resolve_database_url


def _to_dataframe(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    data = list(rows)
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


@dataclass
class DashboardRepository:
    """Lightweight read-only repository for dashboard queries."""

    database_url: str | None = None

    def __post_init__(self) -> None:
        self.database_url = resolve_database_url(self.database_url)
        self._session_factory = get_session_factory(self.database_url)

    def _run_query(self, stmt: Select) -> list[Any]:
        session = self._session_factory()
        try:
            result = session.execute(stmt)
            return result.scalars().all()
        finally:
            session.close()

    def list_dataset_types(self) -> list[str]:
        stmt = select(func.distinct(models.DatasetRun.dataset_type)).order_by(models.DatasetRun.dataset_type)
        session = self._session_factory()
        try:
            return [row[0] for row in session.execute(stmt).all() if row[0]]
        finally:
            session.close()

    def list_dataset_runs(
        self,
        *,
        dataset_slug: str | None = None,
        dataset_type: str | None = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        stmt = select(models.DatasetRun).order_by(models.DatasetRun.run_timestamp.desc()).limit(limit)
        if dataset_slug:
            stmt = stmt.where(models.DatasetRun.dataset_slug == dataset_slug)
        if dataset_type:
            stmt = stmt.where(models.DatasetRun.dataset_type == dataset_type)

        runs = self._run_query(stmt)
        return _to_dataframe(
            {
                "dataset_run_id": run.id,
                "run_identifier": run.run_id,
                "dataset_slug": run.dataset_slug,
                "dataset_type": run.dataset_type,
                "run_timestamp": run.run_timestamp,
                "source_file": run.source_file,
                "created_at": run.created_at,
            }
            for run in runs
        )

    def get_dataset_run(self, dataset_run_id: str) -> models.DatasetRun | None:
        stmt = select(models.DatasetRun).where(models.DatasetRun.id == dataset_run_id)
        runs = self._run_query(stmt)
        if runs:
            return runs[0]
        stmt = select(models.DatasetRun).where(models.DatasetRun.run_id == dataset_run_id)
        runs = self._run_query(stmt)
        return runs[0] if runs else None

    def get_overall_metrics(self, run_id: str) -> dict[str, Any]:
        stmt = select(models.OverallMetrics).where(models.OverallMetrics.dataset_run_id == run_id)
        metrics = self._run_query(stmt)
        if not metrics:
            return {}
        row = metrics[0]
        return {
            "total_trades": row.total_trades,
            "total_pnl": row.total_pnl,
            "roi": row.roi,
            "win_rate": row.win_rate,
            "sharpe": row.sharpe,
            "sortino": row.sortino,
            "avg_pnl_per_trade": row.avg_pnl_per_trade,
            "final_capital": row.final_capital,
        }

    def get_horizon_metrics(self, run_id: str) -> pd.DataFrame:
        stmt = select(models.HorizonMetrics).where(models.HorizonMetrics.dataset_run_id == run_id)
        rows = self._run_query(stmt)
        df = _to_dataframe(
            {
                "horizon_days": row.horizon_days,
                "total_predictions": row.total_predictions,
                "correct_predictions": row.correct_predictions,
                "accuracy": row.accuracy,
                "total_trades": row.total_trades,
                "winning_trades": row.winning_trades,
                "win_rate": row.win_rate,
                "total_pnl": row.total_pnl,
                "avg_pnl_per_trade": row.avg_pnl_per_trade,
                "roi": row.roi,
                "sharpe_ratio": row.sharpe_ratio,
                "brier_score": row.brier_score,
            }
            for row in rows
        )
        if df.empty:
            return df
        return df.sort_values(by="horizon_days", ascending=True)

    def get_predictions(self, run_id: str) -> pd.DataFrame:
        stmt = select(models.Prediction).where(models.Prediction.dataset_run_id == run_id)
        rows = self._run_query(stmt)
        df = _to_dataframe(
            {
                "meeting_date": row.meeting_date,
                "prediction_date": row.prediction_date,
                "contract": row.contract,
                "ticker": row.ticker,
                "prediction_kind": row.prediction_kind,
                "days_before_meeting": row.days_before_meeting,
                "predicted_probability": row.predicted_probability,
                "confidence_lower": row.confidence_lower,
                "confidence_upper": row.confidence_upper,
                "market_price": row.market_price,
                "edge": row.edge,
                "actual_outcome": row.actual_outcome,
                "correct": row.correct,
            }
            for row in rows
        )
        if df.empty:
            return df
        return df.sort_values(by=["meeting_date", "contract", "prediction_date"], ascending=True)

    def get_trades(self, run_id: str) -> pd.DataFrame:
        stmt = select(models.Trade).where(models.Trade.dataset_run_id == run_id)
        rows = self._run_query(stmt)
        df = _to_dataframe(
            {
                "meeting_date": row.meeting_date,
                "prediction_date": row.prediction_date,
                "contract": row.contract,
                "side": row.side,
                "position_size": row.position_size,
                "entry_price": row.entry_price,
                "predicted_probability": row.predicted_probability,
                "edge": row.edge,
                "actual_outcome": row.actual_outcome,
                "pnl": row.pnl,
                "roi": row.roi,
            }
            for row in rows
        )
        if df.empty:
            return df
        return df.sort_values(by=["meeting_date", "prediction_date", "contract"], ascending=True)

    def get_grid_search_results(self, run_id: str) -> pd.DataFrame:
        stmt = select(models.GridSearchResult).where(models.GridSearchResult.dataset_run_id == run_id)
        rows = self._run_query(stmt)
        df = _to_dataframe(
            {
                "total_pnl": row.total_pnl,
                "roi": row.roi,
                "sharpe": row.sharpe,
                "trades": row.trades,
                "win_rate": row.win_rate,
                "slippage": row.slippage,
                "transaction_cost_rate": row.transaction_cost_rate,
                "max_position_size": row.max_position_size,
                "train_window_size": row.train_window_size,
                "test_start_date": row.test_start_date,
                "yes_position_size_pct": row.yes_position_size_pct,
                "no_position_size_pct": row.no_position_size_pct,
                "yes_edge_threshold": row.yes_edge_threshold,
                "no_edge_threshold": row.no_edge_threshold,
            }
            for row in rows
        )
        if df.empty:
            return df
        return df.sort_values(by="roi", ascending=False)
