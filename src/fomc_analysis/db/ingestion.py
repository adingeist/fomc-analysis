"""Helpers for loading analytics artifacts into the database."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from fomc_analysis.db import models


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _is_null(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _to_float(value: object) -> float | None:
    if _is_null(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    if _is_null(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_bool(value: object) -> bool | None:
    if _is_null(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    value_str = str(value).strip().lower()
    if value_str in {"true", "1", "yes"}:
        return True
    if value_str in {"false", "0", "no"}:
        return False
    return None


def _parse_datetime(value: object) -> datetime | None:
    if _is_null(value):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for fmt in (None, "%Y-%m-%d %H:%M:%S"):
        try:
            if fmt:
                parsed = datetime.strptime(text, fmt)
            else:
                parsed = datetime.fromisoformat(text)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _parse_date(value: object) -> date | None:
    if _is_null(value):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    parsed_dt = _parse_datetime(value)
    return parsed_dt.date() if parsed_dt else None


def _file_timestamp(path: Path) -> datetime:
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _compute_hash(paths: Sequence[Path | None]) -> str | None:
    valid_paths = [p for p in paths if p is not None and p.exists()]
    if not valid_paths:
        return None
    digest = hashlib.sha256()
    for path in valid_paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _pick(record: dict[str, object], *keys: str) -> object:
    for key in keys:
        if key in record:
            value = record[key]
            if value not in (None, ""):
                return value
    return None


def _upsert_dataset_run(
    session: Session,
    *,
    dataset_slug: str,
    dataset_type: str,
    run_timestamp: datetime,
    run_id: str | None,
    source_file: str | None,
    source_hash: str | None,
    hyperparameters: dict | None,
    metadata: dict | None,
) -> models.DatasetRun:
    stmt = select(models.DatasetRun)
    if run_id:
        stmt = stmt.where(models.DatasetRun.run_id == run_id)
    else:
        stmt = stmt.where(
            models.DatasetRun.dataset_slug == dataset_slug,
            models.DatasetRun.run_timestamp == run_timestamp,
        )

    run = session.scalars(stmt).first()
    if run is None:
        run = models.DatasetRun(
            dataset_slug=dataset_slug,
            dataset_type=dataset_type,
            run_timestamp=run_timestamp,
        )

    run.run_id = run_id or run.run_id
    run.dataset_type = dataset_type
    run.run_timestamp = run_timestamp
    run.source_file = source_file
    run.source_hash = source_hash
    run.hyperparameters = hyperparameters or {}
    run.metadata_json = metadata or {}

    session.add(run)
    session.flush()
    return run


def _reset_child_rows(session: Session, run_id: str) -> None:
    for model in (
        models.OverallMetrics,
        models.HorizonMetrics,
        models.Prediction,
        models.Trade,
        models.GridSearchResult,
    ):
        session.query(model).filter_by(dataset_run_id=run_id).delete(synchronize_session=False)


def _build_prediction(record: dict[str, object], prediction_kind: str, dataset_run_id: str) -> models.Prediction:
    return models.Prediction(
        dataset_run_id=dataset_run_id,
        meeting_date=_parse_date(_pick(record, "meeting_date")),
        prediction_date=_parse_datetime(_pick(record, "prediction_date", "prediction_generated_at")),
        contract=_pick(record, "contract"),
        ticker=_pick(record, "ticker"),
        event_ticker=_pick(record, "event_ticker"),
        market_status=_pick(record, "market_status"),
        prediction_kind=prediction_kind,
        days_before_meeting=_to_int(_pick(record, "days_before_meeting")),
        days_until_meeting=_to_int(_pick(record, "days_until_meeting")),
        predicted_probability=_to_float(_pick(record, "predicted_probability")),
        confidence_lower=_to_float(_pick(record, "confidence_lower")),
        confidence_upper=_to_float(_pick(record, "confidence_upper")),
        market_price=_to_float(_pick(record, "market_price")),
        edge=_to_float(_pick(record, "edge")),
        actual_outcome=_to_int(_pick(record, "actual_outcome")),
        correct=_to_bool(_pick(record, "correct")),
    )


def _build_trade(record: dict[str, object], dataset_run_id: str) -> models.Trade:
    return models.Trade(
        dataset_run_id=dataset_run_id,
        meeting_date=_parse_date(_pick(record, "meeting_date")),
        prediction_date=_parse_datetime(_pick(record, "prediction_date")),
        contract=_pick(record, "contract"),
        ticker=_pick(record, "ticker"),
        days_before_meeting=_to_int(_pick(record, "days_before_meeting")),
        side=_pick(record, "side"),
        position_size=_to_float(_pick(record, "position_size")),
        entry_price=_to_float(_pick(record, "entry_price")),
        predicted_probability=_to_float(_pick(record, "predicted_probability")),
        edge=_to_float(_pick(record, "edge")),
        actual_outcome=_to_int(_pick(record, "actual_outcome")),
        pnl=_to_float(_pick(record, "pnl")),
        roi=_to_float(_pick(record, "roi")),
    )


def load_backtest_artifacts(
    session: Session,
    *,
    backtest_json: Path,
    dataset_slug: str = "fomc_backtest_v3",
    dataset_type: str = "fomc",
    run_id: str | None = None,
    run_timestamp: datetime | None = None,
    predictions_csv: Path | None = None,
    trades_csv: Path | None = None,
) -> models.DatasetRun:
    data = json.loads(backtest_json.read_text())
    metadata = data.get("metadata", {})

    resolved_timestamp = run_timestamp or _parse_datetime(metadata.get("generated_at")) or _file_timestamp(backtest_json)
    source_hash = _compute_hash([backtest_json, predictions_csv, trades_csv])
    meta_payload = {
        "source_files": {
            "backtest_json": str(backtest_json),
            "predictions_csv": str(predictions_csv) if predictions_csv else None,
            "trades_csv": str(trades_csv) if trades_csv else None,
        }
    }

    dataset_run = _upsert_dataset_run(
        session,
        dataset_slug=dataset_slug,
        dataset_type=dataset_type,
        run_timestamp=resolved_timestamp,
        run_id=run_id,
        source_file=str(backtest_json),
        source_hash=source_hash,
        hyperparameters=metadata,
        metadata=meta_payload,
    )

    _reset_child_rows(session, dataset_run.id)

    overall_metrics = data.get("overall_metrics") or {}
    if overall_metrics:
        session.add(
            models.OverallMetrics(
                dataset_run_id=dataset_run.id,
                total_trades=_to_float(overall_metrics.get("total_trades")),
                total_pnl=_to_float(overall_metrics.get("total_pnl")),
                roi=_to_float(overall_metrics.get("roi")),
                win_rate=_to_float(overall_metrics.get("win_rate")),
                sharpe=_to_float(overall_metrics.get("sharpe")),
                sortino=_to_float(overall_metrics.get("sortino")),
                avg_pnl_per_trade=_to_float(overall_metrics.get("avg_pnl_per_trade")),
                final_capital=_to_float(overall_metrics.get("final_capital")),
            )
        )

    for horizon_str, metrics in (data.get("horizon_metrics") or {}).items():
        session.add(
            models.HorizonMetrics(
                dataset_run_id=dataset_run.id,
                horizon_days=int(horizon_str),
                total_predictions=_to_int(metrics.get("total_predictions")),
                correct_predictions=_to_int(metrics.get("correct_predictions")),
                accuracy=_to_float(metrics.get("accuracy")),
                total_trades=_to_int(metrics.get("total_trades")),
                winning_trades=_to_int(metrics.get("winning_trades")),
                win_rate=_to_float(metrics.get("win_rate")),
                total_pnl=_to_float(metrics.get("total_pnl")),
                avg_pnl_per_trade=_to_float(metrics.get("avg_pnl_per_trade")),
                roi=_to_float(metrics.get("roi")),
                sharpe_ratio=_to_float(metrics.get("sharpe_ratio")),
                brier_score=_to_float(metrics.get("brier_score")),
            )
        )

    prediction_records: list[dict[str, object]]
    if predictions_csv and predictions_csv.exists():
        prediction_records = _read_csv(predictions_csv)
    else:
        prediction_records = data.get("predictions") or []

    predictions = [
        _build_prediction(record, prediction_kind="backtest", dataset_run_id=dataset_run.id)
        for record in prediction_records
    ]
    session.add_all(predictions)

    trade_records: list[dict[str, object]]
    if trades_csv and trades_csv.exists():
        trade_records = _read_csv(trades_csv)
    else:
        trade_records = data.get("trades") or []

    trades = [_build_trade(record, dataset_run_id=dataset_run.id) for record in trade_records]
    session.add_all(trades)

    return dataset_run


def load_grid_search_results(
    session: Session,
    *,
    grid_search_csv: Path,
    dataset_slug: str = "fomc_backtest_v3_grid_search",
    dataset_type: str = "fomc",
    run_id: str | None = None,
    run_timestamp: datetime | None = None,
) -> models.DatasetRun:
    rows = _read_csv(grid_search_csv)
    resolved_timestamp = run_timestamp or _file_timestamp(grid_search_csv)
    dataset_run = _upsert_dataset_run(
        session,
        dataset_slug=dataset_slug,
        dataset_type=dataset_type,
        run_timestamp=resolved_timestamp,
        run_id=run_id,
        source_file=str(grid_search_csv),
        source_hash=_compute_hash([grid_search_csv]),
        hyperparameters={},
        metadata={"source_files": {"grid_search_csv": str(grid_search_csv)}},
    )

    _reset_child_rows(session, dataset_run.id)

    grid_rows = [
        models.GridSearchResult(
            dataset_run_id=dataset_run.id,
            total_pnl=_to_float(row.get("total_pnl")),
            roi=_to_float(row.get("roi")),
            sharpe=_to_float(row.get("sharpe")),
            trades=_to_int(row.get("trades")),
            win_rate=_to_float(row.get("win_rate")),
            slippage=_to_float(row.get("slippage")),
            transaction_cost_rate=_to_float(row.get("transaction_cost_rate")),
            max_position_size=_to_float(row.get("max_position_size")),
            train_window_size=_to_int(row.get("train_window_size")),
            test_start_date=_pick(row, "test_start_date"),
            yes_position_size_pct=_to_float(row.get("yes_position_size_pct")),
            no_position_size_pct=_to_float(row.get("no_position_size_pct")),
            yes_edge_threshold=_to_float(row.get("yes_edge_threshold")),
            no_edge_threshold=_to_float(row.get("no_edge_threshold")),
        )
        for row in rows
    ]
    session.add_all(grid_rows)
    return dataset_run


def load_upcoming_predictions(
    session: Session,
    *,
    predictions_csv: Path,
    dataset_slug: str = "fomc_upcoming_predictions",
    dataset_type: str = "fomc",
    run_id: str | None = None,
    run_timestamp: datetime | None = None,
) -> models.DatasetRun:
    rows = _read_csv(predictions_csv)
    resolved_timestamp = run_timestamp or _file_timestamp(predictions_csv)
    dataset_run = _upsert_dataset_run(
        session,
        dataset_slug=dataset_slug,
        dataset_type=dataset_type,
        run_timestamp=resolved_timestamp,
        run_id=run_id,
        source_file=str(predictions_csv),
        source_hash=_compute_hash([predictions_csv]),
        hyperparameters={"prediction_mode": "live"},
        metadata={"source_files": {"predictions_csv": str(predictions_csv)}},
    )

    _reset_child_rows(session, dataset_run.id)

    predictions = [
        _build_prediction(row, prediction_kind="live", dataset_run_id=dataset_run.id)
        for row in rows
    ]
    session.add_all(predictions)
    return dataset_run
