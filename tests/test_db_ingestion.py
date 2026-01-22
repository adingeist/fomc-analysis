from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from fomc_analysis.db.base import Base
from fomc_analysis.db import models
from fomc_analysis.db.ingestion import (
    load_backtest_artifacts,
    load_grid_search_results,
    load_upcoming_predictions,
)
from fomc_analysis.db.session import get_engine, session_scope


@pytest.fixture()
def temp_db_url(tmp_path) -> str:
    db_path = tmp_path / "analytics.db"
    url = f"sqlite:///{db_path}"
    engine = get_engine(url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield url
    Base.metadata.drop_all(engine)


def test_backtest_ingestion_round_trip(temp_db_url: str) -> None:
    data_dir = Path("tests/data/db")
    backtest_json = data_dir / "backtest_results.json"
    predictions_csv = data_dir / "predictions.csv"
    trades_csv = data_dir / "trades.csv"

    with session_scope(temp_db_url) as session:
        run = load_backtest_artifacts(
            session,
            backtest_json=backtest_json,
            predictions_csv=predictions_csv,
            trades_csv=trades_csv,
            dataset_slug="test_backtest",
            dataset_type="fomc",
            run_id="backtest-fixture",
        )
        first_run_id = run.id

    with session_scope(temp_db_url) as session:
        dataset_runs = session.scalars(select(models.DatasetRun)).all()
        assert len(dataset_runs) == 1
        run = dataset_runs[0]
        assert run.hyperparameters["model_class"] == "BetaBinomialModel"

        horizon_rows = session.scalars(select(models.HorizonMetrics)).all()
        assert {row.horizon_days for row in horizon_rows} == {7, 14}

        prediction_count = session.query(models.Prediction).count()
        trade_count = session.query(models.Trade).count()
        assert prediction_count == 2
        assert trade_count == 2

        overall = session.scalars(select(models.OverallMetrics)).first()
        assert overall.total_trades == pytest.approx(3)

    # Re-run ingestion with identical run_id to ensure idempotency
    with session_scope(temp_db_url) as session:
        load_backtest_artifacts(
            session,
            backtest_json=backtest_json,
            predictions_csv=predictions_csv,
            trades_csv=trades_csv,
            dataset_slug="test_backtest",
            dataset_type="fomc",
            run_id="backtest-fixture",
        )

    with session_scope(temp_db_url) as session:
        assert session.query(models.Prediction).count() == 2
        assert session.query(models.Trade).count() == 2
        assert session.query(models.DatasetRun).count() == 1
        run = session.scalars(select(models.DatasetRun)).first()
        assert run.id == first_run_id


def test_grid_and_upcoming_ingestion(temp_db_url: str) -> None:
    data_dir = Path("tests/data/db")

    with session_scope(temp_db_url) as session:
        load_grid_search_results(
            session,
            grid_search_csv=data_dir / "grid_search.csv",
            dataset_slug="grid_fixture",
            dataset_type="fomc",
            run_id="grid-fixture",
        )
        load_upcoming_predictions(
            session,
            predictions_csv=data_dir / "upcoming_predictions.csv",
            dataset_slug="upcoming_fixture",
            dataset_type="fomc",
            run_id="upcoming-fixture",
        )

    with session_scope(temp_db_url) as session:
        grid_rows = session.scalars(select(models.GridSearchResult)).all()
        assert len(grid_rows) == 2
        assert {row.test_start_date for row in grid_rows} == {"2022-01-01", "2021-01-01"}

        live_predictions = session.scalars(select(models.Prediction)).all()
        assert len(live_predictions) == 2
        assert {pred.prediction_kind for pred in live_predictions} == {"live"}
        assert {pred.ticker for pred in live_predictions} == {
            "KXFEDMENTION-18JUN-AI",
            "KXFEDMENTION-18JUN-BANK",
        }

        assert session.query(models.Trade).count() == 0
