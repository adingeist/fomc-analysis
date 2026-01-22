import pandas as pd
import pytest

from fomc_analysis.backtester_v3 import TimeHorizonBacktester


class DummyModel:
    """Lightweight stand-in for probability model used in unit tests."""
    fit_lengths = []

    def __init__(self, probabilities=None):
        self.probabilities = probabilities or {}
        self.contracts = []

    def fit(self, events: pd.DataFrame) -> None:
        DummyModel.fit_lengths.append(len(events))
        self.contracts = list(events.columns)

    def predict(self, n_future: int = 1) -> pd.DataFrame:
        rows = []
        for contract in self.contracts:
            prob = self.probabilities.get(contract, 0.6)
            rows.append({
                "contract": contract,
                "probability": prob,
                "lower_bound": max(0.0, prob - 0.05),
                "upper_bound": min(1.0, prob + 0.05),
                "uncertainty": 0.05,
            })
        return pd.DataFrame(rows)


def test_overall_metrics_without_trades_includes_final_capital():
    outcomes = pd.DataFrame(columns=["meeting_date", "contract", "outcome", "ticker"])
    historical_prices = pd.DataFrame()

    backtester = TimeHorizonBacktester(outcomes=outcomes, historical_prices=historical_prices)

    metrics = backtester._compute_overall_metrics([], initial_capital=1000.0, final_capital=1000.0)

    assert metrics["total_trades"] == 0
    assert metrics["final_capital"] == 1000.0
    assert metrics["roi"] == 0.0


def test_rolling_window_and_start_date_controls():
    meeting_dates = pd.to_datetime([
        "2024-01-01", "2024-03-01", "2024-05-01",
        "2024-07-01", "2024-09-01", "2024-11-01",
    ])
    outcomes = []
    price_rows = []
    for date in meeting_dates:
        outcomes.append({
            "meeting_date": date,
            "contract": "Projection",
            "outcome": 1,
            "ticker": "PROJ",
        })
        price_rows.append({
            "ticker": "PROJ",
            "meeting_date": date,
            "days_before": 7,
            "prediction_date": (date - pd.Timedelta(days=7)).date(),
            "price": 0.3,
        })

    outcomes_df = pd.DataFrame(outcomes)
    prices_df = pd.DataFrame(price_rows)

    DummyModel.fit_lengths = []
    backtester = TimeHorizonBacktester(
        outcomes=outcomes_df,
        historical_prices=prices_df,
        horizons=[7],
        edge_threshold=0.10,
        position_size_pct=0.10,
        min_train_window=2,
        train_window_size=2,
        test_start_date="2024-07-01",
    )
    result = backtester.run(
        model_class=DummyModel,
        model_params={"probabilities": {"Projection": 0.8}},
        initial_capital=1000.0,
    )

    assert result.trades  # Trades exist after test_start_date
    assert max(DummyModel.fit_lengths) == 2  # Rolling window enforced
    for trade in result.trades:
        assert pd.Timestamp(trade.meeting_date) >= pd.Timestamp("2024-07-01")


def test_trade_constraints_apply_slippage_and_transaction_cost():
    meetings = pd.to_datetime(["2024-01-01", "2024-03-01"])
    outcomes = []
    prices = []
    for date in meetings:
        outcomes.append({
            "meeting_date": date,
            "contract": "Projection",
            "outcome": 1,
            "ticker": "PROJ",
        })
        if date == meetings[1]:
            prices.append({
                "ticker": "PROJ",
                "meeting_date": date,
                "days_before": 7,
                "prediction_date": (date - pd.Timedelta(days=7)).date(),
                "price": 0.4,
            })
    outcomes_df = pd.DataFrame(outcomes)
    prices_df = pd.DataFrame(prices)

    DummyModel.fit_lengths = []
    backtester = TimeHorizonBacktester(
        outcomes=outcomes_df,
        historical_prices=prices_df,
        horizons=[7],
        edge_threshold=0.10,
        position_size_pct=0.60,
        max_position_size=200.0,
        min_train_window=1,
        slippage=0.05,
        transaction_cost_rate=0.02,
        yes_position_size_pct=0.60,
    )
    result = backtester.run(
        model_class=DummyModel,
        model_params={"probabilities": {"Projection": 0.8}},
        initial_capital=1000.0,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.position_size == pytest.approx(200.0)
    assert trade.entry_price == pytest.approx(0.45)

    gross = trade.position_size * (1 - trade.entry_price) / trade.entry_price
    expected_pnl = gross * (1 - 0.07) - trade.position_size * 0.02
    assert trade.pnl == pytest.approx(expected_pnl, rel=1e-6)


def test_directional_thresholds_and_sizing_rules():
    meetings = pd.to_datetime(["2024-01-01", "2024-03-01"])
    outcomes = []
    prices = []
    for date in meetings:
        for contract, ticker, outcome in [
            ("Bull", "BULL", 1),
            ("Bear", "BEAR", 0),
        ]:
            outcomes.append({
                "meeting_date": date,
                "contract": contract,
                "outcome": outcome,
                "ticker": ticker,
            })
        if date == meetings[1]:
            prices.extend([
                {
                    "ticker": "BULL",
                    "meeting_date": date,
                    "days_before": 7,
                    "prediction_date": (date - pd.Timedelta(days=7)).date(),
                    "price": 0.4,
                },
                {
                    "ticker": "BEAR",
                    "meeting_date": date,
                    "days_before": 7,
                    "prediction_date": (date - pd.Timedelta(days=7)).date(),
                    "price": 0.9,
                },
            ])

    outcomes_df = pd.DataFrame(outcomes)
    prices_df = pd.DataFrame(prices)

    DummyModel.fit_lengths = []
    backtester = TimeHorizonBacktester(
        outcomes=outcomes_df,
        historical_prices=prices_df,
        horizons=[7],
        edge_threshold=0.05,
        yes_edge_threshold=0.10,
        no_edge_threshold=0.80,
        position_size_pct=0.20,
        yes_position_size_pct=0.25,
        no_position_size_pct=0.10,
        min_train_window=1,
        min_yes_probability=0.0,
        max_no_probability=1.0,
    )
    result = backtester.run(
        model_class=DummyModel,
        model_params={"probabilities": {"Bull": 0.8, "Bear": 0.2}},
        initial_capital=1000.0,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.contract == "Bull"
    assert trade.side == "YES"
    assert trade.position_size == pytest.approx(250.0)
