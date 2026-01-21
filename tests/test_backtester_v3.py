import pandas as pd

from fomc_analysis.backtester_v3 import TimeHorizonBacktester


def test_overall_metrics_without_trades_includes_final_capital():
    outcomes = pd.DataFrame(columns=["meeting_date", "contract", "outcome", "ticker"])
    historical_prices = pd.DataFrame()

    backtester = TimeHorizonBacktester(outcomes=outcomes, historical_prices=historical_prices)

    metrics = backtester._compute_overall_metrics([], initial_capital=1000.0, final_capital=1000.0)

    assert metrics["total_trades"] == 0
    assert metrics["final_capital"] == 1000.0
    assert metrics["roi"] == 0.0
