"""FOMC backtests API router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from fomc_analysis.api.schemas import (
    FOMCBacktestResponse,
    FOMCBacktestTrade,
    FOMCHorizonMetrics,
)

router = APIRouter(prefix="/fomc/backtests", tags=["fomc-backtests"])


def get_data_service(request: Request):
    """Dependency to get FOMC data service."""
    return request.app.state.fomc_data_service


@router.get("", response_model=FOMCBacktestResponse)
async def get_backtest_results(
    data_service=Depends(get_data_service),
):
    """
    Get FOMC backtest results.

    Returns historical backtest performance including horizon-specific
    metrics, overall statistics, and individual trades.
    """
    result = data_service.get_backtest_results()

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No FOMC backtest results available. Run the backtest first.",
        )

    horizon_metrics = []
    for horizon, metrics in result.get("horizon_metrics", {}).items():
        horizon_metrics.append(FOMCHorizonMetrics(
            horizon_days=int(horizon),
            total_predictions=metrics.get("total_predictions", 0),
            correct_predictions=metrics.get("correct_predictions", 0),
            accuracy=metrics.get("accuracy", 0.0),
            total_trades=metrics.get("total_trades", 0),
            winning_trades=metrics.get("winning_trades", 0),
            win_rate=metrics.get("win_rate", 0.0),
            total_pnl=metrics.get("total_pnl", 0.0),
            avg_pnl_per_trade=metrics.get("avg_pnl_per_trade", 0.0),
            roi=metrics.get("roi", 0.0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
            brier_score=metrics.get("brier_score", 0.0),
        ))

    trades = []
    for trade in result.get("trades", []):
        trades.append(FOMCBacktestTrade(
            meeting_date=trade.get("meeting_date", ""),
            contract=trade.get("contract", ""),
            prediction_date=trade.get("prediction_date", ""),
            days_before_meeting=trade.get("days_before_meeting", 0),
            side=trade.get("side", ""),
            position_size=trade.get("position_size", 0.0),
            entry_price=trade.get("entry_price", 0.0),
            predicted_probability=trade.get("predicted_probability", 0.0),
            edge=trade.get("edge", 0.0),
            actual_outcome=trade.get("actual_outcome", 0),
            pnl=trade.get("pnl", 0.0),
            roi=trade.get("roi", 0.0),
        ))

    return FOMCBacktestResponse(
        horizon_metrics=horizon_metrics,
        overall_metrics=result.get("overall_metrics", {}),
        trades=trades,
        metadata=result.get("metadata", {}),
    )
