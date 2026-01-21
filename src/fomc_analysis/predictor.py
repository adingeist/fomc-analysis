"""Utilities for generating live mention contract predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type
import pandas as pd

from .backtester_v3 import (
    RESOLVED_MARKET_STATUSES,
    RESOLVED_RESULTS,
    fetch_kalshi_contract_outcomes,
)
from .models import BaseModel


@dataclass
class UpcomingPrediction:
    """Single forecast for an unresolved contract market."""

    meeting_date: str
    contract: str
    ticker: Optional[str]
    market_status: str
    prediction_generated_at: str
    predicted_probability: float
    confidence_lower: float
    confidence_upper: float
    market_price: Optional[float]
    edge: Optional[float]
    event_ticker: Optional[str] = None
    days_until_meeting: Optional[int] = None


def _contract_display_name(contract_def: Dict[str, Any]) -> str:
    name = contract_def.get("word", "")
    threshold = contract_def.get("threshold", 1)
    if threshold and threshold > 1:
        return f"{name} ({threshold}+)"
    return name


def _parse_meeting_date(market: Dict[str, Any]) -> Optional[date]:
    meeting_date_str = market.get("expiration_date") or market.get("close_date")
    if not meeting_date_str:
        return None
    try:
        return datetime.fromisoformat(meeting_date_str).date()
    except ValueError:
        return None


def _extract_market_price(market: Dict[str, Any]) -> Optional[float]:
    price_dollars = market.get("last_price_dollars")
    if price_dollars is not None:
        try:
            return float(price_dollars)
        except (TypeError, ValueError):
            pass

    price = market.get("last_price") or market.get("close_price")
    if price is None:
        return None

    try:
        price = float(price)
    except (TypeError, ValueError):
        return None

    if price > 1:
        price /= 100.0
    return price


def _collect_upcoming_markets(contract_data: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    upcoming: List[Dict[str, Any]] = []
    for contract_def in contract_data:
        display_name = _contract_display_name(contract_def)
        for market in contract_def.get("markets", []):
            status = str(market.get("status", "")).lower()
            result_value = str(market.get("result", "")).lower()
            if status in RESOLVED_MARKET_STATUSES or result_value in RESOLVED_RESULTS:
                continue
            meeting_date = _parse_meeting_date(market)
            if meeting_date is None:
                continue
            upcoming.append(
                {
                    "contract": display_name,
                    "market": market,
                    "meeting_date": meeting_date,
                    "status": status,
                }
            )
    return upcoming


def _build_training_matrix(contract_data: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    outcomes_df = fetch_kalshi_contract_outcomes(contract_data, kalshi_client=None)
    if outcomes_df.empty:
        raise ValueError("No resolved Kalshi markets available for training.")

    pivot = outcomes_df.pivot_table(
        index="meeting_date",
        columns="contract",
        values="outcome",
        aggfunc="first",
    )
    # Ensure chronological ordering for any downstream logic
    pivot = pivot.sort_index()
    return pivot


def generate_upcoming_predictions(
    contract_data: Sequence[Dict[str, Any]],
    model_class: Type[BaseModel],
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train the given model on resolved meetings and score open markets."""

    model_params = model_params or {}
    training_matrix = _build_training_matrix(contract_data)

    model = model_class(**model_params)
    model.fit(training_matrix)
    prediction_df = model.predict()
    prediction_lookup = prediction_df.set_index("contract")

    upcoming_markets = _collect_upcoming_markets(contract_data)
    now_utc = datetime.now(timezone.utc)
    today = now_utc.date()
    generated_at = now_utc.isoformat()

    predictions: List[UpcomingPrediction] = []
    skipped_contracts: List[str] = []

    for item in upcoming_markets:
        contract_name = item["contract"]
        if contract_name not in prediction_lookup.index:
            if contract_name not in skipped_contracts:
                skipped_contracts.append(contract_name)
            continue

        pred_row = prediction_lookup.loc[contract_name]
        market = item["market"]
        meeting_date = item["meeting_date"]
        price = _extract_market_price(market)
        edge = (
            float(pred_row["probability"]) - price if price is not None else None
        )
        days_until = (meeting_date - today).days if meeting_date else None

        predictions.append(
            UpcomingPrediction(
                meeting_date=meeting_date.strftime("%Y-%m-%d"),
                contract=contract_name,
                ticker=market.get("ticker"),
                market_status=item["status"],
                prediction_generated_at=generated_at,
                predicted_probability=float(pred_row["probability"]),
                confidence_lower=float(pred_row["lower_bound"]),
                confidence_upper=float(pred_row["upper_bound"]),
                market_price=price,
                edge=edge,
                event_ticker=market.get("event_ticker"),
                days_until_meeting=days_until,
            )
        )

    predictions.sort(key=lambda p: (p.edge is not None, p.edge), reverse=True)

    return {
        "predictions": [asdict(p) for p in predictions],
        "metadata": {
            "generated_at": generated_at,
            "model_class": model_class.__name__,
            "model_params": model_params,
            "total_training_meetings": int(len(training_matrix)),
            "contracts_with_history": int(len(training_matrix.columns)),
            "skipped_contracts": skipped_contracts,
        },
    }


def save_predictions_to_disk(result: Dict[str, Any], output_dir: Path) -> None:
    """Persist JSON + CSV versions of the prediction output."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "predictions.json"
    json_path.write_text(json.dumps(result, indent=2))

    predictions = result.get("predictions", [])
    if predictions:
        df = pd.DataFrame(predictions)
        df.to_csv(output_dir / "predictions.csv", index=False)
