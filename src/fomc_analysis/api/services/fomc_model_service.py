"""FOMC model service for managing predictions and models."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from fomc_analysis.models import BaseModel, BetaBinomialModel, EWMAModel
from fomc_analysis.predictor import generate_upcoming_predictions
from fomc_analysis.kalshi_contract_analyzer import (
    ContractWord,
    fetch_mention_contracts,
)
from fomc_analysis.kalshi_client_factory import KalshiClientProtocol

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PARAMS = {
    "alpha_prior": 1.0,
    "beta_prior": 1.0,
    "half_life": 8,
}


class FOMCModelService:
    """Service for managing FOMC prediction models."""

    def __init__(
        self,
        models_dir: Path = Path("data/fomc_models"),
        contract_data_dir: Path = Path("data/kalshi_analysis"),
        model_class: type[BaseModel] = BetaBinomialModel,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        self.models_dir = Path(models_dir)
        self.contract_data_dir = Path(contract_data_dir)
        self.model_class = model_class
        self.model_params = model_params or DEFAULT_MODEL_PARAMS.copy()

        self._model: Optional[BaseModel] = None
        self._contract_words: List[ContractWord] = []
        self._contract_data: List[Dict[str, Any]] = []
        self._last_trained: Optional[datetime] = None

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.contract_data_dir.mkdir(parents=True, exist_ok=True)

    def load_contract_data(self, kalshi_client: Optional[KalshiClientProtocol] = None) -> None:
        """Load contract data from Kalshi or cached file."""
        contract_words_file = self.contract_data_dir / "contract_words.json"

        if contract_words_file.exists():
            logger.info("Loading cached contract data from %s", contract_words_file)
            data = json.loads(contract_words_file.read_text())
            self._contract_data = data
            self._contract_words = [
                ContractWord(
                    word=c["word"],
                    market_ticker=c["market_ticker"],
                    market_title=c["market_title"],
                    variants=c.get("variants", []),
                    threshold=c.get("threshold"),
                    markets=c.get("markets", []),
                )
                for c in data
            ]
        elif kalshi_client is not None:
            logger.info("Fetching contract data from Kalshi API")
            self._contract_words = fetch_mention_contracts(
                kalshi_client,
                series_ticker="KXFEDMENTION",
                market_status=None,
            )
            self._contract_data = [c.to_dict() for c in self._contract_words]
            contract_words_file.write_text(
                json.dumps(self._contract_data, indent=2, ensure_ascii=False)
            )
        else:
            logger.warning("No contract data available and no Kalshi client provided")

    def train_model(self) -> bool:
        """Train the FOMC prediction model on historical data."""
        if not self._contract_data:
            logger.warning("No contract data to train on")
            return False

        try:
            result = generate_upcoming_predictions(
                self._contract_data,
                self.model_class,
                self.model_params,
            )
            self._last_trained = datetime.now(timezone.utc)

            model_file = self.models_dir / "fomc_model.json"
            model_file.write_text(json.dumps(result, indent=2, default=str))

            logger.info(
                "Model trained with %d predictions",
                len(result.get("predictions", [])),
            )
            return True
        except Exception as e:
            logger.error("Failed to train model: %s", e)
            return False

    def get_predictions(
        self,
        kalshi_client: Optional[KalshiClientProtocol] = None,
    ) -> Dict[str, Any]:
        """Get predictions for the next FOMC meeting."""
        if not self._contract_data:
            self.load_contract_data(kalshi_client)

        if not self._contract_data:
            return {
                "predictions": [],
                "metadata": {"error": "No contract data available"},
            }

        try:
            result = generate_upcoming_predictions(
                self._contract_data,
                self.model_class,
                self.model_params,
            )
            return result
        except Exception as e:
            logger.error("Failed to generate predictions: %s", e)
            return {
                "predictions": [],
                "metadata": {"error": str(e)},
            }

    def get_edges(
        self,
        min_edge: float = 0.05,
        kalshi_client: Optional[KalshiClientProtocol] = None,
    ) -> List[Dict[str, Any]]:
        """Find trading opportunities with positive edge."""
        predictions = self.get_predictions(kalshi_client)
        opportunities = []

        for pred in predictions.get("predictions", []):
            edge = pred.get("edge")
            market_price = pred.get("market_price")

            if edge is None or market_price is None:
                continue

            if abs(edge) >= min_edge:
                signal = "BUY_YES" if edge > 0 else "BUY_NO"
                opportunities.append({
                    "contract": pred.get("contract"),
                    "predicted_probability": pred.get("predicted_probability"),
                    "market_price": market_price,
                    "edge": edge,
                    "signal": signal,
                    "confidence_lower": pred.get("confidence_lower"),
                    "confidence_upper": pred.get("confidence_upper"),
                    "market_ticker": pred.get("ticker"),
                })

        opportunities.sort(key=lambda x: abs(x["edge"]), reverse=True)
        return opportunities

    def get_contracts(
        self,
        kalshi_client: Optional[KalshiClientProtocol] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get FOMC contracts from Kalshi."""
        if not self._contract_words:
            self.load_contract_data(kalshi_client)

        contracts = []
        for cw in self._contract_words:
            for market in cw.markets:
                market_status = str(market.get("status", "")).lower()

                if status and status.lower() != "all":
                    if status.lower() == "active" and market_status not in {"open", "active"}:
                        continue
                    elif status.lower() == "settled" and market_status not in {"settled", "resolved"}:
                        continue

                contracts.append({
                    "market_ticker": market.get("ticker"),
                    "word": cw.word,
                    "threshold": cw.threshold or 1,
                    "status": market_status,
                    "last_price": market.get("last_price"),
                    "yes_bid": market.get("yes_bid"),
                    "yes_ask": market.get("yes_ask"),
                    "expiration_time": market.get("expiration_time"),
                    "result": market.get("result"),
                })

        return contracts

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._last_trained is not None

    @property
    def contract_count(self) -> int:
        """Get number of tracked contracts."""
        return len(self._contract_words)

    @property
    def word_list(self) -> List[str]:
        """Get list of tracked words."""
        return [cw.word for cw in self._contract_words]
