"""Model lifecycle management: train, load, cache, and predict.

On application startup the ModelManager ensures that every known ticker
has trained models available.  It first tries to load persisted models
from disk; if any are missing or corrupt it fetches ground-truth data
from the Kalshi API, trains fresh models, and persists them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from earnings_analysis.ground_truth import (
    EARNINGS_SERIES_TICKERS,
    GroundTruthDataset,
    build_backtest_dataframes,
    fetch_ground_truth,
    load_ground_truth,
    save_ground_truth,
)
from earnings_analysis.models.market_adjusted_model import MarketAdjustedModel
from fomc_analysis.kalshi_client_factory import KalshiClientProtocol

logger = logging.getLogger(__name__)

# Map series ticker -> company ticker
TICKER_MAP: Dict[str, str] = {}
for st in EARNINGS_SERIES_TICKERS:
    for prefix in ("KXEARNINGSMENTION", "KXEARNINGMENTION"):
        if st.startswith(prefix):
            TICKER_MAP[st] = st[len(prefix):]
            break

# Deduplicate to unique company tickers
KNOWN_TICKERS: List[str] = sorted(set(TICKER_MAP.values()))

# Default model hyper-parameters
DEFAULT_MODEL_PARAMS = {
    "alpha_prior": 1.0,
    "beta_prior": 1.0,
    "half_life": 4.0,
    "shrinkage_samples": 3,
    "min_samples_to_trade": 2,
}


class ModelManager:
    """Manage per-ticker, per-word MarketAdjustedModel instances."""

    def __init__(
        self,
        models_dir: Path = Path("data/models"),
        ground_truth_dir: Path = Path("data/ground_truth"),
    ) -> None:
        self.models_dir = models_dir
        self.ground_truth_dir = ground_truth_dir

        # {ticker: {word: MarketAdjustedModel}}
        self.models: Dict[str, Dict[str, MarketAdjustedModel]] = {}

        # Cached ground truth dataset
        self._ground_truth: Optional[GroundTruthDataset] = None

        # Per-ticker training data cached for backtests
        self._features: Dict[str, pd.DataFrame] = {}
        self._outcomes: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def ensure_all_trained(self, client: KalshiClientProtocol) -> None:
        """Ensure every known ticker has trained models.

        Called once at application startup (inside the lifespan).
        """
        logger.info("ModelManager: ensuring all models are trained …")

        # 1. Fetch or load ground truth
        self._ground_truth = self._load_or_fetch_ground_truth(client)

        if not self._ground_truth or not self._ground_truth.contracts:
            logger.warning(
                "No ground truth data available — starting with empty models."
            )
            return

        available_companies = self._ground_truth.companies
        logger.info("Ground truth companies: %s", available_companies)

        # 2. For each company, train/load models
        for company in available_companies:
            self._ensure_ticker_trained(company)

        total = sum(len(words) for words in self.models.values())
        logger.info(
            "ModelManager ready: %d tickers, %d word models loaded.",
            len(self.models),
            total,
        )

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def predict(
        self,
        ticker: str,
        word: str,
        market_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return prediction dict for a single ticker/word pair."""
        ticker = ticker.upper()
        word_key = word.lower()

        ticker_models = self.models.get(ticker, {})
        model = ticker_models.get(word_key)
        if model is None:
            raise KeyError(
                f"No trained model for {ticker}/{word_key}. "
                f"Available words: {list(ticker_models.keys())}"
            )

        pred_df = model.predict(market_price=market_price)
        row = pred_df.iloc[0]
        return row.to_dict()

    def predict_all(
        self,
        ticker: str,
        market_prices: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Predict all words for a ticker, optionally with market prices."""
        ticker = ticker.upper()
        ticker_models = self.models.get(ticker, {})
        if not ticker_models:
            raise KeyError(
                f"No models for ticker {ticker}. "
                f"Available: {list(self.models.keys())}"
            )

        results = []
        for word, model in ticker_models.items():
            mp = (market_prices or {}).get(word)
            pred_df = model.predict(market_price=mp)
            row = pred_df.iloc[0].to_dict()
            row["word"] = word
            results.append(row)
        return results

    # ------------------------------------------------------------------
    # Training data access (for backtests)
    # ------------------------------------------------------------------

    def get_training_data(
        self, ticker: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (features, outcomes) DataFrames for a ticker."""
        ticker = ticker.upper()
        if ticker in self._features and ticker in self._outcomes:
            return self._features[ticker], self._outcomes[ticker]
        raise KeyError(f"No training data cached for {ticker}")

    # ------------------------------------------------------------------
    # Retrain
    # ------------------------------------------------------------------

    def retrain(self, ticker: str, client: KalshiClientProtocol) -> None:
        """Force a full retrain for a single ticker."""
        ticker = ticker.upper()
        logger.info("Retraining models for %s …", ticker)
        self._ground_truth = self._load_or_fetch_ground_truth(
            client, force_fetch=True
        )
        self._ensure_ticker_trained(ticker, force=True)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> List[Dict[str, Any]]:
        """Return training status for each ticker."""
        statuses = []
        for ticker in KNOWN_TICKERS:
            words = list(self.models.get(ticker, {}).keys())
            statuses.append(
                {
                    "ticker": ticker,
                    "words": words,
                    "trained": len(words) > 0,
                    "n_words": len(words),
                }
            )
        return statuses

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_or_fetch_ground_truth(
        self,
        client: KalshiClientProtocol,
        force_fetch: bool = False,
    ) -> Optional[GroundTruthDataset]:
        """Load ground truth from disk or fetch from Kalshi API."""
        gt_file = self.ground_truth_dir / "ground_truth.json"

        if gt_file.exists() and not force_fetch:
            try:
                dataset = load_ground_truth(self.ground_truth_dir)
                logger.info(
                    "Loaded %d settled contracts from disk.",
                    len(dataset.contracts),
                )
                return dataset
            except Exception:
                logger.warning(
                    "Corrupt ground truth on disk — re-fetching.",
                    exc_info=True,
                )

        # Fetch from API
        try:
            dataset = fetch_ground_truth(client=client)
            if dataset.contracts:
                save_ground_truth(dataset, self.ground_truth_dir)
                logger.info(
                    "Fetched and saved %d settled contracts.",
                    len(dataset.contracts),
                )
            return dataset
        except Exception:
            logger.error("Failed to fetch ground truth from Kalshi API.", exc_info=True)
            # Last resort: try disk even if it was corrupt before
            if gt_file.exists():
                try:
                    return load_ground_truth(self.ground_truth_dir)
                except Exception:
                    pass
            return None

    def _ensure_ticker_trained(
        self, company: str, force: bool = False
    ) -> None:
        """Train or load models for a single company ticker."""
        company = company.upper()
        model_dir = self.models_dir / company

        if self._ground_truth is None:
            return

        # Build training dataframes
        features, outcomes, _market_prices = build_backtest_dataframes(
            self._ground_truth, company
        )

        if outcomes.empty:
            logger.info("No outcome data for %s — skipping.", company)
            return

        # Cache training data for backtester
        self._features[company] = features
        self._outcomes[company] = outcomes

        # Train / load a model per word
        self.models.setdefault(company, {})

        for word in outcomes.columns:
            model_path = model_dir / f"{word}.json"

            if model_path.exists() and not force:
                try:
                    model = MarketAdjustedModel(**DEFAULT_MODEL_PARAMS)
                    model.load(str(model_path))
                    self.models[company][word] = model
                    continue
                except Exception:
                    logger.warning(
                        "Corrupt model file %s — retraining.", model_path
                    )

            # Train from scratch
            y = outcomes[word].dropna()
            if len(y) == 0:
                continue

            model = MarketAdjustedModel(**DEFAULT_MODEL_PARAMS)
            x = features.loc[y.index]
            model.fit(x, y)
            model.save(str(model_path))
            self.models[company][word] = model

        logger.info(
            "Ticker %s: %d word models ready.", company, len(self.models[company])
        )
