"""FastAPI application with lifespan for model initialization.

Run with:
    uvicorn earnings_analysis.api.app:app --reload

Or via the CLI entry point:
    earnings-api
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from earnings_analysis.api.routers import backtests, contracts, edges, predictions
from earnings_analysis.api.schemas import HealthResponse, ModelStatus
from earnings_analysis.api.services.model_manager import KNOWN_TICKERS, ModelManager
from fomc_analysis.kalshi_client_factory import get_kalshi_client

# FOMC API imports
from fomc_analysis.api.routers import (
    predictions_router as fomc_predictions_router,
    word_frequencies_router as fomc_word_frequencies_router,
    transcripts_router as fomc_transcripts_router,
    backtests_router as fomc_backtests_router,
    contracts_router as fomc_contracts_router,
    edges_router as fomc_edges_router,
)
from fomc_analysis.api.services import FOMCModelService, FOMCDataService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle.

    On startup:
      1. Initialize the Kalshi client.
      2. Create the ModelManager and ensure all models are trained.
         - If ground truth data is missing on disk, fetch from Kalshi API.
         - If any model files are missing or corrupt, retrain and persist.
      3. Initialize FOMC model and data services.

    The Kalshi client and ModelManager are stored in ``app.state`` for
    dependency injection via ``dependencies.py``.
    """
    logger.info("Starting Kalshi Analysis API (Earnings + FOMC) …")

    # Create data directories if needed
    Path("data/models").mkdir(parents=True, exist_ok=True)
    Path("data/ground_truth").mkdir(parents=True, exist_ok=True)
    Path("data/backtest_results").mkdir(parents=True, exist_ok=True)
    Path("data/fomc_models").mkdir(parents=True, exist_ok=True)
    Path("data/kalshi_analysis").mkdir(parents=True, exist_ok=True)

    # Initialize Kalshi client
    try:
        kalshi_client = get_kalshi_client()
        app.state.kalshi_connected = True
        logger.info("Kalshi client initialized successfully.")
    except Exception as exc:
        logger.error("Failed to initialize Kalshi client: %s", exc)
        app.state.kalshi_connected = False
        kalshi_client = None

    app.state.kalshi_client = kalshi_client

    # Initialize Earnings ModelManager and train/load models
    model_manager = ModelManager(
        models_dir=Path("data/models"),
        ground_truth_dir=Path("data/ground_truth"),
    )

    if kalshi_client is not None:
        try:
            # Run in thread pool since Kalshi client is sync
            await asyncio.to_thread(model_manager.ensure_all_trained, kalshi_client)
        except Exception:
            logger.error("Error during earnings model training", exc_info=True)
    else:
        # Try loading existing models from disk even without Kalshi connection
        logger.warning(
            "No Kalshi client — attempting to load existing models from disk."
        )
        try:
            await asyncio.to_thread(model_manager.ensure_all_trained, None)
        except Exception:
            logger.warning("Could not load earnings models without Kalshi client.")

    app.state.model_manager = model_manager

    n_tickers = len(model_manager.models)
    n_words = sum(len(w) for w in model_manager.models.values())
    logger.info("Earnings models loaded: %d tickers, %d word models.", n_tickers, n_words)

    # Initialize FOMC services
    fomc_model_service = FOMCModelService(
        models_dir=Path("data/fomc_models"),
        contract_data_dir=Path("data/kalshi_analysis"),
    )
    fomc_data_service = FOMCDataService(
        transcripts_dir=Path("data/transcripts"),
        segments_dir=Path("data/segments"),
        contract_mapping_file=Path("configs/contract_mapping.yaml"),
        word_freq_file=Path("results/backtest_v3/word_frequency_timeseries.csv"),
    )

    # Load FOMC contract data
    try:
        await asyncio.to_thread(fomc_model_service.load_contract_data, kalshi_client)
        logger.info("FOMC model service initialized: %d contracts", fomc_model_service.contract_count)
    except Exception:
        logger.warning("Could not load FOMC contract data", exc_info=True)

    app.state.fomc_model_service = fomc_model_service
    app.state.fomc_data_service = fomc_data_service

    logger.info("Startup complete.")

    yield

    # Shutdown
    logger.info("Shutting down Kalshi Analysis API …")
    if kalshi_client is not None:
        try:
            kalshi_client.close()
        except Exception:
            pass


app = FastAPI(
    title="Kalshi Analysis API",
    description=(
        "API for predicting Kalshi mention contract outcomes (Earnings + FOMC), "
        "finding market edges, accessing transcripts and word frequencies, "
        "and running backtests."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Earnings routers under /mentions/earnings
app.include_router(predictions.router, prefix="/api/v1/mentions/earnings")
app.include_router(edges.router, prefix="/api/v1/mentions/earnings")
app.include_router(backtests.router, prefix="/api/v1/mentions/earnings")
app.include_router(contracts.router, prefix="/api/v1/mentions/earnings")

# Include FOMC routers under /mentions/fomc
app.include_router(fomc_predictions_router, prefix="/api/v1/mentions/fomc")
app.include_router(fomc_word_frequencies_router, prefix="/api/v1/mentions/fomc")
app.include_router(fomc_transcripts_router, prefix="/api/v1/mentions/fomc")
app.include_router(fomc_backtests_router, prefix="/api/v1/mentions/fomc")
app.include_router(fomc_contracts_router, prefix="/api/v1/mentions/fomc")
app.include_router(fomc_edges_router, prefix="/api/v1/mentions/fomc")


@app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint showing model status."""
    mm: ModelManager = app.state.model_manager
    statuses = mm.get_status()

    return HealthResponse(
        status="healthy",
        models_loaded=sum(len(mm.models.get(t, {})) for t in KNOWN_TICKERS),
        tickers=[
            ModelStatus(
                ticker=s["ticker"],
                words=s["words"],
                trained=s["trained"],
                n_words=s["n_words"],
            )
            for s in statuses
        ],
        kalshi_connected=getattr(app.state, "kalshi_connected", False),
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with links to docs."""
    return {
        "message": "Kalshi Analysis API (Earnings + FOMC)",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/api/v1/health",
        "endpoints": {
            "mentions/earnings": {
                "predictions": "/api/v1/mentions/earnings/predictions/{ticker}",
                "edges": "/api/v1/mentions/earnings/edges/{ticker}",
                "backtests": "/api/v1/mentions/earnings/backtests/{ticker}",
                "contracts": "/api/v1/mentions/earnings/contracts/{ticker}",
            },
            "mentions/fomc": {
                "predictions": "/api/v1/mentions/fomc/predictions",
                "word_frequencies": "/api/v1/mentions/fomc/word-frequencies",
                "transcripts": "/api/v1/mentions/fomc/transcripts",
                "backtests": "/api/v1/mentions/fomc/backtests",
                "contracts": "/api/v1/mentions/fomc/contracts",
                "edges": "/api/v1/mentions/fomc/edges",
            },
        },
    }


def main():
    """Entry point for the ``earnings-api`` console script."""
    uvicorn.run(
        "earnings_analysis.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
