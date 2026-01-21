import pytest

from fomc_analysis.models import BetaBinomialModel
from fomc_analysis.predictor import generate_upcoming_predictions


def _sample_contract_data():
    return [
        {
            "word": "Inflation",
            "threshold": 1,
            "markets": [
                {
                    "ticker": "INF-2024-01",
                    "status": "resolved",
                    "result": "yes",
                    "expiration_date": "2024-01-01",
                    "close_price": 0.7,
                },
                {
                    "ticker": "INF-2024-02",
                    "status": "settled",
                    "result": "no",
                    "expiration_date": "2024-03-01",
                    "close_price": 0.1,
                },
                {
                    "ticker": "INF-2024-04",
                    "status": "open",
                    "expiration_date": "2024-04-01",
                    "last_price_dollars": "0.45",
                    "event_ticker": "kxfedmention-2024apr",
                },
            ],
        },
        {
            "word": "Recession",
            "threshold": 2,
            "markets": [
                {
                    "ticker": "REC-2024-01",
                    "status": "resolved",
                    "result": "no",
                    "expiration_date": "2024-01-01",
                    "close_price": 0.2,
                },
                {
                    "ticker": "REC-2024-02",
                    "status": "open",
                    "expiration_date": "2024-02-01",
                    "last_price": 30,
                },
            ],
        },
        {
            "word": "New Term",
            "threshold": 1,
            "markets": [
                {
                    "ticker": "NEW-2024-05",
                    "status": "open",
                    "expiration_date": "2024-05-01",
                    "last_price": 60,
                }
            ],
        },
    ]


def test_generate_upcoming_predictions_returns_forecasts():
    contract_data = _sample_contract_data()

    result = generate_upcoming_predictions(
        contract_data,
        BetaBinomialModel,
        {"alpha_prior": 1.0, "beta_prior": 1.0, "half_life": 2},
    )

    predictions = result["predictions"]

    assert len(predictions) == 2  # Only two markets have historical data
    inflation = next(p for p in predictions if p["contract"] == "Inflation")
    recession = next(p for p in predictions if p["contract"] == "Recession (2+)")

    assert inflation["market_price"] == pytest.approx(0.45)
    assert inflation["edge"] == pytest.approx(
        inflation["predicted_probability"] - inflation["market_price"]
    )
    assert inflation["event_ticker"] == "kxfedmention-2024apr"

    assert recession["market_price"] == pytest.approx(0.30)
    assert recession["edge"] == pytest.approx(
        recession["predicted_probability"] - 0.30
    )

    metadata = result["metadata"]
    assert metadata["contracts_with_history"] == 2
    assert "New Term" in metadata["skipped_contracts"]
