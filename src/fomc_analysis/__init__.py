"""
fomc_analysis
=============

This package provides a suite of tools for analysing Federal Reserve
press conference transcripts, extracting the words spoken by
FOMC chairs, mapping those words to prediction‑market
contracts, estimating the probability that a word will be mentioned
in a future press conference and backtesting trading strategies
against historical price data.

Key modules include:

* :mod:`fomc_analysis.data_loader` – functions for loading and
  parsing transcripts from PDF or plain‑text files, and for
  extracting chair-only remarks.
* :mod:`fomc_analysis.contract_mapping` – classes and utilities
  for loading and manipulating the mapping from market contract names
  to lists of phrase synonyms.
* :mod:`fomc_analysis.feature_extraction` – routines to count
  mention frequencies, compute recency‑weighted probabilities and
  assemble feature matrices for modelling.
* :mod:`fomc_analysis.model` – simple probabilistic models (EWMA,
  Beta–Binomial, logistic regression) for estimating the likelihood
  of a mention, and wrappers for scikit‑learn models.
* :mod:`fomc_analysis.backtester` – a flexible backtester that
  applies model outputs to historical market prices, respecting
  bid/ask spreads, edge thresholds and position sizing rules.
* :mod:`fomc_analysis.kalshi_api` – a lightweight wrapper for
  interacting with the Kalshi API, including functions to download
  historical price data.  API credentials must be supplied by the
  user.

See the README.md file for a
complete overview, installation instructions and usage examples.

"""

from .backtester import Backtester, TradeResult
from .contract_mapping import ContractMapping, load_mapping_from_file
from .data_loader import Transcript, extract_powell_text, load_transcripts
from .feature_extraction import (
    beta_binomial_estimator,
    compute_binary_events,
    count_mentions,
    ewma_probabilities,
)
from .kalshi_api import KalshiClient
from .model import BetaBinomialModel, EstimateModel, EwmaModel, LogisticRegressionModel

__all__ = [
    "Transcript",
    "load_transcripts",
    "extract_powell_text",
    "ContractMapping",
    "load_mapping_from_file",
    "count_mentions",
    "compute_binary_events",
    "ewma_probabilities",
    "beta_binomial_estimator",
    "EstimateModel",
    "EwmaModel",
    "BetaBinomialModel",
    "LogisticRegressionModel",
    "Backtester",
    "TradeResult",
    "KalshiClient",
]
