"""
feature_extraction
===================

This module contains functions to convert parsed transcripts and
contract mappings into numerical features for modelling.  It includes
utilities for counting phrase mentions, converting counts to binary
events, computing recency‑weighted probabilities (EWMA), and
implementing a simple Beta–Binomial estimator.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .data_loader import Transcript
from .contract_mapping import ContractMapping


def count_mentions(
    transcripts: List[Transcript], mapping: ContractMapping
) -> pd.DataFrame:
    """Count how many times each contract is mentioned in each transcript.

    Parameters
    ----------
    transcripts: List[Transcript]
        A list of Transcript objects (see :mod:`data_loader`).  The
        transcripts should be sorted chronologically for recency
        weighting to make sense.
    mapping: ContractMapping
        A mapping from contract names to lists of phrase variants.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by transcript date (as strings) with one
        column per contract.  Each entry contains the count of
        mentions of that contract in the Powell‑only text of the
        transcript.  If a transcript does not have a date, its index
        will be the file name.
    """
    rows = []
    index = []
    for t in transcripts:
        text = t.powell_text
        row = {}
        for contract in mapping.contracts():
            row[contract] = mapping.count_in_text(contract, text)
        rows.append(row)
        index.append(t.date or t.file_path.name)
    df = pd.DataFrame(rows, index=index)
    return df


def compute_binary_events(counts: pd.DataFrame, threshold: int = 1) -> pd.DataFrame:
    """Convert counts to binary indicator events.

    Parameters
    ----------
    counts: pandas.DataFrame
        DataFrame of mention counts (output of :func:`count_mentions`).
    threshold: int, default 1
        The minimum count that constitutes a "mention" event.  If
        threshold is 1, any non‑zero count becomes a 1 in the events
        DataFrame.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of the same shape as ``counts`` with 1s where
        counts >= threshold and 0s elsewhere.
    """
    events = (counts >= threshold).astype(int)
    events.index = counts.index
    return events


def ewma_probabilities(events: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """Compute exponentially weighted moving average probabilities.

    For each contract and each transcript, this function computes
    ``p_t = alpha * events_t + (1 - alpha) * p_{t-1}``, starting
    either at 0.5 (uninformative prior) or at the first event
    observation if `events` contains a non‑empty initial value.  You
    can adjust the starting value by supplying an ``init``, but it is
    kept at 0.5 for simplicity.

    Parameters
    ----------
    events: pandas.DataFrame
        DataFrame of binary events (1 if a contract was mentioned at
        least once, 0 otherwise).
    alpha: float, default 0.5
        Smoothing parameter.  Higher values put more weight on
        recent observations.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of the same shape as ``events`` where each entry
        is the EWMA probability estimate for that contract and date.
    """
    probs = pd.DataFrame(index=events.index, columns=events.columns, dtype=float)
    # initialise with 0.5 (uninformative prior)
    p_prev = np.full(events.shape[1], 0.5)
    for i, idx in enumerate(events.index):
        row = events.loc[idx].to_numpy().astype(float)
        p_new = alpha * row + (1.0 - alpha) * p_prev
        probs.loc[idx] = p_new
        p_prev = p_new
    return probs


def beta_binomial_estimator(
    events: pd.DataFrame,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    half_life: Optional[int] = None,
) -> pd.DataFrame:
    """Estimate mention probabilities using a Beta–Binomial model.

    In a Beta–Binomial model the posterior mean probability after
    observing ``n`` events with ``k`` successes is ``(alpha + k) / (alpha
    + beta + n)``.  This function supports optional exponential
    decay so that older observations are downweighted.  When
    ``half_life`` is None, all past events are treated equally.

    Parameters
    ----------
    events: pandas.DataFrame
        DataFrame of binary events (rows are dates, columns are
        contracts).
    alpha_prior: float, default 1.0
        The alpha hyperparameter of the Beta prior.
    beta_prior: float, default 1.0
        The beta hyperparameter of the Beta prior.
    half_life: Optional[int], default None
        If provided, defines the half‑life (in number of pressers)
        for exponential decay.  A smaller half‑life gives more
        weight to recent events.  If None, no decay is applied.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of posterior mean probabilities for each
        contract at each time index.
    """
    n_transcripts = events.shape[0]
    n_contracts = events.shape[1]
    probs = pd.DataFrame(index=events.index, columns=events.columns, dtype=float)
    # initialise counts and totals
    successes = np.zeros(n_contracts, dtype=float)
    totals = np.zeros(n_contracts, dtype=float)
    # compute decay factor per transcript
    if half_life is not None and half_life > 0:
        decay_factor = 0.5 ** (1.0 / half_life)
    else:
        decay_factor = 1.0
    for idx in events.index:
        row = events.loc[idx].to_numpy().astype(float)
        # apply decay to past observations
        successes *= decay_factor
        totals *= decay_factor
        # update counts with current observation
        successes += row
        totals += 1.0
        # posterior mean
        posterior = (alpha_prior + successes) / (alpha_prior + beta_prior + totals)
        probs.loc[idx] = posterior
    return probs