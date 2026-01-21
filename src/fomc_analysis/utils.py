"""
utils
=====

Miscellaneous utility functions used throughout the package.
These helpers perform simple conversions and validations.
"""

def implied_prob_from_price(price_cents: float) -> float:
    """Convert a Kalshi YES price in cents to a probability.

    Parameters
    ----------
    price_cents: float
        Price of the YES contract in cents (0–100).

    Returns
    -------
    float
        Probability implied by the price.
    """
    return price_cents / 100.0


def price_from_prob(prob: float) -> float:
    """Convert a probability to a price in cents.

    Parameters
    ----------
    prob: float
        Probability (0–1).

    Returns
    -------
    float
        Price of a YES contract in cents.
    """
    return prob * 100.0