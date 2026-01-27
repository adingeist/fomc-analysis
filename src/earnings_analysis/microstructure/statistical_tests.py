"""
Statistical significance testing for prediction market trading edges.

Implements rigorous hypothesis testing following methodology from Becker (2025),
which used Mann-Whitney U, z-tests, Welch's t-tests, and correlation analysis
with effect sizes (Cohen's d).

This module validates whether observed trading edges are statistically
significant or likely due to noise / small sample sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class SignificanceResult:
    """Result of a statistical significance test for trading edge."""
    n_trades: int
    mean_return: float
    std_return: float
    t_statistic: float
    p_value: float
    cohens_d: float
    ci_95_lower: float
    ci_95_upper: float
    significant_at_05: bool
    significant_at_01: bool

    def summary(self) -> str:
        sig = "***" if self.significant_at_01 else ("**" if self.significant_at_05 else "ns")
        return (
            f"Edge: {self.mean_return:+.4f} ({sig}), "
            f"n={self.n_trades}, t={self.t_statistic:.2f}, "
            f"p={self.p_value:.4f}, d={self.cohens_d:.3f}, "
            f"95% CI [{self.ci_95_lower:+.4f}, {self.ci_95_upper:+.4f}]"
        )


@dataclass
class CalibrationBinResult:
    """Calibration result for a single probability bin."""
    bin_center: float
    predicted_mean: float
    actual_mean: float
    deviation: float
    n_samples: int
    ci_lower: float
    ci_upper: float


@dataclass
class CalibrationResult:
    """Full calibration test result."""
    bins: List[CalibrationBinResult]
    mean_absolute_error: float
    root_mean_squared_error: float
    max_deviation: float
    brier_score: float
    hosmer_lemeshow_stat: float
    hosmer_lemeshow_p: float
    well_calibrated: bool  # True if H-L test p > 0.05


@dataclass
class BrierDecomposition:
    """Brier score decomposed into reliability, resolution, and uncertainty."""
    brier_score: float
    reliability: float     # Lower is better (calibration error)
    resolution: float      # Higher is better (discriminative ability)
    uncertainty: float     # Base rate uncertainty (fixed for dataset)
    reliability_pct: float  # % of Brier from calibration error
    resolution_pct: float   # % of Brier from discrimination
    sharpness: float        # Avg distance of predictions from 0.5


def test_edge_significance(trade_returns: List[float]) -> SignificanceResult:
    """
    Test if average trade return is significantly different from zero.

    Uses a one-sample t-test with H0: mean(returns) = 0.

    Parameters
    ----------
    trade_returns : list of float
        Per-trade returns (ROI). Positive = profit.

    Returns
    -------
    SignificanceResult
        Detailed significance test results.
    """
    returns = np.array(trade_returns, dtype=float)
    n = len(returns)

    if n < 2:
        return SignificanceResult(
            n_trades=n,
            mean_return=float(np.mean(returns)) if n > 0 else 0.0,
            std_return=0.0,
            t_statistic=0.0,
            p_value=1.0,
            cohens_d=0.0,
            ci_95_lower=0.0,
            ci_95_upper=0.0,
            significant_at_05=False,
            significant_at_01=False,
        )

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1))

    # One-sample t-test: H0 = mean return is 0
    t_stat, p_value = stats.ttest_1samp(returns, 0.0)

    # Effect size (Cohen's d)
    cohens_d = mean_ret / std_ret if std_ret > 0 else 0.0

    # 95% confidence interval
    sem = stats.sem(returns)
    ci_lower, ci_upper = stats.t.interval(
        0.95, df=n - 1, loc=mean_ret, scale=sem
    )

    return SignificanceResult(
        n_trades=n,
        mean_return=mean_ret,
        std_return=std_ret,
        t_statistic=float(t_stat),
        p_value=float(p_value),
        cohens_d=float(cohens_d),
        ci_95_lower=float(ci_lower),
        ci_95_upper=float(ci_upper),
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
    )


def test_yes_no_asymmetry(
    yes_returns: List[float],
    no_returns: List[float],
) -> Dict:
    """
    Test if YES and NO trade returns are significantly different.

    Becker found makers buying NO earn +1.25% vs +0.77% buying YES.
    This tests whether our strategy shows a similar directional asymmetry.

    Parameters
    ----------
    yes_returns : list of float
        Returns from YES-side trades.
    no_returns : list of float
        Returns from NO-side trades.

    Returns
    -------
    dict
        Test results including Welch's t-test, Mann-Whitney U, and effect size.
    """
    yes = np.array(yes_returns, dtype=float)
    no = np.array(no_returns, dtype=float)

    if len(yes) < 2 or len(no) < 2:
        return {
            "n_yes": len(yes),
            "n_no": len(no),
            "yes_mean": float(np.mean(yes)) if len(yes) > 0 else 0.0,
            "no_mean": float(np.mean(no)) if len(no) > 0 else 0.0,
            "significant": False,
            "test": "insufficient_data",
        }

    # Welch's t-test (unequal variances)
    t_stat, p_value = stats.ttest_ind(no, yes, equal_var=False)

    # Mann-Whitney U (non-parametric)
    u_stat, u_p_value = stats.mannwhitneyu(no, yes, alternative="greater")

    # Cohen's d for effect size
    pooled_std = np.sqrt(
        ((len(yes) - 1) * np.var(yes, ddof=1) + (len(no) - 1) * np.var(no, ddof=1))
        / (len(yes) + len(no) - 2)
    )
    cohens_d = (np.mean(no) - np.mean(yes)) / pooled_std if pooled_std > 0 else 0.0

    return {
        "n_yes": len(yes),
        "n_no": len(no),
        "yes_mean": float(np.mean(yes)),
        "no_mean": float(np.mean(no)),
        "yes_std": float(np.std(yes, ddof=1)),
        "no_std": float(np.std(no, ddof=1)),
        "difference": float(np.mean(no) - np.mean(yes)),
        "welch_t": float(t_stat),
        "welch_p": float(p_value),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(u_p_value),
        "cohens_d": float(cohens_d),
        "significant": p_value < 0.05,
        "no_favored": float(np.mean(no)) > float(np.mean(yes)),
    }


def test_calibration(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 10,
) -> CalibrationResult:
    """
    Test model calibration using binned analysis and Hosmer-Lemeshow test.

    A well-calibrated model predicts probabilities that match observed rates:
    when it says 70%, the event should happen ~70% of the time.

    Parameters
    ----------
    predictions : list of float
        Model predicted probabilities (0-1).
    outcomes : list of int
        Actual binary outcomes (0 or 1).
    n_bins : int
        Number of calibration bins (default: 10).

    Returns
    -------
    CalibrationResult
        Calibration test results with bins and H-L test.
    """
    preds = np.array(predictions, dtype=float)
    acts = np.array(outcomes, dtype=float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_results = []

    hl_stat = 0.0  # Hosmer-Lemeshow statistic

    for i in range(n_bins):
        mask = (preds >= bins[i]) & (preds < bins[i + 1])
        if i == n_bins - 1:
            mask = mask | (preds == bins[i + 1])

        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue

        pred_mean = float(preds[mask].mean())
        act_mean = float(acts[mask].mean())
        deviation = act_mean - pred_mean

        # Wilson score interval for actual rate
        if n_in_bin > 0:
            z = 1.96
            p_hat = act_mean
            denom = 1 + z ** 2 / n_in_bin
            center = (p_hat + z ** 2 / (2 * n_in_bin)) / denom
            margin = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n_in_bin)) / n_in_bin) / denom
            ci_lower = max(0, center - margin)
            ci_upper = min(1, center + margin)
        else:
            ci_lower = ci_upper = 0.0

        bin_results.append(CalibrationBinResult(
            bin_center=(bins[i] + bins[i + 1]) / 2,
            predicted_mean=pred_mean,
            actual_mean=act_mean,
            deviation=deviation,
            n_samples=n_in_bin,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        ))

        # Hosmer-Lemeshow statistic contribution
        expected_pos = pred_mean * n_in_bin
        expected_neg = (1 - pred_mean) * n_in_bin
        observed_pos = acts[mask].sum()
        observed_neg = n_in_bin - observed_pos

        if expected_pos > 0:
            hl_stat += (observed_pos - expected_pos) ** 2 / expected_pos
        if expected_neg > 0:
            hl_stat += (observed_neg - expected_neg) ** 2 / expected_neg

    # H-L test has n_bins - 2 degrees of freedom
    df = max(1, len(bin_results) - 2)
    hl_p_value = float(1 - stats.chi2.cdf(hl_stat, df))

    # Aggregate metrics
    deviations = [abs(b.deviation) for b in bin_results]
    mae = float(np.mean(deviations)) if deviations else 0.0
    rmse = float(np.sqrt(np.mean([d ** 2 for d in deviations]))) if deviations else 0.0
    max_dev = float(max(deviations)) if deviations else 0.0

    # Brier score
    brier = float(np.mean((preds - acts) ** 2))

    return CalibrationResult(
        bins=bin_results,
        mean_absolute_error=mae,
        root_mean_squared_error=rmse,
        max_deviation=max_dev,
        brier_score=brier,
        hosmer_lemeshow_stat=float(hl_stat),
        hosmer_lemeshow_p=hl_p_value,
        well_calibrated=hl_p_value > 0.05,
    )


def test_multiple_edges(
    edge_results: Dict[str, List[float]],
    method: str = "fdr_bh",
) -> Dict[str, Dict]:
    """
    Test multiple trading edges with correction for multiple comparisons.

    When testing edge across many word-ticker combinations, raw p-values
    overstate significance. This applies FDR or Bonferroni correction.

    Parameters
    ----------
    edge_results : dict
        Mapping of contract_name -> list of per-trade returns.
    method : str
        Correction method: "fdr_bh" (Benjamini-Hochberg) or "bonferroni".

    Returns
    -------
    dict
        Per-contract results with corrected p-values and rejection decisions.
    """
    # Run individual tests
    contract_names = []
    p_values = []
    individual_results = {}

    for name, returns in edge_results.items():
        result = test_edge_significance(returns)
        individual_results[name] = result
        contract_names.append(name)
        p_values.append(result.p_value)

    if not p_values:
        return {}

    p_array = np.array(p_values)

    # Apply correction
    if method == "bonferroni":
        corrected_p = np.minimum(p_array * len(p_array), 1.0)
        reject = corrected_p < 0.05
    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR correction
        n = len(p_array)
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]
        corrected_sorted = np.zeros(n)

        for i in range(n):
            corrected_sorted[i] = sorted_p[i] * n / (i + 1)

        # Enforce monotonicity (working backwards)
        for i in range(n - 2, -1, -1):
            corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])

        corrected_p = np.zeros(n)
        corrected_p[sorted_indices] = np.minimum(corrected_sorted, 1.0)
        reject = corrected_p < 0.05
    else:
        raise ValueError(f"Unknown correction method: {method}")

    # Build output
    output = {}
    for i, name in enumerate(contract_names):
        result = individual_results[name]
        output[name] = {
            "raw_result": result,
            "raw_p_value": result.p_value,
            "corrected_p_value": float(corrected_p[i]),
            "significant_after_correction": bool(reject[i]),
            "correction_method": method,
        }

    return output


def compute_brier_decomposition(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 10,
) -> BrierDecomposition:
    """
    Decompose Brier score into reliability, resolution, and uncertainty.

    Brier = Reliability - Resolution + Uncertainty

    - Reliability (lower is better): Measures calibration error.
    - Resolution (higher is better): Measures how much predictions
      discriminate between outcomes.
    - Uncertainty: Base rate variance (fixed for a dataset).

    Parameters
    ----------
    predictions : list of float
        Predicted probabilities (0-1).
    outcomes : list of int
        Binary outcomes (0 or 1).
    n_bins : int
        Bins for decomposition.

    Returns
    -------
    BrierDecomposition
        Decomposed Brier score components.
    """
    preds = np.array(predictions, dtype=float)
    acts = np.array(outcomes, dtype=float)
    n = len(preds)

    if n == 0:
        return BrierDecomposition(
            brier_score=0.0,
            reliability=0.0,
            resolution=0.0,
            uncertainty=0.0,
            reliability_pct=0.0,
            resolution_pct=0.0,
            sharpness=0.0,
        )

    # Base rate
    base_rate = float(np.mean(acts))
    uncertainty = base_rate * (1 - base_rate)

    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        mask = (preds >= bins[i]) & (preds < bins[i + 1])
        if i == n_bins - 1:
            mask = mask | (preds == bins[i + 1])

        n_k = int(mask.sum())
        if n_k == 0:
            continue

        pred_mean = float(preds[mask].mean())
        act_mean = float(acts[mask].mean())

        reliability += n_k * (pred_mean - act_mean) ** 2
        resolution += n_k * (act_mean - base_rate) ** 2

    reliability /= n
    resolution /= n

    brier = float(np.mean((preds - acts) ** 2))
    sharpness = float(np.mean(np.abs(preds - 0.5)))

    rel_pct = (reliability / brier * 100) if brier > 0 else 0.0
    res_pct = (resolution / brier * 100) if brier > 0 else 0.0

    return BrierDecomposition(
        brier_score=brier,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        reliability_pct=rel_pct,
        resolution_pct=res_pct,
        sharpness=sharpness,
    )
