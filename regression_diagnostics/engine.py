"""Regression diagnostics engine."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from regression_diagnostics.models import (
    AssumptionResult,
    DiagnosticsResult,
    InfluentialPoint,
    VIFResult,
)


def _ols_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Fit OLS, return (beta, fitted, residuals, r2, adj_r2)."""
    n, p = X.shape
    beta, residuals_ss, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residuals = y - fitted

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p) if n > p else r2

    return beta, fitted, residuals, r2, adj_r2


def _check_normality(residuals: np.ndarray) -> AssumptionResult:
    """Shapiro-Wilk test for normality of residuals."""
    # Shapiro-Wilk is only valid for n <= 5000; use KS test for larger samples
    n = len(residuals)
    if n <= 5000:
        stat, p = stats.shapiro(residuals)
        test_name = "Shapiro-Wilk"
    else:
        stat, p = stats.kstest(residuals, "norm", args=(np.mean(residuals), np.std(residuals)))
        test_name = "Kolmogorov-Smirnov"

    passed = p > 0.05
    severity = "ok" if passed else ("warning" if p > 0.01 else "violation")
    return AssumptionResult(
        name="normality",
        passed=passed,
        statistic=float(stat),
        p_value=float(p),
        detail=f"{test_name}: stat={stat:.4f}, p={p:.4f}. "
               f"{'Residuals appear normally distributed.' if passed else 'Residuals deviate from normality.'}",
        severity=severity,
    )


def _check_homoscedasticity(residuals: np.ndarray, fitted: np.ndarray) -> AssumptionResult:
    """Breusch-Pagan test for homoscedasticity."""
    n = len(residuals)
    sq_resid = residuals ** 2
    # Regress squared residuals on fitted values
    X_bp = np.column_stack([np.ones(n), fitted])
    _, _, resid_bp, r2_bp, _ = _ols_fit(X_bp, sq_resid)

    lm_stat = n * r2_bp
    p = float(stats.chi2.sf(lm_stat, df=1))
    passed = p > 0.05
    severity = "ok" if passed else ("warning" if p > 0.01 else "violation")
    return AssumptionResult(
        name="homoscedasticity",
        passed=passed,
        statistic=float(lm_stat),
        p_value=p,
        detail=f"Breusch-Pagan: LM={lm_stat:.4f}, p={p:.4f}. "
               f"{'Constant variance supported.' if passed else 'Evidence of heteroscedasticity.'}",
        severity=severity,
    )


def _check_autocorrelation(residuals: np.ndarray) -> AssumptionResult:
    """Durbin-Watson test for autocorrelation of residuals."""
    diff = np.diff(residuals)
    dw = float(np.sum(diff ** 2) / np.sum(residuals ** 2))
    # DW ~ 2 means no autocorrelation; < 1.5 or > 2.5 is concerning
    passed = 1.5 <= dw <= 2.5
    severity = "ok" if passed else ("warning" if 1.0 <= dw <= 3.0 else "violation")
    direction = "positive" if dw < 1.5 else ("negative" if dw > 2.5 else "none")
    return AssumptionResult(
        name="autocorrelation",
        passed=passed,
        statistic=dw,
        p_value=None,
        detail=f"Durbin-Watson: {dw:.4f}. "
               f"{'No autocorrelation detected.' if passed else f'Possible {direction} autocorrelation.'}",
        severity=severity,
    )


def _check_linearity(residuals: np.ndarray, fitted: np.ndarray) -> AssumptionResult:
    """Check linearity via correlation between residuals and fitted values (should be ~0)."""
    corr, p = stats.pearsonr(fitted, residuals)
    passed = abs(corr) < 0.1
    severity = "ok" if abs(corr) < 0.1 else ("warning" if abs(corr) < 0.3 else "violation")
    return AssumptionResult(
        name="linearity",
        passed=passed,
        statistic=float(corr),
        p_value=float(p),
        detail=f"Residual-fitted correlation: r={corr:.4f}, p={p:.4f}. "
               f"{'Linear relationship supported.' if passed else 'Possible non-linearity detected.'}",
        severity=severity,
    )


def _compute_vif(X_no_intercept: np.ndarray, predictor_names: list[str]) -> list[VIFResult]:
    """Compute Variance Inflation Factor for each predictor."""
    results = []
    n_predictors = X_no_intercept.shape[1]
    for i in range(n_predictors):
        y_i = X_no_intercept[:, i]
        others = np.delete(X_no_intercept, i, axis=1)
        X_i = np.column_stack([np.ones(len(y_i)), others])
        _, _, _, r2, _ = _ols_fit(X_i, y_i)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        results.append(VIFResult(predictor=predictor_names[i], vif=float(vif)))
    return results


def _compute_cooks_distance(
    X: np.ndarray, y: np.ndarray, residuals: np.ndarray, fitted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Cook's distance and leverage (hat values) for each observation."""
    n, p = X.shape
    # Hat matrix diagonal (leverage)
    try:
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        h = np.diag(H)
    except np.linalg.LinAlgError:
        h = np.full(n, p / n)

    mse = float(np.sum(residuals ** 2) / max(n - p, 1))
    cooks = (residuals ** 2 / (p * mse)) * (h / (1 - h + 1e-10) ** 2)
    return cooks, h


def run_diagnostics(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    cooks_threshold: Optional[float] = None,
    llm_backend: Optional[str] = None,
) -> DiagnosticsResult:
    """
    Run full OLS regression diagnostics.

    Parameters
    ----------
    df:         DataFrame containing outcome and predictor columns.
    outcome:    Name of the dependent variable column.
    predictors: List of predictor column names.
    cooks_threshold:
                Cook's distance threshold for influential points.
                Defaults to 4/n (a common rule of thumb).
    llm_backend:
                Optional LLM backend for narrative interpretation.
    """
    subset = df[[outcome] + predictors].dropna()
    y = subset[outcome].to_numpy(dtype=float)
    X_raw = subset[predictors].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), X_raw])  # add intercept

    n = len(y)
    p = len(predictors)
    threshold = cooks_threshold or 4.0 / n

    _, fitted, residuals, r2, adj_r2 = _ols_fit(X, y)

    assumptions = [
        _check_normality(residuals),
        _check_homoscedasticity(residuals, fitted),
        _check_autocorrelation(residuals),
        _check_linearity(residuals, fitted),
    ]

    vif_results = _compute_vif(X_raw, predictors) if p > 1 else []

    cooks, leverage = _compute_cooks_distance(X, y, residuals, fitted)
    std_resid = residuals / (np.std(residuals) + 1e-10)
    influential = [
        InfluentialPoint(
            index=int(subset.index[i]),
            cooks_distance=float(cooks[i]),
            leverage=float(leverage[i]),
            standardized_residual=float(std_resid[i]),
        )
        for i in range(n)
        if cooks[i] > threshold
    ]

    violations = [a for a in assumptions if a.severity == "violation"]
    vif_violations = [v for v in vif_results if v.severity == "violation"]
    overall_passed = len(violations) == 0 and len(vif_violations) == 0

    result = DiagnosticsResult(
        n_obs=n,
        n_predictors=p,
        r_squared=r2,
        adj_r_squared=adj_r2,
        assumptions=assumptions,
        vif_results=vif_results,
        influential_points=influential,
        overall_passed=overall_passed,
    )

    if llm_backend:
        try:
            from regression_diagnostics.narrator import RegressionNarrator
            narrator = RegressionNarrator(backend=llm_backend)
            result.narrative = narrator.narrate(result.to_dict())
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("LLM narration failed: %s", exc)

    return result
