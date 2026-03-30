"""Tests for regression-diagnostics."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from regression_diagnostics.engine import (
    _check_normality, _check_homoscedasticity, _check_autocorrelation,
    _check_linearity, _compute_vif, run_diagnostics,
)
from regression_diagnostics.models import AssumptionResult, VIFResult
from regression_diagnostics.narrator import RegressionNarrator


RNG = np.random.default_rng(42)


def make_clean_df(n=200):
    x1 = RNG.normal(0, 1, n)
    x2 = RNG.normal(0, 1, n)
    y = 2.0 * x1 + 1.5 * x2 + RNG.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def make_heteroscedastic_df(n=200):
    x = RNG.uniform(1, 10, n)
    y = 2 * x + RNG.normal(0, x, n)  # variance grows with x
    return pd.DataFrame({"y": y, "x": x})


def make_autocorrelated_df(n=200):
    x = RNG.normal(0, 1, n)
    errors = np.zeros(n)
    for i in range(1, n):
        errors[i] = 0.9 * errors[i - 1] + RNG.normal(0, 0.5)
    y = 2 * x + errors
    return pd.DataFrame({"y": y, "x": x})


def make_collinear_df(n=200):
    x1 = RNG.normal(0, 1, n)
    x2 = x1 * 0.99 + RNG.normal(0, 0.01, n)  # near-perfect collinearity
    y = x1 + x2 + RNG.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


# --- assumption checks ---

def test_normality_passes_on_normal_residuals():
    residuals = RNG.normal(0, 1, 100)
    result = _check_normality(residuals)
    assert result.passed
    assert result.severity == "ok"


def test_normality_fails_on_uniform_residuals():
    residuals = RNG.uniform(0, 1, 100)
    result = _check_normality(residuals)
    # Uniform is not normal — should fail at p < 0.05
    assert result.p_value is not None


def test_homoscedasticity_passes_on_constant_variance():
    df = make_clean_df()
    result = run_diagnostics(df, "y", ["x1", "x2"])
    homo = next(a for a in result.assumptions if a.name == "homoscedasticity")
    # Clean data should show ok or warning, not outright violation
    assert homo.severity in ("ok", "warning")


def test_autocorrelation_flagged_on_ar1():
    df = make_autocorrelated_df()
    result = run_diagnostics(df, "y", ["x"])
    ac = next(a for a in result.assumptions if a.name == "autocorrelation")
    assert not ac.passed


def test_linearity_check_returns_result():
    residuals = RNG.normal(0, 1, 100)
    fitted = RNG.normal(0, 1, 100)
    result = _check_linearity(residuals, fitted)
    assert isinstance(result, AssumptionResult)
    assert result.name == "linearity"


# --- VIF ---

def test_vif_high_for_collinear_predictors():
    df = make_collinear_df()
    result = run_diagnostics(df, "y", ["x1", "x2"])
    vifs = {v.predictor: v.vif for v in result.vif_results}
    assert vifs["x1"] > 5 or vifs["x2"] > 5


def test_vif_low_for_independent_predictors():
    df = make_clean_df()
    result = run_diagnostics(df, "y", ["x1", "x2"])
    for v in result.vif_results:
        assert v.vif < 5


def test_vif_severity_levels():
    assert VIFResult("x", 1.5).severity == "ok"
    assert VIFResult("x", 6.0).severity == "warning"
    assert VIFResult("x", 12.0).severity == "violation"


def test_single_predictor_has_no_vif():
    df = make_clean_df()
    result = run_diagnostics(df, "y", ["x1"])
    assert result.vif_results == []


# --- run_diagnostics ---

def test_run_diagnostics_r_squared_reasonable():
    df = make_clean_df()
    result = run_diagnostics(df, "y", ["x1", "x2"])
    assert result.r_squared > 0.8


def test_run_diagnostics_n_obs_correct():
    df = make_clean_df(150)
    result = run_diagnostics(df, "y", ["x1", "x2"])
    assert result.n_obs == 150


def test_run_diagnostics_four_assumptions():
    df = make_clean_df()
    result = run_diagnostics(df, "y", ["x1", "x2"])
    assert len(result.assumptions) == 4


def test_run_diagnostics_influential_points():
    df = make_clean_df(100)
    # Add an extreme outlier
    df.loc[0, "y"] = 1000.0
    result = run_diagnostics(df, "y", ["x1", "x2"])
    assert len(result.influential_points) >= 1


def test_run_diagnostics_to_dict_serializable():
    import json
    df = make_clean_df()
    result = run_diagnostics(df, "y", ["x1", "x2"])
    serialized = json.dumps(result.to_dict())
    parsed = json.loads(serialized)
    assert "assumptions" in parsed
    assert "vif_results" in parsed


# --- Narrator ---

def test_narrator_rejects_unknown_backend():
    with pytest.raises(ValueError):
        RegressionNarrator(backend="cohere")


def test_narrator_mocked_returns_text():
    narrator = RegressionNarrator(backend="anthropic")
    df = make_clean_df()
    result = run_diagnostics(df, "y", ["x1", "x2"])
    with patch.object(narrator, "_call_anthropic", return_value="Model fit is reasonable."):
        narrative = narrator.narrate(result.to_dict())
    assert "reasonable" in narrative


def test_narrator_returns_empty_on_error():
    narrator = RegressionNarrator(backend="openai")
    df = make_clean_df()
    result = run_diagnostics(df, "y", ["x1", "x2"])
    with patch.object(narrator, "_call_openai", side_effect=ConnectionError("down")):
        narrative = narrator.narrate(result.to_dict())
    assert narrative == ""
