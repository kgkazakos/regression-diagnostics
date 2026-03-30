"""Core data models for regression-diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AssumptionResult:
    name: str
    passed: bool
    statistic: Optional[float]
    p_value: Optional[float]
    detail: str
    severity: str  # "ok" | "warning" | "violation"


@dataclass
class InfluentialPoint:
    index: int
    cooks_distance: float
    leverage: float
    standardized_residual: float


@dataclass
class VIFResult:
    predictor: str
    vif: float

    @property
    def severity(self) -> str:
        if self.vif >= 10:
            return "violation"
        if self.vif >= 5:
            return "warning"
        return "ok"


@dataclass
class DiagnosticsResult:
    n_obs: int
    n_predictors: int
    r_squared: float
    adj_r_squared: float

    assumptions: list[AssumptionResult] = field(default_factory=list)
    vif_results: list[VIFResult] = field(default_factory=list)
    influential_points: list[InfluentialPoint] = field(default_factory=list)

    overall_passed: bool = True
    narrative: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_obs": self.n_obs,
            "n_predictors": self.n_predictors,
            "r_squared": round(float(self.r_squared), 4),
            "adj_r_squared": round(float(self.adj_r_squared), 4),
            "overall_passed": bool(self.overall_passed),
            "assumptions": [
                {
                    "name": a.name,
                    "passed": bool(a.passed),
                    "statistic": round(float(a.statistic), 4) if a.statistic is not None else None,
                    "p_value": round(float(a.p_value), 4) if a.p_value is not None else None,
                    "detail": a.detail,
                    "severity": a.severity,
                }
                for a in self.assumptions
            ],
            "vif_results": [
                {"predictor": v.predictor, "vif": round(float(v.vif), 3), "severity": v.severity}
                for v in self.vif_results
            ],
            "influential_points": [
                {
                    "index": int(p.index),
                    "cooks_distance": round(float(p.cooks_distance), 4),
                    "leverage": round(float(p.leverage), 4),
                    "standardized_residual": round(float(p.standardized_residual), 4),
                }
                for p in self.influential_points
            ],
            "narrative": self.narrative,
        }
