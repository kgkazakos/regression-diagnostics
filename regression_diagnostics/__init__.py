"""regression-diagnostics — OLS assumption checks, VIF, Cook's distance."""

from regression_diagnostics.engine import run_diagnostics
from regression_diagnostics.models import DiagnosticsResult

__version__ = "0.1.0"
__all__ = ["run_diagnostics", "DiagnosticsResult"]
