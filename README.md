# regression-diagnostics

OLS regression diagnostics for researchers and analysts — assumption checks, multicollinearity, and influential observations, all in one CLI call.

```
$ regression-diagnostics check data.csv \
    --outcome nps_score --predictors "tenure,feature_usage,support_contacts"

  regression-diagnostics · n=847, p=3, R²=0.4312, Adj-R²=0.4291

  Assumption         Result     Detail
  normality          OK         Shapiro-Wilk: stat=0.9923, p=0.1840
  homoscedasticity   WARNING    Breusch-Pagan: LM=6.2140, p=0.0127
  autocorrelation    OK         Durbin-Watson: 1.9823
  linearity          OK         Residual-fitted correlation: r=0.0031

  VIF
    tenure:          1.23
    feature_usage:   1.87
    support_contacts: 1.64
```

## The problem

Most researchers fit a regression and report the coefficients without ever checking whether the model's assumptions hold. Heteroscedasticity biases your standard errors. High VIF makes your coefficients unstable. A handful of influential observations can drive the entire result. `regression-diagnostics` catches all of this automatically — no manual plotting, no hunting through statsmodels output.

## What it checks

**Normality of residuals** — Shapiro-Wilk test (n ≤ 5,000) or Kolmogorov-Smirnov (n > 5,000). Violation means your standard errors and p-values may be unreliable.

**Homoscedasticity** — Breusch-Pagan Lagrange Multiplier test. Violation (heteroscedasticity) means your standard errors are biased — common with survey and behavioural data where variance grows with the outcome.

**Autocorrelation** — Durbin-Watson statistic. Values below 1.5 or above 2.5 suggest correlated residuals, which invalidates standard inference. Most relevant for longitudinal or time-ordered data.

**Linearity** — Pearson correlation between residuals and fitted values. Should be near zero. Non-zero correlation suggests a non-linear relationship that the model is missing.

**Multicollinearity (VIF)** — Variance Inflation Factor per predictor. VIF ≥ 5 is a warning, VIF ≥ 10 is a violation. High VIF means coefficients are unstable and standard errors are inflated.

**Influential observations (Cook's distance)** — Flags observations that have disproportionate influence on the fitted model. Default threshold is 4/n. Leverage and standardised residuals are also reported.

## Install

```bash
pip install regression-diagnostics
```

With optional LLM narrative layer:

```bash
pip install "regression-diagnostics[openai]"
pip install "regression-diagnostics[anthropic]"
pip install "regression-diagnostics[gemini]"
pip install "regression-diagnostics[all]"
```

## Usage

### CLI

```bash
# Basic diagnostics
regression-diagnostics check survey.csv \
  --outcome nps_score \
  --predictors "tenure,feature_usage,support_contacts"

# Custom Cook's threshold and JSON output
regression-diagnostics check data.csv \
  --outcome satisfaction \
  --predictors "age,usage_days,plan_tier" \
  --cooks-threshold 0.02 \
  --output diagnostics.json

# With LLM narrative interpretation
regression-diagnostics check data.csv \
  --outcome conversion \
  --predictors "sessions,pages_viewed,time_on_site" \
  --llm-backend gemini
```

### Python API

```python
import pandas as pd
from regression_diagnostics import run_diagnostics

df = pd.read_csv("survey.csv")

result = run_diagnostics(
    df,
    outcome="nps_score",
    predictors=["tenure", "feature_usage", "support_contacts"],
)

print(f"R² = {result.r_squared:.4f}")
print(f"Overall passed: {result.overall_passed}")

for assumption in result.assumptions:
    print(f"  {assumption.name}: {assumption.severity} — {assumption.detail}")

for vif in result.vif_results:
    print(f"  VIF {vif.predictor}: {vif.vif:.2f} ({vif.severity})")

print(f"Influential points: {len(result.influential_points)}")

import json
print(json.dumps(result.to_dict(), indent=2))
```

## Severity levels

| Level | Meaning |
|-------|---------|
| `ok` | Assumption holds — no action needed |
| `warning` | Mild violation — worth investigating |
| `violation` | Serious violation — results may be unreliable |

`overall_passed` is `False` if any assumption or VIF reaches `violation` severity.

## Output (JSON)

```json
{
  "n_obs": 847,
  "n_predictors": 3,
  "r_squared": 0.4312,
  "adj_r_squared": 0.4291,
  "overall_passed": false,
  "assumptions": [
    {
      "name": "homoscedasticity",
      "passed": false,
      "statistic": 6.214,
      "p_value": 0.0127,
      "detail": "Breusch-Pagan: LM=6.2140, p=0.0127. Evidence of heteroscedasticity.",
      "severity": "warning"
    }
  ],
  "vif_results": [...],
  "influential_points": [...]
}
```

## Part of the CPR Ecosystem

`regression-diagnostics` is Tool 4.7 in the [CPR Orbital](https://github.com/kgkazakos) series — 40 open-source tools for AI-native product research infrastructure, shipping every Friday from March to December 2026.

**Related tools in this series:**

| Tool | What it does |
|------|-------------|
| [`bayes-ab`](https://github.com/kgkazakos/bayes-ab) | Bayesian A/B testing with posteriors and ROPE decision rules |
| [`scale-validator`](https://github.com/kgkazakos/scale-validator) | Cronbach's alpha, item analysis, factor retention |
| [`power-analyst`](https://github.com/kgkazakos/power-analyst) | Sample size and power curves for common test types |
| [`survey-cleaner`](https://github.com/kgkazakos/survey-cleaner) | Remove speeders, straight-liners and gibberish open-ends |

Part of the broader **Computational Product Research (CPR)** discipline. Core projects — CausalTrack, Synthetic User Council, Co-Reasoning Canvas — are developed separately and target CHI 2027/2028.

## License

MIT
