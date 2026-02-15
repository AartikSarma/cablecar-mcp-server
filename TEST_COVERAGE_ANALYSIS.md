# Test Coverage Analysis

**Date**: 2026-02-15
**Overall Coverage**: 65% (statement), with branch coverage gaps
**Tests**: 281 passing (214 unit + 31 integration + 36 evaluation)

---

## Coverage Summary by Module

| Module | Stmts | Miss | Branch | BrPart | Cover | Priority |
|--------|-------|------|--------|--------|-------|----------|
| `server/tools.py` | 250 | 250 | 100 | 0 | **0%** | CRITICAL |
| `server/main.py` | 46 | 46 | 10 | 0 | **0%** | CRITICAL |
| `analysis/prediction.py` | 141 | 127 | 40 | 0 | **8%** | HIGH |
| `analysis/subgroup.py` | 93 | 85 | 32 | 0 | **6%** | HIGH |
| `analysis/survival.py` | 116 | 106 | 32 | 0 | **7%** | HIGH |
| `analysis/sensitivity.py` | 131 | 114 | 56 | 0 | **9%** | HIGH |
| `analysis/hypothesis.py` | 173 | 56 | 60 | 14 | **64%** | MEDIUM |
| `analysis/regression.py` | 158 | 52 | 44 | 8 | **67%** | MEDIUM |
| `workflow/plan.py` | 37 | 15 | 12 | 0 | **45%** | MEDIUM |
| `evaluation/claude_agent.py` | 142 | 82 | 54 | 6 | **39%** | LOW |
| `data/store.py` | 58 | 7 | 18 | 4 | **86%** | LOW |
| `privacy/guard.py` | 121 | 13 | 66 | 11 | **87%** | LOW |

Modules at or above 90% coverage: `base.py`, `causal.py`, `descriptive.py`, `cohort.py`, `loader.py`, `phi_detector.py`, `policy.py`, `codegen/engine.py`, `provenance.py`, `benchmarks.py`, `dgp.py`, `scoring.py`, `agent.py`, `strobe.py`, `tripod.py`, `pipeline.py`, `state.py`, `cache.py`, and all `__init__.py` files.

---

## Critical Gaps and Recommendations

### 1. `server/tools.py` and `server/main.py` — 0% coverage (CRITICAL)

**What's untested**: The entire MCP server layer — all 4 MCP tools (`get_schema`, `load_data`, `query_cohort`, `execute_analysis`) and the 5 private analysis dispatchers (`_summary_stats`, `_descriptive_analysis`, `_hypothesis_test`, `_regression_analysis`, `_subgroup_analysis`). This is the primary interface between the AI and patient data, and the layer where all privacy sanitization happens.

**Recommended tests**:

```
tests/unit/test_server_tools.py
```

- **`DataServerTools.get_schema()`** — schema returned when no data loaded; schema returned after data load; result is sanitized
- **`DataServerTools.load_data()`** — load CSV directory; load with schema validation; load nonexistent path returns error; PHI columns set on privacy guard after load
- **`DataServerTools.query_cohort()`** — build cohort with inclusion criteria; build with exclusion criteria; query before loading data returns error; cohort is stored for later reuse; result is sanitized
- **`DataServerTools.execute_analysis()`** — dispatch to each of the 5 analysis types; unknown analysis type returns error; missing cohort returns error
- **`DataServerTools._summary_stats()`** — numeric column stats; object column top_values; column filtering; missing table returns error
- **`DataServerTools._descriptive_analysis()`** — stratified vs unstratified; variable filtering; missing table error
- **`DataServerTools._hypothesis_test()`** — each test type (mann_whitney, t_test, kruskal, chi_square); unsupported test error; missing variable error; fewer than 2 groups error
- **`DataServerTools._regression_analysis()`** — logistic model; linear model; unsupported model type error; model fitting failure
- **`DataServerTools._subgroup_analysis()`** — subgroup split; descriptive subgroup; regression subgroup; missing variable error

```
tests/unit/test_server_main.py
```

- **`create_server()`** — server creation returns Server and DataServerTools; tool listing returns 4 tools with correct names; each tool dispatches correctly via `call_tool`
- **Error handling** — unknown tool name returns error; exceptions wrapped in JSON error response

**Impact**: This is the single largest coverage gap and represents the actual interface users interact with. Privacy sanitization at this boundary is safety-critical and should be tested end-to-end.

---

### 2. `analysis/prediction.py` — 8% coverage (HIGH)

**What's untested**: The entire `PredictionModel.run()` method including cross-validation, the estimator factory (`_get_estimator`), feature importance extraction (`_get_feature_importance`), and calibration computation (`_compute_calibration`).

**Currently tested**: Only the import path and class instantiation (via conftest fixtures).

**Recommended tests**:

```
tests/unit/test_prediction.py
```

- **`PredictionModel.run()`** — logistic model with cross-validation; random_forest model; gradient_boosting model; unsupported model type falls back to random_forest with warning; binary outcome validation; missing columns warning; missing table warning; no complete cases warning
- **Cross-validation metrics** — fold_metrics contain expected keys (auroc, auprc, accuracy, sensitivity, specificity, ppv, npv); aggregate_metrics have mean and std
- **`_get_estimator()`** — returns LogisticRegression for "logistic"; RandomForestClassifier for "random_forest"; GradientBoostingClassifier for "gradient_boosting"; fallback for unknown type
- **`_get_feature_importance()`** — logistic model returns absolute coefficients; tree models return feature_importances_; results sorted descending
- **`_compute_calibration()`** — returns deciles with observed/predicted rates; Hosmer-Lemeshow statistic and p-value; too few observations returns error

---

### 3. `analysis/survival.py` — 7% coverage (HIGH)

**What's untested**: `SurvivalAnalysis.run()` including Kaplan-Meier fitting, survival probability extraction at timepoints, confidence intervals, log-rank tests (both 2-group and pairwise 3+ group), and the `_extract_km_results()` helper.

**Recommended tests**:

```
tests/unit/test_survival.py
```

- **`SurvivalAnalysis.run()`** — ungrouped Kaplan-Meier; grouped (2-group) KM with log-rank test; grouped (3+ group) with pairwise log-rank; custom timepoints; timepoints exceeding max observed time; missing columns warning; missing table warning; no complete cases warning; lifelines not installed warning
- **`_extract_km_results()`** — median survival time extraction (finite and "not reached"); survival probabilities at timepoints; confidence intervals at timepoints
- **Log-rank testing** — 2-group comparison returns statistic and p_value; 3+ group produces pairwise results list; log-rank failure handled gracefully

The test fixture needs a time-to-event column and a binary event indicator added to `mini_dataframes` (or a dedicated fixture).

---

### 4. `analysis/subgroup.py` — 6% coverage (HIGH)

**What's untested**: The entire `SubgroupAnalysis` class — `run()`, `_resolve_analysis_fn()`, and `_interaction_test()`.

**Recommended tests**:

```
tests/unit/test_subgroup.py
```

- **`SubgroupAnalysis.run()`** — with a callable `analysis_fn`; with `analysis_type="regression"`; unknown analysis_type warns and returns empty; missing `analysis_fn` and `analysis_type` warns; subgroup_variable not in table; fewer than 2 unique values; per-subgroup results collected; analysis failure in one subgroup handled gracefully
- **`_resolve_analysis_fn()`** — returns callable when `analysis_fn` provided; creates regression wrapper for `analysis_type="regression"`; returns None for unknown type; returns None when neither provided
- **`_interaction_test()`** — regression interaction with binary outcome (likelihood ratio test); regression interaction with continuous outcome (F-test); non-regression analysis returns descriptive note; missing outcome/predictors returns note; interaction term construction for numeric predictors; no numeric predictors note

---

### 5. `analysis/sensitivity.py` — 9% coverage (HIGH)

**What's untested**: The entire `SensitivityAnalysis` class — `run()`, `_build_modified_cohort()`, all 4 modification strategies (`_complete_case`, `_missing_indicator`, `_exclude_outliers`, `_alternate_definition`), and `_compare_results()`.

**Recommended tests**:

```
tests/unit/test_sensitivity.py
```

- **`SensitivityAnalysis.run()`** — complete_case sensitivity; missing_indicator sensitivity; exclude_outliers sensitivity; alternate_definition with transform_fn; unknown type warns; base analysis failure handled; modified cohort build failure handled; sensitivity analysis failure handled
- **`_build_modified_cohort()`** — returns Cohort with modified table; table not found raises KeyError; alternate_definition without transform_fn raises ValueError
- **`_complete_case()`** — drops rows with missing values in specified columns; handles columns not present in dataframe
- **`_missing_indicator()`** — adds `_missing` binary columns; imputes numeric with median; skips columns with no missing data
- **`_exclude_outliers()`** — removes rows outside mean +/- 3 SD; skips non-numeric columns; handles zero-std columns; handles all-missing columns
- **`_compare_results()`** — compares coefficients and aggregate metrics; handles errors in one or both results; reports sample size differences; returns note when no automated comparison possible

---

### 6. `analysis/hypothesis.py` — 64% coverage (MEDIUM)

**What's covered**: auto-detect mann_whitney, explicit t_test, chi_square, batch correction, missing variable/group warnings.

**What's untested** (lines 71-75, 79-81, 182-198, 230-248, 263-264, 310-333, 345-356, 373, 379):
- Unknown test name fallback to "auto" (lines 71-75)
- Table not found in cohort (lines 79-81)
- Auto-selection of Fisher's exact test (when expected cells < 5 in a 2x2 table) (lines 182-198)
- Kruskal-Wallis test execution (lines 230-240)
- ANOVA test execution (lines 247-248, 345-356)
- Fisher's exact test execution (lines 310-319) and non-2x2 fallback (lines 312-316)
- Effect size computation: Cohen's d edge cases (line 373, 379)
- Bonferroni/FDR correction with n_tests > 1 within `run()` (lines 133-148)

**Recommended additional tests**:

- Kruskal-Wallis with 3+ groups
- ANOVA with 3+ groups
- Fisher's exact test with 2x2 table
- Fisher's exact fallback to chi-square for non-2x2 tables
- Auto-detect Fisher's exact for small expected cells
- Multiple-testing correction within run() with `n_tests` parameter
- Cohen's d with zero pooled std
- Cohen's d with n < 2

---

### 7. `analysis/regression.py` — 67% coverage (MEDIUM)

**What's covered**: logistic, linear, categorical predictors, confounders, cox warning, unsupported model, odds ratios, result serialization.

**What's untested** (lines 84, 107-108, 131-134, 167-178, 205-211, 221-223, 231, 238-240, 255, 267-308, 361-363, 376, 380-384):
- Cox proportional hazards model execution (lines 267-308) — the warning is tested but not the actual Cox model run with a time column
- VIF (Variance Inflation Factor) computation (lines 205-211)
- Model fit diagnostics for linear models (lines 221-223, 231)
- Multicollinearity detection (lines 238-240)
- Predictions and residuals extraction (lines 361-363, 376, 380-384)

**Recommended additional tests**:

- Cox model execution with proper time_col and event_col
- VIF computation
- Linear regression with model diagnostics
- Handling of perfect separation in logistic regression

---

### 8. `workflow/plan.py` — 45% coverage (MEDIUM)

**What's untested**: `StudyPlan.summary()` method (lines 55-68) — the string formatting of study plan details.

**Recommended tests**:

- `to_dict()` with fully populated plan
- `summary()` with all fields populated
- `summary()` with minimal fields (empty strings)

---

### 9. `evaluation/claude_agent.py` — 39% coverage (LOW priority)

**What's untested**: The actual Anthropic API interaction loop (`run()` method lines 214-286), tool dispatching in the agent context, result parsing, and fallback handling. This is lower priority because it requires mocking the Anthropic API, and the module is used for benchmarking rather than production data processing.

**Recommended tests** (if prioritized):

- Mock-based test of the agent loop with simulated tool calls
- `_parse_submit()` with various input shapes
- `_fallback_result()` when submit_result is never called
- `_dispatch_tool()` for each tool type
- `_tool_query_data()` with group_by, column filtering
- `_tool_run_regression()` with logistic and linear models, merge behavior

---

## Cross-Cutting Gaps

### A. Privacy boundary testing for server tools

The integration test `test_privacy_boundary.py` tests privacy at the `PrivacyGuard` level, but there are no tests verifying that `DataServerTools` methods actually call `sanitize_for_llm()` on every output path, including error paths. A determined attacker could potentially extract data through carefully crafted error messages.

**Recommendation**: Add integration tests that verify every `DataServerTools` method output contains no raw patient identifiers, even in error cases.

### B. Edge cases in data types

Several analysis modules handle numeric, categorical, and object dtypes, but there are no tests for:
- Datetime columns passed to analysis functions
- Mixed-type columns
- Integer columns that should be treated as categorical (e.g., binary 0/1)
- Very large or very small float values (potential overflow in log/exp)

### C. Concurrency and state management

`DataServerTools` maintains mutable state (`_cohorts` dict, `store`). There are no tests for:
- Multiple cohort creation and independent querying
- Overwriting a cohort with the same name
- Analysis on a cohort after the underlying data store is modified

### D. Error message information leakage

Several error handlers include `traceback.format_exc()` in their return values (`load_data`, `query_cohort`, `execute_analysis` in `tools.py`). Tracebacks may contain file paths, data values, or other sensitive context. There should be tests ensuring tracebacks don't leak PHI.

---

## Prioritized Implementation Order

1. **`test_server_tools.py`** — 0% coverage on the primary user-facing interface and privacy boundary (~30 tests)
2. **`test_prediction.py`** — 8% coverage on ML prediction pipeline (~15 tests)
3. **`test_survival.py`** — 7% coverage on survival analysis (~12 tests)
4. **`test_subgroup.py`** — 6% coverage on subgroup analysis engine (~12 tests)
5. **`test_sensitivity.py`** — 9% coverage on robustness analysis (~15 tests)
6. **Expand `test_analysis.py`** — add missing hypothesis test types and regression edge cases (~10 tests)
7. **`test_plan.py`** — expand workflow plan tests (~3 tests)

Implementing items 1-5 would bring overall coverage from 65% to approximately 85-90%.
