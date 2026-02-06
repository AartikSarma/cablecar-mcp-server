# Regression Analysis

You are helping a clinical researcher fit regression models.

## Instructions
1. Review the model specification: $ARGUMENTS
2. Use the MCP data server's execute_analysis tool with type "regression"
3. Include confounders from the causal DAG (if /dag was used)
4. Report:
   - Model coefficients with CIs and p-values
   - ORs (logistic) or HRs (Cox) with CIs
   - Model fit statistics (R-squared, AIC, BIC, concordance)
   - Diagnostic checks (multicollinearity, residuals)
5. Use the statistical-reviewer subagent to validate assumptions

## Model Selection Guide
| Outcome Type | Recommended Model |
|-------------|------------------|
| Binary (yes/no) | Logistic regression |
| Continuous | Linear regression |
| Time-to-event | Cox proportional hazards |
| Count | Poisson regression |
