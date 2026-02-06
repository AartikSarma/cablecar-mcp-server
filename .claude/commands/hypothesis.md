# Hypothesis Testing

You are helping a clinical researcher test statistical hypotheses.

## Instructions
1. Review the hypothesis: $ARGUMENTS
2. Use the MCP data server's execute_analysis tool with type "hypothesis"
3. Auto-select the appropriate test based on data types and groups
4. Report results with:
   - Test statistic and p-value
   - Effect size with confidence interval
   - Multiple testing correction if applicable
5. Use the statistical-reviewer subagent to validate the choice

## Test Selection Guide
| Variable Type | Groups | Recommended Test |
|--------------|--------|-----------------|
| Continuous, normal | 2 | Independent t-test |
| Continuous, non-normal | 2 | Mann-Whitney U |
| Continuous | 3+ | Kruskal-Wallis |
| Categorical | 2+ | Chi-square / Fisher's exact |
