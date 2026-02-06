# Table 1 - Baseline Characteristics

You are generating a publication-ready Table 1 (baseline characteristics).

## Instructions

1. Check that data is loaded and a cohort is defined: $ARGUMENTS
2. Use the MCP data server's `execute_analysis` tool with type "descriptive"
3. Common Table 1 variables:
   - Demographics: age, sex, race/ethnicity
   - Clinical: admission type, severity scores
   - Comorbidities: if available
   - Outcomes: mortality, LOS
4. Stratify by the main comparison variable (e.g., exposure group)
5. Format as a publication-ready table:
   - Continuous variables: median (IQR) or mean +/- SD
   - Categorical variables: n (%)
   - Include p-values for group comparisons
   - Include standardized mean differences (SMD)
   - Flag SMD > 0.1 as potentially imbalanced

## Privacy Reminder
- All cell counts < 10 will be suppressed
- Report only aggregate statistics
