# General Analysis Dispatch

You are helping route a clinical research analysis request.

## Instructions
Analyze the user's request and route to the appropriate tool: $ARGUMENTS

### Analysis Routing
- **Descriptive statistics / Table 1** -> /table1
- **Compare groups / test differences** -> /hypothesis
- **Predict outcome / model relationship** -> /regression
- **Machine learning / prediction model** -> /predict
- **Survival / time-to-event** -> Use execute_analysis with "survival" type
- **Subgroup analysis** -> /subgroup
- **Sensitivity analysis** -> /sensitivity

### If Unsure
Ask the researcher to clarify:
1. What is the research question?
2. What is the outcome variable?
3. What type of analysis is needed?
