# Prediction Model Building

You are helping a clinical researcher build a prediction model with TRIPOD+AI compliance.

## Instructions
1. Review the prediction task: $ARGUMENTS
2. Use the MCP data server's execute_analysis tool with type "prediction"
3. Follow the TRIPOD+AI framework:
   - Clearly define the prediction question (who, what, when)
   - Specify the target population and timepoint
   - Define predictors and outcome
4. Model development steps:
   - Data splitting (train/validation/test)
   - Feature selection
   - Model training (logistic regression, random forest, gradient boosting)
   - Hyperparameter tuning
   - Internal validation (cross-validation, bootstrapping)
5. Report:
   - Discrimination (AUROC, AUPRC)
   - Calibration (calibration plot, Hosmer-Lemeshow)
   - Clinical utility (decision curve analysis)
   - Feature importance

## Model Selection Guide
| Task | Recommended Starting Model |
|------|---------------------------|
| Binary classification | Logistic regression (baseline), then XGBoost |
| Multiclass | Random forest, then XGBoost |
| Risk score | LASSO logistic regression |

## TRIPOD Compliance
Ensure all TRIPOD+AI checklist items are addressed in reporting.
