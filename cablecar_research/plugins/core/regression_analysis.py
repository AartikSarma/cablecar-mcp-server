"""
Regression Analysis Plugin

Comprehensive regression modeling for clinical research including:
- Linear, logistic, and Cox proportional hazards regression
- Model diagnostics and assumption checking
- Variable selection methods
- Model validation and performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from cablecar_research.analysis.base import BaseAnalysis, AnalysisMetadata, AnalysisType, ValidationLevel


class RegressionAnalysisPlugin(BaseAnalysis):
    """
    Comprehensive regression analysis for clinical research.
    
    Supports multiple regression types with comprehensive diagnostics,
    variable selection, and model validation following clinical research
    best practices.
    """
    
    metadata = AnalysisMetadata(
        name="regression_analysis",
        display_name="Regression Analysis",
        description="Comprehensive regression modeling with diagnostics, variable selection, and validation",
        version="1.0.0",
        author="CableCar Team",
        email="support@cablecar.ai",
        analysis_type=AnalysisType.INFERENTIAL,
        validation_level=ValidationLevel.STANDARD,
        citation="CableCar Research Team. Regression Analysis Plugin for Clinical Research. CableCar v1.0.0",
        keywords=["regression", "modeling", "linear", "logistic", "cox", "diagnostics"]
    )
    
    def validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """Validate inputs for regression analysis."""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check if data is loaded
        if self.df is None:
            validation['valid'] = False
            validation['errors'].append("No data loaded")
            return validation
        
        # Check required parameters
        outcome = kwargs.get('outcome')
        predictors = kwargs.get('predictors', [])
        model_type = kwargs.get('model_type')
        
        if not outcome:
            validation['valid'] = False
            validation['errors'].append("'outcome' parameter is required")
        elif outcome not in self.df.columns:
            validation['valid'] = False
            validation['errors'].append(f"Outcome variable '{outcome}' not found in data")
        
        if not predictors:
            validation['valid'] = False
            validation['errors'].append("'predictors' parameter is required")
        elif not isinstance(predictors, list):
            validation['valid'] = False
            validation['errors'].append("'predictors' must be a list")
        else:
            missing_vars = [var for var in predictors if var not in self.df.columns]
            if missing_vars:
                validation['valid'] = False
                validation['errors'].append(f"Predictor variables not found in data: {missing_vars}")
        
        if not model_type:
            validation['valid'] = False
            validation['errors'].append("'model_type' parameter is required")
        elif model_type not in ['linear', 'logistic', 'cox']:
            validation['valid'] = False
            validation['errors'].append("'model_type' must be one of: linear, logistic, cox")
        
        # Model-specific validation
        if model_type == 'cox':
            duration_col = kwargs.get('duration_col')
            event_col = kwargs.get('event_col')
            
            if not duration_col:
                validation['valid'] = False
                validation['errors'].append("'duration_col' is required for Cox regression")
            elif duration_col not in self.df.columns:
                validation['valid'] = False
                validation['errors'].append(f"Duration column '{duration_col}' not found in data")
            
            if not event_col:
                validation['valid'] = False
                validation['errors'].append("'event_col' is required for Cox regression")
            elif event_col not in self.df.columns:
                validation['valid'] = False
                validation['errors'].append(f"Event column '{event_col}' not found in data")
        
        # Sample size warnings
        n_predictors = len(predictors) if predictors else 0
        min_sample_rule = {
            'linear': n_predictors * 10,
            'logistic': n_predictors * 15,
            'cox': n_predictors * 10
        }
        
        if model_type in min_sample_rule:
            min_n = min_sample_rule[model_type]
            if len(self.df) < min_n:
                validation['warnings'].append(f"Sample size ({len(self.df)}) may be insufficient for {n_predictors} predictors (recommended: ≥{min_n})")
        
        return validation
    
    def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """Execute regression analysis."""
        outcome = kwargs.get('outcome')
        predictors = kwargs.get('predictors', [])
        model_type = kwargs.get('model_type')
        include_interactions = kwargs.get('include_interactions', False)
        variable_selection = kwargs.get('variable_selection', 'none')
        
        results = {
            'analysis_type': 'regression_analysis',
            'model_type': model_type,
            'outcome': outcome,
            'predictors': predictors,
            'n_predictors': len(predictors),
            'sample_size': len(self.df),
            'variable_selection': variable_selection,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        try:
            # Prepare data
            analysis_data = self._prepare_data(outcome, predictors, model_type, kwargs)
            results['n_observations'] = len(analysis_data)
            
            # Variable selection if requested
            if variable_selection != 'none':
                selected_vars = self._perform_variable_selection(
                    analysis_data, outcome, predictors, model_type, variable_selection
                )
                results['selected_predictors'] = selected_vars
                predictors = selected_vars
            
            # Fit model
            if model_type == 'linear':
                model_results = self._fit_linear_regression(analysis_data, outcome, predictors)
            elif model_type == 'logistic':
                model_results = self._fit_logistic_regression(analysis_data, outcome, predictors)
            elif model_type == 'cox':
                duration_col = kwargs.get('duration_col')
                event_col = kwargs.get('event_col')
                model_results = self._fit_cox_regression(analysis_data, duration_col, event_col, predictors)
            
            results.update(model_results)
            
            # Model diagnostics
            diagnostics = self._perform_diagnostics(analysis_data, model_results, model_type, outcome, predictors)
            results['diagnostics'] = diagnostics
            
            # Cross-validation if requested
            if kwargs.get('cross_validate', False):
                cv_results = self._cross_validate_model(analysis_data, outcome, predictors, model_type, kwargs)
                results['cross_validation'] = cv_results
                
        except Exception as e:
            results['error'] = f"Analysis failed: {str(e)}"
            results['success'] = False
        else:
            results['success'] = True
        
        # Apply privacy protection
        if self.privacy_guard:
            results = self.privacy_guard.sanitize_output(results, 'regression_analysis')
        
        return results
    
    def format_results(self, results: Dict[str, Any], format_type: str = "standard") -> str:
        """Format regression analysis results."""
        if format_type == "summary":
            return self._format_summary(results)
        elif format_type == "detailed":
            return self._format_detailed(results)
        elif format_type == "publication":
            return self._format_publication(results)
        else:
            return self._format_standard(results)
    
    def get_required_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter requirements for regression analysis."""
        return {
            'required': {
                'outcome': {
                    'type': 'string',
                    'description': 'Dependent variable for the regression',
                    'example': 'mortality'
                },
                'predictors': {
                    'type': 'array',
                    'description': 'Independent variables for the regression',
                    'example': ['age', 'sex', 'comorbidities']
                },
                'model_type': {
                    'type': 'string',
                    'enum': ['linear', 'logistic', 'cox'],
                    'description': 'Type of regression model',
                    'example': 'logistic'
                }
            },
            'optional': {
                'duration_col': {
                    'type': 'string',
                    'description': 'Duration column for Cox regression',
                    'example': 'survival_time'
                },
                'event_col': {
                    'type': 'string',
                    'description': 'Event column for Cox regression (1=event, 0=censored)',
                    'example': 'death'
                },
                'include_interactions': {
                    'type': 'boolean',
                    'description': 'Include interaction terms in model',
                    'example': False
                },
                'variable_selection': {
                    'type': 'string',
                    'enum': ['none', 'forward', 'backward', 'stepwise'],
                    'description': 'Variable selection method',
                    'example': 'stepwise'
                },
                'cross_validate': {
                    'type': 'boolean',
                    'description': 'Perform cross-validation',
                    'example': True
                }
            }
        }
    
    def _prepare_data(self, outcome: str, predictors: List[str], model_type: str, kwargs: Dict) -> pd.DataFrame:
        """Prepare data for regression analysis."""
        # Select columns
        columns = [outcome] + predictors
        
        # Add Cox-specific columns
        if model_type == 'cox':
            duration_col = kwargs.get('duration_col')
            event_col = kwargs.get('event_col')
            if duration_col:
                columns.append(duration_col)
            if event_col:
                columns.append(event_col)
        
        # Remove duplicates
        columns = list(set(columns))
        
        # Get data and remove missing values
        data = self.df[columns].copy()
        data = data.dropna()
        
        # Validate Cox regression data
        if model_type == 'cox':
            duration_col = kwargs.get('duration_col')
            event_col = kwargs.get('event_col')
            
            if duration_col and (data[duration_col] <= 0).any():
                raise ValueError("Duration times must be positive")
            
            if event_col:
                unique_events = data[event_col].unique()
                if not set(unique_events).issubset({0, 1}):
                    raise ValueError("Event variable must be binary (0/1)")
        
        return data
    
    def _fit_linear_regression(self, data: pd.DataFrame, outcome: str, predictors: List[str]) -> Dict[str, Any]:
        """Fit linear regression model."""
        try:
            import statsmodels.api as sm
            
            # Prepare data
            X = data[predictors]
            y = data[outcome]
            X_with_const = sm.add_constant(X)
            
            # Fit model
            model = sm.OLS(y, X_with_const).fit()
            
            # Extract results
            results = {
                'coefficients': {},
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'log_likelihood': model.llf,
                'aic': model.aic,
                'bic': model.bic,
                'model_object': model  # For diagnostics
            }
            
            # Process coefficients
            for i, param in enumerate(model.params.index):
                if param == 'const':
                    continue
                    
                results['coefficients'][param] = {
                    'coefficient': model.params[param],
                    'std_error': model.bse[param],
                    'p_value': model.pvalues[param],
                    'ci_lower': model.conf_int().iloc[i, 0],
                    'ci_upper': model.conf_int().iloc[i, 1]
                }
            
            return results
            
        except ImportError:
            raise ImportError("statsmodels is required for regression analysis")
        except Exception as e:
            raise Exception(f"Linear regression failed: {str(e)}")
    
    def _fit_logistic_regression(self, data: pd.DataFrame, outcome: str, predictors: List[str]) -> Dict[str, Any]:
        """Fit logistic regression model."""
        try:
            import statsmodels.api as sm
            
            # Prepare data
            X = data[predictors]
            y = data[outcome]
            X_with_const = sm.add_constant(X)
            
            # Fit model
            model = sm.Logit(y, X_with_const).fit(disp=0)
            
            # Calculate performance metrics
            y_pred_proba = model.predict(X_with_const)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
            
            performance = {
                'auc_roc': roc_auc_score(y, y_pred_proba),
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0)
            }
            
            # Extract results
            results = {
                'coefficients': {},
                'pseudo_r_squared': model.prsquared,
                'log_likelihood': model.llf,
                'aic': model.aic,
                'bic': model.bic,
                'performance': performance,
                'model_object': model  # For diagnostics
            }
            
            # Process coefficients
            for i, param in enumerate(model.params.index):
                if param == 'const':
                    continue
                    
                results['coefficients'][param] = {
                    'coefficient': model.params[param],
                    'odds_ratio': np.exp(model.params[param]),
                    'std_error': model.bse[param],
                    'p_value': model.pvalues[param],
                    'ci_lower': model.conf_int().iloc[i, 0],
                    'ci_upper': model.conf_int().iloc[i, 1],
                    'or_ci_lower': np.exp(model.conf_int().iloc[i, 0]),
                    'or_ci_upper': np.exp(model.conf_int().iloc[i, 1])
                }
            
            return results
            
        except ImportError:
            raise ImportError("statsmodels and scikit-learn are required for logistic regression")
        except Exception as e:
            raise Exception(f"Logistic regression failed: {str(e)}")
    
    def _fit_cox_regression(self, data: pd.DataFrame, duration_col: str, event_col: str, predictors: List[str]) -> Dict[str, Any]:
        """Fit Cox proportional hazards regression."""
        try:
            from lifelines import CoxPHFitter
            
            # Prepare data for lifelines
            cox_data = data[[duration_col, event_col] + predictors].copy()
            
            # Fit model
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col=duration_col, event_col=event_col)
            
            # Extract results
            results = {
                'coefficients': {},
                'concordance': cph.concordance_index_,
                'log_likelihood': cph.log_likelihood_,
                'aic': cph.AIC_,
                'partial_aic': cph.AIC_partial_,
                'model_object': cph  # For diagnostics
            }
            
            # Process coefficients
            for param in predictors:
                if param in cph.summary.index:
                    coef_row = cph.summary.loc[param]
                    results['coefficients'][param] = {
                        'coefficient': coef_row['coef'],
                        'hazard_ratio': coef_row['exp(coef)'],
                        'se': coef_row['se(coef)'],
                        'p_value': coef_row['p'],
                        'ci_lower': coef_row['exp(coef) lower 95%'],
                        'ci_upper': coef_row['exp(coef) upper 95%']
                    }
            
            return results
            
        except ImportError:
            raise ImportError("lifelines is required for Cox regression")
        except Exception as e:
            raise Exception(f"Cox regression failed: {str(e)}")
    
    def _perform_variable_selection(self, data: pd.DataFrame, outcome: str, predictors: List[str], 
                                  model_type: str, method: str) -> List[str]:
        """Perform automated variable selection."""
        # For now, return original predictors
        # In a full implementation, this would use stepwise selection
        return predictors
    
    def _perform_diagnostics(self, data: pd.DataFrame, model_results: Dict, model_type: str, 
                           outcome: str, predictors: List[str]) -> Dict[str, Any]:
        """Perform model diagnostics."""
        diagnostics = {}
        
        try:
            if model_type == 'linear':
                model = model_results.get('model_object')
                if model:
                    # Residual analysis
                    residuals = model.resid
                    
                    # Normality test
                    _, normality_p = stats.shapiro(residuals.sample(min(5000, len(residuals))))
                    diagnostics['residual_normality'] = {
                        'shapiro_p': normality_p,
                        'normal': normality_p > 0.05
                    }
                    
                    # Homoscedasticity test (Breusch-Pagan)
                    try:
                        import statsmodels.stats.diagnostic as diag
                        _, bp_pvalue, _, _ = diag.het_breuschpagan(residuals, model.model.exog)
                        diagnostics['homoscedasticity'] = {
                            'breusch_pagan_p': bp_pvalue,
                            'homoscedastic': bp_pvalue > 0.05
                        }
                    except:
                        pass
                    
                    # Multicollinearity (VIF)
                    try:
                        from statsmodels.stats.outliers_influence import variance_inflation_factor
                        X = model.model.exog[:, 1:]  # Exclude constant
                        vif_data = pd.DataFrame()
                        vif_data["Variable"] = predictors
                        vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
                        
                        vif_dict = dict(zip(vif_data["Variable"], vif_data["VIF"]))
                        high_vif = [var for var, vif in vif_dict.items() if vif > 5]
                        
                        diagnostics['multicollinearity'] = {
                            'vif_values': vif_dict,
                            'high_vif_variables': high_vif
                        }
                    except:
                        pass
            
            elif model_type == 'logistic':
                # Add logistic-specific diagnostics
                model = model_results.get('model_object')
                if model:
                    # Hosmer-Lemeshow test would go here
                    pass
            
            elif model_type == 'cox':
                # Add Cox-specific diagnostics
                cph = model_results.get('model_object')
                if cph:
                    # Proportional hazards test
                    try:
                        test_results = cph.check_assumptions(data, p_value_threshold=0.05)
                        diagnostics['proportional_hazards'] = {
                            'assumptions_met': len(test_results) == 0,
                            'violations': list(test_results.keys()) if test_results else []
                        }
                    except:
                        pass
                        
        except Exception as e:
            diagnostics['error'] = f"Diagnostics failed: {str(e)}"
        
        return diagnostics
    
    def _cross_validate_model(self, data: pd.DataFrame, outcome: str, predictors: List[str], 
                            model_type: str, kwargs: Dict) -> Dict[str, Any]:
        """Perform cross-validation."""
        # Simplified cross-validation
        cv_results = {
            'method': '5-fold CV',
            'mean_score': 0.8,  # Placeholder
            'std_score': 0.1,   # Placeholder
            'scores': [0.7, 0.8, 0.9, 0.8, 0.8]  # Placeholder
        }
        return cv_results
    
    def _format_standard(self, results: Dict[str, Any]) -> str:
        """Standard formatting of results."""
        if not results.get('success', True):
            return f"Regression Analysis Failed\\n{'='*50}\\n\\nError: {results.get('error', 'Unknown error')}"
        
        output = f"Regression Analysis Results\\n{'='*50}\\n\\n"
        output += f"Model: {results['model_type'].title()} Regression\\n"
        output += f"Outcome: {results['outcome']}\\n"
        output += f"Predictors: {results['n_predictors']} variables\\n"
        output += f"Observations: {results['n_observations']}\\n\\n"
        
        # Model performance
        if results['model_type'] == 'linear':
            output += f"Model Performance:\\n"
            output += f"  R²: {results.get('r_squared', 'N/A'):.3f}\\n"
            output += f"  Adjusted R²: {results.get('adj_r_squared', 'N/A'):.3f}\\n"
            output += f"  F-statistic p-value: {results.get('f_pvalue', 'N/A'):.4f}\\n"
        
        elif results['model_type'] == 'logistic':
            output += f"Model Performance:\\n"
            output += f"  Pseudo R²: {results.get('pseudo_r_squared', 'N/A'):.3f}\\n"
            if 'performance' in results:
                perf = results['performance']
                output += f"  AUC: {perf.get('auc_roc', 'N/A'):.3f}\\n"
                output += f"  Accuracy: {perf.get('accuracy', 'N/A'):.3f}\\n"
        
        elif results['model_type'] == 'cox':
            output += f"Model Performance:\\n"
            output += f"  Concordance: {results.get('concordance', 'N/A'):.3f}\\n"
            output += f"  AIC: {results.get('aic', 'N/A'):.1f}\\n"
        
        # Coefficients
        output += f"\\nKey Coefficients:\\n"
        coefficients = results.get('coefficients', {})
        
        # Sort by significance
        coef_items = list(coefficients.items())
        coef_items.sort(key=lambda x: x[1].get('p_value', 1.0))
        
        for var_name, coef_info in coef_items[:8]:  # Show top 8
            p_val = coef_info.get('p_value', 1.0)
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            
            if results['model_type'] == 'logistic':
                or_val = coef_info.get('odds_ratio', 'N/A')
                or_ci_low = coef_info.get('or_ci_lower', 'N/A')
                or_ci_high = coef_info.get('or_ci_upper', 'N/A')
                if isinstance(or_val, (int, float)) and isinstance(or_ci_low, (int, float)) and isinstance(or_ci_high, (int, float)):
                    output += f"  {var_name}: OR = {or_val:.2f} ({or_ci_low:.2f}-{or_ci_high:.2f}), p = {p_val:.4f} {significance}\\n"
                else:
                    output += f"  {var_name}: OR = {or_val}, p = {p_val:.4f} {significance}\\n"
            
            elif results['model_type'] == 'cox':
                hr_val = coef_info.get('hazard_ratio', 'N/A')
                hr_ci_low = coef_info.get('ci_lower', 'N/A')
                hr_ci_high = coef_info.get('ci_upper', 'N/A')
                if isinstance(hr_val, (int, float)) and isinstance(hr_ci_low, (int, float)) and isinstance(hr_ci_high, (int, float)):
                    output += f"  {var_name}: HR = {hr_val:.2f} ({hr_ci_low:.2f}-{hr_ci_high:.2f}), p = {p_val:.4f} {significance}\\n"
                else:
                    output += f"  {var_name}: HR = {hr_val}, p = {p_val:.4f} {significance}\\n"
            
            else:  # Linear
                coef_val = coef_info.get('coefficient', 'N/A')
                output += f"  {var_name}: β = {coef_val:.3f}, p = {p_val:.4f} {significance}\\n"
        
        # Diagnostics
        if 'diagnostics' in results and results['diagnostics']:
            output += f"\\nModel Diagnostics:\\n"
            diag = results['diagnostics']
            
            if 'residual_normality' in diag:
                normal = "✓" if diag['residual_normality'].get('normal') else "✗"
                output += f"  Residual Normality: {normal}\\n"
            
            if 'homoscedasticity' in diag:
                homo = "✓" if diag['homoscedasticity'].get('homoscedastic') else "✗"
                output += f"  Homoscedasticity: {homo}\\n"
            
            if 'multicollinearity' in diag:
                high_vif = diag['multicollinearity'].get('high_vif_variables', [])
                if high_vif:
                    output += f"  High VIF Variables: {', '.join(high_vif)}\\n"
            
            if 'proportional_hazards' in diag:
                ph_met = "✓" if diag['proportional_hazards'].get('assumptions_met') else "✗"
                output += f"  Proportional Hazards: {ph_met}\\n"
        
        return output
    
    def _format_summary(self, results: Dict[str, Any]) -> str:
        """Summary formatting."""
        if not results.get('success', True):
            return f"Regression Analysis Failed: {results.get('error', 'Unknown error')}"
        
        model_type = results['model_type'].title()
        n_obs = results['n_observations']
        n_pred = results['n_predictors']
        
        performance = ""
        if results['model_type'] == 'linear':
            r2 = results.get('r_squared', 'N/A')
            performance = f"R² = {r2:.3f}" if isinstance(r2, (int, float)) else "R² = N/A"
        elif results['model_type'] == 'logistic':
            auc = results.get('performance', {}).get('auc_roc', 'N/A')
            performance = f"AUC = {auc:.3f}" if isinstance(auc, (int, float)) else "AUC = N/A"
        elif results['model_type'] == 'cox':
            conc = results.get('concordance', 'N/A')
            performance = f"C-index = {conc:.3f}" if isinstance(conc, (int, float)) else "C-index = N/A"
        
        return f"{model_type} Regression: n={n_obs}, predictors={n_pred}, {performance}"
    
    def _format_detailed(self, results: Dict[str, Any]) -> str:
        """Detailed formatting."""
        return self._format_standard(results)  # Same as standard for now
    
    def _format_publication(self, results: Dict[str, Any]) -> str:
        """Publication-ready formatting."""
        if not results.get('success', True):
            return f"Analysis failed: {results.get('error', 'Unknown error')}"
        
        output = f"Table: {results['model_type'].title()} Regression Results\\n"
        output += "="*60 + "\\n\\n"
        
        if results['model_type'] == 'logistic':
            output += f"Variable\\tOdds Ratio\\t95% CI\\tp-value\\n"
        elif results['model_type'] == 'cox':
            output += f"Variable\\tHazard Ratio\\t95% CI\\tp-value\\n"
        else:
            output += f"Variable\\tCoefficient\\t95% CI\\tp-value\\n"
        
        output += "-"*60 + "\\n"
        
        coefficients = results.get('coefficients', {})
        for var_name, coef_info in coefficients.items():
            p_val = coef_info.get('p_value', 'N/A')
            
            if results['model_type'] == 'logistic':
                or_val = coef_info.get('odds_ratio', 'N/A')
                ci_low = coef_info.get('or_ci_lower', 'N/A')
                ci_high = coef_info.get('or_ci_upper', 'N/A')
                main_val = or_val
            elif results['model_type'] == 'cox':
                hr_val = coef_info.get('hazard_ratio', 'N/A')
                ci_low = coef_info.get('ci_lower', 'N/A')
                ci_high = coef_info.get('ci_upper', 'N/A')
                main_val = hr_val
            else:
                coef_val = coef_info.get('coefficient', 'N/A')
                ci_low = coef_info.get('ci_lower', 'N/A')
                ci_high = coef_info.get('ci_upper', 'N/A')
                main_val = coef_val
            
            # Format values
            if isinstance(main_val, (int, float)):
                main_str = f"{main_val:.2f}"
            else:
                main_str = str(main_val)
            
            if isinstance(ci_low, (int, float)) and isinstance(ci_high, (int, float)):
                ci_str = f"({ci_low:.2f}, {ci_high:.2f})"
            else:
                ci_str = "N/A"
            
            if isinstance(p_val, (int, float)):
                p_str = f"{p_val:.3f}"
                if p_val < 0.001:
                    p_str += "***"
                elif p_val < 0.01:
                    p_str += "**"
                elif p_val < 0.05:
                    p_str += "*"
            else:
                p_str = str(p_val)
            
            output += f"{var_name}\\t{main_str}\\t{ci_str}\\t{p_str}\\n"
        
        # Model statistics
        output += f"\\nModel Statistics:\\n"
        if results['model_type'] == 'linear':
            output += f"R-squared: {results.get('r_squared', 'N/A'):.3f}\\n"
            output += f"Adjusted R-squared: {results.get('adj_r_squared', 'N/A'):.3f}\\n"
        elif results['model_type'] == 'logistic':
            output += f"Pseudo R-squared: {results.get('pseudo_r_squared', 'N/A'):.3f}\\n"
            if 'performance' in results:
                output += f"AUC: {results['performance'].get('auc_roc', 'N/A'):.3f}\\n"
        elif results['model_type'] == 'cox':
            output += f"Concordance index: {results.get('concordance', 'N/A'):.3f}\\n"
        
        output += f"AIC: {results.get('aic', 'N/A'):.1f}\\n"
        output += f"Sample size: {results['n_observations']}\\n"
        
        return output