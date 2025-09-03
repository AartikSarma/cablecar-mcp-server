"""
Regression Analysis Module

Comprehensive regression modeling for clinical research with:
- Multiple regression types (linear, logistic, Cox, mixed-effects)
- Automatic diagnostics and assumption checking
- Variable selection methods
- Model validation and performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Statistical modeling
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
try:
    from statsmodels.stats.diagnostic import jarque_bera
except ImportError:
    from scipy.stats import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, accuracy_score, classification_report, confusion_matrix
)

# Survival analysis
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# Mixed effects (if available)
try:
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_MIXED_EFFECTS = True
except ImportError:
    HAS_MIXED_EFFECTS = False


class RegressionAnalysis:
    """
    Comprehensive regression analysis for clinical research.
    
    Supports:
    - Linear regression with diagnostics
    - Logistic regression
    - Cox proportional hazards
    - Mixed-effects models
    - Variable selection
    - Cross-validation
    """
    
    def __init__(self, df: pd.DataFrame, privacy_guard=None):
        self.df = df.copy()
        self.privacy_guard = privacy_guard
        self.models = {}
        self.results = {}
        
    def linear_regression(self,
                         outcome: str,
                         predictors: List[str],
                         interaction_terms: Optional[List[str]] = None,
                         validate_assumptions: bool = True,
                         cross_validate: bool = True) -> Dict[str, Any]:
        """
        Perform linear regression with comprehensive diagnostics.
        
        Args:
            outcome: Dependent variable
            predictors: List of predictor variables
            interaction_terms: Interaction terms (e.g., ['age*sex'])
            validate_assumptions: Whether to check regression assumptions
            cross_validate: Whether to perform cross-validation
            
        Returns:
            Complete regression results with diagnostics
        """
        # Prepare data
        model_data = self._prepare_regression_data(outcome, predictors, interaction_terms)
        
        if model_data is None:
            return {'error': 'Insufficient data for regression'}
        
        X = model_data['X']
        y = model_data['y']
        formula = model_data['formula']
        
        # Fit OLS model using statsmodels for rich diagnostics
        model = smf.ols(formula, data=model_data['df']).fit()
        
        # Basic results
        results = {
            'model_type': 'linear_regression',
            'formula': formula,
            'n_observations': len(y),
            'n_predictors': len(X.columns),
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'f_pvalue': model.f_pvalue,
            'aic': model.aic,
            'bic': model.bic,
            'log_likelihood': model.llf
        }
        
        # Coefficient table
        coef_df = pd.DataFrame({
            'coefficient': model.params,
            'std_error': model.bse,
            't_statistic': model.tvalues,
            'p_value': model.pvalues,
            'ci_lower': model.conf_int()[0],
            'ci_upper': model.conf_int()[1]
        })
        results['coefficients'] = coef_df.to_dict('index')
        
        # Model diagnostics
        if validate_assumptions:
            diagnostics = self._linear_regression_diagnostics(model, X, y)
            results['diagnostics'] = diagnostics
        
        # Cross-validation
        if cross_validate:
            cv_results = self._cross_validate_linear(X, y)
            results['cross_validation'] = cv_results
        
        # Variable importance (standardized coefficients)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        model_scaled = sm.OLS(y, X_scaled).fit()
        
        results['standardized_coefficients'] = {
            var: coef for var, coef in zip(X.columns, model_scaled.params)
        }
        
        # Store model
        model_id = f"linear_regression_{len(self.models)}"
        self.models[model_id] = {
            'model': model,
            'type': 'linear',
            'data': model_data
        }
        results['model_id'] = model_id
        
        if self.privacy_guard:
            results = self.privacy_guard.sanitize_output(results)
            
        self.results['linear_regression'] = results
        return results
    
    def logistic_regression(self,
                           outcome: str,
                           predictors: List[str],
                           interaction_terms: Optional[List[str]] = None,
                           cross_validate: bool = True) -> Dict[str, Any]:
        """
        Perform logistic regression with model diagnostics.
        
        Args:
            outcome: Binary dependent variable
            predictors: List of predictor variables
            interaction_terms: Interaction terms
            cross_validate: Whether to perform cross-validation
            
        Returns:
            Logistic regression results with odds ratios
        """
        # Prepare data
        model_data = self._prepare_regression_data(outcome, predictors, interaction_terms)
        
        if model_data is None:
            return {'error': 'Insufficient data for regression'}
        
        X = model_data['X']
        y = model_data['y']
        
        # Check if outcome is binary
        if y.nunique() != 2:
            return {'error': 'Outcome must be binary for logistic regression'}
        
        # Fit logistic regression
        model = sm.Logit(y, X).fit(disp=0)
        
        # Basic results
        results = {
            'model_type': 'logistic_regression',
            'formula': model_data['formula'],
            'n_observations': len(y),
            'n_predictors': len(X.columns),
            'pseudo_r_squared': model.prsquared,
            'log_likelihood': model.llf,
            'aic': model.aic,
            'bic': model.bic,
            'likelihood_ratio_pvalue': model.llr_pvalue
        }
        
        # Coefficient table with odds ratios
        coef_df = pd.DataFrame({
            'coefficient': model.params,
            'std_error': model.bse,
            'z_statistic': model.tvalues,
            'p_value': model.pvalues,
            'odds_ratio': np.exp(model.params),
            'or_ci_lower': np.exp(model.conf_int()[0]),
            'or_ci_upper': np.exp(model.conf_int()[1])
        })
        results['coefficients'] = coef_df.to_dict('index')
        
        # Model performance metrics
        y_pred_proba = model.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        results['performance'] = {
            'accuracy': accuracy_score(y, y_pred),
            'auc_roc': roc_auc_score(y, y_pred_proba),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }
        
        # Hosmer-Lemeshow test for goodness of fit
        hl_statistic, hl_pvalue = self._hosmer_lemeshow_test(y, y_pred_proba)
        results['hosmer_lemeshow'] = {
            'statistic': hl_statistic,
            'p_value': hl_pvalue,
            'interpretation': 'Good fit' if hl_pvalue > 0.05 else 'Poor fit'
        }
        
        # Cross-validation
        if cross_validate:
            cv_results = self._cross_validate_logistic(X, y)
            results['cross_validation'] = cv_results
        
        # Store model
        model_id = f"logistic_regression_{len(self.models)}"
        self.models[model_id] = {
            'model': model,
            'type': 'logistic',
            'data': model_data
        }
        results['model_id'] = model_id
        
        if self.privacy_guard:
            results = self.privacy_guard.sanitize_output(results)
            
        self.results['logistic_regression'] = results
        return results
    
    def cox_regression(self,
                      duration_col: str,
                      event_col: str,
                      predictors: List[str],
                      interaction_terms: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform Cox proportional hazards regression.
        
        Args:
            duration_col: Time to event or censoring
            event_col: Event indicator (1=event, 0=censored)
            predictors: List of predictor variables
            interaction_terms: Interaction terms
            
        Returns:
            Cox regression results with hazard ratios
        """
        # Prepare data for survival analysis
        required_cols = [duration_col, event_col] + predictors
        if interaction_terms:
            # Parse interaction terms to get base variables
            for term in interaction_terms:
                vars_in_term = term.replace('*', ' ').replace(':', ' ').split()
                required_cols.extend(vars_in_term)
        
        model_df = self.df[required_cols].dropna()
        
        if len(model_df) < 10:
            return {'error': 'Insufficient data for Cox regression'}
        
        # Add interaction terms
        if interaction_terms:
            for term in interaction_terms:
                if '*' in term:
                    vars_in_term = term.split('*')
                    if len(vars_in_term) == 2:
                        var1, var2 = vars_in_term
                        model_df[term] = model_df[var1] * model_df[var2]
                        predictors.append(term)
        
        # Fit Cox model
        cph = CoxPHFitter()
        cph.fit(model_df, duration_col=duration_col, event_col=event_col)
        
        # Extract results
        results = {
            'model_type': 'cox_regression',
            'n_observations': len(model_df),
            'n_events': model_df[event_col].sum(),
            'n_predictors': len(predictors),
            'log_likelihood': cph.log_likelihood_,
            'aic': cph.AIC_,
            'concordance_index': cph.concordance_index_,
            'log_likelihood_ratio_test': {
                'statistic': cph.log_likelihood_ratio_test().test_statistic,
                'p_value': cph.log_likelihood_ratio_test().p_value
            }
        }
        
        # Coefficient table with hazard ratios
        summary_df = cph.summary
        coef_results = {}
        for idx, row in summary_df.iterrows():
            coef_results[idx] = {
                'coefficient': row['coef'],
                'hazard_ratio': row['exp(coef)'],
                'std_error': row['se(coef)'],
                'z_statistic': row['z'],
                'p_value': row['p'],
                'hr_ci_lower': row['exp(coef) lower 95%'],
                'hr_ci_upper': row['exp(coef) upper 95%']
            }
        
        results['coefficients'] = coef_results
        
        # Proportional hazards assumption test
        ph_test = cph.check_assumptions(model_df, p_value_threshold=0.05, show_plots=False)
        results['proportional_hazards_test'] = {
            'assumptions_met': ph_test.summary['test_statistic'].isna().all() or 
                             (ph_test.summary['p'] > 0.05).all(),
            'test_results': ph_test.summary.to_dict('index') if hasattr(ph_test, 'summary') else {}
        }
        
        # Store model
        model_id = f"cox_regression_{len(self.models)}"
        self.models[model_id] = {
            'model': cph,
            'type': 'cox',
            'data': model_df
        }
        results['model_id'] = model_id
        
        if self.privacy_guard:
            results = self.privacy_guard.sanitize_output(results)
            
        self.results['cox_regression'] = results
        return results
    
    def mixed_effects_regression(self,
                                outcome: str,
                                fixed_effects: List[str],
                                random_effects: List[str],
                                groups: str) -> Dict[str, Any]:
        """
        Perform mixed-effects (hierarchical) regression.
        
        Args:
            outcome: Dependent variable
            fixed_effects: Fixed effect predictors
            random_effects: Random effect predictors
            groups: Grouping variable for random effects
            
        Returns:
            Mixed-effects regression results
        """
        if not HAS_MIXED_EFFECTS:
            return {'error': 'Mixed-effects modeling requires statsmodels with mixed linear model support'}
        
        # Prepare data
        required_cols = [outcome, groups] + fixed_effects + random_effects
        model_df = self.df[required_cols].dropna()
        
        if len(model_df) < 20:
            return {'error': 'Insufficient data for mixed-effects regression'}
        
        # Build formula
        fixed_formula = f"{outcome} ~ " + " + ".join(fixed_effects)
        
        # Fit mixed-effects model
        try:
            model = MixedLM.from_formula(
                fixed_formula, 
                model_df, 
                groups=model_df[groups],
                re_formula=" + ".join(random_effects) if random_effects else "1"
            ).fit()
            
            results = {
                'model_type': 'mixed_effects_regression',
                'formula': fixed_formula,
                'random_effects': random_effects,
                'groups': groups,
                'n_observations': len(model_df),
                'n_groups': model_df[groups].nunique(),
                'log_likelihood': model.llf,
                'aic': model.aic,
                'bic': model.bic
            }
            
            # Fixed effects coefficients
            fixed_coefs = {}
            for idx, coef in model.params.items():
                fixed_coefs[idx] = {
                    'coefficient': coef,
                    'std_error': model.bse[idx],
                    't_statistic': model.tvalues[idx],
                    'p_value': model.pvalues[idx],
                    'ci_lower': model.conf_int().loc[idx, 0],
                    'ci_upper': model.conf_int().loc[idx, 1]
                }
            
            results['fixed_effects'] = fixed_coefs
            
            # Random effects variance components
            results['random_effects_variance'] = {
                'group_var': float(model.cov_re.iloc[0, 0]) if hasattr(model, 'cov_re') else None,
                'residual_var': float(model.scale)
            }
            
            # ICC (Intraclass Correlation Coefficient)
            if 'group_var' in results['random_effects_variance']:
                group_var = results['random_effects_variance']['group_var']
                residual_var = results['random_effects_variance']['residual_var']
                icc = group_var / (group_var + residual_var)
                results['icc'] = icc
            
            # Store model
            model_id = f"mixed_effects_{len(self.models)}"
            self.models[model_id] = {
                'model': model,
                'type': 'mixed_effects',
                'data': model_df
            }
            results['model_id'] = model_id
            
        except Exception as e:
            return {'error': f'Mixed-effects model failed: {str(e)}'}
        
        if self.privacy_guard:
            results = self.privacy_guard.sanitize_output(results)
            
        self.results['mixed_effects'] = results
        return results
    
    def variable_selection(self,
                          outcome: str,
                          candidate_predictors: List[str],
                          method: str = 'stepwise',
                          model_type: str = 'linear',
                          criterion: str = 'aic') -> Dict[str, Any]:
        """
        Perform automated variable selection.
        
        Args:
            outcome: Dependent variable
            candidate_predictors: Pool of potential predictors
            method: Selection method ('forward', 'backward', 'stepwise')
            model_type: Type of regression ('linear', 'logistic')
            criterion: Selection criterion ('aic', 'bic', 'pvalue')
            
        Returns:
            Variable selection results with final model
        """
        # Prepare data
        all_vars = [outcome] + candidate_predictors
        model_df = self.df[all_vars].dropna()
        
        if len(model_df) < len(candidate_predictors) * 10:
            return {'error': 'Insufficient observations for variable selection (need ~10 per variable)'}
        
        results = {
            'method': method,
            'criterion': criterion,
            'n_candidates': len(candidate_predictors),
            'n_observations': len(model_df)
        }
        
        if method == 'stepwise':
            selected_vars = self._stepwise_selection(
                model_df, outcome, candidate_predictors, model_type, criterion
            )
        elif method == 'forward':
            selected_vars = self._forward_selection(
                model_df, outcome, candidate_predictors, model_type, criterion
            )
        elif method == 'backward':
            selected_vars = self._backward_selection(
                model_df, outcome, candidate_predictors, model_type, criterion
            )
        else:
            return {'error': f'Unknown selection method: {method}'}
        
        results['selected_variables'] = selected_vars
        
        # Fit final model with selected variables
        if selected_vars:
            if model_type == 'linear':
                final_model_results = self.linear_regression(outcome, selected_vars, validate_assumptions=False)
            elif model_type == 'logistic':
                final_model_results = self.logistic_regression(outcome, selected_vars, cross_validate=False)
            else:
                return {'error': f'Unknown model type: {model_type}'}
            
            results['final_model'] = final_model_results
        else:
            results['final_model'] = {'error': 'No variables selected'}
        
        if self.privacy_guard:
            results = self.privacy_guard.sanitize_output(results)
            
        self.results['variable_selection'] = results
        return results
    
    def _prepare_regression_data(self, outcome: str, predictors: List[str], 
                                interaction_terms: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Prepare data for regression analysis."""
        all_vars = [outcome] + predictors
        
        # Add variables from interaction terms
        if interaction_terms:
            for term in interaction_terms:
                vars_in_term = term.replace('*', ' ').replace(':', ' ').split()
                all_vars.extend(vars_in_term)
        
        # Remove duplicates while preserving order
        all_vars = list(dict.fromkeys(all_vars))
        
        # Filter to available columns and remove missing data
        available_vars = [var for var in all_vars if var in self.df.columns]
        model_df = self.df[available_vars].dropna()
        
        if len(model_df) < 10:
            return None
        
        # Prepare predictor matrix
        X_vars = [var for var in predictors if var in model_df.columns]
        
        # Add interaction terms
        if interaction_terms:
            for term in interaction_terms:
                if '*' in term:
                    vars_in_term = term.split('*')
                    if len(vars_in_term) == 2 and all(var in model_df.columns for var in vars_in_term):
                        var1, var2 = vars_in_term
                        model_df[term] = model_df[var1] * model_df[var2]
                        X_vars.append(term)
        
        X = model_df[X_vars]
        X = sm.add_constant(X)  # Add intercept
        y = model_df[outcome]
        
        # Create formula for statsmodels
        formula_parts = [outcome, '~'] + X_vars
        formula = ' '.join(formula_parts)
        
        return {
            'X': X,
            'y': y,
            'df': model_df,
            'formula': formula
        }
    
    def _linear_regression_diagnostics(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform comprehensive linear regression diagnostics."""
        diagnostics = {}
        
        # Residuals
        residuals = model.resid
        fitted_values = model.fittedvalues
        
        # Normality of residuals (Jarque-Bera test)
        jb_stat, jb_pvalue = jarque_bera(residuals)
        diagnostics['residual_normality'] = {
            'jarque_bera_statistic': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'normal': jb_pvalue > 0.05
        }
        
        # Homoscedasticity (Breusch-Pagan test)
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X)
        diagnostics['homoscedasticity'] = {
            'breusch_pagan_statistic': bp_stat,
            'breusch_pagan_pvalue': bp_pvalue,
            'homoscedastic': bp_pvalue > 0.05
        }
        
        # Multicollinearity (VIF)
        vif_data = {}
        for i, var in enumerate(X.columns):
            if var != 'const':  # Skip intercept
                try:
                    vif_value = variance_inflation_factor(X.values, i)
                    vif_data[var] = vif_value
                except:
                    vif_data[var] = None
        
        diagnostics['multicollinearity'] = {
            'vif_values': vif_data,
            'high_vif_variables': [var for var, vif in vif_data.items() if vif and vif > 5]
        }
        
        # Outliers and influential observations
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        leverage = influence.hat_matrix_diag
        
        # Identify outliers (Cook's D > 4/n)
        n = len(y)
        outlier_threshold = 4 / n
        outliers = np.where(cooks_d > outlier_threshold)[0]
        
        diagnostics['outliers'] = {
            'cooks_d_threshold': outlier_threshold,
            'n_outliers': len(outliers),
            'outlier_indices': outliers.tolist(),
            'max_cooks_d': float(cooks_d.max()),
            'max_leverage': float(leverage.max())
        }
        
        return diagnostics
    
    def _cross_validate_linear(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation for linear regression."""
        # Use scikit-learn for cross-validation
        X_no_const = X.drop('const', axis=1) if 'const' in X.columns else X
        
        lr = LinearRegression()
        cv_scores = cross_val_score(lr, X_no_const, y, cv=cv_folds, scoring='r2')
        
        return {
            'cv_folds': cv_folds,
            'mean_r2': float(cv_scores.mean()),
            'std_r2': float(cv_scores.std()),
            'cv_scores': cv_scores.tolist()
        }
    
    def _cross_validate_logistic(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation for logistic regression."""
        X_no_const = X.drop('const', axis=1) if 'const' in X.columns else X
        
        lr = LogisticRegression(max_iter=1000)
        
        # ROC AUC scores
        auc_scores = cross_val_score(lr, X_no_const, y, cv=cv_folds, scoring='roc_auc')
        
        # Accuracy scores
        acc_scores = cross_val_score(lr, X_no_const, y, cv=cv_folds, scoring='accuracy')
        
        return {
            'cv_folds': cv_folds,
            'mean_auc': float(auc_scores.mean()),
            'std_auc': float(auc_scores.std()),
            'auc_scores': auc_scores.tolist(),
            'mean_accuracy': float(acc_scores.mean()),
            'std_accuracy': float(acc_scores.std()),
            'accuracy_scores': acc_scores.tolist()
        }
    
    def _hosmer_lemeshow_test(self, y_true: pd.Series, y_prob: pd.Series, groups: int = 10) -> Tuple[float, float]:
        """Perform Hosmer-Lemeshow goodness of fit test."""
        # Create groups based on predicted probabilities
        y_df = pd.DataFrame({'true': y_true, 'prob': y_prob})
        y_df['group'] = pd.qcut(y_df['prob'], q=groups, duplicates='drop')
        
        # Calculate observed and expected for each group
        grouped = y_df.groupby('group')
        obs = grouped['true'].sum()
        exp = grouped['prob'].sum()
        n = grouped.size()
        
        # Hosmer-Lemeshow statistic
        hl_stat = sum((obs - exp)**2 / (exp * (1 - exp/n)))
        
        # P-value from chi-square distribution
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(hl_stat, df=groups-2)
        
        return float(hl_stat), float(p_value)
    
    def _stepwise_selection(self, df: pd.DataFrame, outcome: str, 
                           candidates: List[str], model_type: str, criterion: str) -> List[str]:
        """Perform stepwise variable selection."""
        selected = []
        remaining = candidates.copy()
        
        while remaining:
            # Forward step: try adding each remaining variable
            forward_results = []
            for var in remaining:
                test_vars = selected + [var]
                score = self._evaluate_model(df, outcome, test_vars, model_type, criterion)
                forward_results.append((var, score))
            
            # Find best addition
            best_var, best_score = min(forward_results, key=lambda x: x[1])
            
            # Check if improvement is significant
            current_score = self._evaluate_model(df, outcome, selected, model_type, criterion) if selected else float('inf')
            
            if best_score < current_score:
                selected.append(best_var)
                remaining.remove(best_var)
                
                # Backward step: try removing each selected variable
                if len(selected) > 1:
                    backward_results = []
                    for var in selected:
                        test_vars = [v for v in selected if v != var]
                        score = self._evaluate_model(df, outcome, test_vars, model_type, criterion)
                        backward_results.append((var, score))
                    
                    # Find worst variable to potentially remove
                    worst_var, worst_score = min(backward_results, key=lambda x: x[1])
                    
                    if worst_score < best_score:
                        selected.remove(worst_var)
                        remaining.append(worst_var)
            else:
                break
        
        return selected
    
    def _forward_selection(self, df: pd.DataFrame, outcome: str,
                          candidates: List[str], model_type: str, criterion: str) -> List[str]:
        """Perform forward variable selection."""
        selected = []
        remaining = candidates.copy()
        
        while remaining:
            best_score = float('inf')
            best_var = None
            
            for var in remaining:
                test_vars = selected + [var]
                score = self._evaluate_model(df, outcome, test_vars, model_type, criterion)
                
                if score < best_score:
                    best_score = score
                    best_var = var
            
            # Check if improvement is significant
            current_score = self._evaluate_model(df, outcome, selected, model_type, criterion) if selected else float('inf')
            
            if best_score < current_score:
                selected.append(best_var)
                remaining.remove(best_var)
            else:
                break
        
        return selected
    
    def _backward_selection(self, df: pd.DataFrame, outcome: str,
                           candidates: List[str], model_type: str, criterion: str) -> List[str]:
        """Perform backward variable selection."""
        selected = candidates.copy()
        
        while len(selected) > 1:
            best_score = float('inf')
            worst_var = None
            
            for var in selected:
                test_vars = [v for v in selected if v != var]
                score = self._evaluate_model(df, outcome, test_vars, model_type, criterion)
                
                if score < best_score:
                    best_score = score
                    worst_var = var
            
            # Check if removing variable improves the model
            current_score = self._evaluate_model(df, outcome, selected, model_type, criterion)
            
            if best_score < current_score:
                selected.remove(worst_var)
            else:
                break
        
        return selected
    
    def _evaluate_model(self, df: pd.DataFrame, outcome: str, 
                       predictors: List[str], model_type: str, criterion: str) -> float:
        """Evaluate a model with given predictors."""
        if not predictors:
            return float('inf')
        
        try:
            if model_type == 'linear':
                X = sm.add_constant(df[predictors])
                y = df[outcome]
                model = sm.OLS(y, X).fit()
                
                if criterion == 'aic':
                    return model.aic
                elif criterion == 'bic':
                    return model.bic
                else:
                    return -model.rsquared_adj
                    
            elif model_type == 'logistic':
                X = sm.add_constant(df[predictors])
                y = df[outcome]
                model = sm.Logit(y, X).fit(disp=0)
                
                if criterion == 'aic':
                    return model.aic
                elif criterion == 'bic':
                    return model.bic
                else:
                    return -model.prsquared
        
        except:
            return float('inf')
        
        return float('inf')