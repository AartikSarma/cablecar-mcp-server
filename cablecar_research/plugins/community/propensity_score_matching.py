"""
Propensity Score Matching Analysis Plugin

Community-contributed plugin for propensity score matching analysis
to reduce selection bias in observational studies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from cablecar_research.analysis.base import BaseAnalysis, AnalysisMetadata, AnalysisType, ValidationLevel


class PropensityScoreMatchingPlugin(BaseAnalysis):
    """
    Propensity score matching for causal inference in observational studies.
    
    This plugin implements propensity score matching to create balanced
    treatment and control groups from observational data.
    """
    
    metadata = AnalysisMetadata(
        name="propensity_score_matching",
        display_name="Propensity Score Matching",
        description="Propensity score matching for causal inference to reduce selection bias in observational studies",
        version="1.0.0",
        author="Community Contributor",
        email="community@cablecar.ai",
        analysis_type=AnalysisType.CAUSAL_INFERENCE,
        validation_level=ValidationLevel.STANDARD,
        citation="Community Plugin: Propensity Score Matching for CableCar. Based on Rosenbaum & Rubin (1983).",
        keywords=["propensity", "matching", "causal", "inference", "bias", "treatment", "observational"]
    )
    
    def validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """Validate inputs for propensity score matching."""
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
        
        # Check for required parameters
        treatment_var = kwargs.get('treatment_variable')
        if not treatment_var:
            validation['valid'] = False
            validation['errors'].append("'treatment_variable' parameter is required")
        elif treatment_var not in self.df.columns:
            validation['valid'] = False
            validation['errors'].append(f"Treatment variable '{treatment_var}' not found in data")
        else:
            # Check if treatment variable is binary
            unique_values = self.df[treatment_var].unique()
            if len(unique_values) != 2:
                validation['valid'] = False
                validation['errors'].append("Treatment variable must be binary (exactly 2 unique values)")
        
        covariates = kwargs.get('covariates', [])
        if not covariates:
            validation['valid'] = False
            validation['errors'].append("'covariates' parameter is required (list of variables for propensity score)")
        else:
            missing_covs = [cov for cov in covariates if cov not in self.df.columns]
            if missing_covs:
                validation['valid'] = False
                validation['errors'].append(f"Covariates not found in data: {missing_covs}")
        
        # Check sample size
        if len(self.df) < 50:
            validation['warnings'].append("Small sample size may affect matching quality")
        
        # Check for missing data
        if treatment_var and covariates:
            all_vars = [treatment_var] + covariates
            missing_any = self.df[all_vars].isna().any(axis=1).sum()
            if missing_any > 0:
                validation['warnings'].append(f"{missing_any} observations have missing data and will be excluded")
        
        return validation
    
    def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """Execute propensity score matching analysis."""
        treatment_var = kwargs.get('treatment_variable')
        covariates = kwargs.get('covariates', [])
        matching_method = kwargs.get('matching_method', 'nearest_neighbor')
        caliper = kwargs.get('caliper', 0.2)
        replacement = kwargs.get('replacement', False)
        
        # Remove missing data
        analysis_vars = [treatment_var] + covariates
        analysis_df = self.df[analysis_vars].dropna()
        
        results = {
            'analysis_type': 'propensity_score_matching',
            'treatment_variable': treatment_var,
            'covariates': covariates,
            'original_sample_size': len(self.df),
            'analysis_sample_size': len(analysis_df),
            'matching_method': matching_method,
            'caliper': caliper,
            'replacement': replacement
        }
        
        # Step 1: Estimate propensity scores
        X = analysis_df[covariates]
        y = analysis_df[treatment_var]
        
        # Fit logistic regression for propensity scores
        ps_model = LogisticRegression(random_state=42)
        ps_model.fit(X, y)
        propensity_scores = ps_model.predict_proba(X)[:, 1]
        
        results['propensity_score_model'] = {
            'coefficients': dict(zip(covariates, ps_model.coef_[0])),
            'intercept': ps_model.intercept_[0]
        }
        
        # Step 2: Assess pre-matching balance
        pre_balance = self._assess_balance(analysis_df, covariates, treatment_var)
        results['pre_matching_balance'] = pre_balance
        
        # Step 3: Perform matching
        matched_indices = self._perform_matching(
            propensity_scores, y, matching_method, caliper, replacement
        )
        
        if len(matched_indices) == 0:
            results['error'] = "No matches found - try adjusting caliper or matching method"
            return results
        
        # Create matched dataset
        matched_df = analysis_df.iloc[matched_indices].copy()
        matched_df['propensity_score'] = propensity_scores[matched_indices]
        matched_df['match_id'] = range(len(matched_indices))
        
        results['matched_sample_size'] = len(matched_df)
        results['matching_ratio'] = f"1:1"  # For now, only 1:1 matching implemented
        
        # Step 4: Assess post-matching balance
        post_balance = self._assess_balance(matched_df, covariates, treatment_var)
        results['post_matching_balance'] = post_balance
        
        # Step 5: Balance improvement summary
        balance_improvement = self._summarize_balance_improvement(pre_balance, post_balance)
        results['balance_improvement'] = balance_improvement
        
        # Step 6: Propensity score distribution analysis
        ps_distribution = self._analyze_ps_distribution(
            analysis_df, matched_df, propensity_scores, treatment_var
        )
        results['propensity_score_distribution'] = ps_distribution
        
        # Apply privacy protection
        if self.privacy_guard:
            results = self.privacy_guard.sanitize_output(results, 'propensity_matching')
        
        return results
    
    def format_results(self, results: Dict[str, Any], format_type: str = "standard") -> str:
        """Format propensity score matching results."""
        if format_type == "summary":
            return self._format_summary(results)
        elif format_type == "detailed":
            return self._format_detailed(results)
        elif format_type == "publication":
            return self._format_publication(results)
        else:
            return self._format_standard(results)
    
    def get_required_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter requirements for propensity score matching."""
        return {
            'required': {
                'treatment_variable': {
                    'type': 'string',
                    'description': 'Binary treatment/exposure variable',
                    'example': 'treatment_group'
                },
                'covariates': {
                    'type': 'array',
                    'description': 'List of covariates to include in propensity score model',
                    'example': ['age', 'sex', 'comorbidity_score']
                }
            },
            'optional': {
                'matching_method': {
                    'type': 'string',
                    'enum': ['nearest_neighbor', 'optimal'],
                    'description': 'Matching algorithm to use',
                    'example': 'nearest_neighbor'
                },
                'caliper': {
                    'type': 'number',
                    'description': 'Maximum distance for matching (in standard deviations)',
                    'example': 0.2
                },
                'replacement': {
                    'type': 'boolean',
                    'description': 'Whether to allow matching with replacement',
                    'example': False
                }
            }
        }
    
    def _perform_matching(self, propensity_scores: np.ndarray, treatment: pd.Series,
                         method: str, caliper: float, replacement: bool) -> List[int]:
        """Perform the actual propensity score matching."""
        treated_idx = treatment[treatment == 1].index.tolist()
        control_idx = treatment[treatment == 0].index.tolist()
        
        treated_ps = propensity_scores[treated_idx]
        control_ps = propensity_scores[control_idx]
        
        if method == 'nearest_neighbor':
            return self._nearest_neighbor_matching(
                treated_idx, control_idx, treated_ps, control_ps, caliper, replacement
            )
        else:
            # Default to nearest neighbor
            return self._nearest_neighbor_matching(
                treated_idx, control_idx, treated_ps, control_ps, caliper, replacement
            )
    
    def _nearest_neighbor_matching(self, treated_idx: List[int], control_idx: List[int],
                                 treated_ps: np.ndarray, control_ps: np.ndarray,
                                 caliper: float, replacement: bool) -> List[int]:
        """Nearest neighbor matching implementation."""
        matched_pairs = []
        used_controls = set()
        
        # Calculate caliper in absolute terms
        ps_std = np.std(np.concatenate([treated_ps, control_ps]))
        abs_caliper = caliper * ps_std
        
        for i, t_idx in enumerate(treated_idx):
            t_ps = treated_ps[i]
            
            # Find eligible controls
            if replacement:
                eligible_controls = list(range(len(control_idx)))
            else:
                eligible_controls = [j for j, c_idx in enumerate(control_idx) 
                                   if c_idx not in used_controls]
            
            if not eligible_controls:
                continue
            
            # Calculate distances
            distances = np.abs(control_ps[eligible_controls] - t_ps)
            
            # Apply caliper
            valid_matches = np.where(distances <= abs_caliper)[0]
            if len(valid_matches) == 0:
                continue
            
            # Find nearest neighbor
            nearest_idx = eligible_controls[valid_matches[np.argmin(distances[valid_matches])]]
            c_idx = control_idx[nearest_idx]
            
            matched_pairs.extend([t_idx, c_idx])
            if not replacement:
                used_controls.add(c_idx)
        
        return matched_pairs
    
    def _assess_balance(self, df: pd.DataFrame, covariates: List[str], 
                       treatment_var: str) -> Dict[str, Any]:
        """Assess covariate balance between treatment groups."""
        balance = {}
        
        treated = df[df[treatment_var] == 1]
        control = df[df[treatment_var] == 0]
        
        for covar in covariates:
            covar_balance = {}
            
            if pd.api.types.is_numeric_dtype(df[covar]):
                # Continuous variable
                treated_mean = treated[covar].mean()
                control_mean = control[covar].mean()
                pooled_std = np.sqrt(((treated[covar].var() + control[covar].var()) / 2))
                
                standardized_diff = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0
                
                covar_balance['treated_mean'] = treated_mean
                covar_balance['control_mean'] = control_mean
                covar_balance['standardized_difference'] = standardized_diff
                covar_balance['balanced'] = abs(standardized_diff) < 0.1  # Common threshold
                
                # T-test
                _, p_val = stats.ttest_ind(treated[covar].dropna(), control[covar].dropna())
                covar_balance['p_value'] = p_val
                
            else:
                # Categorical variable
                treated_prop = treated[covar].value_counts(normalize=True)
                control_prop = control[covar].value_counts(normalize=True)
                
                # Chi-square test
                contingency = pd.crosstab(df[covar], df[treatment_var])
                _, p_val, _, _ = stats.chi2_contingency(contingency)
                
                covar_balance['treated_distribution'] = treated_prop.to_dict()
                covar_balance['control_distribution'] = control_prop.to_dict()
                covar_balance['p_value'] = p_val
                covar_balance['balanced'] = p_val > 0.05
            
            balance[covar] = covar_balance
        
        # Overall balance summary
        if covariates:
            balanced_vars = sum(1 for v in balance.values() if v.get('balanced', False))
            balance['overall_balance_rate'] = balanced_vars / len(covariates)
        
        return balance
    
    def _summarize_balance_improvement(self, pre_balance: Dict, post_balance: Dict) -> Dict[str, Any]:
        """Summarize improvement in covariate balance."""
        improvement = {
            'variables_improved': 0,
            'variables_worsened': 0,
            'mean_abs_std_diff_before': 0,
            'mean_abs_std_diff_after': 0
        }
        
        std_diffs_before = []
        std_diffs_after = []
        
        for covar in pre_balance:
            if covar == 'overall_balance_rate':
                continue
                
            pre_std_diff = abs(pre_balance[covar].get('standardized_difference', 0))
            post_std_diff = abs(post_balance[covar].get('standardized_difference', 0))
            
            std_diffs_before.append(pre_std_diff)
            std_diffs_after.append(post_std_diff)
            
            if post_std_diff < pre_std_diff:
                improvement['variables_improved'] += 1
            elif post_std_diff > pre_std_diff:
                improvement['variables_worsened'] += 1
        
        if std_diffs_before:
            improvement['mean_abs_std_diff_before'] = np.mean(std_diffs_before)
            improvement['mean_abs_std_diff_after'] = np.mean(std_diffs_after)
            improvement['overall_improvement'] = improvement['mean_abs_std_diff_before'] - improvement['mean_abs_std_diff_after']
        
        return improvement
    
    def _analyze_ps_distribution(self, original_df: pd.DataFrame, matched_df: pd.DataFrame,
                               propensity_scores: np.ndarray, treatment_var: str) -> Dict[str, Any]:
        """Analyze propensity score distributions."""
        distribution = {}
        
        # Overall distribution statistics
        distribution['mean_ps'] = np.mean(propensity_scores)
        distribution['std_ps'] = np.std(propensity_scores)
        distribution['min_ps'] = np.min(propensity_scores)
        distribution['max_ps'] = np.max(propensity_scores)
        
        # Distribution by treatment group
        treated_ps = propensity_scores[original_df[treatment_var] == 1]
        control_ps = propensity_scores[original_df[treatment_var] == 0]
        
        distribution['treated_ps_mean'] = np.mean(treated_ps)
        distribution['control_ps_mean'] = np.mean(control_ps)
        
        # Overlap assessment
        treated_range = (np.min(treated_ps), np.max(treated_ps))
        control_range = (np.min(control_ps), np.max(control_ps))
        
        overlap_start = max(treated_range[0], control_range[0])
        overlap_end = min(treated_range[1], control_range[1])
        
        distribution['overlap_region'] = (overlap_start, overlap_end)
        distribution['has_overlap'] = overlap_start < overlap_end
        
        return distribution
    
    def _format_standard(self, results: Dict[str, Any]) -> str:
        """Standard formatting of PSM results."""
        output = f"Propensity Score Matching Analysis\n{'='*50}\n\n"
        
        # Sample size summary
        output += f"Original Sample: {results['original_sample_size']:,}\n"
        output += f"Analysis Sample: {results['analysis_sample_size']:,}\n"
        output += f"Matched Sample: {results['matched_sample_size']:,}\n"
        output += f"Matching Ratio: {results.get('matching_ratio', 'N/A')}\n\n"
        
        # Balance improvement
        if 'balance_improvement' in results:
            improvement = results['balance_improvement']
            output += f"Balance Improvement:\n"
            output += f"  Variables Improved: {improvement['variables_improved']}\n"
            output += f"  Variables Worsened: {improvement['variables_worsened']}\n"
            output += f"  Mean |Std Diff| Before: {improvement['mean_abs_std_diff_before']:.3f}\n"
            output += f"  Mean |Std Diff| After: {improvement['mean_abs_std_diff_after']:.3f}\n"
            output += f"  Overall Improvement: {improvement['overall_improvement']:.3f}\n\n"
        
        # Covariate balance
        if 'post_matching_balance' in results:
            output += "Post-Matching Covariate Balance:\n"
            for covar, balance in results['post_matching_balance'].items():
                if covar == 'overall_balance_rate':
                    continue
                    
                if 'standardized_difference' in balance:
                    std_diff = balance['standardized_difference']
                    balanced = "✓" if balance.get('balanced', False) else "✗"
                    output += f"  {covar}: Std Diff = {std_diff:.3f} {balanced}\n"
        
        return output
    
    def _format_summary(self, results: Dict[str, Any]) -> str:
        """Summary format."""
        matched_n = results.get('matched_sample_size', 0)
        original_n = results.get('original_sample_size', 0)
        matching_rate = (matched_n / original_n * 100) if original_n > 0 else 0
        
        output = f"Propensity Score Matching Summary\n"
        output += f"Matched {matched_n:,} of {original_n:,} observations ({matching_rate:.1f}%)\n"
        
        if 'balance_improvement' in results:
            improvement = results['balance_improvement']
            output += f"Balance improved for {improvement['variables_improved']} variables\n"
        
        return output
    
    def _format_detailed(self, results: Dict[str, Any]) -> str:
        """Detailed format with all statistics."""
        return self._format_standard(results)  # For now, same as standard
    
    def _format_publication(self, results: Dict[str, Any]) -> str:
        """Publication-ready format."""
        output = "Propensity Score Matching Results\n"
        output += "="*50 + "\n\n"
        
        output += f"We used propensity score matching to balance treatment groups. "
        output += f"Of {results['original_sample_size']:,} eligible patients, "
        output += f"{results['matched_sample_size']:,} were successfully matched. "
        
        if 'balance_improvement' in results:
            improvement = results['balance_improvement']
            output += f"The matching procedure improved balance for {improvement['variables_improved']} "
            output += f"of {len(results['covariates'])} covariates, with the mean absolute standardized "
            output += f"difference decreasing from {improvement['mean_abs_std_diff_before']:.3f} to "
            output += f"{improvement['mean_abs_std_diff_after']:.3f}."
        
        return output