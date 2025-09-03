"""
Hypothesis Testing Plugin

Comprehensive statistical hypothesis testing for clinical research including:
- Parametric and non-parametric tests
- Multiple comparison corrections
- Effect size calculations
- Power analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from cablecar_research.analysis.base import BaseAnalysis, AnalysisMetadata, AnalysisType, ValidationLevel


class HypothesisTestingPlugin(BaseAnalysis):
    """
    Comprehensive statistical hypothesis testing for clinical research.
    
    Performs appropriate statistical tests based on data types and distributions,
    with support for multiple comparison corrections and effect size calculations.
    """
    
    metadata = AnalysisMetadata(
        name="hypothesis_testing",
        display_name="Hypothesis Testing",
        description="Comprehensive statistical hypothesis testing with multiple comparison corrections and effect sizes",
        version="1.0.0",
        author="CableCar Team",
        email="support@cablecar.ai",
        analysis_type=AnalysisType.INFERENTIAL,
        validation_level=ValidationLevel.STANDARD,
        citation="CableCar Research Team. Hypothesis Testing Plugin for Clinical Research. CableCar v1.0.0",
        keywords=["hypothesis", "statistical", "tests", "pvalue", "comparison", "inferential"]
    )
    
    def validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """Validate inputs for hypothesis testing."""
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
        outcome_vars = kwargs.get('outcome_variables')
        group_var = kwargs.get('group_variable')
        
        if not outcome_vars:
            validation['valid'] = False
            validation['errors'].append("'outcome_variables' parameter is required")
        elif not isinstance(outcome_vars, list):
            validation['valid'] = False
            validation['errors'].append("'outcome_variables' must be a list")
        
        if not group_var:
            validation['valid'] = False
            validation['errors'].append("'group_variable' parameter is required")
        
        # Check if variables exist in data
        if outcome_vars and isinstance(outcome_vars, list):
            missing_vars = [var for var in outcome_vars if var not in self.df.columns]
            if missing_vars:
                validation['valid'] = False
                validation['errors'].append(f"Outcome variables not found in data: {missing_vars}")
        
        if group_var and group_var not in self.df.columns:
            validation['valid'] = False
            validation['errors'].append(f"Group variable '{group_var}' not found in data")
        
        # Check group variable has sufficient groups
        if group_var and group_var in self.df.columns:
            n_groups = self.df[group_var].nunique()
            if n_groups < 2:
                validation['valid'] = False
                validation['errors'].append(f"Group variable '{group_var}' must have at least 2 groups")
            elif n_groups > 10:
                validation['warnings'].append(f"Group variable has {n_groups} groups - consider reducing for clarity")
        
        # Check sample sizes
        if len(self.df) < 30:
            validation['warnings'].append("Small sample size (n<30) may affect statistical power")
        
        return validation
    
    def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """Execute comprehensive hypothesis testing."""
        outcome_vars = kwargs.get('outcome_variables', [])
        group_var = kwargs.get('group_variable')
        test_type = kwargs.get('test_type', 'auto')
        correction_method = kwargs.get('correction_method', 'fdr_bh')
        alpha = kwargs.get('alpha', 0.05)
        
        results = {
            'analysis_type': 'hypothesis_testing',
            'outcome_variables': outcome_vars,
            'group_variable': group_var,
            'n_tests': len(outcome_vars),
            'correction_method': correction_method,
            'alpha_level': alpha,
            'sample_size': len(self.df),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Get groups and their sizes
        groups = self.df[group_var].unique()
        group_sizes = self.df[group_var].value_counts().to_dict()
        results['groups'] = groups.tolist()
        results['group_sizes'] = group_sizes
        
        # Perform tests for each outcome variable
        test_results = {}
        p_values = []
        
        for outcome in outcome_vars:
            test_result = self._perform_single_test(
                outcome, group_var, test_type, alpha
            )
            test_results[outcome] = test_result
            
            if test_result.get('p_value') is not None:
                p_values.append(test_result['p_value'])
        
        results['test_results'] = test_results
        
        # Apply multiple comparison correction
        if len(p_values) > 1 and correction_method != 'none':
            corrected_p_values = self._apply_multiple_comparison_correction(
                p_values, correction_method
            )
            
            # Update results with corrected p-values
            for i, outcome in enumerate(outcome_vars):
                if i < len(corrected_p_values):
                    results['test_results'][outcome]['p_value_corrected'] = corrected_p_values[i]
                    results['test_results'][outcome]['significant_corrected'] = corrected_p_values[i] < alpha
        
        # Summary statistics
        n_significant = sum(1 for test in test_results.values() 
                           if test.get('p_value_corrected', test.get('p_value', 1)) < alpha)
        results['summary'] = {
            'n_tests_performed': len(outcome_vars),
            'n_significant': n_significant,
            'proportion_significant': n_significant / len(outcome_vars) if outcome_vars else 0
        }
        
        # Apply privacy protection
        if self.privacy_guard:
            results = self.privacy_guard.sanitize_output(results, 'hypothesis_testing')
        
        return results
    
    def format_results(self, results: Dict[str, Any], format_type: str = "standard") -> str:
        """Format hypothesis testing results."""
        if format_type == "summary":
            return self._format_summary(results)
        elif format_type == "detailed":
            return self._format_detailed(results)
        elif format_type == "publication":
            return self._format_publication(results)
        else:
            return self._format_standard(results)
    
    def get_required_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter requirements for hypothesis testing."""
        return {
            'required': {
                'outcome_variables': {
                    'type': 'array',
                    'description': 'List of outcome variables to test',
                    'example': ['mortality', 'length_of_stay', 'complications']
                },
                'group_variable': {
                    'type': 'string',
                    'description': 'Grouping variable for comparisons',
                    'example': 'treatment_group'
                }
            },
            'optional': {
                'test_type': {
                    'type': 'string',
                    'enum': ['auto', 'parametric', 'non_parametric'],
                    'description': 'Type of statistical test to use',
                    'example': 'auto'
                },
                'correction_method': {
                    'type': 'string',
                    'enum': ['fdr_bh', 'bonferroni', 'holm', 'none'],
                    'description': 'Multiple comparison correction method',
                    'example': 'fdr_bh'
                },
                'alpha': {
                    'type': 'number',
                    'description': 'Significance level',
                    'example': 0.05
                },
                'include_effect_sizes': {
                    'type': 'boolean',
                    'description': 'Calculate and report effect sizes',
                    'example': True
                }
            }
        }
    
    def _perform_single_test(self, outcome: str, group_var: str, test_type: str, alpha: float) -> Dict[str, Any]:
        """Perform statistical test for a single outcome variable."""
        test_result = {
            'outcome': outcome,
            'test_name': None,
            'test_statistic': None,
            'p_value': None,
            'significant': None,
            'effect_size': None,
            'effect_size_interpretation': None,
            'assumptions_met': None,
            'recommendation': None
        }
        
        try:
            # Get data for this outcome
            outcome_data = self.df[[outcome, group_var]].dropna()
            
            if len(outcome_data) < 3:
                test_result['recommendation'] = "Insufficient data for testing"
                return test_result
            
            # Determine if outcome is continuous or categorical
            is_continuous = pd.api.types.is_numeric_dtype(outcome_data[outcome])
            groups = outcome_data[group_var].unique()
            n_groups = len(groups)
            
            if is_continuous:
                # Continuous outcome
                if n_groups == 2:
                    # Two-sample tests
                    group_data = [outcome_data[outcome_data[group_var] == group][outcome] 
                                 for group in groups]
                    
                    # Check assumptions
                    normality_ok = all(self._check_normality(data) for data in group_data if len(data) > 3)
                    equal_var_ok = self._check_equal_variance(group_data)
                    
                    test_result['assumptions_met'] = {
                        'normality': normality_ok,
                        'equal_variance': equal_var_ok
                    }
                    
                    if test_type == 'parametric' or (test_type == 'auto' and normality_ok and equal_var_ok):
                        # t-test
                        stat, p_val = stats.ttest_ind(group_data[0], group_data[1], equal_var=equal_var_ok)
                        test_result['test_name'] = "Independent t-test"
                        test_result['test_statistic'] = stat
                        test_result['p_value'] = p_val
                        
                        # Cohen's d effect size
                        pooled_std = np.sqrt(((len(group_data[0]) - 1) * group_data[0].std()**2 + 
                                            (len(group_data[1]) - 1) * group_data[1].std()**2) / 
                                           (len(group_data[0]) + len(group_data[1]) - 2))
                        cohens_d = (group_data[0].mean() - group_data[1].mean()) / pooled_std
                        test_result['effect_size'] = abs(cohens_d)
                        test_result['effect_size_interpretation'] = self._interpret_cohens_d(abs(cohens_d))
                        
                    else:
                        # Mann-Whitney U test
                        stat, p_val = stats.mannwhitneyu(group_data[0], group_data[1])
                        test_result['test_name'] = "Mann-Whitney U test"
                        test_result['test_statistic'] = stat
                        test_result['p_value'] = p_val
                        
                        # Effect size r = Z / sqrt(N)
                        z_score = stats.norm.ppf(p_val/2)  # Approximate
                        n_total = len(group_data[0]) + len(group_data[1])
                        effect_r = abs(z_score) / np.sqrt(n_total)
                        test_result['effect_size'] = effect_r
                        test_result['effect_size_interpretation'] = self._interpret_r_effect_size(effect_r)
                
                elif n_groups > 2:
                    # Multi-group tests
                    group_data = [outcome_data[outcome_data[group_var] == group][outcome] 
                                 for group in groups]
                    
                    # Check assumptions
                    normality_ok = all(self._check_normality(data) for data in group_data if len(data) > 3)
                    equal_var_ok = self._check_equal_variance(group_data)
                    
                    test_result['assumptions_met'] = {
                        'normality': normality_ok,
                        'equal_variance': equal_var_ok
                    }
                    
                    if test_type == 'parametric' or (test_type == 'auto' and normality_ok and equal_var_ok):
                        # ANOVA
                        stat, p_val = stats.f_oneway(*group_data)
                        test_result['test_name'] = "One-way ANOVA"
                        test_result['test_statistic'] = stat
                        test_result['p_value'] = p_val
                        
                        # Eta squared effect size
                        ss_between = sum(len(data) * (data.mean() - outcome_data[outcome].mean())**2 
                                       for data in group_data)
                        ss_total = sum((outcome_data[outcome] - outcome_data[outcome].mean())**2)
                        eta_squared = ss_between / ss_total if ss_total > 0 else 0
                        test_result['effect_size'] = eta_squared
                        test_result['effect_size_interpretation'] = self._interpret_eta_squared(eta_squared)
                    else:
                        # Kruskal-Wallis test
                        stat, p_val = stats.kruskal(*group_data)
                        test_result['test_name'] = "Kruskal-Wallis test"
                        test_result['test_statistic'] = stat
                        test_result['p_value'] = p_val
            
            else:
                # Categorical outcome - Chi-square test
                contingency_table = pd.crosstab(outcome_data[outcome], outcome_data[group_var])
                
                # Check minimum expected frequencies
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
                min_expected = expected.min()
                
                test_result['assumptions_met'] = {
                    'min_expected_frequency': min_expected >= 5
                }
                
                if min_expected >= 5:
                    test_result['test_name'] = "Chi-square test"
                    test_result['test_statistic'] = chi2
                    test_result['p_value'] = p_val
                    
                    # Cramer's V effect size
                    n = contingency_table.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * min(contingency_table.shape) - 1))
                    test_result['effect_size'] = cramers_v
                    test_result['effect_size_interpretation'] = self._interpret_cramers_v(cramers_v)
                else:
                    # Fisher's exact test for 2x2 tables
                    if contingency_table.shape == (2, 2):
                        odds_ratio, p_val = stats.fisher_exact(contingency_table)
                        test_result['test_name'] = "Fisher's exact test"
                        test_result['test_statistic'] = odds_ratio
                        test_result['p_value'] = p_val
                    else:
                        test_result['recommendation'] = "Use Fisher's exact test or increase sample size"
            
            test_result['significant'] = test_result['p_value'] < alpha if test_result['p_value'] else False
            
        except Exception as e:
            test_result['recommendation'] = f"Test failed: {str(e)}"
        
        return test_result
    
    def _check_normality(self, data: pd.Series) -> bool:
        """Check normality using Shapiro-Wilk test."""
        if len(data) < 3:
            return False
        try:
            _, p_val = stats.shapiro(data.sample(min(5000, len(data))))
            return p_val > 0.05
        except:
            return False
    
    def _check_equal_variance(self, group_data: List[pd.Series]) -> bool:
        """Check equal variance using Levene's test."""
        if len(group_data) < 2:
            return True
        try:
            _, p_val = stats.levene(*group_data)
            return p_val > 0.05
        except:
            return True
    
    def _apply_multiple_comparison_correction(self, p_values: List[float], method: str) -> List[float]:
        """Apply multiple comparison correction."""
        from statsmodels.stats.multitest import multipletests
        
        try:
            _, corrected_p, _, _ = multipletests(p_values, method=method)
            return corrected_p.tolist()
        except:
            # Fallback to Bonferroni
            return [min(p * len(p_values), 1.0) for p in p_values]
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_r_effect_size(self, r: float) -> str:
        """Interpret r effect size."""
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small"
        elif r < 0.5:
            return "medium"
        else:
            return "large"
    
    def _interpret_eta_squared(self, eta2: float) -> str:
        """Interpret eta squared effect size."""
        if eta2 < 0.01:
            return "negligible"
        elif eta2 < 0.06:
            return "small"
        elif eta2 < 0.14:
            return "medium"
        else:
            return "large"
    
    def _interpret_cramers_v(self, v: float) -> str:
        """Interpret Cramer's V effect size."""
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "small"
        elif v < 0.5:
            return "medium"
        else:
            return "large"
    
    def _format_standard(self, results: Dict[str, Any]) -> str:
        """Standard formatting of results."""
        output = f"Hypothesis Testing Results\\n{'='*50}\\n\\n"
        output += f"Group Variable: {results['group_variable']}\\n"
        output += f"Groups: {', '.join(map(str, results['groups']))}\\n"
        output += f"Total Tests: {results['n_tests']}\\n"
        output += f"Correction Method: {results['correction_method']}\\n\\n"
        
        # Summary
        summary = results['summary']
        output += f"Summary: {summary['n_significant']}/{summary['n_tests_performed']} tests significant "
        output += f"({summary['proportion_significant']:.1%})\\n\\n"
        
        # Individual test results
        output += "Individual Test Results:\\n"
        output += "-" * 40 + "\\n"
        
        for outcome, test_result in results['test_results'].items():
            output += f"\\n{outcome.replace('_', ' ').title()}:\\n"
            
            if test_result.get('test_name'):
                output += f"  Test: {test_result['test_name']}\\n"
                
                p_val = test_result.get('p_value_corrected', test_result.get('p_value'))
                if p_val is not None:
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    output += f"  p-value: {p_val:.4f} {significance}\\n"
                    
                if test_result.get('effect_size'):
                    output += f"  Effect size: {test_result['effect_size']:.3f} ({test_result.get('effect_size_interpretation', 'N/A')})\\n"
                    
                if test_result.get('assumptions_met'):
                    assumptions = test_result['assumptions_met']
                    if isinstance(assumptions, dict):
                        failed_assumptions = [k for k, v in assumptions.items() if not v]
                        if failed_assumptions:
                            output += f"  Assumption violations: {', '.join(failed_assumptions)}\\n"
            else:
                output += f"  {test_result.get('recommendation', 'Test not performed')}\\n"
        
        return output
    
    def _format_summary(self, results: Dict[str, Any]) -> str:
        """Summary formatting."""
        summary = results['summary']
        return f"Hypothesis Testing Summary: {summary['n_significant']}/{summary['n_tests_performed']} tests significant ({summary['proportion_significant']:.1%})"
    
    def _format_detailed(self, results: Dict[str, Any]) -> str:
        """Detailed formatting with all statistics."""
        return self._format_standard(results)  # Same as standard for now
    
    def _format_publication(self, results: Dict[str, Any]) -> str:
        """Publication-ready formatting."""
        output = f"Table: Statistical Test Results\\n{'='*50}\\n\\n"
        output += f"Outcome\\tTest\\tp-value\\tAdjusted p\\tEffect Size\\tInterpretation\\n"
        output += "-" * 80 + "\\n"
        
        for outcome, test_result in results['test_results'].items():
            outcome_name = outcome.replace('_', ' ').title()
            test_name = test_result.get('test_name', 'N/A')
            p_val = test_result.get('p_value', 'N/A')
            p_adj = test_result.get('p_value_corrected', 'N/A')
            effect_size = test_result.get('effect_size', 'N/A')
            interpretation = test_result.get('effect_size_interpretation', 'N/A')
            
            if isinstance(p_val, float):
                p_val = f"{p_val:.3f}"
            if isinstance(p_adj, float):
                p_adj = f"{p_adj:.3f}"
            if isinstance(effect_size, float):
                effect_size = f"{effect_size:.3f}"
            
            output += f"{outcome_name}\\t{test_name}\\t{p_val}\\t{p_adj}\\t{effect_size}\\t{interpretation}\\n"
        
        output += f"\\nMultiple comparison correction: {results['correction_method']}\\n"
        
        return output