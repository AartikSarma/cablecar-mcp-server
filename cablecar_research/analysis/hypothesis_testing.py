"""
Hypothesis Testing Module

Comprehensive statistical testing with proper multiple comparison corrections
and effect size calculations. Designed for clinical research applications.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.contingency_tables import mcnemar
import warnings
warnings.filterwarnings('ignore')


class HypothesisTesting:
    """
    Comprehensive hypothesis testing for clinical research.
    
    Supports:
    - Parametric and non-parametric tests
    - Multiple comparison corrections
    - Effect size calculations
    - Power analysis
    - Assumption checking
    """
    
    def __init__(self, df: pd.DataFrame, privacy_guard=None):
        self.df = df.copy()
        self.privacy_guard = privacy_guard
        self.results = {}
        self.test_history = []
        
    def compare_groups(self,
                      outcome_vars: List[str],
                      group_var: str,
                      test_type: str = 'auto',
                      alpha: float = 0.05,
                      multiple_comparison_method: str = 'fdr_bh') -> Dict[str, Any]:
        """
        Compare multiple outcomes between groups with appropriate statistical tests.
        
        Args:
            outcome_vars: List of outcome variables to test
            group_var: Grouping variable
            test_type: 'auto', 'parametric', 'non_parametric'
            alpha: Significance level
            multiple_comparison_method: Method for multiple comparison correction
            
        Returns:
            Dictionary with test results for all variables
        """
        groups = self.df[group_var].dropna().unique()
        n_groups = len(groups)
        
        if n_groups < 2:
            return {'error': 'Need at least 2 groups for comparison'}
        
        test_results = {}
        p_values = []
        
        for outcome in outcome_vars:
            if outcome not in self.df.columns:
                continue
                
            # Determine appropriate test
            is_categorical = self._is_categorical(outcome)
            
            if is_categorical:
                result = self._test_categorical_outcome(outcome, group_var)
            else:
                if test_type == 'auto':
                    test_type = self._determine_test_type(outcome, group_var)
                    
                if n_groups == 2:
                    result = self._test_continuous_two_groups(outcome, group_var, test_type)
                else:
                    result = self._test_continuous_multiple_groups(outcome, group_var, test_type)
            
            test_results[outcome] = result
            p_values.append(result.get('p_value', 1.0))
            
            # Store in history
            self.test_history.append({
                'outcome': outcome,
                'group_var': group_var,
                'test_type': result.get('test_name', 'unknown'),
                'p_value': result.get('p_value', 1.0)
            })
        
        # Apply multiple comparison correction
        if len(p_values) > 1 and multiple_comparison_method:
            corrected_results = self._apply_multiple_comparison_correction(
                p_values, multiple_comparison_method, alpha
            )
            
            # Update results with corrected p-values
            for i, outcome in enumerate(test_results.keys()):
                test_results[outcome]['p_value_corrected'] = corrected_results['p_corrected'][i]
                test_results[outcome]['significant_corrected'] = corrected_results['reject'][i]
        
        final_result = {
            'group_variable': group_var,
            'n_groups': n_groups,
            'groups': list(groups),
            'n_tests': len(outcome_vars),
            'alpha': alpha,
            'multiple_comparison_method': multiple_comparison_method,
            'test_results': test_results
        }
        
        if self.privacy_guard:
            final_result = self.privacy_guard.sanitize_output(final_result)
            
        self.results['group_comparisons'] = final_result
        return final_result
    
    def _test_categorical_outcome(self, outcome: str, group_var: str) -> Dict[str, Any]:
        """Test categorical outcome between groups."""
        contingency_table = pd.crosstab(self.df[outcome], self.df[group_var])
        
        # Chi-square test
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Cramér's V for effect size
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
        
        result = {
            'test_name': 'Chi-square test',
            'statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'effect_size': cramers_v,
            'effect_size_interpretation': self._interpret_cramers_v(cramers_v),
            'contingency_table': contingency_table.to_dict(),
            'expected_frequencies': expected.tolist()
        }
        
        # Check assumptions
        min_expected = expected.min()
        if min_expected < 5:
            result['assumption_warning'] = f"Minimum expected frequency is {min_expected:.2f} < 5. Consider Fisher's exact test."
            
            # Perform Fisher's exact test for 2x2 tables
            if contingency_table.shape == (2, 2):
                oddsratio, fisher_p = stats.fisher_exact(contingency_table)
                result['fisher_exact_p'] = fisher_p
                result['odds_ratio'] = oddsratio
        
        return result
    
    def _test_continuous_two_groups(self, outcome: str, group_var: str, test_type: str) -> Dict[str, Any]:
        """Test continuous outcome between two groups."""
        groups = self.df[group_var].dropna().unique()
        group1_data = self.df[self.df[group_var] == groups[0]][outcome].dropna()
        group2_data = self.df[self.df[group_var] == groups[1]][outcome].dropna()
        
        if test_type == 'parametric':
            # Independent t-test
            statistic, p_value = stats.ttest_ind(group1_data, group2_data)
            test_name = "Independent t-test"
            
            # Check equal variances assumption
            levene_stat, levene_p = stats.levene(group1_data, group2_data)
            equal_variances = levene_p > 0.05
            
            if not equal_variances:
                # Welch's t-test (unequal variances)
                statistic, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                test_name = "Welch's t-test"
                
            # Cohen's d
            pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                                (len(group2_data) - 1) * group2_data.var()) / 
                               (len(group1_data) + len(group2_data) - 2))
            cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
            
            result = {
                'test_name': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'effect_size': cohens_d,
                'effect_size_interpretation': self._interpret_cohens_d(cohens_d),
                'group1_mean': group1_data.mean(),
                'group1_std': group1_data.std(),
                'group2_mean': group2_data.mean(),
                'group2_std': group2_data.std(),
                'equal_variances': equal_variances,
                'levene_p': levene_p
            }
            
        else:
            # Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            
            # Effect size (rank biserial correlation)
            n1, n2 = len(group1_data), len(group2_data)
            rank_biserial = 2 * statistic / (n1 * n2) - 1
            
            result = {
                'test_name': 'Mann-Whitney U test',
                'statistic': statistic,
                'p_value': p_value,
                'effect_size': rank_biserial,
                'effect_size_interpretation': self._interpret_rank_biserial(rank_biserial),
                'group1_median': group1_data.median(),
                'group1_iqr': group1_data.quantile(0.75) - group1_data.quantile(0.25),
                'group2_median': group2_data.median(),
                'group2_iqr': group2_data.quantile(0.75) - group2_data.quantile(0.25)
            }
        
        # Add sample sizes
        result.update({
            'group1_n': len(group1_data),
            'group2_n': len(group2_data),
            'group1_name': groups[0],
            'group2_name': groups[1]
        })
        
        return result
    
    def _test_continuous_multiple_groups(self, outcome: str, group_var: str, test_type: str) -> Dict[str, Any]:
        """Test continuous outcome between multiple groups."""
        groups = self.df[group_var].dropna().unique()
        group_data = [self.df[self.df[group_var] == group][outcome].dropna() for group in groups]
        
        if test_type == 'parametric':
            # One-way ANOVA
            statistic, p_value = stats.f_oneway(*group_data)
            test_name = "One-way ANOVA"
            
            # Calculate eta-squared (effect size)
            ss_between = sum(len(group) * (group.mean() - self.df[outcome].mean())**2 for group in group_data)
            ss_total = ((self.df[outcome] - self.df[outcome].mean())**2).sum()
            eta_squared = ss_between / ss_total
            
            result = {
                'test_name': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'effect_size': eta_squared,
                'effect_size_interpretation': self._interpret_eta_squared(eta_squared)
            }
            
            # Check assumptions
            # Levene's test for equal variances
            levene_stat, levene_p = stats.levene(*group_data)
            result['equal_variances'] = levene_p > 0.05
            result['levene_p'] = levene_p
            
        else:
            # Kruskal-Wallis test
            statistic, p_value = stats.kruskal(*group_data)
            
            # Epsilon-squared (effect size for Kruskal-Wallis)
            n = sum(len(group) for group in group_data)
            epsilon_squared = (statistic - len(groups) + 1) / (n - len(groups))
            
            result = {
                'test_name': 'Kruskal-Wallis test',
                'statistic': statistic,
                'p_value': p_value,
                'effect_size': epsilon_squared,
                'effect_size_interpretation': self._interpret_epsilon_squared(epsilon_squared)
            }
        
        # Add group statistics
        group_stats = {}
        for i, group in enumerate(groups):
            data = group_data[i]
            group_stats[str(group)] = {
                'n': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'median': data.median(),
                'iqr': data.quantile(0.75) - data.quantile(0.25)
            }
        
        result['group_statistics'] = group_stats
        result['n_groups'] = len(groups)
        
        return result
    
    def paired_tests(self,
                    before_var: str,
                    after_var: str,
                    test_type: str = 'auto') -> Dict[str, Any]:
        """
        Perform paired tests for before/after comparisons.
        
        Args:
            before_var: Variable with baseline measurements
            after_var: Variable with follow-up measurements
            test_type: 'auto', 'parametric', 'non_parametric'
            
        Returns:
            Test results with effect sizes
        """
        # Get paired data (both measurements available)
        paired_data = self.df[[before_var, after_var]].dropna()
        
        if len(paired_data) < 3:
            return {'error': 'Need at least 3 paired observations'}
        
        before_vals = paired_data[before_var]
        after_vals = paired_data[after_var]
        differences = after_vals - before_vals
        
        if test_type == 'auto':
            # Test normality of differences
            _, normality_p = stats.shapiro(differences)
            test_type = 'parametric' if normality_p > 0.05 else 'non_parametric'
        
        if test_type == 'parametric':
            # Paired t-test
            statistic, p_value = stats.ttest_rel(after_vals, before_vals)
            
            # Cohen's d for paired data
            cohens_d = differences.mean() / differences.std()
            
            result = {
                'test_name': 'Paired t-test',
                'statistic': statistic,
                'p_value': p_value,
                'effect_size': cohens_d,
                'effect_size_interpretation': self._interpret_cohens_d(cohens_d),
                'mean_difference': differences.mean(),
                'std_difference': differences.std(),
                'before_mean': before_vals.mean(),
                'after_mean': after_vals.mean()
            }
            
        else:
            # Wilcoxon signed-rank test
            statistic, p_value = stats.wilcoxon(after_vals, before_vals)
            
            # Effect size (r = Z / sqrt(N))
            n = len(paired_data)
            z_score = statistic / np.sqrt(n * (n + 1) * (2 * n + 1) / 6)
            effect_size_r = abs(z_score) / np.sqrt(n)
            
            result = {
                'test_name': 'Wilcoxon signed-rank test',
                'statistic': statistic,
                'p_value': p_value,
                'effect_size': effect_size_r,
                'effect_size_interpretation': self._interpret_r_effect_size(effect_size_r),
                'median_difference': differences.median(),
                'before_median': before_vals.median(),
                'after_median': after_vals.median()
            }
        
        result.update({
            'n_pairs': len(paired_data),
            'normality_p': stats.shapiro(differences)[1] if len(differences) <= 5000 else None
        })
        
        if self.privacy_guard:
            result = self.privacy_guard.sanitize_output(result)
            
        self.results['paired_test'] = result
        return result
    
    def proportion_tests(self,
                        success_counts: List[int],
                        sample_sizes: List[int],
                        group_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Test differences in proportions between groups.
        
        Args:
            success_counts: Number of successes in each group
            sample_sizes: Total sample size for each group
            group_names: Names of groups (optional)
            
        Returns:
            Proportion test results with effect sizes
        """
        if len(success_counts) != len(sample_sizes):
            return {'error': 'success_counts and sample_sizes must have same length'}
        
        if len(success_counts) < 2:
            return {'error': 'Need at least 2 groups for proportion test'}
        
        if group_names is None:
            group_names = [f'Group_{i+1}' for i in range(len(success_counts))]
        
        # Calculate proportions
        proportions = [count / size for count, size in zip(success_counts, sample_sizes)]
        
        # Two-proportion z-test
        if len(success_counts) == 2:
            z_stat, p_value = proportions_ztest(success_counts, sample_sizes)
            
            # Effect size (Cohen's h)
            p1, p2 = proportions[0], proportions[1]
            cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
            
            # Confidence intervals
            ci1 = proportion_confint(success_counts[0], sample_sizes[0])
            ci2 = proportion_confint(success_counts[1], sample_sizes[1])
            
            result = {
                'test_name': 'Two-proportion z-test',
                'z_statistic': z_stat,
                'p_value': p_value,
                'effect_size': cohens_h,
                'effect_size_interpretation': self._interpret_cohens_h(cohens_h),
                'group_results': {
                    group_names[0]: {
                        'successes': success_counts[0],
                        'total': sample_sizes[0],
                        'proportion': proportions[0],
                        'ci_lower': ci1[0],
                        'ci_upper': ci1[1]
                    },
                    group_names[1]: {
                        'successes': success_counts[1],
                        'total': sample_sizes[1],
                        'proportion': proportions[1],
                        'ci_lower': ci2[0],
                        'ci_upper': ci2[1]
                    }
                }
            }
        else:
            # Chi-square test for multiple proportions
            # Create contingency table
            failures = [size - count for size, count in zip(sample_sizes, success_counts)]
            contingency = np.array([success_counts, failures])
            
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            result = {
                'test_name': 'Chi-square test for proportions',
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'group_results': {}
            }
            
            for i, name in enumerate(group_names):
                result['group_results'][name] = {
                    'successes': success_counts[i],
                    'total': sample_sizes[i],
                    'proportion': proportions[i]
                }
        
        if self.privacy_guard:
            result = self.privacy_guard.sanitize_output(result)
            
        self.results['proportion_test'] = result
        return result
    
    def _determine_test_type(self, outcome: str, group_var: str) -> str:
        """Determine whether to use parametric or non-parametric test."""
        groups = self.df[group_var].dropna().unique()
        group_data = [self.df[self.df[group_var] == group][outcome].dropna() for group in groups]
        
        # Check normality for each group
        normality_results = []
        for data in group_data:
            if len(data) >= 3:
                # Use sample if data is large
                sample_data = data.sample(min(5000, len(data)), random_state=42)
                _, p_value = stats.shapiro(sample_data)
                normality_results.append(p_value > 0.05)
            else:
                normality_results.append(True)  # Assume normal for very small samples
        
        # Use parametric if all groups are approximately normal
        return 'parametric' if all(normality_results) else 'non_parametric'
    
    def _is_categorical(self, variable: str) -> bool:
        """Check if variable should be treated as categorical."""
        return (self.df[variable].dtype == 'object' or 
                self.df[variable].nunique() <= 10 or
                self.df[variable].dtype == 'bool')
    
    def _apply_multiple_comparison_correction(self,
                                            p_values: List[float],
                                            method: str,
                                            alpha: float) -> Dict[str, Any]:
        """Apply multiple comparison correction."""
        reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            p_values, alpha=alpha, method=method
        )
        
        return {
            'method': method,
            'alpha_original': alpha,
            'alpha_corrected': alpha_bonf if method == 'bonferroni' else alpha_sidak,
            'p_corrected': p_corrected.tolist(),
            'reject': reject.tolist(),
            'n_significant_original': sum(p < alpha for p in p_values),
            'n_significant_corrected': sum(reject)
        }
    
    # Effect size interpretation methods
    def _interpret_cohens_d(self, d: float) -> str:
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_cramers_v(self, v: float) -> str:
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "small"
        elif v < 0.5:
            return "medium"
        else:
            return "large"
    
    def _interpret_eta_squared(self, eta2: float) -> str:
        if eta2 < 0.01:
            return "negligible"
        elif eta2 < 0.06:
            return "small"
        elif eta2 < 0.14:
            return "medium"
        else:
            return "large"
    
    def _interpret_epsilon_squared(self, eps2: float) -> str:
        return self._interpret_eta_squared(eps2)  # Same interpretation
    
    def _interpret_rank_biserial(self, r: float) -> str:
        abs_r = abs(r)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "small"
        elif abs_r < 0.5:
            return "medium"
        else:
            return "large"
    
    def _interpret_cohens_h(self, h: float) -> str:
        abs_h = abs(h)
        if abs_h < 0.2:
            return "negligible"
        elif abs_h < 0.5:
            return "small"
        elif abs_h < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_r_effect_size(self, r: float) -> str:
        abs_r = abs(r)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "small"
        elif abs_r < 0.5:
            return "medium"
        else:
            return "large"