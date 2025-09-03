"""
Descriptive Statistics Analysis

Generates comprehensive descriptive statistics including:
- Table 1 generation for clinical studies
- Stratified analyses
- Missing data summaries
- Distribution assessments
- Effect size calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DescriptiveAnalysis:
    """
    Comprehensive descriptive statistics for clinical research.
    
    Designed to generate publication-ready tables following
    reporting standards (STROBE, CONSORT, etc.).
    """
    
    def __init__(self, df: pd.DataFrame, privacy_guard=None):
        self.df = df.copy()
        self.privacy_guard = privacy_guard
        self.results = {}
        
    def generate_table1(self, 
                       variables: List[str],
                       stratify_by: Optional[str] = None,
                       categorical_vars: Optional[List[str]] = None,
                       continuous_vars: Optional[List[str]] = None,
                       non_normal_vars: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate Table 1 (baseline characteristics table).
        
        Args:
            variables: List of variables to include
            stratify_by: Variable to stratify by (e.g., treatment group)
            categorical_vars: Variables to treat as categorical
            continuous_vars: Variables to treat as continuous
            non_normal_vars: Continuous variables that are non-normal (use median/IQR)
            
        Returns:
            Dictionary with table data and metadata
        """
        if categorical_vars is None:
            categorical_vars = self._detect_categorical(variables)
        if continuous_vars is None:
            continuous_vars = self._detect_continuous(variables)
        if non_normal_vars is None:
            non_normal_vars = self._detect_non_normal(continuous_vars)
            
        table_data = {}
        
        # Overall column
        table_data['Overall'] = self._calculate_column_stats(
            self.df, variables, categorical_vars, continuous_vars, non_normal_vars
        )
        table_data['Overall']['n'] = len(self.df)
        
        # Stratified columns
        if stratify_by:
            unique_groups = self.df[stratify_by].dropna().unique()
            
            for group in unique_groups:
                group_df = self.df[self.df[stratify_by] == group]
                table_data[f"{stratify_by}_{group}"] = self._calculate_column_stats(
                    group_df, variables, categorical_vars, continuous_vars, non_normal_vars
                )
                table_data[f"{stratify_by}_{group}"]['n'] = len(group_df)
        
        # Add statistical tests if stratified
        if stratify_by:
            table_data['p_values'] = self._calculate_group_comparisons(
                variables, stratify_by, categorical_vars, continuous_vars, non_normal_vars
            )
        
        # Apply privacy protection
        if self.privacy_guard:
            table_data = self.privacy_guard.sanitize_output(table_data)
        
        self.results['table1'] = table_data
        return table_data
    
    def _calculate_column_stats(self, 
                               df: pd.DataFrame, 
                               variables: List[str],
                               categorical_vars: List[str],
                               continuous_vars: List[str],
                               non_normal_vars: List[str]) -> Dict[str, Any]:
        """Calculate statistics for a single column (group)."""
        stats_dict = {}
        
        for var in variables:
            if var not in df.columns:
                continue
                
            var_data = df[var].dropna()
            n_missing = df[var].isna().sum()
            
            if var in categorical_vars:
                # Categorical variable
                value_counts = var_data.value_counts()
                percentages = (value_counts / len(var_data) * 100).round(1)
                
                stats_dict[var] = {
                    'type': 'categorical',
                    'missing': n_missing,
                    'categories': {}
                }
                
                for category in value_counts.index:
                    count = value_counts[category]
                    pct = percentages[category]
                    stats_dict[var]['categories'][category] = f"{count} ({pct}%)"
                    
            elif var in continuous_vars:
                # Continuous variable
                if var in non_normal_vars:
                    # Use median and IQR for non-normal
                    median = var_data.median()
                    q1 = var_data.quantile(0.25)
                    q3 = var_data.quantile(0.75)
                    
                    stats_dict[var] = {
                        'type': 'continuous_non_normal',
                        'missing': n_missing,
                        'median': median,
                        'q1': q1,
                        'q3': q3,
                        'summary': f"{median:.1f} ({q1:.1f}-{q3:.1f})"
                    }
                else:
                    # Use mean and SD for normal
                    mean = var_data.mean()
                    std = var_data.std()
                    
                    stats_dict[var] = {
                        'type': 'continuous_normal',
                        'missing': n_missing,
                        'mean': mean,
                        'std': std,
                        'summary': f"{mean:.1f} ± {std:.1f}"
                    }
        
        return stats_dict
    
    def _calculate_group_comparisons(self,
                                   variables: List[str],
                                   stratify_by: str,
                                   categorical_vars: List[str],
                                   continuous_vars: List[str],
                                   non_normal_vars: List[str]) -> Dict[str, float]:
        """Calculate statistical tests between groups."""
        p_values = {}
        
        groups = self.df[stratify_by].dropna().unique()
        if len(groups) != 2:
            return {}  # Only handle 2-group comparisons for now
        
        group1_data = self.df[self.df[stratify_by] == groups[0]]
        group2_data = self.df[self.df[stratify_by] == groups[1]]
        
        for var in variables:
            if var not in self.df.columns or var == stratify_by:
                continue
                
            try:
                if var in categorical_vars:
                    # Chi-square test for categorical variables
                    contingency_table = pd.crosstab(
                        self.df[var], 
                        self.df[stratify_by], 
                        dropna=False
                    )
                    
                    if contingency_table.size > 0:
                        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
                        p_values[var] = p_val
                        
                elif var in continuous_vars:
                    group1_vals = group1_data[var].dropna()
                    group2_vals = group2_data[var].dropna()
                    
                    if len(group1_vals) == 0 or len(group2_vals) == 0:
                        continue
                    
                    if var in non_normal_vars:
                        # Mann-Whitney U test for non-normal
                        statistic, p_val = stats.mannwhitneyu(
                            group1_vals, group2_vals, alternative='two-sided'
                        )
                        p_values[var] = p_val
                    else:
                        # t-test for normal
                        statistic, p_val = stats.ttest_ind(group1_vals, group2_vals)
                        p_values[var] = p_val
                        
            except Exception as e:
                # Log error but continue with other variables
                continue
        
        return p_values
    
    def _detect_categorical(self, variables: List[str]) -> List[str]:
        """Auto-detect categorical variables."""
        categorical = []
        
        for var in variables:
            if var in self.df.columns:
                if (self.df[var].dtype == 'object' or 
                    self.df[var].nunique() <= 10 or
                    self.df[var].dtype == 'bool'):
                    categorical.append(var)
                    
        return categorical
    
    def _detect_continuous(self, variables: List[str]) -> List[str]:
        """Auto-detect continuous variables."""
        continuous = []
        
        for var in variables:
            if var in self.df.columns:
                if (self.df[var].dtype in ['int64', 'float64'] and 
                    self.df[var].nunique() > 10):
                    continuous.append(var)
                    
        return continuous
    
    def _detect_non_normal(self, continuous_vars: List[str]) -> List[str]:
        """Detect non-normal continuous variables using Shapiro-Wilk test."""
        non_normal = []
        
        for var in continuous_vars:
            if var in self.df.columns:
                data = self.df[var].dropna()
                
                if len(data) < 3:
                    continue
                    
                # Use sample if data is too large
                if len(data) > 5000:
                    data = data.sample(5000, random_state=42)
                
                try:
                    # Shapiro-Wilk test (p < 0.05 indicates non-normal)
                    statistic, p_value = stats.shapiro(data)
                    if p_value < 0.05:
                        non_normal.append(var)
                except:
                    # If test fails, assume normal
                    continue
                    
        return non_normal
    
    def missing_data_analysis(self, variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive missing data analysis.
        
        Args:
            variables: Variables to analyze. If None, analyzes all variables.
            
        Returns:
            Dictionary with missing data patterns and statistics
        """
        if variables is None:
            variables = list(self.df.columns)
        
        # Missing data summary
        missing_summary = {}
        for var in variables:
            if var in self.df.columns:
                n_missing = self.df[var].isna().sum()
                pct_missing = (n_missing / len(self.df)) * 100
                
                missing_summary[var] = {
                    'n_missing': n_missing,
                    'percent_missing': pct_missing,
                    'complete_cases': len(self.df) - n_missing
                }
        
        # Missing data patterns
        missing_patterns = self.df[variables].isna().value_counts().head(10)
        
        # Complete case analysis
        complete_cases = self.df[variables].dropna()
        
        result = {
            'summary': missing_summary,
            'patterns': missing_patterns.to_dict(),
            'complete_cases_n': len(complete_cases),
            'complete_cases_percent': (len(complete_cases) / len(self.df)) * 100,
            'recommendations': self._missing_data_recommendations(missing_summary)
        }
        
        if self.privacy_guard:
            result = self.privacy_guard.sanitize_output(result)
            
        self.results['missing_data'] = result
        return result
    
    def _missing_data_recommendations(self, missing_summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations for handling missing data."""
        recommendations = []
        
        high_missing_vars = [var for var, stats in missing_summary.items() 
                           if stats['percent_missing'] > 20]
        
        if high_missing_vars:
            recommendations.append(
                f"Variables with >20% missing data: {', '.join(high_missing_vars)}. "
                "Consider multiple imputation or sensitivity analyses."
            )
        
        moderate_missing_vars = [var for var, stats in missing_summary.items() 
                               if 5 < stats['percent_missing'] <= 20]
        
        if moderate_missing_vars:
            recommendations.append(
                f"Variables with 5-20% missing data: {', '.join(moderate_missing_vars)}. "
                "Consider imputation methods."
            )
        
        if not high_missing_vars and not moderate_missing_vars:
            recommendations.append("Missing data levels are low (<5%). Complete case analysis may be appropriate.")
        
        return recommendations
    
    def correlation_analysis(self, 
                           variables: List[str],
                           method: str = 'pearson') -> Dict[str, Any]:
        """
        Calculate correlation matrix and identify strong correlations.
        
        Args:
            variables: Variables to include in correlation analysis
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation results and multicollinearity warnings
        """
        # Filter to numeric variables
        numeric_vars = [var for var in variables 
                       if var in self.df.columns and 
                       self.df[var].dtype in ['int64', 'float64']]
        
        if len(numeric_vars) < 2:
            return {'error': 'Need at least 2 numeric variables for correlation analysis'}
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_vars].corr(method=method)
        
        # Find strong correlations (|r| > 0.7)
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j] 
                correlation = corr_matrix.iloc[i, j]
                
                if abs(correlation) > 0.7:
                    strong_correlations.append({
                        'variable1': var1,
                        'variable2': var2,
                        'correlation': correlation
                    })
        
        # Calculate VIF for multicollinearity detection
        vif_values = self._calculate_vif(self.df[numeric_vars])
        
        result = {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'vif_values': vif_values,
            'multicollinearity_warnings': [
                var for var, vif in vif_values.items() if vif > 5
            ]
        }
        
        if self.privacy_guard:
            result = self.privacy_guard.sanitize_output(result)
            
        self.results['correlation'] = result
        return result
    
    def _calculate_vif(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Variance Inflation Factor for variables."""
        from sklearn.linear_model import LinearRegression
        
        vif_dict = {}
        df_clean = df.dropna()
        
        if len(df_clean) < 10 or len(df_clean.columns) < 2:
            return {}
        
        for i, col in enumerate(df_clean.columns):
            try:
                X = df_clean.drop(columns=[col])
                y = df_clean[col]
                
                reg = LinearRegression().fit(X, y)
                r_squared = reg.score(X, y)
                
                vif = 1 / (1 - r_squared) if r_squared < 0.99 else float('inf')
                vif_dict[col] = vif
                
            except:
                vif_dict[col] = None
                
        return vif_dict
    
    def effect_size_analysis(self,
                           outcome_var: str,
                           group_var: str,
                           continuous_outcome: bool = None) -> Dict[str, Any]:
        """
        Calculate effect sizes for group comparisons.
        
        Args:
            outcome_var: The outcome variable
            group_var: The grouping variable
            continuous_outcome: Whether outcome is continuous. Auto-detected if None.
            
        Returns:
            Effect size statistics with confidence intervals
        """
        if continuous_outcome is None:
            continuous_outcome = (self.df[outcome_var].dtype in ['int64', 'float64'] and 
                                self.df[outcome_var].nunique() > 10)
        
        groups = self.df[group_var].dropna().unique()
        if len(groups) != 2:
            return {'error': 'Effect size calculation currently supports only 2 groups'}
        
        group1_data = self.df[self.df[group_var] == groups[0]][outcome_var].dropna()
        group2_data = self.df[self.df[group_var] == groups[1]][outcome_var].dropna()
        
        result = {
            'group1': groups[0],
            'group2': groups[1],
            'group1_n': len(group1_data),
            'group2_n': len(group2_data)
        }
        
        if continuous_outcome:
            # Cohen's d for continuous outcomes
            pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                                (len(group2_data) - 1) * group2_data.var()) / 
                               (len(group1_data) + len(group2_data) - 2))
            
            cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
            
            result.update({
                'effect_size_type': 'cohens_d',
                'effect_size': cohens_d,
                'interpretation': self._interpret_cohens_d(cohens_d),
                'group1_mean': group1_data.mean(),
                'group2_mean': group2_data.mean(),
                'pooled_std': pooled_std
            })
        else:
            # Odds ratio for binary outcomes
            contingency = pd.crosstab(self.df[outcome_var], self.df[group_var])
            
            if contingency.shape == (2, 2):
                a, b = contingency.iloc[0, 0], contingency.iloc[0, 1]
                c, d = contingency.iloc[1, 0], contingency.iloc[1, 1]
                
                # Calculate odds ratio
                odds_ratio = (a * d) / (b * c) if b * c > 0 else float('inf')
                
                # Calculate 95% CI for log OR
                log_or = np.log(odds_ratio)
                se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
                ci_lower = np.exp(log_or - 1.96 * se_log_or)
                ci_upper = np.exp(log_or + 1.96 * se_log_or)
                
                result.update({
                    'effect_size_type': 'odds_ratio',
                    'odds_ratio': odds_ratio,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'contingency_table': contingency.to_dict()
                })
        
        if self.privacy_guard:
            result = self.privacy_guard.sanitize_output(result)
            
        self.results['effect_size'] = result
        return result
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"