"""
Descriptive Statistics Analysis Plugin

Core plugin for generating comprehensive descriptive statistics including
Table 1 generation for clinical studies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from scipy import stats

from cablecar_research.analysis.base import BaseAnalysis, AnalysisMetadata, AnalysisType, ValidationLevel


class DescriptiveStatisticsPlugin(BaseAnalysis):
    """
    Comprehensive descriptive statistics for clinical research.
    
    Generates publication-ready tables following reporting standards
    (STROBE, CONSORT, etc.).
    """
    
    metadata = AnalysisMetadata(
        name="descriptive_statistics",
        display_name="Descriptive Statistics",
        description="Generate comprehensive descriptive statistics including Table 1 for clinical studies",
        version="1.0.0",
        author="CableCar Team",
        email="support@cablecar.ai",
        analysis_type=AnalysisType.DESCRIPTIVE,
        validation_level=ValidationLevel.STANDARD,
        citation="CableCar Research Team. Descriptive Statistics Plugin for Clinical Research. CableCar v1.0.0",
        keywords=["descriptive", "table1", "baseline", "characteristics", "statistics"]
    )
    
    def validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """Validate inputs for descriptive statistics."""
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
        
        # Check for required variables parameter
        variables = kwargs.get('variables')
        if not variables:
            validation['valid'] = False
            validation['errors'].append("'variables' parameter is required")
        elif not isinstance(variables, list):
            validation['valid'] = False
            validation['errors'].append("'variables' must be a list")
        else:
            # Check if variables exist in data
            missing_vars = [var for var in variables if var not in self.df.columns]
            if missing_vars:
                validation['valid'] = False
                validation['errors'].append(f"Variables not found in data: {missing_vars}")
        
        # Check stratification variable if provided
        stratify_by = kwargs.get('stratify_by')
        if stratify_by and stratify_by not in self.df.columns:
            validation['valid'] = False
            validation['errors'].append(f"Stratification variable '{stratify_by}' not found in data")
        
        # Warnings for small sample sizes
        if len(self.df) < 30:
            validation['warnings'].append("Small sample size (n<30) may affect statistical validity")
        
        # Suggestions
        if len(self.df) > 10000:
            validation['suggestions'].append("Large dataset detected - consider sampling for initial exploration")
        
        return validation
    
    def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """Execute descriptive statistics analysis."""
        variables = kwargs.get('variables', [])
        stratify_by = kwargs.get('stratify_by')
        categorical_vars = kwargs.get('categorical_vars', [])
        continuous_vars = kwargs.get('continuous_vars', [])
        non_normal_vars = kwargs.get('non_normal_vars', [])
        include_tests = kwargs.get('include_statistical_tests', False)
        
        results = {
            'analysis_type': 'descriptive_statistics',
            'variables_analyzed': variables,
            'sample_size': len(self.df),
            'stratified': stratify_by is not None,
            'stratification_variable': stratify_by,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Generate overall statistics
        overall_stats = self._compute_descriptive_stats(
            variables, categorical_vars, continuous_vars, non_normal_vars
        )
        results['overall'] = overall_stats
        
        # Generate stratified statistics if requested
        if stratify_by:
            stratified_stats = {}
            p_values = {}
            
            strata = self.df[stratify_by].unique()
            for stratum in strata:
                stratum_data = self.df[self.df[stratify_by] == stratum]
                stratum_analyzer = DescriptiveStatisticsPlugin(stratum_data, self.privacy_guard)
                stratum_stats = stratum_analyzer._compute_descriptive_stats(
                    variables, categorical_vars, continuous_vars, non_normal_vars
                )
                stratified_stats[f"{stratify_by}_{stratum}"] = stratum_stats
            
            # Compute p-values for group comparisons if requested
            if include_tests:
                for var in variables:
                    p_val = self._compute_group_comparison(var, stratify_by)
                    if p_val is not None:
                        p_values[var] = p_val
            
            results['stratified'] = stratified_stats
            if p_values:
                results['p_values'] = p_values
        
        # Apply privacy protection if available
        if self.privacy_guard:
            results = self.privacy_guard.sanitize_output(results, 'descriptive_analysis')
        
        return results
    
    def format_results(self, results: Dict[str, Any], format_type: str = "standard") -> str:
        """Format descriptive statistics results."""
        if format_type == "summary":
            return self._format_summary(results)
        elif format_type == "detailed":
            return self._format_detailed(results)
        elif format_type == "publication":
            return self._format_publication(results)
        else:
            return self._format_standard(results)
    
    def get_required_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter requirements for descriptive statistics."""
        return {
            'required': {
                'variables': {
                    'type': 'array',
                    'description': 'List of variables to analyze',
                    'example': ['age', 'sex', 'diagnosis']
                }
            },
            'optional': {
                'stratify_by': {
                    'type': 'string',
                    'description': 'Variable to stratify analysis by (e.g., treatment group)',
                    'example': 'treatment_group'
                },
                'categorical_vars': {
                    'type': 'array', 
                    'description': 'Variables to treat as categorical',
                    'example': ['sex', 'diagnosis']
                },
                'continuous_vars': {
                    'type': 'array',
                    'description': 'Variables to treat as continuous',
                    'example': ['age', 'weight']
                },
                'non_normal_vars': {
                    'type': 'array',
                    'description': 'Continuous variables that are non-normal (use median/IQR)',
                    'example': ['length_of_stay', 'biomarker_level']
                },
                'include_missing_analysis': {
                    'type': 'boolean',
                    'description': 'Include detailed missing data analysis',
                    'example': True
                },
                'include_statistical_tests': {
                    'type': 'boolean',
                    'description': 'Include statistical tests between groups',
                    'example': True
                },
                'generate_table1': {
                    'type': 'boolean',
                    'description': 'Generate publication-ready Table 1 format',
                    'example': True
                }
            }
        }
    
    def _compute_descriptive_stats(self, variables: List[str], 
                                 categorical_vars: List[str] = None,
                                 continuous_vars: List[str] = None,
                                 non_normal_vars: List[str] = None) -> Dict[str, Any]:
        """Compute descriptive statistics for variables."""
        stats = {}
        
        for var in variables:
            if var not in self.df.columns:
                continue
            
            var_data = self.df[var].dropna()
            var_stats = {'variable': var, 'n_valid': len(var_data)}
            
            # Determine variable type
            if categorical_vars and var in categorical_vars:
                var_type = 'categorical'
            elif continuous_vars and var in continuous_vars:
                var_type = 'continuous'
            else:
                # Auto-detect
                if pd.api.types.is_numeric_dtype(var_data):
                    var_type = 'continuous'
                else:
                    var_type = 'categorical'
            
            var_stats['type'] = var_type
            
            if var_type == 'categorical':
                # Categorical statistics
                value_counts = var_data.value_counts()
                percentages = var_data.value_counts(normalize=True) * 100
                
                categories = {}
                for value, count in value_counts.items():
                    pct = percentages[value]
                    categories[str(value)] = f"{count} ({pct:.1f}%)"
                
                var_stats['categories'] = categories
                var_stats['n_categories'] = len(value_counts)
                
            else:
                # Continuous statistics
                var_stats['mean'] = var_data.mean()
                var_stats['std'] = var_data.std()
                var_stats['median'] = var_data.median()
                var_stats['q25'] = var_data.quantile(0.25)
                var_stats['q75'] = var_data.quantile(0.75)
                var_stats['min'] = var_data.min()
                var_stats['max'] = var_data.max()
                
                # Check for normality (if sample size allows)
                if len(var_data) >= 8:  # Minimum for Shapiro-Wilk
                    try:
                        _, p_val = stats.shapiro(var_data.sample(min(5000, len(var_data))))
                        var_stats['normality_p'] = p_val
                        var_stats['appears_normal'] = p_val > 0.05
                    except:
                        var_stats['appears_normal'] = None
                
                # Determine summary format
                use_median = (non_normal_vars and var in non_normal_vars) or \
                           var_stats.get('appears_normal', True) == False
                
                if use_median:
                    var_stats['summary'] = f"{var_stats['median']:.1f} ({var_stats['q25']:.1f}, {var_stats['q75']:.1f})"
                    var_stats['summary_type'] = 'median_iqr'
                else:
                    var_stats['summary'] = f"{var_stats['mean']:.1f} ± {var_stats['std']:.1f}"
                    var_stats['summary_type'] = 'mean_std'
            
            # Missing data info
            n_missing = self.df[var].isna().sum()
            var_stats['n_missing'] = n_missing
            var_stats['percent_missing'] = (n_missing / len(self.df)) * 100
            
            stats[var] = var_stats
        
        return stats
    
    def _compute_group_comparison(self, variable: str, group_var: str) -> Optional[float]:
        """Compute p-value for group comparison."""
        try:
            groups = self.df[group_var].unique()
            if len(groups) < 2:
                return None
            
            var_data = self.df[variable].dropna()
            group_data = self.df[group_var].dropna()
            
            # Align data
            common_idx = var_data.index.intersection(group_data.index)
            var_aligned = var_data.loc[common_idx]
            group_aligned = group_data.loc[common_idx]
            
            if pd.api.types.is_numeric_dtype(var_aligned):
                # Continuous variable - use t-test or Mann-Whitney
                group_arrays = [var_aligned[group_aligned == group] for group in groups]
                
                if len(groups) == 2:
                    # Two groups - t-test
                    _, p_val = stats.ttest_ind(group_arrays[0], group_arrays[1])
                    return p_val
                else:
                    # Multiple groups - ANOVA
                    _, p_val = stats.f_oneway(*group_arrays)
                    return p_val
            else:
                # Categorical variable - Chi-square test
                contingency = pd.crosstab(var_aligned, group_aligned)
                _, p_val, _, _ = stats.chi2_contingency(contingency)
                return p_val
                
        except Exception:
            return None
    
    def _format_standard(self, results: Dict[str, Any]) -> str:
        """Standard formatting of results."""
        output = f"Descriptive Statistics Analysis\n{'='*50}\n\n"
        output += f"Sample Size: {results['sample_size']:,}\n"
        output += f"Variables Analyzed: {len(results['variables_analyzed'])}\n\n"
        
        # Overall statistics
        output += "Variable Summary:\n"
        for var, stats in results['overall'].items():
            output += f"\n{var.replace('_', ' ').title()}:\n"
            if stats['type'] == 'categorical':
                for category, freq in stats['categories'].items():
                    output += f"  {category}: {freq}\n"
            else:
                output += f"  {stats['summary']} ({stats['summary_type']})\n"
            
            if stats['n_missing'] > 0:
                output += f"  Missing: {stats['n_missing']} ({stats['percent_missing']:.1f}%)\n"
        
        # Stratified results
        if results.get('stratified'):
            output += f"\n\nStratified by {results['stratification_variable']}:\n"
            for stratum, stratum_stats in results['stratified'].items():
                group_name = stratum.replace(f"{results['stratification_variable']}_", "")
                output += f"\n{group_name}:\n"
                for var, stats in stratum_stats.items():
                    if stats['type'] == 'categorical':
                        main_category = max(stats['categories'].items(), key=lambda x: int(x[1].split('(')[0]))
                        output += f"  {var}: {main_category[1]}\n"
                    else:
                        output += f"  {var}: {stats['summary']}\n"
        
        # P-values
        if results.get('p_values'):
            output += "\n\nGroup Comparisons (p-values):\n"
            for var, p_val in results['p_values'].items():
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                output += f"  {var}: p = {p_val:.4f} {significance}\n"
        
        return output
    
    def _format_summary(self, results: Dict[str, Any]) -> str:
        """Summary formatting of results."""
        output = f"Descriptive Statistics Summary\n"
        output += f"Sample: n={results['sample_size']:,}, Variables: {len(results['variables_analyzed'])}\n"
        
        if results.get('p_values'):
            significant = sum(1 for p in results['p_values'].values() if p < 0.05)
            output += f"Significant differences: {significant}/{len(results['p_values'])} variables\n"
        
        return output
    
    def _format_detailed(self, results: Dict[str, Any]) -> str:
        """Detailed formatting with all statistics."""
        return self._format_standard(results)  # For now, same as standard
    
    def _format_publication(self, results: Dict[str, Any]) -> str:
        """Publication-ready Table 1 format."""
        output = "Table 1. Baseline Characteristics\n"
        output += "="*50 + "\n\n"
        
        # Handle both overall sample and stratified results
        if results.get('stratified'):
            # Stratified table format
            stratify_var = results.get('stratification_variable', 'Group')
            strata_names = [k.replace(f"{stratify_var}_", "") 
                           for k in results['stratified'].keys()]
            
            # Header
            output += f"Characteristic\tOverall (N={results['sample_size']:,})"
            for name in strata_names:
                # Get sample size for this stratum
                stratum_key = f"{stratify_var}_{name}"
                if stratum_key in results['stratified']:
                    stratum_size = len([v for v in results['stratified'][stratum_key].values() 
                                      if isinstance(v, dict) and 'n_valid' in v])
                    output += f"\t{name}"
                else:
                    output += f"\t{name}"
            
            if results.get('p_values'):
                output += "\tp-value"
            
            output += "\n" + "-"*80 + "\n"
            
            # Variables
            for var in results['variables_analyzed']:
                if var in results['overall']:
                    stats = results['overall'][var]
                    var_display = var.replace('_', ' ').title()
                    
                    if stats['type'] == 'categorical':
                        output += f"{var_display}\n"
                        for category, freq in stats['categories'].items():
                            output += f"  {category}\t{freq}"
                            
                            # Add stratified data
                            for stratum_name, stratum_data in results['stratified'].items():
                                if var in stratum_data:
                                    cat_data = stratum_data[var]['categories'].get(category, "0 (0.0%)")
                                    output += f"\t{cat_data}"
                                else:
                                    output += "\t-"
                            
                            # Add p-value (only on first category line)
                            if category == list(stats['categories'].keys())[0]:
                                if results.get('p_values') and var in results['p_values']:
                                    p_val = results['p_values'][var]
                                    significance = "*" if p_val < 0.05 else ""
                                    output += f"\t{p_val:.3f}{significance}"
                                elif results.get('p_values'):
                                    output += "\t-"
                            
                            output += "\n"
                    else:
                        # Continuous variable
                        output += f"{var_display}\t{stats['summary']}"
                        
                        # Add stratified data
                        for stratum_name, stratum_data in results['stratified'].items():
                            if var in stratum_data:
                                output += f"\t{stratum_data[var]['summary']}"
                            else:
                                output += "\t-"
                        
                        # Add p-value
                        if results.get('p_values') and var in results['p_values']:
                            p_val = results['p_values'][var]
                            significance = "*" if p_val < 0.05 else ""
                            output += f"\t{p_val:.3f}{significance}"
                        elif results.get('p_values'):
                            output += "\t-"
                        
                        output += "\n"
        else:
            # Simple overall table
            output += f"Characteristic\tOverall (N={results['sample_size']:,})\n"
            output += "-"*40 + "\n"
            
            for var in results['variables_analyzed']:
                if var in results['overall']:
                    stats = results['overall'][var]
                    var_display = var.replace('_', ' ').title()
                    
                    if stats['type'] == 'categorical':
                        output += f"{var_display}\n"
                        for category, freq in stats['categories'].items():
                            output += f"  {category}\t{freq}\n"
                    else:
                        output += f"{var_display}\t{stats['summary']}\n"
        
        # Add footnotes
        output += "\n" + "-"*40 + "\n"
        output += "Continuous variables: mean ± SD or median (IQR)\n"
        output += "Categorical variables: n (%)\n"
        if results.get('p_values'):
            output += "* p < 0.05\n"
        
        return output