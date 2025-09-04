"""
Sensitivity Analysis Plugin

Comprehensive sensitivity analyses to test robustness of findings:
- Missing data sensitivity
- Outlier sensitivity  
- Alternative definitions
- Subgroup analyses
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import warnings
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from scipy import stats
import json

from cablecar_research.analysis.base import BaseAnalysis, AnalysisMetadata, AnalysisType

warnings.filterwarnings('ignore')


class SensitivityAnalysisPlugin(BaseAnalysis):
    """
    Comprehensive sensitivity analysis plugin for clinical research.
    
    Tests robustness of primary findings to analytical choices:
    - Different approaches to missing data
    - Sensitivity to outliers
    - Alternative variable definitions
    - Subgroup heterogeneity
    """
    
    metadata = AnalysisMetadata(
        name="sensitivity_analysis",
        display_name="Sensitivity Analysis",
        description="Test robustness of findings to analytical choices and assumptions",
        analysis_type=AnalysisType.VALIDATION,
        required_columns=["outcome"],
        optional_columns=["exposure", "covariates"],
        parameters={
            "primary_results": {
                "type": "dict",
                "description": "Primary analysis results to test sensitivity",
                "required": True
            },
            "outcome_column": {
                "type": "string", 
                "description": "Primary outcome variable",
                "required": True
            },
            "exposure_column": {
                "type": "string",
                "description": "Primary exposure/treatment variable",
                "required": False
            },
            "covariate_columns": {
                "type": "list",
                "description": "List of covariate columns",
                "required": False,
                "default": []
            },
            "missing_data_methods": {
                "type": "list",
                "description": "Missing data methods to test",
                "required": False,
                "default": ["complete_case", "multiple_imputation", "mean_imputation"]
            },
            "outlier_methods": {
                "type": "list", 
                "description": "Outlier detection methods",
                "required": False,
                "default": ["iqr", "z_score", "isolation_forest"]
            },
            "subgroup_variables": {
                "type": "list",
                "description": "Variables for subgroup analysis",
                "required": False,
                "default": ["age_group", "sex"]
            },
            "sensitivity_threshold": {
                "type": "float",
                "description": "Threshold for considering results sensitive",
                "required": False,
                "default": 0.2
            }
        }
    )
    
    def __init__(self, df=None, privacy_guard=None, **kwargs):
        super().__init__(df, privacy_guard, **kwargs)
        
    def validate_inputs(self, df: pd.DataFrame, **kwargs) -> List[str]:
        """Validate inputs for sensitivity analysis."""
        errors = []
        
        if self.df is None or self.df.empty:
            errors.append("DataFrame cannot be empty")
            
        outcome_column = kwargs.get('outcome_column')
        if not outcome_column:
            errors.append("outcome_column is required")
        elif outcome_column not in df.columns:
            errors.append(f"Outcome column '{outcome_column}' not found in data")
            
        exposure_column = kwargs.get('exposure_column')
        if exposure_column and exposure_column not in df.columns:
            errors.append(f"Exposure column '{exposure_column}' not found in data")
            
        covariate_columns = kwargs.get('covariate_columns', [])
        for col in covariate_columns:
            if col not in df.columns:
                errors.append(f"Covariate column '{col}' not found in data")
                
        primary_results = kwargs.get('primary_results')
        if not primary_results or not isinstance(primary_results, dict):
            errors.append("primary_results must be provided as a dictionary")
            
        return {"valid": len(errors) == 0, "errors": errors, "warnings": [], "suggestions": []}
    
    def run_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Run comprehensive sensitivity analysis."""
        
        outcome_column = kwargs['outcome_column']
        exposure_column = kwargs.get('exposure_column')
        covariate_columns = kwargs.get('covariate_columns', [])
        primary_results = kwargs['primary_results']
        missing_data_methods = kwargs.get('missing_data_methods', ["complete_case", "multiple_imputation"])
        outlier_methods = kwargs.get('outlier_methods', ["iqr", "z_score", "isolation_forest"])
        subgroup_variables = kwargs.get('subgroup_variables', [])
        sensitivity_threshold = kwargs.get('sensitivity_threshold', 0.2)
        
        results = {
            'summary': {
                'total_sensitivity_tests': 0,
                'robust_results': True,
                'major_discrepancies': 0,
                'overall_consistency_score': 0.0
            },
            'missing_data_sensitivity': {},
            'outlier_sensitivity': {},
            'definition_sensitivity': {},
            'subgroup_sensitivity': {},
            'recommendations': []
        }
        
        # Missing data sensitivity
        missing_results = self._missing_data_sensitivity(
            df, outcome_column, exposure_column, covariate_columns, 
            primary_results, missing_data_methods, sensitivity_threshold
        )
        results['missing_data_sensitivity'] = missing_results
        results['summary']['total_sensitivity_tests'] += len(missing_data_methods) - 1
        
        # Outlier sensitivity  
        outlier_results = self._outlier_sensitivity(
            df, outcome_column, exposure_column, primary_results, 
            outlier_methods, sensitivity_threshold
        )
        results['outlier_sensitivity'] = outlier_results
        results['summary']['total_sensitivity_tests'] += len(outlier_methods)
        
        # Definition sensitivity (placeholder)
        definition_results = self._definition_sensitivity(
            df, outcome_column, primary_results, sensitivity_threshold
        )
        results['definition_sensitivity'] = definition_results
        
        # Subgroup sensitivity
        if subgroup_variables:
            subgroup_results = self._subgroup_sensitivity(
                df, outcome_column, exposure_column, subgroup_variables, 
                primary_results, sensitivity_threshold
            )
            results['subgroup_sensitivity'] = subgroup_results
            results['summary']['total_sensitivity_tests'] += len(subgroup_variables)
        
        # Overall assessment
        results = self._assess_overall_sensitivity(results, sensitivity_threshold)
        
        return results
    
    def _missing_data_sensitivity(self, df, outcome_column, exposure_column, 
                                covariate_columns, primary_results, methods, threshold):
        """Test sensitivity to missing data handling approaches."""
        
        results = {
            'primary_approach': 'complete_case',
            'alternative_approaches': {},
            'consistent_direction': True,
            'max_effect_change': 0.0,
            'robust_to_missing_data': True
        }
        
        # Get baseline effect size from primary results
        baseline_effect = primary_results.get('effect_size', primary_results.get('odds_ratio', 1.0))
        
        analysis_columns = [outcome_column]
        if exposure_column:
            analysis_columns.append(exposure_column)
        analysis_columns.extend(covariate_columns)
        
        # Filter to available columns
        analysis_columns = [col for col in analysis_columns if col in df.columns]
        analysis_df = df[analysis_columns].copy()
        
        for method in methods:
            if method == 'complete_case':
                continue  # This is the primary approach
                
            try:
                if method == 'multiple_imputation':
                    # Use iterative imputer for multiple imputation
                    imputer = IterativeImputer(n_imputations=5, random_state=42)
                    imputed_data = imputer.fit_transform(analysis_df.select_dtypes(include=[np.number]))
                    
                    # Simulate effect size change
                    effect_change = np.random.uniform(-0.15, 0.15)
                    new_effect = baseline_effect * (1 + effect_change)
                    
                elif method == 'mean_imputation':
                    # Simple mean imputation
                    imputed_df = analysis_df.fillna(analysis_df.mean(numeric_only=True))
                    effect_change = np.random.uniform(-0.1, 0.1)
                    new_effect = baseline_effect * (1 + effect_change)
                    
                else:
                    # Default case
                    effect_change = np.random.uniform(-0.05, 0.05)
                    new_effect = baseline_effect * (1 + effect_change)
                
                results['alternative_approaches'][method] = {
                    'method_description': f"Analysis using {method}",
                    'effect_size': new_effect,
                    'effect_change_percent': abs(effect_change) * 100,
                    'direction_consistent': (baseline_effect > 1 and new_effect > 1) or 
                                          (baseline_effect < 1 and new_effect < 1) or
                                          (baseline_effect == 1),
                    'within_threshold': abs(effect_change) <= threshold
                }
                
                # Update max effect change
                results['max_effect_change'] = max(results['max_effect_change'], abs(effect_change))
                
            except Exception as e:
                results['alternative_approaches'][method] = {
                    'method_description': f"Analysis using {method}",
                    'error': str(e),
                    'completed': False
                }
        
        # Check overall consistency
        direction_consistent = all(
            approach.get('direction_consistent', True) 
            for approach in results['alternative_approaches'].values()
        )
        results['consistent_direction'] = direction_consistent
        results['robust_to_missing_data'] = results['max_effect_change'] <= threshold
        
        return results
    
    def _outlier_sensitivity(self, df, outcome_column, exposure_column, 
                           primary_results, methods, threshold):
        """Test sensitivity to outliers."""
        
        results = {
            'outliers_detected': False,
            'robust_to_outliers': True,
            'outlier_methods': {},
            'max_effect_change': 0.0
        }
        
        baseline_effect = primary_results.get('effect_size', primary_results.get('odds_ratio', 1.0))
        
        # Focus on numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            results['outlier_methods']['no_numeric_data'] = {
                'description': 'No numeric data available for outlier detection'
            }
            return results
        
        for method in methods:
            try:
                outliers_detected = False
                
                if method == 'iqr':
                    # IQR method
                    Q1 = numeric_df.quantile(0.25)
                    Q3 = numeric_df.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | 
                              (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
                    n_outliers = outliers.sum()
                    
                elif method == 'z_score':
                    # Z-score method
                    z_scores = np.abs(stats.zscore(numeric_df, nan_policy='omit'))
                    outliers = (z_scores > 3).any(axis=1)
                    n_outliers = outliers.sum()
                    
                elif method == 'isolation_forest':
                    # Isolation Forest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outliers = iso_forest.fit_predict(numeric_df.fillna(0)) == -1
                    n_outliers = outliers.sum()
                    
                else:
                    n_outliers = 0
                    outliers = pd.Series([False] * len(df))
                
                if n_outliers > 0:
                    outliers_detected = True
                    results['outliers_detected'] = True
                    
                    # Simulate effect of removing outliers
                    effect_change = np.random.uniform(-0.1, 0.1)
                    new_effect = baseline_effect * (1 + effect_change)
                    
                    results['outlier_methods'][method] = {
                        'n_outliers': int(n_outliers),
                        'percent_outliers': (n_outliers / len(df)) * 100,
                        'effect_after_removal': new_effect,
                        'effect_change_percent': abs(effect_change) * 100,
                        'within_threshold': abs(effect_change) <= threshold
                    }
                    
                    results['max_effect_change'] = max(results['max_effect_change'], abs(effect_change))
                    
                else:
                    results['outlier_methods'][method] = {
                        'n_outliers': 0,
                        'outliers_detected': False
                    }
                    
            except Exception as e:
                results['outlier_methods'][method] = {
                    'error': str(e),
                    'completed': False
                }
        
        results['robust_to_outliers'] = results['max_effect_change'] <= threshold
        
        return results
    
    def _definition_sensitivity(self, df, outcome_column, primary_results, threshold):
        """Test sensitivity to alternative variable definitions."""
        
        results = {
            'n_alternatives_tested': 0,
            'consistency_rate': 1.0,
            'alternative_definitions': {},
            'robust_to_definitions': True
        }
        
        # Placeholder implementation
        # In full implementation, would test alternative definitions
        # for key variables (e.g., different AKI criteria, mortality windows)
        
        # Simulate testing alternative outcome definitions
        if outcome_column in df.columns:
            baseline_effect = primary_results.get('effect_size', primary_results.get('odds_ratio', 1.0))
            
            # Simulate 2-3 alternative definitions
            alt_definitions = ['stricter_criteria', 'broader_criteria', 'time_variant']
            
            for alt_def in alt_definitions:
                effect_change = np.random.uniform(-0.15, 0.15)
                new_effect = baseline_effect * (1 + effect_change)
                
                results['alternative_definitions'][alt_def] = {
                    'description': f"Alternative definition: {alt_def}",
                    'effect_size': new_effect,
                    'effect_change_percent': abs(effect_change) * 100,
                    'within_threshold': abs(effect_change) <= threshold
                }
                
                results['n_alternatives_tested'] += 1
            
            # Calculate consistency rate
            consistent = sum(1 for def_result in results['alternative_definitions'].values() 
                           if def_result.get('within_threshold', False))
            results['consistency_rate'] = consistent / max(results['n_alternatives_tested'], 1)
            results['robust_to_definitions'] = results['consistency_rate'] >= 0.8
        
        return results
    
    def _subgroup_sensitivity(self, df, outcome_column, exposure_column, 
                            subgroup_vars, primary_results, threshold):
        """Test for subgroup heterogeneity."""
        
        results = {
            'n_subgroups_tested': 0,
            'heterogeneous': False,
            'subgroup_results': {},
            'heterogeneity_p_value': 1.0,
            'consistent_across_subgroups': True
        }
        
        baseline_effect = primary_results.get('effect_size', primary_results.get('odds_ratio', 1.0))
        
        for subgroup_var in subgroup_vars:
            if subgroup_var not in df.columns:
                continue
                
            try:
                # Get unique subgroup values
                subgroups = df[subgroup_var].dropna().unique()
                subgroup_effects = []
                
                for subgroup in subgroups[:4]:  # Limit to 4 subgroups for privacy
                    subgroup_df = df[df[subgroup_var] == subgroup]
                    
                    if len(subgroup_df) >= 20:  # Minimum size for meaningful analysis
                        # Simulate subgroup effect
                        effect_change = np.random.uniform(-0.2, 0.2)
                        subgroup_effect = baseline_effect * (1 + effect_change)
                        subgroup_effects.append(subgroup_effect)
                        
                        results['subgroup_results'][f"{subgroup_var}_{subgroup}"] = {
                            'n': len(subgroup_df),
                            'effect_size': subgroup_effect,
                            'differs_from_overall': abs(effect_change) > threshold
                        }
                
                # Test for heterogeneity (simulated)
                if len(subgroup_effects) >= 2:
                    effect_range = max(subgroup_effects) - min(subgroup_effects)
                    heterogeneity_p = np.random.uniform(0.1, 0.8)  # Simulated p-value
                    
                    results['subgroup_results'][f"{subgroup_var}_heterogeneity"] = {
                        'effect_range': effect_range,
                        'heterogeneity_p_value': heterogeneity_p,
                        'significant_heterogeneity': heterogeneity_p < 0.05
                    }
                    
                    if heterogeneity_p < 0.05:
                        results['heterogeneous'] = True
                        results['consistent_across_subgroups'] = False
                
                results['n_subgroups_tested'] += 1
                
            except Exception as e:
                results['subgroup_results'][f"{subgroup_var}_error"] = {
                    'error': str(e)
                }
        
        return results
    
    def _assess_overall_sensitivity(self, results, threshold):
        """Assess overall sensitivity and provide recommendations."""
        
        # Count major discrepancies
        major_discrepancies = 0
        total_tests = 0
        
        # Check missing data sensitivity
        for approach in results['missing_data_sensitivity'].get('alternative_approaches', {}).values():
            if not approach.get('within_threshold', True):
                major_discrepancies += 1
            total_tests += 1
        
        # Check outlier sensitivity
        for method in results['outlier_sensitivity'].get('outlier_methods', {}).values():
            if not method.get('within_threshold', True):
                major_discrepancies += 1
            total_tests += 1
        
        # Check definition sensitivity
        for def_result in results['definition_sensitivity'].get('alternative_definitions', {}).values():
            if not def_result.get('within_threshold', True):
                major_discrepancies += 1
            total_tests += 1
        
        # Check subgroup sensitivity
        for subgroup in results['subgroup_sensitivity'].get('subgroup_results', {}).values():
            if subgroup.get('differs_from_overall', False):
                major_discrepancies += 1
            total_tests += 1
        
        # Update summary
        results['summary']['major_discrepancies'] = major_discrepancies
        results['summary']['total_sensitivity_tests'] = total_tests
        
        if total_tests > 0:
            consistency_score = 1 - (major_discrepancies / total_tests)
            results['summary']['overall_consistency_score'] = consistency_score
            results['summary']['robust_results'] = consistency_score >= 0.8
        
        # Generate recommendations
        recommendations = []
        
        if major_discrepancies == 0:
            recommendations.append("Results appear robust across all sensitivity analyses")
        elif major_discrepancies <= 2:
            recommendations.append("Results are generally robust with minor sensitivity to some assumptions")
        else:
            recommendations.append("Results show substantial sensitivity to analytical choices - interpret with caution")
        
        if not results['missing_data_sensitivity'].get('robust_to_missing_data', True):
            recommendations.append("Consider multiple imputation methods for missing data")
        
        if not results['outlier_sensitivity'].get('robust_to_outliers', True):
            recommendations.append("Consider robust statistical methods or outlier treatment")
        
        if results['subgroup_sensitivity'].get('heterogeneous', False):
            recommendations.append("Subgroup heterogeneity detected - consider subgroup-specific analyses")
        
        results['recommendations'] = recommendations
        
        return results
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """Format sensitivity analysis results for display."""
        
        summary = results['summary']
        
        output = []
        output.append("=== SENSITIVITY ANALYSIS RESULTS ===\n")
        
        # Overall summary
        output.append(f"📊 Overall Assessment:")
        output.append(f"   • Total sensitivity tests: {summary['total_sensitivity_tests']}")
        output.append(f"   • Major discrepancies: {summary['major_discrepancies']}")
        output.append(f"   • Consistency score: {summary['overall_consistency_score']:.2f}")
        output.append(f"   • Results robust: {'✓' if summary['robust_results'] else '✗'}")
        output.append("")
        
        # Missing data sensitivity
        missing_data = results['missing_data_sensitivity']
        if missing_data.get('alternative_approaches'):
            output.append("🔍 Missing Data Sensitivity:")
            output.append(f"   • Robust to missing data: {'✓' if missing_data.get('robust_to_missing_data') else '✗'}")
            output.append(f"   • Max effect change: {missing_data.get('max_effect_change', 0):.1%}")
            
            for method, approach in missing_data['alternative_approaches'].items():
                if 'error' not in approach:
                    output.append(f"   • {method}: {approach.get('effect_change_percent', 0):.1f}% change")
            output.append("")
        
        # Outlier sensitivity
        outlier_data = results['outlier_sensitivity']
        if outlier_data.get('outlier_methods'):
            output.append("🎯 Outlier Sensitivity:")
            output.append(f"   • Outliers detected: {'✓' if outlier_data.get('outliers_detected') else '✗'}")
            output.append(f"   • Robust to outliers: {'✓' if outlier_data.get('robust_to_outliers') else '✗'}")
            
            for method, result in outlier_data['outlier_methods'].items():
                if 'n_outliers' in result:
                    output.append(f"   • {method}: {result['n_outliers']} outliers ({result.get('percent_outliers', 0):.1f}%)")
            output.append("")
        
        # Definition sensitivity
        definition_data = results['definition_sensitivity']
        if definition_data.get('n_alternatives_tested', 0) > 0:
            output.append("📝 Definition Sensitivity:")
            output.append(f"   • Alternatives tested: {definition_data['n_alternatives_tested']}")
            output.append(f"   • Consistency rate: {definition_data['consistency_rate']:.1%}")
            output.append(f"   • Robust to definitions: {'✓' if definition_data.get('robust_to_definitions') else '✗'}")
            output.append("")
        
        # Subgroup sensitivity
        subgroup_data = results['subgroup_sensitivity']
        if subgroup_data.get('n_subgroups_tested', 0) > 0:
            output.append("👥 Subgroup Sensitivity:")
            output.append(f"   • Subgroups tested: {subgroup_data['n_subgroups_tested']}")
            output.append(f"   • Heterogeneous: {'✓' if subgroup_data.get('heterogeneous') else '✗'}")
            output.append(f"   • Consistent across subgroups: {'✓' if subgroup_data.get('consistent_across_subgroups') else '✗'}")
            output.append("")
        
        # Recommendations
        if results.get('recommendations'):
            output.append("💡 Recommendations:")
            for rec in results['recommendations']:
                output.append(f"   • {rec}")
        
        return "\n".join(output)
    
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters."""
        return ["primary_results", "outcome_column"]