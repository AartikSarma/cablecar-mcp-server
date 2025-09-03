"""
Exploratory Data Analysis Plugin

Comprehensive exploratory data analysis for clinical research:
- Data quality assessment
- Distribution analysis
- Correlation analysis
- Missing data patterns
- Outlier detection
- Variable relationships
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import json

from cablecar_research.analysis.base import BaseAnalysis, AnalysisMetadata, AnalysisType

warnings.filterwarnings('ignore')


class ExploratoryDataAnalysisPlugin(BaseAnalysis):
    """
    Comprehensive exploratory data analysis plugin for clinical research.
    
    Provides systematic exploration of clinical datasets:
    - Data quality and completeness assessment
    - Univariate and bivariate analysis
    - Distribution analysis and normality testing
    - Missing data patterns and mechanisms
    - Outlier detection and assessment
    - Variable relationships and correlations
    """
    
    metadata = AnalysisMetadata(
        name="exploratory_data_analysis",
        display_name="Exploratory Data Analysis",
        description="Comprehensive data exploration and quality assessment",
        analysis_type=AnalysisType.EXPLORATORY,
        required_columns=[],
        optional_columns=["all_columns"],
        parameters={
            "target_variable": {
                "type": "string",
                "description": "Primary outcome or target variable for focused analysis",
                "required": False
            },
            "categorical_variables": {
                "type": "list",
                "description": "List of categorical variables to analyze",
                "required": False,
                "default": []
            },
            "continuous_variables": {
                "type": "list",
                "description": "List of continuous variables to analyze",
                "required": False,
                "default": []
            },
            "include_correlations": {
                "type": "boolean",
                "description": "Include correlation analysis",
                "required": False,
                "default": True
            },
            "include_distributions": {
                "type": "boolean",
                "description": "Include distribution analysis",
                "required": False,
                "default": True
            },
            "include_missing_patterns": {
                "type": "boolean",
                "description": "Include missing data pattern analysis",
                "required": False,
                "default": True
            },
            "include_outlier_detection": {
                "type": "boolean",
                "description": "Include outlier detection",
                "required": False,
                "default": True
            },
            "outlier_method": {
                "type": "string",
                "description": "Method for outlier detection",
                "required": False,
                "default": "iqr",
                "choices": ["iqr", "z_score", "isolation_forest"]
            },
            "correlation_method": {
                "type": "string",
                "description": "Correlation method",
                "required": False,
                "default": "pearson",
                "choices": ["pearson", "spearman", "kendall"]
            },
            "significance_level": {
                "type": "float",
                "description": "Significance level for statistical tests",
                "required": False,
                "default": 0.05
            }
        }
    )
    
    def __init__(self, privacy_guard=None):
        super().__init__(privacy_guard)
        
    def validate_inputs(self, df: pd.DataFrame, **kwargs) -> List[str]:
        """Validate inputs for exploratory data analysis."""
        errors = []
        
        if df.empty:
            errors.append("DataFrame cannot be empty")
            
        if len(df) < 10:
            errors.append("Dataset too small for meaningful exploratory analysis (minimum 10 rows)")
            
        target_variable = kwargs.get('target_variable')
        if target_variable and target_variable not in df.columns:
            errors.append(f"Target variable '{target_variable}' not found in data")
            
        categorical_variables = kwargs.get('categorical_variables', [])
        for var in categorical_variables:
            if var not in df.columns:
                errors.append(f"Categorical variable '{var}' not found in data")
                
        continuous_variables = kwargs.get('continuous_variables', [])
        for var in continuous_variables:
            if var not in df.columns:
                errors.append(f"Continuous variable '{var}' not found in data")
                
        return errors
    
    def run_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Run comprehensive exploratory data analysis."""
        
        target_variable = kwargs.get('target_variable')
        categorical_variables = kwargs.get('categorical_variables', [])
        continuous_variables = kwargs.get('continuous_variables', [])
        include_correlations = kwargs.get('include_correlations', True)
        include_distributions = kwargs.get('include_distributions', True)
        include_missing_patterns = kwargs.get('include_missing_patterns', True)
        include_outlier_detection = kwargs.get('include_outlier_detection', True)
        outlier_method = kwargs.get('outlier_method', 'iqr')
        correlation_method = kwargs.get('correlation_method', 'pearson')
        significance_level = kwargs.get('significance_level', 0.05)
        
        results = {
            'summary': {
                'n_observations': len(df),
                'n_variables': len(df.columns),
                'data_quality_score': 0.0,
                'completeness_rate': 0.0
            },
            'data_quality': {},
            'variable_profiles': {},
            'missing_data_analysis': {},
            'correlation_analysis': {},
            'distribution_analysis': {},
            'outlier_analysis': {},
            'target_variable_analysis': {},
            'recommendations': []
        }
        
        # Automatically classify variables if not provided
        if not categorical_variables and not continuous_variables:
            categorical_variables, continuous_variables = self._classify_variables(df)
        
        # Data quality assessment
        quality_results = self._assess_data_quality(df)
        results['data_quality'] = quality_results
        results['summary']['completeness_rate'] = quality_results['overall_completeness']
        results['summary']['data_quality_score'] = quality_results['quality_score']
        
        # Variable profiling
        profile_results = self._profile_variables(df, categorical_variables, continuous_variables)
        results['variable_profiles'] = profile_results
        
        # Missing data analysis
        if include_missing_patterns:
            missing_results = self._analyze_missing_patterns(df)
            results['missing_data_analysis'] = missing_results
        
        # Distribution analysis
        if include_distributions:
            distribution_results = self._analyze_distributions(df, continuous_variables, significance_level)
            results['distribution_analysis'] = distribution_results
        
        # Correlation analysis
        if include_correlations and len(continuous_variables) > 1:
            correlation_results = self._analyze_correlations(df, continuous_variables, correlation_method)
            results['correlation_analysis'] = correlation_results
        
        # Outlier analysis
        if include_outlier_detection:
            outlier_results = self._detect_outliers(df, continuous_variables, outlier_method)
            results['outlier_analysis'] = outlier_results
        
        # Target variable analysis
        if target_variable:
            target_results = self._analyze_target_variable(
                df, target_variable, categorical_variables, continuous_variables
            )
            results['target_variable_analysis'] = target_results
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        results['recommendations'] = recommendations
        
        return results
    
    def _classify_variables(self, df: pd.DataFrame) -> tuple:
        """Automatically classify variables as categorical or continuous."""
        categorical_vars = []
        continuous_vars = []
        
        for col in df.columns:
            if df[col].dtype in ['object', 'category', 'bool']:
                categorical_vars.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                # Check if it looks categorical (few unique values)
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05 or df[col].nunique() <= 10:
                    categorical_vars.append(col)
                else:
                    continuous_vars.append(col)
            else:
                # Default to categorical for unknown types
                categorical_vars.append(col)
        
        return categorical_vars, continuous_vars
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        
        results = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'total_cells': len(df) * len(df.columns),
            'missing_cells': df.isnull().sum().sum(),
            'overall_completeness': 0.0,
            'quality_score': 0.0,
            'column_completeness': {},
            'data_types': {},
            'issues': []
        }
        
        # Calculate completeness
        if results['total_cells'] > 0:
            results['overall_completeness'] = 1 - (results['missing_cells'] / results['total_cells'])
        
        # Column-level completeness
        for col in df.columns:
            completeness = 1 - (df[col].isnull().sum() / len(df))
            results['column_completeness'][col] = completeness
            results['data_types'][col] = str(df[col].dtype)
        
        # Identify quality issues
        if results['overall_completeness'] < 0.9:
            results['issues'].append("High missing data rate (>10%)")
        
        # Check for duplicate rows
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            results['issues'].append(f"{n_duplicates} duplicate rows found")
            results['n_duplicates'] = n_duplicates
        
        # Check for columns with single value
        constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_columns:
            results['issues'].append(f"Constant columns: {constant_columns}")
            results['constant_columns'] = constant_columns
        
        # Calculate quality score (0-1 scale)
        quality_score = results['overall_completeness']
        if n_duplicates == 0:
            quality_score += 0.1
        if not constant_columns:
            quality_score += 0.1
        
        results['quality_score'] = min(quality_score, 1.0)
        
        return results
    
    def _profile_variables(self, df: pd.DataFrame, categorical_vars: List[str], 
                         continuous_vars: List[str]) -> Dict[str, Any]:
        """Create detailed variable profiles."""
        
        results = {
            'categorical_profiles': {},
            'continuous_profiles': {},
            'summary_stats': {
                'n_categorical': len(categorical_vars),
                'n_continuous': len(continuous_vars)
            }
        }
        
        # Profile categorical variables
        for var in categorical_vars:
            if var not in df.columns:
                continue
                
            profile = {
                'data_type': str(df[var].dtype),
                'n_unique': df[var].nunique(),
                'n_missing': df[var].isnull().sum(),
                'missing_rate': df[var].isnull().sum() / len(df),
                'value_counts': {},
                'most_frequent': None,
                'least_frequent': None
            }
            
            # Value counts (limit to top 10 for privacy)
            value_counts = df[var].value_counts().head(10)
            for val, count in value_counts.items():
                if count >= 10:  # Privacy threshold
                    profile['value_counts'][str(val)] = int(count)
            
            if len(value_counts) > 0:
                profile['most_frequent'] = str(value_counts.index[0])
                profile['least_frequent'] = str(value_counts.index[-1])
            
            results['categorical_profiles'][var] = profile
        
        # Profile continuous variables
        for var in continuous_vars:
            if var not in df.columns:
                continue
                
            series = df[var].dropna()
            if len(series) == 0:
                continue
                
            profile = {
                'data_type': str(df[var].dtype),
                'n_missing': df[var].isnull().sum(),
                'missing_rate': df[var].isnull().sum() / len(df),
                'count': len(series),
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'q1': float(series.quantile(0.25)),
                'median': float(series.median()),
                'q3': float(series.quantile(0.75)),
                'max': float(series.max()),
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis()),
                'cv': float(series.std() / series.mean()) if series.mean() != 0 else 0
            }
            
            results['continuous_profiles'][var] = profile
        
        return results
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        
        results = {
            'missing_summary': {},
            'missing_patterns': {},
            'missing_mechanism_assessment': {},
            'recommendations': []
        }
        
        # Column-wise missing summary
        for col in df.columns:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                results['missing_summary'][col] = {
                    'n_missing': int(n_missing),
                    'percent_missing': (n_missing / len(df)) * 100
                }
        
        # Missing patterns (limit to avoid privacy issues)
        if results['missing_summary']:
            missing_df = df.isnull()
            
            # Count missing patterns
            pattern_counts = missing_df.value_counts().head(5)  # Top 5 patterns only
            
            for i, (pattern, count) in enumerate(pattern_counts.items()):
                if count >= 10:  # Privacy threshold
                    pattern_name = f"pattern_{i+1}"
                    results['missing_patterns'][pattern_name] = {
                        'count': int(count),
                        'percent': (count / len(df)) * 100,
                        'description': f"Missing pattern with {sum(pattern)} variables missing"
                    }
        
        # Assess missing mechanisms (simplified)
        high_missing_vars = [
            col for col, info in results['missing_summary'].items() 
            if info['percent_missing'] > 20
        ]
        
        if high_missing_vars:
            results['missing_mechanism_assessment']['high_missing_variables'] = high_missing_vars
            results['recommendations'].append("Consider missing data mechanisms for high-missing variables")
        
        return results
    
    def _analyze_distributions(self, df: pd.DataFrame, continuous_vars: List[str], 
                             significance_level: float) -> Dict[str, Any]:
        """Analyze distributions of continuous variables."""
        
        results = {
            'normality_tests': {},
            'distribution_summary': {},
            'recommendations': []
        }
        
        for var in continuous_vars:
            if var not in df.columns:
                continue
                
            series = df[var].dropna()
            if len(series) < 8:  # Minimum for Shapiro-Wilk
                continue
            
            dist_results = {
                'variable': var,
                'n_observations': len(series),
                'distribution_type': 'unknown'
            }
            
            # Normality tests
            try:
                # Shapiro-Wilk test (for smaller samples)
                if len(series) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(series)
                    dist_results['shapiro_wilk'] = {
                        'statistic': float(shapiro_stat),
                        'p_value': float(shapiro_p),
                        'is_normal': shapiro_p > significance_level
                    }
                
                # Kolmogorov-Smirnov test against normal distribution
                ks_stat, ks_p = stats.kstest(series, 'norm')
                dist_results['kolmogorov_smirnov'] = {
                    'statistic': float(ks_stat),
                    'p_value': float(ks_p),
                    'is_normal': ks_p > significance_level
                }
                
                # Determine likely distribution type
                if dist_results.get('shapiro_wilk', {}).get('is_normal', False) or \
                   dist_results['kolmogorov_smirnov']['is_normal']:
                    dist_results['distribution_type'] = 'normal'
                elif series.skew() > 2:
                    dist_results['distribution_type'] = 'right_skewed'
                elif series.skew() < -2:
                    dist_results['distribution_type'] = 'left_skewed'
                else:
                    dist_results['distribution_type'] = 'approximately_normal'
                
            except Exception as e:
                dist_results['error'] = str(e)
            
            results['normality_tests'][var] = dist_results
        
        # Generate recommendations
        non_normal_vars = [
            var for var, test in results['normality_tests'].items()
            if test.get('distribution_type') not in ['normal', 'approximately_normal']
        ]
        
        if non_normal_vars:
            results['recommendations'].append(
                f"Consider non-parametric tests for non-normal variables: {non_normal_vars[:3]}"
            )
        
        return results
    
    def _analyze_correlations(self, df: pd.DataFrame, continuous_vars: List[str], 
                            method: str) -> Dict[str, Any]:
        """Analyze correlations between continuous variables."""
        
        results = {
            'correlation_matrix': {},
            'strong_correlations': [],
            'method': method,
            'recommendations': []
        }
        
        # Filter to available variables
        available_vars = [var for var in continuous_vars if var in df.columns]
        
        if len(available_vars) < 2:
            return results
        
        # Calculate correlation matrix
        numeric_df = df[available_vars].select_dtypes(include=[np.number])
        
        if not numeric_df.empty:
            try:
                if method == 'pearson':
                    corr_matrix = numeric_df.corr(method='pearson')
                elif method == 'spearman':
                    corr_matrix = numeric_df.corr(method='spearman')
                elif method == 'kendall':
                    corr_matrix = numeric_df.corr(method='kendall')
                else:
                    corr_matrix = numeric_df.corr(method='pearson')
                
                # Convert to dict (only upper triangle to avoid redundancy)
                for i, var1 in enumerate(corr_matrix.columns):
                    for j, var2 in enumerate(corr_matrix.columns):
                        if i < j:  # Upper triangle only
                            corr_value = corr_matrix.loc[var1, var2]
                            if pd.notna(corr_value):
                                pair_key = f"{var1}_vs_{var2}"
                                results['correlation_matrix'][pair_key] = {
                                    'variables': [var1, var2],
                                    'correlation': float(corr_value),
                                    'strength': self._assess_correlation_strength(corr_value)
                                }
                                
                                # Identify strong correlations
                                if abs(corr_value) >= 0.7:
                                    results['strong_correlations'].append({
                                        'variables': [var1, var2],
                                        'correlation': float(corr_value),
                                        'strength': self._assess_correlation_strength(corr_value)
                                    })
                
                # Generate recommendations
                if len(results['strong_correlations']) > 0:
                    results['recommendations'].append(
                        f"Strong correlations detected - consider multicollinearity in modeling"
                    )
                
            except Exception as e:
                results['error'] = str(e)
        
        return results
    
    def _assess_correlation_strength(self, corr_value: float) -> str:
        """Assess correlation strength."""
        abs_corr = abs(corr_value)
        if abs_corr >= 0.9:
            return "very_strong"
        elif abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.5:
            return "moderate"
        elif abs_corr >= 0.3:
            return "weak"
        else:
            return "very_weak"
    
    def _detect_outliers(self, df: pd.DataFrame, continuous_vars: List[str], 
                        method: str) -> Dict[str, Any]:
        """Detect outliers in continuous variables."""
        
        results = {
            'method': method,
            'outlier_summary': {},
            'total_outliers': 0,
            'recommendations': []
        }
        
        for var in continuous_vars:
            if var not in df.columns:
                continue
                
            series = df[var].dropna()
            if len(series) < 10:
                continue
            
            outliers = pd.Series([False] * len(series), index=series.index)
            
            try:
                if method == 'iqr':
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))
                    
                elif method == 'z_score':
                    z_scores = np.abs(stats.zscore(series))
                    outliers = z_scores > 3
                    
                elif method == 'isolation_forest':
                    from sklearn.ensemble import IsolationForest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outliers_pred = iso_forest.fit_predict(series.values.reshape(-1, 1))
                    outliers = pd.Series(outliers_pred == -1, index=series.index)
                
                n_outliers = outliers.sum()
                outlier_percent = (n_outliers / len(series)) * 100
                
                results['outlier_summary'][var] = {
                    'n_outliers': int(n_outliers),
                    'percent_outliers': float(outlier_percent),
                    'outlier_rate_category': self._categorize_outlier_rate(outlier_percent)
                }
                
                results['total_outliers'] += n_outliers
                
            except Exception as e:
                results['outlier_summary'][var] = {'error': str(e)}
        
        # Generate recommendations
        high_outlier_vars = [
            var for var, info in results['outlier_summary'].items()
            if info.get('percent_outliers', 0) > 5
        ]
        
        if high_outlier_vars:
            results['recommendations'].append(
                f"High outlier rates detected in: {high_outlier_vars[:3]}"
            )
        
        return results
    
    def _categorize_outlier_rate(self, percent: float) -> str:
        """Categorize outlier rate."""
        if percent == 0:
            return "none"
        elif percent <= 2:
            return "low"
        elif percent <= 5:
            return "moderate"
        else:
            return "high"
    
    def _analyze_target_variable(self, df: pd.DataFrame, target_var: str, 
                               categorical_vars: List[str], continuous_vars: List[str]) -> Dict[str, Any]:
        """Analyze target variable relationships."""
        
        results = {
            'target_variable': target_var,
            'target_type': 'unknown',
            'target_profile': {},
            'relationships': {},
            'recommendations': []
        }
        
        if target_var not in df.columns:
            return results
        
        # Determine target type
        if target_var in categorical_vars:
            results['target_type'] = 'categorical'
            target_series = df[target_var]
            results['target_profile'] = {
                'n_categories': target_series.nunique(),
                'value_counts': dict(target_series.value_counts().head(10))
            }
            
        elif target_var in continuous_vars:
            results['target_type'] = 'continuous'
            target_series = df[target_var].dropna()
            results['target_profile'] = {
                'mean': float(target_series.mean()),
                'std': float(target_series.std()),
                'min': float(target_series.min()),
                'max': float(target_series.max()),
                'median': float(target_series.median())
            }
        
        # Analyze relationships with other variables (sample up to 5)
        other_continuous = [var for var in continuous_vars[:5] if var != target_var and var in df.columns]
        other_categorical = [var for var in categorical_vars[:5] if var != target_var and var in df.columns]
        
        # Target vs continuous variables
        if results['target_type'] == 'continuous':
            for var in other_continuous:
                try:
                    corr = df[target_var].corr(df[var])
                    if pd.notna(corr):
                        results['relationships'][f'{target_var}_vs_{var}'] = {
                            'type': 'continuous_continuous',
                            'correlation': float(corr),
                            'strength': self._assess_correlation_strength(corr)
                        }
                except:
                    pass
        
        # Generate recommendations
        if results['target_type'] == 'categorical' and results['target_profile'].get('n_categories', 0) > 10:
            results['recommendations'].append("Consider grouping categories for target variable")
        
        return results
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate analysis recommendations based on EDA results."""
        
        recommendations = []
        
        # Data quality recommendations
        quality_score = results['summary']['data_quality_score']
        if quality_score < 0.8:
            recommendations.append("Address data quality issues before analysis")
        
        completeness_rate = results['summary']['completeness_rate']
        if completeness_rate < 0.9:
            recommendations.append("Consider missing data imputation strategies")
        
        # Variable recommendations
        n_continuous = results['variable_profiles']['summary_stats']['n_continuous']
        n_categorical = results['variable_profiles']['summary_stats']['n_categorical']
        
        if n_continuous > 20:
            recommendations.append("Consider dimensionality reduction for high-dimensional data")
        
        if n_categorical > 15:
            recommendations.append("Consider feature selection for categorical variables")
        
        # Distribution recommendations
        if 'distribution_analysis' in results:
            non_normal_count = sum(
                1 for test in results['distribution_analysis']['normality_tests'].values()
                if test.get('distribution_type') not in ['normal', 'approximately_normal']
            )
            if non_normal_count > 0:
                recommendations.append("Consider transformations or non-parametric methods")
        
        # Correlation recommendations
        if 'correlation_analysis' in results and results['correlation_analysis'].get('strong_correlations'):
            recommendations.append("Address multicollinearity before modeling")
        
        # Outlier recommendations
        if 'outlier_analysis' in results and results['outlier_analysis']['total_outliers'] > 0:
            recommendations.append("Investigate and potentially treat outliers")
        
        return recommendations
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """Format EDA results for display."""
        
        summary = results['summary']
        
        output = []
        output.append("=== EXPLORATORY DATA ANALYSIS RESULTS ===\n")
        
        # Data overview
        output.append(f"📊 Dataset Overview:")
        output.append(f"   • Observations: {summary['n_observations']:,}")
        output.append(f"   • Variables: {summary['n_variables']}")
        output.append(f"   • Data completeness: {summary['completeness_rate']:.1%}")
        output.append(f"   • Quality score: {summary['data_quality_score']:.2f}/1.0")
        output.append("")
        
        # Variable profiles summary
        if 'variable_profiles' in results:
            profiles = results['variable_profiles']['summary_stats']
            output.append(f"📈 Variable Types:")
            output.append(f"   • Continuous: {profiles['n_continuous']}")
            output.append(f"   • Categorical: {profiles['n_categorical']}")
            output.append("")
        
        # Data quality issues
        if results.get('data_quality', {}).get('issues'):
            output.append(f"⚠️  Data Quality Issues:")
            for issue in results['data_quality']['issues']:
                output.append(f"   • {issue}")
            output.append("")
        
        # Missing data summary
        if 'missing_data_analysis' in results:
            missing_summary = results['missing_data_analysis'].get('missing_summary', {})
            if missing_summary:
                output.append(f"🔍 Missing Data (Top Variables):")
                sorted_missing = sorted(missing_summary.items(), 
                                      key=lambda x: x[1]['percent_missing'], reverse=True)
                for var, info in sorted_missing[:5]:
                    output.append(f"   • {var}: {info['percent_missing']:.1f}% missing")
                output.append("")
        
        # Distribution analysis
        if 'distribution_analysis' in results:
            normality_tests = results['distribution_analysis'].get('normality_tests', {})
            non_normal = [var for var, test in normality_tests.items() 
                         if test.get('distribution_type') not in ['normal', 'approximately_normal']]
            if non_normal:
                output.append(f"📉 Non-Normal Distributions:")
                for var in non_normal[:5]:
                    dist_type = normality_tests[var].get('distribution_type', 'unknown')
                    output.append(f"   • {var}: {dist_type}")
                output.append("")
        
        # Correlation analysis
        if 'correlation_analysis' in results:
            strong_corr = results['correlation_analysis'].get('strong_correlations', [])
            if strong_corr:
                output.append(f"🔗 Strong Correlations:")
                for corr in strong_corr[:5]:
                    vars_str = ' vs '.join(corr['variables'])
                    output.append(f"   • {vars_str}: {corr['correlation']:.3f} ({corr['strength']})")
                output.append("")
        
        # Outlier analysis
        if 'outlier_analysis' in results:
            outlier_summary = results['outlier_analysis'].get('outlier_summary', {})
            high_outliers = [(var, info) for var, info in outlier_summary.items() 
                           if info.get('percent_outliers', 0) > 2]
            if high_outliers:
                output.append(f"🎯 Variables with Outliers:")
                for var, info in high_outliers[:5]:
                    output.append(f"   • {var}: {info['percent_outliers']:.1f}% outliers")
                output.append("")
        
        # Target variable analysis
        if 'target_variable_analysis' in results:
            target_info = results['target_variable_analysis']
            if target_info.get('target_variable'):
                output.append(f"🎯 Target Variable Analysis:")
                output.append(f"   • Variable: {target_info['target_variable']}")
                output.append(f"   • Type: {target_info['target_type']}")
                
                relationships = target_info.get('relationships', {})
                if relationships:
                    output.append(f"   • Key relationships: {len(relationships)}")
                output.append("")
        
        # Recommendations
        if results.get('recommendations'):
            output.append("💡 Key Recommendations:")
            for rec in results['recommendations']:
                output.append(f"   • {rec}")
        
        return "\n".join(output)
    
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters."""
        return []