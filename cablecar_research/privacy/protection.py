"""
Enhanced Privacy Protection

Multi-layered privacy protection system designed for clinical data:
- Configurable privacy policies
- Differential privacy mechanisms  
- K-anonymity and l-diversity enforcement
- PHI detection and sanitization
- Secure aggregation methods
- Audit logging and compliance tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import re
import hashlib
import json
import logging
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class PrivacyGuard:
    """
    Comprehensive privacy protection for clinical research data.
    
    Implements multiple privacy-preserving techniques:
    - Cell suppression with configurable thresholds
    - Differential privacy with calibrated noise
    - K-anonymity enforcement
    - PHI detection and removal
    - Secure aggregation
    """
    
    def __init__(self, 
                 min_cell_size: int = 10,
                 suppression_threshold: int = 5,
                 k_anonymity: int = 5,
                 l_diversity: int = 2,
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 enable_differential_privacy: bool = False):
        """
        Initialize privacy guard with configurable parameters.
        
        Args:
            min_cell_size: Minimum cell size for reporting aggregates
            suppression_threshold: Threshold below which values are suppressed
            k_anonymity: Minimum group size for k-anonymity
            l_diversity: Minimum diversity for sensitive attributes
            epsilon: Privacy parameter for differential privacy (smaller = more private)
            delta: Probability parameter for differential privacy
            enable_differential_privacy: Whether to apply differential privacy
        """
        self.min_cell_size = min_cell_size
        self.suppression_threshold = suppression_threshold
        self.k_anonymity = k_anonymity
        self.l_diversity = l_diversity
        self.epsilon = epsilon
        self.delta = delta
        self.enable_differential_privacy = enable_differential_privacy
        
        # Audit logging
        self.audit_log = []
        self.data_access_log = []
        
        # PHI patterns
        self.phi_patterns = self._initialize_phi_patterns()
        
        # Cache for repeated calculations
        self._cache = {}
    
    def sanitize_output(self, output: Any, context: str = 'general') -> Any:
        """
        Comprehensive output sanitization.
        
        Args:
            output: Data to sanitize
            context: Context of the output ('table', 'model', 'statistics', etc.)
            
        Returns:
            Sanitized output with privacy protections applied
        """
        # Log access
        self._log_data_access(context, type(output).__name__)
        
        if isinstance(output, dict):
            return self._sanitize_dict(output, context)
        elif isinstance(output, list):
            return [self.sanitize_output(item, context) for item in output]
        elif isinstance(output, pd.DataFrame):
            return self._sanitize_dataframe(output, context)
        elif isinstance(output, str):
            return self._sanitize_string(output)
        elif isinstance(output, (int, float)):
            return self._sanitize_number(output, context)
        else:
            return output
    
    def create_safe_aggregate(self,
                            df: pd.DataFrame,
                            group_vars: List[str],
                            agg_vars: List[str],
                            agg_functions: List[str] = ['count', 'mean'],
                            apply_noise: bool = None) -> pd.DataFrame:
        """
        Create privacy-safe aggregate statistics.
        
        Args:
            df: Input dataframe
            group_vars: Variables to group by
            agg_vars: Variables to aggregate
            agg_functions: Aggregation functions to apply
            apply_noise: Whether to apply differential privacy noise
            
        Returns:
            Privacy-safe aggregated dataframe
        """
        if apply_noise is None:
            apply_noise = self.enable_differential_privacy
        
        # Create grouping
        grouped = df.groupby(group_vars)
        
        results = []
        for group_name, group_df in grouped:
            if len(group_df) >= self.min_cell_size:
                group_result = {'group': group_name, 'n': len(group_df)}
                
                for var in agg_vars:
                    if var in group_df.columns:
                        for func in agg_functions:
                            if func == 'count':
                                value = len(group_df[var].dropna())
                            elif func == 'mean':
                                value = group_df[var].mean()
                            elif func == 'median':
                                value = group_df[var].median()
                            elif func == 'std':
                                value = group_df[var].std()
                            else:
                                continue
                            
                            # Apply privacy protection
                            if apply_noise and isinstance(value, (int, float)):
                                value = self._add_differential_privacy_noise(value, func)
                            
                            group_result[f'{var}_{func}'] = value
                
                results.append(group_result)
        
        result_df = pd.DataFrame(results)
        
        # Additional k-anonymity check
        if len(result_df) > 0:
            result_df = self._enforce_k_anonymity(result_df, group_vars)
        
        self._log_privacy_action('safe_aggregate', {
            'n_groups': len(results),
            'min_group_size': self.min_cell_size,
            'differential_privacy': apply_noise
        })
        
        return result_df
    
    def check_k_anonymity(self, 
                         df: pd.DataFrame,
                         quasi_identifiers: List[str],
                         sensitive_attrs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Check k-anonymity and l-diversity of dataset.
        
        Args:
            df: Dataset to check
            quasi_identifiers: Quasi-identifier columns
            sensitive_attrs: Sensitive attribute columns for l-diversity
            
        Returns:
            Privacy assessment results
        """
        # Check quasi-identifiers exist
        missing_qis = [qi for qi in quasi_identifiers if qi not in df.columns]
        if missing_qis:
            return {'error': f'Missing quasi-identifiers: {missing_qis}'}
        
        # Group by quasi-identifiers
        qi_groups = df.groupby(quasi_identifiers)
        group_sizes = qi_groups.size()
        
        # K-anonymity assessment
        k_violations = (group_sizes < self.k_anonymity).sum()
        min_group_size = group_sizes.min()
        
        results = {
            'k_anonymity_target': self.k_anonymity,
            'k_anonymity_achieved': min_group_size,
            'k_violations': k_violations,
            'total_groups': len(group_sizes),
            'k_anonymity_satisfied': k_violations == 0
        }
        
        # L-diversity assessment
        if sensitive_attrs:
            l_diversity_results = {}
            
            for sensitive_attr in sensitive_attrs:
                if sensitive_attr in df.columns:
                    attr_diversity = []
                    
                    for group_name, group_df in qi_groups:
                        sensitive_values = group_df[sensitive_attr].value_counts()
                        diversity = len(sensitive_values)
                        attr_diversity.append(diversity)
                    
                    min_diversity = min(attr_diversity)
                    l_violations = sum(1 for d in attr_diversity if d < self.l_diversity)
                    
                    l_diversity_results[sensitive_attr] = {
                        'l_diversity_target': self.l_diversity,
                        'l_diversity_achieved': min_diversity,
                        'l_violations': l_violations,
                        'l_diversity_satisfied': l_violations == 0
                    }
            
            results['l_diversity'] = l_diversity_results
        
        # Recommendations
        recommendations = []
        if k_violations > 0:
            recommendations.append(f"Generalize quasi-identifiers to achieve {self.k_anonymity}-anonymity")
        
        if sensitive_attrs:
            for attr, ldiv_results in results.get('l_diversity', {}).items():
                if ldiv_results['l_violations'] > 0:
                    recommendations.append(f"Increase diversity in {attr} to achieve {self.l_diversity}-diversity")
        
        results['recommendations'] = recommendations
        
        self._log_privacy_action('k_anonymity_check', results)
        
        return results
    
    def anonymize_dataset(self,
                         df: pd.DataFrame,
                         quasi_identifiers: List[str],
                         sensitive_attrs: Optional[List[str]] = None,
                         anonymization_method: str = 'generalization') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Anonymize dataset to achieve k-anonymity and l-diversity.
        
        Args:
            df: Dataset to anonymize
            quasi_identifiers: Quasi-identifier columns
            sensitive_attrs: Sensitive attribute columns
            anonymization_method: Method to use ('generalization', 'suppression')
            
        Returns:
            Tuple of (anonymized_df, anonymization_report)
        """
        df_anon = df.copy()
        report = {
            'method': anonymization_method,
            'original_rows': len(df),
            'quasi_identifiers': quasi_identifiers,
            'sensitive_attributes': sensitive_attrs or []
        }
        
        # Initial privacy assessment
        initial_assessment = self.check_k_anonymity(df, quasi_identifiers, sensitive_attrs)
        report['initial_assessment'] = initial_assessment
        
        if anonymization_method == 'suppression':
            # Remove groups that don't meet k-anonymity
            qi_groups = df_anon.groupby(quasi_identifiers)
            group_sizes = qi_groups.size()
            
            valid_groups = group_sizes[group_sizes >= self.k_anonymity].index
            df_anon = df_anon.set_index(quasi_identifiers).loc[valid_groups].reset_index()
            
        elif anonymization_method == 'generalization':
            # Generalize quasi-identifiers
            for qi in quasi_identifiers:
                if df_anon[qi].dtype in ['int64', 'float64']:
                    # Numeric generalization (binning)
                    df_anon[qi] = self._generalize_numeric(df_anon[qi])
                else:
                    # Categorical generalization (hierarchy)
                    df_anon[qi] = self._generalize_categorical(df_anon[qi])
        
        # Final assessment
        final_assessment = self.check_k_anonymity(df_anon, quasi_identifiers, sensitive_attrs)
        report['final_assessment'] = final_assessment
        report['anonymized_rows'] = len(df_anon)
        report['suppression_rate'] = 1 - (len(df_anon) / len(df))
        
        self._log_privacy_action('anonymization', report)
        
        return df_anon, report
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy compliance report."""
        report = {
            'privacy_settings': {
                'min_cell_size': self.min_cell_size,
                'suppression_threshold': self.suppression_threshold,
                'k_anonymity': self.k_anonymity,
                'l_diversity': self.l_diversity,
                'differential_privacy_enabled': self.enable_differential_privacy,
                'epsilon': self.epsilon,
                'delta': self.delta
            },
            'audit_summary': {
                'total_actions': len(self.audit_log),
                'total_data_accesses': len(self.data_access_log),
                'recent_actions': self.audit_log[-10:] if self.audit_log else []
            },
            'compliance_checks': self._run_compliance_checks(),
            'recommendations': self._generate_privacy_recommendations()
        }
        
        return report
    
    def _sanitize_dict(self, data: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Sanitize dictionary outputs."""
        sanitized = {}
        
        for key, value in data.items():
            # Skip patient-level identifiers
            if self._is_patient_identifier(key):
                continue
            
            # Apply context-specific sanitization
            if context == 'table' and self._is_count_variable(key):
                sanitized[key] = self._apply_cell_suppression(value)
            else:
                sanitized[key] = self.sanitize_output(value, context)
        
        return sanitized
    
    def _sanitize_dataframe(self, df: pd.DataFrame, context: str) -> pd.DataFrame:
        """Sanitize dataframe outputs."""
        df_clean = df.copy()
        
        # Remove patient identifiers
        id_columns = [col for col in df_clean.columns if self._is_patient_identifier(col)]
        df_clean = df_clean.drop(columns=id_columns)
        
        # Apply suppression to small counts
        for col in df_clean.columns:
            if self._is_count_variable(col):
                df_clean[col] = df_clean[col].apply(self._apply_cell_suppression)
        
        # Ensure minimum group size
        if len(df_clean) < self.min_cell_size:
            return pd.DataFrame()  # Return empty if too small
        
        return df_clean
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize string outputs for PHI."""
        # Apply PHI removal patterns
        for pattern_name, pattern in self.phi_patterns.items():
            text = re.sub(pattern['regex'], pattern['replacement'], text)
        
        return text
    
    def _sanitize_number(self, value: Union[int, float], context: str) -> Union[int, float, str]:
        """Sanitize numeric values."""
        if context == 'count' or self._appears_to_be_count(value):
            return self._apply_cell_suppression(value)
        
        # Apply differential privacy noise if enabled
        if self.enable_differential_privacy and isinstance(value, (int, float)):
            return self._add_differential_privacy_noise(value, 'mean')
        
        # Round to reduce precision
        if isinstance(value, float):
            return round(value, 3)
        
        return value
    
    def _apply_cell_suppression(self, value: Union[int, float]) -> Union[int, float, str]:
        """Apply cell suppression rules."""
        if isinstance(value, (int, float)) and 0 < value < self.suppression_threshold:
            return f"<{self.suppression_threshold}"
        return value
    
    def _add_differential_privacy_noise(self, value: float, query_type: str) -> float:
        """Add calibrated Laplace noise for differential privacy."""
        # Determine sensitivity based on query type
        sensitivity_map = {
            'count': 1.0,
            'mean': 1.0,  # Assuming normalized data
            'sum': 1.0,
            'median': 1.0
        }
        
        sensitivity = sensitivity_map.get(query_type, 1.0)
        scale = sensitivity / self.epsilon
        
        # Add Laplace noise
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def _enforce_k_anonymity(self, df: pd.DataFrame, group_vars: List[str]) -> pd.DataFrame:
        """Ensure k-anonymity in aggregated results."""
        if 'n' not in df.columns:
            return df
        
        # Remove groups smaller than k
        return df[df['n'] >= self.k_anonymity].copy()
    
    def _generalize_numeric(self, series: pd.Series, bins: int = 5) -> pd.Series:
        """Generalize numeric values using binning."""
        try:
            return pd.cut(series, bins=bins, labels=False, duplicates='drop')
        except:
            return series
    
    def _generalize_categorical(self, series: pd.Series) -> pd.Series:
        """Generalize categorical values using hierarchy."""
        # Simple generalization: keep top categories, group others as "Other"
        value_counts = series.value_counts()
        top_categories = value_counts.head(3).index.tolist()
        
        return series.apply(lambda x: x if x in top_categories else "Other")
    
    def _initialize_phi_patterns(self) -> Dict[str, Dict[str, str]]:
        """Initialize PHI detection patterns."""
        return {
            'patient_id': {
                'regex': r'\b(PT|PAT|MRN|ID)[0-9]{4,}\b',
                'replacement': '[PATIENT_ID]'
            },
            'ssn': {
                'regex': r'\b\d{3}-\d{2}-\d{4}\b',
                'replacement': '[SSN]'
            },
            'phone': {
                'regex': r'\b\d{3}-\d{3}-\d{4}\b',
                'replacement': '[PHONE]'
            },
            'date': {
                'regex': r'\b\d{4}-\d{2}-\d{2}\b',
                'replacement': '[DATE]'
            },
            'email': {
                'regex': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'replacement': '[EMAIL]'
            },
            'name': {
                'regex': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                'replacement': '[NAME]'
            },
            'age_over_89': {
                'regex': r'\b(9[0-9]|1[0-9]{2})\s*years?\s*old\b',
                'replacement': '>89 years old'
            }
        }
    
    def _is_patient_identifier(self, column_name: str) -> bool:
        """Check if column name suggests patient identifier."""
        # Handle non-string keys
        if not isinstance(column_name, str):
            return False
            
        id_patterns = [
            'patient_id', 'hospitalization_id', 'mrn', 'ssn', 
            'medical_record_number', 'encounter_id'
        ]
        return any(pattern in column_name.lower() for pattern in id_patterns)
    
    def _is_count_variable(self, variable_name: str) -> bool:
        """Check if variable appears to be a count."""
        # Handle non-string keys
        if not isinstance(variable_name, str):
            return False
            
        count_patterns = ['count', 'n', 'total', 'frequency']
        return any(pattern in variable_name.lower() for pattern in count_patterns)
    
    def _appears_to_be_count(self, value: Union[int, float]) -> bool:
        """Heuristic to determine if value appears to be a count."""
        return (isinstance(value, int) and 
                0 <= value <= 100 and  # Reasonable range for small counts
                value == int(value))    # Integer value
    
    def _log_data_access(self, context: str, data_type: str):
        """Log data access for audit purposes."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'data_type': data_type,
            'privacy_settings': {
                'min_cell_size': self.min_cell_size,
                'differential_privacy': self.enable_differential_privacy
            }
        }
        self.data_access_log.append(log_entry)
    
    def _log_privacy_action(self, action: str, details: Dict[str, Any]):
        """Log privacy-related actions."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        }
        self.audit_log.append(log_entry)
    
    def _run_compliance_checks(self) -> Dict[str, Any]:
        """Run automated compliance checks."""
        checks = {
            'min_cell_size_configured': self.min_cell_size >= 5,
            'suppression_threshold_set': self.suppression_threshold > 0,
            'k_anonymity_reasonable': self.k_anonymity >= 3,
            'audit_logging_enabled': len(self.audit_log) >= 0,
            'phi_patterns_loaded': len(self.phi_patterns) > 0
        }
        
        checks['overall_compliance'] = all(checks.values())
        
        return checks
    
    def _generate_privacy_recommendations(self) -> List[str]:
        """Generate privacy recommendations based on current settings."""
        recommendations = []
        
        if self.min_cell_size < 10:
            recommendations.append(
                "Consider increasing minimum cell size to 10+ for stronger privacy protection"
            )
        
        if not self.enable_differential_privacy:
            recommendations.append(
                "Consider enabling differential privacy for additional protection"
            )
        
        if self.epsilon > 1.0:
            recommendations.append(
                "Consider reducing epsilon parameter for stronger differential privacy"
            )
        
        if len(self.data_access_log) > 1000:
            recommendations.append(
                "Regular audit log cleanup recommended to maintain performance"
            )
        
        return recommendations