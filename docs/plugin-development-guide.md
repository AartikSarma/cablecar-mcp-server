# CableCar Plugin Development Guide

A comprehensive technical guide for developing analysis plugins for CableCar.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Plugin Interface](#plugin-interface)
3. [Development Workflow](#development-workflow)
4. [Privacy Protection](#privacy-protection)
5. [Testing Strategies](#testing-strategies)
6. [Best Practices](#best-practices)
7. [Common Patterns](#common-patterns)

## Architecture Overview

### Plugin System Design

CableCar uses a modular plugin architecture that allows dynamic loading and execution of analysis modules:

```
cablecar_research/
├── analysis/
│   └── base.py              # BaseAnalysis abstract class
├── plugins/
│   ├── __init__.py          # Plugin discovery system  
│   ├── core/                # Core analyses (built-in)
│   ├── community/           # Community contributions
│   └── contrib/             # Experimental analyses
└── registry.py              # Plugin registry and MCP integration
```

### Plugin Lifecycle

1. **Discovery**: Registry scans plugin directories on initialization
2. **Registration**: Plugins are registered with metadata validation
3. **Tool Generation**: MCP tools are dynamically created from plugin specs
4. **Execution**: Plugins are instantiated with data and executed on demand
5. **Formatting**: Results are formatted according to requested output type

## Plugin Interface

### BaseAnalysis Abstract Class

All plugins inherit from `BaseAnalysis`:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

class BaseAnalysis(ABC):
    """Base class for all CableCar analysis plugins."""
    
    def __init__(self, df: Optional[pd.DataFrame], privacy_guard: Optional['PrivacyGuard']):
        self.df = df
        self.privacy_guard = privacy_guard
    
    @abstractmethod 
    def validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """Validate analysis inputs and parameters."""
        pass
    
    @abstractmethod
    def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """Execute the analysis and return results."""
        pass
    
    @abstractmethod
    def format_results(self, results: Dict[str, Any], format_type: str = "standard") -> str:
        """Format results for display."""
        pass
    
    @abstractmethod
    def get_required_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Define parameter schema for MCP tool generation."""
        pass
```

### Metadata Schema

Plugin metadata provides essential information for discovery and documentation:

```python
@dataclass
class AnalysisMetadata:
    name: str                    # Unique plugin identifier
    display_name: str           # Human-readable name
    description: str            # Brief description
    version: str               # Semantic version
    author: str                # Author name
    email: str                 # Author contact
    analysis_type: AnalysisType # DESCRIPTIVE, INFERENTIAL, PREDICTIVE, EXPLORATORY
    validation_level: ValidationLevel  # BASIC, STANDARD, STRICT
    citation: str              # Preferred citation format
    keywords: List[str]        # Searchable keywords
```

## Development Workflow

### 1. Generate Plugin Template

```bash
python scripts/generate_plugin_template.py \
    --name "survival_analysis" \
    --type "inferential" \
    --author "Dr. Jane Smith" \
    --email "jane.smith@hospital.edu" \
    --description "Cox proportional hazards survival analysis"
```

### 2. Implement Core Methods

#### validate_inputs()

Validate all inputs before analysis execution:

```python
def validate_inputs(self, **kwargs) -> Dict[str, Any]:
    validation = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'suggestions': []
    }
    
    # Check data availability
    if self.df is None:
        validation['valid'] = False
        validation['errors'].append("No data loaded")
        return validation
    
    # Check required parameters
    time_var = kwargs.get('time_variable')
    if not time_var:
        validation['valid'] = False
        validation['errors'].append("'time_variable' parameter is required")
    elif time_var not in self.df.columns:
        validation['valid'] = False
        validation['errors'].append(f"Time variable '{time_var}' not found in data")
    
    # Clinical validation
    event_var = kwargs.get('event_variable')
    if event_var and event_var in self.df.columns:
        unique_events = self.df[event_var].dropna().unique()
        if len(unique_events) > 2:
            validation['warnings'].append(f"Event variable has {len(unique_events)} levels, expected binary")
    
    # Sample size requirements
    if len(self.df) < 50:
        validation['warnings'].append("Small sample size may affect survival model reliability")
    
    return validation
```

#### run_analysis()

Core analysis implementation:

```python
def run_analysis(self, **kwargs) -> Dict[str, Any]:
    # Extract parameters
    time_var = kwargs['time_variable']
    event_var = kwargs['event_variable'] 
    covariates = kwargs.get('covariates', [])
    
    results = {
        'analysis_type': 'survival_analysis',
        'sample_size': len(self.df),
        'time_variable': time_var,
        'event_variable': event_var,
        'n_covariates': len(covariates)
    }
    
    # Prepare data
    analysis_data = self._prepare_survival_data(time_var, event_var, covariates)
    
    # Fit Cox model
    cox_results = self._fit_cox_model(analysis_data, covariates)
    results['cox_model'] = cox_results
    
    # Model diagnostics
    diagnostics = self._assess_proportional_hazards(analysis_data, covariates)
    results['diagnostics'] = diagnostics
    
    # Survival curves
    if kwargs.get('generate_curves', True):
        curves = self._generate_survival_curves(analysis_data)
        results['survival_curves'] = curves
    
    # Apply privacy protection
    if self.privacy_guard:
        results = self.privacy_guard.sanitize_output(results, 'survival_analysis')
    
    return results
```

#### format_results()

Multiple output formats for different use cases:

```python
def format_results(self, results: Dict[str, Any], format_type: str = "standard") -> str:
    formatters = {
        "standard": self._format_standard,
        "summary": self._format_summary,
        "detailed": self._format_detailed,
        "publication": self._format_publication
    }
    
    formatter = formatters.get(format_type, self._format_standard)
    return formatter(results)

def _format_publication(self, results: Dict[str, Any]) -> str:
    """Publication-ready table format."""
    output = f"Table: Cox Proportional Hazards Model Results\\n"
    output += "=" * 60 + "\\n\\n"
    
    cox_results = results.get('cox_model', {})
    coefficients = cox_results.get('coefficients', {})
    
    output += f"{'Variable':<20} {'HR':<10} {'95% CI':<15} {'p-value':<10}\\n"
    output += "-" * 60 + "\\n"
    
    for var, coef_info in coefficients.items():
        hr = coef_info.get('hazard_ratio', 'N/A')
        ci_lower = coef_info.get('ci_lower', 'N/A') 
        ci_upper = coef_info.get('ci_upper', 'N/A')
        p_val = coef_info.get('p_value', 'N/A')
        
        if isinstance(hr, float):
            hr_str = f"{hr:.2f}"
            ci_str = f"[{ci_lower:.2f}, {ci_upper:.2f}]"
            p_str = f"{p_val:.3f}" if p_val < 0.001 else f"{p_val:.3f}"
        else:
            hr_str = str(hr)
            ci_str = "N/A"
            p_str = str(p_val)
        
        output += f"{var:<20} {hr_str:<10} {ci_str:<15} {p_str:<10}\\n"
    
    # Model statistics
    output += f"\\nModel Statistics:\\n"
    output += f"  Concordance: {cox_results.get('concordance', 'N/A')}\\n"
    output += f"  Log-likelihood: {cox_results.get('log_likelihood', 'N/A')}\\n"
    
    return output
```

#### get_required_parameters()

Define parameter schema for MCP tool generation:

```python
def get_required_parameters(self) -> Dict[str, Dict[str, Any]]:
    return {
        'required': {
            'time_variable': {
                'type': 'string',
                'description': 'Variable containing survival time/follow-up duration',
                'example': 'survival_days'
            },
            'event_variable': {
                'type': 'string', 
                'description': 'Binary variable indicating event occurrence (1=event, 0=censored)',
                'example': 'death'
            }
        },
        'optional': {
            'covariates': {
                'type': 'array',
                'description': 'Variables to include as covariates in Cox model',
                'example': ['age', 'sex', 'stage']
            },
            'generate_curves': {
                'type': 'boolean',
                'description': 'Generate Kaplan-Meier survival curves',
                'example': True
            },
            'stratify_by': {
                'type': 'string',
                'description': 'Variable to stratify survival curves by',
                'example': 'treatment_group'
            }
        }
    }
```

### 3. Implement Helper Methods

Break complex analysis into focused helper methods:

```python
def _prepare_survival_data(self, time_var: str, event_var: str, covariates: List[str]) -> pd.DataFrame:
    """Prepare data for survival analysis."""
    # Select relevant columns
    analysis_vars = [time_var, event_var] + covariates
    df_clean = self.df[analysis_vars].copy()
    
    # Remove missing values
    df_clean = df_clean.dropna()
    
    # Validate time variable
    if (df_clean[time_var] <= 0).any():
        raise ValueError("Survival times must be positive")
    
    # Validate event variable
    unique_events = df_clean[event_var].unique()
    if not set(unique_events).issubset({0, 1}):
        raise ValueError("Event variable must be binary (0/1)")
    
    return df_clean

def _fit_cox_model(self, df: pd.DataFrame, covariates: List[str]) -> Dict[str, Any]:
    """Fit Cox proportional hazards model."""
    from lifelines import CoxPHFitter
    
    try:
        cph = CoxPHFitter()
        cph.fit(df, duration_col=df.columns[0], event_col=df.columns[1])
        
        # Extract results
        results = {
            'coefficients': {},
            'concordance': cph.concordance_index_,
            'log_likelihood': cph.log_likelihood_,
            'aic': cph.AIC_
        }
        
        # Process coefficients
        for var in covariates:
            if var in cph.summary.index:
                coef_row = cph.summary.loc[var]
                results['coefficients'][var] = {
                    'coefficient': coef_row['coef'],
                    'hazard_ratio': coef_row['exp(coef)'],
                    'se': coef_row['se(coef)'],
                    'ci_lower': coef_row['exp(coef) lower 95%'],
                    'ci_upper': coef_row['exp(coef) upper 95%'], 
                    'p_value': coef_row['p']
                }
        
        return results
        
    except Exception as e:
        return {'error': f"Cox model fitting failed: {str(e)}"}
```

## Privacy Protection

### Automatic Privacy Enforcement

The privacy guard automatically sanitizes all plugin outputs:

```python
def run_analysis(self, **kwargs) -> Dict[str, Any]:
    results = self._perform_analysis(**kwargs)
    
    # Privacy protection is automatic
    if self.privacy_guard:
        results = self.privacy_guard.sanitize_output(results, self.metadata.name)
    
    return results
```

### Privacy-Safe Practices

#### Small Cell Suppression

```python
def _calculate_group_statistics(self, group_var: str) -> Dict[str, Any]:
    """Calculate statistics with privacy protection."""
    group_counts = self.df[group_var].value_counts()
    
    # Only report groups meeting minimum size
    min_size = self.privacy_guard.min_cell_size if self.privacy_guard else 10
    large_groups = group_counts[group_counts >= min_size]
    
    return {
        'groups_reported': len(large_groups),
        'groups_suppressed': len(group_counts) - len(large_groups),
        'group_statistics': large_groups.to_dict()
    }
```

#### Avoiding Patient Re-identification

```python
def _generate_summary_statistics(self) -> Dict[str, Any]:
    """Generate privacy-safe summary statistics."""
    # Safe: Aggregate statistics
    safe_stats = {
        'mean_age': self.df['age'].mean(),
        'median_los': self.df['length_of_stay'].median(),
        'mortality_rate': self.df['mortality'].mean()
    }
    
    # Unsafe: Individual-level data (never include)
    # unsafe_stats = {
    #     'patient_ages': self.df['age'].tolist(),  # DON'T DO THIS
    #     'individual_outcomes': self.df['outcome'].tolist()  # DON'T DO THIS
    # }
    
    return safe_stats
```

## Testing Strategies

### Test Structure

Organize tests by functionality:

```python
class TestSurvivalAnalysisPlugin:
    
    @pytest.fixture
    def sample_survival_data(self):
        """Create synthetic survival data for testing."""
        np.random.seed(42)
        n = 200
        
        return pd.DataFrame({
            'patient_id': range(1, n + 1),
            'time_to_event': np.random.exponential(100, n),
            'event_occurred': np.random.binomial(1, 0.3, n),
            'age': np.random.normal(65, 15, n),
            'treatment': np.random.choice(['A', 'B'], n)
        })
    
    def test_cox_model_accuracy(self, plugin_instance):
        """Test Cox model produces expected results."""
        # Use data with known survival patterns
        # Verify hazard ratios are in expected range
        pass
    
    def test_survival_curve_generation(self, plugin_instance):
        """Test Kaplan-Meier curve generation."""
        results = plugin_instance.run_analysis(
            time_variable='time_to_event',
            event_variable='event_occurred',
            generate_curves=True
        )
        
        assert 'survival_curves' in results
        curves = results['survival_curves']
        assert 'survival_function' in curves
        assert 'confidence_intervals' in curves
    
    def test_proportional_hazards_assumption(self, plugin_instance):
        """Test proportional hazards assumption checking."""
        # Test with data that violates assumptions
        # Verify diagnostic tests catch violations
        pass
```

### Clinical Validation Tests

```python
def test_clinical_validity(self, plugin_instance):
    """Test clinically realistic scenarios."""
    # Test 1: Higher age should increase hazard
    results = plugin_instance.run_analysis(
        time_variable='time_to_event',
        event_variable='event_occurred', 
        covariates=['age']
    )
    
    age_hr = results['cox_model']['coefficients']['age']['hazard_ratio']
    assert age_hr > 1, "Age should increase hazard ratio"
    
    # Test 2: Treatment should modify survival
    results_with_treatment = plugin_instance.run_analysis(
        time_variable='time_to_event',
        event_variable='event_occurred',
        covariates=['age', 'treatment']
    )
    
    # Treatment effect should be detectable
    treatment_p = results_with_treatment['cox_model']['coefficients']['treatment']['p_value']
    assert treatment_p is not None
```

## Best Practices

### Clinical Research Guidelines

1. **Statistical Assumptions**: Always test and report assumption violations
2. **Multiple Comparisons**: Apply appropriate corrections when testing multiple hypotheses  
3. **Effect Sizes**: Report both statistical significance and clinical significance
4. **Confidence Intervals**: Provide confidence intervals for key estimates
5. **Missing Data**: Document and handle missing data appropriately

### Code Quality

1. **Type Hints**: Use comprehensive type annotations
2. **Error Handling**: Graceful handling of edge cases
3. **Documentation**: Clear docstrings and comments
4. **Modularity**: Break complex analysis into focused methods
5. **Performance**: Optimize for large clinical datasets

### Privacy and Security

1. **No Individual Data**: Never expose individual patient records
2. **Aggregate Only**: Only return aggregated statistics
3. **Minimum Cell Size**: Respect privacy guard thresholds
4. **Audit Trail**: Log all data access for compliance

## Common Patterns

### Stratified Analysis Pattern

```python
def run_stratified_analysis(self, stratify_by: str, **kwargs) -> Dict[str, Any]:
    """Common pattern for stratified analyses."""
    results = {
        'overall_results': self._run_single_analysis(**kwargs),
        'stratified_results': {},
        'group_comparisons': {}
    }
    
    # Analyze each stratum
    strata = self.df[stratify_by].unique()
    for stratum in strata:
        stratum_data = self.df[self.df[stratify_by] == stratum]
        stratum_plugin = self.__class__(stratum_data, self.privacy_guard)
        
        if len(stratum_data) >= 20:  # Minimum size
            stratum_results = stratum_plugin._run_single_analysis(**kwargs)
            results['stratified_results'][f"{stratify_by}_{stratum}"] = stratum_results
    
    # Compare between strata
    if len(results['stratified_results']) >= 2:
        comparison = self._compare_strata(results['stratified_results'])
        results['group_comparisons'] = comparison
    
    return results
```

### Missing Data Handling Pattern

```python
def _handle_missing_data(self, variables: List[str], method: str = 'complete_case') -> pd.DataFrame:
    """Common pattern for missing data handling."""
    df_analysis = self.df[variables].copy()
    
    # Document missingness
    missing_summary = {}
    for var in variables:
        n_missing = df_analysis[var].isna().sum()
        pct_missing = (n_missing / len(df_analysis)) * 100
        missing_summary[var] = {'n_missing': n_missing, 'percent_missing': pct_missing}
    
    # Handle missing data
    if method == 'complete_case':
        df_clean = df_analysis.dropna()
    elif method == 'median_imputation':
        df_clean = df_analysis.fillna(df_analysis.median())
    elif method == 'forward_fill':
        df_clean = df_analysis.fillna(method='ffill')
    else:
        raise ValueError(f"Unknown missing data method: {method}")
    
    # Store missing data info for reporting
    self._missing_data_summary = missing_summary
    
    return df_clean
```

### Model Validation Pattern

```python
def _validate_statistical_model(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
    """Common pattern for model validation."""
    validation = {
        'assumptions_met': True,
        'warnings': [],
        'recommendations': []
    }
    
    # Check model convergence
    if 'convergence_failed' in model_results:
        validation['assumptions_met'] = False
        validation['warnings'].append("Model failed to converge")
        validation['recommendations'].append("Try different starting values or reduce model complexity")
    
    # Check sample size adequacy
    n_parameters = model_results.get('n_parameters', 0)
    n_observations = model_results.get('n_observations', 0)
    if n_observations < (n_parameters * 10):
        validation['warnings'].append("Low observations-to-parameters ratio")
        validation['recommendations'].append("Consider reducing model complexity or collecting more data")
    
    # Check for multicollinearity
    if 'vif_values' in model_results:
        high_vif = [var for var, vif in model_results['vif_values'].items() if vif > 5]
        if high_vif:
            validation['warnings'].append(f"High multicollinearity detected: {high_vif}")
            validation['recommendations'].append("Consider removing correlated variables")
    
    return validation
```

This comprehensive guide provides the technical foundation for developing high-quality analysis plugins for CableCar's modular architecture.