"""
Statistical Analysis Module

Comprehensive clinical research analysis capabilities:
- Descriptive statistics and Table 1 generation
- Hypothesis testing with multiple comparison correction
- Regression modeling with diagnostics
- Machine learning with AutoML
- Sensitivity analyses
"""

from .descriptive import DescriptiveAnalysis
from .hypothesis_testing import HypothesisTesting
from .regression import RegressionAnalysis
from .ml_models import MLAnalysis
from .sensitivity import SensitivityAnalysis

__all__ = [
    'DescriptiveAnalysis',
    'HypothesisTesting', 
    'RegressionAnalysis',
    'MLAnalysis',
    'SensitivityAnalysis'
]