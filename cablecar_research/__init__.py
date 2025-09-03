"""
CableCar Research Library

Connecting AI to Big Longitudinal EMRs for Clinical Analytics and Research

A comprehensive clinical research analysis library designed for:
- Standards-compliant analysis (STROBE, TRIPOD+AI)
- Privacy-preserving data analysis
- Reproducible research workflows
- Multi-site federated analysis

This library can be used standalone or as part of the CableCar MCP Server.
"""

__version__ = "1.0.0"
__author__ = "CableCar Research Team"

from .data_import.loaders import DataLoader
from .analysis.descriptive import DescriptiveAnalysis
from .analysis.hypothesis_testing import HypothesisTesting
from .analysis.regression import RegressionAnalysis
from .analysis.ml_models import MLAnalysis
from .analysis.sensitivity import SensitivityAnalysis
from .reporting.strobe_reporter import STROBEReporter
from .reporting.tripod_reporter import TRIPODReporter
from .reporting.visualizations import Visualizer
from .privacy.protection import PrivacyGuard
from .validation.validators import StatisticalValidator

__all__ = [
    'DataLoader',
    'DescriptiveAnalysis',
    'HypothesisTesting', 
    'RegressionAnalysis',
    'MLAnalysis',
    'SensitivityAnalysis',
    'STROBEReporter',
    'TRIPODReporter',
    'Visualizer',
    'PrivacyGuard',
    'StatisticalValidator'
]