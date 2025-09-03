"""
Synthetic Data Generation Module

Generates realistic clinical datasets for testing and demonstration:
- CLIF-compatible synthetic data
- Realistic clinical patterns and relationships
- Privacy-safe test data
- Configurable sample sizes and complexity
"""

from .generator import CLIFSyntheticGenerator

__all__ = ['CLIFSyntheticGenerator']