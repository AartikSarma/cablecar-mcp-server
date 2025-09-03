"""
Data Import and Schema Management Module

Handles:
- Multiple data formats (CSV, Parquet, FST, databases)
- Schema validation and standardization
- Data quality assessment
- Automated data dictionary generation
"""

from .loaders import DataLoader

__all__ = ['DataLoader']