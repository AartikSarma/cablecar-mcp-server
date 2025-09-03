"""
Data Loader Module

Supports loading clinical data from multiple formats with
automatic schema detection and validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import json
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Universal data loader for clinical research datasets.
    
    Supports:
    - CSV files
    - Parquet files  
    - FST files
    - SQL databases
    - CLIF format validation
    """
    
    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        self.loaded_tables = {}
        self.schema_info = {}
        self.data_dictionary = {}
        
    def load_clif_dataset(self, tables: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load a complete CLIF dataset with standard tables.
        
        Args:
            tables: Specific tables to load. If None, loads all available.
            
        Returns:
            Dictionary of table_name -> DataFrame
        """
        standard_clif_tables = [
            'patient', 'hospitalization', 'adt', 'vitals', 'labs',
            'respiratory_support', 'medication_administration', 
            'patient_assessments', 'position'
        ]
        
        if tables is None:
            tables = standard_clif_tables
            
        loaded_data = {}
        
        for table in tables:
            try:
                df = self.load_table(table)
                if df is not None:
                    loaded_data[table] = df
                    logger.info(f"Loaded {table}: {len(df)} rows")
                else:
                    logger.warning(f"Table {table} not found or empty")
            except Exception as e:
                logger.error(f"Error loading {table}: {e}")
                
        self.loaded_tables = loaded_data
        self._generate_data_dictionary()
        
        return loaded_data
    
    def load_table(self, table_name: str, file_format: str = 'auto') -> Optional[pd.DataFrame]:
        """
        Load a single table with format auto-detection.
        
        Args:
            table_name: Name of the table to load
            file_format: Format hint ('csv', 'parquet', 'fst', 'auto')
            
        Returns:
            DataFrame or None if not found
        """
        if file_format == 'auto':
            file_format = self._detect_format(table_name)
            
        if file_format == 'csv':
            return self._load_csv(table_name)
        elif file_format == 'parquet':
            return self._load_parquet(table_name)
        elif file_format == 'fst':
            return self._load_fst(table_name)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
    
    def _detect_format(self, table_name: str) -> str:
        """Detect file format based on available files."""
        formats = ['csv', 'parquet', 'fst']
        
        for fmt in formats:
            if (self.data_path / f"{table_name}.{fmt}").exists():
                return fmt
                
        raise FileNotFoundError(f"No supported format found for {table_name}")
    
    def _load_csv(self, table_name: str) -> pd.DataFrame:
        """Load CSV file with intelligent parsing."""
        filepath = self.data_path / f"{table_name}.csv"
        
        # Try to load with automatic dtype inference
        df = pd.read_csv(filepath, low_memory=False)
        
        # Parse datetime columns
        datetime_patterns = ['dttm', 'date', 'time']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in datetime_patterns):
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Parse numeric columns that might be strings
        for col in df.select_dtypes(include=['object']).columns:
            if col not in [c for c in df.columns if any(p in c.lower() for p in datetime_patterns)]:
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if not numeric_series.isna().all():
                    df[col] = numeric_series
        
        return df
    
    def _load_parquet(self, table_name: str) -> pd.DataFrame:
        """Load Parquet file."""
        filepath = self.data_path / f"{table_name}.parquet"
        return pd.read_parquet(filepath)
    
    def _load_fst(self, table_name: str) -> pd.DataFrame:
        """Load FST file (requires fst package)."""
        try:
            import fst
            filepath = self.data_path / f"{table_name}.fst"
            return fst.read_fst(filepath)
        except ImportError:
            raise ImportError("FST support requires 'python-fst' package")
    
    def _generate_data_dictionary(self):
        """Generate comprehensive data dictionary from loaded data."""
        dictionary = {}
        
        for table_name, df in self.loaded_tables.items():
            table_info = {
                'n_rows': len(df),
                'n_cols': len(df.columns),
                'columns': {}
            }
            
            for col in df.columns:
                col_info = {
                    'dtype': str(df[col].dtype),
                    'missing_count': df[col].isna().sum(),
                    'missing_percent': (df[col].isna().sum() / len(df)) * 100,
                    'unique_count': df[col].nunique()
                }
                
                # Add type-specific statistics
                if df[col].dtype in ['int64', 'float64']:
                    col_info.update({
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'median': df[col].median()
                    })
                elif df[col].dtype == 'object':
                    # For categorical data
                    value_counts = df[col].value_counts()
                    col_info.update({
                        'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                        'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                        'categories': value_counts.head(10).to_dict()
                    })
                
                table_info['columns'][col] = col_info
            
            dictionary[table_name] = table_info
        
        self.data_dictionary = dictionary
        
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get summary of dataset schema and quality."""
        summary = {
            'tables_loaded': list(self.loaded_tables.keys()),
            'total_tables': len(self.loaded_tables),
            'total_rows': sum(len(df) for df in self.loaded_tables.values()),
            'data_quality': {}
        }
        
        for table_name, df in self.loaded_tables.items():
            missing_data = df.isna().sum().sum()
            total_cells = len(df) * len(df.columns)
            
            summary['data_quality'][table_name] = {
                'completeness': ((total_cells - missing_data) / total_cells) * 100,
                'missing_cells': missing_data,
                'total_cells': total_cells
            }
        
        return summary
    
    def validate_clif_schema(self) -> Dict[str, List[str]]:
        """
        Validate loaded data against expected CLIF schema.
        
        Returns:
            Dictionary of validation errors by table
        """
        clif_required_columns = {
            'patient': ['patient_id', 'race', 'ethnicity', 'sex'],
            'hospitalization': ['hospitalization_id', 'patient_id', 'admission_dttm', 'discharge_dttm'],
            'adt': ['hospitalization_id', 'in_dttm', 'out_dttm', 'location_category'],
            'vitals': ['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
            'labs': ['hospitalization_id', 'lab_collected_dttm', 'lab_category', 'lab_value']
        }
        
        errors = {}
        
        for table_name, required_cols in clif_required_columns.items():
            if table_name in self.loaded_tables:
                df = self.loaded_tables[table_name]
                table_errors = []
                
                for col in required_cols:
                    if col not in df.columns:
                        table_errors.append(f"Missing required column: {col}")
                
                if table_errors:
                    errors[table_name] = table_errors
            else:
                errors[table_name] = [f"Table {table_name} not loaded"]
        
        return errors
    
    def export_data_dictionary(self, output_path: Union[str, Path]) -> None:
        """Export data dictionary to JSON file."""
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            json.dump(self.data_dictionary, f, indent=2, default=str)
        
        logger.info(f"Data dictionary exported to {output_path}")
    
    def merge_core_tables(self) -> pd.DataFrame:
        """
        Merge core CLIF tables for analysis.
        
        Returns:
            Merged DataFrame with patient, hospitalization, and derived features
        """
        if 'patient' not in self.loaded_tables or 'hospitalization' not in self.loaded_tables:
            raise ValueError("Core tables 'patient' and 'hospitalization' must be loaded")
        
        # Start with patient table
        merged = self.loaded_tables['patient'].copy()
        
        # Merge hospitalization
        merged = merged.merge(
            self.loaded_tables['hospitalization'], 
            on='patient_id', 
            how='inner'
        )
        
        # Add derived features
        if 'admission_dttm' in merged.columns and 'discharge_dttm' in merged.columns:
            merged['los_hours'] = (
                pd.to_datetime(merged['discharge_dttm']) - 
                pd.to_datetime(merged['admission_dttm'])
            ).dt.total_seconds() / 3600
            merged['los_days'] = merged['los_hours'] / 24
        
        # Add mortality outcome
        if 'discharge_disposition' in merged.columns:
            merged['mortality'] = (merged['discharge_disposition'] == 'Expired').astype(int)
        
        # Calculate age at admission if birth date available
        if 'date_of_birth' in merged.columns and 'admission_dttm' in merged.columns:
            birth_date = pd.to_datetime(merged['date_of_birth'])
            admission_date = pd.to_datetime(merged['admission_dttm'])
            merged['age_at_admission'] = ((admission_date - birth_date).dt.days / 365.25).round(1)
        
        return merged