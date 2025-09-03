"""
Test Data Import Module

Tests for data loading, schema validation, and quality checking.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from cablecar_research.data_import.loaders import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    @pytest.fixture
    def sample_data_dir(self):
        """Create temporary directory with sample data files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample patient data
            patient_data = {
                'patient_id': ['PT000001', 'PT000002', 'PT000003'],
                'date_of_birth': ['1970-01-01', '1980-02-15', '1990-03-20'],
                'sex': ['Male', 'Female', 'Male'],
                'race': ['White', 'Black or African American', 'Asian']
            }
            patient_df = pd.DataFrame(patient_data)
            patient_df.to_csv(Path(tmpdir) / 'patient.csv', index=False)
            
            # Also create parquet version if pyarrow is available
            try:
                patient_df.to_parquet(Path(tmpdir) / 'patient.parquet', index=False)
            except ImportError:
                pass  # Skip parquet if not available
            
            # Create sample hospitalization data
            hosp_data = {
                'hospitalization_id': ['H00000001', 'H00000002', 'H00000003'],
                'patient_id': ['PT000001', 'PT000002', 'PT000003'],
                'admission_dttm': ['2023-01-01 10:00:00', '2023-01-02 14:30:00', '2023-01-03 08:15:00'],
                'discharge_dttm': ['2023-01-05 16:00:00', '2023-01-08 12:00:00', '2023-01-10 09:30:00'],
                'discharge_disposition': ['Home', 'Expired', 'Home']
            }
            pd.DataFrame(hosp_data).to_csv(Path(tmpdir) / 'hospitalization.csv', index=False)
            
            yield tmpdir
    
    def test_load_clif_dataset(self, sample_data_dir):
        """Test loading CLIF dataset."""
        loader = DataLoader(sample_data_dir)
        datasets = loader.load_clif_dataset(['patient', 'hospitalization'])
        
        assert len(datasets) == 2
        assert 'patient' in datasets
        assert 'hospitalization' in datasets
        assert len(datasets['patient']) == 3
        assert len(datasets['hospitalization']) == 3
    
    def test_merge_core_tables(self, sample_data_dir):
        """Test merging core tables."""
        loader = DataLoader(sample_data_dir)
        loader.load_clif_dataset(['patient', 'hospitalization'])
        
        merged_df = loader.merge_core_tables()
        
        assert len(merged_df) == 3
        assert 'patient_id' in merged_df.columns
        assert 'hospitalization_id' in merged_df.columns
        assert 'mortality' in merged_df.columns
        
        # Check mortality outcome creation
        assert merged_df['mortality'].sum() == 1  # One expired patient
    
    def test_schema_validation(self, sample_data_dir):
        """Test CLIF schema validation."""
        loader = DataLoader(sample_data_dir)
        loader.load_clif_dataset(['patient', 'hospitalization'])
        
        errors = loader.validate_clif_schema()
        
        # Should have minimal errors with our sample data
        assert isinstance(errors, dict)
    
    def test_data_dictionary_generation(self, sample_data_dir):
        """Test data dictionary generation."""
        loader = DataLoader(sample_data_dir)
        loader.load_clif_dataset(['patient', 'hospitalization'])
        
        assert hasattr(loader, 'data_dictionary')
        assert len(loader.data_dictionary) > 0
        
        # Check that patient table info is included
        if 'patient' in loader.data_dictionary:
            patient_info = loader.data_dictionary['patient']
            assert 'n_rows' in patient_info
            assert patient_info['n_rows'] == 3
    
    def test_missing_file_handling(self):
        """Test handling of missing data files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(tmpdir)
            datasets = loader.load_clif_dataset(['nonexistent_table'])
            
            # Should return empty dict without crashing
            assert datasets == {}
    
    @pytest.mark.parametrize("file_format", ["csv", "parquet"])
    def test_load_table_formats(self, sample_data_dir, file_format):
        """Test loading different file formats."""
        loader = DataLoader(sample_data_dir)
        
        if file_format == "csv":
            df = loader.load_table("patient", file_format)
            assert df is not None
            assert len(df) == 3
        elif file_format == "parquet":
            # Skip parquet test if file doesn't exist
            try:
                df = loader.load_table("patient", file_format)
            except FileNotFoundError:
                pytest.skip("Parquet file not available in test data")


if __name__ == "__main__":
    pytest.main([__file__])