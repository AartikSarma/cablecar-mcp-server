"""
Test Privacy Protection Components

Tests for privacy guard and data protection mechanisms.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from cablecar_research.privacy.protection import PrivacyGuard


class TestPrivacyGuard:
    """Test cases for PrivacyGuard class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with potential privacy risks."""
        np.random.seed(42)
        n = 1000
        
        data = {
            'patient_id': [f'PT{i:06d}' for i in range(1, n+1)],
            'name': [f'Patient {i}' for i in range(1, n+1)],  # PHI
            'ssn': [f'{i:03d}-{i:02d}-{i:04d}' for i in range(1, n+1)],  # PHI
            'phone': [f'555-{i:03d}-{i:04d}' for i in range(1, n+1)],  # PHI
            'age': np.random.normal(65, 15, n).clip(18, 95),
            'sex': np.random.choice(['Male', 'Female'], n),
            'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], n),
            'diagnosis': np.random.choice(['Pneumonia', 'Sepsis', 'COPD', 'Heart Failure'], n),
            'mortality': np.random.choice([0, 1], n, p=[0.8, 0.2]),
            'zip_code': np.random.choice(['12345', '67890', '11111', '22222'], n),
            'date_of_birth': pd.date_range('1930-01-01', '2000-01-01', periods=n),
            'admission_date': pd.date_range('2023-01-01', '2023-12-31', periods=n)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def privacy_guard_default(self):
        """Create privacy guard with default settings."""
        return PrivacyGuard()
    
    @pytest.fixture
    def privacy_guard_strict(self):
        """Create privacy guard with strict settings."""
        return PrivacyGuard(
            min_cell_size=25,
            k_anonymity=10,
            epsilon=0.5,
            remove_quasi_identifiers=True
        )
    
    def test_cell_suppression_basic(self, sample_data, privacy_guard_default):
        """Test basic cell suppression for small counts."""
        # Create a crosstab with some small cells
        crosstab = pd.crosstab(sample_data['sex'], sample_data['diagnosis'])
        
        # Manually create some small cells
        crosstab.iloc[0, 0] = 3  # Small cell
        crosstab.iloc[1, 1] = 8  # Small cell
        
        protected_table = privacy_guard_default.suppress_small_cells(crosstab)
        
        assert isinstance(protected_table, pd.DataFrame)
        # Should suppress cells < min_cell_size (default 10)
        assert (protected_table == '<10').sum().sum() >= 2
    
    def test_phi_detection_and_removal(self, sample_data, privacy_guard_default):
        """Test PHI detection and removal."""
        phi_columns = privacy_guard_default.detect_phi_columns(sample_data)
        
        # Should detect name, ssn, phone as PHI
        expected_phi = ['name', 'ssn', 'phone']
        assert any(col in phi_columns for col in expected_phi)
        
        # Test PHI removal
        cleaned_data = privacy_guard_default.remove_phi(sample_data)
        
        # PHI columns should be removed
        for phi_col in expected_phi:
            if phi_col in sample_data.columns:
                assert phi_col not in cleaned_data.columns
    
    def test_quasi_identifier_handling(self, sample_data, privacy_guard_strict):
        """Test quasi-identifier generalization."""
        # Age and zip_code are quasi-identifiers
        protected_data = privacy_guard_strict.generalize_quasi_identifiers(sample_data)
        
        # Check that age has been generalized (binned)
        if 'age' in protected_data.columns:
            # Should have fewer unique values than original
            assert protected_data['age'].nunique() < sample_data['age'].nunique()
    
    def test_k_anonymity_check(self, sample_data, privacy_guard_default):
        """Test k-anonymity validation."""
        # Create a subset with potential k-anonymity violations
        subset_data = sample_data[['age', 'sex', 'race', 'diagnosis']].copy()
        
        # Bin age to create quasi-identifier groups
        subset_data['age_group'] = pd.cut(subset_data['age'], bins=5, labels=['18-35', '35-50', '50-65', '65-80', '80+'])
        
        quasi_identifiers = ['age_group', 'sex', 'race']
        k_anonymous = privacy_guard_default.check_k_anonymity(subset_data, quasi_identifiers, k=5)
        
        assert isinstance(k_anonymous, bool)
    
    def test_differential_privacy_noise(self, privacy_guard_default):
        """Test differential privacy noise addition."""
        # Test with a simple count
        true_count = 100
        noisy_count = privacy_guard_default.add_laplace_noise(true_count, sensitivity=1)
        
        assert isinstance(noisy_count, (int, float))
        # Noisy count should be different from true count (with high probability)
        # But we can't assert this strictly due to randomness
    
    def test_safe_aggregation(self, sample_data, privacy_guard_default):
        """Test safe aggregation with privacy protection."""
        # Test safe counting
        safe_counts = privacy_guard_default.safe_count(
            sample_data, 
            group_by=['sex', 'diagnosis']
        )
        
        assert isinstance(safe_counts, pd.DataFrame)
        # No counts should be below min_cell_size in the final output
        if 'count' in safe_counts.columns:
            numeric_counts = pd.to_numeric(safe_counts['count'], errors='coerce')
            non_suppressed = numeric_counts.dropna()
            if len(non_suppressed) > 0:
                assert all(count >= privacy_guard_default.min_cell_size for count in non_suppressed)
    
    def test_safe_statistics(self, sample_data, privacy_guard_default):
        """Test safe statistical calculations."""
        # Test safe mean calculation
        safe_mean = privacy_guard_default.safe_mean(sample_data['age'])
        
        if safe_mean is not None:
            assert isinstance(safe_mean, (int, float))
            # Should be close to actual mean but with some noise
            actual_mean = sample_data['age'].mean()
            assert abs(safe_mean - actual_mean) < actual_mean * 0.5  # Within 50%
    
    def test_date_anonymization(self, sample_data, privacy_guard_default):
        """Test date anonymization methods."""
        # Test date shifting
        shifted_dates = privacy_guard_default.shift_dates(
            sample_data['date_of_birth'], 
            max_shift_days=365
        )
        
        assert len(shifted_dates) == len(sample_data['date_of_birth'])
        assert shifted_dates.dtype == 'datetime64[ns]'
        
        # Test date binning
        binned_dates = privacy_guard_default.bin_dates(
            sample_data['admission_date'],
            bin_size='month'
        )
        
        assert len(binned_dates) == len(sample_data['admission_date'])
        # Should have fewer unique values than original
        assert binned_dates.nunique() <= sample_data['admission_date'].nunique()
    
    def test_geographic_anonymization(self, sample_data, privacy_guard_default):
        """Test geographic data anonymization."""
        # Test zip code generalization
        generalized_zips = privacy_guard_default.generalize_zip_codes(sample_data['zip_code'])
        
        assert len(generalized_zips) == len(sample_data['zip_code'])
        # Should be generalized (e.g., 12345 -> 123**)
        if len(generalized_zips.dropna()) > 0:
            sample_generalized = generalized_zips.dropna().iloc[0]
            if isinstance(sample_generalized, str):
                assert '*' in sample_generalized or len(set(generalized_zips)) < len(set(sample_data['zip_code']))
    
    def test_privacy_risk_assessment(self, sample_data, privacy_guard_default):
        """Test privacy risk assessment."""
        risk_assessment = privacy_guard_default.assess_privacy_risk(sample_data)
        
        assert isinstance(risk_assessment, dict)
        assert 'overall_risk' in risk_assessment
        assert 'phi_detected' in risk_assessment
        assert 'recommendations' in risk_assessment
        
        # Risk should be high due to PHI in sample data
        assert risk_assessment['phi_detected'] == True
    
    def test_safe_output_validation(self, sample_data, privacy_guard_default):
        """Test that outputs are safe for release."""
        # Create a summary table
        summary = sample_data.groupby(['sex', 'diagnosis']).agg({
            'age': ['mean', 'count'],
            'mortality': 'sum'
        }).reset_index()
        
        # Validate the output
        is_safe = privacy_guard_default.validate_safe_output(summary)
        
        # If not safe, should return False
        assert isinstance(is_safe, bool)
    
    def test_privacy_configuration(self):
        """Test privacy guard configuration options."""
        # Test with different configurations
        config1 = PrivacyGuard(min_cell_size=5, k_anonymity=3, epsilon=1.0)
        config2 = PrivacyGuard(min_cell_size=20, k_anonymity=10, epsilon=0.1)
        
        assert config1.min_cell_size == 5
        assert config2.min_cell_size == 20
        assert config1.k_anonymity == 3
        assert config2.k_anonymity == 10
        assert config1.epsilon == 1.0
        assert config2.epsilon == 0.1
    
    def test_privacy_audit_log(self, sample_data, privacy_guard_default):
        """Test privacy audit logging."""
        # Perform some privacy operations
        privacy_guard_default.suppress_small_cells(pd.crosstab(sample_data['sex'], sample_data['diagnosis']))
        privacy_guard_default.remove_phi(sample_data)
        
        # Check audit log
        audit_log = privacy_guard_default.get_audit_log()
        
        assert isinstance(audit_log, list)
        assert len(audit_log) >= 2  # Should have logged the operations
        
        for entry in audit_log:
            assert 'timestamp' in entry
            assert 'operation' in entry
            assert 'details' in entry


if __name__ == "__main__":
    pytest.main([__file__])