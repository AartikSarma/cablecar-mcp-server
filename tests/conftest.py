"""
Pytest Configuration and Shared Fixtures

Global fixtures and configuration for the test suite.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

from cablecar_research.privacy.protection import PrivacyGuard


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data files."""
    test_dir = tempfile.mkdtemp(prefix="clif_test_")
    yield test_dir
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_clif_data():
    """Create comprehensive CLIF dataset for testing."""
    np.random.seed(42)  # For reproducible tests
    
    n_patients = 1000
    
    # Patient table
    patient_data = {
        'patient_id': [f'PT{i:06d}' for i in range(1, n_patients + 1)],
        'date_of_birth': pd.date_range('1930-01-01', '2000-01-01', periods=n_patients),
        'sex': np.random.choice(['Male', 'Female'], n_patients),
        'race': np.random.choice(['White', 'Black or African American', 'Asian', 'Hispanic', 'Other'], n_patients),
        'height_cm': np.random.normal(170, 15, n_patients).clip(140, 210),
        'weight_kg': np.random.normal(75, 20, n_patients).clip(40, 150)
    }
    
    # Hospitalization table
    hosp_data = {
        'hospitalization_id': [f'H{i:08d}' for i in range(1, n_patients + 1)],
        'patient_id': [f'PT{i:06d}' for i in range(1, n_patients + 1)],
        'admission_dttm': pd.date_range('2020-01-01', '2023-12-31', periods=n_patients),
        'discharge_dttm': None,  # Will be calculated
        'discharge_disposition': None,  # Will be calculated
        'hospital_id': np.random.choice(['HOSP001', 'HOSP002', 'HOSP003'], n_patients),
        'age_at_admission': None,  # Will be calculated
        'charlson_comorbidity_index': np.random.poisson(3, n_patients)
    }
    
    # Calculate derived fields
    admission_dates = pd.to_datetime(hosp_data['admission_dttm'])
    birth_dates = pd.to_datetime(patient_data['date_of_birth'])
    
    # Length of stay (exponential distribution)
    los_days = np.random.exponential(7, n_patients).clip(1, 60)
    discharge_dates = admission_dates + pd.to_timedelta(los_days, unit='D')
    
    # Age at admission
    ages = (admission_dates - birth_dates).dt.days / 365.25
    
    # Mortality (20% rate, higher for older patients and longer LOS)
    mortality_prob = 0.1 + 0.005 * ages + 0.002 * los_days
    mortality_prob = np.clip(mortality_prob, 0, 0.8)
    mortality = np.random.binomial(1, mortality_prob, n_patients)
    
    # Discharge disposition
    alive_dispositions = ['Home', 'Skilled Nursing Facility', 'Rehabilitation', 'Home Health Care']
    discharge_disp = np.where(
        mortality == 1, 
        'Expired',
        np.random.choice(alive_dispositions, n_patients)
    )
    
    # Update hospitalization data
    hosp_data['discharge_dttm'] = discharge_dates
    hosp_data['discharge_disposition'] = discharge_disp
    hosp_data['age_at_admission'] = ages.round(1)
    
    # Vitals table (multiple records per patient)
    n_vitals = n_patients * 5  # 5 vitals per patient on average
    vitals_data = {
        'vitals_id': [f'V{i:010d}' for i in range(1, n_vitals + 1)],
        'hospitalization_id': np.repeat([f'H{i:08d}' for i in range(1, n_patients + 1)], 5),
        'recorded_dttm': None,  # Will be calculated
        'heart_rate': np.random.normal(85, 20, n_vitals).clip(40, 180),
        'systolic_bp': np.random.normal(120, 25, n_vitals).clip(70, 220),
        'diastolic_bp': np.random.normal(75, 15, n_vitals).clip(40, 120),
        'respiratory_rate': np.random.normal(18, 5, n_vitals).clip(8, 40),
        'temperature_c': np.random.normal(37, 1.5, n_vitals).clip(34, 42),
        'oxygen_saturation': np.random.normal(96, 4, n_vitals).clip(80, 100)
    }
    
    # Generate vitals timestamps within hospitalization periods
    vitals_times = []
    for i in range(n_patients):
        start_time = admission_dates.iloc[i]
        end_time = discharge_dates.iloc[i]
        duration = (end_time - start_time).total_seconds()
        
        for j in range(5):
            # Random time within hospitalization
            random_seconds = np.random.uniform(0, duration)
            vitals_time = start_time + pd.Timedelta(seconds=random_seconds)
            vitals_times.append(vitals_time)
    
    vitals_data['recorded_dttm'] = vitals_times
    
    # Labs table
    n_labs = n_patients * 3  # 3 lab draws per patient on average
    labs_data = {
        'labs_id': [f'L{i:010d}' for i in range(1, n_labs + 1)],
        'hospitalization_id': np.repeat([f'H{i:08d}' for i in range(1, n_patients + 1)], 3),
        'collected_dttm': None,  # Will be calculated
        'hemoglobin': np.random.normal(12, 2.5, n_labs).clip(6, 18),
        'white_blood_cell_count': np.random.lognormal(2, 0.5, n_labs).clip(1, 50),
        'platelet_count': np.random.normal(250, 100, n_labs).clip(50, 600),
        'sodium': np.random.normal(140, 5, n_labs).clip(125, 155),
        'potassium': np.random.normal(4.0, 0.8, n_labs).clip(2.5, 6.0),
        'creatinine': np.random.lognormal(0.3, 0.6, n_labs).clip(0.5, 8.0),
        'glucose': np.random.lognormal(4.8, 0.4, n_labs).clip(70, 400)
    }
    
    # Generate lab timestamps
    lab_times = []
    for i in range(n_patients):
        start_time = admission_dates.iloc[i]
        end_time = discharge_dates.iloc[i]
        duration = (end_time - start_time).total_seconds()
        
        for j in range(3):
            random_seconds = np.random.uniform(0, duration)
            lab_time = start_time + pd.Timedelta(seconds=random_seconds)
            lab_times.append(lab_time)
    
    labs_data['collected_dttm'] = lab_times
    
    # Medications table
    common_medications = [
        'Norepinephrine', 'Vasopressin', 'Propofol', 'Midazolam', 'Fentanyl',
        'Heparin', 'Insulin', 'Furosemide', 'Metoprolol', 'Vancomycin'
    ]
    
    n_medications = n_patients * 2  # 2 medications per patient on average
    meds_data = {
        'medication_id': [f'M{i:010d}' for i in range(1, n_medications + 1)],
        'hospitalization_id': np.repeat([f'H{i:08d}' for i in range(1, n_patients + 1)], 2),
        'medication_name': np.tile(np.random.choice(common_medications, n_patients), 2),
        'start_dttm': None,  # Will be calculated
        'end_dttm': None,  # Will be calculated
        'route': np.random.choice(['IV', 'PO', 'SQ', 'IM'], n_medications),
        'dose': np.random.uniform(0.5, 100, n_medications).round(1),
        'dose_unit': np.random.choice(['mg', 'mcg', 'units'], n_medications)
    }
    
    # Generate medication timestamps
    med_start_times = []
    med_end_times = []
    for i in range(n_patients):
        start_time = admission_dates.iloc[i]
        end_time = discharge_dates.iloc[i]
        duration = (end_time - start_time).total_seconds()
        
        for j in range(2):
            # Random start time
            start_seconds = np.random.uniform(0, duration * 0.8)
            med_start = start_time + pd.Timedelta(seconds=start_seconds)
            
            # Random duration for medication
            med_duration = np.random.uniform(duration * 0.1, duration * 0.5)
            med_end = med_start + pd.Timedelta(seconds=med_duration)
            
            med_start_times.append(med_start)
            med_end_times.append(min(med_end, end_time))
    
    meds_data['start_dttm'] = med_start_times
    meds_data['end_dttm'] = med_end_times
    
    return {
        'patient': pd.DataFrame(patient_data),
        'hospitalization': pd.DataFrame(hosp_data),
        'vitals': pd.DataFrame(vitals_data),
        'labs': pd.DataFrame(labs_data),
        'medications': pd.DataFrame(meds_data)
    }


@pytest.fixture(scope="session")
def privacy_guard_test():
    """Standard privacy guard for testing."""
    return PrivacyGuard(
        min_cell_size=5,  # Lower threshold for testing
        k_anonymity=3,
        epsilon=1.0
    )


@pytest.fixture(scope="session")
def privacy_guard_strict():
    """Strict privacy guard for testing edge cases."""
    return PrivacyGuard(
        min_cell_size=25,
        k_anonymity=10,
        epsilon=0.5
    )


@pytest.fixture
def small_test_data():
    """Small dataset for quick tests."""
    np.random.seed(42)
    n = 50
    
    return pd.DataFrame({
        'patient_id': [f'PT{i:06d}' for i in range(1, n + 1)],
        'age': np.random.normal(65, 15, n),
        'sex': np.random.choice(['Male', 'Female'], n),
        'outcome': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'continuous_var': np.random.normal(100, 20, n),
        'categorical_var': np.random.choice(['A', 'B', 'C'], n),
        'los_days': np.random.exponential(5, n)
    })


@pytest.fixture
def ml_test_data():
    """Dataset specifically for ML testing."""
    from sklearn.datasets import make_classification, make_regression
    
    # Classification data
    X_clf, y_clf = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    clf_data = pd.DataFrame(X_clf, columns=[f'feature_{i}' for i in range(10)])
    clf_data['target'] = y_clf
    clf_data['patient_id'] = [f'PT{i:06d}' for i in range(1, 501)]
    
    # Regression data
    X_reg, y_reg = make_regression(
        n_samples=500,
        n_features=8,
        n_informative=5,
        noise=0.1,
        random_state=42
    )
    
    reg_data = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(8)])
    reg_data['target'] = y_reg
    reg_data['patient_id'] = [f'PT{i:06d}' for i in range(1, 501)]
    
    return {
        'classification': clf_data,
        'regression': reg_data
    }


# Configure pytest settings
def pytest_configure(config):
    """Configure pytest settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Skip tests requiring external dependencies if not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available dependencies."""
    try:
        import sklearn
    except ImportError:
        sklearn_skip = pytest.mark.skip(reason="scikit-learn not available")
        for item in items:
            if "sklearn" in item.nodeid or "ml" in item.nodeid.lower():
                item.add_marker(sklearn_skip)
    
    try:
        import matplotlib
    except ImportError:
        matplotlib_skip = pytest.mark.skip(reason="matplotlib not available")
        for item in items:
            if "visualiz" in item.nodeid.lower() or "plot" in item.nodeid.lower():
                item.add_marker(matplotlib_skip)