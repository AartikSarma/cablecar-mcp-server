"""
Test Analysis Components

Tests for descriptive statistics, hypothesis testing, and regression analysis.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from cablecar_research.analysis.descriptive import DescriptiveAnalysis
from cablecar_research.analysis.hypothesis_testing import HypothesisTesting
from cablecar_research.analysis.regression import RegressionAnalysis
from cablecar_research.privacy.protection import PrivacyGuard


class TestDescriptiveAnalysis:
    """Test cases for DescriptiveAnalysis class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample clinical data."""
        np.random.seed(42)
        n = 1000
        
        data = {
            'patient_id': [f'PT{i:06d}' for i in range(1, n+1)],
            'age': np.random.normal(65, 15, n).clip(18, 95),
            'sex': np.random.choice(['Male', 'Female'], n),
            'mortality': np.random.choice([0, 1], n, p=[0.8, 0.2]),
            'los_days': np.random.exponential(7, n).clip(1, 50),
            'mechanical_ventilation': np.random.choice([0, 1], n, p=[0.7, 0.3]),
            'charlson_score': np.random.poisson(3, n),
            'sepsis': np.random.choice([0, 1], n, p=[0.6, 0.4])
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def privacy_guard(self):
        """Create privacy guard instance."""
        return PrivacyGuard(min_cell_size=10)
    
    def test_generate_table1(self, sample_data, privacy_guard):
        """Test Table 1 generation."""
        analyzer = DescriptiveAnalysis(privacy_guard=privacy_guard)
        
        variables = ['age', 'sex', 'los_days', 'mechanical_ventilation', 'charlson_score']
        table1 = analyzer.generate_table1(
            data=sample_data,
            variables=variables,
            stratify_by='mortality'
        )
        
        assert isinstance(table1, dict)
        assert 'summary_stats' in table1
        assert 'stratified_analysis' in table1
        assert len(table1['summary_stats']) == len(variables)
    
    def test_missing_data_analysis(self, sample_data, privacy_guard):
        """Test missing data analysis."""
        # Introduce missing data
        sample_data_missing = sample_data.copy()
        sample_data_missing.loc[0:50, 'age'] = np.nan
        sample_data_missing.loc[100:120, 'charlson_score'] = np.nan
        
        analyzer = DescriptiveAnalysis(privacy_guard=privacy_guard)
        missing_analysis = analyzer.analyze_missing_data(sample_data_missing)
        
        assert isinstance(missing_analysis, dict)
        assert 'missing_patterns' in missing_analysis
        assert 'recommendations' in missing_analysis
    
    def test_effect_size_calculation(self, sample_data, privacy_guard):
        """Test effect size calculations."""
        analyzer = DescriptiveAnalysis(privacy_guard=privacy_guard)
        
        # Cohen's d for continuous variable
        cohens_d = analyzer.calculate_effect_size(
            sample_data, 'age', 'mortality', effect_type='cohens_d'
        )
        
        assert isinstance(cohens_d, float)
        assert not np.isnan(cohens_d)


class TestHypothesisTesting:
    """Test cases for HypothesisTesting class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample clinical data."""
        np.random.seed(42)
        n = 500
        
        data = {
            'group': np.random.choice(['Control', 'Treatment'], n),
            'continuous_outcome': np.random.normal(50, 10, n),
            'binary_outcome': np.random.choice([0, 1], n),
            'categorical_var': np.random.choice(['A', 'B', 'C'], n),
            'age': np.random.normal(65, 15, n)
        }
        
        # Create some group differences
        treatment_mask = data['group'] == 'Treatment'
        data['continuous_outcome'] = np.where(
            treatment_mask, 
            np.random.normal(55, 10, sum(treatment_mask)),
            data['continuous_outcome']
        )
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def privacy_guard(self):
        """Create privacy guard instance."""
        return PrivacyGuard(min_cell_size=5)
    
    def test_compare_groups(self, sample_data, privacy_guard):
        """Test group comparison."""
        tester = HypothesisTesting(privacy_guard=privacy_guard)
        
        results = tester.compare_groups(
            data=sample_data,
            outcome_vars=['continuous_outcome', 'binary_outcome'],
            group_var='group'
        )
        
        assert isinstance(results, dict)
        assert len(results) == 2
        assert all('p_value' in result for result in results.values())
        assert all('effect_size' in result for result in results.values())
    
    def test_multiple_comparisons_correction(self, sample_data, privacy_guard):
        """Test multiple comparisons correction."""
        tester = HypothesisTesting(privacy_guard=privacy_guard)
        
        p_values = [0.01, 0.05, 0.10, 0.20, 0.30]
        corrected = tester.correct_multiple_comparisons(p_values, method='fdr_bh')
        
        assert len(corrected) == len(p_values)
        assert isinstance(corrected, list)
        assert all(isinstance(p, float) for p in corrected)
    
    def test_chi_square_test(self, sample_data, privacy_guard):
        """Test chi-square test."""
        tester = HypothesisTesting(privacy_guard=privacy_guard)
        
        result = tester.chi_square_test(
            data=sample_data,
            var1='group',
            var2='categorical_var'
        )
        
        assert isinstance(result, dict)
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'effect_size' in result


class TestRegressionAnalysis:
    """Test cases for RegressionAnalysis class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for regression."""
        np.random.seed(42)
        n = 800
        
        # Create correlated predictors
        age = np.random.normal(65, 15, n)
        charlson = np.random.poisson(3, n)
        
        # Create outcome with realistic relationships
        linear_combination = (
            0.1 * age + 
            0.3 * charlson + 
            np.random.normal(0, 5, n)
        )
        
        binary_outcome = (linear_combination > np.median(linear_combination)).astype(int)
        continuous_outcome = linear_combination + np.random.normal(0, 2, n)
        
        data = {
            'age': age,
            'charlson_score': charlson,
            'sex': np.random.choice(['Male', 'Female'], n),
            'mechanical_ventilation': np.random.choice([0, 1], n),
            'continuous_outcome': continuous_outcome,
            'binary_outcome': binary_outcome,
            'time_to_event': np.random.exponential(10, n),
            'event_occurred': np.random.choice([0, 1], n, p=[0.7, 0.3])
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def privacy_guard(self):
        """Create privacy guard instance."""
        return PrivacyGuard(min_cell_size=10)
    
    def test_linear_regression(self, sample_data, privacy_guard):
        """Test linear regression analysis."""
        analyzer = RegressionAnalysis(privacy_guard=privacy_guard)
        
        results = analyzer.linear_regression(
            data=sample_data,
            outcome='continuous_outcome',
            predictors=['age', 'charlson_score', 'mechanical_ventilation'],
            validate_assumptions=True
        )
        
        assert isinstance(results, dict)
        assert 'coefficients' in results
        assert 'model_fit' in results
        assert 'assumptions_check' in results
        assert 'r_squared' in results['model_fit']
    
    def test_logistic_regression(self, sample_data, privacy_guard):
        """Test logistic regression analysis."""
        analyzer = RegressionAnalysis(privacy_guard=privacy_guard)
        
        results = analyzer.logistic_regression(
            data=sample_data,
            outcome='binary_outcome',
            predictors=['age', 'charlson_score', 'sex']
        )
        
        assert isinstance(results, dict)
        assert 'coefficients' in results
        assert 'odds_ratios' in results
        assert 'model_fit' in results
    
    def test_cox_regression(self, sample_data, privacy_guard):
        """Test Cox proportional hazards regression."""
        analyzer = RegressionAnalysis(privacy_guard=privacy_guard)
        
        results = analyzer.cox_regression(
            data=sample_data,
            time_col='time_to_event',
            event_col='event_occurred',
            predictors=['age', 'charlson_score']
        )
        
        assert isinstance(results, dict)
        assert 'coefficients' in results
        assert 'hazard_ratios' in results
    
    def test_variable_selection(self, sample_data, privacy_guard):
        """Test automatic variable selection."""
        analyzer = RegressionAnalysis(privacy_guard=privacy_guard)
        
        selected_vars = analyzer.stepwise_selection(
            data=sample_data,
            outcome='continuous_outcome',
            candidate_predictors=['age', 'charlson_score', 'mechanical_ventilation', 'sex'],
            method='forward'
        )
        
        assert isinstance(selected_vars, list)
        assert len(selected_vars) >= 1
        assert all(var in sample_data.columns for var in selected_vars)


if __name__ == "__main__":
    pytest.main([__file__])