"""
Test ML Models and AutoML

Tests for machine learning components including AutoML functionality.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from cablecar_research.analysis.ml_models import MLAnalyzer
from cablecar_research.privacy.protection import PrivacyGuard


class TestMLAnalyzer:
    """Test cases for MLAnalyzer class."""
    
    @pytest.fixture
    def binary_classification_data(self):
        """Create binary classification dataset."""
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_clusters_per_class=1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['outcome'] = y
        df['patient_id'] = [f'PT{i:06d}' for i in range(1, len(df)+1)]
        
        return df
    
    @pytest.fixture
    def multiclass_data(self):
        """Create multiclass classification dataset."""
        X, y = make_classification(
            n_samples=800,
            n_features=8,
            n_informative=4,
            n_redundant=2,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['outcome'] = y
        df['patient_id'] = [f'PT{i:06d}' for i in range(1, len(df)+1)]
        
        return df
    
    @pytest.fixture
    def regression_data(self):
        """Create regression dataset."""
        X, y = make_regression(
            n_samples=1000,
            n_features=8,
            n_informative=5,
            noise=0.1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['outcome'] = y
        df['patient_id'] = [f'PT{i:06d}' for i in range(1, len(df)+1)]
        
        return df
    
    @pytest.fixture
    def survival_data(self):
        """Create survival analysis dataset."""
        np.random.seed(42)
        n = 600
        
        # Create features
        X = np.random.randn(n, 5)
        
        # Create survival times with some relationship to features
        linear_pred = X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2
        survival_times = np.random.exponential(np.exp(-linear_pred) * 10)
        
        # Create censoring
        censoring_times = np.random.exponential(15, n)
        observed_times = np.minimum(survival_times, censoring_times)
        events = survival_times <= censoring_times
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['time'] = observed_times
        df['event'] = events.astype(int)
        df['patient_id'] = [f'PT{i:06d}' for i in range(1, len(df)+1)]
        
        return df
    
    @pytest.fixture
    def privacy_guard(self):
        """Create privacy guard instance."""
        return PrivacyGuard(min_cell_size=10)
    
    def test_auto_ml_binary_classification(self, binary_classification_data, privacy_guard):
        """Test AutoML for binary classification."""
        analyzer = MLAnalyzer(privacy_guard=privacy_guard)
        
        feature_cols = [col for col in binary_classification_data.columns 
                       if col not in ['outcome', 'patient_id']]
        
        results = analyzer.auto_ml(
            data=binary_classification_data,
            outcome='outcome',
            features=feature_cols,
            problem_type='classification',
            cv_folds=3,
            hyperparameter_tuning=False  # Disable for faster testing
        )
        
        assert isinstance(results, dict)
        assert 'best_model' in results
        assert 'model_comparison' in results
        assert 'feature_importance' in results
        assert 'performance_metrics' in results
        
        # Check TRIPOD compliance
        assert 'tripod_compliance' in results
        assert isinstance(results['tripod_compliance'], dict)
    
    def test_auto_ml_multiclass(self, multiclass_data, privacy_guard):
        """Test AutoML for multiclass classification."""
        analyzer = MLAnalyzer(privacy_guard=privacy_guard)
        
        feature_cols = [col for col in multiclass_data.columns 
                       if col not in ['outcome', 'patient_id']]
        
        results = analyzer.auto_ml(
            data=multiclass_data,
            outcome='outcome',
            features=feature_cols,
            problem_type='classification',
            cv_folds=3,
            hyperparameter_tuning=False
        )
        
        assert isinstance(results, dict)
        assert 'best_model' in results
        assert len(np.unique(multiclass_data['outcome'])) == 3
        
        # Check that multiclass metrics are included
        assert 'performance_metrics' in results
    
    def test_auto_ml_regression(self, regression_data, privacy_guard):
        """Test AutoML for regression."""
        analyzer = MLAnalyzer(privacy_guard=privacy_guard)
        
        feature_cols = [col for col in regression_data.columns 
                       if col not in ['outcome', 'patient_id']]
        
        results = analyzer.auto_ml(
            data=regression_data,
            outcome='outcome',
            features=feature_cols,
            problem_type='regression',
            cv_folds=3,
            hyperparameter_tuning=False
        )
        
        assert isinstance(results, dict)
        assert 'best_model' in results
        assert 'model_comparison' in results
        assert 'performance_metrics' in results
        
        # Check regression-specific metrics
        metrics = results['performance_metrics']
        assert 'r2' in metrics or 'mae' in metrics or 'rmse' in metrics
    
    def test_survival_analysis(self, survival_data, privacy_guard):
        """Test survival analysis models."""
        analyzer = MLAnalyzer(privacy_guard=privacy_guard)
        
        feature_cols = [col for col in survival_data.columns 
                       if col not in ['time', 'event', 'patient_id']]
        
        results = analyzer.survival_analysis(
            data=survival_data,
            time_col='time',
            event_col='event',
            features=feature_cols,
            model_type='cox'
        )
        
        assert isinstance(results, dict)
        assert 'model_summary' in results
        assert 'hazard_ratios' in results
        assert 'concordance_index' in results
    
    def test_feature_selection(self, binary_classification_data, privacy_guard):
        """Test feature selection methods."""
        analyzer = MLAnalyzer(privacy_guard=privacy_guard)
        
        feature_cols = [col for col in binary_classification_data.columns 
                       if col not in ['outcome', 'patient_id']]
        
        selected_features = analyzer.feature_selection(
            data=binary_classification_data,
            outcome='outcome',
            features=feature_cols,
            method='univariate',
            k=5
        )
        
        assert isinstance(selected_features, list)
        assert len(selected_features) <= 5
        assert all(feat in feature_cols for feat in selected_features)
    
    def test_model_interpretability(self, binary_classification_data, privacy_guard):
        """Test model interpretability features."""
        analyzer = MLAnalyzer(privacy_guard=privacy_guard)
        
        feature_cols = [col for col in binary_classification_data.columns 
                       if col not in ['outcome', 'patient_id']]
        
        # First run AutoML to get a model
        results = analyzer.auto_ml(
            data=binary_classification_data,
            outcome='outcome',
            features=feature_cols,
            problem_type='classification',
            cv_folds=3,
            hyperparameter_tuning=False
        )
        
        # Test interpretability
        interpretability = analyzer.explain_model(
            results['best_model'],
            binary_classification_data[feature_cols],
            method='shap'
        )
        
        assert isinstance(interpretability, dict)
        assert 'feature_importance' in interpretability
        assert 'shap_values' in interpretability or 'explanation_available' in interpretability
    
    def test_model_validation(self, binary_classification_data, privacy_guard):
        """Test comprehensive model validation."""
        analyzer = MLAnalyzer(privacy_guard=privacy_guard)
        
        feature_cols = [col for col in binary_classification_data.columns 
                       if col not in ['outcome', 'patient_id']]
        
        # Split data for validation
        train_data = binary_classification_data.sample(n=800, random_state=42)
        test_data = binary_classification_data.drop(train_data.index)
        
        validation_results = analyzer.validate_model(
            train_data=train_data,
            test_data=test_data,
            outcome='outcome',
            features=feature_cols,
            model_type='random_forest'
        )
        
        assert isinstance(validation_results, dict)
        assert 'train_performance' in validation_results
        assert 'test_performance' in validation_results
        assert 'overfitting_assessment' in validation_results
    
    def test_ensemble_methods(self, binary_classification_data, privacy_guard):
        """Test ensemble modeling approaches."""
        analyzer = MLAnalyzer(privacy_guard=privacy_guard)
        
        feature_cols = [col for col in binary_classification_data.columns 
                       if col not in ['outcome', 'patient_id']]
        
        ensemble_results = analyzer.create_ensemble(
            data=binary_classification_data,
            outcome='outcome',
            features=feature_cols,
            base_models=['random_forest', 'logistic_regression', 'gradient_boosting'],
            ensemble_method='voting'
        )
        
        assert isinstance(ensemble_results, dict)
        assert 'ensemble_model' in ensemble_results
        assert 'base_model_performance' in ensemble_results
        assert 'ensemble_performance' in ensemble_results
    
    def test_hyperparameter_optimization(self, binary_classification_data, privacy_guard):
        """Test hyperparameter optimization."""
        analyzer = MLAnalyzer(privacy_guard=privacy_guard)
        
        feature_cols = [col for col in binary_classification_data.columns 
                       if col not in ['outcome', 'patient_id']]
        
        # Use small dataset and limited trials for faster testing
        small_data = binary_classification_data.sample(n=200, random_state=42)
        
        hp_results = analyzer.optimize_hyperparameters(
            data=small_data,
            outcome='outcome',
            features=feature_cols,
            model_type='random_forest',
            n_trials=5,  # Small number for testing
            cv_folds=3
        )
        
        assert isinstance(hp_results, dict)
        assert 'best_params' in hp_results
        assert 'best_score' in hp_results
        assert 'optimization_history' in hp_results


if __name__ == "__main__":
    pytest.main([__file__])