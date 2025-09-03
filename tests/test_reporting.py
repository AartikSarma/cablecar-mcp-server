"""
Test Reporting Systems

Tests for STROBE and TRIPOD compliant reporting systems.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from cablecar_research.reporting.strobe_reporter import STROBEReporter
from cablecar_research.reporting.tripod_reporter import TRIPODReporter
from cablecar_research.reporting.visualizations import Visualizer
from cablecar_research.privacy.protection import PrivacyGuard


class TestSTROBEReporter:
    """Test cases for STROBE reporting compliance."""
    
    @pytest.fixture
    def sample_analysis_results(self):
        """Create sample analysis results for STROBE reporting."""
        return {
            'study_design': 'retrospective_cohort',
            'study_period': {'start': '2020-01-01', 'end': '2023-12-31'},
            'population': {
                'total_patients': 5000,
                'inclusion_criteria': ['Age >= 18', 'ICU admission'],
                'exclusion_criteria': ['Missing outcome data']
            },
            'demographics': {
                'age_mean': 65.5,
                'age_sd': 15.2,
                'sex_distribution': {'Male': 0.55, 'Female': 0.45},
                'mortality_rate': 0.18
            },
            'exposures': {
                'mechanical_ventilation': {'exposed': 1500, 'unexposed': 3500},
                'vasopressors': {'exposed': 2000, 'unexposed': 3000}
            },
            'outcomes': {
                'primary_outcome': 'in_hospital_mortality',
                'secondary_outcomes': ['length_of_stay', 'ventilator_free_days']
            },
            'statistical_methods': [
                'Chi-square test for categorical variables',
                'Mann-Whitney U test for continuous variables',
                'Logistic regression for mortality prediction'
            ],
            'main_results': {
                'mechanical_ventilation_mortality_or': 2.35,
                'mechanical_ventilation_ci': [1.85, 2.98],
                'p_value': 0.001
            },
            'limitations': [
                'Retrospective design',
                'Single-center study',
                'Potential confounding by indication'
            ]
        }
    
    @pytest.fixture
    def privacy_guard(self):
        """Create privacy guard instance."""
        return PrivacyGuard(min_cell_size=10)
    
    def test_strobe_checklist_validation(self, sample_analysis_results, privacy_guard):
        """Test STROBE checklist item validation."""
        reporter = STROBEReporter(privacy_guard=privacy_guard)
        
        checklist_compliance = reporter.validate_strobe_compliance(sample_analysis_results)
        
        assert isinstance(checklist_compliance, dict)
        assert 'total_items' in checklist_compliance
        assert 'completed_items' in checklist_compliance
        assert 'missing_items' in checklist_compliance
        assert 'compliance_score' in checklist_compliance
        
        # Should have 22 STROBE items
        assert checklist_compliance['total_items'] == 22
    
    def test_generate_strobe_report_html(self, sample_analysis_results, privacy_guard):
        """Test HTML STROBE report generation."""
        reporter = STROBEReporter(privacy_guard=privacy_guard)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'strobe_report.html')
            
            report_path = reporter.generate_report(
                analysis_results=sample_analysis_results,
                output_format='html',
                output_path=output_path
            )
            
            assert os.path.exists(report_path)
            
            # Check file content
            with open(report_path, 'r') as f:
                content = f.read()
            
            assert '<html>' in content
            assert 'STROBE' in content
            assert str(sample_analysis_results['population']['total_patients']) in content
    
    def test_generate_strobe_report_markdown(self, sample_analysis_results, privacy_guard):
        """Test Markdown STROBE report generation."""
        reporter = STROBEReporter(privacy_guard=privacy_guard)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'strobe_report.md')
            
            report_path = reporter.generate_report(
                analysis_results=sample_analysis_results,
                output_format='markdown',
                output_path=output_path
            )
            
            assert os.path.exists(report_path)
            
            # Check file content
            with open(report_path, 'r') as f:
                content = f.read()
            
            assert '# STROBE' in content
            assert '## Methods' in content
            assert '## Results' in content
    
    def test_strobe_section_generation(self, sample_analysis_results, privacy_guard):
        """Test individual STROBE section generation."""
        reporter = STROBEReporter(privacy_guard=privacy_guard)
        
        # Test title and abstract section
        title_section = reporter.generate_title_abstract_section(sample_analysis_results)
        assert isinstance(title_section, str)
        assert len(title_section) > 0
        
        # Test methods section
        methods_section = reporter.generate_methods_section(sample_analysis_results)
        assert isinstance(methods_section, str)
        assert 'retrospective_cohort' in methods_section.lower()
        
        # Test results section
        results_section = reporter.generate_results_section(sample_analysis_results)
        assert isinstance(results_section, str)
        assert str(sample_analysis_results['population']['total_patients']) in results_section


class TestTRIPODReporter:
    """Test cases for TRIPOD reporting compliance."""
    
    @pytest.fixture
    def sample_model_results(self):
        """Create sample ML model results for TRIPOD reporting."""
        return {
            'model_type': 'prediction_model',
            'model_development': {
                'algorithm': 'random_forest',
                'training_data_size': 4000,
                'validation_data_size': 1000,
                'features': ['age', 'sex', 'charlson_score', 'mechanical_ventilation'],
                'outcome': 'mortality'
            },
            'model_performance': {
                'auc': 0.78,
                'auc_ci': [0.74, 0.82],
                'sensitivity': 0.72,
                'specificity': 0.75,
                'ppv': 0.45,
                'npv': 0.91,
                'calibration_slope': 0.95,
                'calibration_intercept': 0.12,
                'brier_score': 0.15
            },
            'feature_importance': {
                'age': 0.35,
                'charlson_score': 0.28,
                'mechanical_ventilation': 0.22,
                'sex': 0.15
            },
            'model_interpretation': {
                'shap_available': True,
                'feature_interactions': True,
                'partial_dependence_plots': True
            },
            'validation_strategy': {
                'internal_validation': 'cross_validation',
                'cv_folds': 5,
                'temporal_validation': True,
                'external_validation': False
            },
            'clinical_utility': {
                'decision_curve_analysis': True,
                'net_benefit_threshold': 0.15,
                'implementation_considerations': [
                    'Real-time prediction capability',
                    'Integration with EHR systems'
                ]
            }
        }
    
    @pytest.fixture
    def privacy_guard(self):
        """Create privacy guard instance."""
        return PrivacyGuard(min_cell_size=10)
    
    def test_tripod_checklist_validation(self, sample_model_results, privacy_guard):
        """Test TRIPOD checklist item validation."""
        reporter = TRIPODReporter(privacy_guard=privacy_guard)
        
        checklist_compliance = reporter.validate_tripod_compliance(sample_model_results)
        
        assert isinstance(checklist_compliance, dict)
        assert 'total_items' in checklist_compliance
        assert 'completed_items' in checklist_compliance
        assert 'missing_items' in checklist_compliance
        assert 'compliance_score' in checklist_compliance
        
        # Should have 27 TRIPOD items
        assert checklist_compliance['total_items'] == 27
    
    def test_generate_tripod_report(self, sample_model_results, privacy_guard):
        """Test TRIPOD report generation."""
        reporter = TRIPODReporter(privacy_guard=privacy_guard)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'tripod_report.html')
            
            report_path = reporter.generate_report(
                model_results=sample_model_results,
                output_format='html',
                output_path=output_path
            )
            
            assert os.path.exists(report_path)
            
            # Check file content
            with open(report_path, 'r') as f:
                content = f.read()
            
            assert '<html>' in content
            assert 'TRIPOD' in content
            assert 'random_forest' in content
            assert str(sample_model_results['model_performance']['auc']) in content
    
    def test_model_performance_section(self, sample_model_results, privacy_guard):
        """Test model performance section generation."""
        reporter = TRIPODReporter(privacy_guard=privacy_guard)
        
        performance_section = reporter.generate_performance_section(sample_model_results)
        
        assert isinstance(performance_section, str)
        assert 'AUC' in performance_section
        assert str(sample_model_results['model_performance']['auc']) in performance_section
        assert 'calibration' in performance_section.lower()
    
    def test_model_development_section(self, sample_model_results, privacy_guard):
        """Test model development section generation."""
        reporter = TRIPODReporter(privacy_guard=privacy_guard)
        
        development_section = reporter.generate_development_section(sample_model_results)
        
        assert isinstance(development_section, str)
        assert 'random_forest' in development_section
        assert str(sample_model_results['model_development']['training_data_size']) in development_section
    
    def test_ai_specific_reporting(self, sample_model_results, privacy_guard):
        """Test AI-specific TRIPOD+AI reporting elements."""
        reporter = TRIPODReporter(privacy_guard=privacy_guard)
        
        ai_section = reporter.generate_ai_specific_section(sample_model_results)
        
        assert isinstance(ai_section, str)
        assert 'interpretability' in ai_section.lower() or 'explainability' in ai_section.lower()
        assert 'shap' in ai_section.lower()


class TestVisualizer:
    """Test cases for visualization components."""
    
    @pytest.fixture
    def sample_ml_results(self):
        """Create sample ML results for visualization."""
        np.random.seed(42)
        n = 1000
        
        y_true = np.random.choice([0, 1], n, p=[0.7, 0.3])
        y_pred_proba = np.random.beta(2, 5, n)
        
        # Make predictions somewhat realistic
        y_pred_proba = np.where(y_true == 1, 
                               y_pred_proba + 0.3, 
                               y_pred_proba - 0.1)
        y_pred_proba = np.clip(y_pred_proba, 0, 1)
        
        return {
            'y_true': y_true,
            'y_pred_proba': y_pred_proba,
            'durations': np.random.exponential(10, n),
            'events': np.random.choice([0, 1], n, p=[0.6, 0.4])
        }
    
    @pytest.fixture
    def privacy_guard(self):
        """Create privacy guard instance."""
        return PrivacyGuard(min_cell_size=10)
    
    def test_create_roc_curve(self, sample_ml_results, privacy_guard):
        """Test ROC curve creation."""
        visualizer = Visualizer(privacy_guard=privacy_guard)
        
        fig = visualizer.create_roc_curve(
            y_true=sample_ml_results['y_true'],
            y_pred_proba=sample_ml_results['y_pred_proba'],
            title='Test ROC Curve'
        )
        
        assert fig is not None
        assert hasattr(fig, 'axes')
        assert len(fig.axes) == 1
    
    def test_create_calibration_plot(self, sample_ml_results, privacy_guard):
        """Test calibration plot creation."""
        visualizer = Visualizer(privacy_guard=privacy_guard)
        
        fig = visualizer.create_calibration_plot(
            y_true=sample_ml_results['y_true'],
            y_pred_proba=sample_ml_results['y_pred_proba'],
            n_bins=10,
            title='Test Calibration Plot'
        )
        
        assert fig is not None
        assert hasattr(fig, 'axes')
        assert len(fig.axes) == 1
    
    def test_create_forest_plot(self, privacy_guard):
        """Test forest plot creation."""
        visualizer = Visualizer(privacy_guard=privacy_guard)
        
        effect_data = {
            'studies': ['Study 1', 'Study 2', 'Study 3'],
            'effects': [1.2, 0.8, 1.5],
            'ci_lower': [0.9, 0.6, 1.1],
            'ci_upper': [1.6, 1.1, 2.0]
        }
        
        fig = visualizer.create_forest_plot(
            effect_data=effect_data,
            title='Test Forest Plot'
        )
        
        assert fig is not None
        assert hasattr(fig, 'axes')
        assert len(fig.axes) == 1
    
    def test_create_kaplan_meier_plot(self, sample_ml_results, privacy_guard):
        """Test Kaplan-Meier survival plot creation."""
        visualizer = Visualizer(privacy_guard=privacy_guard)
        
        # Create groups for comparison
        groups = np.random.choice(['Group A', 'Group B'], len(sample_ml_results['durations']))
        
        fig = visualizer.create_kaplan_meier_plot(
            durations=sample_ml_results['durations'],
            events=sample_ml_results['events'],
            groups=groups,
            title='Test Survival Curve'
        )
        
        assert fig is not None
        assert hasattr(fig, 'axes')
        assert len(fig.axes) == 1
    
    def test_create_box_plot(self, privacy_guard):
        """Test box plot creation."""
        visualizer = Visualizer(privacy_guard=privacy_guard)
        
        data = {
            'Group A': np.random.normal(50, 10, 100),
            'Group B': np.random.normal(55, 12, 100),
            'Group C': np.random.normal(48, 8, 100)
        }
        
        fig = visualizer.create_box_plot(
            data=data,
            title='Test Box Plot'
        )
        
        assert fig is not None
        assert hasattr(fig, 'axes')
        assert len(fig.axes) == 1
    
    def test_create_consort_diagram(self, privacy_guard):
        """Test CONSORT flow diagram creation."""
        visualizer = Visualizer(privacy_guard=privacy_guard)
        
        flow_data = {
            'assessed': 1500,
            'randomized': 1200,
            'intervention': 600,
            'control': 600,
            'analyzed': 580
        }
        
        fig = visualizer.create_consort_diagram(
            flow_data=flow_data,
            title='Test CONSORT Diagram'
        )
        
        assert fig is not None
        assert hasattr(fig, 'axes')
        assert len(fig.axes) == 1
    
    def test_save_figures(self, privacy_guard):
        """Test figure saving functionality."""
        visualizer = Visualizer(privacy_guard=privacy_guard)
        
        # Create a simple figure
        data = {'Group A': np.random.normal(0, 1, 100)}
        fig = visualizer.create_box_plot(data, title='Test')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_files = visualizer.save_all_figures(
                output_dir=tmpdir,
                formats=['png', 'pdf']
            )
            
            assert isinstance(saved_files, dict)
            assert len(saved_files) >= 1
            
            # Check that files were actually created
            for figure_id, file_paths in saved_files.items():
                for file_path in file_paths:
                    assert os.path.exists(file_path)
    
    def test_create_summary_figure(self, privacy_guard):
        """Test multi-panel summary figure creation."""
        visualizer = Visualizer(privacy_guard=privacy_guard)
        
        analysis_results = {
            'sample_size': 1000,
            'primary_outcome': 'mortality',
            'model_performance': {'auc': 0.75}
        }
        
        fig = visualizer.create_summary_figure(analysis_results)
        
        assert fig is not None
        assert hasattr(fig, 'axes')
        assert len(fig.axes) == 4  # 2x2 subplot layout


if __name__ == "__main__":
    pytest.main([__file__])