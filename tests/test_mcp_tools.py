"""
Test MCP Server Tools

Tests for MCP server tool implementations and integration.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Import MCP server tools
from server.tools.dataset_importer import import_clinical_dataset
from server.tools.study_designer import design_clinical_study
from server.tools.table1_generator import generate_descriptive_table
from server.tools.hypothesis_tester import test_clinical_hypotheses
from server.tools.model_builder import build_prediction_model
from server.tools.code_exporter import export_analysis_code
from server.tools.strobe_generator import generate_strobe_report
from server.tools.tripod_generator import generate_tripod_report


class TestDatasetImporter:
    """Test cases for dataset import tool."""
    
    @pytest.fixture
    def sample_csv_files(self):
        """Create sample CSV files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create patient data
            patient_data = pd.DataFrame({
                'patient_id': ['PT000001', 'PT000002', 'PT000003'],
                'age': [65, 45, 78],
                'sex': ['Male', 'Female', 'Male'],
                'admission_date': ['2023-01-01', '2023-01-02', '2023-01-03']
            })
            patient_path = Path(tmpdir) / 'patient.csv'
            patient_data.to_csv(patient_path, index=False)
            
            # Create hospitalization data
            hosp_data = pd.DataFrame({
                'hospitalization_id': ['H00000001', 'H00000002', 'H00000003'],
                'patient_id': ['PT000001', 'PT000002', 'PT000003'],
                'admission_dttm': ['2023-01-01 10:00:00', '2023-01-02 14:30:00', '2023-01-03 08:15:00'],
                'discharge_dttm': ['2023-01-05 16:00:00', '2023-01-08 12:00:00', '2023-01-10 09:30:00'],
                'mortality': [0, 1, 0]
            })
            hosp_path = Path(tmpdir) / 'hospitalization.csv'
            hosp_data.to_csv(hosp_path, index=False)
            
            yield tmpdir
    
    @pytest.mark.asyncio
    async def test_import_clinical_dataset(self, sample_csv_files):
        """Test clinical dataset import functionality."""
        result = await import_clinical_dataset([{
            "name": "data_path",
            "value": sample_csv_files
        }, {
            "name": "tables",
            "value": ["patient", "hospitalization"]
        }])
        
        assert isinstance(result, list)
        assert len(result) == 1
        
        response = result[0]
        assert response["type"] == "text"
        assert "successfully imported" in response["text"].lower()
    
    @pytest.mark.asyncio
    async def test_import_with_schema_validation(self, sample_csv_files):
        """Test import with schema validation."""
        result = await import_clinical_dataset([{
            "name": "data_path",
            "value": sample_csv_files
        }, {
            "name": "validate_schema",
            "value": True
        }])
        
        assert isinstance(result, list)
        response = result[0]
        assert "schema validation" in response["text"].lower()


class TestStudyDesigner:
    """Test cases for study design tool."""
    
    @pytest.mark.asyncio
    async def test_design_clinical_study(self):
        """Test clinical study design functionality."""
        research_question = "Does mechanical ventilation improve outcomes in ARDS patients?"
        
        result = await design_clinical_study([{
            "name": "research_question",
            "value": research_question
        }])
        
        assert isinstance(result, list)
        assert len(result) == 1
        
        response = result[0]
        assert response["type"] == "text"
        assert "study design" in response["text"].lower()
        assert any(keyword in response["text"].lower() for keyword in 
                  ["cohort", "case-control", "randomized", "observational"])
    
    @pytest.mark.asyncio
    async def test_study_design_with_parameters(self):
        """Test study design with specific parameters."""
        result = await design_clinical_study([{
            "name": "research_question",
            "value": "Effect of early antibiotics on sepsis mortality"
        }, {
            "name": "study_type",
            "value": "retrospective_cohort"
        }, {
            "name": "primary_outcome",
            "value": "in_hospital_mortality"
        }])
        
        response = result[0]
        assert "retrospective" in response["text"].lower()
        assert "mortality" in response["text"].lower()


class TestTable1Generator:
    """Test cases for Table 1 generation tool."""
    
    @pytest.fixture
    def mock_dataset(self):
        """Mock dataset for testing."""
        np.random.seed(42)
        return {
            'patient': pd.DataFrame({
                'patient_id': [f'PT{i:06d}' for i in range(1, 101)],
                'age': np.random.normal(65, 15, 100),
                'sex': np.random.choice(['Male', 'Female'], 100),
                'mortality': np.random.choice([0, 1], 100, p=[0.8, 0.2])
            })
        }
    
    @pytest.mark.asyncio
    @patch('server.tools.table1_generator.get_current_dataset')
    async def test_generate_descriptive_table(self, mock_get_dataset, mock_dataset):
        """Test descriptive table generation."""
        mock_get_dataset.return_value = mock_dataset
        
        result = await generate_descriptive_table([{
            "name": "variables",
            "value": ["age", "sex"]
        }])
        
        assert isinstance(result, list)
        response = result[0]
        assert response["type"] == "text"
        assert "table 1" in response["text"].lower()
    
    @pytest.mark.asyncio
    @patch('server.tools.table1_generator.get_current_dataset')
    async def test_stratified_table1(self, mock_get_dataset, mock_dataset):
        """Test stratified Table 1 generation."""
        mock_get_dataset.return_value = mock_dataset
        
        result = await generate_descriptive_table([{
            "name": "variables",
            "value": ["age", "sex"]
        }, {
            "name": "stratify_by",
            "value": "mortality"
        }])
        
        response = result[0]
        assert "stratified" in response["text"].lower() or "mortality" in response["text"].lower()


class TestHypothesisTester:
    """Test cases for hypothesis testing tool."""
    
    @pytest.fixture
    def mock_dataset(self):
        """Mock dataset for testing."""
        np.random.seed(42)
        return {
            'patient': pd.DataFrame({
                'patient_id': [f'PT{i:06d}' for i in range(1, 201)],
                'age': np.random.normal(65, 15, 200),
                'sex': np.random.choice(['Male', 'Female'], 200),
                'treatment': np.random.choice(['Control', 'Treatment'], 200),
                'outcome': np.random.choice([0, 1], 200),
                'los_days': np.random.exponential(7, 200)
            })
        }
    
    @pytest.mark.asyncio
    @patch('server.tools.hypothesis_tester.get_current_dataset')
    async def test_clinical_hypothesis_testing(self, mock_get_dataset, mock_dataset):
        """Test clinical hypothesis testing."""
        mock_get_dataset.return_value = mock_dataset
        
        result = await test_clinical_hypotheses([{
            "name": "outcome_variables",
            "value": ["outcome", "los_days"]
        }, {
            "name": "group_variable",
            "value": "treatment"
        }])
        
        assert isinstance(result, list)
        response = result[0]
        assert response["type"] == "text"
        assert "p-value" in response["text"].lower() or "statistical" in response["text"].lower()
    
    @pytest.mark.asyncio
    @patch('server.tools.hypothesis_tester.get_current_dataset')
    async def test_multiple_comparisons(self, mock_get_dataset, mock_dataset):
        """Test multiple comparisons correction."""
        mock_get_dataset.return_value = mock_dataset
        
        result = await test_clinical_hypotheses([{
            "name": "outcome_variables",
            "value": ["outcome", "los_days", "age"]
        }, {
            "name": "group_variable",
            "value": "treatment"
        }, {
            "name": "correction_method",
            "value": "fdr_bh"
        }])
        
        response = result[0]
        assert "correction" in response["text"].lower() or "fdr" in response["text"].lower()


class TestModelBuilder:
    """Test cases for prediction model building tool."""
    
    @pytest.fixture
    def mock_dataset(self):
        """Mock dataset for testing."""
        np.random.seed(42)
        return {
            'patient': pd.DataFrame({
                'patient_id': [f'PT{i:06d}' for i in range(1, 501)],
                'age': np.random.normal(65, 15, 500),
                'sex': np.random.choice(['Male', 'Female'], 500),
                'charlson_score': np.random.poisson(3, 500),
                'mechanical_ventilation': np.random.choice([0, 1], 500),
                'mortality': np.random.choice([0, 1], 500, p=[0.7, 0.3])
            })
        }
    
    @pytest.mark.asyncio
    @patch('server.tools.model_builder.get_current_dataset')
    async def test_build_prediction_model(self, mock_get_dataset, mock_dataset):
        """Test prediction model building."""
        mock_get_dataset.return_value = mock_dataset
        
        result = await build_prediction_model([{
            "name": "outcome",
            "value": "mortality"
        }, {
            "name": "predictors",
            "value": ["age", "charlson_score", "mechanical_ventilation"]
        }, {
            "name": "model_type",
            "value": "auto_ml"
        }])
        
        assert isinstance(result, list)
        response = result[0]
        assert response["type"] == "text"
        assert any(keyword in response["text"].lower() for keyword in 
                  ["auc", "accuracy", "model", "performance"])
    
    @pytest.mark.asyncio
    @patch('server.tools.model_builder.get_current_dataset')
    async def test_model_validation(self, mock_get_dataset, mock_dataset):
        """Test model validation features."""
        mock_get_dataset.return_value = mock_dataset
        
        result = await build_prediction_model([{
            "name": "outcome",
            "value": "mortality"
        }, {
            "name": "predictors",
            "value": ["age", "charlson_score"]
        }, {
            "name": "validation_strategy",
            "value": "cross_validation"
        }])
        
        response = result[0]
        assert "validation" in response["text"].lower()


class TestCodeExporter:
    """Test cases for code export tool."""
    
    @pytest.fixture
    def mock_analysis_history(self):
        """Mock analysis history for testing."""
        return [
            {
                'tool_name': 'import_dataset',
                'parameters': {'data_path': '/data', 'tables': ['patient']},
                'timestamp': '2023-01-01T10:00:00'
            },
            {
                'tool_name': 'generate_table1',
                'parameters': {'variables': ['age', 'sex']},
                'timestamp': '2023-01-01T10:05:00'
            },
            {
                'tool_name': 'build_model',
                'parameters': {'outcome': 'mortality', 'predictors': ['age']},
                'timestamp': '2023-01-01T10:10:00'
            }
        ]
    
    @pytest.mark.asyncio
    @patch('server.tools.code_exporter.get_analysis_history')
    async def test_export_python_code(self, mock_get_history, mock_analysis_history):
        """Test Python code export."""
        mock_get_history.return_value = mock_analysis_history
        
        result = await export_analysis_code([{
            "name": "language",
            "value": "python"
        }])
        
        assert isinstance(result, list)
        response = result[0]
        assert response["type"] == "text"
        assert "import" in response["text"]
        assert "def" in response["text"] or "class" in response["text"]
    
    @pytest.mark.asyncio
    @patch('server.tools.code_exporter.get_analysis_history')
    async def test_export_r_code(self, mock_get_history, mock_analysis_history):
        """Test R code export."""
        mock_get_history.return_value = mock_analysis_history
        
        result = await export_analysis_code([{
            "name": "language",
            "value": "R"
        }])
        
        response = result[0]
        assert "library(" in response["text"] or "<-" in response["text"]


class TestSTROBEGenerator:
    """Test cases for STROBE report generation tool."""
    
    @pytest.fixture
    def mock_analysis_results(self):
        """Mock analysis results for STROBE reporting."""
        return {
            'study_design': 'retrospective_cohort',
            'population': {'total_patients': 1000},
            'outcomes': {'primary_outcome': 'mortality'},
            'main_results': {'mortality_rate': 0.15}
        }
    
    @pytest.mark.asyncio
    @patch('server.tools.strobe_generator.get_analysis_results')
    async def test_generate_strobe_report(self, mock_get_results, mock_analysis_results):
        """Test STROBE report generation."""
        mock_get_results.return_value = mock_analysis_results
        
        result = await generate_strobe_report([{
            "name": "output_format",
            "value": "html"
        }])
        
        assert isinstance(result, list)
        response = result[0]
        assert response["type"] == "text"
        assert "strobe" in response["text"].lower()
    
    @pytest.mark.asyncio
    @patch('server.tools.strobe_generator.get_analysis_results')
    async def test_strobe_checklist_compliance(self, mock_get_results, mock_analysis_results):
        """Test STROBE checklist compliance checking."""
        mock_get_results.return_value = mock_analysis_results
        
        result = await generate_strobe_report([{
            "name": "include_checklist",
            "value": True
        }])
        
        response = result[0]
        assert "checklist" in response["text"].lower() or "compliance" in response["text"].lower()


class TestTRIPODGenerator:
    """Test cases for TRIPOD report generation tool."""
    
    @pytest.fixture
    def mock_model_results(self):
        """Mock model results for TRIPOD reporting."""
        return {
            'model_type': 'prediction_model',
            'model_performance': {'auc': 0.78, 'sensitivity': 0.72},
            'validation_strategy': {'internal_validation': 'cross_validation'},
            'feature_importance': {'age': 0.35, 'sex': 0.25}
        }
    
    @pytest.mark.asyncio
    @patch('server.tools.tripod_generator.get_model_results')
    async def test_generate_tripod_report(self, mock_get_results, mock_model_results):
        """Test TRIPOD report generation."""
        mock_get_results.return_value = mock_model_results
        
        result = await generate_tripod_report([{
            "name": "output_format",
            "value": "html"
        }])
        
        assert isinstance(result, list)
        response = result[0]
        assert response["type"] == "text"
        assert "tripod" in response["text"].lower()
    
    @pytest.mark.asyncio
    @patch('server.tools.tripod_generator.get_model_results')
    async def test_tripod_ai_compliance(self, mock_get_results, mock_model_results):
        """Test TRIPOD+AI compliance features."""
        mock_get_results.return_value = mock_model_results
        
        result = await generate_tripod_report([{
            "name": "include_ai_elements",
            "value": True
        }])
        
        response = result[0]
        assert "ai" in response["text"].lower() or "interpretability" in response["text"].lower()


class TestMCPIntegration:
    """Test cases for overall MCP integration."""
    
    def test_tool_parameter_validation(self):
        """Test that all tools handle parameter validation correctly."""
        # This would test parameter validation across all tools
        # Implementation would depend on specific parameter validation logic
        pass
    
    def test_privacy_protection_integration(self):
        """Test that privacy protection is properly integrated across tools."""
        # This would test that privacy guards are properly applied
        pass
    
    def test_error_handling(self):
        """Test error handling across MCP tools."""
        # This would test error handling and graceful degradation
        pass


if __name__ == "__main__":
    pytest.main([__file__])