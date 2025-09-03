"""
Base Analysis Interface

Abstract base class for all CableCar analysis plugins, ensuring consistent
interface and enabling dynamic discovery and loading.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class AnalysisType(Enum):
    """Types of clinical research analyses."""
    DESCRIPTIVE = "descriptive"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    REGRESSION = "regression"
    MACHINE_LEARNING = "machine_learning"
    SURVIVAL_ANALYSIS = "survival_analysis"
    CAUSAL_INFERENCE = "causal_inference"
    TIME_SERIES = "time_series"
    NETWORK_ANALYSIS = "network_analysis"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    CUSTOM = "custom"


class ValidationLevel(Enum):
    """Validation requirements for analyses."""
    MINIMAL = "minimal"        # Basic input checking
    STANDARD = "standard"      # Standard clinical research validation
    STRICT = "strict"          # Comprehensive validation with assumptions
    CUSTOM = "custom"          # Custom validation logic


@dataclass
class AnalysisMetadata:
    """Metadata for analysis plugins."""
    name: str
    display_name: str
    description: str
    version: str
    author: str
    email: Optional[str] = None
    analysis_type: AnalysisType = AnalysisType.CUSTOM
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    citation: Optional[str] = None
    doi: Optional[str] = None
    requirements: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'version': self.version,
            'author': self.author,
            'email': self.email,
            'analysis_type': self.analysis_type.value,
            'validation_level': self.validation_level.value,
            'citation': self.citation,
            'doi': self.doi,
            'requirements': self.requirements or [],
            'keywords': self.keywords or []
        }


class BaseAnalysis(ABC):
    """
    Abstract base class for all CableCar analysis plugins.
    
    This class defines the standard interface that all analysis plugins
    must implement to be compatible with the CableCar system.
    """
    
    # Plugin metadata - must be defined by subclasses
    metadata: AnalysisMetadata = None
    
    def __init__(self, df: pd.DataFrame, privacy_guard=None, **kwargs):
        """
        Initialize analysis with data and privacy protection.
        
        Args:
            df: Input DataFrame containing clinical data
            privacy_guard: Privacy protection instance
            **kwargs: Additional configuration parameters
        """
        self.df = df.copy() if df is not None else None
        self.privacy_guard = privacy_guard
        self.config = kwargs
        self.results = {}
        self._validated = False
        
        # Validate that metadata is properly defined
        if self.metadata is None:
            raise ValueError(f"{self.__class__.__name__} must define metadata")
    
    @abstractmethod
    def validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """
        Validate input data and parameters.
        
        Args:
            **kwargs: Analysis-specific parameters
            
        Returns:
            Dictionary with validation results:
            - 'valid': bool - whether inputs are valid
            - 'errors': List[str] - validation error messages
            - 'warnings': List[str] - validation warnings
            - 'suggestions': List[str] - suggestions for improvement
        """
        pass
    
    @abstractmethod
    def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the analysis.
        
        Args:
            **kwargs: Analysis-specific parameters
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    @abstractmethod
    def format_results(self, results: Dict[str, Any], format_type: str = "standard") -> str:
        """
        Format analysis results for display.
        
        Args:
            results: Raw analysis results
            format_type: Format type ("standard", "detailed", "summary", "publication")
            
        Returns:
            Formatted string representation of results
        """
        pass
    
    def get_required_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get required and optional parameters for this analysis.
        
        Returns:
            Dictionary with parameter specifications:
            {
                'required': {
                    'param_name': {
                        'type': 'string|number|boolean|list',
                        'description': 'Parameter description',
                        'example': 'Example value'
                    }
                },
                'optional': { ... }
            }
        """
        return {'required': {}, 'optional': {}}
    
    def check_data_requirements(self) -> Dict[str, Any]:
        """
        Check if the loaded data meets requirements for this analysis.
        
        Returns:
            Dictionary with data requirement check results
        """
        if self.df is None:
            return {
                'meets_requirements': False,
                'errors': ['No data loaded'],
                'warnings': [],
                'suggestions': ['Load data before running analysis']
            }
        
        return {
            'meets_requirements': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
    
    def get_citation(self) -> str:
        """Get citation information for this analysis."""
        if self.metadata.citation:
            return self.metadata.citation
        
        citation_parts = [self.metadata.author]
        if self.metadata.display_name:
            citation_parts.append(f'"{self.metadata.display_name}"')
        citation_parts.append(f"CableCar Analysis Plugin v{self.metadata.version}")
        
        if self.metadata.doi:
            citation_parts.append(f"DOI: {self.metadata.doi}")
        
        return ". ".join(citation_parts)
    
    def get_documentation(self) -> Dict[str, str]:
        """
        Get comprehensive documentation for this analysis.
        
        Returns:
            Dictionary with documentation sections
        """
        return {
            'name': self.metadata.display_name,
            'description': self.metadata.description,
            'usage': self._generate_usage_docs(),
            'parameters': self._generate_parameter_docs(),
            'examples': self._generate_example_docs(),
            'citation': self.get_citation(),
            'version': self.metadata.version
        }
    
    def _generate_usage_docs(self) -> str:
        """Generate usage documentation."""
        return f"""
Usage:
    from cablecar_research.analysis import {self.__class__.__name__}
    
    analyzer = {self.__class__.__name__}(df, privacy_guard)
    results = analyzer.run_analysis(**parameters)
    formatted = analyzer.format_results(results)
"""
    
    def _generate_parameter_docs(self) -> str:
        """Generate parameter documentation."""
        params = self.get_required_parameters()
        docs = []
        
        if params.get('required'):
            docs.append("Required Parameters:")
            for name, spec in params['required'].items():
                docs.append(f"  {name} ({spec.get('type', 'any')}): {spec.get('description', 'No description')}")
        
        if params.get('optional'):
            docs.append("\nOptional Parameters:")
            for name, spec in params['optional'].items():
                docs.append(f"  {name} ({spec.get('type', 'any')}): {spec.get('description', 'No description')}")
        
        return "\n".join(docs) if docs else "No parameters required."
    
    def _generate_example_docs(self) -> str:
        """Generate example documentation."""
        return """
Example:
    # Basic usage example would be provided by each analysis class
    # analyzer = AnalysisClass(df, privacy_guard)
    # results = analyzer.run_analysis()
"""
    
    def __str__(self) -> str:
        """String representation of the analysis."""
        return f"{self.metadata.display_name} v{self.metadata.version} by {self.metadata.author}"
    
    def __repr__(self) -> str:
        """Developer representation of the analysis."""
        return f"{self.__class__.__name__}(name='{self.metadata.name}', version='{self.metadata.version}')"