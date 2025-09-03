#!/usr/bin/env python
"""
Plugin Template Generator

Script to generate boilerplate code for new CableCar analysis plugins.
This makes it easy for contributors to create new analysis modules
following the standard interface.

Usage:
    python scripts/generate_plugin_template.py --name my_analysis --type descriptive --author "John Doe"
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any
import textwrap


def generate_plugin_template(
    plugin_name: str,
    analysis_type: str,
    author: str,
    email: str = "",
    description: str = "",
    output_dir: str = "cablecar_research/plugins/community"
) -> None:
    """Generate a complete plugin template."""
    
    # Clean plugin name
    clean_name = plugin_name.lower().replace(' ', '_').replace('-', '_')
    class_name = ''.join(word.capitalize() for word in clean_name.split('_')) + 'Plugin'
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate the plugin file
    plugin_content = generate_plugin_code(
        plugin_name=clean_name,
        class_name=class_name,
        analysis_type=analysis_type,
        author=author,
        email=email,
        description=description or f"Analysis plugin for {plugin_name}"
    )
    
    plugin_file = output_path / f"{clean_name}.py"
    with open(plugin_file, 'w', encoding='utf-8') as f:
        f.write(plugin_content)
    
    # Generate test file
    test_content = generate_test_template(
        plugin_name=clean_name,
        class_name=class_name
    )
    
    test_dir = Path("tests/plugins")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = test_dir / f"test_{clean_name}.py"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    # Generate documentation template
    docs_content = generate_docs_template(
        plugin_name=clean_name,
        class_name=class_name,
        description=description or f"Analysis plugin for {plugin_name}",
        author=author
    )
    
    docs_dir = Path("docs/plugins")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    docs_file = docs_dir / f"{clean_name}.md"
    with open(docs_file, 'w', encoding='utf-8') as f:
        f.write(docs_content)
    
    print(f"✓ Plugin template generated successfully!")
    print(f"  Plugin: {plugin_file}")
    print(f"  Tests: {test_file}")
    print(f"  Docs: {docs_file}")
    print()
    print("Next steps:")
    print("1. Implement the analysis logic in the run_analysis() method")
    print("2. Update the parameter requirements in get_required_parameters()")
    print("3. Add comprehensive tests")
    print("4. Update the documentation with usage examples")
    print("5. Test your plugin with: python -m pytest tests/plugins/test_{}.py".format(clean_name))


def generate_plugin_code(
    plugin_name: str,
    class_name: str,
    analysis_type: str,
    author: str,
    email: str,
    description: str
) -> str:
    """Generate the main plugin code."""
    
    template = f'''"""
{class_name}

{description}
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from scipy import stats

from cablecar_research.analysis.base import BaseAnalysis, AnalysisMetadata, AnalysisType, ValidationLevel


class {class_name}(BaseAnalysis):
    """
    {description}
    
    This plugin implements [describe your analysis approach here].
    """
    
    metadata = AnalysisMetadata(
        name="{plugin_name}",
        display_name="{plugin_name.replace('_', ' ').title()}",
        description="{description}",
        version="1.0.0",
        author="{author}",
        email="{email}",
        analysis_type=AnalysisType.{analysis_type.upper()},
        validation_level=ValidationLevel.STANDARD,
        citation="{author}. {class_name} for Clinical Research. CableCar v1.0.0",
        keywords=["{plugin_name}", "{analysis_type}", "clinical_research"]
    )
    
    def validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """Validate inputs for {plugin_name} analysis."""
        validation = {{
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }}
        
        # Check if data is loaded
        if self.df is None:
            validation['valid'] = False
            validation['errors'].append("No data loaded")
            return validation
        
        # TODO: Add specific validation logic for your analysis
        # Example:
        # required_param = kwargs.get('required_parameter')
        # if not required_param:
        #     validation['valid'] = False
        #     validation['errors'].append("'required_parameter' is required")
        
        # Check for minimum sample size
        if len(self.df) < 10:
            validation['valid'] = False
            validation['errors'].append("Insufficient sample size (minimum 10 observations required)")
        
        # Add warnings for small samples
        if len(self.df) < 50:
            validation['warnings'].append("Small sample size may affect result reliability")
        
        return validation
    
    def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """Execute {plugin_name} analysis."""
        
        # TODO: Implement your analysis logic here
        results = {{
            'analysis_type': '{plugin_name}',
            'sample_size': len(self.df),
            'timestamp': pd.Timestamp.now().isoformat()
        }}
        
        # Example analysis structure:
        # Step 1: Data preparation
        # clean_data = self._prepare_data(**kwargs)
        
        # Step 2: Core analysis
        # analysis_results = self._perform_analysis(clean_data, **kwargs)
        # results.update(analysis_results)
        
        # Step 3: Statistical tests (if applicable)
        # if kwargs.get('include_tests', False):
        #     test_results = self._perform_statistical_tests(clean_data, **kwargs)
        #     results['statistical_tests'] = test_results
        
        # Apply privacy protection if available
        if self.privacy_guard:
            results = self.privacy_guard.sanitize_output(results, '{plugin_name}')
        
        return results
    
    def format_results(self, results: Dict[str, Any], format_type: str = "standard") -> str:
        """Format {plugin_name} analysis results."""
        if format_type == "summary":
            return self._format_summary(results)
        elif format_type == "detailed":
            return self._format_detailed(results)
        elif format_type == "publication":
            return self._format_publication(results)
        else:
            return self._format_standard(results)
    
    def get_required_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter requirements for {plugin_name} analysis."""
        return {{
            'required': {{
                # TODO: Define required parameters
                # Example:
                # 'outcome_variable': {{
                #     'type': 'string',
                #     'description': 'Primary outcome variable to analyze',
                #     'example': 'mortality'
                # }}
            }},
            'optional': {{
                # TODO: Define optional parameters
                # Example:
                # 'adjustment_variables': {{
                #     'type': 'array',
                #     'description': 'Variables to adjust for in analysis',
                #     'example': ['age', 'sex']
                # }},
                'include_confidence_intervals': {{
                    'type': 'boolean',
                    'description': 'Include confidence intervals in results',
                    'example': True
                }}
            }}
        }}
    
    def _format_standard(self, results: Dict[str, Any]) -> str:
        \"\"\"Standard formatting of results.\"\"\"
        output = f"{{self.metadata.display_name}} Results\\n{{'='*50}}\\n\\n"
        output += f"Sample Size: {{results['sample_size']:,}}\\n"
        output += f"Analysis completed at: {{results['timestamp']}}\\n\\n"
        
        # TODO: Add specific result formatting
        # Example:
        # if 'main_results' in results:
        #     output += "Main Results:\\n"
        #     for key, value in results['main_results'].items():
        #         output += f"  {{key}}: {{value}}\\n"
        
        return output
    
    def _format_summary(self, results: Dict[str, Any]) -> str:
        \"\"\"Summary formatting of results.\"\"\"
        output = f"{{self.metadata.display_name}} Summary\\n"
        output += f"Sample: n={{results['sample_size']:,}}\\n"
        
        # TODO: Add summary-specific formatting
        
        return output
    
    def _format_detailed(self, results: Dict[str, Any]) -> str:
        \"\"\"Detailed formatting with comprehensive output.\"\"\"
        # For now, same as standard - customize as needed
        return self._format_standard(results)
    
    def _format_publication(self, results: Dict[str, Any]) -> str:
        \"\"\"Publication-ready formatting.\"\"\"
        output = f"Table: {{self.metadata.display_name}} Results\\n"
        output += "="*50 + "\\n\\n"
        
        # TODO: Add publication-quality formatting
        # Follow journal standards for tables/figures
        
        return output
    
    # TODO: Add private helper methods for your analysis
    # def _prepare_data(self, **kwargs) -> pd.DataFrame:
    #     """Prepare data for analysis."""
    #     pass
    # 
    # def _perform_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    #     """Core analysis implementation."""
    #     pass
    # 
    # def _perform_statistical_tests(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    #     """Statistical significance testing."""
    #     pass
'''
    
    return textwrap.dedent(template).strip()


def generate_test_template(plugin_name: str, class_name: str) -> str:
    """Generate test template."""
    
    template = f'''"""
Tests for {class_name}

Comprehensive test suite for the {plugin_name} analysis plugin.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from cablecar_research.plugins.community.{plugin_name} import {class_name}
from cablecar_research.privacy.protection import PrivacyGuard


@pytest.fixture
def sample_data():
    """Create sample clinical data for testing."""
    np.random.seed(42)
    n = 100
    
    return pd.DataFrame({{
        'patient_id': range(1, n + 1),
        'age': np.random.normal(65, 15, n),
        'sex': np.random.choice(['M', 'F'], n),
        'outcome': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'biomarker': np.random.lognormal(0, 1, n),
        'treatment': np.random.choice(['A', 'B'], n)
    }})


@pytest.fixture
def privacy_guard():
    """Create privacy guard for testing."""
    return PrivacyGuard(min_cell_size=5)


@pytest.fixture
def plugin_instance(sample_data, privacy_guard):
    """Create plugin instance for testing."""
    return {class_name}(sample_data, privacy_guard)


class Test{class_name}:
    """Test suite for {class_name}."""
    
    def test_plugin_metadata(self, plugin_instance):
        """Test plugin metadata is properly defined."""
        assert plugin_instance.metadata.name == "{plugin_name}"
        assert plugin_instance.metadata.display_name is not None
        assert plugin_instance.metadata.version is not None
        assert plugin_instance.metadata.author is not None
    
    def test_validate_inputs_valid(self, plugin_instance):
        """Test input validation with valid inputs."""
        validation = plugin_instance.validate_inputs()
        assert validation['valid'] is True
        assert len(validation['errors']) == 0
    
    def test_validate_inputs_no_data(self):
        """Test input validation with no data."""
        plugin = {class_name}(None, None)
        validation = plugin.validate_inputs()
        assert validation['valid'] is False
        assert "No data loaded" in validation['errors']
    
    def test_validate_inputs_small_sample(self, privacy_guard):
        """Test input validation with small sample size."""
        small_data = pd.DataFrame({{'id': [1, 2, 3]}})
        plugin = {class_name}(small_data, privacy_guard)
        validation = plugin.validate_inputs()
        assert validation['valid'] is False
        assert "Insufficient sample size" in validation['errors'][0]
    
    def test_run_analysis_basic(self, plugin_instance):
        """Test basic analysis execution."""
        results = plugin_instance.run_analysis()
        
        assert 'analysis_type' in results
        assert results['analysis_type'] == "{plugin_name}"
        assert 'sample_size' in results
        assert results['sample_size'] == 100
        assert 'timestamp' in results
    
    def test_required_parameters(self, plugin_instance):
        """Test required parameters specification."""
        params = plugin_instance.get_required_parameters()
        
        assert 'required' in params
        assert 'optional' in params
        assert isinstance(params['required'], dict)
        assert isinstance(params['optional'], dict)
    
    def test_format_results_standard(self, plugin_instance):
        """Test standard results formatting."""
        results = plugin_instance.run_analysis()
        formatted = plugin_instance.format_results(results, "standard")
        
        assert "{class_name.replace('Plugin', '')} Results" in formatted
        assert "Sample Size: 100" in formatted
    
    def test_format_results_summary(self, plugin_instance):
        """Test summary results formatting."""
        results = plugin_instance.run_analysis()
        formatted = plugin_instance.format_results(results, "summary")
        
        assert "Summary" in formatted
        assert "n=100" in formatted
    
    def test_format_results_publication(self, plugin_instance):
        """Test publication results formatting."""
        results = plugin_instance.run_analysis()
        formatted = plugin_instance.format_results(results, "publication")
        
        assert "Table:" in formatted
    
    def test_privacy_protection(self, sample_data):
        """Test privacy protection is applied."""
        privacy_guard = MagicMock()
        privacy_guard.sanitize_output.return_value = {{'sanitized': True}}
        
        plugin = {class_name}(sample_data, privacy_guard)
        results = plugin.run_analysis()
        
        privacy_guard.sanitize_output.assert_called_once()
        assert results == {{'sanitized': True}}
    
    # TODO: Add specific tests for your analysis logic
    # def test_specific_analysis_feature(self, plugin_instance):
    #     """Test specific analysis functionality."""
    #     pass
    
    # TODO: Add edge case tests
    # def test_missing_data_handling(self, plugin_instance):
    #     """Test handling of missing data."""
    #     pass
    
    # TODO: Add statistical accuracy tests
    # def test_statistical_accuracy(self, plugin_instance):
    #     """Test statistical calculations are accurate."""
    #     pass
'''
    
    return textwrap.dedent(template).strip()


def generate_docs_template(plugin_name: str, class_name: str, description: str, author: str) -> str:
    """Generate documentation template."""
    
    template = f'''# {class_name.replace('Plugin', '')} Analysis Plugin

{description}

## Overview

The {plugin_name} plugin provides [describe what your analysis does] for clinical research datasets. It implements the standard CableCar plugin interface for consistent integration with the analysis pipeline.

## Features

- [Feature 1]
- [Feature 2] 
- [Feature 3]
- Privacy-preserving analysis with configurable protection levels
- Multiple output formats (standard, detailed, summary, publication)
- Comprehensive input validation

## Usage

### Basic Usage

```python
# Via MCP server
run_{plugin_name}(
    # Add required parameters here
)
```

### Advanced Usage

```python
# Direct usage in Python
from cablecar_research.plugins.community.{plugin_name} import {class_name}
from cablecar_research.privacy.protection import PrivacyGuard

# Load your data
df = pd.read_csv("your_data.csv")
privacy_guard = PrivacyGuard(min_cell_size=10)

# Create plugin instance
plugin = {class_name}(df, privacy_guard)

# Run analysis
results = plugin.run_analysis(
    # parameters here
)

# Format results
formatted = plugin.format_results(results, format_type="publication")
print(formatted)
```

## Parameters

### Required Parameters

- [List required parameters with descriptions]

### Optional Parameters

- [List optional parameters with descriptions]

## Output

The plugin returns a dictionary containing:

- `analysis_type`: Type of analysis performed
- `sample_size`: Number of observations analyzed
- `timestamp`: Analysis completion timestamp
- [List other output fields]

## Statistical Methods

[Describe the statistical methods used]

## Validation and Quality Checks

The plugin performs the following validation checks:

1. Data availability verification
2. Minimum sample size requirements
3. [Additional validation checks]

## Privacy Protection

All outputs are processed through the privacy guard to ensure:

- Small cell suppression (configurable threshold)
- Removal of potentially identifying information
- [Additional privacy protections]

## Examples

### Example 1: Basic Analysis

```python
results = run_{plugin_name}(
    # example parameters
)
```

### Example 2: Advanced Analysis

```python
results = run_{plugin_name}(
    # advanced example parameters
    output_format="publication"
)
```

## Interpretation

[Guidelines for interpreting results]

## Limitations

- [List any limitations or assumptions]

## References

- [Relevant citations or methodological references]

## Author

{author}

## Version History

- v1.0.0: Initial implementation
'''
    
    return textwrap.dedent(template).strip()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate CableCar analysis plugin template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python scripts/generate_plugin_template.py --name survival_analysis --type inferential --author "Jane Smith"
          python scripts/generate_plugin_template.py --name "Time Series Analysis" --type exploratory --author "John Doe" --email john@example.com
        """)
    )
    
    parser.add_argument(
        "--name", "-n",
        required=True,
        help="Name of the analysis plugin (e.g., 'survival_analysis')"
    )
    
    parser.add_argument(
        "--type", "-t",
        required=True,
        choices=["descriptive", "inferential", "predictive", "exploratory"],
        help="Type of analysis"
    )
    
    parser.add_argument(
        "--author", "-a",
        required=True,
        help="Author name"
    )
    
    parser.add_argument(
        "--email", "-e",
        default="",
        help="Author email address"
    )
    
    parser.add_argument(
        "--description", "-d",
        default="",
        help="Brief description of the analysis"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="cablecar_research/plugins/community",
        help="Output directory for plugin files"
    )
    
    args = parser.parse_args()
    
    generate_plugin_template(
        plugin_name=args.name,
        analysis_type=args.type,
        author=args.author,
        email=args.email,
        description=args.description,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()