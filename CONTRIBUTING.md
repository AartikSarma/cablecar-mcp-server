# Contributing to CableCar

Welcome! CableCar thrives on community contributions. This guide will help you add new analysis plugins, improve existing functionality, and contribute to our mission of **Connecting AI to Big Longitudinal EMRs for Clinical Analytics and Research**.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Plugin Development](#plugin-development)  
3. [Code Standards](#code-standards)
4. [Testing Requirements](#testing-requirements)
5. [Documentation](#documentation)
6. [Submission Process](#submission-process)
7. [Review Process](#review-process)
8. [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Basic understanding of clinical research methodology
- Familiarity with pandas, numpy, and scipy

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/clif_mcp_server.git
   cd clif_mcp_server
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

4. Run tests to ensure everything works:
   ```bash
   python -m pytest tests/
   ```

## Plugin Development

### Quick Start: Generate a Plugin Template

The easiest way to create a new plugin is using our template generator:

```bash
python scripts/generate_plugin_template.py \
    --name "my_analysis" \
    --type "descriptive" \
    --author "Your Name" \
    --email "your.email@example.com" \
    --description "Brief description of your analysis"
```

This creates:
- `cablecar_research/plugins/community/my_analysis.py` - Main plugin code
- `tests/plugins/test_my_analysis.py` - Test suite
- `docs/plugins/my_analysis.md` - Documentation

### Plugin Types and Directories

Choose the appropriate directory for your plugin:

- **`plugins/core/`** - Fundamental analyses (descriptive statistics, basic tests)
- **`plugins/community/`** - Community-contributed analyses (preferred for new contributions)
- **`plugins/contrib/`** - Experimental or specialized analyses

### Plugin Architecture

Every plugin must inherit from `BaseAnalysis` and implement these methods:

```python
from cablecar_research.analysis.base import BaseAnalysis, AnalysisMetadata, AnalysisType

class MyAnalysisPlugin(BaseAnalysis):
    metadata = AnalysisMetadata(
        name="my_analysis",
        display_name="My Analysis", 
        description="Description of what this analysis does",
        version="1.0.0",
        author="Your Name",
        email="your.email@example.com",
        analysis_type=AnalysisType.DESCRIPTIVE,  # or INFERENTIAL, PREDICTIVE, EXPLORATORY
        validation_level=ValidationLevel.STANDARD,
        keywords=["keyword1", "keyword2"]
    )
    
    def validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """Validate all inputs before analysis."""
        pass
    
    def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """Execute the analysis and return results.""" 
        pass
    
    def format_results(self, results: Dict[str, Any], format_type: str = "standard") -> str:
        """Format results for display."""
        pass
    
    def get_required_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Define required and optional parameters."""
        pass
```

### Analysis Types

Choose the most appropriate analysis type:

- **DESCRIPTIVE**: Summarize and describe data (means, medians, frequencies)
- **INFERENTIAL**: Test hypotheses and make inferences (t-tests, chi-square, ANOVA)  
- **PREDICTIVE**: Build models to predict outcomes (regression, machine learning)
- **EXPLORATORY**: Discover patterns and generate hypotheses (clustering, PCA)

### Privacy Protection

CableCar enforces privacy protection on all outputs. Your plugin automatically receives a `privacy_guard` instance:

```python
def run_analysis(self, **kwargs) -> Dict[str, Any]:
    results = {
        'sample_size': len(self.df),
        'analysis_results': my_analysis_results
    }
    
    # Privacy protection is automatically applied
    if self.privacy_guard:
        results = self.privacy_guard.sanitize_output(results, 'my_analysis')
    
    return results
```

The privacy guard will:
- Suppress small cell counts (< minimum threshold)
- Remove potential patient identifiers
- Apply differential privacy if enabled

## Code Standards

### Style Guidelines

- Follow PEP 8 
- Use type hints for all function parameters and return values
- Maximum line length: 88 characters (Black formatter standard)
- Use descriptive variable names (especially for clinical terms)

### Documentation Standards

Every function must have a docstring:

```python
def calculate_mortality_rate(self, groups: List[str], adjustment_vars: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate mortality rates by group with optional adjustment.
    
    Args:
        groups: List of grouping variables
        adjustment_vars: Variables to adjust for using stratification
        
    Returns:
        Dictionary with group names as keys and mortality rates as values
        
    Raises:
        ValueError: If groups contain non-existent variables
    """
```

### Clinical Research Best Practices

- Always validate input data (missing values, outliers, data types)
- Report confidence intervals where appropriate
- Use appropriate statistical tests for data types
- Consider multiple comparison corrections
- Document statistical assumptions
- Provide clear interpretation guidance

## Testing Requirements

### Minimum Test Coverage

Every plugin must have:

1. **Metadata tests** - Verify plugin information is complete
2. **Validation tests** - Test input validation with valid/invalid inputs  
3. **Analysis tests** - Test core analysis functionality
4. **Formatting tests** - Test all output formats
5. **Privacy tests** - Verify privacy protection is applied
6. **Edge case tests** - Handle missing data, small samples, etc.

### Test Structure

```python
class TestMyAnalysisPlugin:
    def test_plugin_metadata(self, plugin_instance):
        """Test plugin metadata is properly defined."""
        assert plugin_instance.metadata.name is not None
        assert plugin_instance.metadata.version is not None
        
    def test_validate_inputs_valid(self, plugin_instance):
        """Test input validation with valid inputs."""
        validation = plugin_instance.validate_inputs(param1="value1")
        assert validation['valid'] is True
        
    def test_run_analysis_basic(self, plugin_instance):
        """Test basic analysis execution."""
        results = plugin_instance.run_analysis(param1="value1")
        assert 'analysis_type' in results
        
    def test_privacy_protection(self, sample_data):
        """Test privacy protection is applied."""
        # Test implementation
        pass
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run tests for your specific plugin  
python -m pytest tests/plugins/test_my_analysis.py

# Run with coverage
python -m pytest --cov=cablecar_research tests/
```

## Documentation

### Plugin Documentation

Each plugin needs comprehensive documentation in `docs/plugins/`:

- **Overview** - What the analysis does
- **Clinical Use Cases** - When to use this analysis
- **Parameters** - All required and optional parameters  
- **Output** - What the analysis returns
- **Statistical Methods** - Methods and assumptions
- **Examples** - Code examples with real use cases
- **Interpretation** - How to interpret results
- **Limitations** - Known limitations or assumptions

### Code Comments

- Comment complex statistical procedures
- Explain clinical reasoning behind choices
- Document any literature references
- Clarify parameter choices

## Submission Process

### 1. Development Workflow

```bash
# Create feature branch
git checkout -b feature/my-analysis-plugin

# Develop your plugin
python scripts/generate_plugin_template.py --name my_analysis --type descriptive --author "Your Name"

# Implement analysis logic
# Add comprehensive tests
# Update documentation

# Run tests and verify everything works
python -m pytest tests/plugins/test_my_analysis.py
python -m pytest tests/  # Run all tests to ensure no regressions

# Commit your changes
git add .
git commit -m "Add my_analysis plugin for [brief description]"
git push origin feature/my-analysis-plugin
```

### 2. Pull Request Requirements

Your PR must include:

- [ ] New plugin file in appropriate directory
- [ ] Comprehensive test suite (minimum 80% coverage)
- [ ] Complete documentation
- [ ] Updated `CHANGELOG.md` entry
- [ ] No breaking changes to existing functionality

### 3. PR Template

Use this template for your pull request:

```markdown
## Description
Brief description of the analysis plugin and its clinical use case.

## Type of Change
- [ ] New analysis plugin
- [ ] Enhancement to existing plugin  
- [ ] Bug fix
- [ ] Documentation update

## Clinical Context
Explain when and why researchers would use this analysis.

## Testing
- [ ] All tests pass
- [ ] Added comprehensive test suite
- [ ] Tested with sample clinical data
- [ ] Privacy protection verified

## Documentation  
- [ ] Plugin documentation complete
- [ ] Code comments added
- [ ] Examples provided

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] No console warnings or errors
- [ ] Compatible with existing plugins
```

## Review Process

### Review Criteria

Reviews evaluate:

1. **Clinical Validity** - Is the analysis scientifically sound?
2. **Code Quality** - Clean, readable, well-documented code
3. **Testing** - Comprehensive tests with good coverage
4. **Privacy Compliance** - Proper privacy protection implementation
5. **Integration** - Works with existing CableCar infrastructure
6. **Documentation** - Clear, complete documentation

### Review Timeline

- Initial review: 3-7 days
- Follow-up reviews: 1-3 days  
- Final approval: 1-2 days

### Common Review Feedback

- **Statistical Issues**: Inappropriate test selection, missing assumptions checks
- **Privacy Concerns**: Insufficient privacy protection, potential data leakage
- **Testing Gaps**: Missing edge cases, insufficient coverage
- **Documentation**: Unclear usage examples, missing clinical context
- **Code Style**: PEP 8 violations, missing type hints

## Community Guidelines

### Code of Conduct

- Be respectful and professional
- Focus on constructive feedback
- Welcome newcomers and help them learn
- Clinical research serves patients - prioritize safety and accuracy

### Getting Help

- **Documentation Issues**: Open a documentation issue
- **Plugin Development**: Start a discussion thread
- **Bug Reports**: Use the bug report template
- **Feature Requests**: Use the feature request template

### Recognition

Contributors are recognized:

- Author attribution in plugin metadata
- Contributor list in repository
- Annual contributor recognition
- Co-authorship opportunities for significant contributions

## Advanced Topics

### Custom Validation Logic

```python
def validate_inputs(self, **kwargs) -> Dict[str, Any]:
    validation = super().validate_inputs(**kwargs)
    
    # Custom clinical validation
    outcome_var = kwargs.get('outcome')
    if outcome_var and outcome_var not in ['mortality', 'los', 'complications']:
        validation['warnings'].append(f"Unusual outcome variable: {outcome_var}")
    
    return validation
```

### Integration with External Tools

```python
# Example: Integration with R via rpy2
def run_r_analysis(self, r_script: str) -> Dict[str, Any]:
    """Run R analysis while maintaining privacy protection."""
    # Implementation details
    pass
```

### Performance Optimization

For large datasets:

- Use chunked processing with `pd.read_csv(chunksize=...)`
- Implement sampling strategies for initial exploration  
- Cache intermediate results appropriately
- Profile memory usage for datasets > 1M rows

---

## Questions?

- 📖 **Documentation**: [Link to full documentation]
- 💬 **Discussions**: [GitHub Discussions]
- 🐛 **Issues**: [GitHub Issues]
- 📧 **Email**: [Maintainer email]

Thank you for contributing to CableCar! Your work helps advance clinical research and improve patient care through better data analysis tools.