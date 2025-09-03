# CableCar MCP Server

**Connecting AI to Big Longitudinal EMRs for Clinical Analytics and Research**

A modular, extensible MCP (Model Context Protocol) server for AI-assisted clinical research with privacy-first design and standards-compliant reporting.

## 🌟 Overview

CableCar enables clinical researchers to:
- **Import & analyze** clinical datasets with automatic schema validation
- **Build patient cohorts** using complex clinical criteria  
- **Perform statistical analyses** with comprehensive privacy protection
- **Generate publication-ready reports** following STROBE/TRIPOD+AI standards
- **Export reproducible code** for federated multi-site validation
- **Extend functionality** with community-contributed analysis plugins

## 🏗️ Architecture

CableCar uses a **modular plugin architecture** that makes it easy to add new analysis capabilities:

```
📁 cablecar_research/
├── 🧩 plugins/
│   ├── core/           # Built-in analyses (descriptive stats, regression)
│   ├── community/      # Community contributions 
│   └── contrib/        # Experimental analyses
├── 🔒 privacy/         # Privacy protection & data sanitization
├── 📊 reporting/       # STROBE/TRIPOD+AI compliant reports
└── 🔧 registry.py     # Dynamic plugin discovery & MCP integration
```

### Key Features

🔌 **Unified Plugin Architecture** *(New!)*
- All analysis functions implemented as modular plugins
- Single source of truth - no code duplication
- Automatic plugin discovery and loading  
- Standardized analysis interface with validation
- Dynamic MCP tool generation with full metadata
- Hot-pluggable extensions without server restart

🔒 **Privacy-First Design** 
- Cell suppression for small counts
- Automatic PHI removal
- Configurable privacy levels (Standard/High/Maximum)
- Comprehensive audit trails
- Local processing - data never leaves your system

📈 **Standards Compliance**
- **STROBE reporting** for observational studies
- **TRIPOD+AI** for prediction models with ML interpretability
- Publication-ready outputs with checklists
- Comprehensive sensitivity analysis support

🌐 **Multi-Site Ready**
- Generates portable analysis code (Python/R)
- Federated research support with Docker containers
- No patient data sharing required
- Identical analysis execution across sites

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/AartikSarma/cablecar-mcp-server.git
cd clif_mcp_server

# Install dependencies
pip install -r requirements.txt

# Generate synthetic test data
python synthetic_data/clif_generator.py
```

### Start the MCP Server

```bash
python -m server.main --data-path ./data/synthetic
```

### Claude Desktop Configuration

Add to your Claude Desktop MCP settings:

```json
{
  "mcpServers": {
    "cablecar": {
      "command": "python",
      "args": ["-m", "server.main", "--data-path", "/path/to/your/data"],
      "cwd": "/path/to/clif_mcp_server"
    }
  }
}
```

## 📊 Available Analysis Tools

### Core System Tools (7)
- `import_dataset` - Load and validate clinical datasets with schema checking
- `design_study` - Interactive study design wizard with methodological guidance
- `explore_data` - Comprehensive data exploration with privacy-safe summaries
- `export_analysis_code` - Generate complete, reproducible analysis code for multi-site validation
- `get_analysis_summary` - Get summary of all analyses performed in current session
- `get_privacy_report` - Generate comprehensive privacy compliance report
- `list_available_plugins` - List all available analysis plugins and their capabilities

### Analysis Plugins (9)
All analysis functions are now implemented as modular plugins with standardized interfaces:

**📊 Descriptive & Exploratory**
- `run_descriptive_statistics` - Publication-ready Table 1 with statistical tests and stratification
- `run_exploratory_data_analysis` - Comprehensive EDA with data quality, distributions, correlations, outliers

**🔬 Statistical Testing & Modeling**  
- `run_hypothesis_testing` - Comprehensive statistical hypothesis testing with multiple comparison corrections
- `run_regression_analysis` - Linear, logistic, and Cox regression with diagnostics and validation
- `run_ml_models` - AutoML with validation, interpretability, and TRIPOD+AI compliance

**🎯 Specialized Analyses**
- `run_propensity_score_matching` - Propensity score matching for causal inference
- `run_sensitivity_analysis` - Robustness testing for missing data, outliers, definitions, and subgroups

**📝 Standards-Compliant Reporting**
- `run_strobe_reporter` - STROBE-compliant reports for observational studies  
- `run_tripod_reporter` - TRIPOD+AI compliant reports for prediction model studies

*All plugins are automatically discovered and available as MCP tools with full metadata and validation*

## 🔧 Usage Examples

### Basic Data Analysis Workflow

```python
# 1. Import your clinical dataset
import_dataset(
    data_path="/path/to/clinical/data",
    data_format="clif",
    privacy_level="standard"
)

# 2. Design your study approach
design_study(
    research_question="Does early antibiotic therapy improve outcomes?",
    study_type="analytical"
)

# 3. Generate baseline characteristics table
run_descriptive_statistics(
    variables=["age", "sex", "comorbidities", "severity_score"],
    stratify_by="antibiotic_timing",
    output_format="publication"
)

# 4. Test your hypothesis  
run_hypothesis_testing(
    outcome_variables=["mortality", "length_of_stay"],
    group_variable="antibiotic_timing",
    correction_method="fdr_bh"
)

# 5. Generate publication report
run_strobe_reporter(
    study_design="cohort",
    study_title="Early Antibiotic Therapy and Clinical Outcomes",
    study_objective="To assess the impact of early antibiotic therapy on patient outcomes",
    primary_exposure="antibiotic_timing",
    primary_outcome="mortality",
    include_checklist=true
)

# 6. Export code for multi-site validation
export_analysis_code(
    language="python",
    include_all_analyses=true
)
```

### Advanced Analysis with Plugins

```python
# Comprehensive exploratory data analysis
run_exploratory_data_analysis(
    target_variable="mortality",
    include_correlations=true,
    include_distributions=true,
    include_outlier_detection=true
)

# Propensity score matching for causal inference
run_propensity_score_matching(
    treatment_variable="antibiotic_timing",
    outcome_variables=["mortality", "length_of_stay"], 
    matching_variables=["age", "sex", "severity_score"],
    matching_ratio=1,
    output_format="detailed"
)

# Sensitivity analysis for robustness testing
run_sensitivity_analysis(
    primary_results=previous_analysis_results,
    outcome_column="mortality",
    missing_data_methods=["complete_case", "multiple_imputation"],
    outlier_methods=["iqr", "isolation_forest"],
    sensitivity_threshold=0.2
)

# Generate TRIPOD+AI compliant prediction model report
run_tripod_reporter(
    study_type="development_and_validation",
    model_title="ICU Mortality Prediction Model",
    study_objective="Develop and validate a model to predict ICU mortality",
    intended_use="Clinical decision support for ICU risk stratification",
    outcome_variable="mortality",
    outcome_type="binary",
    predictor_variables=["age", "severity_score", "comorbidities"],
    model_type="machine_learning"
)
```

## 🧩 Extending CableCar: Plugin Development

### Create a New Plugin

Generate a plugin template in seconds:

```bash
python scripts/generate_plugin_template.py \
    --name "survival_analysis" \
    --type "inferential" \
    --author "Dr. Jane Smith" \
    --description "Cox proportional hazards survival analysis"
```

This creates:
- Plugin code with standardized interface
- Comprehensive test suite  
- Complete documentation template

### Plugin Structure

Every plugin implements the `BaseAnalysis` interface:

```python
from cablecar_research.analysis.base import BaseAnalysis, AnalysisMetadata, AnalysisType

class SurvivalAnalysisPlugin(BaseAnalysis):
    metadata = AnalysisMetadata(
        name="survival_analysis",
        display_name="Survival Analysis", 
        description="Cox proportional hazards survival analysis with comprehensive diagnostics",
        analysis_type=AnalysisType.INFERENTIAL,
        required_columns=["time", "event"],
        optional_columns=["covariates"],
        parameters={
            "time_column": {
                "type": "string",
                "description": "Time to event or censoring column",
                "required": True
            },
            "event_column": {
                "type": "string", 
                "description": "Binary event indicator column",
                "required": True
            },
            "covariate_columns": {
                "type": "list",
                "description": "List of covariate columns for Cox model",
                "required": False,
                "default": []
            }
        }
    )
    
    def validate_inputs(self, df: pd.DataFrame, **kwargs) -> List[str]:
        """Validate analysis inputs - returns list of error messages"""
        errors = []
        if 'time_column' not in kwargs:
            errors.append("time_column is required")
        # ... additional validation
        return errors
        
    def run_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute the survival analysis"""
        time_col = kwargs['time_column']
        event_col = kwargs['event_column']
        # ... analysis implementation
        return results
        
    def format_results(self, results: Dict[str, Any]) -> str:
        """Format results for display"""
        # ... formatting implementation
        return formatted_output
        
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters"""
        return ["time_column", "event_column"]
```

Your plugin is automatically:
- Discovered by the registry
- Available as `run_survival_analysis` MCP tool
- Integrated with privacy protection
- Documented and tested

## 📁 Data Formats

CableCar supports multiple clinical data formats:

### CLIF (Common Longitudinal ICU Format)
Standard format for ICU research data:
```
data/
├── patient.csv                    # Demographics, outcomes
├── hospitalization.csv            # Admission/discharge
├── vitals.csv                     # Vital signs  
├── labs.csv                       # Laboratory values
├── respiratory_support.csv        # Ventilation data
└── medication_administration.csv  # Medications
```

### Custom Clinical Datasets
Any CSV-based clinical dataset with automatic schema detection.

## 🔒 Privacy & Security

### Privacy Protection Levels

**Standard** (default)
- Minimum cell size: 10
- Basic PHI removal
- Audit logging

**High** 
- Minimum cell size: 20
- Enhanced data sanitization
- Detailed access logging

**Maximum**
- Minimum cell size: 50
- Differential privacy enabled
- Comprehensive audit trails

### Privacy Features
- **Cell Suppression**: Small counts automatically suppressed
- **PHI Removal**: Patient identifiers stripped from all outputs  
- **Aggregate Only**: No individual patient data in results
- **Audit Trails**: All data access logged for compliance
- **Local Processing**: All data stays on your system

## 🌐 Multi-Site Research

Generate analysis code that runs identically across clinical sites:

1. **Develop Analysis**: Use CableCar to develop your analysis locally
2. **Export Code**: Generate standalone Python/R scripts  
3. **Distribute**: Share code (no patient data) with collaborating sites
4. **Execute**: Each site runs identical analysis on their local data
5. **Aggregate**: Combine summary statistics for multi-site results

```python
# Generated code runs independently of CableCar
export_analysis_code(
    language="python",
    containerize=true,  # Includes Docker setup
    include_all_analyses=true
)
```

## 🤝 Contributing

We welcome contributions! CableCar's modular architecture makes it easy to add new analyses.

### Ways to Contribute
- **Analysis Plugins**: Add new statistical methods or clinical analyses
- **Documentation**: Improve guides and examples
- **Testing**: Enhance test coverage
- **Bug Fixes**: Fix issues and improve reliability

### Getting Started
1. Read our [Contributing Guidelines](CONTRIBUTING.md)
2. Check the [Plugin Development Guide](docs/plugin-development-guide.md)
3. Browse existing [plugins](cablecar_research/plugins/) for examples
4. Use the template generator to create new plugins

### Community
- 📋 **Issues**: [GitHub Issues](https://github.com/yourusername/clif_mcp_server/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/clif_mcp_server/discussions)
- 📚 **Documentation**: [Full Documentation](docs/)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🏥 Clinical Research Focus

CableCar is designed specifically for clinical research needs:

- **Regulatory Compliance**: Follows clinical research reporting standards
- **Privacy by Design**: Built for healthcare data requirements  
- **Reproducible Research**: Supports open science practices
- **Multi-Site Studies**: Enables federated research collaborations
- **Clinical Validation**: Statistical methods appropriate for clinical data

---

*CableCar: Connecting AI to Big Longitudinal EMRs for Clinical Analytics and Research*
