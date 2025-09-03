# CableCar Clinical Research Platform

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

🔌 **Modular Plugin System**
- Automatic plugin discovery and loading
- Standardized analysis interface
- Dynamic MCP tool generation

🔒 **Privacy-First Design** 
- Cell suppression for small counts
- Automatic PHI removal
- Configurable privacy levels
- Comprehensive audit trails

📈 **Standards Compliance**
- STROBE reporting for observational studies
- TRIPOD+AI for prediction models
- Publication-ready outputs

🌐 **Multi-Site Ready**
- Generates portable analysis code
- Federated research support
- No patient data sharing required

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/clif_mcp_server.git
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

### Core System Tools
- `import_dataset` - Load and validate clinical datasets
- `design_study` - Interactive study design guidance
- `explore_data` - Privacy-safe data exploration
- `generate_strobe_report` - STROBE-compliant reports
- `generate_tripod_report` - TRIPOD+AI prediction model reports
- `export_analysis_code` - Generate reproducible analysis scripts

### Analysis Plugins
- `run_descriptive_statistics` - Comprehensive Table 1 generation
- `run_propensity_score_matching` - Causal inference analysis
- `list_available_plugins` - Browse all available analysis plugins

*New plugins are automatically discovered and available as MCP tools*

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
test_hypotheses(
    outcome_variables=["mortality", "length_of_stay"],
    group_variable="antibiotic_timing",
    correction_method="fdr_bh"
)

# 5. Generate publication report
generate_strobe_report(
    output_format="markdown",
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
# Propensity score matching for causal inference
run_propensity_score_matching(
    treatment_variable="antibiotic_timing",
    outcome_variables=["mortality", "length_of_stay"],
    matching_variables=["age", "sex", "severity_score"],
    matching_ratio=1,
    output_format="detailed"
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
from cablecar_research.analysis.base import BaseAnalysis, AnalysisMetadata

class SurvivalAnalysisPlugin(BaseAnalysis):
    metadata = AnalysisMetadata(
        name="survival_analysis",
        display_name="Survival Analysis",
        description="Cox proportional hazards survival analysis",
        analysis_type=AnalysisType.INFERENTIAL,
        # ... additional metadata
    )
    
    def validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """Validate analysis inputs"""
        
    def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """Execute the analysis"""
        
    def format_results(self, results, format_type="standard") -> str:
        """Format results for display"""
        
    def get_required_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Define parameter schema for MCP integration"""
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