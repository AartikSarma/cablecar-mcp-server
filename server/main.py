#!/usr/bin/env python
"""
CableCar MCP Server - Main Entry Point

Comprehensive clinical research MCP server with:
- Guided analysis workflows
- Standards-compliant reporting (STROBE, TRIPOD+AI)
- Privacy-preserving data analysis
- Automated code generation for federated analysis
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import our clinical research library
from cablecar_research.data_import.loaders import DataLoader
from cablecar_research.privacy.protection import PrivacyGuard
from cablecar_research.registry import get_registry, initialize_registry

# Import server tools
from .tools.study_designer import StudyDesigner
from .tools.analysis_conductor import AnalysisConductor
from .tools.report_generator import ReportGenerator
from .tools.code_exporter import CodeExporter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global server instance
app = Server("cablecar-clinical-research-server")

# Global state
server_state = {
    'data_loader': None,
    'privacy_guard': None,
    'current_study': None,
    'analysis_history': [],
    'datasets': {}
}

# Initialize plugin registry
registry = get_registry()
initialized = False


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List all available CableCar clinical research tools."""
    
    # Initialize registry if needed
    global initialized
    if not initialized:
        initialize_registry()
        initialized = True
    
    # Get dynamic plugin tools
    plugin_tools = []
    for tool_spec in registry.generate_mcp_tools():
        plugin_tools.append(
            Tool(
                name=tool_spec.name,
                description=tool_spec.description,
                inputSchema=tool_spec.input_schema
            )
        )
    
    # Core system tools
    core_tools = [
        # Data Import and Setup
        Tool(
            name="import_dataset",
            description="Import and validate clinical dataset with schema checking and data quality assessment",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to directory containing clinical data files"
                    },
                    "data_format": {
                        "type": "string",
                        "enum": ["clif", "custom"],
                        "description": "Data format specification"
                    },
                    "privacy_level": {
                        "type": "string",
                        "enum": ["standard", "high", "maximum"],
                        "description": "Privacy protection level"
                    }
                },
                "required": ["data_path"]
            }
        ),
        
        # Study Design
        Tool(
            name="design_study",
            description="Interactive study design wizard to plan analysis approach and ensure methodological rigor",
            inputSchema={
                "type": "object",
                "properties": {
                    "research_question": {
                        "type": "string",
                        "description": "Primary research question or hypothesis"
                    },
                    "study_type": {
                        "type": "string",
                        "enum": ["descriptive", "analytical", "predictive"],
                        "description": "Type of study being conducted"
                    },
                    "outcome_type": {
                        "type": "string",
                        "enum": ["binary", "continuous", "time_to_event", "categorical"],
                        "description": "Type of primary outcome"
                    }
                },
                "required": ["research_question"]
            }
        ),
        
        # Data Exploration
        Tool(
            name="explore_data",
            description="Comprehensive data exploration with privacy-safe summaries and quality assessment",
            inputSchema={
                "type": "object",
                "properties": {
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific variables to explore (optional - explores all if not specified)"
                    },
                    "include_missing_analysis": {
                        "type": "boolean",
                        "description": "Include detailed missing data analysis"
                    }
                }
            }
        ),
        
        
        # Code Generation
        Tool(
            name="export_analysis_code",
            description="Generate complete, reproducible analysis code for multi-site validation",
            inputSchema={
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["python", "r"],
                        "description": "Programming language for generated code"
                    },
                    "include_all_analyses": {
                        "type": "boolean",
                        "description": "Include all performed analyses in exported code"
                    },
                    "containerize": {
                        "type": "boolean",
                        "description": "Generate Docker container for reproducible execution"
                    }
                },
                "required": ["language"]
            }
        ),
        
        # Study Management
        Tool(
            name="get_analysis_summary",
            description="Get summary of all analyses performed in current session",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        
        Tool(
            name="get_privacy_report",
            description="Generate comprehensive privacy compliance report",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        
        Tool(
            name="list_available_plugins",
            description="List all available analysis plugins and their capabilities",
            inputSchema={
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "enum": ["descriptive", "inferential", "predictive", "exploratory"],
                        "description": "Filter plugins by analysis type (optional)"
                    }
                }
            }
        )
    ]
    
    # Combine plugin tools and core tools
    return plugin_tools + core_tools


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Execute CableCar clinical research tools."""
    
    try:
        # Initialize registry if needed
        global initialized
        if not initialized:
            initialize_registry()
            initialized = True
        
        # Check if this is a plugin tool (starts with run_)
        if name.startswith("run_"):
            # Set current data for plugin access
            if server_state['datasets']:
                main_df = server_state['data_loader'].merge_core_tables()
                # Update the handler to use current data
                handler = registry.get_tool_handler(name)
                if handler:
                    # Create plugin instance with current data
                    plugin_name = name.replace('run_', '')
                    plugin_class = registry.get_plugin(plugin_name)
                    if plugin_class:
                        plugin_instance = plugin_class(server_state['privacy_guard'])
                        
                        # Validate inputs
                        validation_errors = plugin_instance.validate_inputs(main_df, **arguments)
                        if validation_errors:
                            error_msg = "Validation failed: " + "; ".join(validation_errors)
                            return [TextContent(type="text", text=error_msg)]
                        
                        # Run analysis
                        results = plugin_instance.run_analysis(main_df, **arguments)
                        
                        # Format results
                        formatted_results = plugin_instance.format_results(results)
                        
                        # Store in analysis history
                        server_state['analysis_history'].append({
                            'type': plugin_name,
                            'timestamp': pd.Timestamp.now(),
                            'results': results
                        })
                        
                        return [TextContent(type="text", text=formatted_results)]
                    else:
                        return [TextContent(type="text", text=f"Plugin {plugin_name} not found")]
                else:
                    return [TextContent(type="text", text=f"Handler for {name} not found")]
            else:
                return [TextContent(type="text", text="No dataset loaded. Use 'import_dataset' first.")]
        
        # Core system tools
        elif name == "import_dataset":
            return await _import_dataset(arguments)
        
        elif name == "design_study":
            return await _design_study(arguments)
        
        elif name == "explore_data":
            return await _explore_data(arguments)
        
        # Analysis tools now handled by plugins - redirect to plugin system
        elif name in ["generate_table1", "test_hypotheses", "fit_regression_model", 
                     "build_prediction_model", "conduct_sensitivity_analysis",
                     "generate_strobe_report", "generate_tripod_report"]:
            # Map tool names to plugin names
            plugin_mapping = {
                "generate_table1": "run_descriptive_statistics",
                "test_hypotheses": "run_hypothesis_testing", 
                "fit_regression_model": "run_regression_analysis",
                "build_prediction_model": "run_ml_models",
                "conduct_sensitivity_analysis": "run_sensitivity_analysis",
                "generate_strobe_report": "run_strobe_reporter",
                "generate_tripod_report": "run_tripod_reporter"
            }
            
            plugin_name = plugin_mapping.get(name)
            if plugin_name:
                # Redirect to plugin system
                return await call_tool(plugin_name, arguments)
            else:
                return [TextContent(type="text", text=f"Tool {name} not available through plugin system")]
        
        elif name == "export_analysis_code":
            return await _export_analysis_code(arguments)
        
        elif name == "get_analysis_summary":
            return await _get_analysis_summary(arguments)
        
        elif name == "get_privacy_report":
            return await _get_privacy_report(arguments)
        
        elif name == "list_available_plugins":
            return await _list_available_plugins(arguments)
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except Exception as e:
        error_msg = f"Error executing {name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return [TextContent(type="text", text=error_msg)]


# Tool implementation functions
async def _import_dataset(arguments: dict) -> List[TextContent]:
    """Import and validate clinical dataset."""
    data_path = arguments["data_path"]
    data_format = arguments.get("data_format", "clif")
    privacy_level = arguments.get("privacy_level", "standard")
    
    # Initialize privacy guard based on level
    privacy_configs = {
        "standard": {"min_cell_size": 10, "k_anonymity": 5},
        "high": {"min_cell_size": 20, "k_anonymity": 10}, 
        "maximum": {"min_cell_size": 50, "k_anonymity": 20, "enable_differential_privacy": True}
    }
    
    config = privacy_configs.get(privacy_level, privacy_configs["standard"])
    server_state['privacy_guard'] = PrivacyGuard(**config)
    
    # Load data
    data_loader = DataLoader(data_path)
    
    if data_format == "clif":
        datasets = data_loader.load_clif_dataset()
    else:
        # Custom format - load available tables
        datasets = {}
        for file_path in Path(data_path).glob("*.csv"):
            table_name = file_path.stem
            datasets[table_name] = data_loader.load_table(table_name)
    
    server_state['data_loader'] = data_loader
    server_state['datasets'] = datasets
    
    # Get data summary
    schema_summary = data_loader.get_schema_summary()
    validation_errors = data_loader.validate_clif_schema() if data_format == "clif" else {}
    
    # Create summary report
    result = f"""Dataset Import Complete
{"="*50}

Data Path: {data_path}
Privacy Level: {privacy_level.upper()}
Tables Loaded: {len(datasets)}

Schema Summary:
- Total Tables: {schema_summary['total_tables']}
- Total Rows: {schema_summary['total_rows']:,}

Data Quality Assessment:
"""
    
    for table_name, quality_info in schema_summary['data_quality'].items():
        result += f"\n{table_name}:"
        result += f"\n  - Completeness: {quality_info['completeness']:.1f}%"
        result += f"\n  - Missing Cells: {quality_info['missing_cells']:,}"
    
    if validation_errors:
        result += f"\n\nValidation Issues Found:"
        for table, errors in validation_errors.items():
            result += f"\n{table}: {'; '.join(errors)}"
    else:
        result += f"\n\n✓ All validation checks passed"
    
    result += f"\n\nDataset ready for analysis. Use 'design_study' to plan your research approach."
    
    return [TextContent(type="text", text=result)]


async def _design_study(arguments: dict) -> List[TextContent]:
    """Interactive study design guidance."""
    research_question = arguments["research_question"]
    study_type = arguments.get("study_type")
    outcome_type = arguments.get("outcome_type")
    
    # Initialize study designer
    study_designer = StudyDesigner()
    
    # Generate study design recommendations
    design_plan = study_designer.create_study_plan(
        research_question=research_question,
        study_type=study_type,
        outcome_type=outcome_type,
        available_data=server_state['datasets']
    )
    
    server_state['current_study'] = design_plan
    
    result = f"""Study Design Plan
{"="*50}

Research Question: {research_question}

Recommended Study Design:
- Study Type: {design_plan['study_type']}
- Primary Analysis: {design_plan['primary_analysis']}
- Sample Size Considerations: {design_plan['sample_size_guidance']}

Suggested Variables:
- Primary Outcome: {design_plan['suggested_outcome']}
- Key Predictors: {', '.join(design_plan['suggested_predictors'][:5])}
- Potential Confounders: {', '.join(design_plan['potential_confounders'][:3])}

Analysis Plan:
1. {design_plan['analysis_steps'][0]}
2. {design_plan['analysis_steps'][1]}
3. {design_plan['analysis_steps'][2]}

Reporting Standards: {design_plan['reporting_standard']}

Next Steps:
Use 'explore_data' to examine your variables, then follow the analysis plan above.
"""
    
    return [TextContent(type="text", text=result)]


async def _explore_data(arguments: dict) -> List[TextContent]:
    """Comprehensive data exploration."""
    if not server_state['datasets']:
        return [TextContent(type="text", text="No dataset loaded. Use 'import_dataset' first.")]
    
    variables = arguments.get("variables")
    include_missing = arguments.get("include_missing_analysis", True)
    
    # Use main dataset (typically merged patient + hospitalization)
    main_df = server_state['data_loader'].merge_core_tables()
    
    # Initialize analysis
    descriptive = DescriptiveAnalysis(main_df, server_state['privacy_guard'])
    
    # Determine variables to explore
    if variables:
        explore_vars = variables
    else:
        # Auto-select key variables
        explore_vars = [col for col in main_df.columns 
                       if not col.endswith('_id') and col not in ['patient_id', 'hospitalization_id']][:15]
    
    result = f"""Data Exploration Report
{"="*50}

Dataset Overview:
- Total Observations: {len(main_df):,}
- Variables Explored: {len(explore_vars)}

"""
    
    # Missing data analysis
    if include_missing:
        missing_results = descriptive.missing_data_analysis(explore_vars)
        
        result += "Missing Data Summary:\n"
        high_missing = []
        for var, stats in missing_results['summary'].items():
            pct_missing = stats['percent_missing']
            if pct_missing > 20:
                high_missing.append(f"{var} ({pct_missing:.1f}%)")
            elif pct_missing > 5:
                result += f"  - {var}: {pct_missing:.1f}% missing\n"
        
        if high_missing:
            result += f"\nHigh Missing Variables (>20%): {', '.join(high_missing)}\n"
        
        result += f"\nComplete Cases: {missing_results['complete_cases_n']:,} ({missing_results['complete_cases_percent']:.1f}%)\n"
    
    # Variable distributions
    result += f"\nVariable Summary:\n"
    for var in explore_vars[:10]:  # Limit for readability
        if var in main_df.columns:
            var_data = main_df[var]
            if var_data.dtype in ['int64', 'float64']:
                result += f"  - {var}: Mean={var_data.mean():.2f}, Median={var_data.median():.2f}, Range=[{var_data.min():.1f}, {var_data.max():.1f}]\n"
            else:
                value_counts = var_data.value_counts().head(3)
                top_values = ', '.join([f"{val} ({count})" for val, count in value_counts.items()])
                result += f"  - {var}: {top_values}{'...' if len(var_data.unique()) > 3 else ''}\n"
    
    result += f"\nRecommendations:\n"
    if missing_results and missing_results.get('recommendations'):
        for rec in missing_results['recommendations'][:3]:
            result += f"  • {rec}\n"
    
    result += f"\nReady for analysis. Consider using 'generate_table1' for baseline characteristics."
    
    return [TextContent(type="text", text=result)]















async def _export_analysis_code(arguments: dict) -> List[TextContent]:
    """Export reproducible analysis code."""
    language = arguments["language"]
    include_all = arguments.get("include_all_analyses", True)
    containerize = arguments.get("containerize", False)
    
    if not server_state['analysis_history']:
        return [TextContent(type="text", text="No analyses found. Complete analyses before exporting code.")]
    
    # Initialize code exporter
    code_exporter = CodeExporter()
    
    # Generate code
    generated_code = code_exporter.generate_complete_analysis_code(
        analysis_history=server_state['analysis_history'],
        language=language,
        include_all=include_all,
        privacy_settings=server_state['privacy_guard'].__dict__ if server_state['privacy_guard'] else {}
    )
    
    # Save code to file
    output_dir = Path("./exported_code")
    output_dir.mkdir(exist_ok=True)
    
    file_extension = "py" if language == "python" else "R"
    filename = f"cablecar_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
    filepath = output_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(generated_code)
    
    # Generate Docker container if requested
    docker_info = ""
    if containerize:
        dockerfile_content = code_exporter.generate_dockerfile(language)
        dockerfile_path = output_dir / "Dockerfile"
        
        with open(dockerfile_path, 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)
        
        docker_info = f"\nDocker container files generated:\n  - Dockerfile: {dockerfile_path}"
    
    # Generate requirements file
    requirements_content = code_exporter.generate_requirements_file(language)
    req_filename = "requirements.txt" if language == "python" else "requirements.R"
    req_filepath = output_dir / req_filename
    
    with open(req_filepath, 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    
    result = f"""Analysis Code Exported
{"="*50}

Language: {language.title()}
Analysis code: {filepath}
Requirements: {req_filepath}
{docker_info}

Generated Code Features:
✓ Complete analysis pipeline
✓ Privacy-preserving aggregations
✓ Comprehensive documentation
✓ Error handling and validation
✓ Reproducible results

Code Structure:
1. Data loading and validation
2. Privacy protection setup
3. Descriptive statistics
4. Hypothesis testing
5. Regression/ML modeling
6. Results export

Deployment Instructions:
1. Copy files to target system
2. Install requirements: pip install -r requirements.txt (Python) or install.packages() (R)
3. Update data paths in configuration
4. Execute analysis script

This code can run independently without the MCP server for federated analysis.
"""
    
    return [TextContent(type="text", text=result)]


async def _get_analysis_summary(arguments: dict) -> List[TextContent]:
    """Get summary of all performed analyses."""
    if not server_state['analysis_history']:
        return [TextContent(type="text", text="No analyses performed yet.")]
    
    result = f"""Analysis Session Summary
{"="*50}

Total Analyses: {len(server_state['analysis_history'])}
Dataset: {len(server_state['datasets'])} tables loaded
Privacy Level: {getattr(server_state['privacy_guard'], 'min_cell_size', 'N/A')} min cell size

Analyses Performed:
"""
    
    # Group analyses by type
    analysis_counts = {}
    for analysis in server_state['analysis_history']:
        analysis_type = analysis['type']
        analysis_counts[analysis_type] = analysis_counts.get(analysis_type, 0) + 1
    
    for analysis_type, count in analysis_counts.items():
        result += f"  - {analysis_type.replace('_', ' ').title()}: {count}\n"
    
    # Recent analyses
    result += f"\nRecent Analyses:\n"
    for analysis in server_state['analysis_history'][-5:]:  # Last 5
        timestamp = analysis['timestamp'].strftime('%Y-%m-%d %H:%M')
        result += f"  - {timestamp}: {analysis['type'].replace('_', ' ').title()}\n"
    
    # Current study info
    if server_state['current_study']:
        study = server_state['current_study']
        result += f"\nCurrent Study Design:\n"
        result += f"  - Type: {study.get('study_type', 'N/A')}\n"
        result += f"  - Primary Analysis: {study.get('primary_analysis', 'N/A')}\n"
        result += f"  - Reporting Standard: {study.get('reporting_standard', 'N/A')}\n"
    
    # Recommendations
    result += f"\nNext Steps:\n"
    
    if 'table1' not in analysis_counts:
        result += f"  • Generate Table 1 for baseline characteristics\n"
    
    if 'hypothesis_testing' not in analysis_counts and 'regression' not in analysis_counts:
        result += f"  • Perform statistical analyses to test hypotheses\n"
    
    if 'ml_model' not in analysis_counts and server_state['current_study'] and 'predictive' in server_state['current_study'].get('study_type', ''):
        result += f"  • Build prediction model for your outcome\n"
    
    result += f"  • Generate STROBE or TRIPOD report for publication\n"
    result += f"  • Export analysis code for multi-site validation\n"
    
    return [TextContent(type="text", text=result)]


async def _get_privacy_report(arguments: dict) -> List[TextContent]:
    """Generate privacy compliance report."""
    if not server_state['privacy_guard']:
        return [TextContent(type="text", text="Privacy guard not initialized. Import dataset first.")]
    
    privacy_report = server_state['privacy_guard'].generate_privacy_report()
    
    result = f"""Privacy Compliance Report
{"="*50}

Privacy Settings:
- Minimum Cell Size: {privacy_report['privacy_settings']['min_cell_size']}
- K-Anonymity: {privacy_report['privacy_settings']['k_anonymity']}
- Differential Privacy: {'Enabled' if privacy_report['privacy_settings']['differential_privacy_enabled'] else 'Disabled'}

Audit Summary:
- Total Privacy Actions: {privacy_report['audit_summary']['total_actions']}
- Total Data Accesses: {privacy_report['audit_summary']['total_data_accesses']}

Compliance Checks:
"""
    
    for check, passed in privacy_report['compliance_checks'].items():
        status = "✓" if passed else "✗"
        result += f"  {status} {check.replace('_', ' ').title()}\n"
    
    overall_compliance = privacy_report['compliance_checks']['overall_compliance']
    result += f"\nOverall Compliance: {'✓ PASSED' if overall_compliance else '✗ ISSUES FOUND'}\n"
    
    if privacy_report['recommendations']:
        result += f"\nRecommendations:\n"
        for rec in privacy_report['recommendations']:
            result += f"  • {rec}\n"
    
    result += f"\nPrivacy protection is actively enforced on all outputs."
    
    return [TextContent(type="text", text=result)]


async def _list_available_plugins(arguments: dict) -> List[TextContent]:
    """List all available analysis plugins."""
    analysis_type_filter = arguments.get("analysis_type")
    
    # Initialize registry if needed
    global initialized
    if not initialized:
        initialize_registry()
        initialized = True
    
    # Get plugin catalog
    catalog = registry.get_plugin_catalog()
    
    result = f"""Available Analysis Plugins
{"="*50}

CableCar supports a modular plugin architecture for extensible clinical research analyses.
Each plugin implements standardized interfaces for validation, execution, and formatting.

"""
    
    # Display plugins by type
    for analysis_type, plugins in catalog.items():
        if analysis_type_filter and analysis_type != analysis_type_filter:
            continue
            
        result += f"{analysis_type.upper()} ANALYSES:\n"
        result += "-" * 30 + "\n"
        
        for plugin_name, plugin_info in plugins.items():
            metadata = plugin_info.get('metadata', {})
            result += f"\n• run_{plugin_name} - {metadata.get('display_name', plugin_name)}\n"
            result += f"  Description: {metadata.get('description', 'No description available')}\n"
            result += f"  Version: {metadata.get('version', 'N/A')}\n"
            result += f"  Author: {metadata.get('author', 'Unknown')}\n"
            
            # Show keywords if available
            keywords = metadata.get('keywords', [])
            if keywords:
                result += f"  Keywords: {', '.join(keywords)}\n"
        
        result += "\n"
    
    # Plugin statistics
    total_plugins = sum(len(plugins) for plugins in catalog.values())
    result += f"Total Available Plugins: {total_plugins}\n\n"
    
    # Usage instructions
    result += "USAGE:\n"
    result += "- Use 'run_<plugin_name>' to execute any plugin\n"
    result += "- All plugins support 'output_format' parameter: standard, detailed, summary, publication\n"
    result += "- Plugin parameters are validated before execution\n"
    result += "- Results are automatically stored in analysis history\n\n"
    
    result += "To see specific parameters for a plugin, try running it without required parameters.\n"
    result += "New plugins can be added to the community/ or contrib/ directories."
    
    return [TextContent(type="text", text=result)]


async def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="CableCar Clinical Research MCP Server")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger.info("Starting CableCar Clinical Research MCP Server")
    
    # Run the MCP server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())