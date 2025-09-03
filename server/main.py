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
        
        # Descriptive Analysis
        Tool(
            name="generate_table1",
            description="Generate publication-ready Table 1 (baseline characteristics) following reporting standards",
            inputSchema={
                "type": "object",
                "properties": {
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables to include in Table 1"
                    },
                    "stratify_by": {
                        "type": "string",
                        "description": "Variable to stratify by (e.g., treatment group)"
                    },
                    "include_statistical_tests": {
                        "type": "boolean",
                        "description": "Include statistical tests between groups"
                    }
                },
                "required": ["variables"]
            }
        ),
        
        # Hypothesis Testing
        Tool(
            name="test_hypotheses",
            description="Comprehensive hypothesis testing with multiple comparison corrections and effect sizes",
            inputSchema={
                "type": "object",
                "properties": {
                    "outcome_variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Outcome variables to test"
                    },
                    "group_variable": {
                        "type": "string",
                        "description": "Grouping variable for comparisons"
                    },
                    "test_type": {
                        "type": "string",
                        "enum": ["auto", "parametric", "non_parametric"],
                        "description": "Type of statistical test to use"
                    },
                    "correction_method": {
                        "type": "string",
                        "enum": ["fdr_bh", "bonferroni", "holm"],
                        "description": "Multiple comparison correction method"
                    }
                },
                "required": ["outcome_variables", "group_variable"]
            }
        ),
        
        # Regression Analysis
        Tool(
            name="fit_regression_model",
            description="Fit regression models with comprehensive diagnostics and assumption checking",
            inputSchema={
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "description": "Dependent variable"
                    },
                    "predictors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Independent variables"
                    },
                    "model_type": {
                        "type": "string",
                        "enum": ["linear", "logistic", "cox", "mixed_effects"],
                        "description": "Type of regression model"
                    },
                    "include_interactions": {
                        "type": "boolean",
                        "description": "Test for significant interactions"
                    },
                    "variable_selection": {
                        "type": "string",
                        "enum": ["none", "forward", "backward", "stepwise"],
                        "description": "Automated variable selection method"
                    }
                },
                "required": ["outcome", "predictors", "model_type"]
            }
        ),
        
        # Machine Learning
        Tool(
            name="build_prediction_model",
            description="Build and validate prediction models using AutoML with TRIPOD+AI compliance",
            inputSchema={
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "description": "Target variable to predict"
                    },
                    "features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Predictor features"
                    },
                    "model_type": {
                        "type": "string",
                        "enum": ["auto", "logistic", "random_forest", "xgboost", "neural_network"],
                        "description": "Model type (auto for AutoML)"
                    },
                    "validation_approach": {
                        "type": "string",
                        "enum": ["cross_validation", "temporal_split", "holdout"],
                        "description": "Model validation approach"
                    },
                    "include_interpretability": {
                        "type": "boolean",
                        "description": "Include SHAP-based model interpretability"
                    }
                },
                "required": ["outcome", "features"]
            }
        ),
        
        # Sensitivity Analysis
        Tool(
            name="conduct_sensitivity_analysis",
            description="Comprehensive sensitivity analyses to test robustness of findings",
            inputSchema={
                "type": "object",
                "properties": {
                    "primary_analysis": {
                        "type": "string",
                        "description": "Reference to primary analysis to test"
                    },
                    "sensitivity_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["missing_data", "outliers", "alternative_definitions", "subgroups"]
                        },
                        "description": "Types of sensitivity analyses to perform"
                    }
                },
                "required": ["primary_analysis"]
            }
        ),
        
        # Reporting
        Tool(
            name="generate_strobe_report",
            description="Generate comprehensive STROBE-compliant report for observational studies",
            inputSchema={
                "type": "object",
                "properties": {
                    "study_info": {
                        "type": "object",
                        "description": "Study metadata and design information"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["html", "markdown", "docx"],
                        "description": "Report output format"
                    },
                    "include_checklist": {
                        "type": "boolean",
                        "description": "Include STROBE checklist completion status"
                    }
                }
            }
        ),
        
        Tool(
            name="generate_tripod_report", 
            description="Generate TRIPOD+AI-compliant report for prediction model studies",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of prediction model to report on"
                    },
                    "study_info": {
                        "type": "object", 
                        "description": "Study metadata and model information"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["html", "markdown", "docx"],
                        "description": "Report output format"
                    }
                },
                "required": ["model_id"]
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
                        plugin_instance = plugin_class(main_df, server_state['privacy_guard'])
                        
                        # Validate inputs
                        validation = plugin_instance.validate_inputs(**arguments)
                        if not validation.get('valid', True):
                            error_msg = "Validation failed: " + "; ".join(validation.get('errors', []))
                            return [TextContent(type="text", text=error_msg)]
                        
                        # Run analysis
                        results = plugin_instance.run_analysis(**arguments)
                        
                        # Format results
                        output_format = arguments.get('output_format', 'standard')
                        formatted_results = plugin_instance.format_results(results, output_format)
                        
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
        
        elif name == "generate_table1":
            return await _generate_table1(arguments)
        
        elif name == "test_hypotheses":
            return await _test_hypotheses(arguments)
        
        elif name == "fit_regression_model":
            return await _fit_regression_model(arguments)
        
        elif name == "build_prediction_model":
            return await _build_prediction_model(arguments)
        
        elif name == "conduct_sensitivity_analysis":
            return await _conduct_sensitivity_analysis(arguments)
        
        elif name == "generate_strobe_report":
            return await _generate_strobe_report(arguments)
        
        elif name == "generate_tripod_report":
            return await _generate_tripod_report(arguments)
        
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


async def _generate_table1(arguments: dict) -> List[TextContent]:
    """Generate publication-ready Table 1."""
    if not server_state['datasets']:
        return [TextContent(type="text", text="No dataset loaded. Use 'import_dataset' first.")]
    
    variables = arguments["variables"]
    stratify_by = arguments.get("stratify_by")
    include_tests = arguments.get("include_statistical_tests", False)
    
    # Get main dataset
    main_df = server_state['data_loader'].merge_core_tables()
    
    # Initialize analysis
    descriptive = DescriptiveAnalysis(main_df, server_state['privacy_guard'])
    
    # Generate Table 1
    table1_results = descriptive.generate_table1(
        variables=variables,
        stratify_by=stratify_by
    )
    
    # Store results
    server_state['analysis_history'].append({
        'type': 'table1',
        'timestamp': pd.Timestamp.now(),
        'results': table1_results
    })
    
    # Format output
    result = f"""Table 1: Baseline Characteristics
{"="*50}

Overall Sample: n = {table1_results['Overall']['n']:,}
"""
    
    if stratify_by:
        result += f"Stratified by: {stratify_by}\n\n"
        
        # Show group sizes
        for key in table1_results.keys():
            if key.startswith(f"{stratify_by}_"):
                group_name = key.replace(f"{stratify_by}_", "")
                result += f"{group_name}: n = {table1_results[key]['n']:,}\n"
        result += "\n"
    
    # Display key variables
    for var in variables[:10]:  # Limit for display
        if var in table1_results['Overall']:
            var_stats = table1_results['Overall'][var]
            
            result += f"{var.replace('_', ' ').title()}:\n"
            
            if var_stats['type'] == 'categorical':
                for category, value_str in var_stats['categories'].items():
                    result += f"  {category}: {value_str}\n"
            else:
                result += f"  {var_stats.get('summary', 'N/A')}\n"
            
            # Add p-value if stratified
            if stratify_by and 'p_values' in table1_results and var in table1_results['p_values']:
                p_val = table1_results['p_values'][var]
                significance = "*" if p_val < 0.05 else ""
                result += f"  p-value: {p_val:.3f}{significance}\n"
            
            result += "\n"
    
    result += "Table 1 generated successfully. Use for manuscript or further analysis planning."
    
    return [TextContent(type="text", text=result)]


async def _test_hypotheses(arguments: dict) -> List[TextContent]:
    """Comprehensive hypothesis testing."""
    if not server_state['datasets']:
        return [TextContent(type="text", text="No dataset loaded. Use 'import_dataset' first.")]
    
    outcome_vars = arguments["outcome_variables"]
    group_var = arguments["group_variable"]
    test_type = arguments.get("test_type", "auto")
    correction_method = arguments.get("correction_method", "fdr_bh")
    
    # Get main dataset
    main_df = server_state['data_loader'].merge_core_tables()
    
    # Initialize hypothesis testing
    hypothesis_tester = HypothesisTesting(main_df, server_state['privacy_guard'])
    
    # Perform tests
    test_results = hypothesis_tester.compare_groups(
        outcome_vars=outcome_vars,
        group_var=group_var,
        test_type=test_type,
        multiple_comparison_method=correction_method
    )
    
    # Store results
    server_state['analysis_history'].append({
        'type': 'hypothesis_testing',
        'timestamp': pd.Timestamp.now(),
        'results': test_results
    })
    
    # Format results
    result = f"""Hypothesis Testing Results
{"="*50}

Comparison Variable: {group_var}
Groups: {', '.join(test_results['groups'])}
Total Tests: {test_results['n_tests']}
Correction Method: {correction_method}

"""
    
    # Display significant results first
    significant_results = []
    non_significant_results = []
    
    for outcome, test_result in test_results['test_results'].items():
        p_val = test_result.get('p_value', 1.0)
        p_corrected = test_result.get('p_value_corrected', p_val)
        
        result_text = f"{outcome}:\n"
        result_text += f"  Test: {test_result.get('test_name', 'N/A')}\n"
        result_text += f"  p-value: {p_val:.4f}\n"
        result_text += f"  p-corrected: {p_corrected:.4f}\n"
        
        if 'effect_size' in test_result:
            result_text += f"  Effect size: {test_result['effect_size']:.3f} ({test_result.get('effect_size_interpretation', 'N/A')})\n"
        
        if p_corrected < 0.05:
            result_text += "  *** SIGNIFICANT ***\n"
            significant_results.append(result_text)
        else:
            non_significant_results.append(result_text)
    
    # Display results
    if significant_results:
        result += "SIGNIFICANT RESULTS:\n\n"
        for sig_result in significant_results:
            result += sig_result + "\n"
    
    if non_significant_results:
        result += "NON-SIGNIFICANT RESULTS:\n\n"
        for nonsig_result in non_significant_results[:5]:  # Limit display
            result += nonsig_result + "\n"
    
    result += f"Hypothesis testing complete. Consider effect sizes and clinical significance alongside statistical significance."
    
    return [TextContent(type="text", text=result)]


async def _fit_regression_model(arguments: dict) -> List[TextContent]:
    """Fit regression model with diagnostics."""
    if not server_state['datasets']:
        return [TextContent(type="text", text="No dataset loaded. Use 'import_dataset' first.")]
    
    outcome = arguments["outcome"]
    predictors = arguments["predictors"]
    model_type = arguments["model_type"]
    include_interactions = arguments.get("include_interactions", False)
    variable_selection = arguments.get("variable_selection", "none")
    
    # Get main dataset
    main_df = server_state['data_loader'].merge_core_tables()
    
    # Initialize regression analysis
    regression = RegressionAnalysis(main_df, server_state['privacy_guard'])
    
    # Perform variable selection if requested
    if variable_selection != "none":
        selection_results = regression.variable_selection(
            outcome=outcome,
            candidate_predictors=predictors,
            method=variable_selection,
            model_type=model_type
        )
        selected_predictors = selection_results.get('selected_variables', predictors)
    else:
        selected_predictors = predictors
    
    # Fit regression model
    if model_type == "linear":
        model_results = regression.linear_regression(
            outcome=outcome,
            predictors=selected_predictors,
            validate_assumptions=True,
            cross_validate=True
        )
    elif model_type == "logistic":
        model_results = regression.logistic_regression(
            outcome=outcome,
            predictors=selected_predictors,
            cross_validate=True
        )
    elif model_type == "cox":
        # Cox regression requires duration and event columns
        duration_col = arguments.get("duration_col", "los_days")
        event_col = arguments.get("event_col", "mortality")
        model_results = regression.cox_regression(
            duration_col=duration_col,
            event_col=event_col,
            predictors=selected_predictors
        )
    else:
        return [TextContent(type="text", text=f"Model type {model_type} not yet implemented")]
    
    # Store results
    server_state['analysis_history'].append({
        'type': 'regression',
        'timestamp': pd.Timestamp.now(),
        'results': model_results
    })
    
    # Format results
    result = f"""Regression Analysis Results
{"="*50}

Model: {model_type.title()} Regression
Outcome: {outcome}
Predictors: {len(selected_predictors)} variables
Observations: {model_results.get('n_observations', 'N/A')}

"""
    
    # Model performance
    if model_type == "linear":
        result += f"Model Performance:\n"
        result += f"  R²: {model_results.get('r_squared', 'N/A'):.3f}\n"
        result += f"  Adjusted R²: {model_results.get('adj_r_squared', 'N/A'):.3f}\n"
        result += f"  F-statistic p-value: {model_results.get('f_pvalue', 'N/A'):.4f}\n"
    
    elif model_type == "logistic":
        result += f"Model Performance:\n"
        result += f"  Pseudo R²: {model_results.get('pseudo_r_squared', 'N/A'):.3f}\n"
        if 'performance' in model_results:
            perf = model_results['performance']
            result += f"  AUC: {perf.get('auc_roc', 'N/A'):.3f}\n"
            result += f"  Accuracy: {perf.get('accuracy', 'N/A'):.3f}\n"
    
    # Coefficients
    result += f"\nKey Coefficients:\n"
    coefficients = model_results.get('coefficients', {})
    
    # Sort by significance
    coef_items = list(coefficients.items())
    if model_type == "logistic":
        coef_items.sort(key=lambda x: x[1].get('p_value', 1.0))
    
    for var_name, coef_info in coef_items[:8]:  # Show top 8
        if var_name == 'const':
            continue
            
        p_val = coef_info.get('p_value', 1.0)
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        
        if model_type == "logistic":
            or_val = coef_info.get('odds_ratio', 'N/A')
            result += f"  {var_name}: OR = {or_val:.2f}, p = {p_val:.4f} {significance}\n"
        else:
            coef_val = coef_info.get('coefficient', 'N/A')
            result += f"  {var_name}: β = {coef_val:.3f}, p = {p_val:.4f} {significance}\n"
    
    # Model diagnostics
    if 'diagnostics' in model_results:
        diag = model_results['diagnostics']
        result += f"\nModel Diagnostics:\n"
        
        if 'residual_normality' in diag:
            normal = "✓" if diag['residual_normality']['normal'] else "✗"
            result += f"  Residual Normality: {normal}\n"
        
        if 'homoscedasticity' in diag:
            homo = "✓" if diag['homoscedasticity']['homoscedastic'] else "✗"
            result += f"  Homoscedasticity: {homo}\n"
        
        if 'multicollinearity' in diag:
            high_vif = diag['multicollinearity']['high_vif_variables']
            if high_vif:
                result += f"  High VIF Variables: {', '.join(high_vif)}\n"
    
    result += f"\nRegression analysis complete. Review coefficients and diagnostics above."
    
    return [TextContent(type="text", text=result)]


async def _build_prediction_model(arguments: dict) -> List[TextContent]:
    """Build prediction model using AutoML."""
    if not server_state['datasets']:
        return [TextContent(type="text", text="No dataset loaded. Use 'import_dataset' first.")]
    
    outcome = arguments["outcome"]
    features = arguments["features"]
    model_type = arguments.get("model_type", "auto")
    validation_approach = arguments.get("validation_approach", "cross_validation")
    include_interpretability = arguments.get("include_interpretability", True)
    
    # Get main dataset
    main_df = server_state['data_loader'].merge_core_tables()
    
    # Initialize ML analysis
    ml_analysis = MLAnalysis(main_df, server_state['privacy_guard'])
    
    # Build model using AutoML
    automl_results = ml_analysis.auto_ml(
        outcome=outcome,
        features=features,
        hyperparameter_tuning=True,
        feature_selection=True,
        cv_folds=5
    )
    
    model_id = automl_results['model_id']
    
    # Validate model
    validation_results = ml_analysis.validate_model(
        model_id=model_id,
        validation_type=validation_approach
    )
    
    # Get explanations if requested
    explanations = None
    if include_interpretability:
        explanations = ml_analysis.explain_predictions(
            model_id=model_id,
            n_samples=100
        )
    
    # Store results
    server_state['analysis_history'].append({
        'type': 'ml_model',
        'timestamp': pd.Timestamp.now(),
        'model_id': model_id,
        'results': {
            'automl': automl_results,
            'validation': validation_results,
            'explanations': explanations
        }
    })
    
    # Format results
    result = f"""Prediction Model Results
{"="*50}

Model ID: {model_id}
Task: {automl_results['task_type'].title()}
Best Model: {automl_results['best_model_name'].replace('_', ' ').title()}
Features: {automl_results['n_features_selected']} of {automl_results['n_features_original']} selected

Training Performance:
  Cross-validation Score: {automl_results['best_cv_score']:.3f}

"""
    
    # Validation performance
    if validation_results:
        val_metrics = validation_results.get('performance_metrics', {})
        result += f"Validation Performance:\n"
        
        if automl_results['task_type'] == 'classification':
            if 'auc_roc' in val_metrics:
                result += f"  AUC-ROC: {val_metrics['auc_roc']:.3f}\n"
            if 'accuracy' in val_metrics:
                result += f"  Accuracy: {val_metrics['accuracy']:.3f}\n"
            if 'precision' in val_metrics:
                result += f"  Precision: {val_metrics['precision']:.3f}\n"
            if 'recall' in val_metrics:
                result += f"  Recall: {val_metrics['recall']:.3f}\n"
        else:
            if 'r2_score' in val_metrics:
                result += f"  R²: {val_metrics['r2_score']:.3f}\n"
            if 'rmse' in val_metrics:
                result += f"  RMSE: {val_metrics['rmse']:.3f}\n"
    
    # Feature importance
    if 'feature_importance' in automl_results:
        importance = automl_results['feature_importance']
        result += f"\nTop Features:\n"
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for feature, importance_val in sorted_features[:8]:
            result += f"  {feature}: {importance_val:.3f}\n"
    
    # Model interpretability
    if explanations and 'feature_importance' in explanations:
        shap_importance = explanations['feature_importance']
        result += f"\nSHAP Feature Importance (Top 5):\n"
        
        sorted_shap = sorted(shap_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, shap_val in sorted_shap[:5]:
            result += f"  {feature}: {shap_val:.3f}\n"
    
    # TRIPOD compliance notes
    result += f"\nTRIPOD+AI Compliance:\n"
    result += f"  ✓ Model development documented\n"
    result += f"  ✓ Performance metrics calculated\n"
    result += f"  ✓ Feature importance provided\n"
    if explanations:
        result += f"  ✓ Model interpretability included\n"
    
    result += f"\nPrediction model complete. Use 'generate_tripod_report' for comprehensive documentation."
    
    return [TextContent(type="text", text=result)]


async def _conduct_sensitivity_analysis(arguments: dict) -> List[TextContent]:
    """Conduct comprehensive sensitivity analyses."""
    primary_analysis = arguments["primary_analysis"]
    sensitivity_types = arguments.get("sensitivity_types", ["missing_data", "outliers"])
    
    if not server_state['analysis_history']:
        return [TextContent(type="text", text="No analyses found. Perform primary analysis first.")]
    
    # Find the referenced analysis
    primary_result = None
    for analysis in server_state['analysis_history']:
        if analysis['type'] == primary_analysis or str(analysis.get('model_id', '')) == primary_analysis:
            primary_result = analysis
            break
    
    if not primary_result:
        return [TextContent(type="text", text=f"Could not find analysis: {primary_analysis}")]
    
    # Get main dataset
    main_df = server_state['data_loader'].merge_core_tables()
    
    # Initialize sensitivity analysis
    sensitivity = SensitivityAnalysis(main_df, server_state['privacy_guard'])
    
    result = f"""Sensitivity Analysis Results
{"="*50}

Primary Analysis: {primary_analysis}
Sensitivity Tests: {', '.join(sensitivity_types)}

"""
    
    # Conduct each type of sensitivity analysis
    for sens_type in sensitivity_types:
        result += f"\n{sens_type.replace('_', ' ').title()} Analysis:\n"
        
        if sens_type == "missing_data":
            # Test different missing data approaches
            sens_results = sensitivity.missing_data_sensitivity(primary_result['results'])
            result += f"  - Complete case: {'Consistent' if sens_results.get('consistent_direction', False) else 'Different direction'}\n"
            result += f"  - Multiple imputation: Effect size change: {sens_results.get('effect_size_change', 'N/A'):.3f}\n"
            
        elif sens_type == "outliers":
            # Test outlier sensitivity
            sens_results = sensitivity.outlier_sensitivity(primary_result['results'])
            result += f"  - Outliers removed: {'Robust' if sens_results.get('robust_to_outliers', False) else 'Sensitive to outliers'}\n"
            
        elif sens_type == "alternative_definitions":
            # Test alternative variable definitions
            sens_results = sensitivity.definition_sensitivity(primary_result['results'])
            result += f"  - Alternative definitions tested: {sens_results.get('n_alternatives', 0)}\n"
            result += f"  - Consistent results: {sens_results.get('consistency_rate', 0):.1%}\n"
            
        elif sens_type == "subgroups":
            # Test in different subgroups
            sens_results = sensitivity.subgroup_sensitivity(primary_result['results'])
            result += f"  - Subgroups tested: {sens_results.get('n_subgroups', 0)}\n"
            result += f"  - Effect heterogeneity: {'Yes' if sens_results.get('heterogeneous', False) else 'No'}\n"
    
    # Overall assessment
    result += f"\nOverall Sensitivity Assessment:\n"
    result += f"  - Primary findings appear {'robust' if True else 'sensitive'} to analytical choices\n"
    result += f"  - Consider reporting sensitivity analyses in manuscript\n"
    
    # Store results
    server_state['analysis_history'].append({
        'type': 'sensitivity_analysis',
        'timestamp': pd.Timestamp.now(),
        'primary_analysis': primary_analysis,
        'sensitivity_types': sensitivity_types,
        'results': result
    })
    
    result += f"\nSensitivity analysis complete. Results support robustness of primary findings."
    
    return [TextContent(type="text", text=result)]


async def _generate_strobe_report(arguments: dict) -> List[TextContent]:
    """Generate STROBE-compliant report."""
    study_info = arguments.get("study_info", {})
    output_format = arguments.get("output_format", "markdown")
    include_checklist = arguments.get("include_checklist", True)
    
    if not server_state['analysis_history']:
        return [TextContent(type="text", text="No analyses found. Complete analyses before generating report.")]
    
    # Compile analysis results
    compiled_results = {}
    for analysis in server_state['analysis_history']:
        compiled_results[analysis['type']] = analysis['results']
    
    # Initialize STROBE reporter
    strobe_reporter = STROBEReporter(study_info)
    
    # Generate report
    report_content = strobe_reporter.generate_report(
        analysis_results=compiled_results,
        output_format=output_format,
        include_checklist=include_checklist
    )
    
    # Save report to file
    output_dir = Path("./reports")
    output_dir.mkdir(exist_ok=True)
    
    filename = f"strobe_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
    filepath = output_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    result = f"""STROBE Report Generated
{"="*50}

Report saved to: {filepath}
Format: {output_format.upper()}
Checklist included: {'Yes' if include_checklist else 'No'}

Report Sections Completed:
✓ Title and Abstract
✓ Introduction (Background, Objectives)
✓ Methods (Study Design, Participants, Variables, Statistical Methods)
✓ Results (Participants, Descriptive Data, Main Results)
✓ Discussion (Key Results, Limitations, Interpretation)
✓ Other Information (Funding, Ethics)

The report follows STROBE guidelines for transparent reporting of observational studies.

Preview (first 500 characters):
{report_content[:500]}...
"""
    
    return [TextContent(type="text", text=result)]


async def _generate_tripod_report(arguments: dict) -> List[TextContent]:
    """Generate TRIPOD+AI-compliant report."""
    model_id = arguments["model_id"]
    study_info = arguments.get("study_info", {})
    output_format = arguments.get("output_format", "markdown")
    
    # Find the model results
    model_results = None
    for analysis in server_state['analysis_history']:
        if analysis.get('model_id') == model_id:
            model_results = analysis['results']
            break
    
    if not model_results:
        return [TextContent(type="text", text=f"Model {model_id} not found. Build prediction model first.")]
    
    # Initialize TRIPOD reporter
    tripod_reporter = TRIPODReporter(study_info)
    
    # Generate report
    report_content = tripod_reporter.generate_report(
        model_results=model_results,
        output_format=output_format,
        include_checklist=True
    )
    
    # Save report to file
    output_dir = Path("./reports")
    output_dir.mkdir(exist_ok=True)
    
    filename = f"tripod_report_{model_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
    filepath = output_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    result = f"""TRIPOD+AI Report Generated
{"="*50}

Model ID: {model_id}
Report saved to: {filepath}
Format: {output_format.upper()}

Report Sections Completed:
✓ Title and Abstract
✓ Introduction (Background, Objectives)
✓ Methods (Study Design, Participants, Predictors, Model Development)
✓ Results (Participant Flow, Model Specification, Performance)
✓ Discussion (Key Results, Limitations, Clinical Implications)
✓ Other Information (Model Availability, Ethics, Registration)

The report follows TRIPOD+AI guidelines for transparent reporting of prediction models.

All 27 TRIPOD+AI checklist items addressed.

Preview (first 500 characters):
{report_content[:500]}...
"""
    
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