#!/usr/bin/env python
"""
CLIF MCP Server - Consolidated Implementation
Combines the best features from both implementations with clean architecture
"""

import asyncio
import logging
import sys
import json
from pathlib import Path
import argparse

import pandas as pd

from mcp.server import Server  # type: ignore
from mcp.server.stdio import stdio_server  # type: ignore
from mcp.types import Tool, TextContent  # type: ignore

# Import all tool classes
from server.tools.cohort_builder import CohortBuilder
from server.tools.outcomes_analyzer import OutcomesAnalyzer
from server.tools.code_generator import CodeGenerator
from server.tools.data_explorer import DataExplorer
from server.tools.ml_model_builder import MLModelBuilder
from server.tools.comprehensive_dictionary import DynamicCLIFDictionary as ComprehensiveDictionary
from server.tools.derived_variable_creator import DerivedVariableCreator
from server.security.privacy_guard import PrivacyGuard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
app = Server("clif-mcp-server")
current_cohort = None
last_regression_config = None
data_dictionary = {}


class CLIFServer:
    """Main CLIF MCP Server with consolidated functionality"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
        # Initialize tools
        self.cohort_builder = CohortBuilder(data_path)
        self.outcomes_analyzer = OutcomesAnalyzer(data_path)
        self.code_generator = CodeGenerator()
        self.data_explorer = DataExplorer(data_path)
        self.ml_model_builder = MLModelBuilder(data_path)
        self.comprehensive_dictionary = ComprehensiveDictionary(data_path)
        self.derived_variable_creator = DerivedVariableCreator(data_path)
        self.privacy_guard = PrivacyGuard()
        
        # Build data dictionary on startup
        self._build_data_dictionary()
        
    def _build_data_dictionary(self):
        """Build comprehensive data dictionary from actual data"""
        global data_dictionary
        
        try:
            # Get the dictionary info from comprehensive_dictionary
            data_dictionary = self.comprehensive_dictionary.get_variable_info()
            
            # Add actual categorical values from data
            categorical_values = {}
            for table_name in self.data_explorer.tables:
                try:
                    df = pd.read_csv(self.data_path / f"{table_name}.csv", nrows=1000)
                    for col in df.select_dtypes(include=['object']).columns:
                        unique_vals = df[col].dropna().unique()
                        categorical_values[f"{table_name}.{col}"] = list(unique_vals)[:20]
                except Exception as e:
                    logger.warning(f"Could not read {table_name}: {e}")
            
            data_dictionary["actual_categorical_values"] = categorical_values
            data_dictionary["available_tables"] = self.data_explorer.tables
            
            logger.info(f"Data dictionary built with {len(data_dictionary)} entries")
            
        except Exception as e:
            logger.error(f"Error building data dictionary: {e}")
            data_dictionary = {"error": "Could not build data dictionary"}


# Initialize server instance
clif_server = None


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available CLIF analysis tools"""
    
    return [
        Tool(
            name="get_data_context",
            description="Get essential context about available data including tables, variables, and common values - use this FIRST before building cohorts",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="show_data_dictionary",
            description="Show comprehensive data dictionary with all available variables",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="explore_data",
            description="Explore CLIF dataset schema and summary statistics",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Specific table to explore (optional)"
                    }
                }
            }
        ),
        Tool(
            name="build_cohort",
            description="Build patient cohort with flexible criteria",
            inputSchema={
                "type": "object",
                "properties": {
                    "criteria": {
                        "type": "object",
                        "description": "Cohort selection criteria",
                        "properties": {
                            "age_range": {
                                "type": "object",
                                "properties": {
                                    "min": {"type": "number"},
                                    "max": {"type": "number"}
                                }
                            },
                            "require_mechanical_ventilation": {"type": "boolean"},
                            "icu_los_range": {
                                "type": "object",
                                "properties": {
                                    "min": {"type": "number"},
                                    "max": {"type": "number"}
                                }
                            },
                            "exclude_readmissions": {"type": "boolean"},
                            "diagnoses": {"type": "array", "items": {"type": "string"}},
                            "medications": {"type": "array", "items": {"type": "string"}},
                            "lab_criteria": {
                                "type": "array",
                                "description": "Lab-based criteria for identifying clinical conditions using time-series lab data",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "lab_name": {
                                            "type": "string",
                                            "description": "Lab test name from labs table (e.g., 'Creatinine', 'Lactate', 'Platelet')"
                                        },
                                        "condition": {
                                            "type": "string",
                                            "enum": ["increase", "decrease", "above", "below", "between"],
                                            "description": "Type of condition to check in the time-series data"
                                        },
                                        "value": {
                                            "type": "number",
                                            "description": "For 'above'/'below': threshold value. For 'increase'/'decrease': multiplication factor (e.g., 1.5 for 50% increase)"
                                        },
                                        "absolute_increase": {
                                            "type": "number",
                                            "description": "Alternative to multiplication factor - absolute increase amount (e.g., 0.3 for creatinine)"
                                        },
                                        "time_window_hours": {
                                            "type": "number",
                                            "description": "Time window to check for the condition (e.g., 48 hours)",
                                            "default": 48
                                        },
                                        "baseline_window_hours": {
                                            "type": "number",
                                            "description": "For 'increase'/'decrease': how far back to look for baseline value",
                                            "default": 168
                                        },
                                        "min_value": {
                                            "type": "number",
                                            "description": "For 'between' condition: minimum value"
                                        },
                                        "max_value": {
                                            "type": "number",
                                            "description": "For 'between' condition: maximum value"
                                        }
                                    },
                                    "required": ["lab_name", "condition"],
                                    "examples": [
                                        {
                                            "description": "AKI by creatinine increase",
                                            "lab_name": "Creatinine",
                                            "condition": "increase",
                                            "value": 1.5,
                                            "time_window_hours": 48
                                        },
                                        {
                                            "description": "Elevated lactate",
                                            "lab_name": "Lactate",
                                            "condition": "above",
                                            "value": 2.0
                                        }
                                    ]
                                }
                            }
                        }
                    },
                    "save_cohort": {
                        "type": "boolean",
                        "description": "Save cohort for reuse"
                    }
                },
                "required": ["criteria"]
            }
        ),
        Tool(
            name="analyze_outcomes",
            description="Analyze clinical outcomes for cohorts",
            inputSchema={
                "type": "object",
                "properties": {
                    "cohort_id": {
                        "type": "string",
                        "description": "ID of previously saved cohort (optional)"
                    },
                    "outcomes": {
                        "type": "array",
                        "description": "Outcomes to analyze (e.g., mortality, icu_los, hospital_los, ventilator_days, or any numeric column)",
                        "items": {"type": "string"}
                    },
                    "stratify_by": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["outcomes"]
            }
        ),
        Tool(
            name="compare_cohorts",
            description="Compare outcomes between two cohorts",
            inputSchema={
                "type": "object",
                "properties": {
                    "cohort1_id": {"type": "string"},
                    "cohort2_id": {"type": "string"},
                    "outcomes": {
                        "type": "array",
                        "description": "Outcomes to compare between cohorts",
                        "items": {"type": "string"}
                    },
                    "adjustment_method": {
                        "type": "string",
                        "enum": ["unadjusted", "regression", "propensity_score", "iptw"]
                    }
                },
                "required": ["cohort1_id", "cohort2_id", "outcomes"]
            }
        ),
        Tool(
            name="fit_regression",
            description="Fit various regression models (linear, logistic, poisson, negative_binomial, cox, mixed)",
            inputSchema={
                "type": "object",
                "properties": {
                    "outcome": {"type": "string"},
                    "predictors": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "model_type": {
                        "type": "string",
                        "enum": ["linear", "logistic", "poisson", "negative_binomial", "cox", "mixed_linear"]
                    },
                    "interaction_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Interaction terms like 'age*sex'"
                    },
                    "random_effects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Random effects for mixed models"
                    }
                },
                "required": ["outcome", "predictors", "model_type"]
            }
        ),
        Tool(
            name="descriptive_stats",
            description="Generate descriptive statistics for variables",
            inputSchema={
                "type": "object",
                "properties": {
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "stratify_by": {
                        "type": "string",
                        "description": "Variable to stratify by"
                    }
                },
                "required": ["variables"]
            }
        ),
        Tool(
            name="build_ml_model",
            description="Build machine learning models for prediction",
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
                        "description": "Feature columns for prediction"
                    },
                    "model_type": {
                        "type": "string",
                        "description": "Model type (auto, logistic_regression, random_forest, xgboost, etc.)",
                        "default": "auto"
                    },
                    "feature_selection": {
                        "type": "object",
                        "properties": {
                            "method": {
                                "type": "string",
                                "enum": ["kbest", "mutual_info", "rfe"]
                            },
                            "n_features": {"type": "integer"}
                        }
                    },
                    "hyperparameter_tuning": {
                        "type": "boolean",
                        "default": True
                    }
                },
                "required": ["outcome", "features"]
            }
        ),
        Tool(
            name="generate_code",
            description="Generate reproducible analysis code",
            inputSchema={
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["python", "r"]
                    },
                    "include_all": {
                        "type": "boolean",
                        "description": "Include all analyses performed"
                    }
                },
                "required": ["language"]
            }
        ),
        Tool(
            name="create_derived_variable",
            description="Create new derived variables from existing data at different levels (timepoint, hospitalization, patient)",
            inputSchema={
                "type": "object",
                "properties": {
                    "variable_name": {
                        "type": "string",
                        "description": "Name of the new variable to create"
                    },
                    "variable_type": {
                        "type": "string",
                        "enum": ["binary", "categorical", "numeric", "datetime"],
                        "description": "Type of the new variable"
                    },
                    "level": {
                        "type": "string",
                        "enum": ["timepoint", "hospitalization", "patient"],
                        "description": "Level at which to create the variable"
                    },
                    "definition": {
                        "type": "object",
                        "description": "Definition of how to create the variable",
                        "properties": {
                            "source_table": {
                                "type": "string",
                                "description": "Source table for custom variables"
                            },
                            "filters": {
                                "type": "object",
                                "description": "Filters to apply to source data"
                            },
                            "aggregation": {
                                "type": "object",
                                "properties": {
                                    "method": {
                                        "type": "string",
                                        "enum": ["count", "sum", "mean", "max", "min", "first", "last"]
                                    },
                                    "field": {"type": "string"},
                                    "group_by": {"type": "string"}
                                }
                            },
                            "transformation": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "threshold": {"type": "number"},
                                    "operator": {"type": "string"},
                                    "values": {"type": "array"}
                                }
                            },
                            "calculation": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["ratio", "difference"]
                                    },
                                    "numerator": {"type": "string"},
                                    "denominator": {"type": "string"},
                                    "var1": {"type": "string"},
                                    "var2": {"type": "string"},
                                    "tolerance_minutes": {"type": "number"}
                                }
                            }
                        }
                    },
                    "predefined": {
                        "type": "string",
                        "enum": ["mortality", "los_days", "icu_los_days", "mechanical_ventilation", 
                                "ventilator_days", "aki_stage", "max_sofa_score", "total_hospitalizations",
                                "total_icu_days", "ever_ventilated", "mortality_any_admission", "map", 
                                "shock_index"],
                        "description": "Use a predefined common variable instead of custom definition"
                    }
                },
                "required": ["variable_name", "level"]
            }
        ),
        Tool(
            name="list_derived_variables",
            description="List available predefined derived variables and their definitions",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute CLIF analysis tools"""
    global current_cohort, last_regression_config
    
    try:
        if name == "get_data_context":
            # Build a concise summary of available data
            result = "CLIF DATA CONTEXT\n"
            result += "=" * 50 + "\n\n"
            
            # Add CLIF explanation
            result += "CLIF stands for Common Longitudinal ICU Format - a standardized data format\n"
            result += "for intensive care unit patient data that enables multi-site clinical research.\n\n"
            
            # Available tables
            result += "AVAILABLE TABLES:\n"
            for table in clif_server.data_explorer.tables:
                result += f"  • {table}\n"
            
            # Key lab tests
            result += "\nKEY LAB TESTS:\n"
            lab_df = pd.read_csv(clif_server.data_path / "labs.csv", nrows=1000)
            if 'lab_category' in lab_df.columns:
                lab_tests = lab_df['lab_category'].unique()
                for test in sorted(lab_tests)[:15]:
                    result += f"  • {test}\n"
            
            # Key medications
            result += "\nCOMMON MEDICATIONS:\n"
            med_df = pd.read_csv(clif_server.data_path / "medication_administration.csv", nrows=1000)
            if 'medication_name' in med_df.columns:
                meds = med_df['medication_name'].value_counts().head(10)
                for med in meds.index[:10]:
                    result += f"  • {med}\n"
            
            result += "\nCOHORT BUILDING TIPS:\n"
            result += "- Age criteria: use age_range with min/max\n"
            result += "- Lab criteria: specify lab_name, condition (above/below/increase), value, and time_window_hours\n"
            result += "- Medications: list medication names to filter by\n"
            result += "- Ventilation: set require_mechanical_ventilation to true/false\n"
            
            return [TextContent(type="text", text=result)]
            
        elif name == "show_data_dictionary":
            # Ensure data dictionary is built
            if not data_dictionary:
                clif_server._build_data_dictionary()
            
            # Get fresh data from comprehensive dictionary
            all_info = clif_server.comprehensive_dictionary.get_variable_info()
            
            result = "COMPREHENSIVE CLIF DATA DICTIONARY\n"
            result += "=" * 50 + "\n"
            
            # Show regular tables
            for table_name, variables in all_info.items():
                if isinstance(variables, dict) and table_name != 'derived_variables':
                    result += f"\n{table_name.upper()} TABLE:\n"
                    result += "-" * 30 + "\n"
                    if variables:
                        for var_name, var_info in variables.items():
                            if isinstance(var_info, dict):
                                var_type = var_info.get('type', 'unknown')
                                desc = var_info.get('description', '')
                                result += f"  • {var_name} ({var_type}): {desc}\n"
                    else:
                        result += "  (No variables found in this table)\n"
            
            # Show derived variables
            if 'derived_variables' in all_info:
                result += "\nDERIVED VARIABLES:\n"
                result += "-" * 30 + "\n"
                for var_name, var_info in all_info['derived_variables'].items():
                    if isinstance(var_info, dict) and var_info.get('available', False):
                        var_type = var_info.get('type', 'unknown')
                        desc = var_info.get('description', '')
                        result += f"  • {var_name} ({var_type}): {desc}\n"
            
            # Add categorical values if available
            if data_dictionary and 'actual_categorical_values' in data_dictionary:
                result += "\nCATEGORICAL VALUE EXAMPLES:\n"
                result += "-" * 30 + "\n"
                for var_path, values in list(data_dictionary['actual_categorical_values'].items())[:10]:
                    if values:
                        result += f"  • {var_path}: {', '.join(str(v) for v in values[:5])}\n"
                        if len(values) > 5:
                            result += f"    ... and {len(values) - 5} more values\n"
            
            if not all_info or (isinstance(all_info, dict) and not any(v for k, v in all_info.items() if k != 'derived_variables')):
                result += "\nNote: No data dictionary found. This may be because:\n"
                result += "1. The data path is incorrect\n"
                result += "2. The CSV files are not in the expected format\n"
                result += "3. The tables are empty\n"
                
            return [TextContent(type="text", text=result)]
            
        elif name == "explore_data":
            table_name = arguments.get("table_name")
            result = clif_server.data_explorer.explore_schema(table_name)
            return [TextContent(type="text", text=clif_server.privacy_guard.sanitize_output(result))]
            
        elif name == "build_cohort":
            criteria = arguments["criteria"]
            save_cohort = arguments.get("save_cohort", False)
            
            # Include helpful context about available variables
            available_labs = []
            if 'labs' in data_dictionary:
                available_labs = list(data_dictionary['labs'].keys())
            
            cohort_df, cohort_id = clif_server.cohort_builder.build_cohort(criteria)
            current_cohort = cohort_df
            
            if len(cohort_df) == 0:
                result = f"""No patients found matching the specified criteria.

Criteria used: {json.dumps(criteria, indent=2)}

Available lab tests in the data: {', '.join(available_labs[:10])}
For lab criteria, use format:
{{
    "lab_criteria": [{{
        "lab_name": "Creatinine",
        "condition": "above",
        "value": 1.5,
        "time_window_hours": 24
    }}]
}}
"""
                return [TextContent(type="text", text=result)]
            
            if save_cohort:
                clif_server.cohort_builder.save_cohort(cohort_id, cohort_df, criteria)
            
            # Get characteristics
            chars = clif_server.cohort_builder.get_cohort_characteristics(cohort_df)
            
            result = f"""Cohort built successfully!
Cohort ID: {cohort_id}
Size: {len(cohort_df)} hospitalizations
Unique patients: {cohort_df['patient_id'].nunique()}

Demographics:
- Mean age: {chars['demographics']['age_mean']:.1f} years
- Sex distribution: {chars['demographics']['sex_distribution']}

Clinical Characteristics:
- Mortality rate: {chars['clinical']['mortality_rate']:.1%}
- Mean ICU LOS: {chars['clinical']['icu_los_mean']:.1f} days
- Mechanical ventilation: {chars['clinical']['ventilation_rate']:.1%}
"""
            
            return [TextContent(type="text", text=clif_server.privacy_guard.sanitize_output(result))]
            
        elif name == "analyze_outcomes":
            cohort_id = arguments.get("cohort_id")
            outcomes = arguments["outcomes"]
            stratify_by = arguments.get("stratify_by", [])
            
            results = clif_server.outcomes_analyzer.analyze_outcomes(
                cohort_id, outcomes, stratify_by, []
            )
            
            formatted_results = clif_server.outcomes_analyzer.format_results(results)
            return [TextContent(type="text", text=clif_server.privacy_guard.sanitize_output(formatted_results))]
            
        elif name == "compare_cohorts":
            cohort1_id = arguments["cohort1_id"]
            cohort2_id = arguments["cohort2_id"]
            outcomes = arguments["outcomes"]
            adjustment = arguments.get("adjustment_method", "unadjusted")
            
            results = clif_server.outcomes_analyzer.compare_cohorts(
                cohort1_id, cohort2_id, outcomes, adjustment
            )
            
            formatted_results = clif_server.outcomes_analyzer.format_comparison(results)
            return [TextContent(type="text", text=clif_server.privacy_guard.sanitize_output(formatted_results))]
            
        elif name == "fit_regression":
            outcome = arguments["outcome"]
            predictors = arguments["predictors"]
            model_type = arguments["model_type"]
            interaction_terms = arguments.get("interaction_terms", [])
            random_effects = arguments.get("random_effects", [])
            
            # Use current cohort or load data
            if current_cohort is not None:
                df = current_cohort
            else:
                # Load and merge basic data
                patient_df = pd.read_csv(clif_server.data_path / "patient.csv")
                hosp_df = pd.read_csv(clif_server.data_path / "hospitalization.csv")
                df = patient_df.merge(hosp_df, on="patient_id")
            
            # Perform regression analysis
            from server.tools.regression_analyzer import RegressionAnalyzer
            regression = RegressionAnalyzer()
            
            result = regression.fit_model(
                df, outcome, predictors, model_type,
                interaction_terms, random_effects
            )
            
            # Store configuration for code generation
            last_regression_config = {
                'outcome': outcome,
                'predictors': predictors,
                'model_type': model_type,
                'interaction_terms': interaction_terms,
                'random_effects': random_effects
            }
            
            return [TextContent(type="text", text=clif_server.privacy_guard.sanitize_output(result))]
            
        elif name == "descriptive_stats":
            variables = arguments["variables"]
            stratify_by = arguments.get("stratify_by")
            
            # Use current cohort or load data
            if current_cohort is not None:
                df = current_cohort
            else:
                patient_df = pd.read_csv(clif_server.data_path / "patient.csv")
                hosp_df = pd.read_csv(clif_server.data_path / "hospitalization.csv")
                df = patient_df.merge(hosp_df, on="patient_id")
            
            # Generate descriptive statistics
            from server.tools.stats_calculator import StatsCalculator
            stats_calc = StatsCalculator()
            
            result = stats_calc.calculate_descriptive_stats(df, variables, stratify_by)
            
            return [TextContent(type="text", text=clif_server.privacy_guard.sanitize_output(result))]
            
        elif name == "build_ml_model":
            outcome = arguments["outcome"]
            features = arguments["features"]
            model_type = arguments.get("model_type", "auto")
            feature_selection = arguments.get("feature_selection")
            hyperparameter_tuning = arguments.get("hyperparameter_tuning", True)
            
            # Use current cohort or load data
            if current_cohort is not None:
                df = current_cohort
            else:
                patient_df = pd.read_csv(clif_server.data_path / "patient.csv")
                hosp_df = pd.read_csv(clif_server.data_path / "hospitalization.csv")
                df = patient_df.merge(hosp_df, on="patient_id")
                
                # Add derived outcomes if needed
                if outcome == "mortality" and "mortality" not in df.columns:
                    df["mortality"] = (df["discharge_disposition"] == "Expired").astype(int)
                if outcome == "los_days" and "los_days" not in df.columns:
                    df["admission_dttm"] = pd.to_datetime(df["admission_dttm"])
                    df["discharge_dttm"] = pd.to_datetime(df["discharge_dttm"])
                    df["los_days"] = (df["discharge_dttm"] - df["admission_dttm"]).dt.total_seconds() / 86400
            
            # Build model
            result = clif_server.ml_model_builder.build_model(
                df=df,
                outcome=outcome,
                features=features,
                model_type=model_type,
                feature_selection=feature_selection,
                hyperparameter_tuning=hyperparameter_tuning
            )
            
            # Generate report
            report = clif_server.ml_model_builder.generate_model_report(result['model_id'])
            
            # Store for code generation
            clif_server.code_generator.store_analysis({
                'type': 'ml_model',
                'model_info': result,
                'report': report
            })
            
            return [TextContent(type="text", text=clif_server.privacy_guard.sanitize_output(report))]
            
        elif name == "generate_code":
            language = arguments["language"]
            include_all = arguments.get("include_all", True)
            
            # Generate code based on stored analyses
            code = clif_server.code_generator.generate_code(
                "comprehensive_analysis", language, include_all, "script"
            )
            
            # Add regression if available
            if last_regression_config and include_all:
                regression_code = clif_server.code_generator.generate_regression_code(
                    last_regression_config, language
                )
                code += f"\n\n# Regression Analysis\n{regression_code}"
            
            # Save to file
            output_dir = Path.home() / "Desktop"
            output_dir.mkdir(exist_ok=True)
            
            filename = f"clif_analysis_{language}.{language[0]}"
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                f.write(code)
            
            result = f"Generated {language} code saved to: {filepath}\n\nPreview:\n{code[:500]}..."
            
            return [TextContent(type="text", text=result)]
            
        elif name == "create_derived_variable":
            variable_name = arguments["variable_name"]
            level = arguments["level"]
            variable_type = arguments.get("variable_type", "numeric")
            
            # Check if using predefined variable
            if "predefined" in arguments:
                # Get definition from predefined variables
                all_vars = clif_server.derived_variable_creator.get_common_derived_variables()
                
                # Find the predefined variable
                definition = {}
                for level_vars in all_vars.values():
                    if arguments["predefined"] in level_vars:
                        var_info = level_vars[arguments["predefined"]]
                        variable_type = var_info["type"]
                        level = var_info["level"]
                        break
                
                # Create the predefined variable
                result_df = clif_server.derived_variable_creator.create_derived_variable(
                    arguments["predefined"], variable_type, level, definition
                )
            else:
                # Use custom definition
                definition = arguments.get("definition", {})
                result_df = clif_server.derived_variable_creator.create_derived_variable(
                    variable_name, variable_type, level, definition
                )
            
            # Format result
            result = f"Created derived variable: {variable_name}\n"
            result += f"Level: {level}\n"
            result += f"Type: {variable_type}\n"
            result += f"Records created: {len(result_df)}\n\n"
            
            # Show sample
            if len(result_df) > 0:
                result += "Sample values:\n"
                result += result_df.head(10).to_string()
            
            # Update current cohort if it's a hospitalization-level variable
            if level == "hospitalization" and current_cohort is not None:
                current_cohort = current_cohort.merge(
                    result_df, on="hospitalization_id", how="left"
                )
                result += "\n\nVariable added to current cohort."
            
            return [TextContent(type="text", text=clif_server.privacy_guard.sanitize_output(result))]
            
        elif name == "list_derived_variables":
            # Get all predefined variables
            all_vars = clif_server.derived_variable_creator.get_common_derived_variables()
            
            result = "PREDEFINED DERIVED VARIABLES\n"
            result += "=" * 50 + "\n\n"
            
            for level_name, variables in all_vars.items():
                result += f"{level_name.upper()}:\n"
                result += "-" * 30 + "\n"
                
                for var_name, var_info in variables.items():
                    result += f"  • {var_name}: {var_info['description']}\n"
                    result += f"    Type: {var_info['type']}, Level: {var_info['level']}\n"
                
                result += "\n"
            
            result += "\nTo use a predefined variable:\n"
            result += '{"variable_name": "my_mortality", "level": "hospitalization", "predefined": "mortality"}\n\n'
            result += "To create a custom variable, provide a definition instead of predefined."
            
            return [TextContent(type="text", text=result)]
            
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except Exception as e:
        error_msg = f"Error executing {name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return [TextContent(type="text", text=error_msg)]


async def main():
    """Main entry point"""
    global clif_server
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="CLIF MCP Server")
    parser.add_argument("--data-path", default="./data/synthetic", help="Path to CLIF CSV files")
    args = parser.parse_args()
    
    # Initialize server
    print(f"Initializing CLIF MCP Server with data path: {args.data_path}", file=sys.stderr)
    
    try:
        clif_server = CLIFServer(args.data_path)
        print("CLIF Server initialized successfully", file=sys.stderr)
        
    except Exception as e:
        print(f"Error initializing server: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())