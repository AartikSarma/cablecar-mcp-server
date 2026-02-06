"""CableCar Minimal Data Server - MCP entry point.

This server provides exactly 4 tools for data access, all privacy-sanitized.
It is the ONLY interface between Claude and raw patient data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from cablecar.server.tools import DataServerTools

logger = logging.getLogger("cablecar")


def create_server(data_path: str | None = None) -> tuple[Server, DataServerTools]:
    """Create and configure the MCP server with 4 data tools."""

    server = Server("cablecar-data-server")
    tools = DataServerTools(data_path=data_path)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="get_schema",
                description="Get the data schema and data dictionary. Returns table definitions, column types, and relationships. No patient data is returned.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="load_data",
                description="Load and validate a clinical dataset. Returns a privacy-sanitized summary with table counts, validation status, and data quality metrics. Never returns raw patient data.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to directory containing data files (CSV or Parquet)",
                        },
                        "schema": {
                            "type": "string",
                            "description": "Schema to validate against (e.g., 'clif'). Optional.",
                        },
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="query_cohort",
                description="Define a study cohort with inclusion/exclusion criteria. Returns a privacy-sanitized CONSORT flow diagram and cohort summary. Never returns raw patient data.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name for this cohort",
                            "default": "main",
                        },
                        "description": {
                            "type": "string",
                            "description": "Human-readable description of the cohort",
                        },
                        "inclusion": {
                            "type": "array",
                            "description": "Inclusion criteria. Each: {column, op (==,!=,>,<,>=,<=,in,not_in,is_null,not_null), value}",
                            "items": {"type": "object"},
                        },
                        "exclusion": {
                            "type": "array",
                            "description": "Exclusion criteria (same format as inclusion)",
                            "items": {"type": "object"},
                        },
                        "index_table": {
                            "type": "string",
                            "description": "Base table for cohort (default: hospitalization)",
                            "default": "hospitalization",
                        },
                    },
                },
            ),
            Tool(
                name="execute_analysis",
                description="Execute a statistical analysis on a defined cohort. Returns privacy-sanitized results (coefficients, CIs, p-values, aggregated statistics). Never returns raw patient data. Supported types: summary_stats, descriptive, hypothesis, regression, subgroup.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "analysis_type": {
                            "type": "string",
                            "description": "Type: summary_stats, descriptive, hypothesis, regression, subgroup",
                            "enum": ["summary_stats", "descriptive", "hypothesis", "regression", "subgroup"],
                        },
                        "params": {
                            "type": "object",
                            "description": "Analysis-specific parameters",
                        },
                        "cohort_name": {
                            "type": "string",
                            "description": "Which cohort to analyze",
                            "default": "main",
                        },
                    },
                    "required": ["analysis_type"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Dispatch tool calls to DataServerTools."""
        try:
            if name == "get_schema":
                result = tools.get_schema()
            elif name == "load_data":
                result = tools.load_data(
                    path=arguments["path"],
                    schema=arguments.get("schema"),
                )
            elif name == "query_cohort":
                result = tools.query_cohort(
                    name=arguments.get("name", "main"),
                    description=arguments.get("description", ""),
                    inclusion=arguments.get("inclusion"),
                    exclusion=arguments.get("exclusion"),
                    index_table=arguments.get("index_table", "hospitalization"),
                )
            elif name == "execute_analysis":
                result = tools.execute_analysis(
                    analysis_type=arguments["analysis_type"],
                    params=arguments.get("params", {}),
                    cohort_name=arguments.get("cohort_name", "main"),
                )
            else:
                result = {"error": f"Unknown tool: {name}"}

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        except Exception as e:
            error_result = {"error": str(e), "sanitized": True}
            return [TextContent(type="text", text=json.dumps(error_result))]

    return server, tools


async def main():
    """Run the MCP server."""
    parser = argparse.ArgumentParser(description="CableCar Data Server")
    parser.add_argument("--data-path", type=str, help="Path to data directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger.info("Starting CableCar Data Server")

    server, tools = create_server(data_path=args.data_path)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
