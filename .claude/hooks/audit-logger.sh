#!/bin/bash
# PostToolUse hook: Log all MCP data server tool calls for audit trail

INPUT=$(cat)

TOOL=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool',''))" 2>/dev/null)

# Only log MCP tool calls (data server tools)
if [[ "$TOOL" == "mcp__cablecar__"* ]]; then
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    TOOL_NAME="$TOOL"

    mkdir -p audit
    echo "{\"timestamp\": \"$TIMESTAMP\", \"tool\": \"$TOOL_NAME\", \"event\": \"post_tool_use\"}" >> audit/cablecar.audit.jsonl
fi
