#!/bin/bash
# PreToolUse hook: Block direct access to data/ directory
# This prevents Claude from using Read/Grep/Bash to access raw patient data.
# All data access MUST go through the MCP data server's privacy-sanitized tools.

# Read the tool use JSON from stdin
INPUT=$(cat)

TOOL=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool',''))" 2>/dev/null)
INPUT_TEXT=$(echo "$INPUT" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin).get('input',{})))" 2>/dev/null)

# Check if the tool is one that could access files
if [[ "$TOOL" == "Read" || "$TOOL" == "Grep" || "$TOOL" == "Glob" || "$TOOL" == "Bash" || "$TOOL" == "Edit" || "$TOOL" == "Write" ]]; then
    # Check if the input references the data directory
    if echo "$INPUT_TEXT" | python3 -c "
import sys
text = sys.stdin.read()
data_patterns = ['/data/', '/data\"', 'data/synthetic', 'data/hospital']
for pattern in data_patterns:
    if pattern in text:
        # Allow access to cablecar/data/ (source code) but block data/ (patient data)
        if 'cablecar/data/' not in text and 'tests/' not in text:
            sys.exit(1)
sys.exit(0)
" 2>/dev/null; then
        : # Access allowed
    else
        echo '{"error": "BLOCKED: Direct access to data/ directory is not allowed. All data access must go through the MCP data server tools (get_schema, load_data, query_cohort, execute_analysis) which enforce privacy sanitization."}' >&2
        exit 2
    fi
fi
