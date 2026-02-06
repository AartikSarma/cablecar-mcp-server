# Load Clinical Dataset

You are helping a clinical researcher load and validate their dataset.

## Instructions

1. Ask the user for the data path if not provided: $ARGUMENTS
2. Use the MCP data server's `load_data` tool to load the dataset
3. The server will validate against the CLIF schema and return a sanitized summary
4. Review the validation results and report:
   - Number of tables loaded and their sizes
   - Schema validation status (which CLIF tables were found)
   - Data quality issues (missing values, unexpected formats)
   - Any warnings about the data
5. Suggest next steps (typically /define-cohort or /data-dictionary)

## Privacy Reminder
- NEVER attempt to read data files directly - always use the MCP data server
- Only discuss aggregate statistics, never individual patient records
- If the user asks to see raw data, explain the privacy boundary

## Example Flow
```
User: /load-data ./data/hospital_a/
Assistant: I'll load your dataset through the data server...
[Uses MCP load_data tool]
Found 8 CLIF-compliant tables:
- patient: 5,000 patients
- hospitalization: 12,340 encounters
- vitals: 1,234,567 measurements
...
Data quality: 2 warnings (see below)
Suggested next step: /define-cohort to select your study population
```
