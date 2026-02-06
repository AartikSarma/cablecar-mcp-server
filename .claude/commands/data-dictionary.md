# Data Dictionary Explorer

You are helping a clinical researcher understand their dataset's structure and contents.

## Instructions

1. Use the MCP data server's `get_schema` tool to retrieve the data dictionary
2. Present the schema information clearly:
   - List all available tables with descriptions
   - For each table: columns, data types, descriptions
   - Highlight key relationships (foreign keys)
   - Note which variables are available for analysis
3. If the user asks about a specific variable: $ARGUMENTS
   - Show the variable's table, type, and description
   - Show summary statistics (from the sanitized summary)
   - Suggest related variables
4. Help the researcher understand what analyses are possible with available data

## Output Format
Present as organized tables:

### Table: hospitalization
| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| hospitalization_id | string | Unique encounter ID | Primary key |
| patient_id | string | Patient identifier | Links to patient table |
| admission_dttm | datetime | Admission timestamp | |
| discharge_dttm | datetime | Discharge timestamp | |
| age_at_admission | float | Age in years | |

## Privacy Reminder
- Show schema structure freely (no PHI in schema)
- For value distributions, only show aggregated statistics
- Never show example values that could be PHI
