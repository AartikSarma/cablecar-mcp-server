# Define Study Cohort

You are helping a clinical researcher define their study cohort with inclusion/exclusion criteria.

## Instructions

1. Review the user's request: $ARGUMENTS
2. If data hasn't been loaded yet, suggest using /load-data first
3. Help the researcher formulate criteria:
   - **Inclusion criteria**: Who should be IN the study?
   - **Exclusion criteria**: Who should be EXCLUDED?
4. Translate natural language criteria into structured format for the query_cohort tool:
   ```json
   {
     "inclusion": [
       {"column": "age_at_admission", "op": ">=", "value": 18},
       {"column": "location_category", "op": "==", "value": "ICU"}
     ],
     "exclusion": [
       {"column": "los_days", "op": "<", "value": 1}
     ]
   }
   ```
5. Use the MCP data server's `query_cohort` tool to apply criteria
6. Present the CONSORT-style flow diagram:
   ```
   Total hospitalizations: N
   → After inclusion criteria: N (excluded: N)
     → Criterion 1: N remaining
     → Criterion 2: N remaining
   → After exclusion criteria: N (excluded: N)
   → Final cohort: N
   ```
7. Ask if the researcher wants to adjust criteria

## Common Clinical Criteria
- Age: age_at_admission >= 18 (adults)
- ICU: location_category == "ICU" (from ADT table)
- Sepsis: Often requires lab + clinical criteria
- Mechanical ventilation: device_category == "vent"
- Minimum ICU stay: icu_los >= 24 hours

## Privacy Reminder
- Report only aggregate counts in the flow diagram
- All counts < 10 are suppressed by the privacy guard
