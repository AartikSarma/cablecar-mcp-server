# Subgroup Analysis

You are helping a clinical researcher perform subgroup analyses.

## Instructions
1. Review the subgroup plan: $ARGUMENTS
2. Use the MCP data server's execute_analysis tool with type "subgroup"
3. Key considerations:
   - Pre-specify subgroups (avoid data dredging)
   - Test for interaction (is the effect DIFFERENT across subgroups?)
   - Adjust for multiple comparisons
   - Report subgroup-specific estimates with CIs
4. Present as a forest plot layout:
   ```
   Subgroup          N     Effect (95% CI)     P-interaction
   -------------------------------------------------------
   Overall          892    1.45 (1.12-1.88)
   Age < 65         356    1.22 (0.81-1.83)    0.34
   Age >= 65        536    1.61 (1.15-2.25)
   Male             498    1.38 (0.98-1.94)    0.67
   Female           394    1.54 (1.05-2.26)
   ```

## Privacy Reminder
- Suppress subgroups with n < 10
