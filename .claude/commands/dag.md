# Causal DAG Builder

You are helping a clinical researcher build a directed acyclic graph (DAG) for causal reasoning.

## Instructions

1. Review the study design: $ARGUMENTS
2. Help identify variables and their roles:
   - **Exposure** (E): The main variable of interest
   - **Outcome** (O): The result being measured
   - **Confounders** (C): Common causes of exposure AND outcome
   - **Mediators** (M): On the causal pathway from E to O
   - **Colliders** (X): Common effects - DO NOT adjust for these
3. Draw causal arrows between variables
4. Use the causal-advisor subagent for complex DAGs
5. Present the DAG in Mermaid format for visualization
6. Identify:
   - Minimal adjustment set (what to control for)
   - Collider bias warnings
   - Potential unmeasured confounders

## Common ICU Confounders
- Age -> most outcomes
- Severity of illness (SOFA/APACHE) -> treatment decisions AND outcomes
- Comorbidities -> treatment AND outcomes
- Admission type (medical/surgical) -> treatment patterns

## Collider Warning
NEVER adjust for variables that are EFFECTS of both exposure and outcome.
Example: If studying ventilation -> mortality, do NOT adjust for ICU LOS (it's caused by both).

## Output
Present the DAG as a Mermaid diagram and list the recommended adjustment set.
