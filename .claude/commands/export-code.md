# Export Reproducible Analysis Code

You are generating portable, reproducible analysis scripts.

## Instructions
1. Determine the target language: $ARGUMENTS (python or r)
2. Gather the complete analysis provenance from this session:
   - Data source and schema
   - Cohort definition (inclusion/exclusion criteria)
   - All analyses performed
3. Generate a complete, self-contained script that:
   - Loads data from a configurable directory
   - Applies the exact cohort definition
   - Runs all analyses in order
   - Produces publication-ready output
4. The script must work at ANY site with CLIF-formatted data

## Language-Specific Notes
### Python
- Use pandas, scipy, statsmodels, lifelines, scikit-learn
- Include all imports at the top
- Use argparse for the data directory path

### R
- Use tidyverse (dplyr, ggplot2, broom, tidyr)
- Use survival, survminer for time-to-event
- Use glm() for logistic regression with broom::tidy()

## Key Requirement
The generated code MUST be runnable without modification at a different hospital site that has CLIF-formatted data.
