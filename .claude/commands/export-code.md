# Export Reproducible Analysis Code

You are generating a portable, reproducible analysis script that can run at any CLIF-compliant site.

## Arguments
$ARGUMENTS should specify: `<language>` (python or r)

## Process

### Step 1: Gather Provenance
Collect the complete analysis context from this session:
- Study name and description
- Data source path and format (CSV or Parquet)
- Schema name (e.g., "clif")
- Which CLIF tables were actually used
- Cohort definition (all inclusion/exclusion criteria with column, op, value)
- Every analysis step performed, including:
  - Step name and description
  - Analysis type (descriptive, regression, survival, prediction, hypothesis, subgroup)
  - Parameters (outcome, predictors, confounders, model_type, etc.)
  - Key results (sanitized aggregates only, never raw data)

### Step 2: Generate Scaffold
Use the CableCar scaffold system to generate the deterministic boilerplate. Run this via Bash:

```
uv run python -c "
from cablecar.codegen.engine import CodeGenerator
from cablecar.codegen.provenance import AnalysisProvenance

prov = AnalysisProvenance(
    study_name='<study name>',
    data_source='./data',
    schema_name='clif',
    data_format='<csv or parquet>',
    tables_used=[<list of tables>],
    cohort_definition={
        'inclusion': [<criteria>],
        'exclusion': [<criteria>],
    },
)
# Add each analysis step from the session
# prov.add_step(name, description, parameters={...}, analysis_type='...', result_summary={...})

gen = CodeGenerator()
print(gen.generate_scaffold('<language>', prov))
"
```

### Step 3: Fill in Analysis Code
Take the scaffold and replace every `raise NotImplementedError` (Python) or `stop("Fill in analysis code")` (R) with complete, working analysis code that reproduces the exact analyses from this session.

Use the code-generator agent for the analysis implementations, providing:
- The scaffold with its stub functions
- The study provenance context (step parameters and results)
- The target language

### Step 4: Validate
Before presenting the final script, verify:
- [ ] All imports are at the top (no mid-script imports)
- [ ] Data directory is configurable (argparse for Python, commandArgs for R)
- [ ] Every table load has an existence check
- [ ] Cohort criteria match the session exactly
- [ ] Every analysis from the session is implemented (no stubs remain)
- [ ] No hardcoded file paths (only relative to data_dir)
- [ ] No raw patient data appears anywhere in the script
- [ ] R code uses tidyverse style throughout (pipes, dplyr, ggplot2, broom)
- [ ] Python code uses pathlib, f-strings, type hints

### Step 5: Present
Show the complete script in a code block. Then provide:
1. How to run it: `python script.py --data-dir /path/to/clif/data` or `Rscript script.R /path/to/clif/data`
2. Required dependencies (pip packages or R packages)
3. Expected output description

## Key Constraint
The generated code must be runnable WITHOUT modification at a different hospital site that has CLIF-formatted data. The only thing a new site changes is the data directory argument.
