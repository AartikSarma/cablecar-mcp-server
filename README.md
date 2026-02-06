# CableCar v2: AI-Powered Clinical Research Platform

**Connecting AI to Big Longitudinal EMRs for Clinical Analytics and Research**

A hybrid architecture combining a minimal MCP data server (privacy boundary) with Claude Code native tools (skills, subagents, hooks) for AI-assisted clinical research that is privacy-first, reproducible, and standards-compliant.

## Overview

CableCar enables clinical researchers to develop and study research questions with AI assistance while ensuring patient data never leaves the local environment. The platform supports the full research lifecycle: study design, cohort building, statistical analysis, causal reasoning, code generation, and publication reporting.

### Key Capabilities

- **Privacy-first architecture** -- raw patient data exists only in the server process; all outputs pass through PrivacyGuard before reaching the AI
- **Causal reasoning framework** -- DAG construction, backdoor criterion, collider bias detection, minimal adjustment sets
- **Full statistical suite** -- descriptive, hypothesis testing, logistic/linear/Cox regression, prediction models, survival analysis, subgroup analysis, sensitivity analysis
- **Reproducible code generation** -- Jinja2-templated Python and R (tidyverse) scripts that run at any CLIF site without modification
- **Standards-compliant reporting** -- STROBE (22 items) and TRIPOD+AI checklists with auto-population from analysis provenance
- **AI evaluation framework** -- benchmark scenarios with ground-truth effects for testing AI clinical reasoning capabilities

## Architecture

```
+--------------------------------------------------------------+
|  CLAUDE'S CONTEXT (Anthropic API)                            |
|                                                              |
|  Skills (/new-study, /dag, /table1, /regression, etc.)       |
|  Subagents (statistical-reviewer, causal-advisor, etc.)      |
|  ↕ Only sanitized aggregates, never raw data                 |
|  +========================================================+  |
|  ‖  Minimal Data Server (MCP, 4 tools)                    ‖  |
|  ‖  - get_schema()                                        ‖  |
|  ‖  - load_data(path, schema)                             ‖  |
|  ‖  - query_cohort(criteria)                              ‖  |
|  ‖  - execute_analysis(code)                              ‖  |
|  ‖  ALL outputs pass through PrivacyGuard                 ‖  |
|  +========================================================+  |
+---------------------------+----------------------------------+
                            | Only sanitized results cross
                            v
+--------------------------------------------------------------+
|  LOCAL MACHINE                                               |
|  Data Server Process (holds data in memory)                  |
|  ├── Raw data files (CSV/Parquet) loaded into memory         |
|  ├── PrivacyGuard filters ALL output before returning        |
|  ├── Analysis library executes locally                       |
|  └── Cohort/DataStore with caching                           |
|                                                              |
|  PreToolUse hooks block Read/Grep/Bash on data directory     |
+--------------------------------------------------------------+
```

| Layer | Implementation | Purpose |
|-------|---------------|---------|
| Privacy Boundary | Minimal Data Server (MCP, 4 tools) | Only interface to raw data; all output sanitized |
| User Interface | Skills (15 slash commands) | Research workflow steps; never touch raw data |
| Specialized AI | Subagents (5 agents) | Statistical review, causal reasoning, code generation |
| Safety Enforcement | Hooks (2 shell scripts) | Block direct data access, audit logging |

### Why This Architecture

The v1 prototype had 20+ MCP tools consuming significant context. v2 uses exactly 4 MCP tools with simple interfaces. All intelligence (study design, causal reasoning, statistical review, code generation) lives in skills and subagents, which consume zero context until invoked.

## Project Structure

```
cablecar/
├── server/                    # Minimal MCP Data Server (4 tools)
│   ├── main.py                # Server entry point
│   └── tools.py               # get_schema, load_data, query_cohort, execute_analysis
├── schema/                    # Flexible schema system
│   ├── base.py                # SchemaDefinition, TableSpec, ColumnSpec
│   ├── clif.py                # CLIF v2.1.0 complete schema
│   ├── registry.py            # Schema registry + auto-inference
│   └── validator.py           # Structural + type validation
├── data/                      # Data layer (runs inside server process)
│   ├── loader.py              # CSV/Parquet auto-detection
│   ├── store.py               # In-memory DataStore with caching
│   ├── cohort.py              # CohortBuilder + Cohort (architectural linchpin)
│   ├── transforms.py          # Derived variables (mortality, LOS, etc.)
│   └── temporal.py            # Time-series feature extraction
├── analysis/                  # Statistical analysis library
│   ├── base.py                # BaseAnalysis ABC + AnalysisResult
│   ├── descriptive.py         # Table 1, distributions, SMD
│   ├── hypothesis.py          # Group comparisons, multiple testing correction
│   ├── regression.py          # Linear, logistic, Cox PH
│   ├── prediction.py          # ML models, cross-validation
│   ├── survival.py            # Kaplan-Meier, competing risks
│   ├── causal.py              # DAG, backdoor criterion, adjustment sets
│   ├── subgroup.py            # Efficient subgroup engine + interaction tests
│   └── sensitivity.py         # Missing data, outlier, definition sensitivity
├── privacy/                   # Privacy and compliance
│   ├── guard.py               # PrivacyGuard (cell suppression, PHI redaction)
│   ├── phi_detector.py        # PHI detection (SSN, MRN, email, phone, DOB, IP)
│   ├── audit.py               # Persistent JSONL audit trail
│   └── policy.py              # Configurable PrivacyPolicy
├── codegen/                   # Reproducible code generation
│   ├── engine.py              # Jinja2-based CodeGenerator
│   ├── provenance.py          # AnalysisProvenance tracking
│   └── templates/             # Python (*.py.j2) and R (*.R.j2) templates
├── reporting/                 # Publication reporting
│   ├── strobe.py              # STROBE 22-item checklist
│   └── tripod.py              # TRIPOD+AI checklist
├── workflow/                  # Research workflow orchestration
│   ├── pipeline.py            # AnalysisPipeline (step chaining)
│   ├── state.py               # Immutable WorkflowState snapshots
│   ├── plan.py                # StudyPlan (PICO + causal framework)
│   └── cache.py               # ComputationCache
└── evaluation/                # AI capability evaluation
    ├── scenarios.py           # 6 ground-truth clinical scenarios
    ├── graders.py             # 4-dimension output scoring
    └── benchmarks.py          # Benchmark suite runner

.claude/
├── commands/                  # 15 skills (slash commands)
│   ├── new-study.md           # /new-study - PICO framework + study design
│   ├── load-data.md           # /load-data - Import + validate dataset
│   ├── define-cohort.md       # /define-cohort - Inclusion/exclusion criteria
│   ├── data-dictionary.md     # /data-dictionary - Explore schema
│   ├── dag.md                 # /dag - Build causal DAG
│   ├── table1.md              # /table1 - Baseline characteristics
│   ├── analyze.md             # /analyze - General analysis dispatch
│   ├── hypothesis.md          # /hypothesis - Statistical testing
│   ├── regression.md          # /regression - Regression models
│   ├── predict.md             # /predict - ML prediction models
│   ├── subgroup.md            # /subgroup - Subgroup analysis
│   ├── sensitivity.md         # /sensitivity - Sensitivity analyses
│   ├── export-code.md         # /export-code - Generate Python/R scripts
│   ├── report.md              # /report - STROBE/TRIPOD reports
│   └── privacy-check.md       # /privacy-check - Audit compliance
├── agents/                    # 5 subagents
│   ├── causal-advisor.md      # Guides DAG construction and causal reasoning
│   ├── statistical-reviewer.md # Validates analysis choices and assumptions
│   ├── code-generator.md      # Generates Python/R from templates
│   ├── clinical-evaluator.md  # Evaluates AI research capabilities
│   └── cablecar-clinical-evaluator.md
└── hooks/
    ├── block-data-access.sh   # PreToolUse: blocks Read/Grep/Bash on data/
    └── audit-logger.sh        # PostToolUse: logs all data operations

tests/
├── conftest.py                # Shared fixtures (mini schema, synthetic data, cohort)
├── unit/                      # 170 unit tests across 9 test files
│   ├── test_schema.py         # Schema types, validation, registry
│   ├── test_privacy.py        # PrivacyGuard, PHI detection, audit
│   ├── test_data.py           # Loader, store, cohort builder, subgroups
│   ├── test_analysis.py       # Descriptive, hypothesis, regression
│   ├── test_causal.py         # CausalDAG, adjustment sets, collider detection
│   ├── test_codegen.py        # Python/R code generation, provenance
│   ├── test_reporting.py      # STROBE and TRIPOD+AI reports
│   ├── test_workflow.py       # Pipeline, state, cache
│   └── test_evaluation.py     # Scenarios, graders, benchmarks
└── integration/               # 10 integration tests
    ├── test_pipeline.py       # End-to-end workflow
    └── test_privacy_boundary.py # PHI never leaks through any path
```

## Quick Start

### Prerequisites

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone https://github.com/AartikSarma/cablecar-mcp-server.git
cd cablecar-mcp-server

# Install dependencies with UV
uv sync --dev
```

### Synthetic Data

CableCar uses synthetic CLIF data (8,000 patients, 10,000 hospitalizations, 28 tables) from [synthetic_clif](https://github.com/AartikSarma/synthetic_clif):

```bash
# Clone and copy synthetic data
git clone https://github.com/AartikSarma/synthetic_clif.git /tmp/synthetic_clif
cp /tmp/synthetic_clif/synth_clif_10k/*.parquet data/synthetic/
```

### Run Tests

```bash
uv run pytest tests/ -v
```

190 tests, all passing (unit + integration).

### Claude Code Configuration

Add to `.claude/settings.local.json`:

```json
{
  "mcpServers": {
    "cablecar": {
      "command": "uv",
      "args": ["run", "python", "-m", "cablecar.server.main", "--data-path", "./data/synthetic"]
    }
  }
}
```

## Usage

### Research Workflow

```
/new-study   What factors are associated with mortality in ICU sepsis patients?
             -> PICO framework, study design guidance, causal-advisor builds DAG

/load-data   ./data/hospital_x/
             -> Validates against CLIF schema, privacy-filtered summary

/define-cohort  Adult ICU patients with sepsis
             -> CONSORT flow diagram: 10,000 -> 9,669 adults -> ...

/table1      --stratify-by mortality
             -> Baseline characteristics, all cells >= 10

/regression  logistic mortality ~ age + sofa + lactate + ventilation
             -> statistical-reviewer validates, reports ORs, CIs, assumptions

/subgroup    age_group
             -> Filters parent cohort (no reload), forest plot + interaction test

/export-code python
/export-code r
             -> Jinja2 templates produce complete, runnable scripts

/report      strobe
             -> All 22 STROBE items auto-populated from provenance chain
```

### Key API Examples

```python
from cablecar.data.store import DataStore
from cablecar.data.cohort import CohortBuilder, CohortDefinition
from cablecar.analysis.descriptive import DescriptiveAnalysis
from cablecar.analysis.regression import RegressionAnalysis
from cablecar.analysis.causal import CausalDAG
from cablecar.privacy.guard import PrivacyGuard

# Load data
store = DataStore()
store.load("./data/synthetic")

# Build cohort
cohort = CohortBuilder(store).build(CohortDefinition(
    name="adults",
    inclusion_criteria=[{"column": "age_at_admission", "op": ">=", "value": 18}],
))

# Descriptive analysis (Table 1)
table1 = DescriptiveAnalysis(cohort).run(
    variables=["age_at_admission", "hospital_mortality"],
    stratify_by="discharge_category",
)

# Causal DAG
dag = (CausalDAG("mortality_study")
    .add_variable("treatment", role="exposure")
    .add_variable("mortality", role="outcome")
    .add_variable("severity", role="confounder")
    .add_edge("severity", "treatment")
    .add_edge("severity", "mortality")
    .add_edge("treatment", "mortality"))
adjustment_set = dag.get_minimal_adjustment_set()  # {"severity"}

# Regression with confounders from DAG
result = RegressionAnalysis(cohort).run(
    outcome="hospital_mortality",
    predictors=["age_at_admission"],
    confounders=list(adjustment_set),
    model_type="logistic",
)

# Privacy: sanitize before any output reaches an LLM
safe = PrivacyGuard().sanitize_for_llm(result.to_dict())
# safe["sanitized"] == True, small cells suppressed, PHI redacted
```

## Privacy Guarantee

Raw patient data exists **only** in the data server's process memory. PHI cannot reach Anthropic's API through three layers of defense:

1. **PrivacyGuard** -- every server tool output passes through `sanitize_for_llm()` which applies cell suppression (counts < 10), PHI redaction (SSN, MRN, email, phone, DOB, IP), and audit logging
2. **PreToolUse hooks** -- `block-data-access.sh` blocks Claude from using Read/Grep/Bash to access the `data/` directory directly
3. **Architecture** -- analysis code runs inside the server process on raw data, but only returns sanitized coefficients, p-values, and aggregated statistics

## Causal Reasoning

The `CausalDAG` class (entirely new in v2) guides researchers through rigorous causal analysis:

- **Variable roles**: exposure, outcome, confounder, mediator, collider, instrument
- **Minimal adjustment set** via the backdoor criterion
- **Collider bias warnings** -- alerts when conditioning on a collider would induce spurious associations
- **Cycle detection** -- prevents invalid DAG structures
- **Mermaid diagram export** for visualization in markdown

The `/dag` skill and `causal-advisor` subagent walk researchers through DAG construction interactively.

## Code Generation

Generates complete, runnable analysis scripts using Jinja2 templates:

- **Python**: pandas, scipy, statsmodels, lifelines, scikit-learn
- **R**: tidyverse (dplyr, ggplot2, broom, tidyr), survival, survminer

Generated code works at **any site** with CLIF-formatted data without modification, enabling federated multi-site research.

## Evaluation Framework

Six benchmark scenarios with known ground-truth effects for testing AI clinical reasoning:

| Scenario | Difficulty | Domain | Ground Truth |
|----------|-----------|--------|-------------|
| Age-mortality association | Easy | General | Linear increase embedded in data |
| Severity-vasopressor relationship | Easy | General | 90% vs 30% probability |
| Ventilation-LOS (confounded) | Medium | Causal | Severity confounds both |
| Lactate mortality prediction | Medium | Prediction | AUROC 0.55-0.75 expected |
| Confounding by indication | Hard | Causal | Unadjusted association is misleading |
| Age-severity interaction | Hard | General | Both independently affect mortality |

Grading across 4 dimensions: method appropriateness, statistical correctness, causal reasoning, and completeness.

## Data Format

CableCar supports the [CLIF v2.1.0](https://clif-consortium.github.io/website/) schema with 28 tables:

**Core tables**: patient, hospitalization, ADT, vitals, labs, respiratory_support, medication_admin_continuous, patient_assessments

**Additional tables**: procedures, dialysis, intake_output, position, microbiology, sensitivity, ecg, imaging, provider, diagnosis, billing, surgery, transfusion, therapy, nutrition, line_list, code_status, and more.

CSV and Parquet formats are auto-detected. Custom schemas can be registered via `SchemaRegistry`.

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Run full test suite
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=cablecar --cov-report=term-missing
```

### Dependencies

Managed via UV (`pyproject.toml`):

- pandas, numpy, scipy, scikit-learn, statsmodels, lifelines
- networkx (causal DAGs), pydantic (schema types), jinja2 (code generation)
- pyarrow (Parquet support), mcp (Model Context Protocol)

## License

MIT License

---

*CableCar v2 -- 45 Python modules, 15 skills, 5 subagents, 2 hooks, 190 tests, 0 PHI leaks.*
