"""DGP-driven synthetic data generator.

Generates synthetic clinical data from a :class:`DGPSpec`, producing concrete
data tables (e.g. CLIF CSVs) with known ground truth.  The generation follows
the causal structure of the DAG via topological sort, ensuring that parent
variables are generated before their children.

This module is intentionally separate from :mod:`synthetic.generator`
(``CLIFSyntheticGenerator``) for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from cablecar.evaluation.dgp import (
    CausalEdge,
    CLIFMapping,
    DGPSpec,
    DGPVariable,
    ErrorType,
    FunctionalForm,
    GroundTruth,
    MissingnessMechanism,
    VariableType,
)


class DGPSyntheticGenerator:
    """Generate synthetic data from a :class:`DGPSpec`.

    The algorithm:
    1. Topological sort of the DAG to determine generation order.
    2. Root nodes are sampled from their marginal distributions.
    3. Child nodes are generated conditionally on their parents using the
       edge's functional form and effect size.
    4. Measurement noise is applied per ``NoiseSpec``.
    5. Missingness is applied per ``MissingnessSpec``.
    6. Schema mappings place values into concrete data tables.
    """

    def __init__(self, spec: DGPSpec) -> None:
        self.spec = spec
        self.rng = np.random.default_rng(spec.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> tuple[dict[str, pd.DataFrame], GroundTruth]:
        """Generate data tables and return ``(tables, ground_truth)``."""
        patient_data = self._generate_patient_data()
        patient_data = self._apply_noise(patient_data)
        patient_data = self._apply_missingness(patient_data)
        tables = self._map_to_tables(patient_data)
        return tables, self.spec.ground_truth

    def generate_and_save(
        self, output_dir: str | Path,
    ) -> tuple[dict[str, pd.DataFrame], GroundTruth]:
        """Generate, save to disk, and return ``(tables, ground_truth)``.

        Directory layout::

            output_dir/
              tables/
                patient.csv
                hospitalization.csv
                ...
              ground_truth/
                dgp_spec.json
        """
        output_dir = Path(output_dir)
        tables_dir = output_dir / "tables"
        gt_dir = output_dir / "ground_truth"
        tables_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        tables, ground_truth = self.generate()

        for table_name, df in tables.items():
            df.to_csv(tables_dir / f"{table_name}.csv", index=False)

        (gt_dir / "dgp_spec.json").write_text(
            self.spec.model_dump_json(indent=2)
        )

        return tables, ground_truth

    # ------------------------------------------------------------------
    # Step 1: Build the DAG and get topological order
    # ------------------------------------------------------------------

    def _build_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for var in self.spec.variables:
            graph.add_node(var.name)
        for edge in self.spec.edges:
            graph.add_edge(edge.cause, edge.effect)
        return graph

    def _get_generation_order(self) -> list[str]:
        graph = self._build_graph()
        return list(nx.topological_sort(graph))

    def _get_parent_edges(self, child_name: str) -> list[CausalEdge]:
        """Return all edges pointing *into* ``child_name``."""
        return [e for e in self.spec.edges if e.effect == child_name]

    # ------------------------------------------------------------------
    # Step 2 & 3: Generate patient-level data
    # ------------------------------------------------------------------

    def _generate_patient_data(self) -> pd.DataFrame:
        """Generate a DataFrame with one row per patient, one column per DGP variable."""
        n = self.spec.n_patients
        order = self._get_generation_order()
        data: dict[str, np.ndarray] = {}

        for var_name in order:
            var = self.spec.get_variable(var_name)
            parent_edges = self._get_parent_edges(var_name)

            if not parent_edges:
                # Root node -- sample from marginal distribution
                data[var_name] = self._sample_marginal(var, n)
            else:
                # Child node -- conditional generation
                data[var_name] = self._sample_conditional(var, parent_edges, data, n)

        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Marginal sampling
    # ------------------------------------------------------------------

    def _sample_marginal(self, var: DGPVariable, n: int) -> np.ndarray:
        dist = var.distribution
        family = dist.family.lower()
        params = dist.params

        if family == "normal":
            return self.rng.normal(
                loc=params.get("mean", 0),
                scale=params.get("std", 1),
                size=n,
            )
        elif family == "bernoulli":
            return self.rng.binomial(1, params.get("p", 0.5), size=n).astype(float)
        elif family == "uniform":
            return self.rng.uniform(
                low=params.get("low", 0),
                high=params.get("high", 1),
                size=n,
            )
        elif family == "poisson":
            return self.rng.poisson(lam=params.get("lam", 1), size=n).astype(float)
        elif family == "exponential":
            return self.rng.exponential(
                scale=params.get("scale", 1), size=n
            )
        elif family == "categorical":
            categories = var.categories or params.get("categories", ["A", "B"])
            probs = params.get("probabilities", None)
            indices = self.rng.choice(len(categories), size=n, p=probs)
            return indices.astype(float)
        else:
            return self.rng.normal(size=n)

    # ------------------------------------------------------------------
    # Conditional sampling
    # ------------------------------------------------------------------

    def _sample_conditional(
        self,
        var: DGPVariable,
        parent_edges: list[CausalEdge],
        data: dict[str, np.ndarray],
        n: int,
    ) -> np.ndarray:
        """Generate *var* conditional on its parents via functional forms."""
        # Start with the marginal baseline
        dist = var.distribution
        family = dist.family.lower()
        params = dist.params

        if var.variable_type == VariableType.BINARY:
            base_logit = np.full(n, _logit(params.get("p", 0.5)))
            for edge in parent_edges:
                parent_vals = data[edge.cause]
                base_logit += self._apply_functional_form(edge, parent_vals, data)
            prob = _sigmoid(base_logit)
            return self.rng.binomial(1, prob, size=n).astype(float)

        elif var.variable_type == VariableType.COUNT:
            base_log_rate = np.full(n, np.log(max(params.get("lam", 1), 0.01)))
            for edge in parent_edges:
                parent_vals = data[edge.cause]
                base_log_rate += self._apply_functional_form(edge, parent_vals, data)
            rate = np.exp(base_log_rate)
            rate = np.clip(rate, 0.01, 1000)
            return self.rng.poisson(lam=rate).astype(float)

        elif var.variable_type == VariableType.TIME_TO_EVENT:
            base_log_rate = np.full(
                n, np.log(max(params.get("scale", 1), 0.01))
            )
            for edge in parent_edges:
                parent_vals = data[edge.cause]
                base_log_rate += self._apply_functional_form(edge, parent_vals, data)
            scale = np.exp(base_log_rate)
            scale = np.clip(scale, 0.01, 1000)
            return self.rng.exponential(scale=scale)

        else:
            # Continuous: additive model
            base_mean = np.full(n, params.get("mean", 0.0))
            base_std = params.get("std", 1.0)
            for edge in parent_edges:
                parent_vals = data[edge.cause]
                base_mean += self._apply_functional_form(edge, parent_vals, data)
            noise = self.rng.normal(0, base_std, size=n)
            return base_mean + noise

    # ------------------------------------------------------------------
    # Functional form implementations
    # ------------------------------------------------------------------

    def _apply_functional_form(
        self,
        edge: CausalEdge,
        parent_vals: np.ndarray,
        data: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Return the contribution of *parent_vals* via *edge*'s functional form."""
        form = edge.functional_form

        if form == FunctionalForm.LINEAR:
            return edge.effect_size * parent_vals

        elif form == FunctionalForm.LOGISTIC:
            return edge.effect_size * parent_vals

        elif form == FunctionalForm.THRESHOLD:
            threshold = edge.parameters.get("threshold", 0.0)
            effect_above = edge.parameters.get("effect_above", edge.effect_size)
            effect_below = edge.parameters.get("effect_below", 0.0)
            return np.where(
                parent_vals > threshold, effect_above, effect_below
            )

        elif form == FunctionalForm.QUADRATIC:
            quad_coeff = edge.parameters.get("quad_coeff", 0.0)
            return edge.effect_size * parent_vals + quad_coeff * parent_vals**2

        elif form == FunctionalForm.INTERACTION:
            other_var = edge.parameters.get("interaction_variable", "")
            if other_var and other_var in data:
                other_vals = data[other_var]
                return edge.effect_size * parent_vals * other_vals
            return edge.effect_size * parent_vals

        return edge.effect_size * parent_vals

    # ------------------------------------------------------------------
    # Step 4: Noise application
    # ------------------------------------------------------------------

    def _apply_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for var in self.spec.variables:
            if var.noise.error_type == ErrorType.NONE:
                continue
            if var.name not in df.columns:
                continue

            values = df[var.name].values.copy()

            if var.noise.error_type == ErrorType.ADDITIVE_GAUSSIAN:
                noise = self.rng.normal(0, var.noise.magnitude, size=len(values))
                values = values + noise

            elif var.noise.error_type == ErrorType.MULTIPLICATIVE:
                noise = self.rng.normal(1, var.noise.magnitude, size=len(values))
                values = values * noise

            elif var.noise.error_type == ErrorType.MISCLASSIFICATION:
                flip_prob = var.noise.parameters.get(
                    "flip_probability", var.noise.magnitude
                )
                flip_mask = self.rng.random(len(values)) < flip_prob
                if var.variable_type == VariableType.BINARY:
                    values[flip_mask] = 1.0 - values[flip_mask]
                else:
                    values[flip_mask] = self.rng.permutation(values[flip_mask])

            elif var.noise.error_type == ErrorType.ROUNDING:
                digits = var.noise.parameters.get("rounding_digits", 0)
                values = np.round(values, decimals=int(digits))

            df[var.name] = values
        return df

    # ------------------------------------------------------------------
    # Step 5: Missingness application
    # ------------------------------------------------------------------

    def _apply_missingness(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for var in self.spec.variables:
            if var.missingness.mechanism == MissingnessMechanism.NONE:
                continue
            if var.name not in df.columns:
                continue

            prop = var.missingness.proportion
            n = len(df)

            if var.missingness.mechanism == MissingnessMechanism.MCAR:
                mask = self.rng.random(n) < prop
                df.loc[mask, var.name] = np.nan

            elif var.missingness.mechanism == MissingnessMechanism.MAR:
                cond_vars = var.missingness.conditioning_variables
                if cond_vars:
                    # Higher values of conditioning variable -> higher missingness
                    cond_col = cond_vars[0]
                    if cond_col in df.columns:
                        cond_vals = df[cond_col].values
                        # Scale to [0, 1] based on rank
                        ranks = pd.Series(cond_vals).rank(pct=True).values
                        miss_prob = prop * 2 * ranks
                        miss_prob = np.clip(miss_prob, 0, 1)
                        mask = self.rng.random(n) < miss_prob
                        df.loc[mask, var.name] = np.nan
                    else:
                        mask = self.rng.random(n) < prop
                        df.loc[mask, var.name] = np.nan
                else:
                    mask = self.rng.random(n) < prop
                    df.loc[mask, var.name] = np.nan

            elif var.missingness.mechanism == MissingnessMechanism.MNAR:
                # Missingness depends on the variable's own value
                vals = df[var.name].values
                ranks = pd.Series(vals).rank(pct=True).values
                miss_prob = prop * 2 * ranks
                miss_prob = np.clip(miss_prob, 0, 1)
                mask = self.rng.random(n) < miss_prob
                df.loc[mask, var.name] = np.nan

        return df

    # ------------------------------------------------------------------
    # Step 6: Schema mapping to concrete tables
    # ------------------------------------------------------------------

    def _map_to_tables(self, patient_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Map patient-level DGP data to concrete schema tables."""
        n = self.spec.n_patients

        # Always produce patient and hospitalization tables
        patient_ids = [f"P{i:06d}" for i in range(1, n + 1)]
        hosp_ids = [f"H{i:06d}" for i in range(1, n + 1)]

        patient_table = pd.DataFrame({"patient_id": patient_ids})
        hosp_table = pd.DataFrame({
            "hospitalization_id": hosp_ids,
            "patient_id": patient_ids,
        })

        # Collect per-table rows
        table_data: dict[str, list[dict[str, Any]]] = {}

        for var_name, mapping in self.spec.schema_mappings.items():
            if not isinstance(mapping, CLIFMapping):
                continue
            if var_name not in patient_data.columns:
                continue

            values = patient_data[var_name].values

            if mapping.table == "patient":
                patient_table[mapping.column] = values
            elif mapping.table == "hospitalization":
                hosp_table[mapping.column] = values
            elif mapping.category_column and mapping.category_value:
                # Category-based table (vitals, labs, assessments, etc.)
                table_name = mapping.table
                if table_name not in table_data:
                    table_data[table_name] = []
                for i in range(n):
                    val = values[i]
                    if pd.isna(val):
                        continue
                    row: dict[str, Any] = {
                        "hospitalization_id": hosp_ids[i],
                        mapping.category_column: mapping.category_value,
                        mapping.column: val,
                    }
                    # Add datetime placeholder for time-varying tables
                    if mapping.table in (
                        "vitals",
                        "labs",
                        "respiratory_support",
                        "medication_admin_continuous",
                        "patient_assessments",
                    ):
                        dttm_col = _get_datetime_column(mapping.table)
                        if dttm_col:
                            row[dttm_col] = f"2024-01-01T{8 + i % 12:02d}:00:00"
                    table_data[table_name].append(row)
            else:
                # Direct column mapping (non-category)
                if mapping.table == "patient":
                    patient_table[mapping.column] = values
                elif mapping.table == "hospitalization":
                    hosp_table[mapping.column] = values

        tables: dict[str, pd.DataFrame] = {
            "patient": patient_table,
            "hospitalization": hosp_table,
        }

        for table_name, rows in table_data.items():
            if rows:
                tables[table_name] = pd.DataFrame(rows)

        return tables


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _logit(p: float) -> float:
    p = max(min(p, 0.999), 0.001)
    return float(np.log(p / (1 - p)))


def _get_datetime_column(table_name: str) -> str | None:
    """Return the datetime column name for a CLIF time-varying table."""
    mapping = {
        "vitals": "recorded_dttm",
        "labs": "lab_result_dttm",
        "respiratory_support": "recorded_dttm",
        "medication_admin_continuous": "admin_dttm",
        "patient_assessments": "recorded_dttm",
    }
    return mapping.get(table_name)
