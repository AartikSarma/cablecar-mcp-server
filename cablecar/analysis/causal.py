"""Causal reasoning framework with DAG support."""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx


@dataclass
class Variable:
    """A variable in a causal DAG.

    Attributes
    ----------
    name:
        Variable identifier (should match a column name in the data).
    role:
        One of ``"exposure"``, ``"outcome"``, ``"confounder"``,
        ``"mediator"``, ``"collider"``, ``"instrument"``, ``"other"``.
    description:
        Optional human-readable description.
    """

    name: str
    role: str  # "exposure", "outcome", "confounder", "mediator", "collider", "instrument", "other"
    description: str = ""


class CausalDAG:
    """Directed acyclic graph for causal reasoning.

    This is a *framework tool* (not a :class:`BaseAnalysis` subclass).
    It helps researchers specify and validate their causal assumptions
    before running any statistical analysis.

    Features:
    - Fluent API: :meth:`add_variable` and :meth:`add_edge` return ``self``
      for chaining.
    - Cycle detection: adding an edge that would break the DAG property
      raises ``ValueError``.
    - Minimal adjustment set via the backdoor criterion.
    - Collider-bias warnings.
    - Mermaid diagram export for visualisation.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._graph = nx.DiGraph()
        self._variables: dict[str, Variable] = {}

    # ------------------------------------------------------------------
    # Building the DAG
    # ------------------------------------------------------------------

    def add_variable(
        self, name: str, role: str = "other", description: str = "",
    ) -> CausalDAG:
        """Add a variable to the DAG.  Returns *self* for chaining."""
        self._variables[name] = Variable(
            name=name, role=role, description=description,
        )
        self._graph.add_node(name, role=role)
        return self

    def add_edge(self, cause: str, effect: str) -> CausalDAG:
        """Add a causal edge.  Returns *self* for chaining.

        Raises
        ------
        ValueError
            If the edge would create a cycle.
        """
        if cause not in self._graph:
            self.add_variable(cause)
        if effect not in self._graph:
            self.add_variable(effect)
        self._graph.add_edge(cause, effect)
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self._graph):
            self._graph.remove_edge(cause, effect)
            raise ValueError(
                f"Adding edge {cause} -> {effect} would create a cycle"
            )
        return self

    # ------------------------------------------------------------------
    # Querying roles
    # ------------------------------------------------------------------

    def get_exposure(self) -> str | None:
        """Return the node marked as the exposure, or ``None``."""
        for name, var in self._variables.items():
            if var.role == "exposure":
                return name
        return None

    def get_outcome(self) -> str | None:
        """Return the node marked as the outcome, or ``None``."""
        for name, var in self._variables.items():
            if var.role == "outcome":
                return name
        return None

    def get_confounders(self) -> list[str]:
        """Return all nodes explicitly marked as confounders."""
        return [
            name for name, var in self._variables.items()
            if var.role == "confounder"
        ]

    # ------------------------------------------------------------------
    # Adjustment sets
    # ------------------------------------------------------------------

    def get_minimal_adjustment_set(self) -> set[str]:
        """Get the minimal set of variables to adjust for to identify the causal effect.

        Uses the backdoor criterion: block all backdoor paths from
        exposure to outcome.  Excludes colliders (conditioning on a
        collider opens a spurious path).

        This is a simplified implementation that:
        1. Includes all explicitly-tagged confounders.
        2. Includes any node that is an ancestor of the exposure **and**
           has a directed path to the outcome (or vice-versa).
        3. Removes any colliders.
        """
        exposure = self.get_exposure()
        outcome = self.get_outcome()
        if not exposure or not outcome:
            return set()

        adjustment: set[str] = set()

        for node in self._graph.nodes():
            if node in (exposure, outcome):
                continue

            var = self._variables.get(node)

            # Always include explicit confounders
            if var and var.role == "confounder":
                adjustment.add(node)
                continue

            # Include nodes that are ancestors of exposure AND have paths
            # to outcome (backdoor paths)
            is_ancestor_of_exposure = (
                nx.has_path(self._graph, node, exposure)
                or self._graph.has_edge(node, exposure)
            )
            has_path_to_outcome = (
                nx.has_path(self._graph, node, outcome)
                or self._graph.has_edge(node, outcome)
            )

            if is_ancestor_of_exposure and has_path_to_outcome:
                adjustment.add(node)

        # Remove colliders -- should NOT adjust for these
        colliders = {
            name
            for name, var in self._variables.items()
            if var.role == "collider"
        }
        adjustment -= colliders

        return adjustment

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def check_for_collider_bias(self) -> list[str]:
        """Warn if any colliders could induce bias when conditioned on.

        A collider is a node with two or more parents.  Conditioning on
        a collider (or its descendant) opens a non-causal path between
        the parents.
        """
        warnings: list[str] = []

        colliders = [
            name for name, var in self._variables.items()
            if var.role == "collider"
        ]
        for c in colliders:
            parents = list(self._graph.predecessors(c))
            if len(parents) >= 2:
                warnings.append(
                    f"WARNING: '{c}' is a collider (caused by {parents}). "
                    f"Conditioning on it may induce spurious associations."
                )

        return warnings

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram syntax for visualisation."""
        lines = ["graph LR"]

        for node in self._graph.nodes():
            var = self._variables.get(node)
            role = var.role if var else "other"
            style = {
                "exposure": f'{node}["{node} (E)"]:::exposure',
                "outcome": f'{node}["{node} (O)"]:::outcome',
                "confounder": f'{node}["{node} (C)"]:::confounder',
                "mediator": f'{node}["{node} (M)"]:::mediator',
                "collider": f'{node}["{node} (X)"]:::collider',
            }.get(role, f'{node}["{node}"]')
            lines.append(f"    {style}")

        for u, v in self._graph.edges():
            lines.append(f"    {u} --> {v}")

        lines.append("    classDef exposure fill:#e74c3c,color:#fff")
        lines.append("    classDef outcome fill:#2ecc71,color:#fff")
        lines.append("    classDef confounder fill:#f39c12,color:#fff")
        lines.append("    classDef mediator fill:#3498db,color:#fff")
        lines.append("    classDef collider fill:#9b59b6,color:#fff")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a summary of the DAG."""
        return {
            "name": self.name,
            "n_variables": len(self._variables),
            "n_edges": self._graph.number_of_edges(),
            "exposure": self.get_exposure(),
            "outcome": self.get_outcome(),
            "confounders": self.get_confounders(),
            "minimal_adjustment_set": list(self.get_minimal_adjustment_set()),
            "collider_warnings": self.check_for_collider_bias(),
            "is_dag": nx.is_directed_acyclic_graph(self._graph),
        }
