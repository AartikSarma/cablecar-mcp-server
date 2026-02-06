from dataclasses import dataclass, field


@dataclass
class PrivacyPolicy:
    """Configurable privacy policy for data output sanitization."""
    min_cell_size: int = 10  # Suppress cells with count < this
    k_anonymity: int = 5     # Minimum group size for k-anonymity
    suppress_marker: str = "<suppressed>"  # What to show for suppressed values
    redact_phi: bool = True  # Auto-detect and redact PHI
    max_unique_categories: int = 50  # Max categories before treating as high-cardinality
    round_percentages: bool = True  # Round percentages to 1 decimal
    round_means: bool = True  # Round means to 2 decimals
    suppress_extreme_percentiles: bool = True  # Don't show min/max for small groups
    allowed_stats: list[str] = field(default_factory=lambda: [
        "count", "mean", "std", "median", "q1", "q3", "min", "max",
        "missing_count", "missing_pct", "n_unique"
    ])
