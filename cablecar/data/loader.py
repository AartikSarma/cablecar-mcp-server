"""Data loading utilities for CSV and Parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataLoader:
    """Load clinical data files and directories into DataFrames."""

    _FORMAT_MAP = {
        ".csv": "csv",
        ".parquet": "parquet",
        ".pq": "parquet",
    }

    def detect_format(self, filename: str) -> str:
        """Detect file format from extension.

        Returns ``"csv"`` or ``"parquet"``.

        Raises:
            ValueError: If the extension is not recognized.
        """
        suffix = Path(filename).suffix.lower()
        fmt = self._FORMAT_MAP.get(suffix)
        if fmt is None:
            raise ValueError(
                f"Unknown file format '{suffix}'. "
                f"Supported: {sorted(self._FORMAT_MAP.keys())}"
            )
        return fmt

    def load_file(self, path: Path | str) -> pd.DataFrame:
        """Load a single CSV or Parquet file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        fmt = self.detect_format(path.name)
        if fmt == "csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)

    def load_directory(self, path: Path | str) -> dict[str, pd.DataFrame]:
        """Load all CSV/Parquet files in a directory.

        Returns a dict mapping file stems to DataFrames.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        tables: dict[str, pd.DataFrame] = {}
        for ext in self._FORMAT_MAP:
            for file_path in sorted(path.glob(f"*{ext}")):
                stem = file_path.stem
                if stem not in tables:
                    tables[stem] = self.load_file(file_path)
        return tables

    def get_table_summary(self, df: pd.DataFrame, name: str) -> dict:
        """Return a summary dict for a DataFrame."""
        return {
            "name": name,
            "rows": len(df),
            "columns": list(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 3),
        }
