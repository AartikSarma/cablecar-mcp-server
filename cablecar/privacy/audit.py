from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class AuditEntry:
    """A single audit-log entry recording a tool invocation and privacy actions."""
    timestamp: str
    tool_name: str
    action: str
    privacy_actions: list[str] = field(default_factory=list)
    data_accessed: list[str] = field(default_factory=list)
    suppressed_count: int = 0


class AuditLog:
    """Append-only audit trail persisted as a JSONL file.

    Each line in the log file is a JSON-encoded :class:`AuditEntry`.
    """

    def __init__(self, log_path: str | Path | None = None) -> None:
        """Initialize the audit log.

        Parameters
        ----------
        log_path:
            Path to the JSONL log file.  Defaults to
            ``./audit/cablecar.audit.jsonl``.  The parent directory is
            created automatically if it does not exist.
        """
        if log_path is None:
            self._path = Path("./audit/cablecar.audit.jsonl")
        else:
            self._path = Path(log_path)

        # Ensure the directory exists.
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def log(self, entry: AuditEntry) -> None:
        """Append an :class:`AuditEntry` to the log file."""
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry), default=str) + "\n")

    def log_tool_call(
        self,
        tool_name: str,
        action: str,
        privacy_actions: list[str] | None = None,
        data_accessed: list[str] | None = None,
        suppressed_count: int = 0,
    ) -> None:
        """Convenience wrapper that creates and persists an :class:`AuditEntry`.

        Parameters
        ----------
        tool_name:
            Name of the MCP tool that was invoked.
        action:
            High-level description of the action performed.
        privacy_actions:
            List of privacy-related actions taken (e.g. cell suppression,
            PHI redaction).
        data_accessed:
            List of datasets / tables / columns that were accessed.
        suppressed_count:
            Number of values that were suppressed due to small cell sizes.
        """
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            tool_name=tool_name,
            action=action,
            privacy_actions=privacy_actions or [],
            data_accessed=data_accessed or [],
            suppressed_count=suppressed_count,
        )
        self.log(entry)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_entries(self, since: str | None = None) -> list[AuditEntry]:
        """Read entries from the log, optionally filtered by timestamp.

        Parameters
        ----------
        since:
            ISO-8601 timestamp string.  Only entries with a ``timestamp``
            greater than or equal to this value are returned.  When ``None``
            all entries are returned.
        """
        entries: list[AuditEntry] = []
        if not self._path.exists():
            return entries

        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entry = AuditEntry(**data)
                if since is not None and entry.timestamp < since:
                    continue
                entries.append(entry)
        return entries

    def summary(self) -> dict:
        """Return a summary of the audit log.

        Returns
        -------
        dict
            Keys: ``total_entries``, ``entries_by_tool``,
            ``total_suppressions``.
        """
        entries = self.get_entries()
        by_tool: dict[str, int] = {}
        total_suppressions = 0
        for entry in entries:
            by_tool[entry.tool_name] = by_tool.get(entry.tool_name, 0) + 1
            total_suppressions += entry.suppressed_count

        return {
            "total_entries": len(entries),
            "entries_by_tool": by_tool,
            "total_suppressions": total_suppressions,
        }
