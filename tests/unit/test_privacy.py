"""Tests for cablecar.privacy: guard, PHI detector, policy, and audit."""
import json
import pandas as pd
import pytest

from cablecar.privacy.guard import PrivacyGuard
from cablecar.privacy.policy import PrivacyPolicy
from cablecar.privacy.phi_detector import PHIDetector, PHIMatch
from cablecar.privacy.audit import AuditLog, AuditEntry


class TestPrivacyPolicy:
    def test_defaults(self):
        policy = PrivacyPolicy()
        assert policy.min_cell_size == 10
        assert policy.redact_phi is True
        assert policy.suppress_marker == "<suppressed>"

    def test_custom_policy(self):
        policy = PrivacyPolicy(min_cell_size=20, k_anonymity=10)
        assert policy.min_cell_size == 20
        assert policy.k_anonymity == 10


class TestPHIDetector:
    def test_detect_ssn(self):
        detector = PHIDetector()
        matches = detector.scan_text("SSN: 123-45-6789")
        assert any(m.phi_type == "SSN" for m in matches)

    def test_detect_email(self):
        detector = PHIDetector()
        assert detector.contains_phi("john.doe@hospital.org")

    def test_detect_phone(self):
        detector = PHIDetector()
        assert detector.contains_phi("Call 555-123-4567")

    def test_detect_mrn_labeled(self):
        detector = PHIDetector()
        assert detector.contains_phi("MRN: 12345678")

    def test_no_phi_in_clean_text(self):
        detector = PHIDetector()
        assert not detector.contains_phi("Patient age 65, male")

    def test_redact_text(self):
        detector = PHIDetector()
        redacted = detector.redact_text("SSN: 123-45-6789")
        assert "123-45-6789" not in redacted
        assert "REDACTED" in redacted

    def test_scan_dataframe(self):
        detector = PHIDetector()
        df = pd.DataFrame({
            "name": ["John Doe"],
            "email": ["john@example.com"],
            "age": [65],
        })
        matches = detector.scan_dataframe(df, phi_columns=["email"])
        assert any(m.phi_type == "EMAIL" for m in matches)

    def test_extra_patterns(self):
        detector = PHIDetector(extra_patterns={"CUSTOM": r"\bXYZ\d{4}\b"})
        assert detector.contains_phi("ID: XYZ1234")


class TestPrivacyGuard:
    def test_sanitize_dict(self, privacy_guard):
        data = {"count": 5, "label": "test", "total": 100}
        result = privacy_guard.sanitize_for_llm(data, context="test")
        assert result["sanitized"] is True
        # Count of 5 should be suppressed (below min_cell_size=10)
        assert result["data"]["count"] == "<suppressed>"
        assert result["data"]["total"] == 100

    def test_sanitize_string_with_phi(self, privacy_guard):
        text = "Patient SSN: 123-45-6789"
        result = privacy_guard.sanitize_for_llm(text, context="note")
        assert result["sanitized"] is True
        assert "123-45-6789" not in str(result["data"])

    def test_sanitize_string_without_phi(self, privacy_guard):
        text = "Mean age was 65.2 years"
        result = privacy_guard.sanitize_for_llm(text, context="stat")
        assert result["data"] == text

    def test_sanitize_dataframe(self, privacy_guard, mini_dataframes):
        result = privacy_guard.sanitize_for_llm(
            mini_dataframes["patient"], context="patient_table"
        )
        assert result["sanitized"] is True
        # Should have shape info, no raw rows
        assert "shape" in result["data"]
        assert result["data"]["shape"]["rows"] == 50

    def test_suppress_small_cells(self, privacy_guard):
        counts = {"GroupA": 100, "GroupB": 5, "GroupC": 15}
        suppressed = privacy_guard.suppress_small_cells(counts)
        assert suppressed["GroupA"] == 100
        assert suppressed["GroupB"] == "<suppressed>"
        assert suppressed["GroupC"] == 15

    def test_suppress_series(self, privacy_guard):
        series = pd.Series({"A": 100, "B": 3})
        result = privacy_guard.suppress_small_cells(series)
        assert result["B"] == "<suppressed>"

    def test_nested_dict_sanitization(self, privacy_guard):
        data = {
            "outer": {
                "inner_count": 3,
                "inner_value": 42.5,
            }
        }
        result = privacy_guard.sanitize_for_llm(data)
        assert result["data"]["outer"]["inner_count"] == "<suppressed>"
        assert result["data"]["outer"]["inner_value"] == 42.5

    def test_list_sanitization(self, privacy_guard):
        data = {"items": [1, 5, 100]}
        result = privacy_guard.sanitize_for_llm(data)
        assert result["data"]["items"][0] == "<suppressed>"
        assert result["data"]["items"][1] == "<suppressed>"
        assert result["data"]["items"][2] == 100

    def test_phi_in_dict_keys(self, privacy_guard):
        data = {"john@hospital.com": "test"}
        result = privacy_guard.sanitize_for_llm(data)
        assert result["sanitized"] is True
        # The key should be redacted
        keys = list(result["data"].keys())
        assert "john@hospital.com" not in keys

    def test_sanitize_analysis_result(self, privacy_guard):
        result = privacy_guard.sanitize_analysis_result({
            "coefficient": 1.23,
            "p_value": 0.045,
            "n_events": 5,
        })
        assert result["sanitized"] is True
        assert result["data"]["n_events"] == "<suppressed>"
        assert result["data"]["coefficient"] == 1.23

    def test_custom_policy(self, strict_policy):
        guard = PrivacyGuard(policy=strict_policy)
        counts = {"A": 15, "B": 25}
        suppressed = guard.suppress_small_cells(counts)
        assert suppressed["A"] == "<suppressed>"
        assert suppressed["B"] == 25

    def test_zero_not_suppressed(self, privacy_guard):
        data = {"count": 0, "total": 100}
        result = privacy_guard.sanitize_for_llm(data)
        assert result["data"]["count"] == 0

    def test_boolean_not_suppressed(self, privacy_guard):
        data = {"flag": True}
        result = privacy_guard.sanitize_for_llm(data)
        assert result["data"]["flag"] is True


class TestAuditLog:
    def test_log_and_read(self, tmp_path):
        log_path = tmp_path / "test.audit.jsonl"
        audit = AuditLog(log_path=log_path)
        audit.log_tool_call(
            tool_name="get_schema",
            action="loaded schema",
            privacy_actions=["redacted PHI"],
            suppressed_count=2,
        )
        entries = audit.get_entries()
        assert len(entries) == 1
        assert entries[0].tool_name == "get_schema"
        assert entries[0].suppressed_count == 2

    def test_summary(self, tmp_path):
        log_path = tmp_path / "test.audit.jsonl"
        audit = AuditLog(log_path=log_path)
        audit.log_tool_call("tool_a", "action1", suppressed_count=1)
        audit.log_tool_call("tool_a", "action2", suppressed_count=3)
        audit.log_tool_call("tool_b", "action3", suppressed_count=0)

        summary = audit.summary()
        assert summary["total_entries"] == 3
        assert summary["entries_by_tool"]["tool_a"] == 2
        assert summary["total_suppressions"] == 4

    def test_empty_log(self, tmp_path):
        log_path = tmp_path / "empty.audit.jsonl"
        audit = AuditLog(log_path=log_path)
        assert audit.get_entries() == []
        summary = audit.summary()
        assert summary["total_entries"] == 0
