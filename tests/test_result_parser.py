"""
Unit tests for modules/result_parser.py.
"""

import pytest
from modules.result_parser import (
    DiagnosticAnalysis,
    MalformedJSONError,
    ValidationError,
    extract_json_str,
    normalize_severity,
    normalize_subsystem,
    parse_and_validate_analysis,
)


def test_parse_valid_pure_json():
    raw = """{
        "root_cause": "System ran out of memory triggering OOM killer",
        "subsystem": "memory",
        "severity": "HIGH",
        "evidence": [
            "stress-ng invoked oom-killer: gfp_mask=0x1100cca",
            "Out of memory: Killed process 3412 (stress-ng-vm)"
        ],
        "explanation": "Memory consumption exceeded physical limits.",
        "commands": ["free -h", "vmstat 1 5"],
        "possible_fixes": ["Increase swap size", "Limit stress-ng memory"],
        "confidence": "HIGH (clear kernel call trace and OOM notification)",
        "uncertainty": "None"
    }"""

    analysis = parse_and_validate_analysis(raw)
    assert isinstance(analysis, DiagnosticAnalysis)
    assert "OOM killer" in analysis.root_cause
    assert analysis.subsystem == "memory"
    assert analysis.severity == "HIGH"
    assert len(analysis.evidence) == 2
    assert len(analysis.commands) == 2
    assert len(analysis.possible_fixes) == 2
    assert "HIGH" in analysis.confidence


def test_parse_markdown_code_fenced_json():
    raw = """Here is the diagnostic report:
```json
{
    "root_cause": "USB peripheral failed descriptor read",
    "subsystem": "usb",
    "severity": "MEDIUM",
    "evidence": ["usb 1-1: device descriptor read/64, error -110"],
    "explanation": "The USB device timed out during handshake.",
    "commands": ["lsusb -v", "dmesg | grep -i usb"],
    "possible_fixes": ["Check USB cable"],
    "confidence": "HIGH",
    "uncertainty": "Exact hardware fault unknown without testing port"
}
```
Hope this helps!"""

    analysis = parse_and_validate_analysis(raw)
    assert analysis.subsystem == "usb"
    assert analysis.severity == "MEDIUM"
    assert "error -110" in analysis.evidence[0]


def test_parse_with_think_tags():
    raw = """<think>
The user provided a log showing kernel panic.
I need to output valid JSON.
</think>
{
    "root_cause": "Kernel panic due to null pointer dereference",
    "subsystem": "panic",
    "severity": "CRITICAL",
    "evidence": ["Kernel panic - not syncing: Fatal exception"],
    "explanation": "Fatal kernel crash halted the system.",
    "commands": ["journalctl -k -b -1"],
    "possible_fixes": ["Update driver"],
    "confidence": "HIGH",
    "uncertainty": "None"
}"""

    analysis = parse_and_validate_analysis(raw)
    assert analysis.subsystem == "panic"
    assert analysis.severity == "CRITICAL"


def test_parse_malformed_json_raises():
    raw = "Sorry, I cannot analyze this log because it is incomplete."
    with pytest.raises(MalformedJSONError) as exc_info:
        parse_and_validate_analysis(raw)
    assert "Failed to parse" in str(exc_info.value) or "empty" in str(exc_info.value)


def test_parse_empty_input_raises():
    with pytest.raises(MalformedJSONError):
        parse_and_validate_analysis("")


def test_parse_missing_root_cause_raises_validation_error():
    raw = '{"subsystem": "memory", "severity": "HIGH"}'
    with pytest.raises(ValidationError) as exc_info:
        parse_and_validate_analysis(raw)
    assert "missing mandatory field: 'root_cause'" in str(exc_info.value)


def test_severity_normalization():
    assert normalize_severity("high") == "HIGH"
    assert normalize_severity(" Critical ") == "CRITICAL"
    assert normalize_severity("medium") == "MEDIUM"
    assert normalize_severity("low") == "LOW"
    assert normalize_severity("Severity: HIGH") == "HIGH"
    assert normalize_severity("INVALID_VAL") == "UNKNOWN"
    assert normalize_severity(None) == "UNKNOWN"


def test_subsystem_normalization():
    assert normalize_subsystem("Memory Management") == "memory"
    assert normalize_subsystem("USB Subsystem") == "usb"
    assert normalize_subsystem("storage") == "storage"
    assert normalize_subsystem("Unknown Peripheral") == "other"
    assert normalize_subsystem(None) == "other"


def test_type_coercion_for_lists():
    # If the LLM returns a single string instead of a list
    raw = """{
        "root_cause": "Storage drive timeout",
        "subsystem": "storage",
        "severity": "HIGH",
        "evidence": "Single evidence string",
        "commands": "smartctl -a /dev/sda",
        "possible_fixes": null
    }"""
    analysis = parse_and_validate_analysis(raw)
    assert analysis.evidence == ["Single evidence string"]
    assert analysis.commands == ["smartctl -a /dev/sda"]
    assert analysis.possible_fixes == []
