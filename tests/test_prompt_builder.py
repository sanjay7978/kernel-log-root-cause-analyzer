"""
Unit tests for modules/prompt_builder.py.
"""

import pytest
from modules.prompt_builder import (
    build_diagnostic_prompt,
    DIAGNOSTIC_SYSTEM_INSTRUCTIONS,
    JSON_SCHEMA_INSTRUCTIONS,
)


def test_build_prompt_contains_kernel_log():
    sample_log = "[ 1.234567] kernel: EXT4-fs error (device sda1): ext4_lookup:1500: inode #2: comm ls: deleted inode referenced"
    prompt = build_diagnostic_prompt(sample_log)

    assert sample_log in prompt
    assert "KERNEL LOG EVIDENCE TO ANALYZE:" in prompt
    assert "JSON DIAGNOSTIC REPORT:" in prompt


def test_build_prompt_contains_required_rules():
    sample_log = "[ 1.000000] usb 1-1: device descriptor read/64, error -110"
    prompt = build_diagnostic_prompt(sample_log)

    # Core grounding constraints
    assert "Base your conclusions strictly on the provided log lines" in prompt
    assert "Do NOT invent or hallucinate error messages" in prompt
    assert "DISTINGUISH FACT FROM INFERENCE" in prompt
    assert "ADMIT UNCERTAINTY" in prompt
    assert "COMMAND SAFETY" in prompt

    # Required JSON schema fields
    assert '"root_cause"' in prompt
    assert '"subsystem"' in prompt
    assert '"severity"' in prompt
    assert '"evidence"' in prompt
    assert '"explanation"' in prompt
    assert '"commands"' in prompt
    assert '"possible_fixes"' in prompt
    assert '"confidence"' in prompt
    assert '"uncertainty"' in prompt


def test_build_prompt_empty_input_raises_value_error():
    for empty_val in ["", "   ", "\n\n"]:
        with pytest.raises(ValueError) as exc_info:
            build_diagnostic_prompt(empty_val)
        assert "empty kernel log" in str(exc_info.value)


def test_build_prompt_with_optional_context():
    sample_log = "[ 0.500000] pci 0000:00:1f.3: failed to probe device"
    notes = "Server running Ubuntu 22.04 on Dell PowerEdge R740"
    prompt = build_diagnostic_prompt(sample_log, context_notes=notes)

    assert notes in prompt
    assert "ADDITIONAL CONTEXT:" in prompt


def test_prompt_determinism():
    sample_log = "[ 2.0] sd 0:0:0:0: [sda] tag#0 FAILED"
    p1 = build_diagnostic_prompt(sample_log)
    p2 = build_diagnostic_prompt(sample_log)
    assert p1 == p2
