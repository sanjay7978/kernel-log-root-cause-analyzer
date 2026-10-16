"""
Unit tests for modules/preprocessor.py.

Verifies conservative log cleaning, ANSI sequence removal, indentation preservation,
consecutive deduplication, evidence detection, and determinism using pytest.
"""

import time

from modules.preprocessor import (
    preprocess_kernel_log,
    clean_log_text,
    strip_control_and_ansi,
    find_candidate_evidence,
    PreprocessedLog,
    PreprocessingStats,
)


def test_empty_and_whitespace_only_input():
    """Test that empty or whitespace-only inputs return empty results safely."""
    for empty_val in ["", "   ", "\n\n\t  \n"]:
        res = preprocess_kernel_log(empty_val)
        assert res.cleaned_text == ""
        assert res.stats.original_line_count == 0
        assert res.stats.cleaned_line_count == 0
        assert res.candidate_evidence_lines == []


def test_empty_lines_removal():
    """Test that blank and empty lines are removed and counted in stats."""
    raw = "[ 0.0] Line one\n\n   \n[ 0.1] Line two\n\n"
    res = preprocess_kernel_log(raw)

    assert res.stats.empty_lines_removed == 3
    assert res.cleaned_text == "[ 0.0] Line one\n[ 0.1] Line two\n"


def test_whitespace_normalization_preserves_indentation():
    """Test that trailing whitespace is trimmed while leading indentation (Call Traces) is preserved."""
    raw = (
        "[ 1.0] Hardware exception:  \t \n"
        "  [ 1.0]  Call Trace:    \n"
        "   ? dump_stack_lvl+0x48/0x70   \n"
        "   ? panic+0x120/0x340   \n"
    )
    res = preprocess_kernel_log(raw)

    lines = res.cleaned_text.splitlines()
    assert lines[0] == "[ 1.0] Hardware exception:"
    assert lines[1] == "  [ 1.0]  Call Trace:"
    assert lines[2] == "   ? dump_stack_lvl+0x48/0x70"
    assert lines[3] == "   ? panic+0x120/0x340"


def test_consecutive_duplicates_collapsed():
    """Test that identical consecutive flood lines are collapsed with repeat count annotation."""
    flood_line = "[ 2.0] sd 0:0:0:0: [sda] tag#0 FAILED Result: hostbyte=DID_OK"
    raw = "\n".join([flood_line] * 5)

    res = preprocess_kernel_log(raw)
    assert res.stats.duplicate_lines_collapsed == 4
    assert res.stats.cleaned_line_count == 2  # original line + 1 repeat annotation line

    expected_output = (
        f"{flood_line}\n"
        "    [... preceding message repeated 4 times ...]\n"
    )
    assert res.cleaned_text == expected_output


def test_non_consecutive_duplicates_preserved():
    """Test that identical lines separated in time are NOT removed (timeline preservation)."""
    raw = (
        "[ 1.0] link state UP\n"
        "[ 2.0] link state DOWN\n"
        "[ 3.0] link state UP\n"
    )
    res = preprocess_kernel_log(raw)

    # All three chronological events must be preserved
    assert res.stats.cleaned_line_count == 3
    assert res.stats.duplicate_lines_collapsed == 0
    assert res.cleaned_text == raw


def test_ansi_and_control_characters_stripped():
    """Test stripping ANSI color sequences and unprintable control bytes."""
    raw = "\x1b[31;1m[ 0.5] \x07Critical bus fault\x1b[0m\r\n[ 0.6] Normal \x08message\n"
    cleaned = clean_log_text(raw)

    assert "\x1b" not in cleaned
    assert "\x07" not in cleaned
    assert "\x08" not in cleaned
    assert "\r" not in cleaned
    assert cleaned == "[ 0.5] Critical bus fault\n[ 0.6] Normal message\n"


def test_timestamps_and_prefixes_preserved():
    """Test that timestamps and subsystem labels are untouched."""
    raw = (
        "[    0.000000] Linux version 6.8.0-generic\n"
        "[    0.123456] ACPI: Core revision 20230628\n"
        "[    1.500000] usb 1-1: new high-speed USB device number 2\n"
    )
    res = preprocess_kernel_log(raw)
    assert res.cleaned_text == raw


def test_candidate_evidence_detection_does_not_drop_normal_lines():
    """Test that candidate evidence lines are flagged and counted without discarding context lines."""
    raw = (
        "[ 0.1] Booting kernel on CPU 0\n"
        "[ 0.2] ACPI: Early initialization\n"
        "[ 0.3] pci 0000:00:1f.3: failed to probe device\n"
        "[ 0.4] Initializing network stack\n"
        "[ 0.5] Kernel panic - not syncing: Fatal exception\n"
    )
    res = preprocess_kernel_log(raw)

    # 5 total lines kept (contextual lines must not be dropped!)
    assert res.stats.cleaned_line_count == 5
    # 2 candidate evidence lines (failed, panic)
    assert res.stats.candidate_evidence_count == 2
    assert len(res.candidate_evidence_lines) == 2
    assert "failed to probe" in res.candidate_evidence_lines[0]
    assert "Kernel panic" in res.candidate_evidence_lines[1]


def test_preprocessor_determinism():
    """Test that the same input always yields identical output."""
    raw = "[ 1.0] Sample message\n[ 1.0] Sample message\n\n[ 2.0] Done"
    res_one = preprocess_kernel_log(raw)
    res_two = preprocess_kernel_log(raw)

    assert res_one.cleaned_text == res_two.cleaned_text
    assert res_one.stats == res_two.stats


def test_large_input_performance():
    """Test that processing large logs (10,000 lines) completes quickly and safely."""
    line = "[ 1.000000] kernel: normal operational status heartbeat\n"
    raw = line * 10000

    start = time.perf_counter()
    res = preprocess_kernel_log(raw)
    duration = time.perf_counter() - start

    # 10,000 identical lines collapsed to 1 + annotation
    assert res.stats.original_line_count == 10000
    assert res.stats.duplicate_lines_collapsed == 9999
    assert duration < 1.0  # Must finish in under 1 second
