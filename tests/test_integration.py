"""
Integration tests for Phase 1 pipeline on authentic real Linux kernel logs.
"""

from pathlib import Path
import pytest

from modules.log_loader import load_log_file
from modules.preprocessor import preprocess_kernel_log
from config import SAMPLE_LOGS_DIR


def test_real_oom_log_pipeline():
    """Test loading and preprocessing the authentic Ubuntu OOM incident log."""
    oom_log_path = SAMPLE_LOGS_DIR / "memory" / "real_oom_incident.log"
    assert oom_log_path.exists(), f"Log file missing: {oom_log_path}"

    raw_text = load_log_file(oom_log_path)
    assert len(raw_text) > 0
    assert "stress-ng invoked oom-killer" in raw_text

    result = preprocess_kernel_log(raw_text)
    assert result.stats.cleaned_line_count == 35
    assert result.stats.candidate_evidence_count >= 2

    # Verify Call Trace structure was preserved
    assert "Call Trace:" in result.cleaned_text
    assert "  dump_stack_lvl+0x48/0x70" in result.cleaned_text
    assert "Out of memory: Killed process" in result.cleaned_text


def test_real_usb_error_log_pipeline():
    """Test loading and preprocessing the authentic Linux USB error log."""
    usb_log_path = SAMPLE_LOGS_DIR / "usb" / "real_usb_descriptor_error.log"
    assert usb_log_path.exists(), f"Log file missing: {usb_log_path}"

    raw_text = load_log_file(usb_log_path)
    assert len(raw_text) > 0
    assert "unable to enumerate USB device" in raw_text

    result = preprocess_kernel_log(raw_text)
    # 15 raw lines; the repeated "error -110" and "not responding" lines collapse to 11
    assert result.stats.original_line_count == 15
    assert result.stats.cleaned_line_count == 11
    assert result.stats.duplicate_lines_collapsed == 7
    assert result.stats.candidate_evidence_count >= 3

    # Verify all timestamps and device names are preserved
    assert "xhci_hcd" in result.cleaned_text
    assert "error -110" in result.cleaned_text
    assert "error -71" in result.cleaned_text
