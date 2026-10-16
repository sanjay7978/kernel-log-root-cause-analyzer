"""
Comprehensive unit and integration tests for Phase 3A:
- Unified in-memory log analysis (modules/analyzer.py)
- UI service controller functions (modules/ui_service.py)
- Input extraction (uploaded files vs. pasted text)
- Safety constraints and size limits
- Error handling and visual badge formatting
- Gradio Blocks UI hierarchy construction (gradio_app.py)
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import gradio as gr

from config import MAX_LOG_FILE_SIZE_BYTES, SAMPLE_LOGS_DIR
from modules.analyzer import analyze_log, analyze_raw_content
from modules.llm_analyzer import LLMResponse, OllamaClient, OllamaConnectionError
from modules.log_loader import LogFileEmptyError, LogFileTooLargeError
from modules.ui_service import (
    extract_input_content,
    format_commands_list,
    format_evidence_list,
    format_severity_badge,
    format_subsystem_badge,
    load_sample_log_content,
    run_ui_analysis,
)
import gradio_app


@pytest.fixture
def mock_ollama_client():
    """Fixture providing a mock OllamaClient with a valid structured response."""
    mock_client = MagicMock(spec=OllamaClient)
    mock_response = LLMResponse(
        content="""{
            "root_cause": "Process stress-ng exhausted system memory invoking the OOM killer",
            "subsystem": "memory",
            "severity": "HIGH",
            "evidence": [
                "stress-ng invoked oom-killer: gfp_mask=0x1100cca",
                "Out of memory: Killed process 3412 (stress-ng-vm)"
            ],
            "explanation": "High memory allocation triggered out_of_memory in page allocator.",
            "commands": ["free -m", "dmesg -T | grep -i oom"],
            "possible_fixes": ["Configure cgroup limits", "Add swap space"],
            "confidence": "HIGH",
            "uncertainty": "None"
        }""",
        model="qwen3:8b",
        total_duration_ms=2500.0,
        eval_duration_ms=1800.0,
        eval_count=120,
    )
    mock_client.generate.return_value = mock_response
    mock_client.model = "qwen3:8b"
    return mock_client


# ============================================================
# 1. In-Memory Analysis Service Tests (modules/analyzer.py)
# ============================================================

def test_analyze_raw_content_valid(mock_ollama_client):
    raw_log = "[ 1.000000] usb 1-1: device descriptor read/64, error -110\n"
    result = analyze_raw_content(
        raw_content=raw_log,
        source_label="Test Raw Log",
        ollama_client=mock_ollama_client,
    )

    assert result.analysis.subsystem == "memory"
    assert result.analysis.severity == "HIGH"
    assert "total_pipeline_ms" in result.timing_metrics
    assert result.log_file_path == "Test Raw Log"


def test_analyze_raw_content_empty_raises():
    for blank in ["", "   ", "\n\n\t  \n"]:
        with pytest.raises(LogFileEmptyError):
            analyze_raw_content(blank)


def test_analyze_raw_content_size_exceeded_raises():
    huge_text = "A" * (MAX_LOG_FILE_SIZE_BYTES + 10)
    with pytest.raises(LogFileTooLargeError) as exc_info:
        analyze_raw_content(huge_text)
    assert "exceeds safety limit" in str(exc_info.value)


# ============================================================
# 2. Input Extraction Tests (modules/ui_service.py)
# ============================================================

def test_extract_input_content_pasted_text():
    pasted = "[ 0.123] kernel boot\n"
    content, source = extract_input_content(uploaded_file=None, pasted_text=pasted)
    assert content == pasted.strip()
    assert source == "Pasted Log Text"


def test_extract_input_content_uploaded_file(tmp_path: Path):
    test_file = tmp_path / "uploaded_kernel.log"
    test_file.write_text("[ 1.234] Disk I/O error\n", encoding="utf-8")

    content, source = extract_input_content(uploaded_file=str(test_file), pasted_text="")
    assert content == "[ 1.234] Disk I/O error\n"
    assert "uploaded_kernel.log" in source


def test_extract_input_content_no_input_raises_value_error():
    with pytest.raises(ValueError) as exc_info:
        extract_input_content(uploaded_file=None, pasted_text="")
    assert "either uploading a file or pasting" in str(exc_info.value)


# ============================================================
# 3. UI Controller Analysis Function Tests
# ============================================================

def test_run_ui_analysis_with_pasted_log(mock_ollama_client):
    pasted = "[ 120.450123] stress-ng invoked oom-killer: gfp_mask=0x1100cca\n"
    outputs = run_ui_analysis(
        uploaded_file=None,
        pasted_text=pasted,
        ollama_client=mock_ollama_client,
    )

    status_html, root_cause, sub_badge, sev_badge, conf_badge, evidence, exp, cmds, fixes, unc, metrics = outputs

    assert "Analysis Complete" in status_html
    assert "stress-ng" in root_cause
    assert "MEMORY" in sub_badge
    assert "HIGH" in sev_badge
    assert "free -m" in cmds
    assert "cgroup limits" in fixes
    assert "Raw Input Lines" in metrics


def test_run_ui_analysis_with_uploaded_file(mock_ollama_client, tmp_path: Path):
    log_file = tmp_path / "system_oom.log"
    log_file.write_text("[ 120.450123] stress-ng invoked oom-killer: gfp_mask=0x1100cca\n", encoding="utf-8")

    outputs = run_ui_analysis(
        uploaded_file=str(log_file),
        pasted_text=None,
        ollama_client=mock_ollama_client,
    )

    status_html, root_cause, sub_badge, sev_badge, *rest = outputs
    assert "Analysis Complete" in status_html
    assert "system_oom.log" in status_html
    assert "stress-ng" in root_cause


def test_run_ui_analysis_no_input_returns_friendly_error():
    outputs = run_ui_analysis(uploaded_file=None, pasted_text="")
    status_html = outputs[0]
    assert "No Input Provided" in status_html
    assert "Drag and drop a .log file" in status_html


def test_run_ui_analysis_empty_file_returns_error(tmp_path: Path):
    empty_file = tmp_path / "zero_byte.log"
    empty_file.touch()

    outputs = run_ui_analysis(uploaded_file=str(empty_file), pasted_text="")
    status_html = outputs[0]
    assert "Log File Is Empty" in status_html


def test_run_ui_analysis_ollama_unreachable_returns_friendly_error():
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.generate.side_effect = OllamaConnectionError("Connection refused to 127.0.0.1:11434")

    outputs = run_ui_analysis(
        uploaded_file=None,
        pasted_text="[ 0.1] kernel sample line\n",
        ollama_client=mock_client,
    )

    status_html = outputs[0]
    assert "Ollama Daemon Unreachable" in status_html
    assert "ollama serve" in status_html


# ============================================================
# 4. Badge & Component Formatting Tests
# ============================================================

def test_format_severity_badges():
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]:
        badge = format_severity_badge(sev)
        assert sev in badge
        assert "border-radius" in badge


def test_format_subsystem_badge():
    sub_badge = format_subsystem_badge("memory")
    assert "MEMORY" in sub_badge


def test_format_evidence_list():
    evidence = [
        "[ 50.218542] usb 1-1: device descriptor read/64, error -110",
        "[ 51.162018] usb 1-1: reset high-speed USB device",
    ]
    html_out = format_evidence_list(evidence)
    assert "error -110" in html_out
    assert "font-family:ui-monospace" in html_out

    # Empty list
    assert "No direct log line citations" in format_evidence_list([])


def test_format_commands_list():
    cmds = ["dmesg | grep -i usb", "lsusb -v"]
    html_out = format_commands_list(cmds)
    assert "dmesg | grep -i usb" in html_out
    assert "lsusb -v" in html_out

    # Empty list
    assert "No specific diagnostic commands" in format_commands_list([])


# ============================================================
# 5. Sample Log Loader Test
# ============================================================

def test_load_sample_log_content():
    oom_content = load_sample_log_content("memory", "real_oom_incident.log")
    assert "stress-ng invoked oom-killer" in oom_content

    usb_content = load_sample_log_content("usb", "real_usb_descriptor_error.log")
    assert "device descriptor read/64, error -110" in usb_content

    with pytest.raises(FileNotFoundError):
        load_sample_log_content("nonexistent", "missing.log")


def test_load_all_subsystem_sample_logs():
    """Verify that all 9 subsystem sample logs exist, load cleanly, and integrate with UI."""
    assert len(gradio_app.SAMPLE_LOG_CHOICES) == 9
    for label, (cat, fname) in gradio_app.SAMPLE_LOG_CHOICES.items():
        content = load_sample_log_content(cat, fname)
        assert len(content.strip()) > 0
        file_input, pasted_text, status_msg = gradio_app.load_sample_by_choice(label)
        assert file_input is None
        assert pasted_text == content
        assert f"sample_logs/{cat}/{fname}" in status_msg

    # Test invalid choice returns gr.update()
    assert gradio_app.load_sample_by_choice("Invalid Choice")[2] == ""


# ============================================================
# 6. Gradio Blocks UI Construction Test
# ============================================================

def test_create_ui_hierarchy():
    """Verify that create_ui initializes without runtime errors and constructs valid Gradio Blocks."""
    demo = gradio_app.create_ui()
    assert isinstance(demo, gr.Blocks)
    assert demo.title == "Kernel Log Root Cause Analyzer"
