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
from modules.analyzer import PipelineExecutionResult
from modules.preprocessor import PreprocessingStats
from modules.result_parser import DiagnosticAnalysis
from modules.ui_service import (
    extract_input_content,
    format_commands_list,
    format_confidence_badge,
    format_error_html,
    format_evidence_list,
    format_fixes_list,
    format_severity_badge,
    format_subsystem_badge,
    load_sample_log_content,
    read_input,
    render_analysing,
    render_empty_state,
    render_header,
    render_report,
    run_ui_analysis,
)
from modules.ui_theme import CSS, build_theme
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
    report_html, cleaned_log = run_ui_analysis(
        uploaded_file=None,
        pasted_text=pasted,
        ollama_client=mock_ollama_client,
    )

    assert "stress-ng" in report_html            # root cause
    assert "MEMORY" in report_html               # subsystem chip
    assert "kx-sev-high" in report_html          # severity chip
    assert "free -m" in report_html              # commands
    assert "Add swap space" in report_html       # fixes
    assert "raw lines" in report_html            # pipeline details
    assert "stress-ng invoked oom-killer" in cleaned_log


def test_run_ui_analysis_with_uploaded_file(mock_ollama_client, tmp_path: Path):
    log_file = tmp_path / "system_oom.log"
    log_file.write_text("[ 120.450123] stress-ng invoked oom-killer: gfp_mask=0x1100cca\n", encoding="utf-8")

    report_html, cleaned_log = run_ui_analysis(
        uploaded_file=str(log_file),
        pasted_text=None,
        ollama_client=mock_ollama_client,
    )

    assert "system_oom.log" in report_html
    assert "stress-ng" in report_html
    assert cleaned_log.strip() != ""


def test_run_ui_analysis_no_input_returns_friendly_error():
    report_html, cleaned_log = run_ui_analysis(uploaded_file=None, pasted_text="")
    assert "No input provided" in report_html
    assert "drop a .log or .txt file" in report_html
    assert cleaned_log == ""


def test_run_ui_analysis_empty_file_returns_error(tmp_path: Path):
    empty_file = tmp_path / "zero_byte.log"
    empty_file.touch()

    report_html, _ = run_ui_analysis(uploaded_file=str(empty_file), pasted_text="")
    assert "The log is empty" in report_html


def test_run_ui_analysis_ollama_unreachable_returns_friendly_error():
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.generate.side_effect = OllamaConnectionError("Connection refused to 127.0.0.1:11434")

    report_html, _ = run_ui_analysis(
        uploaded_file=None,
        pasted_text="[ 0.1] kernel sample line\n",
        ollama_client=mock_client,
    )

    assert "Ollama is not reachable" in report_html
    assert "ollama serve" in report_html


def test_read_input_returns_the_text_or_an_error_card(tmp_path: Path):
    raw, label, error = read_input(None, "[ 1.0] something failed\n")
    assert (raw, label, error) == ("[ 1.0] something failed", "Pasted Log Text", "")

    raw, label, error = read_input(None, "")
    assert raw == "" and "No input provided" in error


# ============================================================
# 4. Badge & Component Formatting Tests
# ============================================================

def test_format_severity_badges():
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        badge = format_severity_badge(sev)
        assert sev in badge                       # the word is always there, colour is never the only signal
        assert f"kx-sev-{sev.lower()}" in badge
    assert "kx-sev-unknown" in format_severity_badge("UNKNOWN")
    assert "kx-sev-unknown" in format_severity_badge("something else")


def test_format_subsystem_badge():
    sub_badge = format_subsystem_badge("memory")
    assert "MEMORY" in sub_badge


def test_formatters_escape_html_from_the_model():
    hostile = "<script>alert(1)</script>"
    assert "<script>" not in format_evidence_list([hostile])
    assert "<script>" not in format_commands_list([hostile], {hostile: hostile})
    assert "<script>" not in format_confidence_badge(hostile)
    assert "<script>" not in format_fixes_list([hostile])
    assert "<script>" not in format_error_html(hostile, hostile, hostile)


def test_format_evidence_list():
    evidence = [
        "[ 50.218542] usb 1-1: device descriptor read/64, error -110",
        "[ 51.162018] usb 1-1: reset high-speed USB device",
    ]
    html_out = format_evidence_list(evidence)
    assert "error -110" in html_out
    assert "kx-log" in html_out
    assert html_out.count("kx-ln") == 2          # one numbered row per line
    assert "kx-warn" not in html_out

    # Empty list
    assert "No direct log line citations" in format_evidence_list([])


def test_format_evidence_list_marks_lines_not_found_in_the_log():
    html_out = format_evidence_list(["real line", "invented line"], ["invented line"])
    assert html_out.count("Not found in the log") == 1
    assert html_out.count("kx-warn") == 1


def test_format_commands_list():
    cmds = ["dmesg | grep -i usb", "lsusb -v"]
    html_out = format_commands_list(cmds)
    assert "dmesg | grep -i usb" in html_out
    assert "lsusb -v" in html_out
    assert "Review before running" not in html_out

    # Empty list
    assert "No specific diagnostic commands" in format_commands_list([])


def test_format_commands_list_marks_flagged_commands_with_the_reason():
    html_out = format_commands_list(["lsusb", "rm -rf /"], {"rm -rf /": "'rm' deletes files"})
    assert html_out.count("Review before running") == 1
    assert "deletes files" in html_out


def test_format_fixes_list():
    assert format_fixes_list(["Replace the cable", "Update the driver"]).count("<li>") == 2
    assert "No remediation steps" in format_fixes_list([])


# ============================================================
# 4b. Whole-page rendering
# ============================================================

def _result(**changes):
    analysis = DiagnosticAnalysis(
        root_cause="USB device failed to enumerate", subsystem="usb", severity="MEDIUM",
        evidence=["usb 1-1: unable to enumerate USB device"], explanation="Timeouts, then reset.",
        commands=["lsusb"], possible_fixes=["Try another port"], confidence="HIGH (clear)",
        uncertainty="Cable or port?",
    )
    values = dict(
        analysis=analysis, preprocessed_stats=PreprocessingStats(15, 11, 0, 7, 3), cleaned_log_excerpt="x",
        timing_metrics={"total_pipeline_ms": 54000, "llm_inference_ms": 53000, "ollama_prompt_tokens": 1100,
                        "ollama_eval_count": 650},
        log_file_path="Pasted Log Text", model_name="qwen3:8b", original_cleaned_lines=11, sent_lines=11,
    )
    values.update(changes)
    return PipelineExecutionResult(**values)


def test_render_report_contains_every_section_in_one_block():
    html_out = render_report(_result(), "Pasted Log Text")

    for heading in ["Probable root cause", "Evidence from the log", "Explanation", "What the log does not show",
                    "Commands to investigate", "Possible fixes", "Pipeline details"]:
        assert heading in html_out
    assert "USB device failed to enumerate" in html_out
    assert "kx-sev-medium" in html_out and "confidence" in html_out
    assert "54 s" in html_out and "qwen3:8b" in html_out
    assert "Partial log analysed" not in html_out    # nothing was cut


def test_render_report_warns_when_the_log_was_cut():
    html_out = render_report(
        _result(context_truncated=True, original_cleaned_lines=735, sent_lines=100, omitted_lines=635), "big.log",
    )
    assert "Partial log analysed" in html_out
    assert "100 of 735" in html_out

    assert "Ollama also cut the prompt" in render_report(_result(ollama_truncated=True), "x")


def test_render_report_flags_unverified_evidence_and_commands():
    result = _result()
    result.analysis.unverified_evidence = ["usb 1-1: unable to enumerate USB device"]
    result.analysis.command_warnings = {"lsusb": "'lsusb' is invented for this test"}

    html_out = render_report(result, "x")
    assert "Not found in the log" in html_out
    assert "Review before running" in html_out


def test_page_states_empty_analysing_and_header():
    assert "press Analyse" in render_empty_state()
    busy = render_analysing("big.log")
    assert "Analysing" in busy and "big.log" in busy and "kx-bar" in busy
    assert "ollama online" in render_header("qwen3:8b", True)
    assert "ollama not reachable" in render_header("qwen3:8b", False)
    assert "checking ollama" in render_header("qwen3:8b", None)


def test_clear_resets_every_wired_output():
    cleared = gradio_app.clear_inputs_and_outputs()
    assert len(cleared) == 6
    assert cleared[0] is None and cleared[1] == "" and cleared[3] is None
    assert "press Analyse" in cleared[4]
    assert cleared[5] == ""


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


def test_the_page_makes_no_outside_requests_and_gradio_analytics_are_off():
    """The tool promises nothing leaves the machine, so no fonts, CDNs or analytics."""
    import os

    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"
    assert "http" not in CSS and "@import" not in CSS and "url(" not in CSS
    assert gradio_app.create_ui().analytics_enabled is False


def test_theme_builds():
    assert isinstance(build_theme(), gr.themes.Base)
