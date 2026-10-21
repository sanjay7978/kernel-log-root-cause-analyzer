"""
Unit tests for modules/report_export.py: the Markdown and JSON reports and the saved files.
"""

import json
from pathlib import Path

import pytest

from modules import report_export
from modules.analyzer import PipelineExecutionResult
from modules.event_extractor import summarise_events
from modules.preprocessor import PreprocessingStats
from modules.report_export import (
    KEEP_REPORTS,
    cleanup_report_files,
    report_data,
    save_report_files,
    to_json,
    to_markdown,
)
from modules.result_parser import DiagnosticAnalysis

LOG = (
    "[   50.218542] usb 1-1: device descriptor read/64, error -110\n"
    "[   57.950201] usb 1-1: unable to enumerate USB device\n"
)


def _result(**changes):
    analysis = DiagnosticAnalysis(
        root_cause="USB device failed to enumerate | after timeouts", subsystem="usb", severity="MEDIUM",
        evidence=["usb 1-1: unable to enumerate USB device", "kernel BUG at usb.c:1!"],
        explanation="Timeouts, then reset.", commands=["lsusb", "rm -rf /tmp/x"], possible_fixes=["Try another port"],
        confidence="HIGH (clear)", uncertainty="Cable or port?",
    )
    analysis.unverified_evidence = ["kernel BUG at usb.c:1!"]
    analysis.command_warnings = {"rm -rf /tmp/x": "'rm' deletes files"}
    values = dict(
        analysis=analysis, preprocessed_stats=PreprocessingStats(15, 11, 0, 7, 3), cleaned_log_excerpt=LOG,
        timing_metrics={"total_pipeline_ms": 54000, "llm_inference_ms": 53000, "ollama_prompt_tokens": 1100,
                        "ollama_eval_count": 650},
        log_file_path="Pasted Log Text", model_name="qwen3:8b", original_cleaned_lines=11, sent_lines=11,
        timeline=summarise_events(LOG),
    )
    values.update(changes)
    return PipelineExecutionResult(**values)


@pytest.fixture(autouse=True)
def _clean_folder():
    yield
    cleanup_report_files()


def test_json_report_holds_the_analysis_timeline_pipeline_and_checks():
    data = json.loads(to_json(_result(), "Pasted Log Text"))

    assert data["source"] == "Pasted Log Text" and data["model"] == "qwen3:8b"
    assert data["analysis"]["subsystem"] == "usb"
    assert data["analysis"]["unverified_evidence"] == ["kernel BUG at usb.c:1!"]
    assert data["analysis"]["command_warnings"] == {"rm -rf /tmp/x": "'rm' deletes files"}
    assert data["timeline"]["total_events"] == 2
    assert data["pipeline"]["total_seconds"] == 54.0 and data["pipeline"]["prompt_tokens"] == 1100
    assert "probable root cause" in data["note"]


def test_markdown_report_has_every_section_and_marks_the_flagged_items():
    md = to_markdown(_result(), "Pasted Log Text")

    for heading in ["# Kernel log analysis report", "## Probable root cause", "## Timeline", "## Evidence from the log",
                    "## Explanation", "## What the log does not show", "## Commands to investigate",
                    "## Possible fixes", "## Pipeline details"]:
        assert heading in md
    assert "**Subsystem:** usb  |  **Severity:** MEDIUM" in md
    assert "`kernel BUG at usb.c:1!`  *(not found in the log)*" in md
    assert "`rm -rf /tmp/x`  *(review before running: 'rm' deletes files)*" in md
    assert "| 50.219 s | usb | usb 1-1: device descriptor read/64, error -110 | 1 |" in md
    assert "It can be wrong" in md                       # the disclaimer is in every report
    assert "Partial log analysed" not in md


def test_markdown_escapes_pipes_in_table_cells_and_says_when_the_log_was_cut():
    result = _result(context_truncated=True, original_cleaned_lines=735, sent_lines=100, omitted_lines=635)
    result.timeline.groups[0].text = "a | b"

    md = to_markdown(result, "big.log")
    assert "a \\| b" in md
    assert "Only 100 of 735" in md


def test_markdown_copes_with_no_evidence_commands_fixes_or_timeline():
    result = _result(timeline=summarise_events("[ 1.0] Linux version 6.5.0\n"))
    result.analysis.evidence, result.analysis.commands, result.analysis.possible_fixes = [], [], []

    md = to_markdown(result, "x")
    assert "No log lines were quoted." in md and "No commands suggested." in md and "No fixes suggested." in md
    assert "## Timeline" not in md


def test_files_are_saved_and_only_the_latest_few_are_kept(monkeypatch):
    stamps = iter(f"20260101-0000{n:02d}" for n in range(30))
    monkeypatch.setattr(report_export.time, "strftime", lambda fmt: next(stamps))

    paths = [save_report_files(f"# report {n}", json.dumps({"n": n})) for n in range(KEEP_REPORTS + 3)]

    last_md, last_json = map(Path, paths[-1])
    assert last_md.read_text() == f"# report {KEEP_REPORTS + 2}"
    assert json.loads(last_json.read_text()) == {"n": KEEP_REPORTS + 2}
    assert len(list(last_md.parent.glob("*.md"))) == KEEP_REPORTS
    assert not Path(paths[0][0]).exists()            # the oldest were removed


def test_report_data_notes_when_ollama_cut_the_prompt():
    assert report_data(_result(ollama_truncated=True), "x")["pipeline"]["ollama_cut_prompt"] is True
