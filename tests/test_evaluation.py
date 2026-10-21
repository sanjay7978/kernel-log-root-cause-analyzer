"""
Unit tests for modules/evaluation.py and evaluate.py, using a fake analyzer (no Ollama needed).
"""

import json
import sys

import pytest

import evaluate
from modules.analyzer import PipelineExecutionResult
from modules.evaluation import (
    format_report,
    load_dataset,
    run_evaluation,
    summarise,
)
from modules.preprocessor import PreprocessingStats
from modules.result_parser import DiagnosticAnalysis


def _result(subsystem="usb", severity="HIGH", unverified=(), flagged=None, seconds=60.0, truncated=False):
    analysis = DiagnosticAnalysis(
        root_cause=f"{subsystem} problem", subsystem=subsystem, severity=severity,
        evidence=["a", "b", "c"], commands=["lsusb", "rm x"], confidence="HIGH (clear)",
    )
    analysis.unverified_evidence = list(unverified)
    analysis.command_warnings = dict(flagged or {})
    return PipelineExecutionResult(
        analysis=analysis, preprocessed_stats=PreprocessingStats(10, 10, 0, 0, 2), cleaned_log_excerpt="x",
        timing_metrics={"total_pipeline_ms": seconds * 1000, "ollama_prompt_tokens": 1200, "ollama_eval_count": 700},
        context_truncated=truncated, original_cleaned_lines=10, sent_lines=10,
    )


ENTRIES = [
    {"category": "usb", "filename": "sample_logs/usb/a.log", "path": "/x/a.log", "description": "usb thing"},
    {"category": "gpu", "filename": "sample_logs/gpu/b.log", "path": "/x/b.log", "description": "gpu thing"},
]


def test_dataset_loads_all_nine_sample_logs_and_they_exist():
    entries = load_dataset()
    assert len(entries) == 9
    assert {e["category"] for e in entries} == {"acpi", "driver", "filesystem", "gpu", "memory",
                                                "network", "panic", "storage", "usb"}
    from pathlib import Path
    assert all(Path(e["path"]).is_file() for e in entries)


def test_dataset_can_be_limited_to_one_category():
    assert [e["category"] for e in load_dataset(category="usb")] == ["usb"]
    assert load_dataset(category="nonexistent") == []


def test_every_log_is_run_the_requested_number_of_times():
    calls = []

    def fake(path, model=None):
        calls.append((path, model))
        return _result()

    records = run_evaluation(ENTRIES, runs=3, analyze=fake, model="m")
    assert len(records) == 6
    assert calls.count(("/x/a.log", "m")) == 3
    assert [r["run"] for r in records if r["category"] == "usb"] == [1, 2, 3]
    assert all(r["ok"] for r in records)


def test_a_failed_run_is_recorded_and_the_rest_continue():
    results = iter([_result(), RuntimeError("Ollama is down")])

    def fake(path, model=None):
        item = next(results)
        if isinstance(item, Exception):
            raise item
        return item

    seen = []
    records = run_evaluation([ENTRIES[0], ENTRIES[1]], runs=1, analyze=fake, progress=seen.append)
    # first entry ok, second entry hits the error
    assert [r["ok"] for r in records] == [True, False]
    assert "RuntimeError: Ollama is down" in records[1]["error"]
    assert len(seen) == 2


def test_summary_counts_correct_subsystems_evidence_commands_and_failures():
    records = [
        {**_record(_result("usb"), ENTRIES[0], 1)},
        {**_record(_result("usb", unverified=["a"], flagged={"rm x": "deletes"}, truncated=True), ENTRIES[0], 2)},
        {**_record(_result("driver", severity="MEDIUM"), ENTRIES[1], 1)},   # wrong subsystem for gpu
        {"category": "gpu", "file": "f", "run": 2, "ok": False, "error": "TimeoutError: x", "expected": "gpu thing"},
    ]
    s = summarise(records)

    assert (s["runs"], s["answered"], s["failed"]) == (4, 3, 1)
    assert s["subsystem_correct"] == 2
    assert s["evidence_total"] == 9 and s["evidence_unverified"] == 1
    assert s["commands_total"] == 6 and s["commands_flagged"] == 1
    assert s["severities"] == {"HIGH": 2, "MEDIUM": 1}
    assert s["confidences"] == {"HIGH": 3}
    assert s["context_truncated_runs"] == 1
    assert s["seconds"]["mean"] == 60.0
    gpu = next(c for c in s["per_category"] if c["category"] == "gpu")
    assert (gpu["runs"], gpu["answered"], gpu["subsystem_correct"]) == (2, 1, 0)
    assert gpu["subsystems"] == {"driver": 1}


def _record(result, entry, run):
    from modules.evaluation import record_from_result
    return record_from_result(entry, run, result)


def test_summary_handles_a_run_where_everything_failed():
    records = [{"category": "usb", "file": "f", "run": 1, "ok": False, "error": "x", "expected": "e"}]
    s = summarise(records)

    assert s["answered"] == 0 and s["seconds"] is None
    assert "0 answered, 1 failed" in format_report(s, records, "m")


def test_report_shows_totals_and_root_causes_for_manual_judging():
    records = [_record(_result("usb"), ENTRIES[0], 1), _record(_result("usb"), ENTRIES[0], 2)]
    report = format_report(summarise(records), records, "qwen3:8b")

    assert "Evaluation of qwen3:8b" in report
    assert "Subsystem as expected   : 2 of 2 (100%)" in report
    assert "[usb] expected: usb thing" in report
    assert "run 1: usb problem" in report and "run 2: usb problem" in report


def test_cli_runs_saves_results_and_prints_the_report(monkeypatch, tmp_path, capsys):
    out = tmp_path / "res.json"

    def fake_run(entries, runs, model, progress):
        records = [_record(_result("usb"), entries[0], r) for r in range(1, runs + 1)]
        for r in records:
            progress(r)
        return records

    monkeypatch.setattr(evaluate, "run_evaluation", fake_run)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--category", "usb", "--runs", "2", "--out", str(out)])

    assert evaluate.main() == 0
    saved = json.loads(out.read_text())
    assert saved["summary"]["runs"] == 2 and len(saved["runs"]) == 2
    text = capsys.readouterr().out
    assert "Evaluating" in text and "Subsystem as expected" in text


def test_cli_rejects_zero_runs(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--runs", "0"])
    with pytest.raises(SystemExit):
        evaluate.main()
