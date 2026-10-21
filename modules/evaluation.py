"""
Evaluation Module for Kernel Log Root Cause Analyzer.

Runs the full analysis pipeline over the labelled sample logs, several times
each, and summarises how well the local model did: whether it answered in a
usable form, named the right subsystem, quoted real lines, and how long it took.

Whether the *root cause* is right cannot be decided by a script, so each run
keeps the model's root cause next to the expected description from
sample_logs/metadata.json for a person to judge.
"""

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config import BASE_DIR
from modules.analyzer import PipelineExecutionResult, analyze_log


def load_dataset(logs_dir: Optional[Path] = None, category: Optional[str] = None) -> List[Dict[str, str]]:
    """Read sample_logs/metadata.json and return its entries (optionally for one category)."""
    root = Path(logs_dir) if logs_dir else BASE_DIR / "sample_logs"
    entries = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    for entry in entries:
        # filenames in metadata.json are relative to the project root
        entry["path"] = str((root.parent / entry["filename"]).resolve())
    if category:
        entries = [e for e in entries if e["category"] == category]
    return sorted(entries, key=lambda e: e["category"])


def record_from_result(entry: Dict[str, str], run: int, result: PipelineExecutionResult) -> Dict[str, Any]:
    """One row of results from a successful pipeline run."""
    a = result.analysis
    t = result.timing_metrics
    return {
        "category": entry["category"], "file": entry["filename"], "run": run, "ok": True, "error": None,
        "subsystem": a.subsystem, "severity": a.severity, "confidence": a.confidence,
        "root_cause": a.root_cause, "expected": entry.get("description", ""),
        "evidence_total": len(a.evidence), "evidence_unverified": len(a.unverified_evidence),
        "commands_total": len(a.commands), "commands_flagged": len(a.command_warnings),
        "total_s": round(t.get("total_pipeline_ms", 0) / 1000, 1),
        "prompt_tokens": int(t.get("ollama_prompt_tokens", 0)),
        "eval_tokens": int(t.get("ollama_eval_count", 0)),
        "cleaned_lines": result.original_cleaned_lines, "sent_lines": result.sent_lines,
        "context_truncated": result.context_truncated, "ollama_truncated": result.ollama_truncated,
    }


def record_from_error(entry: Dict[str, str], run: int, error: Exception) -> Dict[str, Any]:
    """One row for a run that failed (Ollama down, unparseable answer, ...)."""
    return {
        "category": entry["category"], "file": entry["filename"], "run": run, "ok": False,
        "error": f"{type(error).__name__}: {error}", "expected": entry.get("description", ""),
    }


def run_evaluation(
    entries: List[Dict[str, str]],
    runs: int = 3,
    analyze: Callable[..., PipelineExecutionResult] = analyze_log,
    model: Optional[str] = None,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Run every log `runs` times. A failed run is recorded, not raised, so one bad answer does not stop the rest."""
    records: List[Dict[str, Any]] = []
    for entry in entries:
        for run in range(1, runs + 1):
            try:
                record = record_from_result(entry, run, analyze(entry["path"], model=model))
            except Exception as err:  # noqa: BLE001 - any failure is a result worth recording
                record = record_from_error(entry, run, err)
            records.append(record)
            if progress:
                progress(record)
    return records


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Totals over all runs, plus one row per category."""
    good = [r for r in records if r["ok"]]
    times = [r["total_s"] for r in good]
    evidence = sum(r["evidence_total"] for r in good)
    unverified = sum(r["evidence_unverified"] for r in good)

    def correct_subsystem(r: Dict[str, Any]) -> bool:
        return r["subsystem"] == r["category"]

    per_category = []
    for category in dict.fromkeys(r["category"] for r in records):
        rows = [r for r in records if r["category"] == category]
        ok_rows = [r for r in rows if r["ok"]]
        per_category.append({
            "category": category,
            "runs": len(rows),
            "answered": len(ok_rows),
            "subsystem_correct": sum(correct_subsystem(r) for r in ok_rows),
            "subsystems": dict(Counter(r["subsystem"] for r in ok_rows)),
            "severities": dict(Counter(r["severity"] for r in ok_rows)),
            "mean_s": round(statistics.mean(r["total_s"] for r in ok_rows), 1) if ok_rows else None,
        })

    return {
        "runs": len(records),
        "answered": len(good),
        "failed": len(records) - len(good),
        "subsystem_correct": sum(correct_subsystem(r) for r in good),
        "evidence_total": evidence,
        "evidence_unverified": unverified,
        "commands_total": sum(r["commands_total"] for r in good),
        "commands_flagged": sum(r["commands_flagged"] for r in good),
        "severities": dict(Counter(r["severity"] for r in good)),
        "confidences": dict(Counter(r["confidence"].split()[0].upper() if r["confidence"] else "?" for r in good)),
        "context_truncated_runs": sum(r["context_truncated"] for r in good),
        "ollama_truncated_runs": sum(r["ollama_truncated"] for r in good),
        "seconds": {
            "mean": round(statistics.mean(times), 1), "median": round(statistics.median(times), 1),
            "min": min(times), "max": max(times),
        } if times else None,
        "per_category": per_category,
    }


def format_report(summary: Dict[str, Any], records: List[Dict[str, Any]], model: str) -> str:
    """A plain-text report: totals, a per-category table, and root causes for a person to judge."""
    s = summary
    lines = [f"Evaluation of {model}", "=" * 72]
    lines.append(f"Runs                    : {s['runs']} ({s['answered']} answered, {s['failed']} failed)")
    if s["answered"]:
        pct = 100 * s["subsystem_correct"] / s["answered"]
        lines.append(f"Subsystem as expected   : {s['subsystem_correct']} of {s['answered']} ({pct:.0f}%)")
        lines.append(f"Evidence lines in log   : {s['evidence_total'] - s['evidence_unverified']} of {s['evidence_total']}")
        lines.append(f"Commands flagged        : {s['commands_flagged']} of {s['commands_total']}")
        lines.append(f"Severity                : {s['severities']}")
        lines.append(f"Confidence              : {s['confidences']}")
        lines.append(f"Logs cut to fit context : {s['context_truncated_runs']} run(s); Ollama cut the prompt in {s['ollama_truncated_runs']} run(s)")
        sec = s["seconds"]
        lines.append(f"Seconds per analysis    : mean {sec['mean']}, median {sec['median']}, min {sec['min']}, max {sec['max']}")
    lines += ["", "Per category", "-" * 72, f"{'category':<12}{'answered':<10}{'subsystem ok':<14}{'returned':<30}{'mean s':<8}"]
    for c in s["per_category"]:
        returned = ", ".join(f"{k} x{v}" for k, v in c["subsystems"].items()) or "-"
        mean = "-" if c["mean_s"] is None else c["mean_s"]
        lines.append(f"{c['category']:<12}{str(c['answered']) + '/' + str(c['runs']):<10}{c['subsystem_correct']:<14}{returned:<30}{mean!s:<8}")
    lines += ["", "Root causes to judge by hand (model answer vs expected)", "-" * 72]
    for category in dict.fromkeys(r["category"] for r in records):
        rows = [r for r in records if r["category"] == category]
        lines.append(f"[{category}] expected: {rows[0]['expected']}")
        for r in rows:
            answer = r["root_cause"] if r["ok"] else f"FAILED - {r['error']}"
            lines.append(f"   run {r['run']}: {answer}")
    return "\n".join(lines)
