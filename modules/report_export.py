"""
Report Export Module for Kernel Log Root Cause Analyzer.

Turns one analysis into a file someone can keep, attach or hand in:

  * Markdown: a readable report (root cause, timeline, evidence, commands, fixes)
  * JSON: the same content as data, with the pipeline numbers and the checks

Both say plainly that the diagnosis is a probable cause produced by a local
language model, and mark evidence and commands the checks flagged.
"""

import json
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from modules.analyzer import PipelineExecutionResult

DISCLAIMER = (
    "This is a probable root cause produced by a local language model from the log above. "
    "It can be wrong. Check the evidence and review every command before running it."
)
KEEP_REPORTS = 10

_report_dir: Optional[Path] = None


def report_data(result: PipelineExecutionResult, source_label: str) -> Dict[str, Any]:
    """Everything in the report as plain data."""
    s, t = result.preprocessed_stats, result.timing_metrics
    return {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source_label,
        "model": result.model_name,
        "analysis": result.analysis.to_dict(),
        "timeline": result.timeline.to_dict() if result.timeline else None,
        "pipeline": {
            "raw_lines": s.original_line_count,
            "cleaned_lines": s.cleaned_line_count,
            "repeats_collapsed": s.duplicate_lines_collapsed,
            "lines_sent_to_model": result.sent_lines or s.cleaned_line_count,
            "log_cut_to_fit_model": result.context_truncated,
            "omitted_lines": result.omitted_lines,
            "ollama_cut_prompt": result.ollama_truncated,
            "prompt_tokens": int(t.get("ollama_prompt_tokens", 0)),
            "answer_tokens": int(t.get("ollama_eval_count", 0)),
            "model_seconds": round(t.get("llm_inference_ms", 0) / 1000, 1),
            "total_seconds": round(t.get("total_pipeline_ms", 0) / 1000, 1),
        },
        "note": DISCLAIMER,
    }


def to_json(result: PipelineExecutionResult, source_label: str) -> str:
    return json.dumps(report_data(result, source_label), indent=2, ensure_ascii=False)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _fmt_time(seconds: Optional[float]) -> str:
    return f"{seconds:.3f} s" if seconds is not None else "-"


def to_markdown(result: PipelineExecutionResult, source_label: str) -> str:
    """A readable report."""
    a = result.analysis
    data = report_data(result, source_label)
    p = data["pipeline"]
    out = [
        "# Kernel log analysis report", "",
        f"- **Source:** {source_label}",
        f"- **Model:** {result.model_name} (run locally with Ollama)",
        f"- **Generated:** {data['generated']}",
        f"- **Analysis time:** {p['total_seconds']:g} s", "",
        "## Probable root cause", "", a.root_cause, "",
        f"**Subsystem:** {a.subsystem}  |  **Severity:** {a.severity}  |  **Confidence:** {a.confidence}", "",
    ]
    if result.context_truncated:
        out += [f"> **Partial log analysed.** Only {p['lines_sent_to_model']} of {result.original_cleaned_lines} "
                f"lines were sent to the model ({p['omitted_lines']} routine lines left out).", ""]

    groups = result.timeline.groups if result.timeline else []
    if groups:
        out += ["## Timeline", "", "| Time | Subsystem | Event | Count |", "| --- | --- | --- | --- |"]
        for g in groups:
            out.append(f"| {_fmt_time(g.first_time)} | {g.subsystem} | {_cell(g.text)} | {g.count} |")
        out.append("")

    out += ["## Evidence from the log", ""]
    if a.evidence:
        for number, item in enumerate(a.evidence, start=1):
            flag = "  *(not found in the log)*" if item in a.unverified_evidence else ""
            out.append(f"{number}. `{item.strip()}`{flag}")
    else:
        out.append("No log lines were quoted.")
    out += ["", "## Explanation", "", a.explanation or "None given.", "",
            "## What the log does not show", "", a.uncertainty or "None given.", "",
            "## Commands to investigate", ""]
    if a.commands:
        for cmd in a.commands:
            reason = a.command_warnings.get(cmd)
            out.append(f"- `{cmd}`" + (f"  *(review before running: {reason})*" if reason else ""))
    else:
        out.append("No commands suggested.")
    out += ["", "## Possible fixes", ""]
    out += [f"{n}. {fix}" for n, fix in enumerate(a.possible_fixes, start=1)] or ["No fixes suggested."]
    out += [
        "", "## Pipeline details", "",
        f"- Raw lines: {p['raw_lines']}", f"- Cleaned lines: {p['cleaned_lines']} ({p['repeats_collapsed']} repeats collapsed)",
        f"- Lines sent to the model: {p['lines_sent_to_model']}",
        f"- Tokens: {p['prompt_tokens']} in, {p['answer_tokens']} out",
        f"- Model time: {p['model_seconds']:g} s", "", "---", "", f"*{DISCLAIMER}*", "",
    ]
    return "\n".join(out)


def save_report_files(markdown: str, json_text: str) -> Tuple[str, str]:
    """
    Write the two files to a temporary folder and return their paths.

    Only the latest few reports are kept, so the folder does not grow without limit.
    """
    global _report_dir
    if _report_dir is None or not _report_dir.exists():
        _report_dir = Path(tempfile.mkdtemp(prefix="kernel-log-reports-"))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    md_path = _report_dir / f"kernel-log-report-{stamp}.md"
    json_path = _report_dir / f"kernel-log-report-{stamp}.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json_text, encoding="utf-8")

    for suffix in (".md", ".json"):
        old = sorted(_report_dir.glob(f"kernel-log-report-*{suffix}"))[:-KEEP_REPORTS]
        for path in old:
            path.unlink(missing_ok=True)
    return str(md_path), str(json_path)


def cleanup_report_files() -> None:
    """Remove the temporary folder (used by tests)."""
    global _report_dir
    if _report_dir is not None:
        shutil.rmtree(_report_dir, ignore_errors=True)
        _report_dir = None
