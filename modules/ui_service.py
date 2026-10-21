"""
UI Service Module for Kernel Log Root Cause Analyzer.

Framework-agnostic helpers for the Gradio interface:
  - reading the log from an upload or pasted text, with input validation
  - running the analysis pipeline (modules/analyzer.py)
  - rendering the result as ONE block of HTML, styled by modules/ui_theme.py

All styling lives in CSS classes (prefix "kx-"); nothing here carries inline
colours. The whole report is returned as a single string so the page updates
in one step instead of redrawing a dozen separate components.

Keeps all business logic out of gradio_app.py so it can be tested without a browser.
"""

import html
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config import MAX_LOG_FILE_SIZE_BYTES, OLLAMA_MODEL, SAMPLE_LOGS_DIR
from modules.analyzer import PipelineExecutionResult, analyze_raw_content
from modules.answer_checks import locate_evidence
from modules.event_extractor import EventSummary
from modules.llm_analyzer import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from modules.log_loader import (
    LogFileEmptyError,
    LogFileNotFoundError,
    LogFileTooLargeError,
    load_log_file,
)
from modules.report_export import to_json, to_markdown
from modules.result_parser import MalformedJSONError, ValidationError

logger = logging.getLogger("kernel_log_analyzer.ui_service")

SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW")

# Most rows the timeline and the log viewer will draw, so the page stays light on long logs
MAX_TIMELINE_ROWS = 30
MAX_VIEWER_LINES = 400

# What every analysis returns: (report_html, cleaned_log_text, markdown_report, json_report)
Outcome = Tuple[str, str, str, str]


# ----------------------------------------------------------------------------
# Small pieces
# ----------------------------------------------------------------------------

def format_severity_badge(severity: str) -> str:
    """Severity chip: a coloured square plus the word, so colour is never the only signal."""
    sev = (severity or "UNKNOWN").upper()
    css = sev.lower() if sev in SEVERITIES else "unknown"
    return f'<span class="kx-chip kx-sev kx-sev-{css}">{html.escape(sev)}</span>'


def format_subsystem_badge(subsystem: str) -> str:
    """Affected subsystem chip."""
    return f'<span class="kx-chip kx-sub">{html.escape((subsystem or "other").upper())}</span>'


def format_confidence_badge(confidence: str) -> str:
    """The model's own confidence statement, shown as its words, not as a probability."""
    return f'<span class="kx-chip kx-conf"><span class="kx-k">confidence</span>{html.escape(confidence or "n/a")}</span>'


def format_evidence_list(
    evidence_items: List[str],
    unverified: Optional[List[str]] = None,
    lines: Optional[Dict[str, int]] = None,
) -> str:
    """
    Quoted log lines, shown like a log viewer with a line gutter.

    Lines in `unverified` could not be found in the log and are marked in amber.
    `lines` maps a quote to its line number in the log viewer; those rows link to it.
    """
    if not evidence_items:
        return '<p class="kx-empty">No direct log line citations extracted.</p>'
    not_found = set(unverified or [])
    where = lines or {}
    rows = []
    for number, item in enumerate(evidence_items, start=1):
        missing = item in not_found
        note = (
            '<span class="kx-note">Not found in the log. The model may have misquoted or invented this line.</span>'
        ) if missing else ""
        line_no = where.get(item)
        jump = f'<a class="kx-jump" href="#kx-L{line_no}" title="Show this line in the log">line {line_no}</a>' if line_no else ""
        rows.append(
            f'<div class="kx-row{" kx-warn" if missing else ""}">'
            f'<span class="kx-ln">{number}</span>'
            f'<span class="kx-body"><code>{html.escape(item.strip())}</code>{note}</span>{jump}</div>'
        )
    return f'<div class="kx-log">{"".join(rows)}</div>'


def format_commands_list(commands: List[str], warnings: Optional[Dict[str, str]] = None) -> str:
    """
    Suggested commands. Click one to select it for copying.

    Commands in `warnings` could change the system, or are not recognised as
    read-only, and are marked with the reason.
    """
    if not commands:
        return '<p class="kx-empty">No specific diagnostic commands suggested.</p>'
    flagged = warnings or {}
    rows = []
    for cmd in commands:
        reason = flagged.get(cmd)
        note = f'<span class="kx-note">Review before running: {html.escape(reason)}</span>' if reason else ""
        rows.append(
            f'<div class="kx-row{" kx-warn" if reason else ""}">'
            f'<span class="kx-prompt">$</span>'
            f'<span class="kx-body"><code>{html.escape(cmd.strip())}</code>{note}</span></div>'
        )
    return f'<div class="kx-cmds">{"".join(rows)}</div>'


def format_fixes_list(items: List[str]) -> str:
    """Possible fixes as a numbered list."""
    if not items:
        return '<p class="kx-empty">No remediation steps suggested.</p>'
    return '<ol class="kx-fixes">' + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ol>"


def format_error_html(title: str, message: str, suggestion: Optional[str] = None) -> str:
    """An error card with what went wrong and, when known, what to do about it."""
    hint = f'<p class="kx-hint">{html.escape(suggestion)}</p>' if suggestion else ""
    return (
        f'<div class="kx-error" role="alert"><div class="kx-error-title">{html.escape(title)}</div>'
        f'<p>{html.escape(message)}</p>{hint}</div>'
    )


def _section(title: str, body: str) -> str:
    return f'<section><div class="kx-label">{html.escape(title)}</div>{body}</section>'


def _text_block(text: str, empty: str = "None given.") -> str:
    text = (text or "").strip()
    return f'<p class="kx-text">{html.escape(text)}</p>' if text else f'<p class="kx-empty">{empty}</p>'


def _when(seconds: Optional[float]) -> str:
    return f"{seconds:.2f} s" if seconds is not None else ""


def render_timeline(summary: Optional[EventSummary], viewable_lines: int = MAX_VIEWER_LINES) -> str:
    """What happened, in order: one row per burst of events, each linking to its line in the log viewer."""
    if summary is None or not summary.groups:
        return '<p class="kx-empty">No warnings or errors were found to build a timeline from.</p>'
    counts = " &middot; ".join(f"{html.escape(name)} {count}" for name, count in summary.subsystem_counts.items())
    facts = [f"{summary.total_events} events", counts]
    if summary.first_event_time is not None:
        facts.append(f"first at {summary.first_event_time:g} s")
    if summary.span_seconds:
        facts.append(f"spans {summary.span_seconds:g} s")
    rows = []
    for group in summary.groups[:MAX_TIMELINE_ROWS]:
        times = _when(group.first_time)
        repeat = f'<span class="kx-n">&times;{group.count}</span>' if group.count > 1 else '<span class="kx-n"></span>'
        inner = (
            f'<span class="kx-t">{times or f"line {group.first_line}"}</span>'
            f'<span class="kx-sys">{html.escape(group.subsystem)}</span>'
            f'<span class="kx-et">{html.escape(group.text)}</span>{repeat}'
        )
        if group.first_line <= viewable_lines:
            rows.append(f'<a class="kx-ev" href="#kx-L{group.first_line}">{inner}</a>')
        else:
            rows.append(f'<div class="kx-ev">{inner}</div>')
    more = len(summary.groups) - MAX_TIMELINE_ROWS
    tail = f'<p class="kx-empty">and {more} more entries</p>' if more > 0 else ""
    return f'<p class="kx-sum">{" &middot; ".join(facts)}</p><div class="kx-tl">{"".join(rows)}</div>{tail}'


def render_log_viewer(log_text: str, hit_lines: Optional[Set[int]] = None) -> str:
    """
    The log the model saw, numbered, with the lines it quoted highlighted.

    One element per line (the number comes from CSS), so even a long log stays light.
    """
    hits = hit_lines or set()
    all_lines = log_text.splitlines()
    rows = []
    for number, line in enumerate(all_lines[:MAX_VIEWER_LINES], start=1):
        classes = ["kx-vl"]
        if number in hits:
            classes.append("kx-hit")
        if line.lstrip().startswith("[..."):
            classes.append("kx-gap")
        ident = f' id="kx-L{number}"'
        rows.append(f'<span class="{" ".join(classes)}"{ident} data-n="{number}">{html.escape(line)}\n</span>')
    cut = len(all_lines) - MAX_VIEWER_LINES
    tail = f'<p class="kx-empty">and {cut} more lines (use the Log sent to the model box)</p>' if cut > 0 else ""
    return f'<pre class="kx-view" tabindex="0">{"".join(rows)}</pre>{tail}'


# ----------------------------------------------------------------------------
# Whole-page states
# ----------------------------------------------------------------------------

def render_header(model: str = OLLAMA_MODEL, ollama_online: Optional[bool] = None) -> str:
    """Top bar: name on the left; model, locality and the Ollama status on the right."""
    if ollama_online is None:
        status = '<span><span class="kx-dot"></span>checking ollama</span>'
    elif ollama_online:
        status = '<span><span class="kx-dot kx-ok"></span>ollama online</span>'
    else:
        status = '<span><span class="kx-dot kx-bad"></span>ollama not reachable</span>'
    return (
        '<header class="kx-top">'
        '<div class="kx-brand"><span class="kx-mark">kernel<b>/</b>log analyzer</span>'
        '<span class="kx-tag">probable root cause from dmesg and journalctl, on this machine</span></div>'
        f'<div class="kx-env"><span>{html.escape(model)}</span><span>local only</span>{status}</div>'
        '</header>'
    )


def render_empty_state() -> str:
    """What the report pane shows before anything has been analysed."""
    return (
        '<div class="kx-start">'
        '<p class="kx-lead">Paste or upload a kernel log, or load a sample incident, then press Analyse.</p>'
        '<p class="kx-flow">clean log &rarr; build prompt &rarr; local model &rarr; checked answer</p>'
        '<p>You get the probable root cause, the log lines that support it, and what to check next. '
        'Each quoted line is checked against your log, and suggested commands are only shown, never run.</p>'
        '<p>The model runs on this computer. Nothing is sent anywhere.</p>'
        '</div>'
    )


def render_analysing(source_label: str = "") -> str:
    """Shown immediately while the local model works, so the page never looks frozen."""
    source = f' <span class="kx-meta">{html.escape(source_label)}</span>' if source_label else ""
    return (
        '<div class="kx-busy" role="status" aria-live="polite">'
        f'<p class="kx-lead">Analysing{source}</p>'
        '<p class="kx-sub">The local model usually needs about a minute. Longer logs take longer.</p>'
        '<div class="kx-bar"></div></div>'
    )


def render_live_status() -> str:
    """Live-capture tab: what this computer can capture."""
    from modules.live_collector import get_collector_capabilities

    caps = get_collector_capabilities()
    if caps["is_linux"]:
        dmesg = "found" if caps["has_dmesg"] else "not found"
        journalctl = "found" if caps["has_journalctl"] else "not found"
        return (
            f'<div class="kx-live kx-on"><b>Linux host ({html.escape(caps["platform"])})</b><br>'
            f'Live capture is available. <code>dmesg</code>: {dmesg}. <code>journalctl</code>: {journalctl}.</div>'
        )
    return (
        f'<div class="kx-live"><b>{html.escape(caps["platform"])} host</b><br>'
        'Live capture needs a Linux kernel. Run this app inside a Linux machine to use it, '
        'or use the Paste and Upload tabs here.</div>'
    )


# Kept under its original name for callers and tests
get_live_host_status_html = render_live_status


def _warning_banner(result: PipelineExecutionResult) -> str:
    parts = []
    if result.context_truncated:
        parts.append(
            f"This log was too long for the local model, so only {result.sent_lines:,} of "
            f"{result.original_cleaned_lines:,} lines were sent ({result.omitted_lines:,} routine lines left out; "
            "the earliest and latest warnings were kept with their context)."
        )
    if result.ollama_truncated:
        parts.append("Ollama also cut the prompt, so this diagnosis may be unreliable. Try a shorter log.")
    if not parts:
        return ""
    return f'<div class="kx-warnbox"><b>Partial log analysed.</b> {html.escape(" ".join(parts))}</div>'


def _metrics(result: PipelineExecutionResult, live: bool) -> str:
    s, t = result.preprocessed_stats, result.timing_metrics
    rows = [("source", result.log_file_path or "")]
    if live:
        rows.append(("capture time", f'{t.get("collect_ms", 0) / 1000:.2f} s'))
    rows += [
        ("raw lines", f"{s.original_line_count:,}"),
        ("cleaned lines", f"{s.cleaned_line_count:,}"),
        ("sent to model", f"{result.sent_lines or s.cleaned_line_count:,}"),
        ("repeats collapsed", f"{s.duplicate_lines_collapsed:,}"),
        ("model", result.model_name),
        ("prompt tokens", f'{t.get("ollama_prompt_tokens", 0):.0f}'),
        ("answer tokens", f'{t.get("ollama_eval_count", 0):.0f}'),
        ("model time", f'{t.get("llm_inference_ms", 0) / 1000:.1f} s'),
        ("total time", f'{t.get("total_pipeline_ms", 0) / 1000:.1f} s'),
    ]
    body = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>" for k, v in rows)
    return f'<details class="kx-details"><summary>Pipeline details</summary><dl class="kx-metrics">{body}</dl></details>'


def render_report(result: PipelineExecutionResult, source_label: str, live: bool = False) -> str:
    """The whole diagnostic report as one HTML string."""
    a = result.analysis
    seconds = result.timing_metrics.get("total_pipeline_ms", 0) / 1000
    head = (
        f'<section><div class="kx-label">Probable root cause</div>'
        f'<h2 class="kx-cause">{html.escape(a.root_cause)}</h2>'
        f'<div class="kx-chips">{format_subsystem_badge(a.subsystem)}{format_severity_badge(a.severity)}'
        f'{format_confidence_badge(a.confidence)}</div>'
        f'<div class="kx-meta">{html.escape(source_label)} &middot; {html.escape(result.model_name)} &middot; {seconds:.0f} s</div>'
        f'</section>'
    )
    log_text = result.cleaned_log_excerpt
    where = locate_evidence(a.evidence, log_text)
    timeline_note = (
        '<p class="kx-empty">The timeline covers the lines that were sent to the model.</p>'
        if result.context_truncated else ""
    )
    return (
        '<div class="kx-report">'
        + _warning_banner(result)
        + head
        + _section("What happened, in order", render_timeline(result.timeline) + timeline_note)
        + _section("Evidence from the log", format_evidence_list(a.evidence, a.unverified_evidence, where))
        + _section("Explanation", _text_block(a.explanation))
        + _section("What the log does not show", _text_block(a.uncertainty))
        + _section("Commands to investigate", format_commands_list(a.commands, a.command_warnings))
        + _section("Possible fixes", format_fixes_list(a.possible_fixes))
        + _section("The log the model saw", render_log_viewer(log_text, set(where.values())))
        + _metrics(result, live)
        + "</div>"
    )


# ----------------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------------

def extract_input_content(
    uploaded_file: Any,
    pasted_text: Optional[str],
) -> Tuple[str, str]:
    """
    Extract and validate raw log text from either an uploaded file or pasted string.

    Returns:
        Tuple[str, str]: (raw_content, source_label)

    Raises:
        ValueError: If neither input is provided.
    """
    pasted = (pasted_text or "").strip()
    has_file = uploaded_file is not None
    has_text = len(pasted) > 0

    if not has_file and not has_text:
        raise ValueError("Please provide a kernel log by either uploading a file or pasting log text.")

    if has_file and has_text:
        # User supplied both: prefer the uploaded file if present, but log informational notice
        logger.info("Both file upload and pasted text provided; prioritizing uploaded file.")

    if has_file:
        # Gradio provides file as a temp file path string or an object with .name
        file_path_str = uploaded_file if isinstance(uploaded_file, (str, Path)) else getattr(uploaded_file, "name", str(uploaded_file))
        path_obj = Path(file_path_str)
        content = load_log_file(path_obj)
        return content, f"Uploaded File ({path_obj.name})"

    return pasted, "Pasted Log Text"


def read_input(uploaded_file: Any, pasted_text: Optional[str]) -> Tuple[str, str, str]:
    """
    Read the log, turning any problem into a ready-to-show error card.

    Returns (raw_content, source_label, error_html). error_html is "" when reading worked.
    """
    try:
        raw, label = extract_input_content(uploaded_file, pasted_text)
        return raw, label, ""
    except ValueError as err:
        return "", "", format_error_html("No input provided", str(err), "Paste the log text, or drop a .log or .txt file.")
    except LogFileNotFoundError as err:
        return "", "", format_error_html("File not found", str(err), "Check that the file still exists.")
    except LogFileEmptyError as err:
        return "", "", format_error_html("The log is empty", str(err), "Provide a kernel log with dmesg or journalctl text in it.")
    except LogFileTooLargeError as err:
        max_mb = MAX_LOG_FILE_SIZE_BYTES // (1024 * 1024)
        return "", "", format_error_html("File is too large", str(err), f"Keep logs under {max_mb} MB.")
    except Exception as exc:  # noqa: BLE001 - show anything unexpected as a card, not a stack trace
        return "", "", format_error_html("Could not read the input", f"Unable to read input: {exc}")


def _pipeline_error_card(exc: Exception, model: Optional[str]) -> str:
    """One error card for every way the model step can fail."""
    if isinstance(exc, OllamaConnectionError):
        return format_error_html("Ollama is not reachable", str(exc),
                                 "Start Ollama (open the app, or run `ollama serve` in a terminal) and try again.")
    if isinstance(exc, OllamaModelNotFoundError):
        return format_error_html("Model is not installed", str(exc), f"Install it with: ollama pull {model or OLLAMA_MODEL}")
    if isinstance(exc, OllamaTimeoutError):
        return format_error_html("The model took too long", str(exc),
                                 "Free up some memory, try a shorter log, or raise OLLAMA_TIMEOUT_SECONDS.")
    if isinstance(exc, OllamaResponseError):
        return format_error_html("Ollama returned an error", str(exc), "Try again. If it keeps happening, restart Ollama.")
    if isinstance(exc, (MalformedJSONError, ValidationError)):
        return format_error_html("The model's answer was unusable", f"The answer did not match the expected format: {exc}",
                                 "Run the analysis again.")
    logger.exception("Unexpected error during analysis")
    return format_error_html("Analysis failed", f"An unexpected error occurred: {exc}", "Check the terminal for details.")


# ----------------------------------------------------------------------------
# Controllers: each returns (report_html, cleaned_log_text, markdown_report, json_report)
# ----------------------------------------------------------------------------

def _outcome(result: PipelineExecutionResult, source_label: str, live: bool = False) -> Outcome:
    """The page, the cleaned log and the two downloadable reports for one finished analysis."""
    return (
        render_report(result, source_label, live=live),
        result.cleaned_log_excerpt,
        to_markdown(result, source_label),
        to_json(result, source_label),
    )


def analyze_prepared(
    raw_content: str,
    source_label: str,
    model_override: Optional[str] = None,
    ollama_client: Optional[OllamaClient] = None,
) -> Outcome:
    """Run the pipeline on log text that has already been read."""
    try:
        result = analyze_raw_content(
            raw_content=raw_content,
            source_label=source_label,
            model=model_override,
            ollama_client=ollama_client,
        )
    except Exception as exc:  # noqa: BLE001
        return _pipeline_error_card(exc, model_override), "", "", ""
    return _outcome(result, source_label)


def run_ui_analysis(
    uploaded_file: Any,
    pasted_text: Optional[str],
    model_override: Optional[str] = None,
    ollama_client: Optional[OllamaClient] = None,
) -> Outcome:
    """Upload or pasted text in; an Outcome (report_html, cleaned_log_text, markdown, json) out."""
    raw, label, error = read_input(uploaded_file, pasted_text)
    if error:
        return error, "", "", ""
    return analyze_prepared(raw, label, model_override, ollama_client)


def run_live_ui_analysis(
    collector_type: str = "auto",
    boot_offset: int = 0,
    lines: int = 1000,
    model_override: Optional[str] = None,
    ollama_client: Optional[OllamaClient] = None,
) -> Outcome:
    """Capture the live kernel log and analyse it; returns an Outcome (see the top of this module)."""
    from modules.analyzer import analyze_live
    from modules.live_collector import (
        CollectionExecutionError,
        CollectorNotFoundError,
        PermissionDeniedError,
        PlatformNotSupportedError,
    )

    try:
        result = analyze_live(
            collector_type=collector_type,
            boot_offset=boot_offset,
            lines=lines,
            model=model_override,
            ollama_client=ollama_client,
        )
    except PlatformNotSupportedError as err:
        return format_error_html("Live capture is not available here", str(err),
                                 "Run the app on a Linux machine, or use stored logs in the Upload and Paste tabs."), "", "", ""
    except PermissionDeniedError as err:
        return format_error_html("Permission denied reading the kernel log", str(err),
                                 "Run with sudo, grant CAP_SYSLOG, or relax kernel.dmesg_restrict."), "", "", ""
    except CollectorNotFoundError as err:
        return format_error_html("Collector not found", str(err),
                                 "Install util-linux (for dmesg) or systemd (for journalctl)."), "", "", ""
    except CollectionExecutionError as err:
        return format_error_html("Live capture failed", str(err), "Check that the kernel ring buffer has content."), "", "", ""
    except Exception as exc:  # noqa: BLE001
        return _pipeline_error_card(exc, model_override), "", "", ""

    return _outcome(result, result.log_file_path or "Live Linux capture", live=True)


def load_sample_log_content(category: str, filename: str) -> str:
    """Load text content of a bundled sample log for quick testing."""
    target = SAMPLE_LOGS_DIR / category / filename
    if not target.exists():
        raise FileNotFoundError(f"Sample log not found: {target}")
    return target.read_text(encoding="utf-8", errors="replace")
