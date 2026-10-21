"""
UI Service Module for Kernel Log Root Cause Analyzer (Phase 3A).

Provides pure, framework-agnostic helper functions for the Gradio UI:
- Ingestion of uploaded files and pasted text
- Input validation and size checks
- Execution of the common analysis pipeline (modules/analyzer.py)
- Formatting of structured diagnosis into rich, accessible HTML/Markdown components
- Safe error handling with human-readable error messages

Keeps all business logic out of gradio_app.py for 100% automated testability.
"""

import html
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from config import (
    MAX_LOG_FILE_SIZE_BYTES,
    OLLAMA_MODEL,
    SAMPLE_LOGS_DIR,
)
from modules.analyzer import PipelineExecutionResult, analyze_log, analyze_raw_content
from modules.llm_analyzer import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from modules.log_loader import (
    LogFileAccessError,
    LogFileEmptyError,
    LogFileNotFoundError,
    LogFileTooLargeError,
    LogLoaderError,
    LogNotAFileError,
    load_log_file,
)
from modules.result_parser import MalformedJSONError, ResultParserError, ValidationError

logger = logging.getLogger("kernel_log_analyzer.ui_service")


SEVERITY_STYLES = {
    "CRITICAL": {
        "bg": "#fee2e2",
        "text": "#991b1b",
        "border": "#f87171",
        "icon": "🔴",
    },
    "HIGH": {
        "bg": "#ffedd5",
        "text": "#9a3412",
        "border": "#fb923c",
        "icon": "🟠",
    },
    "MEDIUM": {
        "bg": "#fef9c3",
        "text": "#854d0e",
        "border": "#facc15",
        "icon": "🟡",
    },
    "LOW": {
        "bg": "#ecfdf5",
        "text": "#065f46",
        "border": "#34d399",
        "icon": "🟢",
    },
    "UNKNOWN": {
        "bg": "#f3f4f6",
        "text": "#374151",
        "border": "#9ca3af",
        "icon": "⚪",
    },
}


def format_severity_badge(severity: str) -> str:
    """Format severity into an accessible visual badge."""
    sev_upper = severity.upper()
    style = SEVERITY_STYLES.get(sev_upper, SEVERITY_STYLES["UNKNOWN"])
    return (
        f'<span style="display:inline-flex; align-items:center; gap:6px; '
        f'background-color:{style["bg"]}; color:{style["text"]}; '
        f'border:1px solid {style["border"]}; border-radius:9999px; '
        f'padding:4px 14px; font-weight:700; font-size:14px; letter-spacing:0.05em;">'
        f'{style["icon"]} {sev_upper}</span>'
    )


def format_subsystem_badge(subsystem: str) -> str:
    """Format affected subsystem name into a styled badge."""
    sub_upper = subsystem.upper()
    return (
        f'<span style="display:inline-flex; align-items:center; '
        f'background-color:#eff6ff; color:#1e40af; border:1px solid #93c5fd; '
        f'border-radius:9999px; padding:4px 14px; font-weight:600; font-size:14px;">'
        f'⚙️ {sub_upper}</span>'
    )


def format_confidence_badge(confidence: str) -> str:
    """Format diagnostic confidence with clear distinction from probability."""
    conf_safe = html.escape(confidence)
    return (
        f'<span style="display:inline-flex; align-items:center; '
        f'background-color:#f5f3ff; color:#5b21b6; border:1px solid #c4b5fd; '
        f'border-radius:9999px; padding:4px 14px; font-weight:600; font-size:14px;">'
        f'🔍 Confidence: {conf_safe}</span>'
    )


def format_markdown_list(items: List[str], empty_msg: str = "*(None reported)*") -> str:
    """Format list of strings into clean markdown bullet points."""
    if not items:
        return empty_msg
    return "\n".join(f"* `{html.escape(item)}`" if item.startswith(("-", "ls", "dm", "fr", "to", "ps", "jo", "vm", "sm", "ip")) else f"* {html.escape(item)}" for item in items)


def format_evidence_list(evidence_items: List[str]) -> str:
    """Format factual log evidence lines with distinct terminal-style blockquote."""
    if not evidence_items:
        return "*(No direct log line citations extracted)*"
    blocks = []
    for item in evidence_items:
        clean_line = html.escape(item.strip())
        blocks.append(
            f'<div style="background-color:#1e293b; color:#f8fafc; font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,monospace; '
            f'font-size:13px; padding:8px 12px; border-radius:6px; margin-bottom:6px; overflow-x:auto; white-space:pre-wrap; border-left:4px solid #38bdf8;">'
            f'{clean_line}</div>'
        )
    return "\n".join(blocks)


def format_commands_list(commands: List[str]) -> str:
    """Format troubleshooting commands into copyable code blocks."""
    if not commands:
        return "*(No specific diagnostic commands suggested)*"
    blocks = []
    for cmd in commands:
        clean_cmd = html.escape(cmd.strip())
        blocks.append(
            f'<div style="background-color:#0f172a; color:#38bdf8; font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,monospace; '
            f'font-size:13px; padding:8px 12px; border-radius:6px; margin-bottom:6px; display:flex; align-items:center; gap:8px;">'
            f'<span style="color:#64748b;">$</span> <code>{clean_cmd}</code></div>'
        )
    return "\n".join(blocks)


def format_error_html(title: str, message: str, suggestion: Optional[str] = None) -> str:
    """Generate a clean, professional error notification card."""
    title_safe = html.escape(title)
    msg_safe = html.escape(message)
    sugg_html = f'<div style="margin-top:8px; font-size:13px; color:#475569;">💡 <strong>Suggestion:</strong> {html.escape(suggestion)}</div>' if suggestion else ""

    return (
        f'<div style="background-color:#fef2f2; border:1px solid #fecaca; border-left:5px solid #ef4444; '
        f'border-radius:8px; padding:16px; margin:12px 0;">'
        f'<div style="font-weight:700; color:#991b1b; font-size:15px; display:flex; align-items:center; gap:8px;">'
        f'⚠️ {title_safe}</div>'
        f'<div style="color:#7f1d1d; font-size:14px; margin-top:6px; line-height:1.5;">{msg_safe}</div>'
        f'{sugg_html}'
        f'</div>'
    )


def extract_input_content(
    uploaded_file: Any,
    pasted_text: Optional[str],
) -> Tuple[str, str]:
    """
    Extract and validate raw log text from either an uploaded file or pasted string.

    Returns:
        Tuple[str, str]: (raw_content, source_label)

    Raises:
        ValueError: If neither input is provided, or both are provided ambiguously.
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


def run_ui_analysis(
    uploaded_file: Any,
    pasted_text: Optional[str],
    model_override: Optional[str] = None,
    ollama_client: Optional[OllamaClient] = None,
) -> Tuple[str, str, str, str, str, str, str, str, str, str, str]:
    """
    UI Controller entrypoint: Coordinates input extraction, pipeline analysis,
    and returns UI component outputs.

    Returns an 11-tuple matching the Gradio output components:
      1. status_html
      2. root_cause_text
      3. subsystem_badge_html
      4. severity_badge_html
      5. confidence_badge_html
      6. evidence_html
      7. explanation_text
      8. commands_html
      9. fixes_markdown
      10. uncertainty_text
      11. metrics_markdown
    """
    empty_result = ("", "", "", "", "", "", "", "", "", "", "")

    # 1. Extract log content
    try:
        raw_content, source_label = extract_input_content(uploaded_file, pasted_text)
    except ValueError as err:
        return (
            format_error_html("No Input Provided", str(err), "Drag and drop a .log file or paste text into the box below."),
            *empty_result[1:],
        )
    except LogFileNotFoundError as err:
        return (
            format_error_html("File Not Found", str(err), "Ensure the selected file exists on disk."),
            *empty_result[1:],
        )
    except LogFileEmptyError as err:
        return (
            format_error_html("Log File Is Empty", str(err), "Provide a kernel log containing valid dmesg or journalctl text."),
            *empty_result[1:],
        )
    except LogFileTooLargeError as err:
        max_mb = MAX_LOG_FILE_SIZE_BYTES // (1024 * 1024)
        return (
            format_error_html("File Exceeds Size Limit", str(err), f"Keep uploaded kernel logs below {max_mb} MB."),
            *empty_result[1:],
        )
    except Exception as exc:
        return (
            format_error_html("Input Extraction Error", f"Unable to read input: {exc}"),
            *empty_result[1:],
        )

    # 2. Run analysis pipeline
    try:
        result = analyze_raw_content(
            raw_content=raw_content,
            source_label=source_label,
            model=model_override,
            ollama_client=ollama_client,
        )
    except OllamaConnectionError as err:
        return (
            format_error_html(
                "Ollama Daemon Unreachable",
                str(err),
                "Make sure Ollama is running locally. Open a terminal and run: `ollama serve`",
            ),
            *empty_result[1:],
        )
    except OllamaModelNotFoundError as err:
        model_name = model_override or OLLAMA_MODEL
        return (
            format_error_html(
                "Model Not Installed",
                str(err),
                f"Pull the required model in your terminal using: `ollama pull {model_name}`",
            ),
            *empty_result[1:],
        )
    except OllamaTimeoutError as err:
        return (
            format_error_html(
                "Inference Timeout",
                str(err),
                "The local model took too long to respond. Ensure your machine has sufficient free RAM.",
            ),
            *empty_result[1:],
        )
    except (MalformedJSONError, ValidationError) as err:
        return (
            format_error_html(
                "Diagnostic Validation Error",
                f"The local LLM response failed schema validation: {err}",
                "Try running the analysis again or verify that the model has not been corrupted.",
            ),
            *empty_result[1:],
        )
    except Exception as exc:
        logger.exception("Unexpected error during UI analysis")
        return (
            format_error_html(
                "Unexpected Processing Failure",
                f"An unexpected internal error occurred: {exc}",
                "Check application terminal logs for technical traceback details.",
            ),
            *empty_result[1:],
        )

    # 3. Format structured output for UI
    a = result.analysis
    t = result.timing_metrics
    s = result.preprocessed_stats

    status_banner = (
        f'<div style="background-color:#f0fdf4; border:1px solid #bbf7d0; border-left:5px solid #22c55e; '
        f'border-radius:8px; padding:12px 16px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">'
        f'<div><span style="font-weight:700; color:#166534;">✅ Analysis Complete</span> — Source: <code>{html.escape(source_label)}</code></div>'
        f'<div style="font-size:13px; color:#15803d;">Model: <strong>{html.escape(result.model_name)}</strong> | '
        f'Total time: <strong>{t.get("total_pipeline_ms", 0):.0f} ms</strong> (LLM: {t.get("llm_inference_ms", 0):.0f} ms)</div>'
        f'</div>'
    )

    if result.context_truncated or result.ollama_truncated:
        parts = []
        if result.context_truncated:
            parts.append(
                f"This log was too long for the local model, so only {result.sent_lines:,} of "
                f"{result.original_cleaned_lines:,} lines were sent ({result.omitted_lines:,} routine lines left out, "
                f"keeping the earliest and latest warnings with their context)."
            )
        if result.ollama_truncated:
            parts.append("Ollama also cut the prompt, so this diagnosis may be unreliable. Try a shorter log.")
        status_banner += (
            '<div style="background-color:#fffbeb; border:1px solid #fde68a; border-left:5px solid #f59e0b; '
            'border-radius:8px; padding:12px 16px; margin-bottom:12px; color:#92400e; font-size:14px;">'
            f'⚠️ <strong>Partial log analysed.</strong> {html.escape(" ".join(parts))}</div>'
        )

    root_cause_display = a.root_cause
    subsystem_badge = format_subsystem_badge(a.subsystem)
    severity_badge = format_severity_badge(a.severity)
    confidence_badge = format_confidence_badge(a.confidence)

    evidence_display = format_evidence_list(a.evidence)
    explanation_display = a.explanation
    commands_display = format_commands_list(a.commands)
    fixes_display = format_markdown_list(a.possible_fixes, empty_msg="*(No remediation steps suggested)*")
    uncertainty_display = a.uncertainty

    metrics_table = (
        f"| Metric | Value |\n"
        f"| :--- | :--- |\n"
        f"| **Source** | `{source_label}` |\n"
        f"| **Raw Input Lines** | {s.original_line_count:,} |\n"
        f"| **Cleaned Lines** | {s.cleaned_line_count:,} |\n"
        f"| **Lines Sent To Model** | {result.sent_lines or s.cleaned_line_count:,} |\n"
        f"| **Noise Reduction** | {s.reduction_percentage:.1f}% |\n"
        f"| **Candidate Markers Flagged** | {s.candidate_evidence_count} |\n"
        f"| **Local Model Used** | `{result.model_name}` |\n"
        f"| **LLM Eval Tokens** | {t.get('ollama_eval_count', 0):.0f} tokens |\n"
        f"| **Preprocessing Time** | {t.get('preprocess_ms', 0):.2f} ms |\n"
        f"| **LLM Inference Time** | {t.get('llm_inference_ms', 0):.2f} ms |\n"
        f"| **Total Pipeline Time** | {t.get('total_pipeline_ms', 0):.2f} ms |\n"
    )

    return (
        status_banner,
        root_cause_display,
        subsystem_badge,
        severity_badge,
        confidence_badge,
        evidence_display,
        explanation_display,
        commands_display,
        fixes_display,
        uncertainty_display,
        metrics_table,
    )


def load_sample_log_content(category: str, filename: str) -> str:
    """Load text content of a pre-collected sample log from disk for quick UI testing."""
    target = SAMPLE_LOGS_DIR / category / filename
    if not target.exists():
        raise FileNotFoundError(f"Sample log not found: {target}")
    return target.read_text(encoding="utf-8", errors="replace")


def get_live_host_status_html() -> str:
    """
    Generate an HTML card summarizing host platform and kernel collection capabilities.
    """
    from modules.live_collector import get_collector_capabilities

    caps = get_collector_capabilities()
    os_name = caps["platform"]

    if caps["is_linux"]:
        dmesg_status = "✅ Available" if caps["has_dmesg"] else "❌ Not found"
        journalctl_status = "✅ Available" if caps["has_journalctl"] else "❌ Not found"
        return (
            f'<div style="background-color:#f0fdf4; border:1px solid #86efac; border-radius:8px; padding:12px 16px; margin:8px 0;">'
            f'<div style="font-weight:700; color:#166534; display:flex; align-items:center; gap:8px;">'
            f'🐧 Linux Host Environment ({os_name})</div>'
            f'<div style="font-size:13px; color:#15803d; margin-top:4px;">'
            f'Live collection is active. <code>dmesg</code>: {dmesg_status} | <code>journalctl</code>: {journalctl_status}</div>'
            f'</div>'
        )

    return (
        f'<div style="background-color:#f8fafc; border:1px solid #cbd5e1; border-left:4px solid #0284c7; border-radius:8px; padding:12px 16px; margin:8px 0;">'
        f'<div style="font-weight:700; color:#0f172a; display:flex; align-items:center; gap:8px;">'
        f'🍎 {os_name} Development Host Detected</div>'
        f'<div style="font-size:13px; color:#475569; margin-top:4px; line-height:1.4;">'
        f'Native Linux kernel collection requires Linux. On macOS, live collection runs inside your Ubuntu VM. '
        f'For macOS testing, use the <strong>Upload Log File</strong> or <strong>Paste Log Text</strong> tabs.</div>'
        f'</div>'
    )


def run_live_ui_analysis(
    collector_type: str = "auto",
    boot_offset: int = 0,
    lines: int = 1000,
    model_override: Optional[str] = None,
    ollama_client: Optional[OllamaClient] = None,
) -> Tuple[str, str, str, str, str, str, str, str, str, str, str, str]:
    """
    UI Controller entrypoint for live Linux kernel capture.

    Returns a 12-tuple matching the Gradio output components:
      1. status_html
      2. root_cause_text
      3. subsystem_badge_html
      4. severity_badge_html
      5. confidence_badge_html
      6. evidence_html
      7. explanation_text
      8. commands_html
      9. fixes_markdown
      10. uncertainty_text
      11. metrics_markdown
      12. cleaned_log_excerpt
    """
    empty_result = ("", "", "", "", "", "", "", "", "", "", "", "")

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
        return (
            format_error_html(
                "Host Platform Incompatible",
                str(err),
                "Run this command inside your Ubuntu VM, or use stored sample logs in the Upload/Paste tabs.",
            ),
            *empty_result[1:],
        )
    except PermissionDeniedError as err:
        return (
            format_error_html(
                "Kernel Log Read Permission Denied",
                str(err),
                "Run with sudo or grant CAP_SYSLOG / disable kernel.dmesg_restrict.",
            ),
            *empty_result[1:],
        )
    except CollectorNotFoundError as err:
        return (
            format_error_html(
                "Collector Binary Not Found",
                str(err),
                "Install util-linux (for dmesg) or systemd (for journalctl).",
            ),
            *empty_result[1:],
        )
    except CollectionExecutionError as err:
        return (
            format_error_html(
                "Live Collection Failed",
                str(err),
                "Verify the kernel ring buffer is populated.",
            ),
            *empty_result[1:],
        )
    except OllamaConnectionError as err:
        return (
            format_error_html(
                "Ollama Daemon Unreachable",
                str(err),
                "Make sure Ollama is running locally: `ollama serve`",
            ),
            *empty_result[1:],
        )
    except Exception as exc:
        logger.exception("Unexpected error during live UI analysis")
        return (
            format_error_html(
                "Live Analysis Failed",
                f"An unexpected error occurred: {exc}",
            ),
            *empty_result[1:],
        )

    a = result.analysis
    t = result.timing_metrics
    s = result.preprocessed_stats

    status_banner = (
        f'<div style="background-color:#f0fdf4; border:1px solid #bbf7d0; border-left:5px solid #22c55e; '
        f'border-radius:8px; padding:12px 16px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">'
        f'<div><span style="font-weight:700; color:#166534;">✅ Live Capture Complete</span> — Source: <code>{html.escape(result.log_file_path or "Live Linux")}</code></div>'
        f'<div style="font-size:13px; color:#15803d;">Model: <strong>{html.escape(result.model_name)}</strong> | '
        f'Capture: <strong>{t.get("collect_ms", 0):.0f} ms</strong> | '
        f'Total time: <strong>{t.get("total_pipeline_ms", 0):.0f} ms</strong></div>'
        f'</div>'
    )

    return (
        status_banner,
        a.root_cause,
        format_subsystem_badge(a.subsystem),
        format_severity_badge(a.severity),
        format_confidence_badge(a.confidence),
        format_evidence_list(a.evidence),
        a.explanation,
        format_commands_list(a.commands),
        format_markdown_list(a.possible_fixes, empty_msg="*(No remediation steps suggested)*"),
        a.uncertainty,
        (
            f"| Metric | Value |\n"
            f"| :--- | :--- |\n"
            f"| **Source** | `{result.log_file_path}` |\n"
            f"| **Capture Time** | {t.get('collect_ms', 0):.2f} ms |\n"
            f"| **Raw Lines Captured** | {s.original_line_count:,} |\n"
            f"| **Cleaned Lines** | {s.cleaned_line_count:,} |\n"
        f"| **Lines Sent To Model** | {result.sent_lines or s.cleaned_line_count:,} |\n"
            f"| **Noise Reduction** | {s.reduction_percentage:.1f}% |\n"
            f"| **Candidate Markers Flagged** | {s.candidate_evidence_count} |\n"
            f"| **Local Model Used** | `{result.model_name}` |\n"
            f"| **LLM Eval Tokens** | {t.get('ollama_eval_count', 0):.0f} tokens |\n"
            f"| **LLM Inference Time** | {t.get('llm_inference_ms', 0):.2f} ms |\n"
            f"| **Total Pipeline Time** | {t.get('total_pipeline_ms', 0):.2f} ms |\n"
        ),
        result.cleaned_log_excerpt,
    )

