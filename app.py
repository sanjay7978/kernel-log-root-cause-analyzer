#!/usr/bin/env python3
"""
CLI Application for Kernel Log Root Cause Analyzer (Phase 1 & Phase 2).

Integrates:
  Stored Kernel Log -> Loader -> Preprocessor -> Prompt Builder -> Ollama (Qwen) -> Parser -> Diagnostic Report

Fails gracefully on errors without exposing raw Python tracebacks.
"""

import argparse
import json
import sys
from pathlib import Path

from config import DEFAULT_LIVE_LOG_LINES, OLLAMA_MODEL
from modules.analyzer import analyze_live, analyze_log
from modules.live_collector import (
    CollectionExecutionError,
    CollectorNotFoundError,
    PermissionDeniedError,
    PlatformNotSupportedError,
)
from modules.llm_analyzer import (
    OllamaConnectionError,
    OllamaError,
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
from modules.preprocessor import preprocess_kernel_log
from modules.result_parser import MalformedJSONError, ResultParserError, ValidationError


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="LLM-Driven Dynamic Kernel Log Root Cause Analyzer (Phase 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full Phase 2 Root Cause Analysis using local Qwen model:
  python app.py sample_logs/memory/real_oom_incident.log

  # Analyze using a specific model override:
  python app.py sample_logs/usb/real_usb_descriptor_error.log --model qwen3:8b

  # Phase 1 mode: Preprocess only (no LLM call):
  python app.py sample_logs/memory/real_oom_incident.log --preprocess-only

  # Output as raw structured JSON:
  python app.py sample_logs/memory/real_oom_incident.log --json
        """,
    )

    parser.add_argument(
        "log_path",
        nargs="?",
        default=None,
        help="Path to stored kernel log file to analyze (omit when using --ui).",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the interactive Gradio web application (Phase 3A).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Ollama model name override (default: '{OLLAMA_MODEL}').",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Run Phase 1 preprocessing and noise metrics only (bypass LLM).",
    )
    parser.add_argument(
        "--preview-lines",
        type=int,
        default=25,
        help="Lines of cleaned log to preview in preprocess-only mode (default: 25).",
    )
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Display candidate evidence lines detected by preprocessor.",
    )
    parser.add_argument(
        "--save-clean",
        type=str,
        default=None,
        help="Save cleaned log text to disk.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output analysis report as formatted JSON.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Capture and analyze live kernel logs directly from the host (auto-detects collector).",
    )
    parser.add_argument(
        "--live-dmesg",
        action="store_true",
        help="Capture and analyze live kernel logs using dmesg specifically.",
    )
    parser.add_argument(
        "--live-journalctl",
        action="store_true",
        help="Capture and analyze live kernel logs using journalctl -k specifically.",
    )
    parser.add_argument(
        "--boot-offset",
        type=int,
        default=0,
        help="Boot offset for journalctl (default 0 for current boot, -1 for previous boot).",
    )
    parser.add_argument(
        "--live-lines",
        type=int,
        default=DEFAULT_LIVE_LOG_LINES,
        help=f"Maximum lines to capture during live collection (default: {DEFAULT_LIVE_LOG_LINES}).",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Save complete analysis report (JSON format) to disk.",
    )

    return parser.parse_args()


def display_analysis_report(result) -> None:
    """Format and print the structured diagnostic report to stdout."""
    sep = "=" * 68
    sub_sep = "-" * 68

    a = result.analysis
    t = result.timing_metrics
    stats = result.preprocessed_stats

    print(sep)
    print(" KERNEL LOG ROOT CAUSE ANALYSIS REPORT")
    print(sep)
    print(f" Target File       : {result.log_file_path}")
    print(f" Local Model Used  : {result.model_name}")
    print(f" Pipeline Metrics  : {stats.original_line_count} raw lines -> {stats.cleaned_line_count} cleaned ({stats.reduction_percentage:.1f}% reduction)")
    print(f" Total Elapsed     : {t.get('total_pipeline_ms', 0):.0f} ms (Inference: {t.get('llm_inference_ms', 0):.0f} ms | {t.get('ollama_eval_count', 0):.0f} tokens)")
    if result.context_truncated:
        print(f" WARNING           : log too long for the model; sent {result.sent_lines} of {result.original_cleaned_lines} lines ({result.omitted_lines} omitted)")
    if result.ollama_truncated:
        print(" WARNING           : Ollama cut the prompt itself; the diagnosis may be unreliable. Lower --live-lines or raise OLLAMA_NUM_CTX.")
    print(sep)

    print("\n[PROBABLE ROOT CAUSE]")
    print(f"  {a.root_cause}")

    print("\n[AFFECTED SUBSYSTEM]")
    print(f"  {a.subsystem.upper()}")

    print("\n[SEVERITY]")
    print(f"  {a.severity}")

    print("\n[LOG EVIDENCE]")
    if a.evidence:
        for idx, item in enumerate(a.evidence, start=1):
            flag = "  [NOT FOUND IN LOG]" if item in a.unverified_evidence else ""
            print(f"  {idx}. {item}{flag}")
    else:
        print("  (No direct log citations extracted)")

    print("\n[EXPLANATION]")
    print(f"  {a.explanation}")

    print("\n[RECOMMENDED TROUBLESHOOTING COMMANDS]")
    if a.commands:
        for idx, cmd in enumerate(a.commands, start=1):
            warning = a.command_warnings.get(cmd)
            print(f"  {idx}. {cmd}" + (f"  [REVIEW FIRST: {warning}]" if warning else ""))
    else:
        print("  (No specific commands recommended)")

    print("\n[POSSIBLE FIXES / REMEDIATION]")
    if a.possible_fixes:
        for idx, fix in enumerate(a.possible_fixes, start=1):
            print(f"  {idx}. {fix}")
    else:
        print("  (No fixes suggested)")

    print("\n[DIAGNOSTIC CONFIDENCE]")
    print(f"  {a.confidence}")

    print("\n[UNCERTAINTY & LIMITATIONS]")
    print(f"  {a.uncertainty}")

    print("\n" + sep)


def run_preprocess_only(args, log_path: Path) -> int:
    """Execute Phase 1 preprocessing mode."""
    try:
        raw_content = load_log_file(log_path)
    except LogLoaderError as err:
        print(f"\n[ERROR] Ingestion failure: {err}\n", file=sys.stderr)
        return 1

    preprocessed = preprocess_kernel_log(raw_content)
    stats = preprocessed.stats

    sep = "=" * 68
    print(sep)
    print(" KERNEL LOG PREPROCESSOR (PHASE 1 MODE)")
    print(sep)
    print(f" File                 : {log_path.resolve()}")
    print(f" Original Lines       : {stats.original_line_count}")
    print(f" Cleaned Lines        : {stats.cleaned_line_count}")
    print(f" Empty Lines Removed  : {stats.empty_lines_removed}")
    print(f" Duplicates Collapsed : {stats.duplicate_lines_collapsed}")
    print(f" Noise Reduction      : {stats.reduction_percentage:.2f}%")
    print(f" Candidate Markers    : {stats.candidate_evidence_count}")
    print(sep)

    lines = preprocessed.cleaned_text.splitlines()
    preview = lines[: args.preview_lines]
    print(f" PREVIEW (First {len(preview)} of {len(lines)} lines):")
    print("-" * 68)
    for line in preview:
        print(line)
    if len(lines) > args.preview_lines:
        print(f" ... [{len(lines) - args.preview_lines} lines omitted]")
    print(sep)

    if args.save_clean:
        dest = Path(args.save_clean).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(preprocessed.cleaned_text, encoding="utf-8")
        print(f"\n[SUCCESS] Cleaned log saved to: {dest}")

    return 0


def main() -> int:
    """Main CLI entrypoint."""
    args = parse_arguments()

    # Launch Gradio graphical interface if requested
    if args.ui:
        import gradio_app
        gradio_app.launch()
        return 0

    # Check if live capture mode is requested
    is_live = args.live or args.live_dmesg or args.live_journalctl

    if not is_live and not args.log_path:
        print(
            "\n[ERROR] No kernel log file provided. Specify a file path, "
            "use a live capture flag (--live, --live-dmesg, --live-journalctl), "
            "or use --ui to launch the web interface.\n",
            file=sys.stderr,
        )
        print("Usage: python app.py <path_to_log_file>   OR   python app.py --live   OR   python app.py --ui\n")
        return 1

    if is_live:
        collector = "dmesg" if args.live_dmesg else ("journalctl" if args.live_journalctl else "auto")
        try:
            print(f"\nCapturing live kernel log using '{collector}' (model: '{args.model or OLLAMA_MODEL}')...")
            result = analyze_live(
                collector_type=collector,
                boot_offset=args.boot_offset,
                lines=args.live_lines,
                model=args.model,
            )
        except PlatformNotSupportedError as err:
            print(f"\n[ERROR] Platform Incompatible:\n  {err}\n", file=sys.stderr)
            return 1
        except PermissionDeniedError as err:
            print(f"\n[ERROR] Permission Denied:\n  {err}\n  Hint: Run with sudo or grant CAP_SYSLOG.\n", file=sys.stderr)
            return 1
        except CollectorNotFoundError as err:
            print(f"\n[ERROR] Collector Tool Not Found:\n  {err}\n", file=sys.stderr)
            return 1
        except CollectionExecutionError as err:
            print(f"\n[ERROR] Live Collection Execution Error:\n  {err}\n", file=sys.stderr)
            return 1
        except OllamaConnectionError as err:
            print(f"\n[ERROR] Ollama Connection Failure:\n  {err}\n  Hint: Start Ollama by running 'ollama serve' in a terminal.\n", file=sys.stderr)
            return 1
        except OllamaModelNotFoundError as err:
            print(f"\n[ERROR] Model Not Available:\n  {err}\n", file=sys.stderr)
            return 1
        except OllamaTimeoutError as err:
            print(f"\n[ERROR] Inference Timeout:\n  {err}\n", file=sys.stderr)
            return 1
        except OllamaResponseError as err:
            print(f"\n[ERROR] Ollama Server Error:\n  {err}\n", file=sys.stderr)
            return 1
        except MalformedJSONError as err:
            print(f"\n[ERROR] Model output could not be parsed into JSON:\n  {err}\n", file=sys.stderr)
            return 1
        except ValidationError as err:
            print(f"\n[ERROR] Model output failed schema validation:\n  {err}\n", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"\n[UNEXPECTED ERROR] {exc}\n", file=sys.stderr)
            return 1
    else:
        log_path = Path(args.log_path)

        # 1. Phase 1 mode check
        if args.preprocess_only:
            return run_preprocess_only(args, log_path)

        # 2. Phase 2 Full Pipeline Execution
        try:
            print(f"\nAnalyzing '{log_path.name}' using local model '{args.model or OLLAMA_MODEL}' via Ollama...")
            result = analyze_log(file_path=log_path, model=args.model)

        except LogFileNotFoundError as err:
            print(f"\n[ERROR] File not found: {err}\n", file=sys.stderr)
            return 1
        except LogNotAFileError as err:
            print(f"\n[ERROR] Invalid path: {err}\n", file=sys.stderr)
            return 1
        except LogFileEmptyError as err:
            print(f"\n[ERROR] Log file is empty: {err}\n", file=sys.stderr)
            return 1
        except LogFileTooLargeError as err:
            print(f"\n[ERROR] File exceeds size limit: {err}\n", file=sys.stderr)
            return 1
        except LogFileAccessError as err:
            print(f"\n[ERROR] Permission error: {err}\n", file=sys.stderr)
            return 1
        except OllamaConnectionError as err:
            print(f"\n[ERROR] Ollama Connection Failure:\n  {err}\n  Hint: Start Ollama by running 'ollama serve' in a terminal.\n", file=sys.stderr)
            return 1
        except OllamaModelNotFoundError as err:
            print(f"\n[ERROR] Model Not Available:\n  {err}\n", file=sys.stderr)
            return 1
        except OllamaTimeoutError as err:
            print(f"\n[ERROR] Inference Timeout:\n  {err}\n", file=sys.stderr)
            return 1
        except OllamaResponseError as err:
            print(f"\n[ERROR] Ollama Server Error:\n  {err}\n", file=sys.stderr)
            return 1
        except MalformedJSONError as err:
            print(f"\n[ERROR] Model output could not be parsed into JSON:\n  {err}\n", file=sys.stderr)
            return 1
        except ValidationError as err:
            print(f"\n[ERROR] Model output failed schema validation:\n  {err}\n", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"\n[UNEXPECTED ERROR] {exc}\n", file=sys.stderr)
            return 1

    # 3. Output formatting
    if args.json:
        report_data = {
            "analysis": result.analysis.to_dict(),
            "pipeline_stats": {
                "original_lines": result.preprocessed_stats.original_line_count,
                "cleaned_lines": result.preprocessed_stats.cleaned_line_count,
                "reduction_percentage": result.preprocessed_stats.reduction_percentage,
            },
            "timings_ms": result.timing_metrics,
            "model": result.model_name,
            "file": result.log_file_path,
        }
        print(json.dumps(report_data, indent=2))
    else:
        display_analysis_report(result)

    # 4. Optional: Save full report
    if args.save_report:
        report_dest = Path(args.save_report).resolve()
        report_dest.parent.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "analysis": result.analysis.to_dict(),
            "timings_ms": result.timing_metrics,
            "model": result.model_name,
            "file": result.log_file_path,
        }
        report_dest.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        print(f"[SUCCESS] Analysis report saved to: {report_dest}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
