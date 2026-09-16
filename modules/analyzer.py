"""
Orchestration Pipeline Module for Kernel Log Root Cause Analyzer (Phase 2 & 3A).

Integrates the end-to-end pipeline:
  Kernel Log Input (Stored File, Uploaded File, or Pasted Text)
            ↓
       Log Loader / Input Validation
            ↓
       Preprocessor
            ↓
      Prompt Builder
            ↓
    Local LLM (Ollama)
            ↓
      Result Parser
            ↓
  Validated Diagnostic Report

Provides a single, unified analysis engine for both the CLI (app.py) and
the Gradio graphical interface (gradio_app.py). Tracks performance timing
metrics across all pipeline stages with structured Python logging.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

from config import MAX_LOG_FILE_SIZE_BYTES, OLLAMA_MODEL
from modules.llm_analyzer import OllamaClient
from modules.log_loader import (
    LogFileEmptyError,
    LogFileTooLargeError,
    load_log_file,
)
from modules.preprocessor import PreprocessingStats, preprocess_kernel_log
from modules.prompt_builder import build_diagnostic_prompt
from modules.result_parser import DiagnosticAnalysis, parse_and_validate_analysis

logger = logging.getLogger("kernel_log_analyzer.pipeline")


@dataclass
class PipelineExecutionResult:
    """Complete diagnostic result with pipeline performance metrics."""
    analysis: DiagnosticAnalysis
    preprocessed_stats: PreprocessingStats
    cleaned_log_excerpt: str
    timing_metrics: Dict[str, float]
    log_file_path: Optional[str] = None
    model_name: str = OLLAMA_MODEL


def analyze_raw_content(
    raw_content: str,
    source_label: str = "Pasted Log",
    model: Optional[str] = None,
    ollama_client: Optional[OllamaClient] = None,
    initial_timings: Optional[Dict[str, float]] = None,
) -> PipelineExecutionResult:
    """
    Execute the root cause analysis pipeline for in-memory kernel log text.
    Used for pasted text, uploaded file content, and stored files.

    Parameters:
        raw_content: Raw text of the kernel log.
        source_label: Descriptive label for the source (e.g. file path or 'Pasted Log').
        model: Optional model name override.
        ollama_client: Optional pre-configured OllamaClient (useful for testing/mocking).
        initial_timings: Optional pre-recorded timings (e.g. file loading time).

    Returns:
        PipelineExecutionResult: Container with validated analysis, stats, and timings.

    Raises:
        LogFileEmptyError: If raw_content contains no usable characters.
        LogFileTooLargeError: If raw_content exceeds MAX_LOG_FILE_SIZE_BYTES.
    """
    total_start = time.perf_counter()
    timings: Dict[str, float] = dict(initial_timings or {})

    # 1. Input validation & size safety check
    if not raw_content or not raw_content.strip():
        logger.warning("Empty or blank log content supplied from %s", source_label)
        raise LogFileEmptyError(f"Log content is empty (0 bytes or only whitespace) from {source_label}.")

    content_size_bytes = len(raw_content.encode("utf-8", errors="replace"))
    if content_size_bytes > MAX_LOG_FILE_SIZE_BYTES:
        logger.error(
            "Log content size (%d bytes) exceeds safety limit of %d bytes from %s",
            content_size_bytes,
            MAX_LOG_FILE_SIZE_BYTES,
            source_label,
        )
        raise LogFileTooLargeError(
            f"Log content size ({content_size_bytes} bytes) exceeds safety limit of {MAX_LOG_FILE_SIZE_BYTES} bytes."
        )

    logger.info("Starting log analysis for source: '%s' (size: %d bytes)", source_label, content_size_bytes)

    # 2. Preprocess conservatively
    t1 = time.perf_counter()
    preprocessed = preprocess_kernel_log(raw_content)
    timings["preprocess_ms"] = round((time.perf_counter() - t1) * 1000, 2)

    logger.info(
        "Preprocessing complete: %d lines -> %d cleaned lines (%.1f%% noise reduction, %d candidate markers)",
        preprocessed.stats.original_line_count,
        preprocessed.stats.cleaned_line_count,
        preprocessed.stats.reduction_percentage,
        preprocessed.stats.candidate_evidence_count,
    )

    if not preprocessed.cleaned_text.strip():
        logger.warning("Log content yielded 0 usable lines after preprocessing from %s", source_label)
        raise LogFileEmptyError("Log content contained no valid lines after preprocessing.")

    # 3. Build diagnostic prompt
    t2 = time.perf_counter()
    prompt = build_diagnostic_prompt(preprocessed.cleaned_text)
    timings["prompt_build_ms"] = round((time.perf_counter() - t2) * 1000, 2)

    # 4. Invoke Local LLM through Ollama
    client = ollama_client or OllamaClient(model=model or OLLAMA_MODEL)
    target_model = model or getattr(client, "model", OLLAMA_MODEL)
    logger.info("Submitting prompt to Ollama model '%s'...", target_model)

    t3 = time.perf_counter()
    llm_resp = client.generate(prompt=prompt, model=model)
    timings["llm_inference_ms"] = round((time.perf_counter() - t3) * 1000, 2)
    timings["ollama_eval_duration_ms"] = llm_resp.eval_duration_ms
    timings["ollama_eval_count"] = float(llm_resp.eval_count)

    logger.info(
        "Ollama generation finished in %.2f ms (eval: %d tokens, model: %s)",
        timings["llm_inference_ms"],
        llm_resp.eval_count,
        llm_resp.model,
    )

    # 5. Parse and validate LLM output
    t4 = time.perf_counter()
    analysis = parse_and_validate_analysis(llm_resp.content)
    timings["parse_validate_ms"] = round((time.perf_counter() - t4) * 1000, 2)

    timings["total_pipeline_ms"] = round((time.perf_counter() - total_start + timings.get("load_ms", 0) / 1000) * 1000, 2)

    logger.info(
        "Analysis complete for '%s'. Subsystem: %s, Severity: %s, Confidence: %s",
        source_label,
        analysis.subsystem,
        analysis.severity,
        analysis.confidence,
    )

    return PipelineExecutionResult(
        analysis=analysis,
        preprocessed_stats=preprocessed.stats,
        cleaned_log_excerpt=preprocessed.cleaned_text,
        timing_metrics=timings,
        log_file_path=source_label,
        model_name=llm_resp.model,
    )


def analyze_log(
    file_path: Union[str, Path],
    model: Optional[str] = None,
    ollama_client: Optional[OllamaClient] = None,
) -> PipelineExecutionResult:
    """
    Execute the root cause analysis pipeline for a stored log file path.
    Loads the file safely and delegates to analyze_raw_content.

    Parameters:
        file_path: Path to stored kernel log file.
        model: Optional model name override.
        ollama_client: Optional pre-configured OllamaClient (useful for testing/mocking).

    Returns:
        PipelineExecutionResult: Container with validated analysis, stats, and timings.
    """
    target_path = Path(file_path).expanduser().resolve()
    logger.info("Loading log file: %s", target_path)

    # Load file with full boundary and safety checks
    t0 = time.perf_counter()
    raw_content = load_log_file(target_path)
    load_duration_ms = round((time.perf_counter() - t0) * 1000, 2)

    return analyze_raw_content(
        raw_content=raw_content,
        source_label=str(target_path),
        model=model,
        ollama_client=ollama_client,
        initial_timings={"load_ms": load_duration_ms},
    )


def analyze_live(
    collector_type: str = "auto",
    boot_offset: int = 0,
    lines: Optional[int] = None,
    model: Optional[str] = None,
    ollama_client: Optional[OllamaClient] = None,
) -> PipelineExecutionResult:
    """
    Capture live kernel logs from the Linux host (dmesg or journalctl)
    and execute the root cause analysis pipeline.

    Parameters:
        collector_type: 'auto', 'dmesg', or 'journalctl'.
        boot_offset: 0 for current boot, -1 for previous boot.
        lines: Maximum number of lines to capture.
        model: Optional model override.
        ollama_client: Optional pre-configured OllamaClient.

    Returns:
        PipelineExecutionResult: Container with validated analysis, stats, and timings.
    """
    from modules.live_collector import collect_live_kernel_log

    t0 = time.perf_counter()
    raw_content, source_label = collect_live_kernel_log(
        collector_type=collector_type,
        boot_offset=boot_offset,
        lines=lines,
    )
    collect_duration_ms = round((time.perf_counter() - t0) * 1000, 2)

    return analyze_raw_content(
        raw_content=raw_content,
        source_label=source_label,
        model=model,
        ollama_client=ollama_client,
        initial_timings={"collect_ms": collect_duration_ms},
    )
