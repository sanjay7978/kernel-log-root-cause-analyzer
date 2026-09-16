"""
Log Preprocessor Module for Kernel Log Root Cause Analyzer (Phase 1).

Conservatively normalizes, sanitizes, and deduplicates raw kernel log text.
Preserves timestamps, hardware prefixes, call traces, and surrounding context.

NOTE: This module identifies candidate evidence lines for statistical metrics,
but strictly DOES NOT classify, diagnose, or infer root causes.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

try:
    from config import EVIDENCE_MARKERS
except ImportError:
    EVIDENCE_MARKERS = (
        "ERROR", "WARN", "WARNING", "failed", "failure", "timeout",
        "critical", "panic", "BUG", "Oops", "Call Trace", "I/O error",
        "segfault", "out of memory", "killed process",
    )

# Regular expression to match ANSI escape codes (color codes, cursor movements, etc.)
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Non-printable control characters to remove (excluding tab \t and newline \n)
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class PreprocessingStats:
    """Statistical summary of the preprocessing stage."""
    original_line_count: int = 0
    cleaned_line_count: int = 0
    empty_lines_removed: int = 0
    duplicate_lines_collapsed: int = 0
    candidate_evidence_count: int = 0

    @property
    def reduction_percentage(self) -> float:
        """Percentage of lines reduced during cleaning."""
        if self.original_line_count == 0:
            return 0.0
        reduced = self.original_line_count - self.cleaned_line_count
        return round((reduced / self.original_line_count) * 100.0, 2)


@dataclass
class PreprocessedLog:
    """Result object containing cleaned text, metrics, and candidate evidence."""
    cleaned_text: str
    stats: PreprocessingStats
    candidate_evidence_lines: List[str] = field(default_factory=list)


def strip_control_and_ansi(text: str) -> str:
    """
    Remove ANSI escape sequences and non-printable control characters.
    Preserves standard formatting characters like tabs and newlines.
    """
    # 1. Remove ANSI escape sequences
    cleaned = ANSI_ESCAPE_PATTERN.sub("", text)
    # 2. Normalize Windows/Mac line endings (\r\n or \r -> \n)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    # 3. Strip remaining non-printable control characters
    cleaned = CONTROL_CHAR_PATTERN.sub("", cleaned)
    return cleaned


def find_candidate_evidence(line: str) -> bool:
    """
    Determine whether a line contains candidate diagnostic markers.

    IMPORTANT: A match does NOT indicate a confirmed root cause.
    It indicates a line worthy of diagnostic review or statistical highlighting.
    """
    line_lower = line.lower()
    for marker in EVIDENCE_MARKERS:
        # Check marker either lowercase or case-sensitive if capitalized specifically
        if marker.lower() in line_lower:
            return True
    return False


def preprocess_kernel_log(
    raw_log: str,
    collapse_consecutive_duplicates: bool = True,
) -> PreprocessedLog:
    """
    Preprocess raw kernel log text conservatively.

    Steps:
      1. Sanitize ANSI escape codes and unprintable control characters.
      2. Split into individual lines.
      3. Strip trailing whitespace while preserving leading indentation.
      4. Remove empty / whitespace-only lines.
      5. Collapse consecutive exact duplicate lines to prevent log flooding.
      6. Record candidate evidence markers without diagnostic filtering.

    Parameters:
        raw_log: Raw log content string from loader.
        collapse_consecutive_duplicates: If True, collapses consecutive identical lines.

    Returns:
        PreprocessedLog containing cleaned_text, PreprocessingStats, and candidate lines.
    """
    if not raw_log or not raw_log.strip():
        return PreprocessedLog(
            cleaned_text="",
            stats=PreprocessingStats(0, 0, 0, 0, 0),
            candidate_evidence_lines=[],
        )

    # Step 1: Control sequence & ANSI cleanup
    sanitized = strip_control_and_ansi(raw_log)
    raw_lines = sanitized.splitlines()
    total_original = len(raw_lines)

    cleaned_lines: List[str] = []
    candidate_lines: List[str] = []
    empty_removed = 0
    duplicates_collapsed = 0

    prev_line: str = ""
    repeat_count: int = 0

    def flush_repeats():
        nonlocal repeat_count, duplicates_collapsed
        if repeat_count > 0:
            cleaned_lines.append(f"    [... preceding message repeated {repeat_count} times ...]")
            duplicates_collapsed += repeat_count
            repeat_count = 0

    # Step 2: Iterate over lines
    for raw_line in raw_lines:
        # Normalize: strip trailing whitespace, keep leading indentation (critical for call traces)
        line = raw_line.rstrip()

        # Step 3: Remove empty / whitespace-only lines
        if not line:
            empty_removed += 1
            continue

        # Step 4: Consecutive duplicate handling
        if collapse_consecutive_duplicates:
            if line == prev_line:
                repeat_count += 1
                continue
            else:
                flush_repeats()
                prev_line = line
        else:
            prev_line = line

        cleaned_lines.append(line)

        # Step 5: Identify candidate evidence (purely for highlighting & metrics)
        if find_candidate_evidence(line):
            candidate_lines.append(line)

    # Flush any remaining repeats at end of file
    flush_repeats()

    stats = PreprocessingStats(
        original_line_count=total_original,
        cleaned_line_count=len(cleaned_lines),
        empty_lines_removed=empty_removed,
        duplicate_lines_collapsed=duplicates_collapsed,
        candidate_evidence_count=len(candidate_lines),
    )

    cleaned_text = "\n".join(cleaned_lines)
    if cleaned_text:
        cleaned_text += "\n"

    return PreprocessedLog(
        cleaned_text=cleaned_text,
        stats=stats,
        candidate_evidence_lines=candidate_lines,
    )


def clean_log_text(raw_log: str) -> str:
    """
    Convenience wrapper returning solely the cleaned log string.
    """
    return preprocess_kernel_log(raw_log).cleaned_text
