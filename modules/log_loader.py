"""
Log Loader Module for Kernel Log Root Cause Analyzer (Phase 1).

Responsible solely for safely ingesting stored kernel log files from the filesystem.
Validates file existence, file type, permissions, size constraints, and handles
encoding nuances gracefully.

This module intentionally contains NO diagnostic logic, subsystem heuristics, or LLM logic.
"""

from pathlib import Path
from typing import Union

try:
    from config import MAX_LOG_FILE_SIZE_BYTES
except ImportError:
    # Fallback to 50 MB if imported outside project root
    MAX_LOG_FILE_SIZE_BYTES = 50 * 1024 * 1024


class LogLoaderError(Exception):
    """Base exception for all log loading failures."""
    pass


class LogFileNotFoundError(LogLoaderError):
    """Raised when the specified log file does not exist."""
    pass


class LogNotAFileError(LogLoaderError):
    """Raised when the specified path is a directory or special device, not a file."""
    pass


class LogFileEmptyError(LogLoaderError):
    """Raised when the log file is 0 bytes or contains no text."""
    pass


class LogFileAccessError(LogLoaderError):
    """Raised when permissions or I/O errors prevent reading the log file."""
    pass


class LogFileTooLargeError(LogLoaderError):
    """Raised when the log file exceeds the maximum allowed size threshold."""
    pass


def load_log_file(
    file_path: Union[str, Path],
    max_size_bytes: int = MAX_LOG_FILE_SIZE_BYTES,
) -> str:
    """
    Safely load a raw kernel log file from disk into a string.

    Parameters:
        file_path: Absolute or relative path to the kernel log file.
        max_size_bytes: Maximum allowed file size in bytes (defaults to config value).

    Returns:
        str: Raw log file content.

    Raises:
        LogFileNotFoundError: If the file does not exist on disk.
        LogNotAFileError: If the path points to a directory or non-regular file.
        LogFileEmptyError: If the file size is 0 bytes or contains only whitespace.
        LogFileTooLargeError: If file size exceeds max_size_bytes.
        LogFileAccessError: If permission denied or unexpected OS error occurs.
    """
    if file_path is None or str(file_path).strip() == "":
        raise LogFileNotFoundError("No log file path was provided.")

    target_path = Path(file_path).expanduser().resolve()

    # 1. Verification of existence
    if not target_path.exists():
        raise LogFileNotFoundError(f"Log file does not exist: {target_path}")

    # 2. Verification that target is a regular file
    if not target_path.is_file():
        raise LogNotAFileError(f"Expected a regular file, but found directory or special file: {target_path}")

    # 3. File size and safety checks
    try:
        file_size = target_path.stat().st_size
    except OSError as exc:
        raise LogFileAccessError(f"Unable to read file statistics for {target_path}: {exc}") from exc

    if file_size == 0:
        raise LogFileEmptyError(f"Log file is empty (0 bytes): {target_path}")

    if file_size > max_size_bytes:
        raise LogFileTooLargeError(
            f"Log file size ({file_size} bytes) exceeds safety limit of {max_size_bytes} bytes: {target_path}"
        )

    # 4. Safe read with encoding handling
    # Uses utf-8 with replacement for corrupted bytes or unexpected terminal control sequences
    try:
        with open(target_path, mode="r", encoding="utf-8", errors="replace") as file_handle:
            content = file_handle.read()
    except PermissionError as exc:
        raise LogFileAccessError(f"Permission denied reading file: {target_path}") from exc
    except OSError as exc:
        raise LogFileAccessError(f"I/O error reading file {target_path}: {exc}") from exc

    # 5. Content check
    if not content.strip():
        raise LogFileEmptyError(f"Log file contains only whitespace or blank characters: {target_path}")

    return content
