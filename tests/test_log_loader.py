"""
Unit tests for modules/log_loader.py.

Verifies defensive file ingestion, edge cases, error conditions,
size bounds, and encoding resilience using pytest.
"""

import os
import stat
# pyrefly: ignore [missing-import]
import pytest
from pathlib import Path

from modules.log_loader import (
    load_log_file,
    LogLoaderError,
    LogFileNotFoundError,
    LogNotAFileError,
    LogFileEmptyError,
    LogFileTooLargeError,
    LogFileAccessError,
)


def test_load_valid_file(tmp_path: Path):
    """Test loading a normal, valid log file returns exact content."""
    sample_content = "[    0.000000] Linux version 6.8.0-generic\n[    0.000001] Initializing cgroup subsys cpuset\n"
    log_file = tmp_path / "dmesg.log"
    log_file.write_text(sample_content, encoding="utf-8")

    result = load_log_file(log_file)
    assert result == sample_content


def test_load_accepts_str_and_path(tmp_path: Path):
    """Test that both str and Path instances are handled identically."""
    log_file = tmp_path / "valid.log"
    log_file.write_text("kernel log line\n", encoding="utf-8")

    result_path = load_log_file(log_file)
    result_str = load_log_file(str(log_file))

    assert result_path == result_str == "kernel log line\n"


def test_load_nonexistent_file(tmp_path: Path):
    """Test that querying a nonexistent file raises LogFileNotFoundError."""
    missing = tmp_path / "does_not_exist.log"
    with pytest.raises(LogFileNotFoundError) as exc_info:
        load_log_file(missing)

    assert "does not exist" in str(exc_info.value)
    assert isinstance(exc_info.value, LogLoaderError)


def test_load_empty_or_none_path():
    """Test that empty or None path arguments raise LogFileNotFoundError."""
    with pytest.raises(LogFileNotFoundError):
        load_log_file("")

    with pytest.raises(LogFileNotFoundError):
        load_log_file(None)  # type: ignore


def test_load_directory_as_file(tmp_path: Path):
    """Test that passing a directory path raises LogNotAFileError."""
    dir_path = tmp_path / "some_dir"
    dir_path.mkdir()

    with pytest.raises(LogNotAFileError) as exc_info:
        load_log_file(dir_path)

    assert "Expected a regular file" in str(exc_info.value)


def test_load_zero_byte_file(tmp_path: Path):
    """Test that an empty 0-byte file raises LogFileEmptyError."""
    empty_file = tmp_path / "empty.log"
    empty_file.touch()

    with pytest.raises(LogFileEmptyError) as exc_info:
        load_log_file(empty_file)

    assert "0 bytes" in str(exc_info.value)


def test_load_whitespace_only_file(tmp_path: Path):
    """Test that a file containing only whitespace/newlines raises LogFileEmptyError."""
    blank_file = tmp_path / "blank.log"
    blank_file.write_text("   \n\t  \n  \n", encoding="utf-8")

    with pytest.raises(LogFileEmptyError) as exc_info:
        load_log_file(blank_file)

    assert "whitespace or blank" in str(exc_info.value)


def test_load_file_too_large(tmp_path: Path):
    """Test that a file exceeding max_size_bytes raises LogFileTooLargeError."""
    large_file = tmp_path / "large.log"
    large_file.write_text("A" * 200, encoding="utf-8")

    # Set safety limit to 100 bytes for this test
    with pytest.raises(LogFileTooLargeError) as exc_info:
        load_log_file(large_file, max_size_bytes=100)

    assert "exceeds safety limit" in str(exc_info.value)


def test_load_problematic_non_utf8_encoding(tmp_path: Path):
    """Test that invalid byte sequences are handled safely without throwing UnicodeDecodeError."""
    corrupted_file = tmp_path / "corrupted.log"
    # Write invalid UTF-8 sequences (0xFF, 0xFE, 0xC0)
    corrupted_file.write_bytes(b"[ 1.234567] Driver initialized \xff\xfe corrupted \xc0 data\n")

    result = load_log_file(corrupted_file)
    assert "[ 1.234567] Driver initialized" in result
    assert "\ufffd" in result  # Replacement character should be present
    assert "corrupted" in result


def test_load_unreadable_file_permissions(tmp_path: Path):
    """Test that an unreadable file due to missing permissions raises LogFileAccessError."""
    unreadable_file = tmp_path / "restricted.log"
    unreadable_file.write_text("secure content\n", encoding="utf-8")

    # Remove all read permissions
    os.chmod(unreadable_file, 0)

    try:
        with pytest.raises(LogFileAccessError) as exc_info:
            load_log_file(unreadable_file)
        assert "Permission denied" in str(exc_info.value) or "Unable to read" in str(exc_info.value)
    finally:
        # Restore permissions so pytest tmp_path cleanup can delete the file cleanly
        os.chmod(unreadable_file, stat.S_IRUSR | stat.S_IWUSR)
