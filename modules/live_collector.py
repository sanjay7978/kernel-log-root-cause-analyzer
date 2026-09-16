"""
Live Linux Kernel Log Collector Module (Phase 3B).

Responsible for safely capturing live Linux kernel logs via `dmesg` and `journalctl`,
including previous-boot investigations (-b -1).

Safety & Compatibility Guarantees:
- Strictly executes commands with shell=False and validated argument lists.
- Enforces execution timeouts and line limits to prevent buffer bloat.
- Gracefully detects non-Linux host platforms (e.g. macOS development) and provides
  informative platform guidance.
- Distinguishes permission denials (kernel.dmesg_restrict) from missing binaries.
"""

import logging
import platform
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from config import (
    DEFAULT_LIVE_LOG_LINES,
    LIVE_COLLECTION_TIMEOUT_SECONDS,
)

logger = logging.getLogger("kernel_log_analyzer.live_collector")


class LiveCollectionError(Exception):
    """Base exception for all live log collection failures."""
    pass


class PlatformNotSupportedError(LiveCollectionError):
    """Raised when live collection is invoked on a non-Linux OS."""
    pass


class CollectorNotFoundError(LiveCollectionError):
    """Raised when neither dmesg nor journalctl is available in PATH."""
    pass


class PermissionDeniedError(LiveCollectionError):
    """Raised when dmesg_restrict or lack of root/CAP_SYSLOG prevents reading."""
    pass


class CollectionExecutionError(LiveCollectionError):
    """Raised when subprocess fails or times out."""
    pass


def is_linux() -> bool:
    """Check whether the host operating system is Linux."""
    return platform.system() == "Linux"


def get_collector_capabilities() -> Dict[str, Any]:
    """
    Inspect the host environment and return available kernel log collection tools.
    """
    os_name = platform.system()
    has_linux = is_linux()
    dmesg_path = shutil.which("dmesg")
    journalctl_path = shutil.which("journalctl")

    return {
        "platform": os_name,
        "is_linux": has_linux,
        "has_dmesg": bool(dmesg_path),
        "dmesg_path": dmesg_path,
        "has_journalctl": bool(journalctl_path),
        "journalctl_path": journalctl_path,
        "can_collect": has_linux and bool(dmesg_path or journalctl_path),
    }


def execute_collector_command(
    cmd: List[str],
    timeout: int = LIVE_COLLECTION_TIMEOUT_SECONDS,
) -> str:
    """
    Execute a system command safely with shell=False and strict timeout.
    """
    logger.info("Executing collector command: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
        )
    except FileNotFoundError as exc:
        raise CollectorNotFoundError(f"Collector command '{cmd[0]}' not found on this system.") from exc
    except subprocess.TimeoutExpired as exc:
        raise CollectionExecutionError(
            f"Command '{' '.join(cmd)}' timed out after {timeout} seconds."
        ) from exc
    except OSError as exc:
        raise CollectionExecutionError(f"OS error executing '{' '.join(cmd)}': {exc}") from exc

    if proc.returncode != 0:
        err_msg = (proc.stderr or proc.stdout or "").strip()
        err_lower = err_msg.lower()

        if "permission denied" in err_lower or "operation not permitted" in err_lower:
            raise PermissionDeniedError(
                f"Permission denied executing '{cmd[0]}'. "
                "Ensure your user has read permissions (CAP_SYSLOG) or run with sudo. "
                f"Details: {err_msg}"
            )
        raise CollectionExecutionError(
            f"Command '{' '.join(cmd)}' failed with exit code {proc.returncode}. Details: {err_msg}"
        )

    output = proc.stdout
    if not output or not output.strip():
        raise CollectionExecutionError(f"Command '{' '.join(cmd)}' returned an empty log buffer.")

    return output


def collect_dmesg(
    lines: Optional[int] = DEFAULT_LIVE_LOG_LINES,
    timeout: int = LIVE_COLLECTION_TIMEOUT_SECONDS,
) -> str:
    """
    Capture the kernel ring buffer using dmesg.
    Attempts ISO timestamps first, falling back to standard dmesg.
    """
    if not shutil.which("dmesg"):
        raise CollectorNotFoundError("The 'dmesg' binary was not found in PATH.")

    # Try ISO timestamps for high-precision time data
    cmd = ["dmesg", "--time-format", "iso"]
    try:
        raw_output = execute_collector_command(cmd, timeout=timeout)
    except CollectionExecutionError:
        # Fallback to standard dmesg (e.g. older busybox or kernel versions without --time-format)
        logger.info("dmesg --time-format iso unsupported; falling back to standard dmesg")
        raw_output = execute_collector_command(["dmesg"], timeout=timeout)

    output_lines = raw_output.splitlines()
    if lines and lines > 0 and len(output_lines) > lines:
        logger.info("Truncating dmesg buffer from %d to latest %d lines", len(output_lines), lines)
        output_lines = output_lines[-lines:]

    return "\n".join(output_lines) + "\n"


def collect_journalctl(
    boot_offset: int = 0,
    lines: Optional[int] = DEFAULT_LIVE_LOG_LINES,
    timeout: int = LIVE_COLLECTION_TIMEOUT_SECONDS,
) -> str:
    """
    Capture kernel messages from systemd journal (journalctl -k).
    Supports current boot (0), previous boot (-1), and line capping.
    """
    if not shutil.which("journalctl"):
        raise CollectorNotFoundError("The 'journalctl' binary was not found in PATH.")

    cmd = ["journalctl", "-k", "--no-pager", f"-b{boot_offset}"]
    if lines and lines > 0:
        cmd.extend(["-n", str(lines)])

    return execute_collector_command(cmd, timeout=timeout)


def collect_live_kernel_log(
    collector_type: str = "auto",
    boot_offset: int = 0,
    lines: Optional[int] = DEFAULT_LIVE_LOG_LINES,
    timeout: int = LIVE_COLLECTION_TIMEOUT_SECONDS,
) -> Tuple[str, str]:
    """
    Unified entry point for capturing live Linux kernel logs.

    Parameters:
        collector_type: 'auto', 'dmesg', or 'journalctl'.
        boot_offset: 0 for current boot, -1 for previous boot (journalctl only).
        lines: Maximum number of lines to retain.
        timeout: Execution timeout in seconds.

    Returns:
        Tuple[str, str]: (raw_kernel_log_text, source_description)

    Raises:
        PlatformNotSupportedError: If running on a non-Linux OS (e.g. macOS).
        CollectorNotFoundError: If requested tool is missing.
        PermissionDeniedError: If permissions are insufficient.
        CollectionExecutionError: If execution fails or times out.
    """
    if not is_linux():
        host_os = platform.system()
        raise PlatformNotSupportedError(
            f"Live kernel log collection requires a native Linux host (detected: {host_os}). "
            "On macOS, please upload or paste stored logs, or run this application inside an Ubuntu VM."
        )

    col_type = (collector_type or "auto").strip().lower()
    has_dmesg = bool(shutil.which("dmesg"))
    has_journalctl = bool(shutil.which("journalctl"))

    # Auto-detection strategy
    if col_type == "auto":
        # If examining a previous boot, journalctl is required
        if boot_offset != 0:
            if not has_journalctl:
                raise CollectorNotFoundError(
                    f"Previous boot analysis (boot_offset={boot_offset}) requires 'journalctl', which was not found."
                )
            col_type = "journalctl"
        elif has_dmesg:
            col_type = "dmesg"
        elif has_journalctl:
            col_type = "journalctl"
        else:
            raise CollectorNotFoundError("Neither 'dmesg' nor 'journalctl' was found on this Linux system.")

    # Execute requested collector
    if col_type == "dmesg":
        if boot_offset != 0:
            logger.warning("dmesg only inspects the active boot ring buffer; ignoring boot_offset=%d", boot_offset)
        log_text = collect_dmesg(lines=lines, timeout=timeout)
        label = f"Live Linux dmesg (latest {lines or 'all'} lines)"
        return log_text, label

    if col_type == "journalctl":
        log_text = collect_journalctl(boot_offset=boot_offset, lines=lines, timeout=timeout)
        boot_desc = "current boot" if boot_offset == 0 else f"boot offset {boot_offset}"
        label = f"Live Linux journalctl -k ({boot_desc}, latest {lines or 'all'} lines)"
        return log_text, label

    raise ValueError(f"Unknown collector type: '{collector_type}'. Use 'auto', 'dmesg', or 'journalctl'.")
