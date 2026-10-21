"""
Comprehensive unit and integration tests for Phase 3B:
- Platform check & capability detection (modules/live_collector.py)
- Mocked subprocess executions (dmesg, journalctl, ISO timestamp fallback)
- Safe execution, timeout, not found, and permission handling
- Live capture pipeline orchestration (modules/analyzer.py analyze_live)
- UI live capture controller (modules/ui_service.py run_live_ui_analysis)
"""

import platform
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from modules.analyzer import analyze_live
from modules.live_collector import (
    CollectionExecutionError,
    CollectorNotFoundError,
    PermissionDeniedError,
    PlatformNotSupportedError,
    collect_dmesg,
    collect_journalctl,
    collect_live_kernel_log,
    execute_collector_command,
    get_collector_capabilities,
    is_linux,
)
from modules.llm_analyzer import LLMResponse, OllamaClient
from modules.ui_service import (
    get_live_host_status_html,
    run_live_ui_analysis,
)


@pytest.fixture
def mock_ollama_client():
    """Fixture providing a mock OllamaClient with a valid structured response."""
    mock_client = MagicMock(spec=OllamaClient)
    mock_response = LLMResponse(
        content="""{
            "root_cause": "Process stress-ng exhausted system memory invoking the OOM killer",
            "subsystem": "memory",
            "severity": "HIGH",
            "evidence": [
                "stress-ng invoked oom-killer: gfp_mask=0x1100cca",
                "Out of memory: Killed process 3412 (stress-ng-vm)"
            ],
            "explanation": "High memory allocation triggered out_of_memory in page allocator.",
            "commands": ["free -m", "dmesg -T | grep -i oom"],
            "possible_fixes": ["Configure cgroup limits", "Add swap space"],
            "confidence": "HIGH",
            "uncertainty": "None"
        }""",
        model="qwen3:8b",
        total_duration_ms=2500.0,
        eval_duration_ms=1800.0,
        eval_count=120,
    )
    mock_client.generate.return_value = mock_response
    mock_client.model = "qwen3:8b"
    return mock_client


# ============================================================
# 1. Platform & Capability Detection
# ============================================================

def test_is_linux_current_host():
    """Verify is_linux reflects whether the host OS is Linux."""
    expected = platform.system() == "Linux"
    assert is_linux() == expected


def test_is_linux_mocked():
    """Verify is_linux properly interprets mocked platform names."""
    with patch("platform.system", return_value="Linux"):
        assert is_linux() is True

    with patch("platform.system", return_value="Darwin"):
        assert is_linux() is False

    with patch("platform.system", return_value="Windows"):
        assert is_linux() is False


def test_get_collector_capabilities_non_linux():
    """Verify that on non-Linux hosts, collectors are reported as unavailable."""
    with patch("modules.live_collector.is_linux", return_value=False), \
         patch("platform.system", return_value="Darwin"):
        caps = get_collector_capabilities()
        assert caps["can_collect"] is False
        assert caps["is_linux"] is False
        assert caps["platform"] == "Darwin"


def test_get_collector_capabilities_linux():
    """Verify capability detection when host is Linux and tools are present."""
    with patch("modules.live_collector.is_linux", return_value=True), \
         patch("modules.live_collector.shutil.which") as mock_which, \
         patch("platform.system", return_value="Linux"):

        # Both tools found
        mock_which.side_effect = lambda cmd: f"/usr/bin/{cmd}"
        caps = get_collector_capabilities()
        assert caps["can_collect"] is True
        assert caps["has_dmesg"] is True
        assert caps["has_journalctl"] is True
        assert caps["is_linux"] is True

        # Only dmesg found
        mock_which.side_effect = lambda cmd: "/bin/dmesg" if cmd == "dmesg" else None
        caps = get_collector_capabilities()
        assert caps["has_dmesg"] is True
        assert caps["has_journalctl"] is False
        assert caps["can_collect"] is True


# ============================================================
# 2. Command Execution Safety & Error Handling
# ============================================================

def test_execute_collector_command_success():
    """Verify successful execution returns stdout without error."""
    mock_proc = subprocess.CompletedProcess(
        args=["dmesg"],
        returncode=0,
        stdout="[ 0.000000] Linux version 6.8.0\n[ 0.000001] Command line: BOOT_IMAGE=/vmlinuz\n",
        stderr="",
    )
    with patch("modules.live_collector.subprocess.run", return_value=mock_proc) as mock_run:
        output = execute_collector_command(["dmesg"], timeout=5)
        assert "Linux version 6.8.0" in output
        mock_run.assert_called_once_with(
            ["dmesg"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            shell=False,
            check=False,
        )


def test_execute_collector_command_permission_denied():
    """Verify permission denied stderr triggers PermissionDeniedError."""
    mock_proc = subprocess.CompletedProcess(
        args=["dmesg"],
        returncode=1,
        stdout="",
        stderr="dmesg: read kernel buffer failed: Operation not permitted",
    )
    with patch("modules.live_collector.subprocess.run", return_value=mock_proc):
        with pytest.raises(PermissionDeniedError) as exc_info:
            execute_collector_command(["dmesg"])
        assert "Permission denied" in str(exc_info.value)


def test_execute_collector_command_not_found():
    """Verify FileNotFoundError triggers CollectorNotFoundError."""
    with patch("modules.live_collector.subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(CollectorNotFoundError) as exc_info:
            execute_collector_command(["dmesg"])
        assert "not found on this system" in str(exc_info.value)


def test_execute_collector_command_timeout():
    """Verify subprocess timeout triggers CollectionExecutionError."""
    with patch("modules.live_collector.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["dmesg"], timeout=15)):
        with pytest.raises(CollectionExecutionError) as exc_info:
            execute_collector_command(["dmesg"], timeout=15)
        assert "timed out after 15 seconds" in str(exc_info.value)


def test_execute_collector_command_empty_output():
    """Verify empty stdout triggers CollectionExecutionError."""
    mock_proc = subprocess.CompletedProcess(
        args=["dmesg"],
        returncode=0,
        stdout="   \n\n  ",
        stderr="",
    )
    with patch("modules.live_collector.subprocess.run", return_value=mock_proc):
        with pytest.raises(CollectionExecutionError) as exc_info:
            execute_collector_command(["dmesg"])
        assert "returned an empty log buffer" in str(exc_info.value)


# ============================================================
# 3. dmesg Collection
# ============================================================

def test_collect_dmesg_tool_not_found_raises():
    """Verify collect_dmesg raises CollectorNotFoundError if dmesg is missing."""
    with patch("modules.live_collector.shutil.which", return_value=None):
        with pytest.raises(CollectorNotFoundError) as exc_info:
            collect_dmesg()
        assert "The 'dmesg' binary was not found" in str(exc_info.value)


def test_collect_dmesg_success_with_line_truncation():
    """Verify collect_dmesg truncates to recent lines."""
    raw_lines = [f"[ {i}.000000] Kernel log line {i}" for i in range(100)]
    mock_proc = subprocess.CompletedProcess(
        args=["dmesg", "--time-format", "iso"],
        returncode=0,
        stdout="\n".join(raw_lines) + "\n",
        stderr="",
    )
    with patch("modules.live_collector.shutil.which", return_value="/bin/dmesg"), \
         patch("modules.live_collector.subprocess.run", return_value=mock_proc):
        content = collect_dmesg(lines=25)

        lines = content.strip().splitlines()
        assert len(lines) == 25
        assert "Kernel log line 99" in lines[-1]


def test_collect_dmesg_fallback_from_timestamp():
    """Verify collect_dmesg falls back to standard dmesg if --time-format iso fails."""
    call_count = 0

    def mock_run_impl(cmd, **kwargs):
        nonlocal call_count
        call_count += 1
        if "--time-format" in cmd:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="dmesg: unrecognized option '--time-format'",
            )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="[ 0.123] Standard dmesg line\n",
            stderr="",
        )

    with patch("modules.live_collector.shutil.which", return_value="/bin/dmesg"), \
         patch("modules.live_collector.subprocess.run", side_effect=mock_run_impl):
        content = collect_dmesg()
        assert "Standard dmesg line" in content
        assert call_count == 2


# ============================================================
# 4. journalctl Collection
# ============================================================

def test_collect_journalctl_tool_not_found_raises():
    """Verify collect_journalctl raises CollectorNotFoundError if journalctl is missing."""
    with patch("modules.live_collector.shutil.which", return_value=None):
        with pytest.raises(CollectorNotFoundError) as exc_info:
            collect_journalctl()
        assert "The 'journalctl' binary was not found" in str(exc_info.value)


def test_collect_journalctl_boot_offset_and_lines():
    """Verify journalctl arguments include -k, -b<boot>, -n <lines>."""
    mock_proc = subprocess.CompletedProcess(
        args=["journalctl", "-k", "--no-pager", "-b-1", "-n", "500"],
        returncode=0,
        stdout="Sep 16 20:00:00 host kernel: Linux version 6.8.0\n",
        stderr="",
    )
    with patch("modules.live_collector.shutil.which", return_value="/bin/journalctl"), \
         patch("modules.live_collector.subprocess.run", return_value=mock_proc) as mock_run:
        content = collect_journalctl(boot_offset=-1, lines=500)

        assert "Linux version 6.8.0" in content
        mock_run.assert_called_once_with(
            ["journalctl", "-k", "--no-pager", "-b-1", "-n", "500"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            shell=False,
            check=False,
        )


# ============================================================
# 5. Unified Live Collector Routing
# ============================================================

def test_collect_live_kernel_log_on_non_linux_raises():
    """Verify collect_live_kernel_log raises PlatformNotSupportedError on non-Linux."""
    with patch("modules.live_collector.is_linux", return_value=False):
        with pytest.raises(PlatformNotSupportedError) as exc_info:
            collect_live_kernel_log()
        assert "requires a native Linux host" in str(exc_info.value)


def test_collect_live_kernel_log_auto_selection():
    """Verify auto collector prioritizes dmesg for current boot, journalctl for previous boot."""
    with patch("modules.live_collector.is_linux", return_value=True):
        # Case A: Current boot with dmesg available
        with patch("modules.live_collector.shutil.which", side_effect=lambda cmd: f"/bin/{cmd}"), \
             patch("modules.live_collector.collect_dmesg", return_value="dmesg output\n"):
            content, label = collect_live_kernel_log(collector_type="auto", boot_offset=0)
            assert content == "dmesg output\n"
            assert "Live Linux dmesg" in label

        # Case B: Previous boot (-1) auto-selects journalctl
        with patch("modules.live_collector.shutil.which", side_effect=lambda cmd: f"/bin/{cmd}"), \
             patch("modules.live_collector.collect_journalctl", return_value="journalctl output\n"):
            content, label = collect_live_kernel_log(collector_type="auto", boot_offset=-1)
            assert content == "journalctl output\n"
            assert "boot offset -1" in label

        # Case C: Neither tool found
        with patch("modules.live_collector.shutil.which", return_value=None):
            with pytest.raises(CollectorNotFoundError) as exc_info:
                collect_live_kernel_log(collector_type="auto")
            assert "Neither 'dmesg' nor 'journalctl' was found" in str(exc_info.value)


def test_collect_live_kernel_log_invalid_collector():
    """Verify unsupported collector raises ValueError on Linux."""
    with patch("modules.live_collector.is_linux", return_value=True):
        with pytest.raises(ValueError) as exc_info:
            collect_live_kernel_log(collector_type="syslog_ng")
        assert "Unknown collector type: 'syslog_ng'" in str(exc_info.value)


# ============================================================
# 6. Pipeline Live Orchestration (modules/analyzer.py)
# ============================================================

def test_analyze_live_pipeline(mock_ollama_client):
    """Verify analyze_live connects collector to analysis pipeline."""
    sample_log = (
        "[ 120.450123] stress-ng invoked oom-killer: gfp_mask=0x1100cca\n"
        "[ 120.450130] Out of memory: Killed process 3412 (stress-ng-vm)\n"
    )

    with patch("modules.live_collector.collect_live_kernel_log", return_value=(sample_log, "dmesg (last 100 lines)")):
        result = analyze_live(collector_type="dmesg", ollama_client=mock_ollama_client)

        assert result.analysis.subsystem == "memory"
        assert result.analysis.severity == "HIGH"
        assert "collect_ms" in result.timing_metrics
        assert result.timing_metrics["collect_ms"] >= 0
        assert result.log_file_path == "dmesg (last 100 lines)"
        assert "stress-ng" in result.cleaned_log_excerpt


# ============================================================
# 7. UI Service Live Integration (modules/ui_service.py)
# ============================================================

def test_get_live_host_status_html_darwin():
    """Verify the host status card says live capture needs Linux when on macOS."""
    with patch("platform.system", return_value="Darwin"), \
         patch("modules.live_collector.is_linux", return_value=False):
        html = get_live_host_status_html()
        assert "Darwin host" in html
        assert "needs a Linux kernel" in html


def test_get_live_host_status_html_linux():
    """Verify the host status card reflects Linux when active."""
    with patch("platform.system", return_value="Linux"), \
         patch("modules.live_collector.is_linux", return_value=True), \
         patch("modules.live_collector.shutil.which", side_effect=lambda cmd: f"/bin/{cmd}"):
        html = get_live_host_status_html()
        assert "Linux host" in html
        assert "Live capture is available" in html
        assert "kx-on" in html


def test_run_live_ui_analysis_on_darwin_unmocked():
    """Verify run_live_ui_analysis on current macOS returns friendly error HTML."""
    if platform.system() != "Linux":
        report_html, cleaned_log, markdown, json_text = run_live_ui_analysis()
        assert "Live capture is not available here" in report_html
        assert "Linux machine" in report_html
        assert (cleaned_log, markdown, json_text) == ("", "", "")


def test_run_live_ui_analysis_mocked_success(mock_ollama_client):
    """Verify run_live_ui_analysis returns the whole report as one HTML string, plus the cleaned log."""
    sample_log = (
        "[ 120.450123] stress-ng invoked oom-killer: gfp_mask=0x1100cca\n"
        "[ 120.450130] Out of memory: Killed process 3412 (stress-ng-vm)\n"
    )

    with patch("modules.live_collector.collect_live_kernel_log", return_value=(sample_log, "dmesg (live)")):
        outputs = run_live_ui_analysis(ollama_client=mock_ollama_client)

        assert len(outputs) == 4
        report_html, cleaned_log, markdown, json_text = outputs

        assert "stress-ng" in report_html
        assert "MEMORY" in report_html
        assert "HIGH" in report_html
        assert "oom-killer" in report_html
        assert "free -m" in report_html
        assert "capture time" in report_html          # live runs show how long the capture took
        assert "dmesg (live)" in report_html
        assert "stress-ng" in cleaned_log
        assert "dmesg (live)" in markdown and "dmesg (live)" in json_text


def test_run_live_ui_analysis_permission_error():
    """Verify permission error is formatted nicely in UI output."""
    with patch("modules.analyzer.analyze_live", side_effect=PermissionDeniedError("dmesg read failed: Operation not permitted")):
        report_html, cleaned_log, markdown, json_text = run_live_ui_analysis()
        assert "Permission denied reading the kernel log" in report_html
        assert "CAP_SYSLOG" in report_html
        assert (cleaned_log, markdown, json_text) == ("", "", "")
