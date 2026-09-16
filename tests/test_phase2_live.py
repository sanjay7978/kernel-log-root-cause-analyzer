"""
Live integration tests for Phase 2 using the local Ollama daemon.
Automatically skips if Ollama is unreachable on localhost:11434.
"""

from pathlib import Path
import pytest

from config import SAMPLE_LOGS_DIR
from modules.analyzer import analyze_log
from modules.llm_analyzer import OllamaClient


@pytest.fixture(scope="module")
def check_ollama():
    client = OllamaClient()
    if not client.check_health():
        pytest.skip("Local Ollama daemon is not running on http://localhost:11434")


def test_live_oom_analysis(check_ollama):
    """Test live root cause analysis on the authentic OOM incident log."""
    log_path = SAMPLE_LOGS_DIR / "memory" / "real_oom_incident.log"
    result = analyze_log(log_path)

    assert result.analysis.subsystem.lower() == "memory"
    assert result.analysis.severity in ["HIGH", "CRITICAL"]
    assert len(result.analysis.evidence) > 0
    assert len(result.analysis.commands) > 0
    assert len(result.analysis.possible_fixes) > 0
    assert result.timing_metrics["llm_inference_ms"] > 0


def test_live_usb_analysis(check_ollama):
    """Test live root cause analysis on the authentic USB error log."""
    log_path = SAMPLE_LOGS_DIR / "usb" / "real_usb_descriptor_error.log"
    result = analyze_log(log_path)

    assert result.analysis.subsystem.lower() == "usb"
    assert result.analysis.severity in ["MEDIUM", "HIGH"]
    assert len(result.analysis.evidence) > 0
    assert len(result.analysis.commands) > 0
    assert result.timing_metrics["llm_inference_ms"] > 0
