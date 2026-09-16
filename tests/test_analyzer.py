"""
Unit tests for modules/analyzer.py (orchestration pipeline).
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from config import SAMPLE_LOGS_DIR
from modules.analyzer import PipelineExecutionResult, analyze_log
from modules.llm_analyzer import LLMResponse, OllamaClient


def test_analyze_log_with_mock_client():
    oom_log_path = SAMPLE_LOGS_DIR / "memory" / "real_oom_incident.log"
    assert oom_log_path.exists()

    # Create mock OllamaClient
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

    result = analyze_log(oom_log_path, ollama_client=mock_client)

    assert isinstance(result, PipelineExecutionResult)
    assert result.analysis.subsystem == "memory"
    assert result.analysis.severity == "HIGH"
    assert "stress-ng" in result.analysis.root_cause
    assert len(result.analysis.evidence) == 2
    assert "total_pipeline_ms" in result.timing_metrics
    assert "llm_inference_ms" in result.timing_metrics
    assert result.timing_metrics["load_ms"] > 0 or result.timing_metrics["preprocess_ms"] >= 0
    assert result.preprocessed_stats.cleaned_line_count == 35
