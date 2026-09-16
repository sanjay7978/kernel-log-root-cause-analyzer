"""
Unit tests for modules/llm_analyzer.py using mock HTTP responses.
"""

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from modules.llm_analyzer import (
    LLMResponse,
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    query_ollama,
)


@patch("urllib.request.urlopen")
def test_ollama_client_generate_success(mock_urlopen):
    # Mock successful HTTP 200 response from Ollama
    mock_response = MagicMock()
    mock_response.status = 200
    fake_body = {
        "response": '{"root_cause": "OOM killer", "severity": "HIGH"}',
        "total_duration": 1_500_000_000,
        "load_duration": 500_000_000,
        "eval_duration": 1_000_000_000,
        "eval_count": 50,
    }
    mock_response.read.return_value = json.dumps(fake_body).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    client = OllamaClient(model="qwen3:8b")
    result = client.generate("Analyze this log")

    assert isinstance(result, LLMResponse)
    assert '{"root_cause": "OOM killer"' in result.content
    assert result.model == "qwen3:8b"
    assert result.total_duration_ms == 1500.0
    assert result.eval_count == 50


def test_ollama_empty_prompt_raises():
    client = OllamaClient()
    with pytest.raises(OllamaResponseError) as exc_info:
        client.generate("")
    assert "Prompt cannot be empty" in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_ollama_connection_error(mock_urlopen):
    # Simulate connection refused / daemon not running
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    client = OllamaClient()
    with pytest.raises(OllamaConnectionError) as exc_info:
        client.generate("test prompt")
    assert "Cannot connect to Ollama" in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_ollama_model_not_found_404(mock_urlopen):
    # Simulate 404 HTTPError
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="http://localhost:11434/api/generate",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=None,
    )

    client = OllamaClient(model="nonexistent_model:latest")
    with pytest.raises(OllamaModelNotFoundError) as exc_info:
        client.generate("test prompt")
    assert "not found in Ollama" in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_ollama_timeout_error(mock_urlopen):
    import socket
    mock_urlopen.side_effect = urllib.error.URLError(socket.timeout("timed out"))

    client = OllamaClient(timeout=10)
    with pytest.raises(OllamaTimeoutError) as exc_info:
        client.generate("test prompt")
    assert "timed out after 10s" in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_ollama_empty_response_error(mock_urlopen):
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps({"response": ""}).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    client = OllamaClient()
    with pytest.raises(OllamaResponseError) as exc_info:
        client.generate("test prompt")
    assert "empty response" in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_ollama_health_check(mock_urlopen):
    mock_response = MagicMock()
    mock_response.status = 200
    mock_urlopen.return_value.__enter__.return_value = mock_response

    client = OllamaClient()
    assert client.check_health() is True

    mock_urlopen.side_effect = Exception("error")
    assert client.check_health() is False
