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


# ============================================================
# Retrying transient server failures
# ============================================================

def _ok_response():
    response = MagicMock()
    response.status = 200
    response.read.return_value = json.dumps({"response": '{"root_cause": "x"}', "eval_count": 5}).encode("utf-8")
    cm = MagicMock()
    cm.__enter__.return_value = response
    return cm


def _http_error(code):
    return urllib.error.HTTPError(url="http://localhost:11434/api/generate", code=code, msg="err", hdrs={}, fp=None)


@patch("time.sleep")
@patch("urllib.request.urlopen")
def test_http_500_is_retried_and_then_succeeds(mock_urlopen, mock_sleep):
    mock_urlopen.side_effect = [_http_error(500), _ok_response()]

    result = OllamaClient(retry_delay=2).generate("prompt")

    assert result.eval_count == 5
    assert mock_urlopen.call_count == 2
    mock_sleep.assert_called_once_with(2)


@patch("time.sleep")
@patch("urllib.request.urlopen")
def test_dropped_connection_is_retried(mock_urlopen, mock_sleep):
    import http.client
    mock_urlopen.side_effect = [http.client.RemoteDisconnected("closed"), ConnectionResetError(), _ok_response()]

    result = OllamaClient(max_retries=2).generate("prompt")

    assert result.eval_count == 5
    assert mock_urlopen.call_count == 3


@patch("time.sleep")
@patch("urllib.request.urlopen")
def test_server_error_that_keeps_happening_is_raised_after_the_retries(mock_urlopen, mock_sleep):
    mock_urlopen.side_effect = _http_error(500)

    with pytest.raises(OllamaResponseError) as exc_info:
        OllamaClient(max_retries=2).generate("prompt")

    assert "HTTP error 500" in str(exc_info.value)
    assert mock_urlopen.call_count == 3  # first attempt plus two retries


@patch("time.sleep")
@patch("urllib.request.urlopen")
def test_a_dropped_connection_that_keeps_happening_reports_a_connection_error(mock_urlopen, mock_sleep):
    import http.client
    mock_urlopen.side_effect = http.client.RemoteDisconnected("closed")

    with pytest.raises(OllamaConnectionError) as exc_info:
        OllamaClient(max_retries=1).generate("prompt")

    assert "closed the connection" in str(exc_info.value)
    assert mock_urlopen.call_count == 2


@patch("time.sleep")
@patch("urllib.request.urlopen")
def test_errors_that_will_not_fix_themselves_are_not_retried(mock_urlopen, mock_sleep):
    import socket

    # daemon not running, unknown model, client error, timeout
    for error, expected in [
        (urllib.error.URLError("Connection refused"), OllamaConnectionError),
        (_http_error(404), OllamaModelNotFoundError),
        (_http_error(400), OllamaResponseError),
        (urllib.error.URLError(socket.timeout("timed out")), OllamaTimeoutError),
    ]:
        mock_urlopen.reset_mock()
        mock_urlopen.side_effect = error
        with pytest.raises(expected):
            OllamaClient(max_retries=2).generate("prompt")
        assert mock_urlopen.call_count == 1
    mock_sleep.assert_not_called()


@patch("time.sleep")
@patch("urllib.request.urlopen")
def test_retries_can_be_turned_off(mock_urlopen, mock_sleep):
    mock_urlopen.side_effect = _http_error(503)

    with pytest.raises(OllamaResponseError):
        OllamaClient(max_retries=0).generate("prompt")

    assert mock_urlopen.call_count == 1
