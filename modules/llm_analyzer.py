"""
LLM Analyzer Module for Kernel Log Root Cause Analyzer (Phase 2).

Responsible solely for low-level HTTP communication with the local Ollama daemon.
Sends prompts, sets inference parameters (including format='json'), and returns
the raw LLM response along with inference performance metrics.

Uses only the Python standard library (urllib.request, json).
Contains NO diagnostic rules, prompt formatting, or UI rendering.
"""

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SECONDS,
)


class OllamaError(Exception):
    """Base exception for Ollama interactions."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when Ollama daemon is unreachable on localhost."""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested model is not pulled in Ollama."""
    pass


class OllamaTimeoutError(OllamaError):
    """Raised when inference exceeds the timeout threshold."""
    pass


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an unexpected status code or empty payload."""
    pass


@dataclass
class LLMResponse:
    """Encapsulates the raw completion text and execution timing metadata."""
    content: str
    model: str
    total_duration_ms: float = 0.0
    load_duration_ms: float = 0.0
    eval_duration_ms: float = 0.0
    eval_count: int = 0
    raw_payload: Optional[Dict[str, Any]] = None


class OllamaClient:
    """HTTP Client for communicating with the local Ollama REST API."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT_SECONDS,
        temperature: float = OLLAMA_TEMPERATURE,
        num_ctx: int = OLLAMA_NUM_CTX,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx

    def check_health(self) -> bool:
        """
        Verify that Ollama daemon is reachable and responding.
        Returns True if reachable, False otherwise.
        """
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        force_json: bool = True,
    ) -> LLMResponse:
        """
        Send a prompt to Ollama and return the generated completion.

        Parameters:
            prompt: Text prompt to submit.
            model: Optional model override. Defaults to self.model.
            force_json: If True, instructs Ollama to constrain generation to valid JSON.

        Returns:
            LLMResponse: Container with content and timing metrics.

        Raises:
            OllamaConnectionError: If daemon is not running.
            OllamaModelNotFoundError: If model does not exist.
            OllamaTimeoutError: If request times out.
            OllamaResponseError: If server responds with an error or empty content.
        """
        if not prompt or not prompt.strip():
            raise OllamaResponseError("Prompt cannot be empty.")

        target_model = model or self.model
        endpoint = f"{self.base_url}/api/generate"

        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }

        if force_json:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status = resp.status
                if status != 200:
                    raise OllamaResponseError(f"Ollama server returned unexpected status HTTP {status}")
                raw_bytes = resp.read()
                resp_json = json.loads(raw_bytes.decode("utf-8"))

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise OllamaModelNotFoundError(
                    f"Model '{target_model}' not found in Ollama. Pull it using: ollama pull {target_model}"
                ) from exc
            raise OllamaResponseError(f"Ollama HTTP error {exc.code}: {exc.reason}") from exc

        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise OllamaTimeoutError(
                    f"Ollama inference timed out after {self.timeout}s on model '{target_model}'."
                ) from exc
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Ensure the Ollama daemon is running (`ollama serve`). Details: {exc.reason}"
            ) from exc

        except (socket.timeout, TimeoutError) as exc:
            raise OllamaTimeoutError(
                f"Ollama inference timed out after {self.timeout}s on model '{target_model}'."
            ) from exc

        except json.JSONDecodeError as exc:
            raise OllamaResponseError(f"Malformed JSON response received from Ollama API: {exc}") from exc

        content = resp_json.get("response", "")
        if not content and not resp_json.get("thinking"):
            raise OllamaResponseError(f"Ollama returned empty response for model '{target_model}'.")

        # Extract timing metrics (Ollama returns durations in nanoseconds)
        total_ns = resp_json.get("total_duration", 0)
        load_ns = resp_json.get("load_duration", 0)
        eval_ns = resp_json.get("eval_duration", 0)
        eval_count = resp_json.get("eval_count", 0)

        return LLMResponse(
            content=content,
            model=target_model,
            total_duration_ms=round(total_ns / 1_000_000, 2),
            load_duration_ms=round(load_ns / 1_000_000, 2),
            eval_duration_ms=round(eval_ns / 1_000_000, 2),
            eval_count=eval_count,
            raw_payload=resp_json,
        )


def query_ollama(
    prompt: str,
    model: Optional[str] = None,
    timeout: int = OLLAMA_TIMEOUT_SECONDS,
) -> str:
    """
    Convenience function to send a prompt to Ollama and return the content string.
    """
    client = OllamaClient(model=model or OLLAMA_MODEL, timeout=timeout)
    response = client.generate(prompt=prompt)
    return response.content
