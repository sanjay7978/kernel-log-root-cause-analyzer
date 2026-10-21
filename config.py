"""
Configuration settings for Kernel Log Root Cause Analyzer (Phase 1, 2 & 3A).

Centralizes path definitions, environment constants, Ollama settings,
Gradio web server configurations, and processing limits. All paths are anchored
using pathlib.Path to ensure cross-platform compatibility between macOS development
and Linux execution environments.
"""

import os
from pathlib import Path

# Project Root Directory
BASE_DIR: Path = Path(__file__).resolve().parent

# Sample Logs Directory
SAMPLE_LOGS_DIR: Path = BASE_DIR / "sample_logs"

# Supported Subsystem Categories for Sample Datasets and Diagnosis
CATEGORIES: tuple[str, ...] = (
    "usb",
    "filesystem",
    "memory",
    "network",
    "gpu",
    "driver",
    "storage",
    "acpi",
    "panic",
)

# Safety thresholds for file ingestion & pasted text
# Protects against runaway memory usage if a multi-gigabyte log dump is provided
MAX_LOG_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB safety limit

# Informational / Candidate Evidence Patterns
# NOTE: These tokens are used solely for statistical counting and diagnostic
# evidence highlighting. They MUST NOT be used for root-cause classification logic.
EVIDENCE_MARKERS: tuple[str, ...] = (
    "ERROR",
    "WARN",
    "WARNING",
    "failed",
    "failure",
    "timeout",
    "critical",
    "panic",
    "BUG",
    "Oops",
    "Call Trace",
    "I/O error",
    "segfault",
    "out of memory",
    "killed process",
)

# ============================================================
# Phase 2: Local LLM (Ollama) Settings
# ============================================================

# Ollama REST API Base URL
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Selected Primary Local Qwen Model
# Configurable via environment variable; defaults to installed qwen3:8b
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")

# Request timeout in seconds (allows adequate time for local GPU/Unified memory inference).
# Long prompts take a while to read before the first token, so this is generous.
OLLAMA_TIMEOUT_SECONDS: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))

# Context window requested from Ollama, in tokens. Ollama accepts only about half
# of it as prompt (the rest is left for the answer), so the prompt budget below is
# num_ctx // 2. Ollama's own default is 4096, which leaves room for only ~2,000
# tokens, roughly 60 lines of log.
OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "16384"))

# Prompt budgeting. Kernel logs cost about 2.2-3.5 characters per token; we plan
# with the low end so the estimate errs on the side of keeping the prompt short.
CHARS_PER_TOKEN: float = float(os.getenv("CHARS_PER_TOKEN", "2.2"))
PROMPT_SAFETY_TOKENS: int = int(os.getenv("PROMPT_SAFETY_TOKENS", "128"))

# Generation temperature: lower temperature (0.1) reduces hallucination and ensures factual consistency
OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

# Ollama sometimes answers a long generation with an HTTP 500 or drops the connection,
# with nothing in its own log. The request can safely be repeated, so retry a few times.
OLLAMA_MAX_RETRIES: int = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
OLLAMA_RETRY_DELAY_SECONDS: float = float(os.getenv("OLLAMA_RETRY_DELAY_SECONDS", "3"))

# Allowed severity levels
VALID_SEVERITIES: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

# ============================================================
# Phase 3A: Gradio UI Web Server Settings
# ============================================================

GRADIO_SERVER_NAME: str = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_SERVER_PORT: int = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE: bool = os.getenv("GRADIO_SHARE", "False").lower() in ("true", "1", "yes")

# ============================================================
# Phase 3B: Live Linux Kernel Collection Settings
# ============================================================

DEFAULT_LIVE_LOG_LINES: int = int(os.getenv("DEFAULT_LIVE_LOG_LINES", "1000"))
SUPPORTED_LIVE_COLLECTORS: tuple[str, ...] = ("auto", "dmesg", "journalctl")
LIVE_COLLECTION_TIMEOUT_SECONDS: int = int(os.getenv("LIVE_COLLECTION_TIMEOUT_SECONDS", "15"))
