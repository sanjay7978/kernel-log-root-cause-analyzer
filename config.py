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

# Request timeout in seconds (allows adequate time for local GPU/Unified memory inference)
OLLAMA_TIMEOUT_SECONDS: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

# Generation temperature: lower temperature (0.1) reduces hallucination and ensures factual consistency
OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

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
