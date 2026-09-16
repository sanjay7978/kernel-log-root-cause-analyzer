# LLM-Driven Dynamic Kernel Log Root Cause Analyzer Using Lightweight On-Device Models

An intelligent Linux kernel log diagnostic system designed to analyze system incidents using lightweight, local on-device LLMs (such as Qwen3:8B running via Ollama).

---

## 📌 Project Overview & Objectives

Modern Linux kernel logs (`dmesg`, `journalctl -k`) are verbose and challenging to interpret during high-pressure incidents. When memory is exhausted, storage controllers reset, or drivers lock up, identifying the root cause requires deep kernel expertise.

This project provides an end-to-end automated pipeline that ingests raw kernel logs, sanitizes noise, and uses a **locally hosted lightweight LLM** to extract:
* **🎯 Probable Root Cause**
* **⚙️ Affected Subsystem** (Memory, Storage, USB, Network, GPU, Driver, Filesystem, ACPI, Panic, Other)
* **⚠️ Severity Level** (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) with distinct visual indicators
* **📜 Verifiable Log Evidence** (direct quotes and line citations from the log)
* **💡 Technical Explanation** (plain English deduction distinguishing facts from inference)
* **🛠️ Safe Linux Troubleshooting Commands** (read-only diagnostics prioritized)
* **🔧 Actionable Remediation / Fix Steps**
* **❓ Diagnostic Confidence & Uncertainty Boundaries**

---

## 🏗️ Unified Pipeline Architecture (Phases 1, 2 & 3A)

```text
       ┌── Stored File (CLI / Disk)
       │
       ├── Uploaded File (.log, .txt via Gradio)
       │
       └── Pasted Log Text (Gradio Textbox)
                ↓
          Input Validation & Bounds Checking (50 MB Safety Ceiling)
                ↓
       Phase 1: Preprocessor (ANSI Stripping, Call Trace Indentation, Flood Deduplication)
                ↓
       Phase 2: Prompt Builder (Diagnostic Persona, Strict Grounding, JSON Schema)
                ↓
       Local LLM: Qwen3:8B via Ollama (Zero-Shot On-Device Inference)
                ↓
       Phase 2: Response Parser (JSON Extraction, Thinking-Tag Removal, Schema Validation)
                ↓
       Validated Diagnostic Analysis Object
          ├── Terminal CLI Dashboard & JSON Output (app.py)
          └── Professional Gradio Web Interface (gradio_app.py)
```

> [!IMPORTANT]
> **Scope Notice:** Phase 3A is about **Productization** of stored, uploaded, and pasted logs. Live collection from a running Linux kernel (`dmesg`, `journalctl -k`) and continuous monitoring belong to **Phase 3B**.

---

## 📂 Project Structure

```text
kernel_log_root_cause_analyzer/
├── .gitignore
├── README.md                # Project documentation & usage instructions
├── config.py                # Paths, categories, Ollama model, and Gradio server settings
├── requirements.txt         # Dependencies (pytest, gradio)
├── app.py                   # Unified CLI application (Phase 1 & Phase 2 & Phase 3A launcher)
├── gradio_app.py            # Professional Gradio web dashboard (Phase 3A)
│
├── modules/
│   ├── __init__.py
│   ├── log_loader.py        # Safe ingestion, validation, and error handling
│   ├── preprocessor.py      # Conservative cleaning, deduplication & noise metrics
│   ├── prompt_builder.py    # Diagnostic prompt construction & grounding rules
│   ├── llm_analyzer.py      # Zero-dependency HTTP client for local Ollama REST API
│   ├── result_parser.py     # JSON extraction, schema validation, and fallback parsing
│   ├── analyzer.py          # Unified orchestration service for in-memory and file inputs
│   └── ui_service.py        # Pure UI controller logic, badge formatting & error handling
│
├── sample_logs/             # Ground-truth evaluation dataset (real Linux logs)
│   ├── metadata.json        # Incident catalog (category, origin source, description)
│   ├── acpi/
│   ├── driver/
│   ├── filesystem/
│   ├── gpu/
│   ├── memory/              # e.g., real_oom_incident.log
│   ├── network/
│   ├── panic/
│   ├── storage/
│   └── usb/                 # e.g., real_usb_descriptor_error.log
│
└── tests/
    ├── __init__.py
    ├── test_log_loader.py       # 10 loader unit tests
    ├── test_preprocessor.py     # 10 preprocessor unit tests
    ├── test_prompt_builder.py   # 5 prompt construction tests
    ├── test_llm_analyzer.py     # 7 mock Ollama HTTP client tests
    ├── test_result_parser.py    # 9 parser and validation tests
    ├── test_analyzer.py         # 1 orchestration pipeline test
    ├── test_phase3a_ui.py       # 17 UI service, input extraction & Gradio tests
    ├── test_integration.py      # 2 Phase 1 real-log pipeline tests
    └── test_phase2_live.py      # 2 live Ollama end-to-end tests
```

---

## ⚙️ Environment & Model Setup

### 1. Python Virtual Environment
```bash
# Navigate to the project root
cd /Users/sanjay/Desktop/kernel_log_root_cause_analyzer

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (pytest, gradio)
pip install -r requirements.txt
```

### 2. Local Ollama & Qwen3:8B Setup
Make sure the Ollama background daemon is running:
```bash
ollama serve
```
Verify the model is installed:
```bash
ollama list
```
If `qwen3:8b` is not installed:
```bash
ollama pull qwen3:8b
```

---

## 🖥️ Launching the Gradio Web Application (Phase 3A)

You can launch the web interface in two equivalent ways:

```bash
# Option 1: Direct launcher
python gradio_app.py

# Option 2: Unified CLI launcher
python app.py --ui
```

Once started, open your web browser at:
👉 **`http://127.0.0.1:7860`**

### Key UI Capabilities:
1. **📄 Upload Log File:** Drag-and-drop any text-based kernel log file (`.log`, `.txt`).
2. **📋 Paste Log Text:** Directly paste terminal output from `dmesg` or `journalctl -k`.
3. **⚡ 1-Click Sample Incident Loaders:** Click **"Load Real Memory OOM Log"** or **"Load Real USB Timeout Log"** to instantly test the system with real-world failure logs without hunting for files.
4. **🔍 Evidence vs. Inference Distinction:**
   * **Evidence:** Factual lines directly quoted from the log (terminal-styled callout).
   * **Explanation:** Inferences explaining why the kernel reacted this way.
   * **Uncertainty & Limitations:** Explicitly documents what cannot be proven from this log alone.
5. **📊 Technical Performance Accordion:** Inspect input line counts, noise reduction percentage, exact execution timings (preprocessing, prompt construction, Ollama inference, and parsing), along with the sanitized log text.

---

## ⌨️ Command Line Interface (CLI Backward Compatibility)

The existing Phase 2 CLI remains 100% backward compatible:

```bash
# Analyze a stored log file (Full Phase 2 Pipeline):
python app.py sample_logs/memory/real_oom_incident.log

# Analyze a USB hardware failure log:
python app.py sample_logs/usb/real_usb_descriptor_error.log

# Export output as raw structured JSON:
python app.py sample_logs/memory/real_oom_incident.log --json

# Save analysis report to file:
python app.py sample_logs/memory/real_oom_incident.log --save-report report.json

# Phase 1 mode: Preprocessing & noise metrics only (no LLM call):
python app.py sample_logs/memory/real_oom_incident.log --preprocess-only
```

---

## 🔒 Security & Safety Principles

1. **Untrusted Input:** Uploaded files and pasted text are treated as untrusted strings. They are sanitized, validated against size boundaries (50 MB limit), and never executed.
2. **Zero Shell Execution:** Recommended troubleshooting commands are displayed strictly as informational guidance. The application **never executes commands automatically** via subshells or system calls.
3. **100% Private & Air-Gapped:** Zero data leaves your machine. All inference occurs locally on your Apple Silicon / unified memory hardware via Ollama.

---

## 🧪 Automated Testing

The project maintains a comprehensive automated test suite with **61 offline tests** and **2 live tests**:

### Fast Offline Test Suite (Runs in ~1.2s without requiring Ollama):
```bash
pytest -v --ignore=tests/test_phase2_live.py
```

### Live Ollama Integration Test Suite (Evaluates real Qwen3:8B on real incidents):
```bash
pytest -v tests/test_phase2_live.py
```

### Test Breakdown (63 Total Passing Tests):
* **Log Loader (10 tests):** Boundary checks, missing files, directories, empty files, size ceilings, permissions, and non-UTF-8 decoding.
* **Preprocessor (10 tests):** ANSI stripping, duplicate collapsing, Call Trace indentation preservation, and noise metrics.
* **Prompt Builder (5 tests):** Grounding constraints, schema requirements, context injection, and determinism.
* **LLM Analyzer (7 tests):** HTTP status handling, mock completions, 404 model handling, timeout detection, daemon health checks.
* **Result Parser (9 tests):** Valid JSON, markdown fences, thinking-tag removal, malformed JSON recovery, severity normalization, list type coercion.
* **Orchestration & Integration (3 tests):** Unified analysis on real and mock logs.
* **Phase 3A UI Service & Gradio (17 tests):** Input extraction, pasted text handling, file upload handling, size validation, UI formatting, badge generation, error card generation, and Gradio Blocks hierarchy.
* **Live Ollama Tests (2 tests):** Authentic end-to-end OOM and USB incident analyses using the live local Qwen model.

---

## ⚠️ Important Limitations & Roadmap

* **Diagnostic Assistant, Not Infallible Oracle:** The analyzer provides probable root causes based strictly on visible log evidence. If an issue occurred before logging started or was overwritten in the cyclic ring buffer, the system will explicitly state this uncertainty.
* **Reserved for Phase 3B:**
  * Live kernel ring buffer streaming (`dmesg -w`, `journalctl -f`)
  * Subprocess capture on live Linux hosts
  * Multi-boot historical log analysis
