# LLM-Driven Dynamic Kernel Log Root Cause Analyzer Using Lightweight On-Device Models

An intelligent Linux kernel log diagnostic system designed to analyze system incidents using lightweight, local on-device LLMs (such as Qwen3:8B running via Ollama).

---

## 📌 Project Overview & Objectives

Modern Linux kernel logs (`dmesg`, `journalctl -k`) are verbose and challenging to interpret during high-pressure incidents. When memory is exhausted, storage controllers reset, or drivers lock up, identifying the root cause requires deep kernel expertise.

This project provides an end-to-end automated pipeline that ingests raw kernel logs (from stored files, direct paste, or live Linux collection), sanitizes noise, and uses a **locally hosted lightweight LLM** to extract:
* **🎯 Probable Root Cause**
* **⚙️ Affected Subsystem** (Memory, Storage, USB, Network, GPU, Driver, Filesystem, ACPI, Panic, Other)
* **⚠️ Severity Level** (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) with distinct visual indicators
* **📜 Verifiable Log Evidence** (direct quotes and line citations from the log)
* **💡 Technical Explanation** (plain English deduction distinguishing facts from inference)
* **🛠️ Safe Linux Troubleshooting Commands** (read-only diagnostics prioritized)
* **🔧 Actionable Remediation / Fix Steps**
* **❓ Diagnostic Confidence & Uncertainty Boundaries**

---

## 🏗️ Unified Pipeline Architecture (Phases 1, 2, 3A & 3B)

```text
       ┌── Stored File (CLI / Disk)
       │
       ├── Uploaded File (.log, .txt via Gradio)
       │
       ├── Pasted Log Text (Gradio Textbox)
       │
       └── Live Linux Capture (dmesg / journalctl -k on host or Ubuntu VM)
                ↓
          Input Validation & Safety Constraints (50 MB Limit, Non-Destructive Subprocess)
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

> [!NOTE]
> **Live Linux Collection:** Live kernel ring buffer capture uses non-destructive, safe read utilities (`dmesg`, `journalctl -k`) with strictly isolated arguments (`shell=False`). When running on macOS or Windows development hosts, capability detection automatically alerts the user and guides them to use the Ubuntu VM or stored sample logs.

---

## 📂 Project Structure

```text
kernel_log_root_cause_analyzer/
├── .gitignore
├── README.md                # Project documentation & usage instructions
├── config.py                # Paths, categories, Ollama model, live settings, and Gradio server settings
├── requirements.txt         # Dependencies (pytest, gradio)
├── app.py                   # Unified CLI application (Phases 1, 2, 3A & 3B)
├── gradio_app.py            # Professional Gradio web dashboard (Phases 3A & 3B)
│
├── modules/
│   ├── __init__.py
│   ├── log_loader.py        # Safe ingestion, validation, and error handling
│   ├── preprocessor.py      # Conservative cleaning, deduplication & noise metrics
│   ├── prompt_builder.py    # Diagnostic prompt construction & grounding rules
│   ├── llm_analyzer.py      # Zero-dependency HTTP client for local Ollama REST API
│   ├── result_parser.py     # JSON extraction, schema validation, and fallback parsing
│   ├── live_collector.py    # Safe Linux live kernel log capture (dmesg, journalctl, capability detection)
│   ├── analyzer.py          # Unified orchestration service for in-memory, file, and live inputs
│   └── ui_service.py        # Pure UI controller logic, badge formatting & live capture handling
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
    ├── test_live_collector.py   # 23 live collector, subprocess mock & UI live tests
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

### 3. Long Logs and the Context Window
Ollama only reads about **half** of the configured context window as prompt (the other half is kept for the answer), and when a prompt is longer it silently drops the middle, which removes the instructions at the top. To avoid that, the analyzer fits the log into the model's budget *before* sending it:

* The instructions and JSON schema are never cut.
* Lines around warnings and errors are kept, with as many neighbouring lines as the budget allows; routine lines far from any warning are dropped first.
* The earliest events (usually the cause) and the latest events (the final state) are favoured over the middle of a long log.
* Every gap is marked `[... N lines omitted ...]`, the model is told about it, and the CLI and web UI show a warning with the number of lines sent.

Settings (environment variables):

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `OLLAMA_NUM_CTX` | `16384` | Context window requested from Ollama. Prompt budget is half of this. Lower it (e.g. `8192`) for less memory and faster runs on long logs, at the cost of seeing fewer lines. |
| `CHARS_PER_TOKEN` | `2.2` | Characters per token used to estimate the prompt size. Kernel logs cost 2.2-3.5, so the default errs on the safe side. |
| `OLLAMA_TIMEOUT_SECONDS` | `300` | Long prompts take a while to read before the first token. |

With the default, `qwen3:8b` uses about 7.8 GB of memory (5.9 GB at Ollama's own 4096 default). Short logs are not affected.

### 4. Answer Checks
Small models sometimes quote lines that are not in the log, or suggest commands that change the system. After the model answers, `modules/answer_checks.py` checks both and flags what it finds. Nothing is removed from the answer, and commands are never run.

* **Evidence:** each quoted line is looked up in the log that was actually sent to the model. Quotes that cannot be found are marked *Not found in the log* (CLI: `[NOT FOUND IN LOG]`). Differences in spacing, case, quotes, the leading timestamp, or a `...` in the middle are tolerated.
* **Commands:** each command is checked against a list of known read-only tools. Anything that could change the system (`rm`, `fsck` without `-n`, `dmesg -C`, `ethtool -s`, redirects into files, ...) or that is not on the list is marked *Review before running* with the reason (CLI: `[REVIEW FIRST: ...]`). The check does not validate a command's options, only whether it is safe to read from.
* The same information is in the JSON output as `unverified_evidence` and `command_warnings`.

---

## 🖥️ Launching the Gradio Web Application

You can launch the web interface in two equivalent ways:

```bash
# Option 1: Direct launcher
python gradio_app.py

# Option 2: Unified CLI launcher
python app.py --ui
```

Once started, open your web browser at:
👉 **`http://127.0.0.1:7860`**

### UI Tabs & Features:
1. **📄 Upload Log File:** Drag-and-drop any text-based kernel log file (`.log`, `.txt`).
2. **📋 Paste Log Text:** Directly paste terminal output from `dmesg` or `journalctl -k`.
3. **🐧 Live Linux Capture:**
   * **Host Capability Indicator:** Detects whether running on a native Linux host or macOS/Windows development machine.
   * **Collection Tool Selection:** `auto` (detects `journalctl` or `dmesg`), `dmesg`, or `journalctl`.
   * **Boot Offset Selection:** Current boot (`0`), previous boot (`-1` for post-crash analysis), or historical boots (`-2`, etc.).
   * **Line Count Slider:** Adjust between 100 and 5000 lines.
4. **⚡ 1-Click Sample Incident Loaders:** Click **"Load Real Memory OOM Log"** or **"Load Real USB Timeout Log"** to instantly test the system with real-world failure logs.
5. **🔍 Evidence vs. Inference Distinction:**
   * **Evidence:** Factual lines directly quoted from the log (terminal-styled callout).
   * **Explanation:** Inferences explaining why the kernel reacted this way.
   * **Uncertainty & Limitations:** Explicitly documents what cannot be proven from this log alone.
6. **📊 Technical Performance Accordion:** Inspect raw line counts, noise reduction percentage, exact execution timings (capture, preprocessing, prompt construction, Ollama inference, and parsing), along with the sanitized log text.

---

## ⌨️ Command Line Interface (CLI)

The CLI provides complete support for stored logs, live captures, and batch inspection:

### 1. Live Linux Kernel Log Analysis:
```bash
# Capture live kernel log using auto-detected collector:
python app.py --live

# Capture live log using dmesg specifically (latest 500 lines):
python app.py --live-dmesg --live-lines 500

# Capture previous boot kernel log using journalctl (post-crash investigation):
python app.py --live-journalctl --boot-offset -1

# Output live analysis as structured JSON:
python app.py --live --json
```

### 2. Stored File Analysis:
```bash
# Analyze a stored log file (Full Pipeline):
python app.py sample_logs/memory/real_oom_incident.log

# Analyze a USB hardware failure log:
python app.py sample_logs/usb/real_usb_descriptor_error.log

# Export output as raw structured JSON:
python app.py sample_logs/memory/real_oom_incident.log --json

# Save analysis report to file:
python app.py sample_logs/memory/real_oom_incident.log --save-report report.json

# Phase 1 mode: Preprocessing & noise metrics only (bypass LLM):
python app.py sample_logs/memory/real_oom_incident.log --preprocess-only
```

---

## 🔒 Security & Safety Principles

1. **Non-Destructive Execution:** All live collector commands (`dmesg`, `journalctl -k`) are read-only and invoked strictly with `shell=False` and isolated parameter arrays.
2. **Untrusted Input Safety:** Uploaded files and pasted text are treated as untrusted strings, bounded to 50 MB, and never executed as code.
3. **Informational Recommendations:** Diagnostic commands shown in the report are recommendations for the administrator and are **never executed automatically**.
4. **100% Private & Air-Gapped:** Zero data leaves your machine. All inference occurs locally on your hardware via Ollama.

---

## 🧪 Automated Testing

The project maintains a comprehensive automated test suite with **84 offline tests** and **2 live Ollama tests**:

### Fast Offline Test Suite (Runs in ~1.2s without requiring Ollama):
```bash
pytest -v --ignore=tests/test_phase2_live.py
```

### Live Ollama Integration Test Suite (Evaluates real Qwen3:8B on real incidents):
```bash
pytest -v tests/test_phase2_live.py
```

### Test Breakdown (86 Total Tests):
* **Live Collector (23 tests):** Platform detection, capability inspection, mocked `dmesg`/`journalctl` executions, ISO timestamp fallback, permission errors, timeout handling, CLI arguments, and UI live integration.
* **Phase 3A UI Service & Gradio (17 tests):** Input extraction, pasted text handling, file upload handling, size validation, UI formatting, badge generation, error card generation, and Gradio Blocks hierarchy.
* **Log Loader (10 tests):** Boundary checks, missing files, directories, empty files, size ceilings, permissions, and non-UTF-8 decoding.
* **Preprocessor (10 tests):** ANSI stripping, duplicate collapsing, Call Trace indentation preservation, and noise metrics.
* **Result Parser (9 tests):** Valid JSON, markdown fences, thinking-tag removal, malformed JSON recovery, severity normalization, list type coercion.
* **LLM Analyzer (7 tests):** HTTP status handling, mock completions, 404 model handling, timeout detection, daemon health checks.
* **Prompt Builder (5 tests):** Grounding constraints, schema requirements, context injection, and determinism.
* **Orchestration & Integration (3 tests):** Unified analysis on real and mock logs.
* **Live Ollama Tests (2 tests):** Authentic end-to-end OOM and USB incident analyses using the live local Qwen model.
