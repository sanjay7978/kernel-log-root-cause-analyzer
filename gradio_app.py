#!/usr/bin/env python3
"""
Gradio Web Application for Kernel Log Root Cause Analyzer (Phase 3A).

Provides a graphical user interface for analyzing Linux kernel logs
using a lightweight on-device LLM (Qwen3:8B via Ollama).

Supports:
- Drag-and-drop log file uploads (.log, .txt)
- Direct log text pasting
- 1-click sample incident loading (OOM, USB)
- Evidence vs. Inference vs. Uncertainty visual distinction
- Processing performance metrics and cleaned log inspection
"""

import logging
from pathlib import Path
from typing import Optional

import gradio as gr

from config import (
    CATEGORIES,
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
    GRADIO_SHARE,
    OLLAMA_MODEL,
    SAMPLE_LOGS_DIR,
)
from modules.ui_service import (
    get_live_host_status_html,
    load_sample_log_content,
    run_live_ui_analysis,
    run_ui_analysis,
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kernel_log_analyzer.gradio")

# Custom CSS for modern enterprise diagnostic dashboard styling
CUSTOM_CSS = """
/* Container styling */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Header styling */
.header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #ffffff;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.header-title {
    font-size: 26px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 12px;
}

.header-subtitle {
    font-size: 14px;
    color: #94a3b8;
    margin-top: 6px;
    line-height: 1.5;
}

.model-badge {
    background: #1e293b;
    border: 1px solid #334155;
    color: #38bdf8;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin-top: 10px;
}

/* Diagnostic Card styling */
.diag-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
}

.diag-title {
    font-size: 15px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.root-cause-banner {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 5px solid #0284c7;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

.root-cause-text {
    font-size: 18px;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.4;
}

/* Callout info boxes */
.info-callout {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #475569;
    line-height: 1.5;
}
"""


SAMPLE_LOG_CHOICES = {
    "Memory: OOM Killer (stress-ng)": ("memory", "real_oom_incident.log"),
    "USB: Device Descriptor Timeout (-110)": ("usb", "real_usb_descriptor_error.log"),
    "Filesystem: EXT4 Metadata Corruption (Read-Only)": ("filesystem", "real_ext4_corruption.log"),
    "Storage: SATA / ATA Read I/O Media Error (UNC)": ("storage", "real_ata_io_error.log"),
    "Network: NETDEV Watchdog Transmit Queue Timeout": ("network", "real_netdev_watchdog_tx_timeout.log"),
    "GPU: AMDGPU Ring Timeout & Reset Recovery": ("gpu", "real_amdgpu_ring_timeout.log"),
    "Driver: Intel iwlwifi Missing Firmware Probe Failure": ("driver", "real_driver_probe_failure.log"),
    "Panic: Fatal Kernel NULL Pointer Dereference (#PF)": ("panic", "real_kernel_panic_null_deref.log"),
    "ACPI: BIOS AML Namespace Lookup Error (AE_NOT_FOUND)": ("acpi", "real_acpi_bios_error.log"),
}


def load_oom_example():
    """Helper to populate pasted text box with authentic OOM sample log."""
    try:
        content = load_sample_log_content("memory", "real_oom_incident.log")
        return None, content, "Loaded real Out-Of-Memory incident from sample_logs/memory/real_oom_incident.log"
    except Exception as exc:
        return None, "", f"Failed to load sample: {exc}"


def load_usb_example():
    """Helper to populate pasted text box with authentic USB error sample log."""
    try:
        content = load_sample_log_content("usb", "real_usb_descriptor_error.log")
        return None, content, "Loaded real USB descriptor timeout incident from sample_logs/usb/real_usb_descriptor_error.log"
    except Exception as exc:
        return None, "", f"Failed to load sample: {exc}"


def load_sample_by_choice(choice: str):
    """Helper to populate pasted text box from chosen sample log."""
    if not choice or choice not in SAMPLE_LOG_CHOICES:
        return None, gr.update(), ""
    cat, filename = SAMPLE_LOG_CHOICES[choice]
    try:
        content = load_sample_log_content(cat, filename)
        return None, content, f"Loaded real {choice} incident from sample_logs/{cat}/{filename}"
    except Exception as exc:
        return None, "", f"Failed to load sample: {exc}"


def clear_inputs_and_outputs():
    """Reset all UI inputs and outputs."""
    return (
        None,   # file upload
        "",     # pasted text
        "",     # status info
        "",     # status html
        "",     # root cause
        "",     # subsystem
        "",     # severity
        "",     # confidence
        "",     # evidence
        "",     # explanation
        "",     # commands
        "",     # fixes
        "",     # uncertainty
        "",     # metrics table
        "",     # preprocessed log view
    )


def create_ui():
    """Build the Gradio Blocks UI hierarchy."""
    with gr.Blocks(title="Kernel Log Root Cause Analyzer") as demo:
        # 1. Header Banner
        gr.HTML(
            f"""
            <div class="header-box">
                <div class="header-title">
                    <span>🐧 Kernel Log Root Cause Analyzer</span>
                </div>
                <div class="header-subtitle">
                    Intelligent Linux kernel log diagnostics powered by local, lightweight on-device LLMs.
                    Ingests raw <code>dmesg</code> / <code>journalctl</code> outputs, sanitizes noise, and isolates probable root causes with factual evidence.
                </div>
                <div>
                    <span class="model-badge">⚡ Engine: Ollama / {OLLAMA_MODEL}</span>
                    <span class="model-badge" style="background:#0f172a; color:#a7f3d0; border-color:#065f46;">🔒 100% Private (On-Device Inference)</span>
                    <span class="model-badge" style="background:#0f172a; color:#fde047; border-color:#854d0e;">📋 Phase 3B: Stored, Pasted & Live Linux Capture</span>
                </div>
            </div>
            """
        )

        # 2. Main Input Section
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### 📥 Step 1: Provide Kernel Log")

                with gr.Tabs():
                    with gr.TabItem("📄 Upload Log File", id="tab_file"):
                        log_file_input = gr.File(
                            label="Upload kernel log file (.log, .txt, dmesg capture)",
                            file_count="single",
                            type="filepath",
                        )
                        gr.Markdown(
                            "<small style='color:#64748b;'>Accepts plain text files containing <code>dmesg</code> or <code>journalctl -k</code> captures up to 50 MB.</small>"
                        )

                    with gr.TabItem("📋 Paste Log Text", id="tab_paste"):
                        pasted_log_input = gr.Textbox(
                            label="Paste raw Linux kernel log text",
                            placeholder="[  120.450123] kernel: Out of memory: Killed process 3412 (stress-ng)...",
                            lines=10,
                            max_lines=25,
                        )

                    with gr.TabItem("🐧 Live Linux Capture", id="tab_live"):
                        gr.HTML(value=get_live_host_status_html())
                        with gr.Row():
                            live_collector_radio = gr.Radio(
                                choices=["auto", "dmesg", "journalctl"],
                                value="auto",
                                label="Collection Utility",
                                info="auto selects journalctl if systemd is active, dmesg otherwise",
                            )
                        with gr.Row():
                            boot_offset_input = gr.Number(
                                value=0,
                                label="Boot Offset (journalctl -b)",
                                precision=0,
                                info="0 = current boot, -1 = previous boot (post-crash investigation)",
                            )
                            live_lines_slider = gr.Slider(
                                minimum=100,
                                maximum=5000,
                                value=1000,
                                step=100,
                                label="Max Lines to Capture",
                                info="Recent lines to read from ring buffer",
                            )
                        analyze_live_btn = gr.Button("🐧 Capture & Analyze Live Kernel Log", variant="primary")

                # Action and Sample Buttons
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Run Root Cause Analysis (File/Pasted)", variant="primary", scale=2)
                    clear_btn = gr.Button("↺ Clear All", variant="secondary", scale=1)

                # Quick sample incident loader
                with gr.Row():
                    sample_info = gr.Markdown("<small style='color:#475569;'>⚡ Quick test with real Linux incidents:</small>")
                with gr.Row():
                    load_oom_btn = gr.Button("📂 Load Real Memory OOM Log", size="sm")
                    load_usb_btn = gr.Button("📂 Load Real USB Timeout Log", size="sm")
                with gr.Row():
                    sample_dropdown = gr.Dropdown(
                        choices=["-- Select a sample log to load --"] + list(SAMPLE_LOG_CHOICES.keys()),
                        value="-- Select a sample log to load --",
                        label="⚡ Or Select from All 9 Real Subsystem Incidents",
                        interactive=True,
                    )

            # Status and Architecture Notes Column
            with gr.Column(scale=3):
                gr.Markdown("### ℹ️ Analysis Architecture")
                gr.HTML(
                    """
                    <div class="info-callout">
                        <strong>Pipeline Mechanics:</strong>
                        <ol style="padding-left:18px; margin:6px 0;">
                            <li><strong>Validation:</strong> Ensures non-empty, safe text bounds.</li>
                            <li><strong>Preprocessing:</strong> Strips ANSI codes, collapses repeated line bursts, preserves call traces.</li>
                            <li><strong>Grounding:</strong> Enforces factual citations without hallucinating hardware causes.</li>
                            <li><strong>Inference:</strong> Evaluates evidence using your local Qwen model.</li>
                        </ol>
                        <hr style="border:none; border-top:1px solid #cbd5e1; margin:10px 0;">
                        <span style="color:#0369a1; font-weight:600;">🔒 Safe Execution:</span>
                        Linux commands shown in reports are diagnostic recommendations. They are <strong>never executed automatically</strong>.
                    </div>
                    """
                )
                status_feedback = gr.Markdown(visible=True)

        # 3. Status Notification Banner
        status_banner_html = gr.HTML()

        # 4. Results Section
        gr.Markdown("---")
        gr.Markdown("## 📋 Diagnostic Analysis Report")

        # Prominent Root Cause Headline & High-Level Metadata
        with gr.Group():
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("#### 🎯 Probable Root Cause")
                    root_cause_out = gr.Markdown(
                        value="*Awaiting log submission. Upload or paste a log above and click 'Run Root Cause Analysis'.*"
                    )
                with gr.Column(scale=1):
                    gr.Markdown("#### ⚙️ Affected Subsystem")
                    subsystem_badge_out = gr.HTML(value="<span style='color:#94a3b8;'>—</span>")
                    gr.Markdown("#### ⚠️ Severity")
                    severity_badge_out = gr.HTML(value="<span style='color:#94a3b8;'>—</span>")
                    gr.Markdown("#### 🔍 Confidence")
                    confidence_badge_out = gr.HTML(value="<span style='color:#94a3b8;'>—</span>")

        # Two-column layout for detailed diagnostics
        with gr.Row():
            # Left Column: Evidence & Technical Explanation
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📜 Verifiable Log Evidence (Facts)")
                    gr.Markdown(
                        "<small style='color:#64748b;'>Direct line citations extracted from the submitted log supporting the diagnosis:</small>"
                    )
                    evidence_out = gr.HTML()

                with gr.Group():
                    gr.Markdown("### 💡 Technical Explanation (Inference)")
                    gr.Markdown(
                        "<small style='color:#64748b;'>Deductions explaining why the kernel reacted this way and what the symptoms indicate:</small>"
                    )
                    explanation_out = gr.Markdown()

                with gr.Group():
                    gr.Markdown("### ❓ Diagnostic Uncertainty & Boundaries")
                    gr.Markdown(
                        "<small style='color:#64748b;'>Information that cannot be established from this log alone:</small>"
                    )
                    uncertainty_out = gr.Markdown()

            # Right Column: Actionable Recommendations
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🛠️ Recommended Troubleshooting Commands")
                    gr.Markdown(
                        "<small style='color:#64748b;'>Safe, read-only Linux diagnostic commands to investigate the issue:</small>"
                    )
                    commands_out = gr.HTML()

                with gr.Group():
                    gr.Markdown("### 🔧 Possible Fixes & Remediation")
                    gr.Markdown(
                        "<small style='color:#64748b;'>Potential steps to resolve the underlying condition:</small>"
                    )
                    fixes_out = gr.Markdown()

        # 5. Technical Details & Performance Metrics (Accordion)
        with gr.Accordion("📊 Pipeline Performance Metrics & Preprocessed Log", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    metrics_table_out = gr.Markdown()
                with gr.Column(scale=1):
                    gr.Markdown("##### Preprocessed Evidence Log")
                    preprocessed_log_view = gr.Textbox(
                        label="Cleaned log fed to the model",
                        lines=12,
                        interactive=False,
                    )

        # ============================================================
        # Event Callbacks
        # ============================================================

        # Wrapper function to update the preprocessed log view as well
        def handle_analysis(file_obj, pasted_txt):
            outputs = run_ui_analysis(file_obj, pasted_txt)
            # Extract cleaned log excerpt if analysis was successful
            cleaned_excerpt = ""
            try:
                raw, src = modules.ui_service.extract_input_content(file_obj, pasted_txt)
                cleaned_excerpt = modules.preprocessor.preprocess_kernel_log(raw).cleaned_text
            except Exception:
                cleaned_excerpt = "(No preprocessed log available)"

            return (
                *outputs,
                cleaned_excerpt,
            )

        # Analyze button click
        analyze_btn.click(
            fn=handle_analysis,
            inputs=[log_file_input, pasted_log_input],
            outputs=[
                status_banner_html,
                root_cause_out,
                subsystem_badge_out,
                severity_badge_out,
                confidence_badge_out,
                evidence_out,
                explanation_out,
                commands_out,
                fixes_out,
                uncertainty_out,
                metrics_table_out,
                preprocessed_log_view,
            ],
            show_progress="full",
        )

        # Quick sample loading buttons
        load_oom_btn.click(
            fn=load_oom_example,
            inputs=[],
            outputs=[log_file_input, pasted_log_input, status_feedback],
        )

        load_usb_btn.click(
            fn=load_usb_example,
            inputs=[],
            outputs=[log_file_input, pasted_log_input, status_feedback],
        )

        sample_dropdown.change(
            fn=load_sample_by_choice,
            inputs=[sample_dropdown],
            outputs=[log_file_input, pasted_log_input, status_feedback],
        )

        # Live Linux analyze button click
        analyze_live_btn.click(
            fn=run_live_ui_analysis,
            inputs=[live_collector_radio, boot_offset_input, live_lines_slider],
            outputs=[
                status_banner_html,
                root_cause_out,
                subsystem_badge_out,
                severity_badge_out,
                confidence_badge_out,
                evidence_out,
                explanation_out,
                commands_out,
                fixes_out,
                uncertainty_out,
                metrics_table_out,
                preprocessed_log_view,
            ],
            show_progress="full",
        )

        # Clear button
        clear_btn.click(
            fn=clear_inputs_and_outputs,
            inputs=[],
            outputs=[
                log_file_input,
                pasted_log_input,
                status_feedback,
                status_banner_html,
                root_cause_out,
                subsystem_badge_out,
                severity_badge_out,
                confidence_badge_out,
                evidence_out,
                explanation_out,
                commands_out,
                fixes_out,
                uncertainty_out,
                metrics_table_out,
                preprocessed_log_view,
            ],
        )

    return demo


def launch(server_name: str = GRADIO_SERVER_NAME, server_port: int = GRADIO_SERVER_PORT, share: bool = GRADIO_SHARE):
    """Launch the Gradio web interface."""
    import modules.preprocessor
    import modules.ui_service

    demo = create_ui()
    logger.info("Starting Gradio web server on http://%s:%d", server_name, server_port)
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    import modules.preprocessor
    import modules.ui_service
    launch()
