#!/usr/bin/env python3
"""
Gradio Web Application for Kernel Log Root Cause Analyzer.

A two-pane workbench: the log goes in on the left (paste, upload or live
capture), the diagnostic report comes out on the right. Analysis uses a
lightweight on-device LLM (Qwen3:8B via Ollama), so nothing leaves the machine.

The page is kept light on purpose:
  - the report is ONE HTML block (modules/ui_service.py), not a dozen widgets
  - clicking Analyse shows a "working" state at once, then swaps in the report
  - styling is plain CSS in modules/ui_theme.py (no web fonts, no scripts)
"""

import logging
import os

# Gradio sends anonymous usage statistics and checks its latest version online by default.
# This tool promises that nothing leaves the machine, so both are switched off before Gradio loads.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402  (must come after the environment variable above)

from config import (
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
    GRADIO_SHARE,
    OLLAMA_MODEL,
)
from modules.llm_analyzer import OllamaClient
from modules.ui_service import (
    analyze_prepared,
    load_sample_log_content,
    read_input,
    render_analysing,
    render_empty_state,
    render_header,
    render_live_status,
    run_live_ui_analysis,
)
from modules.ui_theme import CSS, build_theme

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kernel_log_analyzer.gradio")


SAMPLE_LOG_CHOICES = {
    "Memory: OOM killer (stress-ng)": ("memory", "real_oom_incident.log"),
    "USB: device descriptor timeout (-110)": ("usb", "real_usb_descriptor_error.log"),
    "Filesystem: EXT4 corruption, read-only remount": ("filesystem", "real_ext4_corruption.log"),
    "Storage: SATA read error (UNC)": ("storage", "real_ata_io_error.log"),
    "Network: transmit queue timeout (e1000e)": ("network", "real_netdev_watchdog_tx_timeout.log"),
    "GPU: AMDGPU ring timeout and reset": ("gpu", "real_amdgpu_ring_timeout.log"),
    "Driver: iwlwifi firmware missing": ("driver", "real_driver_probe_failure.log"),
    "Panic: NULL pointer dereference": ("panic", "real_kernel_panic_null_deref.log"),
    "ACPI: BIOS AML namespace error": ("acpi", "real_acpi_bios_error.log"),
}

PLACEHOLDER = "[  120.450123] stress-ng invoked oom-killer: gfp_mask=0x1100cca ...\n[  120.450255] Out of memory: Killed process 3412 (stress-ng-vm)"


def load_sample_by_choice(choice: str):
    """Populate the paste box from the chosen sample log. Returns (file, pasted_text, note)."""
    if not choice or choice not in SAMPLE_LOG_CHOICES:
        return None, gr.update(), ""
    cat, filename = SAMPLE_LOG_CHOICES[choice]
    try:
        content = load_sample_log_content(cat, filename)
        return None, content, f"Loaded sample: {choice} (sample_logs/{cat}/{filename})"
    except Exception as exc:
        return None, "", f"Failed to load sample: {exc}"


def clear_inputs_and_outputs():
    """Reset every input and the report. Order matches the outputs wired in create_ui()."""
    return (
        None,                  # file upload
        "",                    # pasted text
        "",                    # sample note
        None,                  # sample picker
        render_empty_state(),  # report
        "",                    # cleaned log
    )


def _header_with_status() -> str:
    """Header with the live Ollama status (checked once, when the page opens)."""
    return render_header(OLLAMA_MODEL, OllamaClient().check_health())


def create_ui():
    """Build the Gradio Blocks UI."""
    with gr.Blocks(title="Kernel Log Root Cause Analyzer", analytics_enabled=False) as demo:
        header = gr.HTML(render_header(OLLAMA_MODEL, None))

        with gr.Row(equal_height=False):
            # ---------------- input pane ----------------
            with gr.Column(scale=5, min_width=340, elem_classes="kx-sticky"):
                gr.HTML('<div class="kx-label kx-pane-title"><i>01</i>Input</div>')

                with gr.Tabs() as input_tabs:
                    with gr.TabItem("Paste", id="paste"):
                        pasted_log_input = gr.Textbox(
                            show_label=False,
                            placeholder=PLACEHOLDER,
                            lines=14,
                            max_lines=14,
                        )
                    with gr.TabItem("Upload", id="upload"):
                        log_file_input = gr.File(
                            label="Kernel log file (.log or .txt, up to 50 MB)",
                            file_count="single",
                            type="filepath",
                        )
                    with gr.TabItem("Live capture", id="live"):
                        gr.HTML(render_live_status())
                        live_collector_radio = gr.Radio(
                            choices=["auto", "dmesg", "journalctl"],
                            value="auto",
                            label="Collector",
                            info="auto uses dmesg for the current boot and journalctl for earlier boots",
                        )
                        with gr.Row():
                            boot_offset_input = gr.Number(
                                value=0, label="Boot", precision=0,
                                info="0 current, -1 previous (after a crash)",
                            )
                            live_lines_slider = gr.Slider(
                                minimum=100, maximum=5000, value=1000, step=100,
                                label="Lines", info="most recent lines to read",
                            )
                        analyze_live_btn = gr.Button("Capture and analyse", variant="primary")

                with gr.Row():
                    analyze_btn = gr.Button("Analyse", variant="primary", scale=3)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)

                sample_dropdown = gr.Dropdown(
                    choices=list(SAMPLE_LOG_CHOICES),
                    value=None,
                    label="Load a sample incident",
                    info="Puts one of the bundled logs in the Paste tab",
                )
                sample_note = gr.Markdown(elem_classes="kx-note-line")

            # ---------------- report pane ----------------
            with gr.Column(scale=7, min_width=420):
                gr.HTML('<div class="kx-label kx-pane-title"><i>02</i>Report</div>')
                report_out = gr.HTML(render_empty_state())
                with gr.Accordion("Log sent to the model", open=False):
                    cleaned_log_view = gr.Textbox(
                        show_label=False, lines=12, max_lines=20, interactive=False,
                        placeholder="The cleaned log appears here after an analysis.",
                    )

        # ============================================================
        # Events
        # ============================================================
        outputs = [report_out, cleaned_log_view]

        def analyse_input(file_obj, pasted_txt):
            """Check the input, show 'analysing' at once, then the report."""
            raw, label, error = read_input(file_obj, pasted_txt)
            if error:
                yield error, ""
                return
            yield render_analysing(label), ""
            yield analyze_prepared(raw, label)

        def analyse_live(collector, boot_offset, lines):
            yield render_analysing("live capture"), ""
            yield run_live_ui_analysis(collector, int(boot_offset or 0), int(lines))

        def pick_sample(choice):
            file_value, text_value, note = load_sample_by_choice(choice)
            switch = gr.Tabs(selected="paste") if choice in SAMPLE_LOG_CHOICES else gr.update()
            return file_value, text_value, note, switch

        analyze_btn.click(
            analyse_input, [log_file_input, pasted_log_input], outputs,
            show_progress="hidden", trigger_mode="once",
        )
        analyze_live_btn.click(
            analyse_live, [live_collector_radio, boot_offset_input, live_lines_slider], outputs,
            show_progress="hidden", trigger_mode="once",
        )
        sample_dropdown.change(
            pick_sample, [sample_dropdown], [log_file_input, pasted_log_input, sample_note, input_tabs],
            show_progress="hidden",
        )
        clear_btn.click(
            clear_inputs_and_outputs, [],
            [log_file_input, pasted_log_input, sample_note, sample_dropdown, report_out, cleaned_log_view],
            show_progress="hidden",
        )
        demo.load(_header_with_status, None, header, show_progress="hidden")

    return demo


def launch(server_name: str = GRADIO_SERVER_NAME, server_port: int = GRADIO_SERVER_PORT, share: bool = GRADIO_SHARE):
    """Launch the Gradio web interface."""
    demo = create_ui()
    logger.info("Starting Gradio web server on http://%s:%d", server_name, server_port)
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        theme=build_theme(),
        css=CSS,
        footer_links=[],
    )


if __name__ == "__main__":
    launch()
