"""
Theme for the Gradio web interface: a dark, console-style look.

Everything here is plain CSS and Gradio theme variables. There are no web
fonts, no CDN links and no scripts, so the page loads instantly, works offline
and sends nothing to anyone, which matches the tool's "everything stays on this
machine" design.

The colours live in one place (PALETTE). Severity colours always come with a
text label in the markup, so meaning never depends on colour alone.
"""

import inspect

import gradio as gr

PALETTE = {
    "bg": "#0e1116",        # page
    "panel": "#151a21",     # cards and inputs
    "panel_2": "#1b222b",   # raised: hover, table header
    "line": "#252c36",      # borders
    "line_2": "#323b48",    # stronger borders, focus rings
    "text": "#d6dde6",
    "muted": "#8b96a5",
    "accent": "#4fb3a5",    # the one chrome colour
    "accent_ink": "#0b1512",
    "crit": "#f0605d",
    "high": "#ee8f3f",
    "med": "#e6c34a",
    "low": "#5fbf84",
    "warn": "#e0a030",
}

FONT_SANS = ["system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue", "sans-serif"]
FONT_MONO = ["ui-monospace", "SF Mono", "Menlo", "Consolas", "Liberation Mono", "monospace"]


def build_theme() -> gr.themes.Base:
    """Gradio theme with every variable pinned to the dark palette (light and dark mode look the same)."""
    p = PALETTE
    base = gr.themes.Base(
        primary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.slate,
        font=FONT_SANS,
        font_mono=FONT_MONO,
        radius_size=gr.themes.sizes.radius_sm,
        spacing_size=gr.themes.sizes.spacing_sm,
        text_size=gr.themes.sizes.text_md,
    )
    values = {
        "body_background_fill": p["bg"],
        "body_text_color": p["text"],
        "body_text_color_subdued": p["muted"],
        "background_fill_primary": p["panel"],
        "background_fill_secondary": p["bg"],
        "border_color_primary": p["line"],
        "border_color_accent": p["accent"],
        "color_accent_soft": p["panel_2"],
        "shadow_drop": "none",
        "shadow_drop_lg": "none",
        "shadow_inset": "none",
        "block_background_fill": p["panel"],
        "block_border_color": p["line"],
        "block_border_width": "1px",
        "block_shadow": "none",
        "block_label_background_fill": "transparent",
        "block_label_text_color": p["muted"],
        "block_title_text_color": p["muted"],
        "block_info_text_color": p["muted"],
        "input_background_fill": p["bg"],
        "input_border_color": p["line"],
        "input_border_color_focus": p["accent"],
        "input_shadow": "none",
        "input_shadow_focus": "none",
        "input_placeholder_color": "#5b6574",
        "button_primary_background_fill": p["accent"],
        "button_primary_background_fill_hover": "#63c4b6",
        "button_primary_text_color": p["accent_ink"],
        "button_primary_border_color": p["accent"],
        "button_secondary_background_fill": "transparent",
        "button_secondary_background_fill_hover": p["panel_2"],
        "button_secondary_text_color": p["text"],
        "button_secondary_border_color": p["line_2"],
        "link_text_color": p["accent"],
        "link_text_color_hover": "#63c4b6",
        "link_text_color_active": p["accent"],
        "link_text_color_visited": p["accent"],
        "table_even_background_fill": p["panel"],
        "table_odd_background_fill": p["bg"],
        "table_border_color": p["line"],
        "panel_background_fill": p["panel"],
        "panel_border_color": p["line"],
    }
    # same values for the dark variants, so the page never flips to a light look
    values.update({f"{key}_dark": value for key, value in list(values.items())})
    # keep only variables this version of Gradio knows about, so a rename never breaks start-up
    known = set(inspect.signature(gr.themes.Base.set).parameters)
    return base.set(**{key: value for key, value in values.items() if key in known})


def _css() -> str:
    p = PALETTE
    return """
:root, .dark, body {
  --kx-bg: %(bg)s; --kx-panel: %(panel)s; --kx-panel-2: %(panel_2)s;
  --kx-line: %(line)s; --kx-line-2: %(line_2)s; --kx-text: %(text)s; --kx-muted: %(muted)s;
  --kx-accent: %(accent)s; --kx-crit: %(crit)s; --kx-high: %(high)s; --kx-med: %(med)s;
  --kx-low: %(low)s; --kx-warn: %(warn)s;
  --kx-mono: ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  color-scheme: dark;
}
html, body { background: var(--kx-bg) !important; }

/* ---- page frame ------------------------------------------------------ */
.gradio-container { max-width: 1360px !important; padding: 0 20px 40px !important; background: var(--kx-bg) !important; }
footer, .built-with, .show-api { display: none !important; }
.gradio-container .prose, .gradio-container .block { box-shadow: none !important; }

/* ---- header ---------------------------------------------------------- */
.kx-top { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
  padding: 18px 0 14px; border-bottom: 1px solid var(--kx-line); margin-bottom: 20px; }
.kx-brand { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.kx-mark { font-family: var(--kx-mono); font-size: 17px; font-weight: 600; letter-spacing: 0; color: var(--kx-text); }
.kx-mark b { color: var(--kx-accent); font-weight: 600; }
.kx-tag { color: var(--kx-muted); font-size: 13px; }
.kx-env { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; font-family: var(--kx-mono); font-size: 12px; color: var(--kx-muted); }
.kx-env span { white-space: nowrap; }
.kx-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%%; margin-right: 6px; background: var(--kx-muted); }
.kx-dot.kx-ok { background: var(--kx-low); }
.kx-dot.kx-bad { background: var(--kx-crit); }

/* ---- section labels --------------------------------------------------- */
.kx-label { font-family: var(--kx-mono); font-size: 11px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--kx-muted); margin: 0 0 10px; }
.kx-label i { font-style: normal; color: var(--kx-accent); margin-right: 8px; }
.kx-pane-title { margin-bottom: 12px !important; }

/* ---- gradio widgets --------------------------------------------------- */
.gradio-container .tab-wrapper, .gradio-container .tab-container { border-color: var(--kx-line) !important; }
.gradio-container button[role="tab"] { font-family: var(--kx-mono); font-size: 12px; letter-spacing: .04em; color: var(--kx-muted);
  background: transparent !important; border: 0 !important; padding: 8px 14px; }
.gradio-container button[role="tab"][aria-selected="true"] { color: var(--kx-text); box-shadow: inset 0 -2px 0 var(--kx-accent); }
.gradio-container textarea, .gradio-container input[type="text"], .gradio-container input[type="number"] {
  font-family: var(--kx-mono) !important; font-size: 12.5px !important; line-height: 1.55 !important; }
.gradio-container button.primary { font-weight: 600; letter-spacing: .01em; }
.gradio-container button.secondary { border: 1px solid var(--kx-line-2) !important; }
.gradio-container button.lg, .gradio-container button.md { min-height: 40px; }
.kx-note-line { color: var(--kx-muted); font-size: 12.5px; margin: 6px 0 0; min-height: 18px; }
.kx-note-line p { margin: 0 !important; color: var(--kx-muted) !important; font-size: 12.5px !important; }

/* dropdown list: solid, above everything, same look as the rest */
.gradio-container ul.options { background: var(--kx-panel) !important; border: 1px solid var(--kx-line-2) !important; border-radius: 4px !important;
  z-index: 60 !important; box-shadow: 0 8px 24px rgba(0,0,0,.5) !important; max-height: 280px; }
.gradio-container ul.options li { color: var(--kx-text) !important; font-size: 13px; padding: 7px 12px !important; }
.gradio-container ul.options li:hover, .gradio-container ul.options li.selected, .gradio-container ul.options li[aria-selected="true"] { background: var(--kx-panel-2) !important; }
.gradio-container ul.options li.selected { box-shadow: inset 3px 0 0 var(--kx-accent); }

/* the input pane stays in view while a long report scrolls (wide screens only) */
@media (min-width: 861px) { .kx-sticky { position: sticky; top: 16px; align-self: flex-start; } }

/* accordion that holds the cleaned log */
.gradio-container .label-wrap { font-family: var(--kx-mono); font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--kx-muted); }

/* ---- report ----------------------------------------------------------- */
.kx-report { color: var(--kx-text); }
.kx-report section { margin: 0 0 22px; }
.kx-report p { margin: 0 0 8px; line-height: 1.6; }
.kx-cause { font-size: 21px; line-height: 1.35; font-weight: 600; margin: 0 0 14px; letter-spacing: -.005em; color: var(--kx-text); }
.kx-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
.kx-chip { display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--kx-line-2); border-radius: 4px;
  padding: 3px 9px; font-family: var(--kx-mono); font-size: 12px; letter-spacing: .04em; color: var(--kx-text); background: var(--kx-bg); }
.kx-chip .kx-k { color: var(--kx-muted); text-transform: lowercase; letter-spacing: 0; }
.kx-sev::before { content: ""; width: 8px; height: 8px; background: var(--kx-muted); display: inline-block; }
.kx-sev-critical { border-color: var(--kx-crit); } .kx-sev-critical::before { background: var(--kx-crit); }
.kx-sev-high { border-color: var(--kx-high); }     .kx-sev-high::before { background: var(--kx-high); }
.kx-sev-medium { border-color: var(--kx-med); }    .kx-sev-medium::before { background: var(--kx-med); }
.kx-sev-low { border-color: var(--kx-low); }       .kx-sev-low::before { background: var(--kx-low); }
.kx-meta { font-family: var(--kx-mono); font-size: 12px; color: var(--kx-muted); }

.kx-log, .kx-cmds { border: 1px solid var(--kx-line); border-radius: 4px; background: var(--kx-bg); overflow: hidden; }
.kx-row { display: flex; gap: 12px; align-items: flex-start; padding: 7px 12px; border-top: 1px solid var(--kx-line);
  font-family: var(--kx-mono); font-size: 12.5px; line-height: 1.55; border-left: 3px solid transparent; }
.kx-row:first-child { border-top: 0; }
.kx-ln, .kx-prompt { color: #5b6574; user-select: none; min-width: 1.4em; text-align: right; }
.kx-prompt { text-align: left; min-width: 0; color: var(--kx-accent); }
.kx-row code { font-family: inherit; background: transparent; padding: 0; color: var(--kx-text); white-space: pre-wrap; word-break: break-word; flex: 1; border: 0; }
.kx-cmds .kx-row code { user-select: all; cursor: copy; }
.kx-row.kx-warn { border-left-color: var(--kx-warn); background: rgba(224,160,48,.06); }
.kx-note { display: block; width: 100%%; margin-top: 2px; font-family: system-ui, sans-serif; font-size: 12px; color: var(--kx-warn); }
.kx-row .kx-body { flex: 1; min-width: 0; }

.kx-text { white-space: pre-wrap; color: var(--kx-text); }
.kx-fixes { margin: 0; padding-left: 20px; } .kx-fixes li { margin: 0 0 6px; line-height: 1.55; padding-left: 4px; }
.kx-fixes li::marker { color: var(--kx-accent); font-family: var(--kx-mono); }
.kx-empty { color: var(--kx-muted); margin: 0; }

.kx-warnbox { border: 1px solid var(--kx-warn); border-left-width: 3px; border-radius: 4px; padding: 10px 14px; margin: 0 0 20px;
  background: rgba(224,160,48,.07); font-size: 13.5px; line-height: 1.55; }
.kx-warnbox b { color: var(--kx-warn); font-weight: 600; }

.kx-error { border: 1px solid var(--kx-crit); border-left-width: 3px; border-radius: 4px; padding: 14px 16px; background: rgba(240,96,93,.07); }
.kx-error-title { font-weight: 600; color: var(--kx-crit); margin-bottom: 6px; }
.kx-error p { margin: 0 0 6px; line-height: 1.55; }
.kx-hint { color: var(--kx-muted); font-size: 13px; }

.kx-details { border-top: 1px solid var(--kx-line); padding-top: 12px; }
.kx-details summary { cursor: pointer; font-family: var(--kx-mono); font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--kx-muted); list-style: none; }
.kx-details summary::-webkit-details-marker { display: none; }
.kx-details summary::before { content: "+"; color: var(--kx-accent); margin-right: 8px; display: inline-block; width: 10px; }
.kx-details[open] summary::before { content: "\\2212"; }
.kx-metrics { display: grid; grid-template-columns: max-content 1fr; gap: 6px 22px; margin: 14px 0 0; font-family: var(--kx-mono); font-size: 12.5px; }
.kx-metrics dt { color: var(--kx-muted); } .kx-metrics dd { margin: 0; color: var(--kx-text); }

/* ---- empty and busy states -------------------------------------------- */
.kx-start { border: 1px dashed var(--kx-line-2); border-radius: 4px; padding: 26px 24px; color: var(--kx-muted); }
.kx-start .kx-lead { color: var(--kx-text); font-size: 15px; margin: 0 0 10px; }
.kx-start .kx-flow { font-family: var(--kx-mono); font-size: 12px; margin: 0 0 14px; color: var(--kx-accent); }
.kx-start p { margin: 0 0 6px; line-height: 1.6; font-size: 13.5px; }
.kx-busy { border: 1px solid var(--kx-line); border-radius: 4px; padding: 22px 24px; background: var(--kx-panel); }
.kx-busy .kx-lead { margin: 0 0 4px; font-size: 15px; } .kx-busy .kx-sub { margin: 0 0 16px; color: var(--kx-muted); font-size: 13px; }
.kx-bar { height: 3px; background: var(--kx-line); overflow: hidden; border-radius: 2px; position: relative; }
.kx-bar::after { content: ""; position: absolute; inset: 0; width: 35%%; background: var(--kx-accent); transform: translateX(-100%%);
  animation: kx-slide 1.4s ease-in-out infinite; }
@keyframes kx-slide { to { transform: translateX(300%%); } }
@media (prefers-reduced-motion: reduce) { .kx-bar::after { animation: none; width: 100%%; transform: none; opacity: .5; } }

/* ---- live tab status --------------------------------------------------- */
.kx-live { border: 1px solid var(--kx-line); border-left: 3px solid var(--kx-muted); border-radius: 4px; padding: 10px 14px; margin: 0 0 12px; font-size: 13px; line-height: 1.55; color: var(--kx-muted); }
.kx-live b { color: var(--kx-text); font-weight: 600; } .kx-live.kx-on { border-left-color: var(--kx-low); }
.kx-live code { font-family: var(--kx-mono); color: var(--kx-text); }

@media (max-width: 860px) { .gradio-container { padding: 0 12px 28px !important; } .kx-cause { font-size: 18px; } }
""" % p


CSS = _css()
