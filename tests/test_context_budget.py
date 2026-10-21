"""
Unit tests for modules/context_budget.py and its use in the analysis pipeline.

Checks that long logs are reduced on our terms (instructions kept, warnings kept
with context, gaps marked and counted) instead of being cut silently by Ollama.
"""

from unittest.mock import MagicMock

from modules.analyzer import analyze_raw_content, prepare_log_for_prompt
from modules.context_budget import fit_to_budget, log_char_budget
from modules.llm_analyzer import LLMResponse, OllamaClient

VALID_ANSWER = """{
    "root_cause": "OOM killer terminated stress-ng", "subsystem": "memory", "severity": "HIGH",
    "evidence": ["Out of memory: Killed process 3412"], "explanation": "x",
    "commands": ["free -h"], "possible_fixes": ["add swap"], "confidence": "HIGH", "uncertainty": "none"
}"""


def _noise(n, start=0):
    return [f"[{i + start:8.3f}] systemd[1]: routine service message number {i + start}" for i in range(n)]


def test_short_log_is_returned_unchanged():
    text = "\n".join(_noise(10)) + "\n"
    fit = fit_to_budget(text, max_chars=10_000)

    assert fit.truncated is False
    assert fit.text == text
    assert fit.omitted_lines == 0
    assert fit.kept_lines == fit.original_lines == 10


def test_long_log_is_reduced_to_the_budget():
    text = "\n".join(_noise(2000)) + "\n"
    fit = fit_to_budget(text, max_chars=5_000)

    assert fit.truncated is True
    assert len(fit.text) <= 5_000 + 200
    assert fit.kept_lines + fit.omitted_lines == fit.original_lines == 2000


def test_warning_lines_and_neighbours_survive_and_gaps_are_marked():
    lines = _noise(1000)
    lines[500] = "[ 500.000] EXT4-fs error (device sda2): deleted inode referenced"
    text = "\n".join(lines) + "\n"
    fit = fit_to_budget(text, max_chars=4_000)

    assert "EXT4-fs error" in fit.text
    # two lines of context on each side are kept with it
    assert "routine service message number 498" in fit.text
    assert "routine service message number 502" in fit.text
    assert "lines omitted ...]" in fit.text
    assert "routine service message number 100" not in fit.text


def test_earliest_and_latest_warnings_are_kept_and_the_middle_is_dropped():
    lines = _noise(3000)
    for i in range(0, 3000, 50):
        lines[i] = f"[{i:8.3f}] driver: failed to probe device {i}"
    text = "\n".join(lines) + "\n"
    fit = fit_to_budget(text, max_chars=6_000)

    assert "failed to probe device 0" in fit.text        # first event (likely the cause)
    assert "failed to probe device 2950" in fit.text     # last event (final state)
    assert "failed to probe device 1500" not in fit.text # middle is the first to go


def test_log_without_any_warnings_keeps_start_and_end():
    text = "\n".join(_noise(1000)) + "\n"
    fit = fit_to_budget(text, max_chars=3_000)

    assert fit.truncated is True
    assert "message number 0" in fit.text
    assert "message number 999" in fit.text


def test_one_enormous_line_cannot_blow_the_budget():
    text = "[ 1.000] kernel: error " + "x" * 200_000 + "\n" + "\n".join(_noise(50)) + "\n"
    fit = fit_to_budget(text, max_chars=3_000)

    assert len(fit.text) < 4_000


def test_omission_marker_uses_singular_for_one_line():
    lines = _noise(200)
    lines[100] = "[ 1.0] something failed"
    text = "\n".join(lines) + "\n"
    fit = fit_to_budget(text, max_chars=len(text) - 20)

    assert "1 lines omitted" not in fit.text


def test_log_char_budget_uses_half_of_the_context_window():
    # 16384 // 2 = 8192 prompt tokens, less 128 safety = 8064 tokens, at 2.0 chars per token, less overhead
    assert log_char_budget(16384, overhead_chars=3000, chars_per_token=2.0, safety_tokens=128) == 8064 * 2 - 3000
    assert log_char_budget(100, overhead_chars=10_000, chars_per_token=2.0, safety_tokens=128) == 0


def test_prepare_log_for_prompt_keeps_short_logs_whole_and_adds_no_note():
    text = "\n".join(_noise(30)) + "\n"
    fit, note = prepare_log_for_prompt(text)

    assert fit.truncated is False
    assert note is None


def test_pipeline_reports_truncation_and_keeps_instructions_in_the_prompt():
    lines = _noise(3000)
    lines[1500] = "[1500.000] Out of memory: Killed process 3412 (stress-ng-vm)"
    raw = "\n".join(lines) + "\n"

    client = MagicMock(spec=OllamaClient)
    client.model = "qwen3:8b"
    client.generate.return_value = LLMResponse(content=VALID_ANSWER, model="qwen3:8b", eval_count=50,
                                               raw_payload={"prompt_eval_count": 4000})
    result = analyze_raw_content(raw, source_label="long", ollama_client=client)

    assert result.context_truncated is True
    assert result.omitted_lines > 0
    assert result.sent_lines + result.omitted_lines == result.original_cleaned_lines == 3000
    assert result.ollama_truncated is False
    assert "Out of memory: Killed process" in result.cleaned_log_excerpt

    prompt = client.generate.call_args.kwargs["prompt"]
    assert "FACTUAL GROUNDING" in prompt                 # instructions intact
    assert "KERNEL LOG EVIDENCE TO ANALYZE" in prompt
    assert "lines were left out" in prompt               # the model is told about the gaps


def test_pipeline_flags_when_ollama_cuts_the_prompt_anyway():
    raw = "\n".join(_noise(10)) + "\n[ 1.0] kernel: error\n"
    client = MagicMock(spec=OllamaClient)
    client.model = "qwen3:8b"
    client.generate.return_value = LLMResponse(content=VALID_ANSWER, model="qwen3:8b", eval_count=50,
                                               raw_payload={"prompt_eval_count": 8194})
    result = analyze_raw_content(raw, source_label="short", ollama_client=client)

    assert result.context_truncated is False
    assert result.ollama_truncated is True


def test_ollama_request_sets_the_context_window(monkeypatch):
    import json
    import urllib.request
    from config import OLLAMA_NUM_CTX

    captured = {}

    class FakeResp:
        status = 200
        def read(self):
            return json.dumps({"response": "{}", "eval_count": 1}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    OllamaClient().generate("hello")

    assert captured["body"]["options"]["num_ctx"] == OLLAMA_NUM_CTX


def test_leftover_budget_is_used_for_more_context_around_warnings():
    lines = _noise(2000)
    lines[1000] = "[1000.000] EXT4-fs error (device sda2): deleted inode referenced"
    text = "\n".join(lines) + "\n"
    fit = fit_to_budget(text, max_chars=20_000)

    # far more than the two minimum neighbours are kept, but not the whole log
    assert fit.truncated is True
    assert "routine service message number 990" in fit.text
    assert "routine service message number 1010" in fit.text
    assert len(fit.text) > 15_000
    assert len(fit.text) <= 20_000 + 200


def test_all_chosen_context_is_kept_even_when_warnings_are_front_heavy():
    lines = _noise(1500)
    for i in (5, 10, 15, 20, 25):
        lines[i] = f"[{i}.000] usb 1-1: device descriptor read error {i}"
    text = "\n".join(lines) + "\n"
    fit = fit_to_budget(text, max_chars=8_000)

    for i in (5, 10, 15, 20, 25):
        assert f"read error {i}" in fit.text
