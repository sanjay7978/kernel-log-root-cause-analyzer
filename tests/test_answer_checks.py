"""
Unit tests for modules/answer_checks.py and how the checks reach the pipeline output.
"""

from unittest.mock import MagicMock

import pytest

from modules.analyzer import analyze_raw_content
from modules.answer_checks import (
    check_command,
    check_commands,
    evidence_in_log,
    find_unverified_evidence,
    locate_evidence,
)
from modules.llm_analyzer import LLMResponse, OllamaClient
from modules.ui_service import format_commands_list, format_evidence_list

LOG = (
    "[  120.450255] Out of memory: Killed process 3412 (stress-ng-vm) total-vm:3938048kB\n"
    "[   57.950125] usb 1-1: device not accepting address 4, error -71\n"
    "[   57.950201] usb 1-1: unable to enumerate USB device\n"
)


# ---------------------------------------------------------------- evidence

@pytest.mark.parametrize("quote", [
    "[  120.450255] Out of memory: Killed process 3412 (stress-ng-vm) total-vm:3938048kB",   # exact
    "Out of memory: Killed process 3412 (stress-ng-vm)",                                      # no timestamp, partial line
    "  out of   memory: killed process 3412  ",                                               # spacing and case
    "\"usb 1-1: unable to enumerate USB device\"",                                            # quoted
    "> 120.450255] Out of memory: Killed process 3412",                                       # damaged timestamp
    "usb 1-1: device not accepting address 4 ... error -71",                                  # shortened with ...
])
def test_quotes_that_are_in_the_log_are_verified(quote):
    assert evidence_in_log(quote, LOG) is True


@pytest.mark.parametrize("quote", [
    "usb 1-1: device descriptor read/64, error -110",   # plausible but not in this log
    "kernel BUG at mm/slub.c:4012!",                    # invented
    "error -71 ... usb 1-1: device not accepting",      # pieces in the wrong order
    "",
])
def test_quotes_that_are_not_in_the_log_are_flagged(quote):
    assert evidence_in_log(quote, LOG) is False


def test_find_unverified_evidence_returns_only_the_missing_lines():
    evidence = ["usb 1-1: unable to enumerate USB device", "kernel BUG at mm/slub.c:4012!"]
    assert find_unverified_evidence(evidence, LOG) == ["kernel BUG at mm/slub.c:4012!"]
    assert find_unverified_evidence([], LOG) == []


def test_locate_evidence_gives_the_line_number_of_each_quote():
    where = locate_evidence(
        ["usb 1-1: unable to enumerate USB device",       # last line
         "[  120.450255] Out of memory: Killed process",   # with its timestamp, first line
         "usb 1-1: device not accepting address 4 ... error -71",   # shortened with ...
         "kernel BUG at mm/slub.c:4012!"],                  # not in the log
        LOG,
    )
    assert where == {
        "usb 1-1: unable to enumerate USB device": 3,
        "[  120.450255] Out of memory: Killed process": 1,
        "usb 1-1: device not accepting address 4 ... error -71": 2,
    }


def test_locate_evidence_with_nothing_to_find():
    assert locate_evidence([], LOG) == {} and locate_evidence(["x"], "") == {}


# ---------------------------------------------------------------- commands

@pytest.mark.parametrize("command", [
    "dmesg | grep -i usb",
    "dmesg --time-format iso",
    "journalctl -k -b -1",
    "sudo smartctl -a /dev/sda",
    "lspci -v 2>/dev/null",
    "free -h && uname -r",
    "grep -i 'a|b' /var/log/syslog",          # a | inside quotes is not a pipe
    "dpkg -l | grep firmware",
    "ethtool -S eth0",
    "ip addr show",
    "systemctl status NetworkManager",
    "fsck -n /dev/sda2",
    "cat /sys/module/usbcore/parameters/autosuspend",
    "find /lib/firmware -name 'iwlwifi*'",
    "",
])
def test_read_only_commands_are_not_flagged(command):
    assert check_command(command) is None


@pytest.mark.parametrize("command, reason_part", [
    ("rm -rf /tmp/x", "deletes files"),
    ("dd if=/dev/zero of=/dev/sda", "writes raw data"),
    ("mkfs.ext4 /dev/sda1", "formats"),
    ("fsck -f /dev/sda2", "can modify the disk"),
    ("dmesg -C", "clears the kernel log"),
    ("echo 1 > /proc/sys/vm/drop_caches", "writes output to a file"),
    ("ethtool -s eth0 speed 10 duplex full", "network card settings"),
    ("ip link set eth0 down", "network configuration"),
    ("systemctl restart NetworkManager", "start, stop or change services"),
    ("find /lib/firmware -name '*.ucode' -delete", "delete files"),
    ("sysctl -w vm.swappiness=10", "kernel settings"),
    ("sed -i s/a/b/ /etc/fstab", "edits files"),
    ("apt install linux-firmware", "install or remove software"),
    ("fdisk /dev/sda", "partition table"),
    ("smartctl -t long /dev/sda", "self-test"),
    ("cat $(find / -name x)", "command substitution"),
    ("ls | xargs rm", "runs other commands"),
    ("sudo rm /etc/hosts", "deletes files"),
    ("foo --bar", "not on the list of known read-only commands"),
    ("dmesg | grep 'unterminated", "could not be parsed"),
])
def test_commands_that_change_the_system_or_are_unknown_are_flagged(command, reason_part):
    reason = check_command(command)
    assert reason is not None and reason_part in reason


def test_any_segment_of_a_pipeline_can_flag_it():
    assert check_command("dmesg | tee /tmp/out.txt") is not None
    assert check_command("dmesg && rm -rf /") is not None


def test_check_commands_maps_only_flagged_ones():
    result = check_commands(["dmesg", "fsck -f /dev/sda2", "lsblk"])
    assert list(result) == ["fsck -f /dev/sda2"]


# ---------------------------------------------------------------- pipeline and display

def _client(answer_json):
    client = MagicMock(spec=OllamaClient)
    client.model = "qwen3:8b"
    client.generate.return_value = LLMResponse(content=answer_json, model="qwen3:8b", eval_count=10,
                                               raw_payload={"prompt_eval_count": 500})
    return client


ANSWER = """{
    "root_cause": "USB enumeration failed", "subsystem": "usb", "severity": "MEDIUM",
    "evidence": ["usb 1-1: unable to enumerate USB device", "usb 1-1: invented line that is not logged"],
    "explanation": "x", "commands": ["lsusb", "fsck -f /dev/sda2"],
    "possible_fixes": ["try another port"], "confidence": "HIGH", "uncertainty": "none"
}"""


def test_pipeline_attaches_the_checks_to_the_analysis():
    result = analyze_raw_content(LOG, source_label="t", ollama_client=_client(ANSWER))
    a = result.analysis

    assert a.unverified_evidence == ["usb 1-1: invented line that is not logged"]
    assert list(a.command_warnings) == ["fsck -f /dev/sda2"]
    as_dict = a.to_dict()
    assert as_dict["unverified_evidence"] == a.unverified_evidence
    assert as_dict["command_warnings"] == a.command_warnings


def test_evidence_is_checked_against_the_lines_the_model_was_actually_sent():
    # The line exists in the original log but was cut to fit the context window
    lines = [f"[{i}.000] systemd[1]: routine message number {i}" for i in range(4000)]
    lines[2000] = "[2000.000] something error happened here"
    raw = "\n".join(lines) + "\n"
    answer = ANSWER.replace("usb 1-1: unable to enumerate USB device", "routine message number 1000")
    result = analyze_raw_content(raw, source_label="t", ollama_client=_client(answer))

    assert result.context_truncated is True
    assert "routine message number 1000" in result.analysis.unverified_evidence


def test_ui_marks_unverified_evidence_and_flagged_commands():
    evidence_html = format_evidence_list(["real line", "made up line"], ["made up line"])
    assert evidence_html.count("Not found in the log") == 1

    commands_html = format_commands_list(["lsusb", "rm -rf /"], {"rm -rf /": "'rm' deletes files"})
    assert commands_html.count("Review before running") == 1
    assert "deletes files" in commands_html


def test_ui_formatters_still_work_without_the_checks():
    assert "Not found in the log" not in format_evidence_list(["line"])
    assert "Review before running" not in format_commands_list(["lsusb"])
