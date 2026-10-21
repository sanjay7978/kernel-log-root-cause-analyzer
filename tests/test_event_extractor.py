"""
Unit tests for modules/event_extractor.py: subsystem tags, events, bursts and the summary.
"""

from pathlib import Path

import pytest

from config import SAMPLE_LOGS_DIR
from modules.event_extractor import (
    extract_events,
    summarise_events,
    tag_subsystem,
)
from modules.preprocessor import preprocess_kernel_log


@pytest.mark.parametrize("message, expected", [
    ("usb 1-1: device descriptor read/64, error -110", "usb"),
    ("xhci_hcd 0000:00:14.0: host not responding", "usb"),
    ("ata1.00: failed command: READ FPDMA QUEUED", "storage"),
    ("blk_update_request: I/O error, dev sda, sector 4096", "storage"),
    ("EXT4-fs error (device sda2): deleted inode referenced", "filesystem"),
    ("Aborting journal on device sda2-8.", "filesystem"),
    ("Out of memory: Killed process 3412 (stress-ng-vm)", "memory"),
    ("NETDEV WATCHDOG: eth0 (e1000e): transmit queue 0 timed out", "network"),
    ("e1000e 0000:00:1f.6 eth0: NIC Link is Down", "network"),
    ("amdgpu 0000:03:00.0: amdgpu: GPU reset begin!", "gpu"),
    ("ACPI BIOS Error (bug): Could not resolve symbol", "acpi"),
    ("iwlwifi 0000:00:14.3: Direct firmware load for x.ucode failed with error -2", "driver"),
    ("Kernel panic - not syncing: Fatal exception", "panic"),
    ("Oops: 0000 [#1] PREEMPT SMP NOPTI", "panic"),
    ("systemd[1]: Started Daily apt download activities.", None),
])
def test_messages_are_tagged_with_the_subsystem_they_name(message, expected):
    assert tag_subsystem(message) == expected


def test_identical_repeats_become_one_entry_with_a_count_and_a_different_event_starts_a_new_one():
    log = (
        "[   50.218542] usb 1-1: device descriptor read/64, error -110\n"
        "[   50.454231] usb 1-1: device descriptor read/64, error -110\n"
        "[   50.690112] usb 1-1: device descriptor read/64, error -110\n"
        "[   51.162018] usb 1-1: reset high-speed USB device number 4 using xhci_hcd\n"
        "[   56.278234] usb 1-1: device descriptor read/64, error -110\n"
    )
    groups = summarise_events(log).groups

    assert [(g.count, g.first_line) for g in groups] == [(3, 1), (1, 4), (1, 5)]
    assert groups[0].first_time == 50.218542 and groups[0].last_time == 50.690112


def test_lines_with_different_error_codes_are_never_merged():
    log = (
        "[ 1.0] usb 1-1: device descriptor read/64, error -110\n"
        "[ 2.0] usb 1-1: device descriptor read/64, error -71\n"
    )
    assert [g.count for g in summarise_events(log).groups] == [1, 1]


def test_numbers_that_only_identify_a_file_or_address_do_not_split_a_burst():
    log = (
        "[ 4.1] iwlwifi 0000:00:14.3: Direct firmware load for iwlwifi-QuZ-a0-hr-b0-77.ucode failed with error -2\n"
        "[ 4.2] iwlwifi 0000:00:14.3: Direct firmware load for iwlwifi-QuZ-a0-hr-b0-76.ucode failed with error -2\n"
        "[ 4.3] iwlwifi 0000:00:14.3: Direct firmware load for iwlwifi-QuZ-a0-hr-b0-75.ucode failed with error -2\n"
    )
    assert [g.count for g in summarise_events(log).groups] == [3]


def test_lines_the_preprocessor_collapsed_are_counted_in_the_burst():
    raw = "".join(f"[ {50 + i * 0.2:.3f}] usb 1-1: device descriptor read/64, error -110\n" for i in range(5))
    cleaned = preprocess_kernel_log(raw).cleaned_text
    assert "repeated 4 times" in cleaned

    summary = summarise_events(cleaned)
    assert summary.total_events == 5
    assert summary.groups[0].count == 5
    assert summary.groups[0].last_time == pytest.approx(50.8)


def test_a_line_with_no_name_of_its_own_takes_the_subsystem_of_the_line_above():
    log = (
        "[ 1420.890128] NETDEV WATCHDOG: eth0 (e1000e): transmit queue 0 timed out\n"
        "[ 1420.890135] WARNING: CPU: 1 PID: 0 at net/sched/sch_generic.c:525 dev_watchdog+0x235/0x240\n"
    )
    assert [e.subsystem for e in extract_events(log)] == ["network", "network"]


def test_a_nameless_line_far_from_any_tagged_line_stays_other():
    log = (
        "[ 10.0] usb 1-1: device descriptor read/64, error -110\n"
        "[ 90.0] something failed in a service\n"
    )
    assert [e.subsystem for e in extract_events(log)] == ["usb", "other"]


def test_a_warning_that_merely_mentions_a_function_offset_is_still_an_event():
    log = "[ 1420.890135] WARNING: CPU: 1 PID: 0 at net/sched/sch_generic.c:525 dev_watchdog+0x235/0x240\n"
    assert len(extract_events(log)) == 1


def test_stack_frames_and_trace_headers_are_not_events():
    log = (
        "[ 68.214051] BUG: kernel NULL pointer dereference, address: 0000000000000028\n"
        "[ 68.214126] Call Trace:\n"
        "[ 68.214142]  ? page_fault_oops+0x99/0x1b0\n"
        "[ 68.214250] Kernel panic - not syncing: Fatal exception\n"
    )
    assert [e.line for e in extract_events(log)] == [1, 4]


def test_omission_markers_and_blank_lines_are_ignored_but_line_numbers_still_count_them():
    log = (
        "    [... 25 lines omitted ...]\n"
        "\n"
        "[ 5.0] usb 1-1: unable to enumerate USB device\n"
    )
    assert [e.line for e in extract_events(log)] == [3]


def test_summary_counts_and_times():
    log = (
        "[ 10.0] usb 1-1: device descriptor read/64, error -110\n"
        "[ 11.0] usb 1-1: unable to enumerate USB device\n"
        "[ 12.5] EXT4-fs error (device sda2): bad\n"
    )
    summary = summarise_events(log)

    assert summary.total_events == 3
    assert summary.subsystem_counts == {"usb": 2, "filesystem": 1}
    assert summary.first_event_time == 10.0
    assert summary.span_seconds == 2.5
    assert summary.to_dict()["timeline"][0]["subsystem"] == "usb"


def test_logs_without_seconds_still_give_a_timeline_in_line_order():
    log = (
        "Oct 16 21:34:48 host kernel: eth0: link is down\n"
        "Oct 16 21:34:49 host kernel: eth0: link is down\n"
    )
    summary = summarise_events(log)

    assert summary.total_events == 2 and len(summary.groups) == 1
    assert summary.first_event_time is None and summary.span_seconds is None


def test_a_log_with_nothing_notable_has_an_empty_timeline():
    summary = summarise_events("[ 1.0] Linux version 6.5.0\n[ 2.0] Command line: quiet\n")
    assert summary.groups == [] and summary.total_events == 0


@pytest.mark.parametrize("path", sorted(Path(SAMPLE_LOGS_DIR).rglob("*.log")), ids=lambda p: p.parent.name)
def test_every_sample_log_gets_a_timeline_whose_main_subsystem_is_its_own(path):
    summary = summarise_events(preprocess_kernel_log(path.read_text()).cleaned_text)

    assert summary.groups, "a failure log should have events"
    assert next(iter(summary.subsystem_counts)) == path.parent.name
