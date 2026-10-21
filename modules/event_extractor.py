"""
Event Extractor Module for Kernel Log Root Cause Analyzer.

Turns a cleaned kernel log into a short, ordered story of what happened:

  1. every line is tagged with the kernel subsystem it belongs to, from driver
     names and message patterns (usb, storage, filesystem, memory, network,
     gpu, acpi, driver, panic). Lines that carry no name of their own, such as
     the indented lines of a call trace, take the subsystem of the line above;
  2. lines that matter (warnings, errors, resets, remounts, link changes) are
     picked out as events, in time order;
  3. consecutive events that say the same thing (numbers aside) are grouped into
     one burst, so eight identical timeouts read as one entry with a count,
     while a different event in between starts a new entry;
  4. a summary is computed: events per subsystem, the time of the first event,
     and how long the log spans.

This is a reading aid for the person using the tool. The tags are a heuristic,
they are NOT used to decide the root cause, and the model never sees them.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple

from modules.preprocessor import find_candidate_evidence, split_timestamp

# A line with no subsystem of its own takes the one above it, if that line is
# close enough (seconds, or lines when the log has no seconds).
INHERIT_GAP_SECONDS = 2.0
INHERIT_GAP_LINES = 6

_OMISSION_LINE = re.compile(r"^\s*\[\.\.\. \d+ lines? omitted \.\.\.\]\s*$")
# "[... preceding message repeated 3 times, last at [   50.926040] ...]" from the preprocessor
_REPEAT_NOTE = re.compile(r"^\s*\[\.\.\. preceding message repeated (\d+) times?(?:, last at (\[[^\]]*\]))? \.\.\.\]\s*$")
_SECONDS = re.compile(r"\[\s*(\d+\.\d+)\]")

# First match wins. Patterns are matched against the lower-cased message.
_RULES: List[Tuple[str, Pattern[str]]] = [(name, re.compile(pattern)) for name, pattern in [
    ("panic", r"kernel panic|not syncing|^oops\b|\boops: \d|bug: kernel (null|paging)|kernel bug at|general protection fault"),
    ("memory", r"out of memory|oom-kill|oom_reaper|invoked oom-killer|killed process|page allocation failure|mem-info"),
    ("usb", r"\busb \d|\bxhci|\behci|\buhci|usbcore|usb-storage|new (high|full|low|super)-speed usb"),
    ("gpu", r"amdgpu|\[drm|nouveau|\bi915\b|radeon|nvidia|gpu reset|ring \w+ timeout|vram"),
    ("acpi", r"^acpi|\bacpi (bios )?(error|exception)|acpi:|ae_not_found|ae_already_exists"),
    ("filesystem", r"ext4-fs|\bext4:|\bxfs\b|btrfs|f2fs|jbd2|remounting filesystem|aborting journal|\bvfs:|fat-fs|ntfs"),
    ("storage", r"\bata\d|\bsd \d+:\d+|\[sd[a-z]+\]|\bnvme|\bscsi|blk_update_request|buffer i/o error|i/o error, dev|libata|medium error|\bnvme"),
    ("driver", r"firmware|probe of .* failed|direct firmware load|failed to load|iwlwifi|\bmodule\b.*(unknown|failed)"),
    ("network", r"netdev watchdog|\beth\d|\benp\w+|\bwlan\d|e1000|r8169|\bigb\b|ixgbe|nic link|link is (up|down)|transmit queue"),
]]

# A line that is only a stack frame, such as "? page_fault_oops+0x99/0x1b0"; frames are detail, not events.
_STACK_FRAME = re.compile(r"\??\s*[\w.]+\+0x[0-9a-f]+/0x[0-9a-f]+(\s*\[\w+\])?")

# Lines that are not warnings by keyword but matter for the story.
_STATE_CHANGE = re.compile(
    r"\breset\b|remounting|read-only|link is down|nic link is down|unable to|not responding|not accepting|"
    r"aborting|timed out|timeout|disabled|failed|killed|frozen|exception emask|oom-kill"
)
# Indented or trace-style lines inherit the subsystem of the line above.
_CONTINUATION = re.compile(r"^(call trace|rip:|rsp:|code:|<task>|</task>|<irq>|</irq>|---\[|\?|cpu:|hardware name|modules linked|mem-info|node \d|tasks state|\[ +pid)")


@dataclass
class Event:
    """One notable line."""
    line: int                 # 1-based line number in the text that was analysed
    time: Optional[float]     # seconds, when the log has them
    subsystem: str
    text: str                 # the message without its timestamp
    repeats: int = 0          # extra copies the preprocessor collapsed into this line
    last_time: Optional[float] = None


@dataclass
class EventGroup:
    """A burst of events from one subsystem."""
    first_line: int
    last_line: int
    first_time: Optional[float]
    last_time: Optional[float]
    subsystem: str
    count: int
    text: str                 # the first message of the burst


@dataclass
class EventSummary:
    groups: List[EventGroup] = field(default_factory=list)
    subsystem_counts: Dict[str, int] = field(default_factory=dict)
    total_events: int = 0
    first_event_time: Optional[float] = None
    span_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_events": self.total_events,
            "subsystem_counts": self.subsystem_counts,
            "first_event_time": self.first_event_time,
            "span_seconds": self.span_seconds,
            "timeline": [
                {"first_line": g.first_line, "last_line": g.last_line, "first_time": g.first_time,
                 "last_time": g.last_time, "subsystem": g.subsystem, "count": g.count, "text": g.text}
                for g in self.groups
            ],
        }


def tag_subsystem(message: str) -> Optional[str]:
    """The subsystem a message names, or None when it names none."""
    lowered = message.lower().strip()
    for subsystem, pattern in _RULES:
        if pattern.search(lowered):
            return subsystem
    return None


def _seconds(timestamp: str) -> Optional[float]:
    match = _SECONDS.search(timestamp)
    return float(match.group(1)) if match else None


def _signature(message: str) -> str:
    """The message with its numbers masked, except error codes, so repeats can be recognised."""
    lowered = message.lower().strip()
    lowered = re.sub(r"0x[0-9a-f]+", "0x#", lowered)
    return re.sub(r"(?<!error )(?<!error -)(?<![a-z])\d+", "#", lowered)


def extract_events(log_text: str) -> List[Event]:
    """Tag every line, then keep the lines that tell the story, in order."""
    events: List[Event] = []
    current = "other"
    tagged_line = 0
    tagged_time: Optional[float] = None
    for number, line in enumerate(log_text.splitlines(), start=1):
        if not line.strip() or _OMISSION_LINE.match(line):
            continue

        repeat = _REPEAT_NOTE.match(line)
        if repeat:
            if events:
                events[-1].repeats += int(repeat.group(1))
                events[-1].last_time = _seconds(repeat.group(2) or "") or events[-1].last_time
            continue

        timestamp, message = split_timestamp(line)
        time = _seconds(timestamp)
        tag = tag_subsystem(message)
        is_continuation = line[:1] in (" ", "\t") or bool(_CONTINUATION.match(message.lower()))
        if tag is not None:
            current, tagged_line, tagged_time = tag, number, time
        else:
            near = (
                time - tagged_time <= INHERIT_GAP_SECONDS if time is not None and tagged_time is not None
                else number - tagged_line <= INHERIT_GAP_LINES
            )
            tag = current if (is_continuation or near) and current != "other" else "other"

        notable = find_candidate_evidence(message) or bool(_STATE_CHANGE.search(message.lower()))
        is_frame = bool(_STACK_FRAME.fullmatch(message.strip()))
        if notable and not is_frame and not message.lower().startswith("call trace"):
            events.append(Event(line=number, time=time, subsystem=tag, text=message.strip(), last_time=time))
    return events


def summarise_events(log_text: str) -> EventSummary:
    """Events, bursts and totals for a cleaned log."""
    events = extract_events(log_text)
    summary = EventSummary(total_events=sum(1 + e.repeats for e in events))
    if not events:
        return summary

    group: Optional[EventGroup] = None
    group_key = ("", "")
    for event in events:
        copies = 1 + event.repeats
        key = (event.subsystem, _signature(event.text))
        if group is not None and key == group_key:
            group.count += copies
            group.last_line = event.line
            group.last_time = event.last_time if event.last_time is not None else event.time
        else:
            group = EventGroup(event.line, event.line, event.time,
                               event.last_time if event.last_time is not None else event.time,
                               event.subsystem, copies, event.text)
            group_key = key
            summary.groups.append(group)
        summary.subsystem_counts[event.subsystem] = summary.subsystem_counts.get(event.subsystem, 0) + copies

    times = [t for e in events for t in (e.time, e.last_time) if t is not None]
    if times:
        summary.first_event_time = events[0].time
        summary.span_seconds = round(max(times) - min(times), 3)
    summary.subsystem_counts = dict(sorted(summary.subsystem_counts.items(), key=lambda kv: -kv[1]))
    return summary
