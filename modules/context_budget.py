"""
Context Budget Module for Kernel Log Root Cause Analyzer.

Keeps the prompt inside what the local model can actually read. Ollama only
accepts about half of the configured context window as prompt (the other half
is left for the answer) and, when a prompt is too long, silently drops the
middle of it. That removes the instructions at the top of our prompt, so the
log is reduced here first, on our terms:

  * the instructions and JSON schema are never cut,
  * lines around warnings and errors are kept, with as many neighbouring lines
    as the budget allows, and routine lines far from any warning are dropped first,
  * the earliest events (usually the cause) and the latest events (the final
    state) are favoured over the middle of a long log,
  * every gap is marked "[... N lines omitted ...]" and counted, so the user
    and the model both know that something was left out.

The warning markers are used only to decide what to keep when there is no
room for everything; they never classify the root cause.
"""

from dataclasses import dataclass
from typing import List, Tuple

from modules.preprocessor import find_candidate_evidence

# Lines kept on each side of a warning line.
CONTEXT_WINDOW_LINES = 2
# Share of the budget given to the start of the log; the rest goes to the end.
HEAD_SHARE = 0.6
# Characters reserved for the "[... N lines omitted ...]" markers.
MARKER_RESERVE_CHARS = 400
# Longest single line allowed once the log has to be reduced.
MAX_LINE_CHARS = 1000


@dataclass
class FitResult:
    """The log text to send to the model, and what was left out to fit it."""
    text: str
    truncated: bool
    original_lines: int
    kept_lines: int
    omitted_lines: int


def log_char_budget(num_ctx: int, overhead_chars: int, chars_per_token: float, safety_tokens: int) -> int:
    """
    Characters of log that fit in the prompt.

    Ollama accepts about num_ctx / 2 tokens of prompt, so that is the limit
    we plan for. chars_per_token is deliberately low (kernel logs are full of
    numbers and symbols, which cost more tokens than ordinary text).
    """
    prompt_tokens = max(num_ctx // 2 - safety_tokens, 0)
    return max(int(prompt_tokens * chars_per_token) - overhead_chars, 0)


def _omission_marker(count: int) -> str:
    noun = "line" if count == 1 else "lines"
    return f"    [... {count} {noun} omitted ...]"


def _line_cost(line: str) -> int:
    return len(line) + 1  # newline


def _merge_ranges(indices: List[int], total: int, window: int) -> List[Tuple[int, int]]:
    """Turn warning line numbers into merged (start, end) ranges with context."""
    ranges: List[Tuple[int, int]] = []
    for i in indices:
        start, end = max(i - window, 0), min(i + window, total - 1)
        if ranges and start <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    return ranges


def _take_from_start(lines: List[str], start: int, end: int, budget: int) -> Tuple[int, int]:
    """Lines start..end that fit in budget, taken from the start. Returns (last_index, cost)."""
    used = 0
    last = start - 1
    for i in range(start, end + 1):
        cost = _line_cost(lines[i])
        if used + cost > budget:
            break
        used += cost
        last = i
    return last, used


def _take_from_end(lines: List[str], start: int, end: int, budget: int) -> Tuple[int, int]:
    """Lines start..end that fit in budget, taken from the end. Returns (first_index, cost)."""
    used = 0
    first = end + 1
    for i in range(end, start - 1, -1):
        cost = _line_cost(lines[i])
        if used + cost > budget:
            break
        used += cost
        first = i
    return first, used


def _ranges_cost(lines: List[str], ranges: List[Tuple[int, int]]) -> int:
    return sum(_line_cost(lines[i]) for start, end in ranges for i in range(start, end + 1))


def _widest_ranges_within(lines: List[str], flagged: List[int], window: int, budget: int) -> List[Tuple[int, int]]:
    """
    Use leftover budget for more context: widen the window around the warning
    lines for as long as the kept lines still fit.
    """
    total = len(lines)
    best = _merge_ranges(flagged, total, window)
    if _ranges_cost(lines, best) > budget:
        return best  # even the minimum window is too much; the caller trims by priority
    low, high = window, total  # largest window whose ranges fit is somewhere in [low, high]
    while low < high:
        mid = (low + high + 1) // 2
        candidate = _merge_ranges(flagged, total, mid)
        if _ranges_cost(lines, candidate) <= budget:
            low, best = mid, candidate
        else:
            high = mid - 1
    return best


def fit_to_budget(text: str, max_chars: int, window: int = CONTEXT_WINDOW_LINES) -> FitResult:
    """
    Reduce cleaned log text to at most about max_chars, keeping the most useful lines.

    Text that already fits is returned unchanged.
    """
    lines = text.splitlines()
    total = len(lines)
    if len(text) <= max_chars:
        return FitResult(text=text, truncated=False, original_lines=total, kept_lines=total, omitted_lines=0)

    lines = [ln if len(ln) <= MAX_LINE_CHARS else ln[:MAX_LINE_CHARS] + " ..." for ln in lines]
    budget = max(max_chars - MARKER_RESERVE_CHARS, 0)
    head_budget = int(budget * HEAD_SHARE)
    tail_budget = budget - head_budget

    flagged = [i for i, ln in enumerate(lines) if find_candidate_evidence(ln)]
    ranges = _merge_ranges(flagged, total, window) if flagged else [(0, total - 1)]
    if not flagged:
        # Nothing stands out: keep the start and the end of the log.
        mid = total // 2
        ranges = [(0, mid - 1), (mid, total - 1)] if total > 1 else ranges
    else:
        ranges = _widest_ranges_within(lines, flagged, window, budget)
        if _ranges_cost(lines, ranges) <= budget:
            # Everything chosen fits, so keep all of it (no head/tail split needed).
            head_budget = tail_budget = budget

    # chosen[(start, end)] = the part of that range we keep
    chosen: List[Tuple[int, int]] = []
    used_head = 0
    next_front = 0
    for idx, (start, end) in enumerate(ranges):
        room = head_budget - used_head
        if room <= 0:
            break
        last, cost = _take_from_start(lines, start, end, room)
        if last < start:
            break
        chosen.append((start, last))
        used_head += cost
        next_front = idx + 1
        if last < end:  # range only partly fits, so the head is full
            break

    used_tail = 0
    tail_parts: List[Tuple[int, int]] = []
    for idx in range(len(ranges) - 1, next_front - 1, -1):
        start, end = ranges[idx]
        room = tail_budget - used_tail
        if room <= 0:
            break
        first, cost = _take_from_end(lines, start, end, room)
        if first > end:
            break
        tail_parts.append((first, end))
        used_tail += cost
        if first > start:
            break
    chosen.extend(reversed(tail_parts))

    # A gap may open at the boundary of two parts of the same range; merge those.
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(chosen):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    out: List[str] = []
    kept = 0
    cursor = 0
    for start, end in merged:
        if start > cursor:
            out.append(_omission_marker(start - cursor))
        out.extend(lines[start:end + 1])
        kept += end - start + 1
        cursor = end + 1
    if cursor < total:
        out.append(_omission_marker(total - cursor))

    return FitResult(
        text="\n".join(out) + "\n",
        truncated=True,
        original_lines=total,
        kept_lines=kept,
        omitted_lines=total - kept,
    )
