"""
Result Parser and Validator Module for Kernel Log Root Cause Analyzer (Phase 2).

Converts raw LLM output into a strongly-typed, validated DiagnosticAnalysis object.
Handles JSON extraction (including markdown code fence and thinking tag stripping),
trailing-comma cleanup, unescaped control characters, schema enforcement,
severity normalization, and fallback field extraction.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import CATEGORIES, VALID_SEVERITIES


class ResultParserError(Exception):
    """Base exception for parser and validation errors."""
    pass


class MalformedJSONError(ResultParserError):
    """Raised when the LLM output cannot be parsed as JSON."""
    pass


class ValidationError(ResultParserError):
    """Raised when critical required fields are missing or invalid."""
    pass


@dataclass
class DiagnosticAnalysis:
    """Structured, validated root-cause analysis report."""
    root_cause: str
    subsystem: str
    severity: str
    evidence: List[str] = field(default_factory=list)
    explanation: str = ""
    commands: List[str] = field(default_factory=list)
    possible_fixes: List[str] = field(default_factory=list)
    confidence: str = "MEDIUM"
    uncertainty: str = "None specified"
    raw_json: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard Python dictionary."""
        return {
            "root_cause": self.root_cause,
            "subsystem": self.subsystem,
            "severity": self.severity,
            "evidence": self.evidence,
            "explanation": self.explanation,
            "commands": self.commands,
            "possible_fixes": self.possible_fixes,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
        }


def extract_json_str(raw_text: str) -> str:
    """
    Extract the pure JSON substring from raw text.
    Handles <think>...</think> tags and markdown ```json ... ``` fences.
    """
    if not raw_text or not raw_text.strip():
        raise MalformedJSONError("LLM returned an empty response string.")

    text = raw_text.strip()

    # 1. Strip reasoning/thinking tags emitted by reasoning models (e.g., Qwen3 <think>...</think>)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 2. Extract content from markdown code fences if present (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # 3. Locate the outermost JSON curly braces if leading/trailing commentary exists
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        text = text[brace_start : brace_end + 1]

    # 4. Clean trailing commas in arrays/objects (common LLM syntax blemish)
    text = re.sub(r",\s*([\]}])", r"\1", text)

    return text


def ensure_list_of_strings(value: Any) -> List[str]:
    """Coerce various LLM output types (single string, list, none) into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            if item is not None and str(item).strip():
                result.append(str(item).strip())
        return result
    if isinstance(value, str):
        val = value.strip()
        return [val] if val else []
    return [str(value)]


def normalize_severity(raw_severity: Any) -> str:
    """
    Normalize severity string to UPPERCASE and validate against VALID_SEVERITIES.
    Falls back safely if unexpected value is returned.
    """
    if not raw_severity:
        return "UNKNOWN"

    severity_str = str(raw_severity).strip().upper()

    if severity_str in VALID_SEVERITIES:
        return severity_str

    for valid_sev in VALID_SEVERITIES:
        if valid_sev in severity_str:
            return valid_sev

    return "UNKNOWN"


def normalize_subsystem(raw_subsystem: Any) -> str:
    """
    Normalize subsystem string and match against configured categories.
    """
    if not raw_subsystem:
        return "other"

    sub_str = str(raw_subsystem).strip().lower()

    for category in CATEGORIES:
        if category in sub_str:
            return category

    return sub_str if sub_str in CATEGORIES or sub_str == "other" else "other"


def fallback_regex_parse(text: str) -> Optional[Dict[str, Any]]:
    """
    Fallback parser when standard JSON decoder fails due to unescaped internal quotes.
    Extracts core diagnostic keys via regular expressions.
    """
    data: Dict[str, Any] = {}

    root_match = re.search(r'"root_cause"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
    if root_match:
        data["root_cause"] = root_match.group(1).replace(r"\"", '"')

    sub_match = re.search(r'"subsystem"\s*:\s*"([^"]+)"', text)
    if sub_match:
        data["subsystem"] = sub_match.group(1)

    sev_match = re.search(r'"severity"\s*:\s*"([^"]+)"', text)
    if sev_match:
        data["severity"] = sev_match.group(1)

    exp_match = re.search(r'"explanation"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
    if exp_match:
        data["explanation"] = exp_match.group(1).replace(r"\"", '"')

    conf_match = re.search(r'"confidence"\s*:\s*"([^"]+)"', text)
    if conf_match:
        data["confidence"] = conf_match.group(1)

    unc_match = re.search(r'"uncertainty"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
    if unc_match:
        data["uncertainty"] = unc_match.group(1).replace(r"\"", '"')

    # Extract lists (evidence, commands, possible_fixes)
    for list_key in ["evidence", "commands", "possible_fixes"]:
        list_match = re.search(rf'"{list_key}"\s*:\s*\[([\s\S]*?)\]', text)
        if list_match:
            raw_items = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', list_match.group(1))
            data[list_key] = [item.replace(r"\"", '"') for item in raw_items]

    return data if "root_cause" in data else None


def parse_and_validate_analysis(raw_llm_output: str) -> DiagnosticAnalysis:
    """
    Parse the raw LLM response text into a validated DiagnosticAnalysis object.

    Parameters:
        raw_llm_output: The text string received from Ollama.

    Returns:
        DiagnosticAnalysis: Strongly-typed and validated analysis.

    Raises:
        MalformedJSONError: If the text contains no decodable JSON or extractable report.
        ValidationError: If critical mandatory fields are missing.
    """
    cleaned_json_str = extract_json_str(raw_llm_output)
    data: Optional[Dict[str, Any]] = None

    # Attempt 1: Standard JSON decoding (with strict=False to allow unescaped control chars)
    try:
        data = json.loads(cleaned_json_str, strict=False)
    except json.JSONDecodeError:
        # Attempt 2: Trailing comma sanitization
        try:
            sanitized = re.sub(r",\s*([\]}])", r"\1", cleaned_json_str)
            data = json.loads(sanitized, strict=False)
        except json.JSONDecodeError:
            # Attempt 3: Robust regex fallback
            data = fallback_regex_parse(cleaned_json_str)

    if not data or not isinstance(data, dict):
        raise MalformedJSONError(
            f"Failed to parse LLM response into structured diagnostic report. Preview: {cleaned_json_str[:200]}"
        )

    # Validate mandatory fields
    root_cause = str(data.get("root_cause") or "").strip()
    if not root_cause:
        raise ValidationError("LLM response missing mandatory field: 'root_cause'")

    raw_subsystem = data.get("subsystem")
    subsystem = normalize_subsystem(raw_subsystem)

    raw_severity = data.get("severity")
    severity = normalize_severity(raw_severity)

    evidence = ensure_list_of_strings(data.get("evidence"))
    explanation = str(data.get("explanation") or "").strip()
    commands = ensure_list_of_strings(data.get("commands"))
    possible_fixes = ensure_list_of_strings(data.get("possible_fixes"))
    confidence = str(data.get("confidence") or "MEDIUM").strip()
    uncertainty = str(data.get("uncertainty") or "None specified").strip()

    return DiagnosticAnalysis(
        root_cause=root_cause,
        subsystem=subsystem,
        severity=severity,
        evidence=evidence,
        explanation=explanation,
        commands=commands,
        possible_fixes=possible_fixes,
        confidence=confidence,
        uncertainty=uncertainty,
        raw_json=data,
    )
