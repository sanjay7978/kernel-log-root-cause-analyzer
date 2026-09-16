"""
Prompt Builder Module for Kernel Log Root Cause Analyzer (Phase 2).

Converts preprocessed kernel log text into a structured, evidence-grounded
diagnostic prompt for the local LLM. Enforces factual grounding, safety,
and a strict JSON response schema.
"""

from typing import Optional
from config import CATEGORIES, VALID_SEVERITIES


DIAGNOSTIC_SYSTEM_INSTRUCTIONS = f"""You are a senior Linux Kernel Diagnostic Specialist and Reliability Engineer.
Your task is to analyze sanitized Linux kernel log messages and determine the probable root cause of the incident.

STRICT DIAGNOSTIC RULES:
1. FACTUAL GROUNDING: Base your conclusions strictly on the provided log lines. Do NOT invent or hallucinate error messages, devices, or symptoms not present in the evidence.
2. DISTINGUISH FACT FROM INFERENCE:
   - "Evidence" must be direct quotes or exact details from the log lines.
   - "Inference" is your technical deduction based on those facts.
3. ADMIT UNCERTAINTY: If the logs are incomplete, ambiguous, or only show symptoms without the primary root cause, explicitly state this in the "uncertainty" field. Do not claim certainty when the evidence does not establish it.
4. SUBSYSTEM CLASSIFICATION: Choose the most affected subsystem from this exact list:
   {', '.join(CATEGORIES)}, other
5. SEVERITY CLASSIFICATION: Assign one of these exact severity ratings:
   {', '.join(VALID_SEVERITIES)}
   - CRITICAL: Kernel panic, hardware unrecoverable fault, complete system lockup/data corruption risk.
   - HIGH: Out of memory killer invoked, filesystem remounted read-only, core device failure.
   - MEDIUM: Driver retry/timeout loops, peripheral enumeration failure, non-fatal link drops.
   - LOW: Informational warnings, degraded performance, transient errors successfully recovered.
6. COMMAND SAFETY: Prioritize safe, read-only diagnostic commands (e.g., dmesg, journalctl -k, free -h, lsusb, lsblk, lspci). Do not recommend destructive commands (like rm, mkfs, or dd) casually. Any remediation commands must be clearly labeled as potential fixes.
7. OUTPUT FORMAT: You must respond ONLY with a valid JSON object. Do not include markdown code block ticks (```json), markdown formatting, or introductory/closing remarks outside the JSON.
"""

JSON_SCHEMA_INSTRUCTIONS = """
Your output must strictly follow this JSON schema:
{
  "root_cause": "Concise statement of the probable root cause",
  "subsystem": "usb | filesystem | memory | network | gpu | driver | storage | acpi | panic | other",
  "severity": "LOW | MEDIUM | HIGH | CRITICAL",
  "evidence": [
    "Exact relevant quote or line from the log demonstrating the problem"
  ],
  "explanation": "Clear, plain English explanation of what happened, why the kernel reacted this way, and what the consequences are",
  "commands": [
    "Recommended read-only Linux diagnostic commands to investigate further"
  ],
  "possible_fixes": [
    "Safe, actionable steps or configuration checks to remediate the issue"
  ],
  "confidence": "HIGH | MEDIUM | LOW (with brief rationale)",
  "uncertainty": "Explanation of missing information or alternative possibilities if evidence is incomplete"
}
"""


def build_diagnostic_prompt(cleaned_kernel_log: str, context_notes: Optional[str] = None) -> str:
    """
    Construct the diagnostic prompt from preprocessed kernel log text.

    Parameters:
        cleaned_kernel_log: Sanitized kernel log text from Phase 1 preprocessor.
        context_notes: Optional user or environment notes.

    Returns:
        str: Complete prompt ready to send to the local LLM.

    Raises:
        ValueError: If cleaned_kernel_log is empty or contains only whitespace.
    """
    if not cleaned_kernel_log or not cleaned_kernel_log.strip():
        raise ValueError("Cannot build diagnostic prompt from empty kernel log content.")

    log_section = cleaned_kernel_log.strip()
    notes_section = f"\nADDITIONAL CONTEXT:\n{context_notes.strip()}\n" if context_notes else ""

    prompt = (
        f"{DIAGNOSTIC_SYSTEM_INSTRUCTIONS}\n"
        f"{JSON_SCHEMA_INSTRUCTIONS}\n"
        f"========================================\n"
        f"KERNEL LOG EVIDENCE TO ANALYZE:\n"
        f"========================================\n"
        f"{log_section}\n"
        f"{notes_section}"
        f"========================================\n"
        f"JSON DIAGNOSTIC REPORT:"
    )

    return prompt
