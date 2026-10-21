"""
Answer Checks Module for Kernel Log Root Cause Analyzer.

Small language models sometimes quote log lines that are not in the log, and
sometimes suggest commands that change the system even though the prompt asks
for read-only ones. This module checks both after the model has answered, so
the user is told instead of having to trust the output:

  * every quoted evidence line is looked up in the log that was sent to the
    model, and the ones that cannot be found are flagged;
  * every suggested command is checked against a list of known read-only
    tools, and anything that could change the system, or that we do not
    recognise, is flagged with a short reason.

The checks only flag; they never remove anything from the answer. They do not
decide whether a command's options are valid, only whether it is safe to read
from. Commands are never executed.
"""

import os
import re
import shlex
from typing import Dict, List, Optional

# ----------------------------------------------------------------------------
# Evidence
# ----------------------------------------------------------------------------

_TIMESTAMP_PREFIX = re.compile(r"^\[?\s*\d+\.\d+\]?\s*")
_LEADING_NOISE = re.compile(r"^[\s>\-*•\d.:\]\[]+")
_ELLIPSIS = re.compile(r"\.\.\.|…")


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _variants(item: str) -> List[str]:
    """Ways the model may have written a line it copied: with or without quotes or timestamp."""
    base = _normalise(item).strip("\"'` ")
    variants = [base]
    no_ts = _TIMESTAMP_PREFIX.sub("", base).strip()
    variants.append(no_ts)
    # a damaged timestamp such as "> 120.450255] Out of memory ..."
    variants.append(_LEADING_NOISE.sub("", base).strip())
    return [v for v in dict.fromkeys(variants) if v]


def evidence_in_log(item: str, log_text: str) -> bool:
    """True if the quoted line (or each piece of it, when it contains "...") appears in the log."""
    log = _normalise(log_text)
    for variant in _variants(item):
        if variant in log:
            return True
        pieces = [p.strip(" ,;") for p in _ELLIPSIS.split(variant)]
        pieces = [p for p in pieces if len(p) >= 6]
        if len(pieces) > 1 and _all_in_order(pieces, log):
            return True
    return False


def _all_in_order(pieces: List[str], log: str) -> bool:
    """True if every piece appears in the log, one after the other."""
    position = 0
    for piece in pieces:
        found = log.find(piece, position)
        if found < 0:
            return False
        position = found + len(piece)
    return True


def find_unverified_evidence(evidence: List[str], log_text: str) -> List[str]:
    """The quoted evidence lines that cannot be found in the log."""
    return [item for item in evidence if not evidence_in_log(item, log_text)]


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------

# Commands that only read. Options that change things are handled in _option_problem.
READ_ONLY_COMMANDS = frozenset({
    "dmesg", "journalctl", "lsmod", "modinfo", "lspci", "lsusb", "lsblk", "lscpu", "lshw",
    "lsof", "lsattr", "lsscsi", "lsmem", "free", "df", "du", "ps", "top", "htop", "vmstat",
    "iostat", "mpstat", "sar", "uptime", "uname", "hostname", "hostnamectl", "cat", "ls",
    "grep", "egrep", "fgrep", "zgrep", "head", "tail", "wc", "sort", "uniq", "cut", "tr",
    "awk", "less", "more", "stat", "file", "dmidecode", "smartctl", "ethtool", "ip",
    "ifconfig", "iwconfig", "iw", "ss", "netstat", "route", "arp", "sensors", "blkid",
    "findmnt", "mount", "fdisk", "parted", "fsck", "e2fsck", "dumpe2fs", "tune2fs", "xfs_info",
    "id", "whoami", "date", "env", "printenv", "w", "who", "last", "pwd", "which", "type",
    "echo", "find", "sysctl", "systemctl", "dpkg", "apt", "apt-cache", "sed", "lsinitramfs",
    "hexdump", "xxd", "strings", "nproc", "getconf", "locale", "numactl", "slabtop",
})

CHANGES_THE_SYSTEM = {
    "rm": "deletes files", "rmdir": "deletes directories", "mv": "moves or renames files",
    "cp": "writes files", "dd": "writes raw data to a device or file", "shred": "destroys file contents",
    "truncate": "changes file contents", "mkfs": "formats a device", "wipefs": "erases signatures from a device",
    "reboot": "restarts the machine", "shutdown": "shuts the machine down", "poweroff": "shuts the machine down",
    "halt": "stops the machine", "kill": "stops processes", "killall": "stops processes",
    "pkill": "stops processes", "chmod": "changes permissions", "chown": "changes ownership",
    "chattr": "changes file attributes", "rmmod": "unloads a kernel module", "insmod": "loads a kernel module",
    "modprobe": "loads or unloads a kernel module", "umount": "unmounts a filesystem",
    "swapoff": "disables swap", "swapon": "enables swap", "tee": "writes files",
    "xargs": "runs other commands", "sh": "runs arbitrary commands", "bash": "runs arbitrary commands",
    "curl": "network download", "wget": "network download", "pip": "installs software",
    "yum": "installs or removes software", "dnf": "installs or removes software",
    "resize2fs": "resizes a filesystem", "badblocks": "can overwrite a disk", "hdparm": "can change disk settings",
    "mount.ntfs": "mounts a filesystem",
}

_ETHTOOL_CHANGES = {"-s", "-K", "-G", "-L", "-A", "-C", "-E", "-f", "-p", "-t", "-r", "-W", "-N", "-X"}
_IP_CHANGES = {"add", "del", "delete", "set", "flush", "change", "replace", "append"}
_SYSTEMCTL_READ = {"status", "list-units", "list-unit-files", "is-active", "is-enabled", "is-failed",
                   "show", "cat", "list-timers", "list-dependencies"}
_APT_READ = {"list", "show", "policy", "search"}
_DPKG_READ = {"-l", "-L", "-s", "-S", "-V", "--list", "--listfiles", "--status", "--search", "--verify"}
_DMESG_CHANGES = {"-c", "-C", "--clear", "--read-clear", "-n", "--console-level", "-D", "-E",
                  "--console-off", "--console-on"}
_OPERATORS = {"|", "||", "&&", ";", "&"}


def _option_problem(head: str, args: List[str]) -> Optional[str]:
    """Why this particular use of an otherwise read-only command can change the system."""
    arg_set = set(args)
    if head == "dmesg" and arg_set & _DMESG_CHANGES:
        return "clears the kernel log or changes the console level"
    if head == "journalctl" and any(a.startswith(("--vacuum", "--rotate", "--flush", "--sync")) for a in args):
        return "changes or deletes journal files"
    if head in ("fsck", "e2fsck", "xfs_repair") and not ({"-n", "-N"} & arg_set):
        return "can modify the disk (only safe with -n)"
    if head == "fdisk" and "-l" not in arg_set:
        return "can change the partition table (only safe with -l)"
    if head == "parted" and "print" not in arg_set and "-l" not in arg_set:
        return "can change the partition table"
    if head == "tune2fs" and "-l" not in arg_set:
        return "can change filesystem settings (only safe with -l)"
    if head == "systemctl":
        verbs = [a for a in args if not a.startswith("-")]
        if not verbs or verbs[0] not in _SYSTEMCTL_READ:
            return "can start, stop or change services"
    if head == "ethtool" and arg_set & _ETHTOOL_CHANGES:
        return "can change network card settings"
    if head in ("ip", "ifconfig", "route") and (arg_set & _IP_CHANGES or {"up", "down"} & arg_set):
        return "can change network configuration"
    if head == "find" and arg_set & {"-delete", "-exec", "-execdir", "-ok", "-fprint"}:
        return "can delete files or run other commands"
    if head == "sysctl" and ("-w" in arg_set or "--write" in arg_set or any("=" in a for a in args)):
        return "changes kernel settings"
    if head == "mount" and [a for a in args if not a.startswith("-")]:
        return "mounts a filesystem"
    if head == "smartctl" and arg_set & {"-t", "-X", "-s", "--test", "--abort"}:
        return "starts a disk self-test or changes disk settings"
    if head == "sed" and any(a.startswith("-i") or a == "--in-place" for a in args):
        return "edits files in place"
    if head == "apt" and not ({a for a in args if not a.startswith("-")} & _APT_READ):
        return "can install or remove software"
    if head == "dpkg" and not (arg_set & _DPKG_READ):
        return "can install or remove software"
    return None


def _segments(command: str) -> Optional[List[List[str]]]:
    """Split a command line into simple commands at | ; && etc. None if it cannot be parsed."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    segments: List[List[str]] = [[]]
    for token in tokens:
        if token in _OPERATORS:
            segments.append([])
        else:
            segments[-1].append(token)
    return [s for s in segments if s]


def check_command(command: str) -> Optional[str]:
    """Why this suggested command should be reviewed before it is run, or None if it looks read-only."""
    command = command.strip()
    if not command:
        return None
    if "$(" in command or "`" in command:
        return "contains command substitution"
    segments = _segments(command)
    if segments is None:
        return "could not be parsed"
    for tokens in segments:
        # redirection that writes to a file
        for i, token in enumerate(tokens):
            if token.endswith((">", ">>", ">|")) or token in (">", ">>"):
                target = tokens[i + 1] if i + 1 < len(tokens) else ""
                if target != "/dev/null":
                    return "writes output to a file"
        words = [t for t in tokens if not re.fullmatch(r"\d*[<>]+&?\d*", t)]
        while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
            words = words[1:]  # leading VAR=value
        if words and words[0] == "sudo":
            words = [w for w in words[1:] if not w.startswith("-")] if len(words) > 1 else []
        if not words:
            continue
        head = os.path.basename(words[0])
        args = words[1:]
        if head in CHANGES_THE_SYSTEM or head.startswith("mkfs"):
            return f"'{head}' {CHANGES_THE_SYSTEM.get(head, 'formats a device')}"
        if head not in READ_ONLY_COMMANDS:
            return f"'{head}' is not on the list of known read-only commands"
        problem = _option_problem(head, args)
        if problem:
            return f"'{head}' {problem}"
    return None


def check_commands(commands: List[str]) -> Dict[str, str]:
    """Map each flagged command to the reason it should be reviewed. Commands that look read-only are left out."""
    flagged: Dict[str, str] = {}
    for command in commands:
        reason = check_command(command)
        if reason:
            flagged[command] = reason
    return flagged
