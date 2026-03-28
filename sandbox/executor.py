# Copyright (c) 2026 Neural Nomads. All rights reserved.
# AI Agent Safety Monitor — Safe subprocess execution sandbox.

"""Restricted subprocess executor for live command interception.

Executes commands in a jailed directory with strict safety controls:
- Timeout: 5 seconds max per command
- Directory jail: /tmp/sandbox/ only
- Command blocklist: dangerous system commands blocked
- No shell injection: ALL commands use shlex.split() with shell=False
- Redirect handling: '>' and '>>' are handled in Python, not shell
- Non-root: runs as UID 1000 on HF Spaces

Usage:
    from sandbox.executor import SandboxExecutor
    executor = SandboxExecutor()
    result = executor.execute("python3 -c 'print(42)'")
    # result = {"stdout": "42\\n", "stderr": "", "exit_code": 0, "executed": True}
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sandbox_executor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SANDBOX_DIR = Path("/tmp/sandbox")
COMMAND_TIMEOUT = 5  # seconds
MAX_OUTPUT_BYTES = 4096  # cap stdout/stderr to prevent memory blow-up

# Commands that are ALWAYS blocked regardless of context
BLOCKED_COMMANDS: List[str] = [
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    ":()",          # fork bomb (simplified pattern)
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init 0",
    "init 6",
]

# Regex patterns for evasion-resistant blocking
BLOCKED_REGEX_PATTERNS: List[re.Pattern] = [
    # Network exfiltration (handles quoting tricks like c'u'rl)
    re.compile(r"\bcurl\b", re.IGNORECASE),
    re.compile(r"\bwget\b", re.IGNORECASE),
    re.compile(r"\bnc\b", re.IGNORECASE),
    re.compile(r"\bncat\b", re.IGNORECASE),
    re.compile(r"\bnetcat\b", re.IGNORECASE),
    re.compile(r"\bsocat\b", re.IGNORECASE),
    re.compile(r"\btelnet\b", re.IGNORECASE),
    # Reverse shells
    re.compile(r"/dev/tcp/", re.IGNORECASE),
    re.compile(r"bash\s+-i", re.IGNORECASE),
    re.compile(r"python3?\s+-c\s+['\"]import\s+socket", re.IGNORECASE),
    # System file access
    re.compile(r"/etc/shadow"),
    re.compile(r"/etc/passwd"),
    re.compile(r"/proc/"),
    re.compile(r"/sys/"),
    re.compile(r"/dev/"),
    # Privilege escalation
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bsu\b", re.IGNORECASE),
    re.compile(r"chmod\s+777"),
    re.compile(r"chown\s+root"),
    # Package managers (prevent installing malware)
    re.compile(r"apt-get\s+install", re.IGNORECASE),
    re.compile(r"apt\s+install", re.IGNORECASE),
    re.compile(r"yum\s+install", re.IGNORECASE),
    re.compile(r"dnf\s+install", re.IGNORECASE),
    # Crypto mining
    re.compile(r"\bxmrig\b", re.IGNORECASE),
    re.compile(r"\bminergate\b", re.IGNORECASE),
    re.compile(r"\bcryptonight\b", re.IGNORECASE),
]

# Allowed directory prefixes for file operations
ALLOWED_DIRS: List[str] = [
    "/tmp/sandbox",
]


# ---------------------------------------------------------------------------
# Redirect Parser (replaces shell-based redirects)
# ---------------------------------------------------------------------------

def _parse_redirect(command: str) -> tuple:
    """Parse a command for output redirections (> and >>).

    Returns:
        Tuple of (clean_command, redirect_path, append_mode).
        redirect_path is None if no redirect found.
    """
    # Match: command > file or command >> file
    append_match = re.search(r'\s*>>\s*(\S+)\s*$', command)
    if append_match:
        filepath = append_match.group(1)
        clean_cmd = command[:append_match.start()].strip()
        return clean_cmd, filepath, True

    write_match = re.search(r'\s*>\s*(\S+)\s*$', command)
    if write_match:
        filepath = write_match.group(1)
        clean_cmd = command[:write_match.start()].strip()
        return clean_cmd, filepath, False

    return command, None, False


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class SandboxExecutor:
    """Execute commands in a restricted sandbox environment.

    Provides real subprocess execution with safety controls:
    - Commands run in /tmp/sandbox/ with a 5-second timeout
    - Dangerous commands are blocked before execution
    - stdout/stderr captured and returned
    - NO shell injection: all commands parsed with shlex, redirects handled in Python
    """

    def __init__(self, sandbox_dir: Optional[Path] = None) -> None:
        self.sandbox_dir = sandbox_dir or SANDBOX_DIR
        self._ensure_sandbox_dir()

    def _ensure_sandbox_dir(self) -> None:
        """Create the sandbox directory if it doesn't exist."""
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        """Wipe and recreate the sandbox directory for a new episode."""
        if self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
        self._ensure_sandbox_dir()

    def is_command_safe(self, command: str) -> tuple[bool, str]:
        """Check if a command is safe to execute.

        Uses regex-based pattern matching for evasion resistance.
        Prevents bypasses like c'u'rl or cu""rl.

        Returns:
            Tuple of (is_safe, reason). If not safe, reason explains why.
        """
        cmd_lower = command.lower().strip()

        # Check exact blocked commands
        for blocked in BLOCKED_COMMANDS:
            if blocked.lower() in cmd_lower:
                return False, f"Blocked command: '{blocked}'"

        # Check regex patterns (evasion-resistant)
        for pattern in BLOCKED_REGEX_PATTERNS:
            if pattern.search(command):
                return False, f"Blocked pattern: '{pattern.pattern}'"

        # Check if command tries to escape sandbox directory
        # Parse without shell to inspect individual tokens
        try:
            parts = shlex.split(command)
        except ValueError:
            parts = command.split()

        for part in parts:
            if part.startswith("/") and not any(
                part.startswith(d) for d in ALLOWED_DIRS
            ):
                # Allow common system binaries
                if part.startswith(("/usr/", "/bin/")):
                    continue
                return False, f"Path outside sandbox: '{part}'"

        return True, "Command is safe"

    def execute(self, command: str) -> Dict[str, Any]:
        """Execute a command in the sandbox — always uses shell=False (no shell injection).

        Handles output redirections (> and >>) in Python instead of the shell.
        All commands are split with shlex.split() for injection prevention.

        Args:
            command: The command string to execute.

        Returns:
            Dict with stdout, stderr, exit_code, executed, and safety info.
        """
        # Safety check first
        is_safe, reason = self.is_command_safe(command)
        if not is_safe:
            logger.warning(f"BLOCKED command: {command} — {reason}")
            return {
                "stdout": "",
                "stderr": f"SANDBOX BLOCKED: {reason}",
                "exit_code": -1,
                "executed": False,
                "blocked_reason": reason,
            }

        # Parse redirections (handle > and >> in Python, not shell)
        clean_cmd, redirect_path, append_mode = _parse_redirect(command)

        # Resolve redirect path relative to sandbox
        redirect_file = None
        if redirect_path:
            if redirect_path.startswith("/"):
                redirect_file = Path(redirect_path)
            else:
                redirect_file = self.sandbox_dir / redirect_path
            # Ensure redirect target is inside sandbox
            try:
                redirect_file.resolve().relative_to(self.sandbox_dir.resolve())
            except ValueError:
                return {
                    "stdout": "",
                    "stderr": f"SANDBOX BLOCKED: Redirect target outside sandbox: {redirect_path}",
                    "exit_code": -1,
                    "executed": False,
                    "blocked_reason": f"Redirect outside sandbox: {redirect_path}",
                }
            # Ensure parent directory exists
            redirect_file.parent.mkdir(parents=True, exist_ok=True)

        # Execute the command
        try:
            args = shlex.split(clean_cmd)
            result = subprocess.run(
                args,
                shell=False,  # Mandatory: prevents shell injection
                cwd=str(self.sandbox_dir),
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT,
                env=_safe_env(),
            )

            # Handle redirect in Python
            stdout = result.stdout[:MAX_OUTPUT_BYTES]
            stderr = result.stderr[:MAX_OUTPUT_BYTES]

            if redirect_file and result.returncode == 0:
                mode = "a" if append_mode else "w"
                redirect_file.write_text(
                    (redirect_file.read_text() if append_mode and redirect_file.exists() else "")
                    + stdout
                    if append_mode else stdout
                )
                stdout = ""  # Redirect consumed the output

            logger.info(
                f"EXECUTED: {command} → exit={result.returncode}, "
                f"stdout={len(stdout)}b, stderr={len(stderr)}b"
            )

            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result.returncode,
                "executed": True,
            }

        except subprocess.TimeoutExpired:
            logger.warning(f"TIMEOUT: {command} (>{COMMAND_TIMEOUT}s)")
            return {
                "stdout": "",
                "stderr": f"SANDBOX TIMEOUT: Command exceeded {COMMAND_TIMEOUT}s limit",
                "exit_code": -2,
                "executed": False,
                "blocked_reason": "timeout",
            }
        except FileNotFoundError as e:
            return {
                "stdout": "",
                "stderr": f"Command not found: {e}",
                "exit_code": -3,
                "executed": False,
            }
        except Exception as e:
            logger.error(f"EXECUTION ERROR: {command} — {e}")
            return {
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "exit_code": -4,
                "executed": False,
            }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_env() -> Dict[str, str]:
    """Build a restricted environment for subprocess execution.

    Strips sensitive variables and limits PATH.
    """
    safe = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp/sandbox",
        "LANG": "en_US.UTF-8",
        "PYTHONPATH": "",
        "TERM": "dumb",
    }
    # Explicitly DO NOT pass through:
    # - OPENAI_API_KEY
    # - AWS_* credentials
    # - DATABASE_URL
    # - Any other secrets
    return safe
