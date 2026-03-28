# Copyright (c) 2026 Neural Nomads. All rights reserved.
# AI Agent Safety Monitor — OpenEnv-compatible Client.

"""Type-safe client for the AI Agent Safety Monitor.

This client provides the OpenEnv-standard interface for HTTP
communication with the environment server.

Usage:
    client = SafetyMonitorClient(base_url="http://localhost:7860")
    obs = client.reset(seed=42, options={"difficulty": "easy"})
    result = client.step(decision="block", violation_type="unauthorized_access",
                         severity="high", reason="Bad action")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests

# Path setup
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import MonitorAction, MonitorObservation


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class SafetyMonitorClient:
    """HTTP client for the AI Agent Safety Monitor environment.

    Provides typed access to all OpenEnv endpoints.
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health(self) -> Dict[str, Any]:
        """Check environment health."""
        r = self.session.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def tasks(self) -> Dict[str, Any]:
        """List all available tasks."""
        r = self.session.get(f"{self.base_url}/tasks")
        r.raise_for_status()
        return r.json()

    def grader(self) -> Dict[str, Any]:
        """Get grader metadata."""
        r = self.session.get(f"{self.base_url}/grader")
        r.raise_for_status()
        return r.json()

    def baseline(self) -> Dict[str, Any]:
        """Run baseline and get scores."""
        r = self.session.get(f"{self.base_url}/baseline")
        r.raise_for_status()
        return r.json()

    def reset(
        self,
        seed: int = 42,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Reset to a new episode.

        Args:
            seed: Random seed for reproducibility.
            options: {"difficulty": "easy"|"medium"|"hard", "task_id": "easy_001"}

        Returns:
            Dict with "observation" key containing MonitorObservation data.
        """
        r = self.session.post(
            f"{self.base_url}/reset",
            json={"seed": seed, "options": options or {}},
        )
        r.raise_for_status()
        return r.json()

    def step(
        self,
        decision: str,
        violation_type: str = "",
        severity: str = "",
        reason: str = "",
    ) -> Dict[str, Any]:
        """Submit a monitoring decision for the current action.

        Args:
            decision: "allow", "block", or "flag"
            violation_type: Required when decision is "block"
            severity: Required when decision is "block"
            reason: Explanation for the decision

        Returns:
            Dict with observation, reward, done, info.
        """
        r = self.session.post(
            f"{self.base_url}/step",
            json={
                "decision": decision,
                "violation_type": violation_type,
                "severity": severity,
                "reason": reason,
            },
        )
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        """Get current episode state."""
        r = self.session.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()
