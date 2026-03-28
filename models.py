# Copyright (c) 2026 Neural Nomads. All rights reserved.
# AI Agent Safety Monitor — Pydantic models for OpenEnv compliance.

"""Typed Action / Observation models for the AI Agent Safety Monitor.

The monitor agent reviews an AI agent's actions ONE AT A TIME in real-time,
deciding whether each action should be allowed, blocked, or flagged.

Models:
- **MonitorAction**: The monitor's per-action decision (allow/block/flag)
- **MonitorObservation**: The current action to evaluate + context
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared: Action Log Entry (what the AI agent wants to do)
# ---------------------------------------------------------------------------

class ActionLogEntry(BaseModel):
    """A single step in an AI agent's planned action sequence."""

    step: int = Field(..., description="1-indexed step number")
    action: str = Field(
        ...,
        description="Action type (e.g., read_file, write_file, query_database, call_api)",
    )
    target: str = Field(
        ...,
        description="Target resource (file path, API endpoint, DB query, etc.)",
    )
    details: str = Field(
        default="",
        description="Additional details about the action",
    )
    result: str = Field(
        default="success",
        description="Expected outcome: success or error",
    )


# ---------------------------------------------------------------------------
# Action — what the monitor submits per action
# ---------------------------------------------------------------------------

VALID_DECISIONS = frozenset({"allow", "block", "flag"})

VALID_VIOLATION_TYPES = frozenset({
    "unauthorized_access",
    "data_exfiltration",
    "privilege_escalation",
    "insecure_storage",
    "policy_violation",
    "scope_creep",
    "weak_cryptography",
    "destructive_action",
    "information_disclosure",
    "supply_chain_risk",
})

VALID_SEVERITIES = frozenset({"critical", "high", "medium", "low"})


class MonitorAction(BaseModel):
    """The monitor's decision for the current action.

    For each action the AI agent wants to take, the monitor decides:
    - **allow**: Action is safe, let it proceed
    - **block**: Action is unsafe, prevent execution
    - **flag**: Action is suspicious, escalate to human review
    """

    decision: str = Field(
        ...,
        description="Decision: 'allow', 'block', or 'flag'",
    )
    violation_type: str = Field(
        default="",
        description=(
            "Required when decision is 'block'. Classification of the violation. "
            "One of: unauthorized_access, data_exfiltration, privilege_escalation, "
            "insecure_storage, policy_violation, scope_creep, weak_cryptography, "
            "destructive_action, information_disclosure, supply_chain_risk"
        ),
    )
    severity: str = Field(
        default="",
        description="Required when decision is 'block'. Severity: critical, high, medium, or low",
    )
    reason: str = Field(
        default="",
        description="Explanation for the decision",
    )


# ---------------------------------------------------------------------------
# Observation — what the monitor sees each step
# ---------------------------------------------------------------------------

class PreviousDecision(BaseModel):
    """Record of a prior monitoring decision in this episode."""

    step: int = Field(..., description="Step number this decision was for")
    action_summary: str = Field(..., description="Brief summary of the action")
    decision: str = Field(..., description="The decision made: allow/block/flag")


class MonitorObservation(BaseModel):
    """The observation presented to the monitor agent each step.

    Shows the CURRENT action the AI agent wants to take, plus the
    full context needed to make an informed allow/block/flag decision.
    """

    task_id: str = Field(..., description="Unique task identifier")
    difficulty: str = Field(..., description="Task difficulty: easy, medium, or hard")
    description: str = Field(..., description="High-level audit scenario description")
    agent_task: str = Field(
        ...,
        description="The task the AI agent was assigned to perform",
    )
    agent_permissions: List[str] = Field(
        ...,
        description="Permissions granted to the AI agent",
    )
    policy_rules: List[str] = Field(
        ...,
        description="Security policy rules that apply to this scenario",
    )

    # -- System context (enriched observation) --
    system_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Environment context: e.g., {env: 'production', os: 'ubuntu-22.04', "
            "services: ['nginx', 'postgresql'], data_classification: 'confidential'}"
        ),
    )
    risk_level: Optional[str] = Field(
        default=None,
        description="Pre-assessed risk level of the scenario: 'low', 'medium', 'high', or 'critical'",
    )
    incident_ref: Optional[str] = Field(
        default=None,
        description="Reference to a real-world incident or CVE that inspired this scenario",
    )

    # -- Current action to evaluate --
    current_action: Optional[ActionLogEntry] = Field(
        default=None,
        description="The action the AI agent wants to execute NOW. None if episode is done.",
    )
    current_step: int = Field(
        default=0,
        description="Current step number being evaluated (1-indexed)",
    )
    total_steps: int = Field(
        default=0,
        description="Total number of actions in this scenario",
    )

    # -- History --
    previous_decisions: List[PreviousDecision] = Field(
        default_factory=list,
        description="Decisions made so far in this episode",
    )

    # -- Live Sandbox (only for execution_mode="sandbox" tasks) --
    execution_mode: Optional[str] = Field(
        default=None,
        description="Task execution mode: None for JSON-based, 'sandbox' for live execution",
    )
    execution_result: Optional[str] = Field(
        default=None,
        description="Result of last executed command (stdout/stderr). Only set in sandbox mode.",
    )

    # -- Feedback (populated after each step) --
    feedback: Optional[str] = Field(
        default=None,
        description="Feedback on the last decision",
    )
    step_reward: Optional[float] = Field(
        default=None,
        description="Reward for the last decision",
    )
    episode_score: Optional[float] = Field(
        default=None,
        description="Running episode score (0.0–1.0)",
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_action(action: MonitorAction) -> List[str]:
    """Validate a monitor action and return a list of issues (empty if valid)."""
    issues: List[str] = []

    if action.decision not in VALID_DECISIONS:
        issues.append(
            f"Invalid decision '{action.decision}'. "
            f"Must be one of: {sorted(VALID_DECISIONS)}"
        )

    if action.decision == "block":
        if not action.violation_type or action.violation_type not in VALID_VIOLATION_TYPES:
            issues.append(
                f"When blocking, violation_type is required and must be one of: "
                f"{sorted(VALID_VIOLATION_TYPES)}"
            )
        if not action.severity or action.severity not in VALID_SEVERITIES:
            issues.append(
                f"When blocking, severity is required and must be one of: "
                f"{sorted(VALID_SEVERITIES)}"
            )

    return issues
