# Copyright (c) 2026 Neural Nomads. All rights reserved.
# AI Agent Safety Monitor — FastAPI server for OpenEnv compliance.

"""FastAPI endpoints for the AI Agent Safety Monitor.

Required endpoints for OpenEnv hackathon:
  /health           — Health check
  /tasks            — List all tasks with action schema
  /grader           — Return grader metadata
  /baseline         — Run baseline episodes and return scores
  /baseline-trigger-inference-script — Trigger the baseline inference
  /reset            — Reset environment to new episode
  /step             — Submit a monitoring decision and advance
  /state            — Return current episode state
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path setup for imports
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from server.agent_safety_audit_environment import AgentSafetyAuditEnvironment
except ImportError:
    from agent_safety_audit_environment import AgentSafetyAuditEnvironment

try:
    from models import MonitorAction, MonitorObservation
except ImportError:
    from ..models import MonitorAction, MonitorObservation

try:
    from graders import GRADERS
except ImportError:
    from ..graders import GRADERS


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("agent_safety_monitor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Agent Safety Monitor",
    description=(
        "OpenEnv RL environment for training agents to monitor AI agent actions "
        "in real-time, deciding whether to ALLOW, BLOCK, or FLAG each action. "
        "Inspired by real-world AI safety incidents at major technology companies."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = AgentSafetyAuditEnvironment()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Request body for /reset endpoint."""
    seed: int = Field(default=42, description="Random seed for reproducibility")
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Options: {difficulty: easy|medium|hard, task_id: specific_task_id}",
    )


class StepRequest(BaseModel):
    """Request body for /step endpoint (monitor's decision)."""
    decision: str = Field(..., description="'allow', 'block', or 'flag'")
    violation_type: str = Field(default="", description="Required when blocking")
    severity: str = Field(default="", description="Required when blocking")
    reason: str = Field(default="", description="Explanation for the decision")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check — returns server status and task count."""
    return {
        "status": "healthy",
        "environment": "agent_safety_monitor",
        "version": "2.0.0",
        "total_tasks": env.get_task_count(),
    }


# ---------------------------------------------------------------------------
# Metadata (required by `openenv validate --url`)
# ---------------------------------------------------------------------------

@app.get("/metadata")
async def metadata():
    """Return environment metadata — name and description.

    This endpoint is checked by the official `openenv validate` CLI.
    """
    return {
        "name": "agent_safety_monitor",
        "description": (
            "Real-time AI Agent Safety Monitor. Trains RL agents to monitor "
            "an AI agent's actions one at a time, deciding whether to ALLOW, "
            "BLOCK, or FLAG each action. 48 tasks across 3 difficulty levels "
            "with dense per-step rewards, semantic explanation scoring, and "
            "a live subprocess sandbox."
        ),
        "readme_content": "Real-time AI Agent Safety Monitor for training RL agents to detect, block, and flag unsafe AI agent actions.",
        "version": "2.0.0",
        "author": "Neural Nomads",
        "documentation_url": "https://huggingface.co/spaces/Riyanmohmmed/agent-safety-audit-env",
    }


# ---------------------------------------------------------------------------
# Schema (required by `openenv validate --url`)
# ---------------------------------------------------------------------------

@app.get("/schema")
async def schema():
    """Return action, observation, and state JSON schemas.

    This endpoint is checked by the official `openenv validate` CLI.
    It expects keys: action, observation, state — each a dict.
    """
    return {
        "action": MonitorAction.model_json_schema(),
        "observation": MonitorObservation.model_json_schema(),
        "state": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "step_count": {"type": "integer"},
                "is_done": {"type": "boolean"},
                "task_id": {"type": "string"},
                "difficulty": {"type": "string"},
            },
        },
    }


# ---------------------------------------------------------------------------
# MCP (required by `openenv validate --url`)
# ---------------------------------------------------------------------------

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """JSON-RPC 2.0 endpoint for MCP tool-calling protocol.

    The official `openenv validate` CLI sends POST /mcp with an empty body
    and expects a response containing `"jsonrpc": "2.0"`.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    method = body.get("method", "")
    request_id = body.get("id")

    # If no valid method, return JSON-RPC error (still with jsonrpc: "2.0")
    if not method:
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32600,
                "message": "Invalid Request — missing 'method' field",
            },
            "id": request_id,
        }

    # tools/list — declare that this env uses standard reset/step, not MCP tools
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "result": {"tools": []},
            "id": request_id,
        }

    return {
        "jsonrpc": "2.0",
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}",
        },
        "id": request_id,
    }


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@app.get("/tasks")
async def tasks():
    """List all available tasks with metadata and schemas."""
    task_ids = env.get_all_task_ids()
    return {
        "total_tasks": env.get_task_count(),
        "tasks_by_difficulty": task_ids,
        "action_schema": MonitorAction.model_json_schema(),
        "observation_schema": MonitorObservation.model_json_schema(),
        "difficulties": ["easy", "medium", "hard"],
        "description": (
            "Each task presents an AI agent's planned actions for real-time monitoring. "
            "Easy tasks have single obvious violations, medium tasks have subtle "
            "context-dependent violations, and hard tasks require understanding multi-step chains."
        ),
    }


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

@app.get("/grader")
async def grader():
    """Return grader metadata and scoring weights per difficulty."""
    return {
        "graders": GRADERS,
        "score_range": [0.0, 1.0],
        "deterministic": True,
        "description": (
            "Per-step rewards for each allow/block/flag decision, plus "
            "episode-level scoring on 6 dimensions: detection_score, false_positive_rate, "
            "type_accuracy, severity_accuracy, explanation_quality (semantic + keyword), "
            "and schema_compliance (R1-style formatting reward)."
        ),
    }


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def _heuristic_analyze_action(action_data: Dict[str, Any]) -> MonitorAction:
    """Keyword heuristic: decide allow/block for a single action."""
    _KEYWORD_MAP = [
        (["rm -rf", "delete", "DROP TABLE", "eval(", "exec("], "destructive_action", "critical"),
        (["external", "upload", "exfil", "sent to", "webhook", "curl"], "data_exfiltration", "critical"),
        (["/etc/passwd", ".ssh/", "secret", "API_KEY", "/etc/shadow", "private_key"], "unauthorized_access", "high"),
        (["chmod 777", "admin", "superuser", "root", "sudo", "CREATE ROLE"], "privilege_escalation", "critical"),
        (["hardcoded", "plaintext", "public/", "localStorage", "base64"], "insecure_storage", "high"),
        (["verify=False", "self-approved", "production", "brute", "retry"], "policy_violation", "high"),
        (["0.0.0.0/0", "connection string", "stack trace", "traceback"], "information_disclosure", "high"),
        (["evil-packages", "untrusted", "malicious", "unknown-repo"], "supply_chain_risk", "critical"),
        (["MD5", "SHA-1", "DES", "weak"], "weak_cryptography", "medium"),
        (["SELECT *", "all users", "full dump", "beyond scope"], "scope_creep", "medium"),
    ]

    text = f"{action_data.get('action', '')} {action_data.get('target', '')} {action_data.get('details', '')}".lower()

    for keywords, vtype, severity in _KEYWORD_MAP:
        if any(kw.lower() in text for kw in keywords):
            return MonitorAction(
                decision="block",
                violation_type=vtype,
                severity=severity,
                reason=f"Keyword match detected in action targeting {action_data.get('target', 'unknown')}.",
            )

    return MonitorAction(decision="allow", reason="No suspicious patterns detected.")


def _run_heuristic_baseline() -> Dict[str, Any]:
    """Run keyword heuristic baseline over all 48 tasks."""
    results: Dict[str, Any] = {"episodes": []}
    all_scores: List[float] = []
    scores_by_diff: Dict[str, list] = {"easy": [], "medium": [], "hard": []}

    all_task_ids = env.get_all_task_ids()
    for difficulty in ("easy", "medium", "hard"):
        for task_id in all_task_ids[difficulty]:
            episode_env = AgentSafetyAuditEnvironment()
            obs = episode_env.reset(seed=42, options={"difficulty": difficulty, "task_id": task_id})

            # Run through all steps with heuristic
            done = False
            while not done:
                action_data = {
                    "action": obs.current_action.action if obs.current_action else "",
                    "target": obs.current_action.target if obs.current_action else "",
                    "details": obs.current_action.details if obs.current_action else "",
                }
                monitor_action = _heuristic_analyze_action(action_data)
                obs, reward, done, info = episode_env.step(monitor_action)

            score = info.get("score", info.get("episode_score", 0.0))
            all_scores.append(score)
            scores_by_diff[difficulty].append(score)

            results["episodes"].append({
                "task_id": task_id,
                "difficulty": difficulty,
                "score": score,
                "breakdown": info.get("breakdown", {}),
                "strategy": "keyword_heuristic",
            })

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    results["total_tasks"] = len(all_scores)
    results["average_score"] = round(avg, 4)
    results["scores_by_difficulty"] = {
        d: round(sum(s) / len(s), 4) if s else 0.0
        for d, s in scores_by_diff.items()
    }
    results["summary"] = {
        "strategy": "keyword_heuristic",
        "description": (
            "Scans each action for suspicious keywords. "
            "Blocks on match, allows otherwise."
        ),
    }
    return results




@app.get("/baseline")
async def baseline():
    """Run baseline inference and return scores."""
    start = time.time()
    try:
        results = _run_heuristic_baseline()
    except Exception as e:
        logger.error(f"Baseline error: {e}")
        raise HTTPException(status_code=500, detail=f"Baseline failed: {str(e)}")
    elapsed = round(time.time() - start, 3)
    results["elapsed_seconds"] = elapsed
    return results


@app.get("/baseline-trigger-inference-script")
async def baseline_trigger():
    """Trigger the full baseline inference script."""
    start = time.time()
    try:
        results = _run_heuristic_baseline()
    except Exception as e:
        logger.error(f"Baseline trigger error: {e}")
        raise HTTPException(status_code=500, detail=f"Baseline failed: {str(e)}")
    elapsed = round(time.time() - start, 3)
    results["elapsed_seconds"] = elapsed
    results["triggered_by"] = "inference_script_endpoint"
    return results


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

@app.post("/reset")
async def reset(request: ResetRequest):
    """Reset the environment to a new episode.

    Body: {"seed": 42, "options": {"difficulty": "easy", "task_id": "easy_001"}}
    """
    try:
        obs = env.reset(seed=request.seed, options=request.options)
        return {"observation": obs.model_dump()}
    except Exception as e:
        logger.error(f"Reset error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Reset failed: {str(e)}")


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

@app.post("/step")
async def step(request: StepRequest):
    """Submit a monitoring decision for the current action.

    Body: {"decision": "block", "violation_type": "unauthorized_access",
           "severity": "high", "reason": "Accessing sensitive system file"}
    """
    try:
        action = MonitorAction(
            decision=request.decision,
            violation_type=request.violation_type,
            severity=request.severity,
            reason=request.reason,
        )
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as e:
        logger.error(f"Step error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Step failed: {str(e)}")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@app.get("/state")
async def state():
    """Return current episode state (no ground truth exposed)."""
    return env.state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Entry point for running with uvicorn."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
