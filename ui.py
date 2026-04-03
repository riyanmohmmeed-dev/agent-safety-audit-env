# Copyright (c) 2026 Neural Nomads. All rights reserved.
# AI Agent Safety Monitor — Gradio Web UI

"""Gradio interface for the AI Agent Safety Monitor.

Minimal, dark, beautiful. One card per action, three buttons, instant feedback.
Mounted on FastAPI at `/` so judges see the UI immediately on HuggingFace Spaces.
"""

from __future__ import annotations

import gradio as gr
import json
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# UI State Management
# ---------------------------------------------------------------------------

class UIState:
    """Tracks the current episode state for the UI."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.task_id = ""
        self.difficulty = ""
        self.agent_task = ""
        self.permissions: List[str] = []
        self.policy_rules: List[str] = []
        self.system_context: Dict[str, Any] = {}
        self.risk_level = ""
        self.incident_ref = ""
        self.current_action = None
        self.current_step = 0
        self.total_steps = 0
        self.previous_decisions: List[Dict] = []
        self.feedback = ""
        self.step_reward = 0.0
        self.episode_score = 0.0
        self.done = False
        self.breakdown: Dict[str, float] = {}
        self.decision_history: List[Dict] = []


ui_state = UIState()


# ---------------------------------------------------------------------------
# Environment Interaction (direct import, no HTTP)
# ---------------------------------------------------------------------------

def get_env():
    """Get the global environment instance from the server module."""
    from server.app import env
    return env


def start_episode(difficulty: str) -> Tuple[str, str, str, str, str, str, str, str]:
    """Start a new episode and return the first observation."""
    env = get_env()
    ui_state.reset()

    obs = env.reset(seed=None, options={"difficulty": difficulty})
    ui_state.active = True
    ui_state.task_id = obs.task_id
    ui_state.difficulty = obs.difficulty
    ui_state.agent_task = obs.agent_task
    ui_state.permissions = obs.agent_permissions
    ui_state.policy_rules = obs.policy_rules
    ui_state.system_context = obs.system_context or {}
    ui_state.risk_level = obs.risk_level or ""
    ui_state.incident_ref = obs.incident_ref or ""
    ui_state.current_action = obs.current_action
    ui_state.current_step = obs.current_step
    ui_state.total_steps = obs.total_steps

    return _build_ui_outputs()


def submit_decision(
    decision: str,
    violation_type: str = "",
    severity: str = "",
    reason: str = "",
) -> Tuple[str, str, str, str, str, str, str, str]:
    """Submit a decision and return the updated observation."""
    if not ui_state.active:
        return _build_idle_outputs("No active episode. Pick a difficulty to start.")

    from models import MonitorAction
    env = get_env()

    action = MonitorAction(
        decision=decision,
        violation_type=violation_type if decision == "block" else "",
        severity=severity if decision == "block" else "",
        reason=reason,
    )

    obs, reward, done, info = env.step(action)
    ui_state.step_reward = reward
    ui_state.feedback = info.get("step_feedback", "")

    # Record decision
    ui_state.decision_history.append({
        "step": ui_state.current_step,
        "decision": decision,
        "reward": round(reward, 3),
    })

    if done:
        ui_state.done = True
        ui_state.episode_score = info.get("score", info.get("episode_score", 0.0))
        ui_state.breakdown = info.get("breakdown", {})
        return _build_done_outputs()
    else:
        ui_state.current_action = obs.current_action
        ui_state.current_step = obs.current_step
        return _build_ui_outputs()


# ---------------------------------------------------------------------------
# UI Output Builders
# ---------------------------------------------------------------------------

def _difficulty_badge(diff: str) -> str:
    colors = {
        "easy": "#22c55e",
        "medium": "#f59e0b",
        "grey_area": "#a855f7",
        "hard": "#ef4444",
    }
    labels = {
        "easy": "EASY",
        "medium": "MEDIUM",
        "grey_area": "GREY AREA",
        "hard": "HARD",
    }
    c = colors.get(diff, "#6b7280")
    l = labels.get(diff, diff.upper())
    return f'<span style="background:{c}; color:#fff; padding:4px 14px; border-radius:20px; font-size:13px; font-weight:700; letter-spacing:1px;">{l}</span>'


def _reward_indicator(reward: float) -> str:
    if reward > 0.05:
        return f'<span style="color:#22c55e; font-weight:700;">+{reward:.2f} ✓</span>'
    elif reward < -0.05:
        return f'<span style="color:#ef4444; font-weight:700;">{reward:.2f} ✗</span>'
    else:
        return f'<span style="color:#6b7280; font-weight:700;">{reward:.2f}</span>'


def _build_action_card() -> str:
    """Build the main action card HTML."""
    action = ui_state.current_action
    if action is None:
        return ""

    step_label = f"Step {ui_state.current_step} of {ui_state.total_steps}"
    progress_pct = (ui_state.current_step / ui_state.total_steps) * 100

    action_type_colors = {
        "read_file": "#3b82f6",
        "write_file": "#f59e0b",
        "execute_command": "#ef4444",
        "call_api": "#8b5cf6",
        "query_database": "#06b6d4",
    }
    action_color = action_type_colors.get(action.action, "#6b7280")

    # Build decision history dots
    dots = ""
    for d in ui_state.decision_history:
        dot_colors = {"allow": "#22c55e", "block": "#ef4444", "flag": "#f59e0b"}
        dot_color = dot_colors.get(d["decision"], "#6b7280")
        step_num = d["step"]
        dec_name = d["decision"]
        dec_reward = d["reward"]
        dots += f'<span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:{dot_color}; margin:0 3px;" title="Step {step_num}: {dec_name} ({dec_reward:+.2f})"></span>'

    feedback_html = ""
    if ui_state.feedback and ui_state.decision_history:
        last = ui_state.decision_history[-1]
        border_colors = {"allow": "#22c55e", "block": "#ef4444", "flag": "#f59e0b"}
        border_c = border_colors.get(last["decision"], "#6b7280")
        last_dec_upper = last["decision"].upper()
        last_reward_html = _reward_indicator(last["reward"])
        feedback_text = ui_state.feedback
        feedback_html = f'''
        <div style="margin-top:16px; padding:12px 16px; border-radius:10px; background:rgba(255,255,255,0.03); border-left:3px solid {border_c};">
            <div style="font-size:12px; color:#9ca3af; margin-bottom:4px;">Last Decision: {last_dec_upper} → {last_reward_html}</div>
            <div style="font-size:13px; color:#d1d5db;">{feedback_text}</div>
        </div>'''

    return f'''
    <div style="position:relative;">
        <!-- Progress bar -->
        <div style="height:3px; background:#1e293b; border-radius:2px; margin-bottom:24px;">
            <div style="height:3px; background:linear-gradient(90deg, #06b6d4, #8b5cf6); border-radius:2px; width:{progress_pct}%; transition:width 0.5s ease;"></div>
        </div>

        <!-- Header -->
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
            <div>
                <span style="font-size:13px; color:#64748b; font-weight:500;">{step_label}</span>
                <span style="margin-left:12px;">{_difficulty_badge(ui_state.difficulty)}</span>
            </div>
            <span style="font-size:12px; color:#475569; font-family:monospace;">{ui_state.task_id}</span>
        </div>

        <!-- Decision history dots -->
        <div style="margin-bottom:20px; min-height:16px;">{dots}</div>

        <!-- Action Card -->
        <div style="background:#0f172a; border:1px solid #1e293b; border-radius:16px; padding:28px; margin-bottom:16px;">

            <!-- Action type badge -->
            <div style="margin-bottom:16px;">
                <span style="background:{action_color}20; color:{action_color}; padding:5px 12px; border-radius:8px; font-size:12px; font-weight:600; font-family:monospace; letter-spacing:0.5px;">{action.action}</span>
            </div>

            <!-- Target -->
            <div style="font-family:'JetBrains Mono', 'Fira Code', monospace; font-size:15px; color:#e2e8f0; margin-bottom:16px; word-break:break-all; line-height:1.5;">
                {action.target}
            </div>

            <!-- Details -->
            <div style="font-size:14px; color:#94a3b8; line-height:1.7;">
                {action.details}
            </div>
        </div>

        {feedback_html}
    </div>'''


def _build_context_panel() -> str:
    """Build the context side panel."""
    rules_html = "".join(
        f'<li style="margin-bottom:8px; color:#94a3b8; font-size:13px; line-height:1.5;">{rule}</li>'
        for rule in ui_state.policy_rules
    )

    perms_html = "".join(
        f'<span style="background:#1e293b; color:#94a3b8; padding:3px 10px; border-radius:6px; font-size:12px; margin:2px 4px 2px 0; display:inline-block; font-family:monospace;">{p}</span>'
        for p in ui_state.permissions
    )

    risk_colors = {"low": "#22c55e", "medium": "#f59e0b", "high": "#f97316", "critical": "#ef4444"}
    risk_c = risk_colors.get(ui_state.risk_level, "#6b7280")

    return f'''
    <div style="font-size:14px;">
        <!-- Agent Task -->
        <div style="margin-bottom:20px;">
            <div style="font-size:11px; color:#475569; font-weight:600; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:8px;">Mission</div>
            <div style="color:#cbd5e1; line-height:1.6;">{ui_state.agent_task}</div>
        </div>

        <!-- Risk Level -->
        <div style="margin-bottom:20px;">
            <div style="font-size:11px; color:#475569; font-weight:600; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:8px;">Risk Level</div>
            <span style="color:{risk_c}; font-weight:700; font-size:13px;">{ui_state.risk_level.upper() if ui_state.risk_level else "—"}</span>
        </div>

        <!-- Permissions -->
        <div style="margin-bottom:20px;">
            <div style="font-size:11px; color:#475569; font-weight:600; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:8px;">Permissions</div>
            <div>{perms_html}</div>
        </div>

        <!-- Policy Rules -->
        <div>
            <div style="font-size:11px; color:#475569; font-weight:600; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:8px;">Policy Rules</div>
            <ul style="padding-left:16px; margin:0;">{rules_html}</ul>
        </div>
    </div>'''


def _build_ui_outputs():
    """Return all UI component values for an active episode."""
    action_card = _build_action_card()
    context = _build_context_panel()
    # Return: action_html, context_html, status, btn_vis, block_opts_vis, reason_val, step_info, done_panel
    return (
        action_card,
        context,
        "",
        gr.update(visible=True),   # decision buttons
        gr.update(visible=False),  # block options
        "",                        # reason cleared
        f"Step {ui_state.current_step}/{ui_state.total_steps}",
        gr.update(visible=False),  # done panel hidden
    )


def _build_idle_outputs(msg: str = ""):
    """Return UI state for no active episode."""
    welcome = f'''
    <div style="text-align:center; padding:60px 20px;">
        <div style="font-size:48px; margin-bottom:16px;">🛡️</div>
        <div style="font-size:22px; color:#e2e8f0; font-weight:600; margin-bottom:10px;">AI Agent Safety Monitor</div>
        <div style="font-size:14px; color:#64748b; max-width:400px; margin:0 auto; line-height:1.7;">
            Review an AI agent's actions in real-time.<br/>
            Decide: <span style="color:#22c55e; font-weight:600;">ALLOW</span>,
            <span style="color:#ef4444; font-weight:600;">BLOCK</span>, or
            <span style="color:#f59e0b; font-weight:600;">FLAG</span> each action.
        </div>
        <div style="font-size:13px; color:#475569; margin-top:20px;">{msg}</div>
    </div>'''
    return (
        welcome, "", "", gr.update(visible=False), gr.update(visible=False),
        "", "", gr.update(visible=False),
    )


def _build_done_outputs():
    """Return UI state when episode is complete."""
    score = ui_state.episode_score
    score_color = "#22c55e" if score >= 0.7 else "#f59e0b" if score >= 0.4 else "#ef4444"

    # Build breakdown bars
    bars = ""
    for dim, val in ui_state.breakdown.items():
        label = dim.replace("_", " ").title()
        bar_color = "#22c55e" if val >= 0.7 else "#f59e0b" if val >= 0.4 else "#ef4444"
        bars += f'''
        <div style="margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:12px; color:#94a3b8;">{label}</span>
                <span style="font-size:12px; color:#e2e8f0; font-weight:600;">{val:.2f}</span>
            </div>
            <div style="height:6px; background:#1e293b; border-radius:3px;">
                <div style="height:6px; background:{bar_color}; border-radius:3px; width:{val*100}%; transition:width 0.8s ease;"></div>
            </div>
        </div>'''

    # Decision timeline
    timeline = ""
    for d in ui_state.decision_history:
        t_colors = {"allow": "#22c55e", "block": "#ef4444", "flag": "#f59e0b"}
        t_icons = {"allow": "✓", "block": "✗", "flag": "⚑"}
        d_color = t_colors.get(d["decision"], "#6b7280")
        d_icon = t_icons.get(d["decision"], "?")
        step_num = d["step"]
        dec_name = d["decision"]
        timeline += f'<span style="display:inline-flex; align-items:center; justify-content:center; width:32px; height:32px; border-radius:50%; background:{d_color}20; color:{d_color}; font-size:14px; margin:0 4px; font-weight:700;" title="Step {step_num}: {dec_name}">{d_icon}</span>'

    done_html = f'''
    <div style="text-align:center; padding:40px 20px;">
        <!-- Score circle -->
        <div style="display:inline-flex; align-items:center; justify-content:center; width:120px; height:120px; border-radius:50%; border:4px solid {score_color}; margin-bottom:20px;">
            <span style="font-size:32px; font-weight:700; color:{score_color};">{score:.0%}</span>
        </div>

        <div style="font-size:18px; color:#e2e8f0; font-weight:600; margin-bottom:6px;">Episode Complete</div>
        <div style="font-size:13px; color:#64748b; margin-bottom:24px;">
            {ui_state.task_id} · {_difficulty_badge(ui_state.difficulty)} · {ui_state.total_steps} steps
        </div>

        <!-- Decision timeline -->
        <div style="margin-bottom:28px;">{timeline}</div>

        <!-- Breakdown bars -->
        <div style="max-width:400px; margin:0 auto; text-align:left;">
            {bars}
        </div>

        <div style="margin-top:24px; font-size:13px; color:#475569;">Pick a new difficulty to play again.</div>
    </div>'''

    return (
        done_html,
        "",
        "",
        gr.update(visible=False),   # hide decision buttons
        gr.update(visible=False),
        "",
        "Done",
        gr.update(visible=False),
    )


# ---------------------------------------------------------------------------
# Event Handlers
# ---------------------------------------------------------------------------

def on_difficulty_click(difficulty: str):
    return start_episode(difficulty)


def on_allow_click(reason: str):
    return submit_decision("allow", reason=reason)


def on_block_click(violation_type: str, severity: str, reason: str):
    return submit_decision("block", violation_type=violation_type, severity=severity, reason=reason)


def on_flag_click(reason: str):
    return submit_decision("flag", reason=reason)


def toggle_block_opts(visible: bool):
    return gr.update(visible=visible)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* === ANIMATED BACKGROUND === */
.gradio-container {
    background: #020617 !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    position: relative;
}

.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(6, 182, 212, 0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(139, 92, 246, 0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 80%, rgba(34, 197, 94, 0.03) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
    animation: bgPulse 8s ease-in-out infinite alternate;
}

@keyframes bgPulse {
    0% { opacity: 0.6; }
    100% { opacity: 1; }
}

.dark {
    --background-fill-primary: #020617 !important;
    --background-fill-secondary: #0f172a !important;
    --border-color-primary: #1e293b !important;
    --body-text-color: #e2e8f0 !important;
    --body-text-color-subdued: #64748b !important;
}

#main-panel {
    background: #020617 !important;
    border: none !important;
}

/* === DIFFICULTY BUTTONS with glow === */
.diff-btn {
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    background: #0f172a !important;
    min-width: 100px !important;
    position: relative;
    overflow: hidden;
}

.diff-btn::after {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    width: 0; height: 0;
    border-radius: 50%;
    background: rgba(255,255,255,0.1);
    transform: translate(-50%,-50%);
    transition: width 0.4s, height 0.4s;
}
.diff-btn:hover::after {
    width: 200px; height: 200px;
}

.diff-btn:hover {
    transform: translateY(-2px) scale(1.02) !important;
    border-color: #334155 !important;
}

.diff-easy { color: #22c55e !important; }
.diff-easy:hover { background: rgba(34,197,94,0.1) !important; border-color: #22c55e !important; box-shadow: 0 0 20px rgba(34,197,94,0.2) !important; }

.diff-medium { color: #f59e0b !important; }
.diff-medium:hover { background: rgba(245,158,11,0.1) !important; border-color: #f59e0b !important; box-shadow: 0 0 20px rgba(245,158,11,0.2) !important; }

.diff-grey { color: #a855f7 !important; }
.diff-grey:hover { background: rgba(168,85,247,0.1) !important; border-color: #a855f7 !important; box-shadow: 0 0 20px rgba(168,85,247,0.2) !important; }

.diff-hard { color: #ef4444 !important; }
.diff-hard:hover { background: rgba(239,68,68,0.1) !important; border-color: #ef4444 !important; box-shadow: 0 0 20px rgba(239,68,68,0.2) !important; }

/* === DECISION BUTTONS with pulse animation === */
@keyframes pulseGreen {
    0%, 100% { box-shadow: 0 4px 14px rgba(34, 197, 94, 0.25); }
    50% { box-shadow: 0 4px 24px rgba(34, 197, 94, 0.4); }
}
@keyframes pulseRed {
    0%, 100% { box-shadow: 0 4px 14px rgba(239, 68, 68, 0.25); }
    50% { box-shadow: 0 4px 24px rgba(239, 68, 68, 0.4); }
}
@keyframes pulseAmber {
    0%, 100% { box-shadow: 0 4px 14px rgba(245, 158, 11, 0.25); }
    50% { box-shadow: 0 4px 24px rgba(245, 158, 11, 0.4); }
}

.allow-btn {
    background: linear-gradient(135deg, #15803d, #22c55e) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 28px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    animation: pulseGreen 3s ease-in-out infinite;
}
.allow-btn:hover {
    transform: translateY(-3px) scale(1.03) !important;
    box-shadow: 0 8px 30px rgba(34, 197, 94, 0.4) !important;
    animation: none;
}

.block-btn {
    background: linear-gradient(135deg, #b91c1c, #ef4444) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 28px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    animation: pulseRed 3s ease-in-out infinite;
}
.block-btn:hover {
    transform: translateY(-3px) scale(1.03) !important;
    box-shadow: 0 8px 30px rgba(239, 68, 68, 0.4) !important;
    animation: none;
}

.flag-btn {
    background: linear-gradient(135deg, #b45309, #f59e0b) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 28px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    animation: pulseAmber 3s ease-in-out infinite;
}
.flag-btn:hover {
    transform: translateY(-3px) scale(1.03) !important;
    box-shadow: 0 8px 30px rgba(245, 158, 11, 0.4) !important;
    animation: none;
}

/* === INPUTS === */
textarea, select, input {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
textarea:focus, select:focus, input:focus {
    border-color: #06b6d4 !important;
    box-shadow: 0 0 0 3px rgba(6,182,212,0.1) !important;
    outline: none !important;
}

/* === LABELS === */
label {
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
}

/* === HIDE FOOTER === */
footer { display: none !important; }

/* === SCROLLBAR === */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #020617; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* === ANIMATED GRADIENT TEXT (for header) === */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.gradient-title {
    background: linear-gradient(135deg, #06b6d4, #8b5cf6, #22c55e, #06b6d4);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 6s ease infinite;
}

/* === LIVE INDICATOR PULSE === */
@keyframes livePulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.5); }
}
.live-dot {
    animation: livePulse 2s ease-in-out infinite;
}
"""


# ---------------------------------------------------------------------------
# Build Gradio Interface
# ---------------------------------------------------------------------------

def create_ui() -> gr.Blocks:
    """Create and return the Gradio Blocks interface."""

    with gr.Blocks(
        title="AI Agent Safety Monitor",
    ) as demo:

        # Inject CSS via HTML (Gradio 6.0+ moved css/theme to launch())
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # --- Header ---
        gr.HTML('''
        <div style="display:flex; align-items:center; justify-content:space-between; padding:16px 0; border-bottom:1px solid #1e293b; margin-bottom:20px;">
            <div style="display:flex; align-items:center; gap:12px;">
                <span style="font-size:24px;">🛡️</span>
                <div>
                    <div class="gradient-title" style="font-size:17px; font-weight:700; letter-spacing:-0.3px;">Agent Safety Monitor</div>
                    <div style="font-size:11px; color:#475569;">69 tasks · 4 difficulties · deterministic grading</div>
                </div>
            </div>
            <div style="display:flex; gap:6px; align-items:center;">
                <span class="live-dot" style="width:8px; height:8px; border-radius:50%; background:#22c55e; display:inline-block;"></span>
                <span style="font-size:12px; color:#64748b;">Live</span>
            </div>
        </div>
        ''')

        # --- Difficulty Selector ---
        with gr.Row():
            easy_btn = gr.Button("EASY", elem_classes=["diff-btn", "diff-easy"], size="sm")
            medium_btn = gr.Button("MEDIUM", elem_classes=["diff-btn", "diff-medium"], size="sm")
            grey_btn = gr.Button("GREY AREA", elem_classes=["diff-btn", "diff-grey"], size="sm")
            hard_btn = gr.Button("HARD", elem_classes=["diff-btn", "diff-hard"], size="sm")

        gr.HTML('<div style="height:12px;"></div>')

        # --- Main Content ---
        with gr.Row():
            # Left: Action Card (2/3 width)
            with gr.Column(scale=2):
                action_html = gr.HTML(
                    value=_build_idle_outputs()[0],
                    elem_id="action-card",
                )

                # Decision buttons
                with gr.Group(visible=False) as decision_group:
                    reason_input = gr.Textbox(
                        placeholder="Why are you making this decision? (optional but improves score)",
                        label="Reason",
                        lines=1,
                        max_lines=2,
                    )

                    # Block-specific options (hidden by default)
                    with gr.Group(visible=False) as block_opts:
                        with gr.Row():
                            violation_dd = gr.Dropdown(
                                choices=[
                                    "unauthorized_access", "data_exfiltration",
                                    "privilege_escalation", "insecure_storage",
                                    "policy_violation", "scope_creep",
                                    "weak_cryptography", "destructive_action",
                                    "information_disclosure", "supply_chain_risk",
                                ],
                                label="Violation Type",
                                value="policy_violation",
                            )
                            severity_dd = gr.Dropdown(
                                choices=["low", "medium", "high", "critical"],
                                label="Severity",
                                value="high",
                            )

                    with gr.Row():
                        allow_btn = gr.Button("✓  ALLOW", elem_classes=["allow-btn"], scale=1)
                        block_btn = gr.Button("✗  BLOCK", elem_classes=["block-btn"], scale=1)
                        flag_btn = gr.Button("⚑  FLAG", elem_classes=["flag-btn"], scale=1)

            # Right: Context Panel (1/3 width)
            with gr.Column(scale=1):
                context_html = gr.HTML(value="", elem_id="context-panel")

        # Hidden state elements
        status_text = gr.Textbox(visible=False)
        step_info = gr.Textbox(visible=False)
        done_panel = gr.HTML(visible=False)

        # --- All UI outputs ---
        all_outputs = [
            action_html, context_html, status_text,
            decision_group, block_opts, reason_input,
            step_info, done_panel,
        ]

        # --- Wire difficulty buttons ---
        easy_btn.click(fn=lambda: start_episode("easy"), outputs=all_outputs)
        medium_btn.click(fn=lambda: start_episode("medium"), outputs=all_outputs)
        grey_btn.click(fn=lambda: start_episode("grey_area"), outputs=all_outputs)
        hard_btn.click(fn=lambda: start_episode("hard"), outputs=all_outputs)

        # --- Wire decision buttons ---
        allow_btn.click(
            fn=lambda r: on_allow_click(r),
            inputs=[reason_input],
            outputs=all_outputs,
        )
        block_btn.click(
            fn=lambda vt, sv, r: on_block_click(vt, sv, r),
            inputs=[violation_dd, severity_dd, reason_input],
            outputs=all_outputs,
        )
        flag_btn.click(
            fn=lambda r: on_flag_click(r),
            inputs=[reason_input],
            outputs=all_outputs,
        )

        # Show block options when block button is focused
        block_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[block_opts],
        )

    return demo
