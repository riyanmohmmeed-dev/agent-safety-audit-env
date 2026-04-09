# Copyright (c) 2026 Neural Nomads. All rights reserved.
# AI Agent Safety Monitor — SENTINEL UI with animated guardian character.

"""Gradio interface for the AI Agent Safety Monitor.

SENTINEL — a premium cybersecurity dashboard featuring an animated guardian
character that reacts to each decision: analyzing (cyan), allowing (green),
blocking (red), or flagging (amber). Built with liquid glass design language.
"""

from __future__ import annotations

import gradio as gr
import json
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Character Image Loading (base64 embed for Gradio compatibility)
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent / "static"

def _load_character_b64(name: str) -> str:
    """Load a character PNG as base64 data URI for embedding in HTML."""
    path = _STATIC_DIR / f"sentinel_{name}.png"
    if path.exists():
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode()
        return f"data:image/png;base64,{b64}"
    return ""

# Pre-load all character states
CHAR_ANALYZING = _load_character_b64("analyzing")
CHAR_ALLOWING = _load_character_b64("allowing")
CHAR_BLOCKING = _load_character_b64("blocking")
CHAR_FLAGGING = _load_character_b64("flagging")


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
        self.last_decision = ""  # Track for character state


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
    ui_state.last_decision = "analyzing"

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
    ui_state.last_decision = decision

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
# Character HTML Builder
# ---------------------------------------------------------------------------

def _build_character(state: str = "analyzing", size: int = 180) -> str:
    """Build the animated character HTML for the given state."""
    images = {
        "analyzing": CHAR_ANALYZING,
        "allow": CHAR_ALLOWING,
        "block": CHAR_BLOCKING,
        "flag": CHAR_FLAGGING,
    }
    glows = {
        "analyzing": "0 0 40px rgba(6,182,212,0.4), 0 0 80px rgba(6,182,212,0.15)",
        "allow": "0 0 40px rgba(34,197,94,0.5), 0 0 80px rgba(34,197,94,0.2)",
        "block": "0 0 40px rgba(239,68,68,0.5), 0 0 80px rgba(239,68,68,0.2)",
        "flag": "0 0 40px rgba(245,158,11,0.5), 0 0 80px rgba(245,158,11,0.2)",
    }
    labels = {
        "analyzing": "ANALYZING",
        "allow": "CLEARED",
        "block": "BLOCKED",
        "flag": "FLAGGED",
    }
    colors = {
        "analyzing": "#06b6d4",
        "allow": "#22c55e",
        "block": "#ef4444",
        "flag": "#f59e0b",
    }

    img_src = images.get(state, images["analyzing"])
    glow = glows.get(state, glows["analyzing"])
    label = labels.get(state, "ANALYZING")
    color = colors.get(state, "#06b6d4")

    return f'''
    <div class="sentinel-character" style="text-align:center; padding:8px 0;">
        <div style="position:relative; display:inline-block;">
            <img src="{img_src}" alt="Sentinel {state}"
                 style="width:{size}px; height:{size}px; object-fit:contain;
                        filter:drop-shadow({glow});
                        animation:sentinelFloat 3s ease-in-out infinite;
                        transition:all 0.5s cubic-bezier(0.4,0,0.2,1);" />
        </div>
        <div style="margin-top:8px;">
            <span style="background:{color}20; color:{color}; border:1px solid {color}40;
                         padding:4px 16px; border-radius:20px; font-size:11px;
                         font-weight:700; letter-spacing:2px;
                         animation:statusPulse 2s ease-in-out infinite;">{label}</span>
        </div>
    </div>'''


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
    return f'<span style="background:{c}20; color:{c}; border:1px solid {c}40; padding:4px 14px; border-radius:20px; font-size:11px; font-weight:700; letter-spacing:1.5px;">{l}</span>'


def _reward_indicator(reward: float) -> str:
    if reward > 0.05:
        return f'<span style="color:#22c55e; font-weight:700;">+{reward:.2f}</span>'
    elif reward < -0.05:
        return f'<span style="color:#ef4444; font-weight:700;">{reward:.2f}</span>'
    else:
        return f'<span style="color:#6b7280; font-weight:700;">{reward:.2f}</span>'


def _build_action_card() -> str:
    """Build the main action card HTML with scanning animation."""
    action = ui_state.current_action
    if action is None:
        return ""

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
        dots += f'<span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:{dot_color}; margin:0 3px; box-shadow:0 0 6px {dot_color}80;" title="Step {d["step"]}: {d["decision"]} ({d["reward"]:+.2f})"></span>'

    # Feedback from last decision
    feedback_html = ""
    if ui_state.feedback and ui_state.decision_history:
        last = ui_state.decision_history[-1]
        border_colors = {"allow": "#22c55e", "block": "#ef4444", "flag": "#f59e0b"}
        border_c = border_colors.get(last["decision"], "#6b7280")
        feedback_html = f'''
        <div style="margin-top:16px; padding:12px 16px; border-radius:12px;
                    background:rgba(255,255,255,0.02); border-left:3px solid {border_c};
                    backdrop-filter:blur(10px);">
            <div style="font-size:11px; color:#64748b; margin-bottom:4px; letter-spacing:0.5px;">
                LAST: {last["decision"].upper()} {_reward_indicator(last["reward"])}
            </div>
            <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">{ui_state.feedback}</div>
        </div>'''

    # Character state based on last decision
    char_state = ui_state.last_decision if ui_state.last_decision in ["allow", "block", "flag"] else "analyzing"

    return f'''
    <div style="position:relative;">
        <!-- Header bar -->
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:12px; color:#475569; font-weight:700; letter-spacing:1px;">
                    STEP {ui_state.current_step} / {ui_state.total_steps}
                </span>
                {_difficulty_badge(ui_state.difficulty)}
            </div>
            <span style="font-size:11px; color:#334155; font-family:'JetBrains Mono',monospace;">{ui_state.task_id}</span>
        </div>

        <!-- Animated progress bar -->
        <div style="height:3px; background:#0f172a; border-radius:2px; margin-bottom:16px; overflow:hidden; position:relative;">
            <div style="height:100%; background:linear-gradient(90deg, #06b6d4, #8b5cf6, #06b6d4);
                        background-size:200% 100%; width:{progress_pct}%;
                        transition:width 0.5s cubic-bezier(0.4,0,0.2,1);
                        animation:progressShimmer 2s linear infinite;"></div>
        </div>

        <!-- Decision dots -->
        <div style="margin-bottom:16px; min-height:14px; display:flex; gap:4px; align-items:center;">{dots}</div>

        <!-- Character + Action Card side by side -->
        <div style="display:flex; gap:16px; align-items:flex-start;">

            <!-- Sentinel Character -->
            <div style="flex-shrink:0; width:180px;">
                {_build_character(char_state, 160)}
            </div>

            <!-- Action Card (glass panel with scan line) -->
            <div style="flex:1; background:rgba(15,23,42,0.6); backdrop-filter:blur(20px);
                        border:1px solid rgba(255,255,255,0.06); border-radius:16px;
                        padding:24px; position:relative; overflow:hidden;
                        box-shadow:0 8px 32px rgba(0,0,0,0.4);">

                <!-- Scan line animation -->
                <div style="position:absolute; top:0; left:0; right:0; height:2px;
                            background:linear-gradient(90deg, transparent, {action_color}80, transparent);
                            animation:scanLine 3s ease-in-out infinite;"></div>

                <!-- Action type badge -->
                <div style="margin-bottom:20px;">
                    <span style="background:{action_color}15; color:{action_color}; border:1px solid {action_color}30;
                                 padding:5px 14px; border-radius:8px; font-size:11px; font-weight:700;
                                 font-family:'JetBrains Mono',monospace; letter-spacing:0.5px;">{action.action}</span>
                </div>

                <!-- Target path -->
                <div style="background:rgba(0,0,0,0.3); border-left:3px solid {action_color};
                            padding:12px 16px; border-radius:4px 10px 10px 4px;
                            font-family:'JetBrains Mono','Fira Code',monospace; font-size:13px;
                            color:#f1f5f9; margin-bottom:20px; word-break:break-all; line-height:1.6;">
                    {action.target}
                </div>

                <!-- Details -->
                <div style="font-size:14px; color:#94a3b8; line-height:1.7;">
                    {action.details}
                </div>
            </div>
        </div>

        {feedback_html}
    </div>'''


def _build_context_panel() -> str:
    """Build the context side panel with glass tiles."""
    rules_html = "".join(
        f'<li style="margin-bottom:8px; color:#94a3b8; font-size:12px; line-height:1.6; padding-left:16px; position:relative;">'
        f'<span style="position:absolute; left:0; top:7px; width:5px; height:5px; border-radius:50%; background:#06b6d4; box-shadow:0 0 6px #06b6d480;"></span>{rule}</li>'
        for rule in ui_state.policy_rules
    )

    perms_html = "".join(
        f'<span style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); color:#cbd5e1; padding:3px 10px; border-radius:6px; font-size:11px; margin:0 4px 4px 0; display:inline-block; font-family:monospace;">{p}</span>'
        for p in ui_state.permissions
    )

    risk_colors = {"low": "#22c55e", "medium": "#f59e0b", "high": "#f97316", "critical": "#ef4444"}
    risk_c = risk_colors.get(ui_state.risk_level, "#64748b")
    risk_txt = ui_state.risk_level.upper() if ui_state.risk_level else "N/A"

    return f'''
    <div style="display:flex; flex-direction:column; gap:12px;">
        <!-- Mission tile -->
        <div class="glass-tile" style="background:rgba(15,23,42,0.5); backdrop-filter:blur(16px);
                    border:1px solid rgba(255,255,255,0.05); border-radius:14px; padding:18px;">
            <div style="font-size:10px; color:#475569; font-weight:700; text-transform:uppercase; letter-spacing:2px; margin-bottom:8px;">MISSION</div>
            <div style="color:#e2e8f0; font-size:13px; font-weight:500; line-height:1.6;">{ui_state.agent_task}</div>
        </div>

        <!-- Risk Level tile -->
        <div class="glass-tile" style="background:rgba(15,23,42,0.5); backdrop-filter:blur(16px);
                    border:1px solid rgba(255,255,255,0.05); border-radius:14px; padding:18px;">
            <div style="font-size:10px; color:#475569; font-weight:700; text-transform:uppercase; letter-spacing:2px; margin-bottom:8px;">THREAT LEVEL</div>
            <span style="background:{risk_c}15; border:1px solid {risk_c}40; color:{risk_c};
                         padding:4px 12px; border-radius:6px; font-weight:700; font-size:11px;
                         letter-spacing:1px; animation:{'threatPulse 2s ease-in-out infinite' if ui_state.risk_level in ['high','critical'] else 'none'};">{risk_txt}</span>
        </div>

        <!-- Permissions tile -->
        <div class="glass-tile" style="background:rgba(15,23,42,0.5); backdrop-filter:blur(16px);
                    border:1px solid rgba(255,255,255,0.05); border-radius:14px; padding:18px;">
            <div style="font-size:10px; color:#475569; font-weight:700; text-transform:uppercase; letter-spacing:2px; margin-bottom:8px;">PERMISSIONS</div>
            <div>{perms_html if perms_html else '<span style="color:#334155; font-size:12px; font-style:italic;">None granted</span>'}</div>
        </div>

        <!-- Policy Rules tile -->
        <div class="glass-tile" style="background:rgba(15,23,42,0.5); backdrop-filter:blur(16px);
                    border:1px solid rgba(255,255,255,0.05); border-radius:14px; padding:18px;">
            <div style="font-size:10px; color:#475569; font-weight:700; text-transform:uppercase; letter-spacing:2px; margin-bottom:8px;">POLICY RULES</div>
            <ul style="list-style:none; padding:0; margin:0;">{rules_html if rules_html else '<li style="color:#334155; font-size:12px; font-style:italic;">No policies</li>'}</ul>
        </div>
    </div>'''


def _build_ui_outputs():
    """Return all UI component values for an active episode."""
    action_card = _build_action_card()
    context = _build_context_panel()
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
    """Return UI state for no active episode — welcome screen with Sentinel."""
    welcome = f'''
    <div style="text-align:center; padding:40px 20px;">
        <!-- Sentinel character (analyzing/idle state) -->
        {_build_character("analyzing", 220)}

        <div style="margin-top:16px;">
            <div class="gradient-title" style="font-size:28px; font-weight:800; letter-spacing:-0.5px;">
                S E N T I N E L
            </div>
            <div style="font-size:13px; color:#475569; margin-top:6px; letter-spacing:0.5px;">
                AI Agent Safety Monitor
            </div>
        </div>

        <div style="font-size:13px; color:#64748b; max-width:400px; margin:20px auto 0; line-height:1.8;">
            Review an AI agent's actions in real-time.<br/>
            Decide: <span style="color:#22c55e; font-weight:600;">ALLOW</span>,
            <span style="color:#ef4444; font-weight:600;">BLOCK</span>, or
            <span style="color:#f59e0b; font-weight:600;">FLAG</span> each action.
        </div>

        <div style="display:flex; justify-content:center; gap:16px; margin-top:24px; font-size:12px; color:#334155;">
            <span>69 Tasks</span>
            <span style="color:#1e293b;">|</span>
            <span>10 Grading Components</span>
            <span style="color:#1e293b;">|</span>
            <span>4 Difficulty Levels</span>
        </div>

        <div style="font-size:12px; color:#334155; margin-top:16px;">{msg}</div>
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
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="font-size:11px; color:#64748b;">{label}</span>
                <span style="font-size:11px; color:#e2e8f0; font-weight:600;">{val:.2f}</span>
            </div>
            <div style="height:4px; background:#0f172a; border-radius:2px;">
                <div style="height:4px; background:{bar_color}; border-radius:2px; width:{val*100}%;
                            box-shadow:0 0 8px {bar_color}40;
                            transition:width 0.8s ease;"></div>
            </div>
        </div>'''

    # Decision timeline
    timeline = ""
    for d in ui_state.decision_history:
        t_colors = {"allow": "#22c55e", "block": "#ef4444", "flag": "#f59e0b"}
        t_icons = {"allow": "\u2713", "block": "\u2717", "flag": "\u26A0"}
        d_color = t_colors.get(d["decision"], "#6b7280")
        d_icon = t_icons.get(d["decision"], "?")
        timeline += f'<span style="display:inline-flex; align-items:center; justify-content:center; width:30px; height:30px; border-radius:50%; background:{d_color}15; color:{d_color}; font-size:13px; margin:0 3px; font-weight:700; border:1px solid {d_color}30;" title="Step {d["step"]}: {d["decision"]}">{d_icon}</span>'

    # Character shows the overall result
    char_state = "allow" if score >= 0.5 else "block"

    done_html = f'''
    <div style="text-align:center; padding:30px 20px;">
        {_build_character(char_state, 140)}

        <!-- Score ring -->
        <div style="display:inline-flex; align-items:center; justify-content:center;
                    width:100px; height:100px; border-radius:50%;
                    border:3px solid {score_color}; margin:16px 0;
                    box-shadow:0 0 20px {score_color}30, inset 0 0 20px {score_color}10;
                    animation:scoreReveal 1s ease-out;">
            <span style="font-size:26px; font-weight:700; color:{score_color};">{score:.0%}</span>
        </div>

        <div style="font-size:16px; color:#e2e8f0; font-weight:600; margin-bottom:4px;">Episode Complete</div>
        <div style="font-size:12px; color:#475569; margin-bottom:20px;">
            {ui_state.task_id} &middot; {_difficulty_badge(ui_state.difficulty)} &middot; {ui_state.total_steps} steps
        </div>

        <!-- Timeline -->
        <div style="margin-bottom:24px;">{timeline}</div>

        <!-- Breakdown -->
        <div style="max-width:380px; margin:0 auto; text-align:left;">
            {bars}
        </div>

        <div style="margin-top:20px; font-size:12px; color:#334155;">Pick a new difficulty to play again.</div>
    </div>'''

    return (
        done_html,
        "",
        "",
        gr.update(visible=False),
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


# ---------------------------------------------------------------------------
# Custom CSS — SENTINEL Theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* === SENTINEL DARK THEME === */
.gradio-container {
    background: #020617 !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* Aurora background */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse at 15% 30%, rgba(6,182,212,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 20%, rgba(139,92,246,0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 90%, rgba(34,197,94,0.03) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
    animation: auroraPulse 10s ease-in-out infinite alternate;
}

@keyframes auroraPulse {
    0% { opacity: 0.5; }
    50% { opacity: 1; }
    100% { opacity: 0.7; }
}

.dark {
    --background-fill-primary: #020617 !important;
    --background-fill-secondary: #0f172a !important;
    --border-color-primary: #1e293b !important;
    --body-text-color: #e2e8f0 !important;
    --body-text-color-subdued: #64748b !important;
}

/* === SENTINEL CHARACTER ANIMATIONS === */
@keyframes sentinelFloat {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
}

@keyframes statusPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

@keyframes scanLine {
    0% { transform: translateY(0); opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% { transform: translateY(200px); opacity: 0; }
}

@keyframes progressShimmer {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

@keyframes threatPulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px rgba(239,68,68,0.3); }
    50% { opacity: 0.7; box-shadow: 0 0 16px rgba(239,68,68,0.5); }
}

@keyframes scoreReveal {
    0% { transform: scale(0.5); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}

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

/* === DIFFICULTY BUTTONS === */
.diff-btn {
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    background: rgba(15,23,42,0.5) !important;
    backdrop-filter: blur(10px);
    min-width: 100px !important;
}

.diff-btn:hover {
    transform: translateY(-2px) scale(1.02) !important;
}

.diff-easy { color: #22c55e !important; border-color: rgba(34,197,94,0.2) !important; }
.diff-easy:hover { background: rgba(34,197,94,0.1) !important; border-color: #22c55e !important; box-shadow: 0 0 20px rgba(34,197,94,0.15) !important; }

.diff-medium { color: #f59e0b !important; border-color: rgba(245,158,11,0.2) !important; }
.diff-medium:hover { background: rgba(245,158,11,0.1) !important; border-color: #f59e0b !important; box-shadow: 0 0 20px rgba(245,158,11,0.15) !important; }

.diff-grey { color: #a855f7 !important; border-color: rgba(168,85,247,0.2) !important; }
.diff-grey:hover { background: rgba(168,85,247,0.1) !important; border-color: #a855f7 !important; box-shadow: 0 0 20px rgba(168,85,247,0.15) !important; }

.diff-hard { color: #ef4444 !important; border-color: rgba(239,68,68,0.2) !important; }
.diff-hard:hover { background: rgba(239,68,68,0.1) !important; border-color: #ef4444 !important; box-shadow: 0 0 20px rgba(239,68,68,0.15) !important; }

/* === DECISION BUTTONS with glow === */
@keyframes pulseGlow {
    0%, 100% { filter: brightness(1); }
    50% { filter: brightness(1.15); }
}

.allow-btn {
    background: linear-gradient(135deg, #15803d, #22c55e) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 28px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    transition: all 0.3s !important;
    animation: pulseGlow 3s ease-in-out infinite;
    box-shadow: 0 4px 20px rgba(34,197,94,0.25) !important;
}
.allow-btn:hover {
    transform: translateY(-3px) scale(1.03) !important;
    box-shadow: 0 8px 30px rgba(34,197,94,0.4) !important;
    animation: none;
}

.block-btn {
    background: linear-gradient(135deg, #b91c1c, #ef4444) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 28px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    transition: all 0.3s !important;
    animation: pulseGlow 3s ease-in-out infinite;
    box-shadow: 0 4px 20px rgba(239,68,68,0.25) !important;
}
.block-btn:hover {
    transform: translateY(-3px) scale(1.03) !important;
    box-shadow: 0 8px 30px rgba(239,68,68,0.4) !important;
    animation: none;
}

.flag-btn {
    background: linear-gradient(135deg, #b45309, #f59e0b) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 28px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    transition: all 0.3s !important;
    animation: pulseGlow 3s ease-in-out infinite;
    box-shadow: 0 4px 20px rgba(245,158,11,0.25) !important;
}
.flag-btn:hover {
    transform: translateY(-3px) scale(1.03) !important;
    box-shadow: 0 8px 30px rgba(245,158,11,0.4) !important;
    animation: none;
}

/* === INPUTS === */
textarea, select, input {
    background: rgba(15,23,42,0.6) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
textarea:focus, select:focus, input:focus {
    border-color: #06b6d4 !important;
    box-shadow: 0 0 0 3px rgba(6,182,212,0.1) !important;
    outline: none !important;
}

label {
    color: #475569 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}

footer { display: none !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #020617; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* === LIVE INDICATOR === */
@keyframes livePulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.5); }
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
        title="SENTINEL — AI Agent Safety Monitor",
    ) as demo:

        # Inject CSS
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # --- Header ---
        gr.HTML('''
        <div style="display:flex; align-items:center; justify-content:space-between; padding:12px 0; border-bottom:1px solid rgba(255,255,255,0.04); margin-bottom:16px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:32px; height:32px; border-radius:10px; background:linear-gradient(135deg, #06b6d4, #8b5cf6);
                            display:flex; align-items:center; justify-content:center; font-size:16px;
                            box-shadow:0 0 15px rgba(6,182,212,0.3);">
                    &#x1f6e1;
                </div>
                <div>
                    <div class="gradient-title" style="font-size:15px; font-weight:700; letter-spacing:2px;">SENTINEL</div>
                    <div style="font-size:10px; color:#334155; letter-spacing:0.5px;">AI Agent Safety Monitor &middot; Neural Nomads</div>
                </div>
            </div>
            <div style="display:flex; gap:6px; align-items:center;">
                <span class="live-dot" style="width:7px; height:7px; border-radius:50%; background:#22c55e; display:inline-block;"></span>
                <span style="font-size:11px; color:#475569; font-weight:500;">Live</span>
            </div>
        </div>
        ''')

        # --- Difficulty Selector ---
        with gr.Row():
            easy_btn = gr.Button("EASY", elem_classes=["diff-btn", "diff-easy"], size="sm")
            medium_btn = gr.Button("MEDIUM", elem_classes=["diff-btn", "diff-medium"], size="sm")
            grey_btn = gr.Button("GREY AREA", elem_classes=["diff-btn", "diff-grey"], size="sm")
            hard_btn = gr.Button("HARD", elem_classes=["diff-btn", "diff-hard"], size="sm")

        gr.HTML('<div style="height:8px;"></div>')

        # --- Main Content ---
        with gr.Row():
            # Left: Action Card (2/3 width)
            with gr.Column(scale=2):
                action_html = gr.HTML(
                    value=_build_idle_outputs()[0],
                    elem_id="action-card",
                )

                # Decision buttons
                with gr.Column(visible=False) as decision_group:
                    reason_input = gr.Textbox(
                        placeholder="Why are you making this decision? (optional but improves score)",
                        label="Reason",
                        lines=1,
                        max_lines=2,
                    )

                    # Block-specific options
                    with gr.Column(visible=False) as block_opts:
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
                        allow_btn = gr.Button("\u2713  ALLOW", elem_classes=["allow-btn"], scale=1, min_width=10)
                        block_btn = gr.Button("\u2717  BLOCK", elem_classes=["block-btn"], scale=1, min_width=10)
                        flag_btn = gr.Button("\u26A0  FLAG", elem_classes=["flag-btn"], scale=1, min_width=10)

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
