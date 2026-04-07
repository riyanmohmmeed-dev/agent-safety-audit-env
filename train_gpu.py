#!/usr/bin/env python3
"""
train_gpu.py — PPO Training on Apple Silicon MPS GPU
=====================================================
Proximal Policy Optimization with Generalized Advantage Estimation.
Runs on Apple M3 Pro (14-core GPU) via PyTorch MPS backend.

Usage:
    python train_gpu.py --episodes 2000       # Full training run
    python train_gpu.py --episodes 500 --lr 3e-4  # Quick test
"""

import argparse, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from server.agent_safety_audit_environment import AgentSafetyAuditEnvironment
from models import MonitorAction

# ── Device Selection ─────────────────────────────────────────────────────────
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"  Device: {DEVICE} ({'Apple Silicon GPU' if DEVICE.type == 'mps' else 'CPU'})")

# ── Feature Extraction (expanded to 32 features) ────────────────────────────
RISK_MAP = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0, "unknown": 0.5}
DANGER_KW = ["rm","delete","drop","eval","exec","sudo","chmod","passwd","curl","wget",
             "ssh","secret","api_key","token","exfil","upload","external","admin","root",
             "superuser","plaintext","hardcoded","base64","md5","production","verify=false",
             "select *","dump","0.0.0.0","kill","shutdown","reboot"]
SAFE_KW = ["read","list","log","print","check","validate","test","query","get","fetch",
           "status","health","info","count","view","display","show","describe"]
VTYPES_MAP = {"unauthorized_access":0,"data_exfiltration":1,"privilege_escalation":2,
              "insecure_storage":3,"policy_violation":4,"scope_creep":5,
              "weak_cryptography":6,"destructive_action":7,"information_disclosure":8,
              "supply_chain_risk":9}
SEVERITY_MAP = {"low":0.25,"medium":0.5,"high":0.75,"critical":1.0}

def extract_features(obs) -> np.ndarray:
    f = np.zeros(32, dtype=np.float32)
    if not hasattr(obs, "risk_level"):
        return f
    
    risk = obs.risk_level or "unknown"
    f[0] = RISK_MAP.get(risk, 0.5)
    f[1] = min((obs.current_step or 1) / max(obs.total_steps or 1, 1), 1.0)
    f[2] = float(obs.current_step or 1) / 15.0  # absolute step position
    
    action = obs.current_action
    prev = obs.previous_decisions or []
    perms = obs.agent_permissions or []
    rules = obs.policy_rules or []
    
    txt = ""
    if action:
        txt = f"{action.action} {action.target} {action.details}".lower()
    
    # Danger/safe keyword signals
    danger_count = sum(1 for k in DANGER_KW if k in txt)
    safe_count = sum(1 for k in SAFE_KW if k in txt)
    f[3] = min(danger_count / 5.0, 1.0)
    f[4] = min(safe_count / 3.0, 1.0)
    f[5] = 1.0 if danger_count > 0 and safe_count == 0 else 0.0  # pure danger
    f[6] = 1.0 if safe_count > 0 and danger_count == 0 else 0.0  # pure safe
    
    # Text length features
    f[7] = min(len(txt) / 200.0, 1.0)
    f[8] = min(txt.count(" ") / 20.0, 1.0)  # word count proxy
    
    # Operation type encoding (one-hot-ish)
    file_ops = ["file","write","read","open","save","delete","rm","mkdir","copy","move"]
    net_ops = ["http","request","curl","wget","api","upload","download","ssh","socket","dns"]
    sys_ops = ["sudo","chmod","chown","kill","process","service","daemon","cron","systemctl"]
    db_ops = ["sql","query","insert","select","drop","database","table","mongo","redis"]
    f[9] = 1.0 if any(op in txt for op in file_ops) else 0.0
    f[10] = 1.0 if any(op in txt for op in net_ops) else 0.0
    f[11] = 1.0 if any(op in txt for op in sys_ops) else 0.0
    f[12] = 1.0 if any(op in txt for op in db_ops) else 0.0
    
    # Execution result
    f[13] = 1.0 if hasattr(obs, "execution_result") and obs.execution_result else 0.0
    
    # Context size
    f[14] = min(len(perms) / 5.0, 1.0)
    f[15] = min(len(rules) / 5.0, 1.0)
    
    # Difficulty encoding
    diff = getattr(obs, "difficulty", "medium") or "medium"
    f[16] = 1.0 if diff == "easy" else 0.0
    f[17] = 1.0 if diff == "medium" else 0.0
    f[18] = 1.0 if diff == "grey_area" else 0.0
    f[19] = 1.0 if diff == "hard" else 0.0
    
    # Previous decisions history
    if prev:
        decs = [d.decision if hasattr(d, "decision") else d.get("decision", "") for d in prev]
        n = max(len(decs), 1)
        f[20] = sum(1 for d in decs if d == "allow") / n
        f[21] = sum(1 for d in decs if d == "block") / n
        f[22] = sum(1 for d in decs if d == "flag") / n
        f[23] = 1.0 if any(d == "block" for d in decs) else 0.0
        f[24] = 1.0 if any(d == "flag" for d in decs) else 0.0
        # Last decision
        last = decs[-1] if decs else ""
        f[25] = 1.0 if last == "allow" else 0.0
        f[26] = 1.0 if last == "block" else 0.0
        f[27] = 1.0 if last == "flag" else 0.0
    
    # Sandbox mode
    f[28] = 1.0 if getattr(obs, "execution_mode", None) == "sandbox" else 0.0
    
    # High-risk patterns
    f[29] = 1.0 if any(p in txt for p in ["/etc/passwd","/etc/shadow","../"]) else 0.0
    f[30] = 1.0 if any(p in txt for p in ["rm -rf","drop table","format "]) else 0.0
    f[31] = 1.0 if any(p in txt for p in ["password","secret","credential","private_key"]) else 0.0
    
    return f


# ── Actor-Critic Network ────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """Shared backbone with separate actor (policy) and critic (value) heads."""
    
    def __init__(self, input_dim=32, hidden_dim=128, num_actions=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)
    
    def get_action_and_value(self, state: np.ndarray):
        x = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        logits, value = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return (action.item(), dist.log_prob(action).item(),
                dist.entropy().item(), value.squeeze().item())
    
    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.log_prob(actions), dist.entropy(), values.squeeze(-1)


# ── PPO Algorithm ────────────────────────────────────────────────────────────

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    advantages = []
    gae = 0.0
    next_value = 0.0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        next_value = values[i]
    return advantages


VTYPES = ["policy_violation","unauthorized_access","data_exfiltration",
          "destructive_action","privilege_escalation","insecure_storage",
          "scope_creep","weak_cryptography","information_disclosure","supply_chain_risk"]

def pick_violation_info(features):
    """Heuristic violation type selection based on features."""
    if features[30] > 0.5:  # destructive pattern
        return "destructive_action", "critical"
    if features[29] > 0.5:  # path traversal
        return "unauthorized_access", "high"
    if features[31] > 0.5:  # credentials
        return "information_disclosure", "high"
    if features[10] > 0.5:  # network
        return "data_exfiltration", "high"
    if features[11] > 0.5:  # system
        return "privilege_escalation", "critical"
    if features[12] > 0.5:  # database
        return "data_exfiltration", "high"
    if features[3] > 0.3:   # general danger
        return "policy_violation", "high" if features[0] > 0.5 else "medium"
    return "policy_violation", "medium"


def train_ppo(
    num_episodes=2000,
    lr=3e-4,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    ppo_epochs=4,
    batch_size=64,
    save_path="training_results",
):
    os.makedirs(save_path, exist_ok=True)
    
    model = ActorCritic().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training buffers
    all_states, all_actions, all_log_probs = [], [], []
    all_rewards, all_values, all_dones = [], [], []
    
    # Metrics
    ep_scores, ep_rewards, ep_lengths = [], [], []
    diff_scores = {"easy": [], "medium": [], "grey_area": [], "hard": []}
    dec_counts = {"allow": 0, "block": 0, "flag": 0}
    diffs = ["easy", "medium", "grey_area", "hard"]
    update_count = 0
    
    print(f"\n{'='*65}")
    print(f"  PPO Training — AI Agent Safety Monitor (M3 Pro MPS GPU)")
    print(f"  Episodes: {num_episodes} | LR: {lr} | γ: {gamma} | λ: {lam}")
    print(f"  Clip: {clip_eps} | Entropy: {entropy_coef} | PPO epochs: {ppo_epochs}")
    print(f"  Architecture: 32→128→128→3 (Actor-Critic, LayerNorm)")
    print(f"{'='*65}\n")
    
    t0 = time.time()
    update_interval = 128  # update every N steps
    total_steps_collected = 0
    
    for ep in range(num_episodes):
        diff = diffs[ep % 4]
        env = AgentSafetyAuditEnvironment()
        obs = env.reset(seed=42 + ep * 7, options={"difficulty": diff})
        
        done = False
        ep_reward_sum = 0.0
        steps = 0
        
        while not done:
            feat = extract_features(obs)
            action_idx, log_prob, entropy, value = model.get_action_and_value(feat)
            action_str = {0: "allow", 1: "block", 2: "flag"}[action_idx]
            dec_counts[action_str] += 1
            
            vt, sev = "", ""
            if action_str in ("block", "flag"):
                vt, sev = pick_violation_info(feat)
            
            action = MonitorAction(
                decision=action_str, violation_type=vt, severity=sev,
                reason=f"PPO policy (ep {ep+1}, step {steps+1})",
            )
            
            next_obs, reward, done, info = env.step(action)
            
            all_states.append(feat)
            all_actions.append(action_idx)
            all_log_probs.append(log_prob)
            all_rewards.append(reward)
            all_values.append(value)
            all_dones.append(float(done))
            
            ep_reward_sum += reward
            obs = next_obs
            steps += 1
            total_steps_collected += 1
            
            # PPO update when buffer is full
            if total_steps_collected >= update_interval:
                # Compute GAE
                advantages = compute_gae(all_rewards, all_values, all_dones, gamma, lam)
                returns = [a + v for a, v in zip(advantages, all_values)]
                
                # Convert to tensors
                states_t = torch.FloatTensor(np.array(all_states)).to(DEVICE)
                actions_t = torch.LongTensor(all_actions).to(DEVICE)
                old_log_probs_t = torch.FloatTensor(all_log_probs).to(DEVICE)
                returns_t = torch.FloatTensor(returns).to(DEVICE)
                advantages_t = torch.FloatTensor(advantages).to(DEVICE)
                advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
                
                # PPO update epochs
                n = len(all_states)
                for _ in range(ppo_epochs):
                    indices = np.random.permutation(n)
                    for start in range(0, n, batch_size):
                        end = min(start + batch_size, n)
                        idx = indices[start:end]
                        
                        batch_states = states_t[idx]
                        batch_actions = actions_t[idx]
                        batch_old_lp = old_log_probs_t[idx]
                        batch_returns = returns_t[idx]
                        batch_adv = advantages_t[idx]
                        
                        new_log_probs, entropy, values = model.evaluate(batch_states, batch_actions)
                        
                        # PPO clipped objective
                        ratio = torch.exp(new_log_probs - batch_old_lp)
                        surr1 = ratio * batch_adv
                        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_adv
                        actor_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss
                        value_loss = nn.functional.mse_loss(values, batch_returns)
                        
                        # Entropy bonus
                        entropy_loss = -entropy.mean()
                        
                        loss = actor_loss + value_coef * value_loss + entropy_coef * entropy_loss
                        
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        optimizer.step()
                
                update_count += 1
                all_states.clear(); all_actions.clear(); all_log_probs.clear()
                all_rewards.clear(); all_values.clear(); all_dones.clear()
                total_steps_collected = 0
        
        score = info.get("score", info.get("episode_score", 0.0))
        ep_scores.append(score)
        ep_rewards.append(ep_reward_sum)
        ep_lengths.append(steps)
        diff_scores[diff].append(score)
        
        if (ep + 1) % 20 == 0 or ep == 0:
            avg_s = sum(ep_scores[-20:]) / len(ep_scores[-20:])
            best = max(ep_scores[-20:])
            elapsed = time.time() - t0
            print(f"  Ep {ep+1:5d}/{num_episodes} | Score: {score:.3f} | "
                  f"Avg(20): {avg_s:.3f} | Best(20): {best:.3f} | "
                  f"Steps: {steps} | {diff:10s} | Updates: {update_count} | "
                  f"{elapsed:.1f}s")
    
    elapsed = time.time() - t0
    
    # Save results
    results = {
        "algorithm": "PPO (Proximal Policy Optimization)",
        "device": str(DEVICE),
        "episodes": num_episodes,
        "learning_rate": lr, "gamma": gamma, "lambda": lam,
        "clip_epsilon": clip_eps, "entropy_coef": entropy_coef,
        "ppo_epochs": ppo_epochs, "batch_size": batch_size,
        "training_time_seconds": round(elapsed, 2),
        "architecture": "ActorCritic: 32→128(LN)→128(LN)→[Actor:64→3, Critic:64→1]",
        "feature_dim": 32,
        "total_updates": update_count,
        "initial_avg_score": round(sum(ep_scores[:20]) / 20, 4),
        "final_avg_score": round(sum(ep_scores[-20:]) / 20, 4),
        "best_score": round(max(ep_scores), 4),
        "improvement": round(sum(ep_scores[-20:]) / 20 - sum(ep_scores[:20]) / 20, 4),
        "decision_distribution": {k: round(v / max(sum(dec_counts.values()), 1), 3)
                                   for k, v in dec_counts.items()},
        "scores_by_difficulty": {d: round(sum(s) / max(len(s), 1), 4)
                                  for d, s in diff_scores.items()},
        "episode_scores": [round(s, 4) for s in ep_scores],
        "episode_rewards": [round(r, 4) for r in ep_rewards],
    }
    
    with open(f"{save_path}/ppo_training_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    torch.save(model.state_dict(), f"{save_path}/ppo_policy.pt")
    
    print(f"\n{'='*65}")
    print(f"  PPO TRAINING COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*65}")
    print(f"  Initial Avg: {results['initial_avg_score']:.4f}")
    print(f"  Final Avg:   {results['final_avg_score']:.4f}")
    print(f"  Improvement: {results['improvement']:+.4f}")
    print(f"  Best Score:  {results['best_score']:.4f}")
    print(f"  Updates:     {update_count}")
    print(f"  Decisions:   {results['decision_distribution']}")
    for d, s in results["scores_by_difficulty"].items():
        print(f"    {d:12s}: {s:.4f}")
    
    # Generate convergence plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        n = len(ep_scores)
        w = max(n // 20, 10)
        def smooth(d, win):
            return [sum(d[max(0, i-win):i+1]) / len(d[max(0, i-win):i+1]) for i in range(len(d))]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"PPO Training — AI Safety Monitor ({DEVICE}, {num_episodes} episodes)",
                     fontsize=15, fontweight="bold")
        
        # Score convergence
        ax = axes[0, 0]
        ax.plot(range(n), ep_scores, alpha=0.15, color="#4A90D9", linewidth=0.5)
        ax.plot(range(n), smooth(ep_scores, w), color="#1A3A5C", linewidth=2.5, label="Smoothed")
        ax.axhline(y=results["initial_avg_score"], color="#E74C3C", linestyle="--", alpha=0.5, label="Initial avg")
        ax.axhline(y=results["final_avg_score"], color="#2ECC71", linestyle="--", alpha=0.5, label="Final avg")
        ax.set_xlabel("Episode"); ax.set_ylabel("Score"); ax.set_title("Score Convergence")
        ax.legend(); ax.grid(True, alpha=0.3)
        
        # Reward convergence
        ax = axes[0, 1]
        ax.plot(range(n), ep_rewards, alpha=0.15, color="#E67E22", linewidth=0.5)
        ax.plot(range(n), smooth(ep_rewards, w), color="#D35400", linewidth=2.5, label="Smoothed")
        ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward"); ax.set_title("Reward Convergence")
        ax.legend(); ax.grid(True, alpha=0.3)
        
        # By difficulty
        ax = axes[1, 0]
        colors = ["#2ECC71", "#F39C12", "#E74C3C", "#8E44AD"]
        for i, (d, c) in enumerate(zip(diffs, colors)):
            ds = [ep_scores[j] for j in range(n) if j % 4 == i]
            if ds:
                ax.plot(range(len(ds)), smooth(ds, max(len(ds) // 10, 5)),
                        color=c, linewidth=2, label=d)
        ax.set_xlabel("Episode (per difficulty)"); ax.set_ylabel("Score")
        ax.set_title("Learning by Difficulty"); ax.legend(); ax.grid(True, alpha=0.3)
        
        # Score distribution comparison
        ax = axes[1, 1]
        first = ep_scores[:min(50, n // 4)]
        last = ep_scores[-min(50, n // 4):]
        ax.hist(first, bins=15, alpha=0.6, color="#E74C3C", label=f"First {len(first)} ep")
        ax.hist(last, bins=15, alpha=0.6, color="#2ECC71", label=f"Last {len(last)} ep")
        ax.set_xlabel("Score"); ax.set_ylabel("Count")
        ax.set_title("Score Distribution: Before vs After"); ax.legend(); ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/ppo_convergence.png", dpi=150, bbox_inches="tight")
        print(f"\n  Plot: {save_path}/ppo_convergence.png")
    except Exception as e:
        print(f"\n  (Plot generation failed: {e})")
    
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PPO training on MPS GPU")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--save-path", default="training_results")
    a = p.parse_args()
    train_ppo(a.episodes, a.lr, a.gamma, clip_eps=a.clip_eps,
              entropy_coef=a.entropy_coef, ppo_epochs=a.ppo_epochs,
              save_path=a.save_path)
