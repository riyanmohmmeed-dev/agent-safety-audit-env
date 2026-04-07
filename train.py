"""
train.py — RL Training Pipeline for AI Agent Safety Monitor
=============================================================
Pure numpy REINFORCE (policy gradient) implementation.
No PyTorch/TensorFlow required — trains on CPU in minutes.

Usage:
    python train.py                     # Train 200 episodes
    python train.py --episodes 500      # Custom episode count
"""

import argparse, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from server.agent_safety_audit_environment import AgentSafetyAuditEnvironment
from models import MonitorAction

# ── Feature Extraction ──────────────────────────────────────────────────────

RISK_MAP = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0, "unknown": 0.5}
DANGER_KW = ["rm","delete","drop","eval","exec","sudo","chmod","passwd","curl","wget",
             "ssh","secret","api_key","token","exfil","upload","external","admin","root",
             "superuser","plaintext","hardcoded","base64","md5","production","verify=false",
             "select *","dump","0.0.0.0"]
SAFE_KW = ["read","list","log","print","check","validate","test","query","get","fetch",
           "status","health","info","count"]

def extract_features(obs) -> np.ndarray:
    f = np.zeros(16, dtype=np.float64)
    if hasattr(obs, "risk_level"):
        f[0] = RISK_MAP.get(obs.risk_level or "unknown", 0.5)
        f[1] = min((obs.current_step or 1) / max(obs.total_steps or 1, 1), 1.0)
        action = obs.current_action
        prev = obs.previous_decisions or []
        perms = obs.agent_permissions or []
        rules = obs.policy_rules or []
    else:
        action, prev, perms, rules = None, [], [], []
    
    txt = ""
    if action:
        txt = f"{action.action} {action.target} {action.details}".lower()
    
    if prev:
        decs = [d.decision if hasattr(d,"decision") else d.get("decision","") for d in prev]
        f[2] = 1.0 if any(d=="block" for d in decs) else 0.0
        f[3] = 1.0 if any(d=="flag" for d in decs) else 0.0
        n = max(len(decs),1)
        f[13] = sum(1 for d in decs if d=="allow")/n
        f[14] = sum(1 for d in decs if d=="block")/n
        f[15] = sum(1 for d in decs if d=="flag")/n
    
    f[4] = min(sum(1 for k in DANGER_KW if k in txt)/5.0, 1.0)
    f[5] = min(sum(1 for k in SAFE_KW if k in txt)/3.0, 1.0)
    f[6] = min(len(txt)/200.0, 1.0)
    f[7] = 1.0 if any(w in txt for w in ["file","write","open","save","delete","rm","mkdir"]) else 0.0
    f[8] = 1.0 if any(w in txt for w in ["http","request","curl","wget","api","upload","ssh"]) else 0.0
    f[9] = 1.0 if any(w in txt for w in ["sudo","chmod","chown","kill","process","service"]) else 0.0
    f[10] = 1.0 if hasattr(obs,"execution_result") and obs.execution_result else 0.0
    f[11] = min(len(perms)/5.0, 1.0)
    f[12] = min(len(rules)/5.0, 1.0)
    return f

# ── Pure Numpy Policy Network ───────────────────────────────────────────────

class NumpyPolicy:
    """2-layer neural network policy trained with REINFORCE. Pure numpy."""
    
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=3, lr=1e-3):
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0/input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim//2) * np.sqrt(2.0/hidden_dim)
        self.b2 = np.zeros(hidden_dim//2)
        self.W3 = np.random.randn(hidden_dim//2, output_dim) * np.sqrt(2.0/(hidden_dim//2))
        self.b3 = np.zeros(output_dim)
        self.lr = lr
        
        # Episode storage
        self.log_probs_grads: List[Dict] = []
        self.rewards: List[float] = []
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()
    
    def forward(self, x):
        self.h1 = self._relu(x @ self.W1 + self.b1)
        self.h2 = self._relu(self.h1 @ self.W2 + self.b2)
        logits = self.h2 @ self.W3 + self.b3
        probs = self._softmax(logits)
        return probs
    
    def select_action(self, state: np.ndarray) -> Tuple[int, str]:
        probs = self.forward(state)
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= probs.sum()
        action = np.random.choice(3, p=probs)
        
        # Store gradient info for REINFORCE
        self.log_probs_grads.append({
            "action": action, "probs": probs.copy(),
            "state": state.copy(), "h1": self.h1.copy(), "h2": self.h2.copy(),
        })
        
        return action, {0:"allow", 1:"block", 2:"flag"}[action]
    
    def update(self, gamma=0.99) -> float:
        if not self.rewards:
            return 0.0
        
        # Compute discounted returns
        R, returns = 0.0, []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # REINFORCE gradient update
        for i, (info, G) in enumerate(zip(self.log_probs_grads, returns)):
            probs = info["probs"]
            action = info["action"]
            state = info["state"]
            h1 = info["h1"]
            h2 = info["h2"]
            
            # d(log_prob)/d(logits) = (one_hot - probs) for the selected action
            dlogits = -probs.copy()
            dlogits[action] += 1.0
            dlogits *= G  # Scale by return
            
            # Backprop through W3
            dW3 = np.outer(h2, dlogits)
            db3 = dlogits
            
            # Backprop through layer 2
            dh2 = dlogits @ self.W3.T
            dh2 *= (h2 > 0).astype(float)  # ReLU derivative
            dW2 = np.outer(h1, dh2)
            db2 = dh2
            
            # Backprop through layer 1
            dh1 = dh2 @ self.W2.T
            dh1 *= (h1 > 0).astype(float)
            dW1 = np.outer(state, dh1)
            db1 = dh1
            
            # Gradient ascent (maximizing expected reward)
            self.W3 += self.lr * dW3
            self.b3 += self.lr * db3
            self.W2 += self.lr * dW2
            self.b2 += self.lr * db2
            self.W1 += self.lr * dW1
            self.b1 += self.lr * db1
        
        total = sum(self.rewards)
        self.log_probs_grads.clear()
        self.rewards.clear()
        return total

# ── Training ─────────────────────────────────────────────────────────────────

VTYPES = ["policy_violation","unauthorized_access","data_exfiltration",
          "destructive_action","privilege_escalation"]

def train(num_episodes=200, lr=1e-3, gamma=0.99, save_path="training_results"):
    os.makedirs(save_path, exist_ok=True)
    policy = NumpyPolicy(lr=lr)
    
    ep_rewards, ep_scores, ep_lengths = [], [], []
    diff_scores = {"easy":[],"medium":[],"grey_area":[],"hard":[]}
    dec_counts = {"allow":0,"block":0,"flag":0}
    diffs = ["easy","medium","grey_area","hard"]
    
    print(f"{'='*60}")
    print(f"  REINFORCE Training — AI Agent Safety Monitor (numpy)")
    print(f"  Episodes: {num_episodes} | LR: {lr} | γ: {gamma}")
    print(f"  Policy: 16→64→32→3 (allow/block/flag)")
    print(f"{'='*60}\n")
    
    t0 = time.time()
    
    for ep in range(num_episodes):
        diff = diffs[ep % 4]
        env = AgentSafetyAuditEnvironment()
        obs = env.reset(seed=42+ep*7, options={"difficulty": diff})
        done, steps = False, 0
        
        while not done:
            feat = extract_features(obs)
            aidx, astr = policy.select_action(feat)
            dec_counts[astr] += 1
            
            vt, sev = "", ""
            if astr == "block":
                vt = "destructive_action" if feat[4]>0.3 else "policy_violation"
                sev = "critical" if feat[0]>0.7 else "high"
            
            action = MonitorAction(decision=astr, violation_type=vt, severity=sev,
                                   reason=f"RL policy (step {steps+1})")
            obs, reward, done, info = env.step(action)
            policy.rewards.append(reward)
            steps += 1
        
        ep_reward = policy.update(gamma)
        score = info.get("score", info.get("episode_score", 0.0))
        ep_rewards.append(ep_reward)
        ep_scores.append(score)
        ep_lengths.append(steps)
        diff_scores[diff].append(score)
        
        if (ep+1) % 10 == 0 or ep == 0:
            avg = sum(ep_scores[-10:])/len(ep_scores[-10:])
            print(f"  Ep {ep+1:4d}/{num_episodes} | Score: {score:.3f} | "
                  f"Avg(10): {avg:.3f} | Steps: {steps} | {diff:10s} | "
                  f"{time.time()-t0:.1f}s")
    
    elapsed = time.time() - t0
    
    results = {
        "algorithm": "REINFORCE (pure numpy)",
        "episodes": num_episodes,
        "learning_rate": lr, "gamma": gamma,
        "training_time_seconds": round(elapsed, 2),
        "architecture": "Linear(16,64)→ReLU→Linear(64,32)→ReLU→Linear(32,3)",
        "initial_avg_score": round(sum(ep_scores[:20])/20, 4),
        "final_avg_score": round(sum(ep_scores[-20:])/20, 4),
        "best_score": round(max(ep_scores), 4),
        "improvement": round(sum(ep_scores[-20:])/20 - sum(ep_scores[:20])/20, 4),
        "decision_distribution": {k: round(v/max(sum(dec_counts.values()),1),3) for k,v in dec_counts.items()},
        "scores_by_difficulty": {d: round(sum(s)/max(len(s),1),4) for d,s in diff_scores.items()},
        "episode_scores": [round(s,4) for s in ep_scores],
        "episode_rewards": [round(r,4) for r in ep_rewards],
    }
    
    with open(f"{save_path}/training_metrics.json","w") as f:
        json.dump(results, f, indent=2)
    
    # Save weights
    np.savez(f"{save_path}/policy_weights.npz",
             W1=policy.W1, b1=policy.b1, W2=policy.W2, b2=policy.b2,
             W3=policy.W3, b3=policy.b3)
    
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"  Initial Avg: {results['initial_avg_score']:.4f}")
    print(f"  Final Avg:   {results['final_avg_score']:.4f}")
    print(f"  Improvement: {results['improvement']:+.4f}")
    print(f"  Best Score:  {results['best_score']:.4f}")
    print(f"  Decisions:   {results['decision_distribution']}")
    for d,s in results["scores_by_difficulty"].items():
        print(f"    {d:12s}: {s:.4f}")
    
    # Generate convergence plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        n = len(ep_scores)
        w = max(n//20, 5)
        def smooth(d, win):
            return [sum(d[max(0,i-win):i+1])/len(d[max(0,i-win):i+1]) for i in range(len(d))]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("REINFORCE Training — AI Agent Safety Monitor", fontsize=14, fontweight="bold")
        
        ax = axes[0,0]
        ax.plot(range(n), ep_scores, alpha=0.3, color="#4A90D9", linewidth=0.5)
        ax.plot(range(n), smooth(ep_scores,w), color="#2C5F8A", linewidth=2, label=f"Smoothed")
        ax.set_xlabel("Episode"); ax.set_ylabel("Score"); ax.set_title("Score Convergence")
        ax.legend(); ax.grid(True, alpha=0.3)
        
        ax = axes[0,1]
        ax.plot(range(n), ep_rewards, alpha=0.3, color="#E67E22", linewidth=0.5)
        ax.plot(range(n), smooth(ep_rewards,w), color="#D35400", linewidth=2, label="Smoothed")
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax.set_title("Reward Convergence")
        ax.legend(); ax.grid(True, alpha=0.3)
        
        ax = axes[1,0]
        colors = ["#2ECC71","#F39C12","#E74C3C","#8E44AD"]
        for i,(d,c) in enumerate(zip(diffs,colors)):
            ds = [ep_scores[j] for j in range(n) if j%4==i]
            if ds: ax.plot(range(len(ds)), smooth(ds,max(len(ds)//10,3)), color=c, linewidth=2, label=d)
        ax.set_xlabel("Episode"); ax.set_ylabel("Score"); ax.set_title("By Difficulty")
        ax.legend(); ax.grid(True, alpha=0.3)
        
        ax = axes[1,1]
        ax.hist(ep_scores[:20], bins=12, alpha=0.6, color="#E74C3C", label="First 20")
        ax.hist(ep_scores[-20:], bins=12, alpha=0.6, color="#2ECC71", label="Last 20")
        ax.set_xlabel("Score"); ax.set_ylabel("Count"); ax.set_title("Before vs After")
        ax.legend(); ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/convergence.png", dpi=150, bbox_inches="tight")
        print(f"\n  Plot: {save_path}/convergence.png")
    except ImportError:
        print("\n  (matplotlib not available — skipping plot)")
    
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--save-path", default="training_results")
    a = p.parse_args()
    train(a.episodes, a.lr, a.gamma, a.save_path)
