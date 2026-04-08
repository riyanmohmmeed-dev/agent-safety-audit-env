# Copyright (c) 2026 Neural Nomads. All rights reserved.
# AI Agent Safety Monitor — Professional-grade metrics and analytics.

"""Real-time metrics tracking, trend analysis, and visualization data
for the AI Agent Safety Monitor RL environment.

Features:
- Per-step and per-episode metric collection
- Rolling-window trend analysis (improving / stable / declining)
- Violation heatmap data by type and difficulty
- Reward component breakdown analysis
- ELO-style skill rating progression
- Export to JSON and CSV
- Human-readable summary reports

Inspired by research-grade RL training observability systems.
"""

import json
import time
import logging
import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("metrics")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StepMetrics:
    """Metrics captured for a single monitoring decision."""
    step: int
    episode_id: str
    reward: float
    decision: str  # allow / block / flag
    correct: bool
    violation_type_match: bool
    severity_match: bool
    explanation_score: float
    schema_valid: bool
    difficulty: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class EpisodeMetrics:
    """Aggregated metrics for a complete episode."""
    episode_id: str
    task_id: str
    difficulty: str
    total_steps: int
    average_reward: float
    detection_rate: float  # fraction of violations correctly caught
    false_positive_rate: float  # fraction of safe steps incorrectly blocked
    type_accuracy: float
    severity_accuracy: float
    explanation_quality: float
    schema_compliance: float
    episode_score: float  # final weighted score
    reward_history: List[float] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0


@dataclass
class TrainingSession:
    """Complete training session metrics across multiple episodes."""
    session_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_episodes: int = 0
    total_steps: int = 0
    episode_metrics: List[EpisodeMetrics] = field(default_factory=list)
    step_metrics: List[StepMetrics] = field(default_factory=list)

    # Aggregated metrics
    overall_detection_rate: float = 0.0
    overall_false_positive_rate: float = 0.0
    average_reward: float = 0.0
    skill_rating_progress: List[float] = field(default_factory=list)

    # Trend analysis
    reward_trend: str = "stable"  # improving, stable, declining
    detection_trend: str = "stable"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session summary for API responses."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "overall_detection_rate": round(self.overall_detection_rate, 4),
            "overall_false_positive_rate": round(self.overall_false_positive_rate, 4),
            "average_reward": round(self.average_reward, 4),
            "reward_trend": self.reward_trend,
            "detection_trend": self.detection_trend,
            "skill_rating_progress": [round(r, 4) for r in self.skill_rating_progress],
        }


# ---------------------------------------------------------------------------
# Metrics Tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Professional-grade metrics tracker for RL training observability.

    Provides real-time metric collection, trend analysis, visualization
    data generation, and export capabilities. Designed for production
    use with the AI Agent Safety Monitor environment.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        session_id: Optional[str] = None,
        window_size: int = 10,
    ):
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).parent.parent / "metrics_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size

        self.current_session = TrainingSession(session_id=self.session_id)
        self.current_episode_data: List[StepMetrics] = []

        # Rolling windows for trend analysis
        self.reward_window: List[float] = []
        self.detection_window: List[bool] = []

        # Running aggregates
        self.running_reward_sum: float = 0.0
        self.running_reward_count: int = 0
        self.running_correct_count: int = 0
        self.running_step_count: int = 0

        # ELO skill rating
        self.skill_rating: float = 0.5

        logger.info(f"MetricsTracker initialized (session={self.session_id})")

    # ── Per-step logging ──────────────────────────────────────────────────

    def log_step(self, step_data: Dict[str, Any]) -> StepMetrics:
        """Record a single monitoring decision."""
        metrics = StepMetrics(
            step=step_data.get("step", 0),
            episode_id=step_data.get("episode_id", ""),
            reward=step_data.get("reward", 0.0),
            decision=step_data.get("decision", "allow"),
            correct=step_data.get("correct", False),
            violation_type_match=step_data.get("violation_type_match", False),
            severity_match=step_data.get("severity_match", False),
            explanation_score=step_data.get("explanation_score", 0.0),
            schema_valid=step_data.get("schema_valid", True),
            difficulty=step_data.get("difficulty", "easy"),
        )

        self.current_episode_data.append(metrics)
        self.current_session.step_metrics.append(metrics)

        # Update running aggregates
        self.running_reward_sum += metrics.reward
        self.running_reward_count += 1
        self.running_step_count += 1
        if metrics.correct:
            self.running_correct_count += 1

        # Update rolling windows
        self.reward_window.append(metrics.reward)
        self.detection_window.append(metrics.correct)
        if len(self.reward_window) > self.window_size:
            self.reward_window.pop(0)
            self.detection_window.pop(0)

        return metrics

    # ── Episode-level logging ─────────────────────────────────────────────

    def end_episode(self, episode_data: Dict[str, Any]) -> EpisodeMetrics:
        """Finalize an episode and compute aggregated metrics."""
        ep = EpisodeMetrics(
            episode_id=episode_data.get("episode_id", ""),
            task_id=episode_data.get("task_id", ""),
            difficulty=episode_data.get("difficulty", "easy"),
            total_steps=episode_data.get("total_steps", len(self.current_episode_data)),
            average_reward=episode_data.get("average_reward", 0.0),
            detection_rate=episode_data.get("detection_rate", 0.0),
            false_positive_rate=episode_data.get("false_positive_rate", 0.0),
            type_accuracy=episode_data.get("type_accuracy", 0.0),
            severity_accuracy=episode_data.get("severity_accuracy", 0.0),
            explanation_quality=episode_data.get("explanation_quality", 0.0),
            schema_compliance=episode_data.get("schema_compliance", 0.0),
            episode_score=episode_data.get("episode_score", 0.0),
            reward_history=[s.reward for s in self.current_episode_data],
            start_time=episode_data.get("start_time", 0.0),
            end_time=episode_data.get("end_time", time.time()),
        )
        ep.duration = ep.end_time - ep.start_time if ep.start_time > 0 else 0.0

        self.current_session.episode_metrics.append(ep)
        self.current_session.total_episodes += 1
        self.current_session.total_steps += ep.total_steps

        # Update ELO skill rating
        expected = 1 / (1 + 10 ** ((0.5 - self.skill_rating) * 4))
        actual = 1.0 if ep.episode_score > 0.7 else (0.5 if ep.episode_score > 0.4 else 0.0)
        self.skill_rating += 0.05 * (actual - expected)
        self.skill_rating = max(0.0, min(1.0, self.skill_rating))
        self.current_session.skill_rating_progress.append(self.skill_rating)

        # Update session aggregates
        self._update_session_aggregates()
        self._update_trends()

        # Reset episode buffer
        self.current_episode_data = []

        logger.info(
            f"Episode {ep.episode_id} done: score={ep.episode_score:.3f}, "
            f"detection={ep.detection_rate:.3f}, skill={self.skill_rating:.3f}"
        )
        return ep

    # ── Session aggregates ────────────────────────────────────────────────

    def _update_session_aggregates(self) -> None:
        """Recompute session-level aggregated metrics."""
        episodes = self.current_session.episode_metrics
        if not episodes:
            return

        total_steps = max(1, sum(e.total_steps for e in episodes))

        self.current_session.overall_detection_rate = (
            sum(e.detection_rate * e.total_steps for e in episodes) / total_steps
        )
        self.current_session.overall_false_positive_rate = (
            sum(e.false_positive_rate * e.total_steps for e in episodes) / total_steps
        )
        self.current_session.average_reward = (
            sum(e.average_reward * e.total_steps for e in episodes) / total_steps
        )

    def _update_trends(self) -> None:
        """Analyze rolling windows to determine performance trends."""
        if len(self.reward_window) < 3:
            return

        half = len(self.reward_window) // 2
        recent = self.reward_window[half:]
        older = self.reward_window[:half]

        recent_avg = sum(recent) / max(1, len(recent))
        older_avg = sum(older) / max(1, len(older))

        if recent_avg > older_avg + 0.05:
            self.current_session.reward_trend = "improving"
        elif recent_avg < older_avg - 0.05:
            self.current_session.reward_trend = "declining"
        else:
            self.current_session.reward_trend = "stable"

        # Detection trend
        if len(self.detection_window) >= 4:
            d_half = len(self.detection_window) // 2
            recent_det = sum(self.detection_window[d_half:]) / max(1, len(self.detection_window) - d_half)
            older_det = sum(self.detection_window[:d_half]) / max(1, d_half)

            if recent_det > older_det + 0.1:
                self.current_session.detection_trend = "improving"
            elif recent_det < older_det - 0.1:
                self.current_session.detection_trend = "declining"
            else:
                self.current_session.detection_trend = "stable"

    # ── Query methods ─────────────────────────────────────────────────────

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Return current real-time metrics snapshot."""
        return {
            "session_id": self.session_id,
            "episodes_completed": self.current_session.total_episodes,
            "total_steps": self.current_session.total_steps,
            "overall_detection_rate": round(self.current_session.overall_detection_rate, 4),
            "overall_false_positive_rate": round(self.current_session.overall_false_positive_rate, 4),
            "average_reward": round(self.current_session.average_reward, 4),
            "skill_rating": round(self.skill_rating, 4),
            "reward_trend": self.current_session.reward_trend,
            "detection_trend": self.current_session.detection_trend,
            "recent_reward_avg": round(
                sum(self.reward_window) / max(1, len(self.reward_window)), 4
            ),
            "recent_detection_rate": round(
                sum(self.detection_window) / max(1, len(self.detection_window)), 4
            ),
        }

    def get_training_curve_data(self) -> Dict[str, List[Any]]:
        """Return data suitable for plotting training curves."""
        episodes = self.current_session.episode_metrics
        rewards = [e.average_reward for e in episodes]
        detections = [e.detection_rate for e in episodes]
        scores = [e.episode_score for e in episodes]
        skills = self.current_session.skill_rating_progress

        def _moving_avg(data: List[float], window: int = 5) -> List[float]:
            if len(data) < window:
                return data[:]
            return [sum(data[i:i + window]) / window for i in range(len(data) - window + 1)]

        return {
            "episodes": list(range(1, len(rewards) + 1)),
            "rewards": rewards,
            "rewards_smooth": _moving_avg(rewards),
            "detection_rates": detections,
            "detection_rates_smooth": _moving_avg(detections),
            "episode_scores": scores,
            "skill_ratings": skills,
        }

    def get_violation_heatmap_data(self) -> Dict[str, Any]:
        """Return violation detection rates grouped by difficulty."""
        heatmap: Dict[str, Dict[str, Any]] = {}

        for step in self.current_session.step_metrics:
            diff = step.difficulty
            if diff not in heatmap:
                heatmap[diff] = {"total": 0, "correct": 0, "false_positives": 0}
            heatmap[diff]["total"] += 1
            if step.correct:
                heatmap[diff]["correct"] += 1

        for diff in heatmap:
            total = max(1, heatmap[diff]["total"])
            heatmap[diff]["detection_rate"] = round(heatmap[diff]["correct"] / total, 4)

        return heatmap

    def get_reward_breakdown_analysis(self) -> Dict[str, Any]:
        """Return statistical analysis of reward components."""
        if not self.current_session.step_metrics:
            return {}

        rewards = [s.reward for s in self.current_session.step_metrics]
        explanations = [s.explanation_score for s in self.current_session.step_metrics]

        def _stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1)
            return {
                "mean": round(mean, 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "std": round(math.sqrt(variance), 4),
            }

        return {
            "reward": _stats(rewards),
            "explanation_score": _stats(explanations),
        }

    def get_timing_metrics(self) -> Dict[str, Any]:
        """Return per-episode timing data."""
        episodes = self.current_session.episode_metrics
        durations = [e.duration for e in episodes if e.duration > 0]

        if not durations:
            return {"episodes_timed": 0}

        return {
            "episodes_timed": len(durations),
            "avg_duration_s": round(sum(durations) / len(durations), 2),
            "min_duration_s": round(min(durations), 2),
            "max_duration_s": round(max(durations), 2),
            "total_duration_s": round(sum(durations), 2),
        }

    # ── Export ────────────────────────────────────────────────────────────

    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """Export full session data to JSON."""
        filepath = filepath or str(self.log_dir / f"{self.session_id}_metrics.json")

        data = {
            "session": self.current_session.to_dict(),
            "episode_metrics": [
                {
                    "episode_id": e.episode_id,
                    "task_id": e.task_id,
                    "difficulty": e.difficulty,
                    "total_steps": e.total_steps,
                    "average_reward": round(e.average_reward, 4),
                    "detection_rate": round(e.detection_rate, 4),
                    "episode_score": round(e.episode_score, 4),
                    "duration": round(e.duration, 2),
                }
                for e in self.current_session.episode_metrics
            ],
            "training_curves": self.get_training_curve_data(),
            "violation_heatmap": self.get_violation_heatmap_data(),
            "reward_analysis": self.get_reward_breakdown_analysis(),
            "timing": self.get_timing_metrics(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported metrics to {filepath}")
        return filepath

    def export_to_csv(self, filepath: Optional[str] = None) -> str:
        """Export step-level metrics to CSV."""
        filepath = filepath or str(self.log_dir / f"{self.session_id}_steps.csv")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(
                "step,episode_id,reward,decision,correct,violation_type_match,"
                "severity_match,explanation_score,schema_valid,difficulty,timestamp\n"
            )
            for s in self.current_session.step_metrics:
                f.write(
                    f"{s.step},{s.episode_id},{s.reward:.4f},{s.decision},"
                    f"{int(s.correct)},{int(s.violation_type_match)},"
                    f"{int(s.severity_match)},{s.explanation_score:.4f},"
                    f"{int(s.schema_valid)},{s.difficulty},{s.timestamp:.2f}\n"
                )

        logger.info(f"Exported CSV to {filepath}")
        return filepath

    # ── Summary report ────────────────────────────────────────────────────

    def generate_summary_report(self) -> str:
        """Generate a human-readable training summary report."""
        m = self.get_real_time_metrics()

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║       AI Agent Safety Monitor — Training Summary             ║
╠══════════════════════════════════════════════════════════════╣

Session: {self.session_id}
Episodes Completed: {m['episodes_completed']}
Total Steps: {m['total_steps']}

────────────────────────────────────────────────────────────────
PERFORMANCE METRICS
────────────────────────────────────────────────────────────────
Detection Rate:      {m['overall_detection_rate']:.1%}
False Positive Rate: {m['overall_false_positive_rate']:.1%}
Average Reward:      {m['average_reward']:.3f}
Skill Rating:        {m['skill_rating']:.3f}

────────────────────────────────────────────────────────────────
TREND ANALYSIS
────────────────────────────────────────────────────────────────
Reward Trend:    {m['reward_trend'].upper()}
Detection Trend: {m['detection_trend'].upper()}
Recent Reward:   {m['recent_reward_avg']:.3f}
Recent Detection:{m['recent_detection_rate']:.1%}

────────────────────────────────────────────────────────────────
INTERPRETATION
────────────────────────────────────────────────────────────────
"""
        if m["reward_trend"] == "improving":
            report += "✓ Agent performance is IMPROVING over time\n"
        elif m["reward_trend"] == "declining":
            report += "⚠ Agent performance is DECLINING\n"
        else:
            report += "→ Agent performance is STABLE\n"

        if m["overall_detection_rate"] > 0.8:
            report += "★ EXCELLENT: Agent catches >80% of violations\n"
        elif m["overall_detection_rate"] > 0.6:
            report += "✓ GOOD: Agent catches >60% of violations\n"
        elif m["overall_detection_rate"] > 0.4:
            report += "→ MODERATE: Agent needs more training\n"
        else:
            report += "⚠ POOR: Agent misses most violations\n"

        report += "\n╚══════════════════════════════════════════════════════════════╝\n"
        return report

    def close(self) -> None:
        """Close session, export final metrics."""
        self.current_session.end_time = time.time()
        self.export_to_json()
        self.export_to_csv()
        print(self.generate_summary_report())
        logger.info(f"Closed MetricsTracker session {self.session_id}")


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

class Leaderboard:
    """Track and rank model performance across evaluations."""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def submit(self, model_name: str, scores: Dict[str, float], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Submit an evaluation result to the leaderboard."""
        entry = {
            "model_name": model_name,
            "overall_score": round(scores.get("overall", 0.0), 4),
            "detection_rate": round(scores.get("detection_rate", 0.0), 4),
            "false_positive_rate": round(scores.get("false_positive_rate", 0.0), 4),
            "type_accuracy": round(scores.get("type_accuracy", 0.0), 4),
            "explanation_quality": round(scores.get("explanation_quality", 0.0), 4),
            "submitted_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.entries.append(entry)
        self.entries.sort(key=lambda e: e["overall_score"], reverse=True)
        rank = next(i + 1 for i, e in enumerate(self.entries) if e is entry)
        entry["rank"] = rank
        return entry

    def get_leaderboard(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """Return the top N entries sorted by overall score."""
        for i, entry in enumerate(self.entries[:top_n]):
            entry["rank"] = i + 1
        return self.entries[:top_n]

    def get_model_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Return all submissions for a specific model."""
        return [e for e in self.entries if e["model_name"] == model_name]


# ---------------------------------------------------------------------------
# Global instances
# ---------------------------------------------------------------------------

_global_tracker: Optional[MetricsTracker] = None
_global_leaderboard: Optional[Leaderboard] = None


def get_tracker(session_id: Optional[str] = None) -> MetricsTracker:
    """Get or create the global metrics tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = MetricsTracker(session_id=session_id)
    return _global_tracker


def get_leaderboard() -> Leaderboard:
    """Get or create the global leaderboard."""
    global _global_leaderboard
    if _global_leaderboard is None:
        _global_leaderboard = Leaderboard()
    return _global_leaderboard
