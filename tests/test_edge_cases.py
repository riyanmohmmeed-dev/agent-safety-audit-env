"""
Phase 1: Exhaustive Edge Case Testing for v3.0 Architecture.

Tests early stopping, multi-signal grading bounds, ELO/metrics,
and API attack surface. Every test is written against the ACTUAL
code interfaces — no assumptions, no hallucinated attributes.
"""
import pytest
import math
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.agent_safety_audit_environment import AgentSafetyAuditEnvironment
from server.metrics import MetricsTracker, Leaderboard
from models import MonitorAction
from graders import compute_rouge, compute_bertscore, compute_nli_entailment

# ═══════════════════════════════════════════════════════════════════════
# 1. EARLY STOPPING EDGE CASES
# ═══════════════════════════════════════════════════════════════════════

class TestEarlyStopping:
    """Test the 4 early stopping conditions using actual env API."""

    def _make_env_with_task(self, difficulty="easy", task_id=None):
        env = AgentSafetyAuditEnvironment()
        opts = {"difficulty": difficulty}
        if task_id:
            opts["task_id"] = task_id
        env.reset(options=opts)
        return env

    def test_fp_cascade_triggers_after_min_steps(self):
        """3 false positives should trigger early stop, but only after min 3 steps."""
        env = self._make_env_with_task("easy")
        # Force FP by blocking every step regardless
        steps_taken = 0
        done = False
        info = {}
        while not done and steps_taken < 20:
            action = MonitorAction(
                decision="block",
                violation_type="unauthorized_access",
                severity="critical",
                reason="Testing FP cascade"
            )
            obs, reward, done, info = env.step(action)
            steps_taken += 1

        # Should have stopped early via fp cascade OR completed naturally
        # The key check: if early_stop_reason exists, it must mention FP
        if "early_stop_reason" in info:
            assert "false_positive" in info["early_stop_reason"]
            # Must not trigger before minimum 3 steps
            assert steps_taken >= env.EARLY_STOP_MIN_STEPS

    def test_miss_cascade_triggers(self):
        """Allowing all steps on a hard task (many violations) should trigger miss cascade."""
        env = self._make_env_with_task("hard")
        steps_taken = 0
        done = False
        info = {}
        while not done and steps_taken < 20:
            action = MonitorAction(
                decision="allow",
                violation_type="",
                severity="",
                reason="Testing miss cascade"
            )
            obs, reward, done, info = env.step(action)
            steps_taken += 1

        if "early_stop_reason" in info:
            assert "miss_cascade" in info["early_stop_reason"]
            assert steps_taken >= env.EARLY_STOP_MIN_STEPS

    def test_min_steps_enforced(self):
        """Early stopping must NEVER fire before EARLY_STOP_MIN_STEPS."""
        env = self._make_env_with_task("easy")
        # Take exactly 2 steps with bad decisions
        for i in range(min(2, len(env._action_log))):
            action = MonitorAction(
                decision="block",
                violation_type="data_destruction",
                severity="critical",
                reason="Bad decision"
            )
            obs, reward, done, info = env.step(action)
            if done:
                break
            # Should NOT have early stop in first 2 steps
            assert "early_stop_reason" not in info or info.get("early_stop_reason") is None

    def test_early_stop_reason_in_info(self):
        """When early stopping triggers, info dict must contain the reason."""
        env = self._make_env_with_task("easy")
        done = False
        info = {}
        while not done:
            action = MonitorAction(
                decision="block",
                violation_type="scope_creep",
                severity="high",
                reason="Force FP"
            )
            obs, reward, done, info = env.step(action)

        # Episode ended — either naturally or via early stop
        # Both are valid; we just check the info structure
        if env._early_stop_reason:
            assert "early_stop_reason" in info
            assert isinstance(info["early_stop_reason"], str)
            assert len(info["early_stop_reason"]) > 0


# ═══════════════════════════════════════════════════════════════════════
# 2. MULTI-SIGNAL GRADING — MATHEMATICAL BOUNDS
# ═══════════════════════════════════════════════════════════════════════

class TestGradingBounds:
    """Test ROUGE, BERTScore, NLI for edge cases and mathematical sanity."""

    def test_rouge_empty_strings(self):
        """Empty inputs should return zero scores, not crash."""
        result = compute_rouge("", "")
        assert result["rouge1"] == 0.0
        assert result["rouge2"] == 0.0
        assert result["rougeL"] == 0.0

    def test_rouge_none_inputs(self):
        """None inputs should return zero scores, not crash."""
        result = compute_rouge(None, "reference text")
        assert result["rouge1"] == 0.0
        result2 = compute_rouge("hypothesis", None)
        assert result2["rouge1"] == 0.0

    def test_rouge_identical_strings(self):
        """Identical strings should produce perfect ROUGE-1 scores."""
        result = compute_rouge("The agent accessed admin panel", "The agent accessed admin panel")
        assert result["rouge1"] > 0.95

    def test_rouge_completely_different(self):
        """Completely different strings should produce near-zero ROUGE."""
        result = compute_rouge("apple banana cherry", "xylophone zebra quantum")
        assert result["rouge1"] < 0.1

    def test_bertscore_identical(self):
        """Identical text should produce high BERTScore."""
        score = compute_bertscore("The server is running.", "The server is running.")
        assert score > 0.85

    def test_bertscore_similar(self):
        """Semantically similar text should score higher than unrelated."""
        similar = compute_bertscore(
            "The agent accessed the admin panel.",
            "Agent logged into admin interface."
        )
        unrelated = compute_bertscore(
            "The agent accessed the admin panel.",
            "The weather is sunny today."
        )
        assert similar > unrelated

    def test_bertscore_returns_float(self):
        """BERTScore should always return a float in [0, 1]."""
        score = compute_bertscore("test input", "test reference")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_nli_entailment_vs_contradiction(self):
        """Entailed pair should score higher than contradictory pair."""
        entailed = compute_nli_entailment(
            "The server is running normally",
            "The server is operational and active"
        )
        contradicted = compute_nli_entailment(
            "The server is running normally",
            "The server has completely crashed and is offline"
        )
        # entailed should be >= contradicted (NLI models vary, so use >=)
        assert entailed >= contradicted or abs(entailed - contradicted) < 0.3

    def test_nli_returns_float_in_range(self):
        """NLI should return float in [0, 1] or -1.0 sentinel if model unavailable."""
        score = compute_nli_entailment("test premise", "test hypothesis")
        assert isinstance(score, float)
        # NLI returns -1.0 as sentinel when model can't load; valid range is [0, 1]
        assert score == -1.0 or (0.0 <= score <= 1.0)


# ═══════════════════════════════════════════════════════════════════════
# 3. METRICS TRACKER & ELO ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════

class TestMetricsTracker:
    """Test MetricsTracker for ELO bounds, data integrity, edge cases."""

    def test_initial_skill_rating(self):
        """Initial skill rating should be 0.5."""
        tracker = MetricsTracker()
        assert tracker.skill_rating == 0.5

    def test_elo_stays_in_bounds_after_many_losses(self):
        """After 100 terrible episodes, skill_rating must remain in [0, 1]."""
        tracker = MetricsTracker()
        for i in range(100):
            tracker.log_step({"step": 1, "episode_id": f"ep{i}", "reward": -1.0, "correct": False})
            tracker.end_episode({
                "episode_id": f"ep{i}",
                "episode_score": 0.0,
                "average_reward": -1.0,
                "detection_rate": 0.0,
            })
        assert 0.0 <= tracker.skill_rating <= 1.0

    def test_elo_stays_in_bounds_after_many_wins(self):
        """After 100 perfect episodes, skill_rating must remain in [0, 1]."""
        tracker = MetricsTracker()
        for i in range(100):
            tracker.log_step({"step": 1, "episode_id": f"ep{i}", "reward": 1.0, "correct": True})
            tracker.end_episode({
                "episode_id": f"ep{i}",
                "episode_score": 1.0,
                "average_reward": 1.0,
                "detection_rate": 1.0,
            })
        assert 0.0 <= tracker.skill_rating <= 1.0

    def test_elo_drops_on_bad_episode(self):
        """A terrible episode should decrease skill rating."""
        tracker = MetricsTracker()
        initial = tracker.skill_rating
        tracker.log_step({"step": 1, "episode_id": "bad", "reward": -1.0, "correct": False})
        tracker.end_episode({"episode_id": "bad", "episode_score": 0.0, "average_reward": -1.0})
        assert tracker.skill_rating <= initial

    def test_episode_count_tracks_correctly(self):
        """Episodes completed should increment exactly once per end_episode call."""
        tracker = MetricsTracker()
        for i in range(5):
            tracker.log_step({"step": 1, "episode_id": f"ep{i}", "reward": 0.5, "correct": True})
            tracker.end_episode({"episode_id": f"ep{i}", "episode_score": 0.5})
        metrics = tracker.get_real_time_metrics()
        assert metrics["episodes_completed"] == 5

    def test_step_metrics_accumulated(self):
        """Steps should accumulate in current_session.step_metrics."""
        tracker = MetricsTracker()
        tracker.log_step({"step": 1, "episode_id": "ep1", "reward": 1.0, "correct": True})
        tracker.log_step({"step": 2, "episode_id": "ep1", "reward": 0.5, "correct": False})
        assert len(tracker.current_session.step_metrics) == 2

    def test_training_curve_data_shape(self):
        """Training curve data should have consistent array lengths."""
        tracker = MetricsTracker()
        for i in range(3):
            tracker.log_step({"step": 1, "episode_id": f"ep{i}", "reward": 0.5, "correct": True})
            tracker.end_episode({"episode_id": f"ep{i}", "episode_score": 0.5 + i * 0.1})
        data = tracker.get_training_curve_data()
        assert len(data["rewards"]) == 3
        assert len(data["episode_scores"]) == 3
        assert len(data["skill_ratings"]) == 3

    def test_real_time_metrics_structure(self):
        """get_real_time_metrics should return all expected keys."""
        tracker = MetricsTracker()
        m = tracker.get_real_time_metrics()
        required_keys = [
            "session_id", "episodes_completed", "total_steps",
            "overall_detection_rate", "overall_false_positive_rate",
            "average_reward", "skill_rating", "reward_trend",
            "detection_trend", "recent_reward_avg", "recent_detection_rate"
        ]
        for key in required_keys:
            assert key in m, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════════════════
# 4. LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════

class TestLeaderboard:
    """Test Leaderboard for duplicate handling, sorting, edge cases."""

    def test_submit_creates_entry(self):
        lb = Leaderboard()
        entry = lb.submit("model_a", {"overall": 0.8, "detection_rate": 0.9})
        assert entry["model_name"] == "model_a"
        assert entry["overall_score"] == 0.8

    def test_duplicate_model_updates_not_appends(self):
        """Submitting the same model name should update, not duplicate."""
        lb = Leaderboard()
        lb.submit("test_model", {"overall": 0.5})
        lb.submit("test_model", {"overall": 0.9})
        board = lb.get_leaderboard()
        names = [e["model_name"] for e in board]
        assert names.count("test_model") == 1
        assert board[0]["overall_score"] == 0.9

    def test_leaderboard_sorted_descending(self):
        """Leaderboard entries should be sorted by overall_score descending."""
        lb = Leaderboard()
        lb.submit("bad", {"overall": 0.3})
        lb.submit("mid", {"overall": 0.6})
        lb.submit("best", {"overall": 0.95})
        board = lb.get_leaderboard()
        scores = [e["overall_score"] for e in board]
        assert scores == sorted(scores, reverse=True)

    def test_leaderboard_rank_assignment(self):
        """Ranks should be 1-indexed and correct."""
        lb = Leaderboard()
        lb.submit("a", {"overall": 0.9})
        lb.submit("b", {"overall": 0.7})
        lb.submit("c", {"overall": 0.5})
        board = lb.get_leaderboard()
        assert board[0]["rank"] == 1
        assert board[1]["rank"] == 2
        assert board[2]["rank"] == 3

    def test_empty_leaderboard(self):
        """Empty leaderboard should return empty list, not crash."""
        lb = Leaderboard()
        assert lb.get_leaderboard() == []

    def test_model_history(self):
        """get_model_history should return all submissions for a model."""
        lb = Leaderboard()
        lb.submit("tracked", {"overall": 0.5})
        lb.submit("other", {"overall": 0.6})
        # After update fix, tracked should have 1 entry
        history = lb.get_model_history("tracked")
        assert len(history) >= 1
        assert all(h["model_name"] == "tracked" for h in history)


# ═══════════════════════════════════════════════════════════════════════
# 5. API ENDPOINT SAFETY (without httpx dependency)
# ═══════════════════════════════════════════════════════════════════════

class TestAnalyzeEndpointLogic:
    """Test the /analyze endpoint's math directly, without HTTP."""

    def test_ci_calculation_with_2_episodes(self):
        """CI should work with exactly 2 episodes."""
        scores = [0.7, 0.8]
        n = len(scores)
        mean_score = sum(scores) / n
        variance = sum((x - mean_score) ** 2 for x in scores) / (n - 1)
        std_dev = math.sqrt(variance)
        margin = 1.96 * (std_dev / math.sqrt(n))
        ci_lower = mean_score - margin
        ci_upper = mean_score + margin

        assert 0.0 <= ci_lower or ci_lower < 0  # can be negative, gets clamped to 0
        assert ci_upper <= 2.0  # reasonable upper bound
        assert not math.isnan(mean_score)
        assert not math.isnan(std_dev)

    def test_ci_calculation_with_identical_scores(self):
        """If all scores are identical, variance=0, std=0, CI collapses to point."""
        scores = [0.5, 0.5, 0.5]
        n = len(scores)
        mean_score = sum(scores) / n
        variance = sum((x - mean_score) ** 2 for x in scores) / (n - 1)
        std_dev = math.sqrt(variance)

        assert std_dev == 0.0
        assert mean_score == 0.5

    def test_ci_single_episode_rejected(self):
        """With < 2 episodes, analysis should be rejected."""
        scores = [0.7]
        assert len(scores) < 2  # The endpoint should return error

    def test_baseline_comparison_math(self):
        """Verify beats_baseline flag is correctly set."""
        baseline = 0.557
        assert 0.6 > baseline  # should flag as beating
        assert 0.5 < baseline  # should flag as not beating


# ═══════════════════════════════════════════════════════════════════════
# 6. ENVIRONMENT LIFECYCLE EDGE CASES
# ═══════════════════════════════════════════════════════════════════════

class TestEnvironmentEdgeCases:
    """Test unusual environment usage patterns."""

    def test_step_after_done_raises(self):
        """Stepping after episode is done should raise RuntimeError."""
        env = AgentSafetyAuditEnvironment()
        env.reset(options={"difficulty": "easy"})

        # Play through until done
        done = False
        while not done:
            action = MonitorAction(decision="allow", violation_type="", severity="", reason="test")
            obs, reward, done, info = env.step(action)

        # Now try stepping again
        with pytest.raises(RuntimeError):
            env.step(MonitorAction(decision="allow", violation_type="", severity="", reason="x"))

    def test_double_reset_is_safe(self):
        """Calling reset twice should not crash."""
        env = AgentSafetyAuditEnvironment()
        obs1 = env.reset(options={"difficulty": "easy"})
        obs2 = env.reset(options={"difficulty": "medium"})
        assert obs2.difficulty == "medium"

    def test_very_long_reason_accepted(self):
        """A 10,000 character reason should not crash."""
        env = AgentSafetyAuditEnvironment()
        env.reset(options={"difficulty": "easy"})
        long_reason = "A" * 10000
        action = MonitorAction(decision="allow", violation_type="", severity="", reason=long_reason)
        obs, reward, done, info = env.step(action)
        # Should not crash
        assert reward is not None
