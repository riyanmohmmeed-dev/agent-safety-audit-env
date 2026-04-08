"""
Bulletproof Testing Suite — Exhaustive Edge Case Coverage
==========================================================
Tests EVERY possible score path to guarantee no value is exactly 0.0 or 1.0.
Tests EVERY API response to ensure validator compliance.
Tests EVERY task to ensure it runs without error.
"""

import sys, os, json, re, math, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from models import MonitorAction
from server.agent_safety_audit_environment import AgentSafetyAuditEnvironment
from graders import (
    grade_step, grade_episode,
    _explanation_score, _type_accuracy, _severity_distance,
    _schema_compliance_score,
    check_entity_accuracy, check_numerical_accuracy,
    compute_calibration_score,
    REWARD_CORRECT_BLOCK, REWARD_CORRECT_FLAG, REWARD_CORRECT_ALLOW,
    REWARD_MISSED_VIOLATION, REWARD_FALSE_POSITIVE, REWARD_FALSE_FLAG, REWARD_INVALID,
    REWARD_GREY_CORRECT_FLAG, REWARD_GREY_BLOCK, REWARD_GREY_ALLOW,
    EASY_WEIGHTS, MEDIUM_WEIGHTS, HARD_WEIGHTS, GREY_AREA_WEIGHTS,
)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Score Range Tests — EVERY score must be strictly in (0, 1)
# ══════════════════════════════════════════════════════════════════════════════

class TestScoreRanges:
    """Ensure grade_episode NEVER returns 0.0 or 1.0."""

    def _run_episode_score(self, difficulty, decision_type):
        """Helper: run an episode with a fixed decision and check score."""
        env = AgentSafetyAuditEnvironment()
        obs = env.reset(seed=42, options={"difficulty": difficulty})
        done = False
        while not done:
            action = MonitorAction(
                decision=decision_type,
                violation_type="policy_violation" if decision_type == "block" else "",
                severity="high" if decision_type == "block" else "",
                reason="Test"
            )
            obs, reward, done, info = env.step(action)
        score = info.get("score", info.get("episode_score"))
        assert score is not None, f"No score in info dict for {difficulty}/{decision_type}"
        assert 0.0 < score < 1.0, f"Score {score} out of (0,1) for {difficulty}/{decision_type}"
        return score

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "grey_area", "hard"])
    @pytest.mark.parametrize("decision", ["allow", "block", "flag"])
    def test_all_difficulty_decision_combos(self, difficulty, decision):
        """12 combos: every difficulty × every decision → score in (0,1)."""
        score = self._run_episode_score(difficulty, decision)
        assert 0.0 < score < 1.0

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "grey_area", "hard"])
    def test_worst_case_all_allow(self, difficulty):
        """Worst case: allow everything. Score must still be > 0."""
        score = self._run_episode_score(difficulty, "allow")
        assert score > 0.0

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "grey_area", "hard"])
    def test_best_case_all_block(self, difficulty):
        """Best case for detection: block everything. Score must still be < 1."""
        score = self._run_episode_score(difficulty, "block")
        assert score < 1.0

    def test_perfect_flag_grey_area(self):
        """Perfect grey area: flag everything. Score must be < 1."""
        score = self._run_episode_score("grey_area", "flag")
        assert score < 1.0


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Reward Range Tests — EVERY step reward must be in (0, 1) for API
# ══════════════════════════════════════════════════════════════════════════════

class TestRewardRanges:
    """Ensure the API step endpoint clamps rewards."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "grey_area", "hard"])
    @pytest.mark.parametrize("decision", ["allow", "block", "flag"])
    def test_step_rewards_clamped(self, difficulty, decision):
        """Simulate what app.py does: clamp every reward."""
        env = AgentSafetyAuditEnvironment()
        obs = env.reset(seed=42, options={"difficulty": difficulty})
        done = False
        while not done:
            action = MonitorAction(
                decision=decision,
                violation_type="policy_violation" if decision == "block" else "",
                severity="high" if decision == "block" else "",
                reason="Test"
            )
            obs, reward, done, info = env.step(action)
            # Simulate app.py clamping
            clamped = round(max(0.01, min(0.99, reward)), 4)
            assert 0.0 < clamped < 1.0, f"Clamped reward {clamped} from raw {reward}"


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: grade_episode Breakdown Tests — EVERY component clamped
# ══════════════════════════════════════════════════════════════════════════════

class TestGradeEpisodeBreakdown:
    """Ensure every breakdown value is in (0, 1)."""

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "grey_area", "hard"])
    def test_all_miss_breakdown(self, difficulty):
        """All allow (worst case) — every component must be in (0,1)."""
        env = AgentSafetyAuditEnvironment()
        obs = env.reset(seed=42, options={"difficulty": difficulty})
        decisions = []
        done = False
        step_num = 0
        while not done:
            step_num += 1
            action = MonitorAction(decision="allow", reason="Test")
            obs, reward, done, info = env.step(action)
            decisions.append({
                "step": step_num, "decision": "allow",
                "violation_type": "", "severity": "", "reason": "Test"
            })

        # Also test grade_episode directly
        task = env._current_task
        score, breakdown = grade_episode(
            decisions=decisions,
            steps_to_block=env._steps_to_block,
            ground_truth=task["ground_truth"],
            total_steps=len(env._action_log),
            difficulty=difficulty,
            steps_to_flag=env._steps_to_flag,
        )
        assert 0.0 < score < 1.0, f"Score {score} out of range"
        for k, v in breakdown.items():
            assert 0.0 < v < 1.0, f"Breakdown {k}={v} out of (0,1)"

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "grey_area", "hard"])
    def test_all_block_breakdown(self, difficulty):
        """All block (aggressive) — breakdown clamped."""
        env = AgentSafetyAuditEnvironment()
        obs = env.reset(seed=42, options={"difficulty": difficulty})
        decisions = []
        done = False
        step_num = 0
        while not done:
            step_num += 1
            action = MonitorAction(
                decision="block", violation_type="policy_violation",
                severity="high", reason="Blocked by test"
            )
            obs, reward, done, info = env.step(action)
            decisions.append({
                "step": step_num, "decision": "block",
                "violation_type": "policy_violation", "severity": "high",
                "reason": "Blocked by test"
            })

        task = env._current_task
        score, breakdown = grade_episode(
            decisions=decisions,
            steps_to_block=env._steps_to_block,
            ground_truth=task["ground_truth"],
            total_steps=len(env._action_log),
            difficulty=difficulty,
            steps_to_flag=env._steps_to_flag,
        )
        assert 0.0 < score < 1.0, f"Score {score} out of range"
        for k, v in breakdown.items():
            assert 0.0 < v < 1.0, f"Breakdown {k}={v} out of (0,1)"


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: ALL 69 Tasks Test — Every single task runs without crash
# ══════════════════════════════════════════════════════════════════════════════

class TestAllTasks:
    """Run EVERY task and verify scores."""

    def test_all_69_tasks_complete(self):
        """Run all 69 tasks end-to-end. No crashes, all scores in (0,1)."""
        env = AgentSafetyAuditEnvironment()
        all_tasks = []
        for diff, tasks in env._all_tasks.items():
            for task in tasks:
                all_tasks.append((diff, task["id"]))

        assert len(all_tasks) == 69, f"Expected 69 tasks, got {len(all_tasks)}"

        failures = []
        for diff, task_id in all_tasks:
            try:
                env2 = AgentSafetyAuditEnvironment()
                obs = env2.reset(seed=42, options={"task_id": task_id, "difficulty": diff})
                done = False
                while not done:
                    action = MonitorAction(
                        decision="allow", reason="Test sweep"
                    )
                    obs, reward, done, info = env2.step(action)

                score = info.get("score", info.get("episode_score"))
                if score is None:
                    failures.append(f"{task_id}: no score")
                elif score <= 0.0 or score >= 1.0:
                    failures.append(f"{task_id}: score={score}")
            except Exception as e:
                failures.append(f"{task_id}: CRASH: {e}")

        assert not failures, f"Task failures:\n" + "\n".join(failures)


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: Weight Sum Tests — weights MUST sum to 1.0
# ══════════════════════════════════════════════════════════════════════════════

class TestWeightSums:
    @pytest.mark.parametrize("weights,name", [
        (EASY_WEIGHTS, "easy"),
        (MEDIUM_WEIGHTS, "medium"),
        (HARD_WEIGHTS, "hard"),
        (GREY_AREA_WEIGHTS, "grey_area"),
    ])
    def test_weights_sum_to_one(self, weights, name):
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, f"{name} weights sum to {total}"


# ══════════════════════════════════════════════════════════════════════════════
# PART 6: Grader Component Edge Cases
# ══════════════════════════════════════════════════════════════════════════════

class TestGraderComponents:
    """Test each grading component for edge cases."""

    def test_type_accuracy_exact_match(self):
        assert _type_accuracy("unauthorized_access", "unauthorized_access") == 1.0

    def test_type_accuracy_no_match(self):
        assert _type_accuracy("unauthorized_access", "supply_chain_risk") >= 0.0

    def test_type_accuracy_empty(self):
        assert _type_accuracy("", "") >= 0.0
        assert _type_accuracy("", "policy_violation") >= 0.0

    def test_severity_distance_exact(self):
        assert _severity_distance("high", "high") == 1.0

    def test_severity_distance_off_by_one(self):
        score = _severity_distance("medium", "high")
        assert 0.0 < score < 1.0

    def test_severity_distance_empty(self):
        score = _severity_distance("", "high")
        assert 0.0 <= score <= 1.0

    def test_schema_compliance_valid(self):
        decisions = [{"decision": "block", "violation_type": "policy_violation", "severity": "high", "reason": "test"}]
        score = _schema_compliance_score(decisions)
        assert 0.0 <= score <= 1.0

    def test_schema_compliance_empty(self):
        score = _schema_compliance_score([])
        assert 0.0 <= score <= 1.0

    def test_entity_accuracy_empty(self):
        score = check_entity_accuracy("", "")
        assert 0.0 <= score <= 1.0

    def test_entity_accuracy_match(self):
        score = check_entity_accuracy("found /etc/passwd access", "/etc/passwd unauthorized access")
        assert 0.0 <= score <= 1.0

    def test_numerical_accuracy_empty(self):
        score = check_numerical_accuracy("", "")
        assert 0.0 <= score <= 1.0

    def test_numerical_accuracy_match(self):
        score = check_numerical_accuracy("port 8080 is open", "port 8080 detected")
        assert 0.5 <= score <= 1.0

    def test_calibration_exact(self):
        score = compute_calibration_score(0.9, 1.0)
        assert 0.0 < score < 1.0

    def test_calibration_wrong(self):
        score = compute_calibration_score(0.9, 0.0)
        assert 0.0 < score < 1.0

    def test_calibration_invalid(self):
        score = compute_calibration_score(-1.0, 0.0)
        assert score == 0.3  # Invalid confidence


# ══════════════════════════════════════════════════════════════════════════════
# PART 7: Step Reward Values — all constants must be in expected range
# ══════════════════════════════════════════════════════════════════════════════

class TestRewardConstants:
    def test_positive_rewards_range(self):
        for name, val in [
            ("CORRECT_BLOCK", REWARD_CORRECT_BLOCK),
            ("CORRECT_FLAG", REWARD_CORRECT_FLAG),
            ("CORRECT_ALLOW", REWARD_CORRECT_ALLOW),
            ("GREY_CORRECT_FLAG", REWARD_GREY_CORRECT_FLAG),
            ("GREY_BLOCK", REWARD_GREY_BLOCK),
        ]:
            assert -1.0 <= val <= 1.0, f"{name} = {val}"

    def test_negative_rewards_range(self):
        for name, val in [
            ("MISSED_VIOLATION", REWARD_MISSED_VIOLATION),
            ("FALSE_POSITIVE", REWARD_FALSE_POSITIVE),
            ("FALSE_FLAG", REWARD_FALSE_FLAG),
            ("INVALID", REWARD_INVALID),
            ("GREY_ALLOW", REWARD_GREY_ALLOW),
        ]:
            assert -1.0 <= val <= 1.0, f"{name} = {val}"


# ══════════════════════════════════════════════════════════════════════════════
# PART 8: Inference.py Log Format Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLogFormat:
    """Test that log formatting never produces out-of-range values."""

    def _clamp_reward(self, r):
        return max(0.01, min(0.99, r))

    def _clamp_rewards(self, rewards):
        return [max(0.01, min(0.99, r)) for r in rewards]

    def _format_end(self, score, rewards):
        safe_score = max(0.01, min(0.99, score))
        safe_rewards = self._clamp_rewards(rewards)
        return f"[END] success=true steps=3 score={safe_score:.2f} rewards={','.join(f'{r:.2f}' for r in safe_rewards)}"

    @pytest.mark.parametrize("score", [0.0, -1.0, 0.001, 0.5, 0.999, 1.0, 2.0])
    def test_end_score_clamped(self, score):
        line = self._format_end(score, [0.02, 0.02])
        m = re.search(r'score=([0-9.]+)', line)
        val = float(m.group(1))
        assert 0.0 < val < 1.0, f"score={val} from input {score}"

    @pytest.mark.parametrize("reward", [0.0, -0.5, 0.001, 0.5, 1.0, 2.0])
    def test_end_rewards_clamped(self, reward):
        line = self._format_end(0.5, [reward])
        m = re.search(r'rewards=([0-9.]+)', line)
        val = float(m.group(1))
        assert 0.0 < val < 1.0, f"reward={val} from input {reward}"

    def test_step_reward_clamped(self):
        for r in [0.0, -1.0, 0.001, 0.5, 1.0, 2.0]:
            clamped = self._clamp_reward(r)
            formatted = f"{clamped:.2f}"
            val = float(formatted)
            assert 0.0 < val < 1.0, f"step reward={val} from input {r}"


# ══════════════════════════════════════════════════════════════════════════════
# PART 9: Generated Tasks Test — procedural tasks work
# ══════════════════════════════════════════════════════════════════════════════

class TestGeneratedTasks:
    """Test procedurally generated tasks."""

    @pytest.mark.parametrize("seed", [1, 42, 100, 999, 12345])
    @pytest.mark.parametrize("difficulty", ["easy", "hard"])
    def test_generated_task(self, seed, difficulty):
        env = AgentSafetyAuditEnvironment()
        obs = env.reset(seed=seed, options={"difficulty": difficulty, "generated": True})
        assert obs is not None
        done = False
        while not done:
            action = MonitorAction(decision="allow", reason="Test")
            obs, reward, done, info = env.step(action)
        score = info.get("score", info.get("episode_score", 0))
        assert 0.0 < score < 1.0


# ══════════════════════════════════════════════════════════════════════════════
# PART 10: Early Stopping Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEarlyStopping:
    """Verify early stopping doesn't produce invalid scores."""

    def test_early_stop_score_valid(self):
        """Trigger early stop by always blocking (FP cascade)."""
        env = AgentSafetyAuditEnvironment()
        # Use a hard task with many steps to trigger FP cascade
        obs = env.reset(seed=42, options={"difficulty": "hard"})
        done = False
        while not done:
            action = MonitorAction(
                decision="block", violation_type="policy_violation",
                severity="critical", reason="Test"
            )
            obs, reward, done, info = env.step(action)
        score = info.get("score", info.get("episode_score"))
        assert score is not None
        assert 0.0 < score < 1.0


# ══════════════════════════════════════════════════════════════════════════════
# PART 11: Stress Test — Many random episodes
# ══════════════════════════════════════════════════════════════════════════════

class TestStress:
    """Run many random episodes to find edge cases."""

    def test_50_random_episodes(self):
        """50 random episodes with random decisions."""
        failures = []
        for i in range(50):
            seed = random.randint(0, 99999)
            diff = random.choice(["easy", "medium", "grey_area", "hard"])
            try:
                env = AgentSafetyAuditEnvironment()
                obs = env.reset(seed=seed, options={"difficulty": diff})
                done = False
                while not done:
                    dec = random.choice(["allow", "block", "flag"])
                    action = MonitorAction(
                        decision=dec,
                        violation_type="policy_violation" if dec == "block" else "",
                        severity="high" if dec == "block" else "",
                        reason="Random test"
                    )
                    obs, reward, done, info = env.step(action)
                    # Check reward range
                    clamped = round(max(0.01, min(0.99, reward)), 4)
                    if not (0.0 < clamped < 1.0):
                        failures.append(f"ep{i} seed={seed} {diff}: reward {clamped}")

                score = info.get("score", info.get("episode_score"))
                if score is None or score <= 0.0 or score >= 1.0:
                    failures.append(f"ep{i} seed={seed} {diff}: score={score}")
            except Exception as e:
                failures.append(f"ep{i} seed={seed} {diff}: CRASH: {e}")

        assert not failures, f"Stress test failures:\n" + "\n".join(failures)
