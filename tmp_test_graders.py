import sys
sys.path.append("/Users/mohammedriyan/Projects/agent_safety_audit_env")
from graders import grade_episode
from models import ActionDecision

decisions = [
    ActionDecision(step=1, decision="block", violation_type="unauthorized_access", severity="high", reason="test")
]

score, breakdown = grade_episode(
    decisions=decisions,
    steps_to_block=[1],
    expected_violation="unauthorized_access",
    expected_severity="high",
    difficulty="easy",
    total_steps=1,
    schema_compliant=True
)
print("SCORE:", score)
print("BREAKDOWN:", breakdown)
