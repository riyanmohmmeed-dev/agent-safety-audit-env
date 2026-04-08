import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graders import compute_bertscore, compute_rouge, compute_nli_entailment, _get_bert_scorer
from server.metrics import MetricsTracker

class TestV3Features(unittest.TestCase):
    def test_bertscore_loads(self):
        scorer = _get_bert_scorer()
        self.assertIsNotNone(scorer)
        
    def test_bertscore_similarity(self):
        score1 = compute_bertscore("The agent accessed the admin panel.", "Agent logged into admin interface.")
        score2 = compute_bertscore("The agent accessed the admin panel.", "The user bought a television.")
        self.assertGreater(score1, score2)
        
    def test_metrics_tracker_analysis(self):
        t = MetricsTracker()
        for _ in range(5):
            t.log_step({'step': 1, 'episode_id': 'ep1', 'reward': 0.5, 'decision': 'allow', 'correct': True, 'difficulty': 'easy'})
            t.end_episode({'episode_id': 'ep1', 'average_reward': 0.8, 'detection_rate': 0.9, 'episode_score': 0.85})
        
        m = t.get_real_time_metrics()
        self.assertEqual(m["episodes_completed"], 5)
        
if __name__ == '__main__':
    unittest.main()
