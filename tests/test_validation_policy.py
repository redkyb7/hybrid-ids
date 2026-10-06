"""Regression checks for domain preservation and deployed Unknown semantics."""
import sys
import unittest
from pathlib import Path
import numpy as np

MODEL = Path(__file__).resolve().parents[1] / 'deep learning model'
if MODEL.is_dir():
    sys.path.insert(0, str(MODEL))
from validation_policy import macro_scores, runtime_attack_mask, select_runtime_threshold


class ValidationPolicyTests(unittest.TestCase):
    def test_uncertain_benign_is_unknown_alert_not_benign(self):
        probabilities = np.array([[.6, .4], [.99, .01], [.4, .6], [.01, .99]])
        self.assertEqual(runtime_attack_mask(probabilities, 0, .8).tolist(), [True, False, True, True])

    def test_threshold_does_not_improve_score_by_hiding_uncertain_attacks(self):
        result = select_runtime_threshold(np.array([[.6, .4], [.4, .6]]), [0, 1], 0, [0, .8])
        self.assertEqual(result['global_threshold'], 0)
        self.assertEqual(result['validation_score'], 1)

    def test_large_cic_domain_cannot_hide_failed_live_domain(self):
        truth = [0, 1] * 100 + [0, 1]
        prediction = [0, 1] * 100 + [1, 0]
        scores = macro_scores(truth, prediction, {'cic': 200, 'live': 2})
        self.assertEqual(scores, {'cic': 1, 'live': 0})

    def test_incomplete_domain_partition_fails(self):
        with self.assertRaises(ValueError):
            macro_scores([0, 1], [0, 1], {'cic': 1})


if __name__ == '__main__':
    unittest.main()
