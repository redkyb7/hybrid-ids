"""Prevent standalone DL metrics from appearing as a combined IDS score."""

import json
import tempfile
import unittest
from pathlib import Path

from frontend.metrics import load_model_metrics


class TestDashboardMetrics(unittest.TestCase):
    def test_only_shared_evaluation_supplies_combined_attack_f1(self):
        self.assertEqual(load_model_metrics(None), {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            path.write_text(json.dumps({"macro_f1": 0.77, "accuracy": 0.98}))
            self.assertEqual(load_model_metrics(path), {})
            path.write_text(json.dumps({
                "test_report": {"combined_attack_f1": 0.82}
            }))
            self.assertEqual(load_model_metrics(path), {"combined_attack_f1": 0.82})


if __name__ == "__main__":
    unittest.main()
