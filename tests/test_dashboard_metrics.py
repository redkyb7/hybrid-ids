"""Read only the selected DL model's standalone classification metric."""

import json
import tempfile
import unittest
from pathlib import Path

from frontend.metrics import load_dl_metrics


class TestDashboardMetrics(unittest.TestCase):
    def test_dl_macro_f1_comes_from_dl_evaluation_metrics(self):
        self.assertEqual(load_dl_metrics(None), {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            path.write_text(json.dumps({"test_report": {"combined_attack_f1": 0.82}}))
            self.assertEqual(load_dl_metrics(path), {})
            path.write_text(json.dumps({"macro_f1": 0.7736512441454672}))
            self.assertEqual(load_dl_metrics(path), {"dl_macro_f1": 0.7736512441454672})
            path.write_text(json.dumps({"macro_f1": 1.2}))
            self.assertEqual(load_dl_metrics(path), {})


if __name__ == "__main__":
    unittest.main()
