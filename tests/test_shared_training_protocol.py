"""Check that the proposed Colab protocol keeps flows and labels aligned."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_shared_models import ML_EXCLUDE, prepare_shared_data, report_test


class TestSharedTrainingProtocol(unittest.TestCase):
    def test_deduplication_precedes_one_common_split(self):
        import json
        with (ROOT / "updated_models" / "extracted" / "dl" /
              "metadata.json").open(encoding="utf-8") as source:
            metadata = json.load(source)
        names = metadata["feature_names"]
        classes = metadata["classes"]
        rng = np.random.default_rng(42)
        features = rng.random((160, len(names)), dtype=np.float32)
        labels = np.repeat(classes, 20)
        frame = pd.DataFrame(features, columns=names)
        frame["ClassLabel"] = labels
        duplicate = frame.iloc[::20].copy()
        conflicting = frame.iloc[[0]].copy()
        conflicting["ClassLabel"] = classes[1]
        invalid = frame.iloc[[1]].copy()
        invalid[names[0]] = np.nan
        same_signature = frame.iloc[[2]].copy()
        excluded_name = next(name for name in names if name in ML_EXCLUDE)
        same_signature[excluded_name] += 10
        frame = pd.concat([frame, duplicate, conflicting, invalid,
                           same_signature],
                          ignore_index=True)

        batches = [frame.iloc[:80], frame.iloc[80:160], frame.iloc[160:]]
        with patch("scripts.train_shared_models.iter_dataset_batches",
                   return_value=batches):
            (X, y, encoder, all_names, ml_names, row_ids,
             train, validation, test, counts) = prepare_shared_data(Path("unused"))
        self.assertEqual(len(X), 159)
        self.assertEqual(counts["ambiguous_removed"], 2)
        self.assertEqual(counts["invalid_rows"], 1)
        self.assertEqual(len(counts["dedup_signature_columns"]), 32)
        self.assertEqual(np.count_nonzero(np.isin(row_ids, [2, 170])), 1)
        self.assertEqual(len(np.unique(X, axis=0)), len(X))
        self.assertEqual(len(set(train) | set(validation) | set(test)), len(X))
        self.assertTrue(set(train).isdisjoint(validation))
        self.assertTrue(set(train).isdisjoint(test))
        self.assertTrue(set(validation).isdisjoint(test))
        self.assertEqual(len(row_ids), len(y))
        self.assertTrue(set(ml_names).issubset(all_names))

    def test_unknown_after_stage1_remains_an_alert(self):
        truth = np.array([0, 1, 1, 0])
        gate = np.array([0.9, 0.9, 0.9, 0.05])
        dl = np.array([[0.6, 0.4], [0.65, 0.35],
                       [0.05, 0.95], [0.2, 0.8]])
        policy = {"stage1_threshold": 0.5, "dl_threshold": 0.7}
        report = report_test(truth, gate, dl, policy, ["Benign", "DDoS"])
        self.assertAlmostEqual(report["combined_attack_recall"], 1.0)
        self.assertAlmostEqual(report["combined_false_positive_rate"], 0.5)
        self.assertEqual(report["unknown_after_gate_count"], 2)
        self.assertAlmostEqual(
            report["by_true_class"]["Benign"]["correct_class_fraction"], 0.5)
        self.assertAlmostEqual(
            report["by_true_class"]["DDoS"]["correct_class_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
