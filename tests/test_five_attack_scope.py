"""Check that narrowing the DL labels does not turn attacks into benign flows."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deep learning model"))
import config
from preprocess import _remap_training_labels, _validate_dataset_schema
sys.path.insert(0, str(ROOT / "backend"))
from hybrid_engine import HybridIDSEngine


class FiveAttackScopeTests(unittest.TestCase):
    def test_all_source_attacks_stay_attacks(self):
        source = pd.Series(config.SOURCE_CLASSES * 2, name="ClassLabel")
        frame = pd.DataFrame({"ClassLabel": source})
        _validate_dataset_schema(frame)
        mapped = _remap_training_labels(frame["ClassLabel"])
        self.assertEqual(set(mapped), set(config.EXPECTED_CLASSES))
        self.assertEqual(mapped[source == "Infiltration"].unique().tolist(), ["Other Attack"])
        self.assertEqual(mapped[source == "Webattack"].unique().tolist(), ["Other Attack"])
        self.assertTrue(all(mapped[source != "Benign"] != "Benign"))
        self.assertEqual(len(mapped), len(source))

    def test_unexpected_source_class_is_rejected(self):
        frame = pd.DataFrame({"ClassLabel": [*config.SOURCE_CLASSES, "Unseen"]})
        with self.assertRaisesRegex(ValueError, "Unexpected classes"):
            _validate_dataset_schema(frame)

    def test_runtime_accepts_new_bundle_and_flags_other_attack(self):
        class FakeML:
            classes_ = [0, 1]
            feature_names_in_ = ["feature"]

            def predict_proba(self, _):
                return np.array([[0.1, 0.9]])

        class FakeDL:
            feature_order = ["feature"]
            label_encoder = SimpleNamespace(classes_=np.array(sorted(config.EXPECTED_CLASSES)))

            def predict_detailed(self, _):
                return {"label": "Other Attack", "confidence": 0.8}

        fake_module = SimpleNamespace(NIDSClassifier=lambda **_: FakeDL())
        with patch("hybrid_engine.joblib.load", side_effect=[FakeML(), ["feature"]]), \
             patch("hybrid_engine.importlib.import_module", return_value=fake_module):
            engine = HybridIDSEngine(artifact_root="unused", stage1_threshold=0.1)
        verdict = engine.classify_flow({"feature": 1})
        self.assertEqual(verdict["verdict"], "MALICIOUS")
        self.assertEqual(verdict["attack_type"], "Other Attack")
        self.assertEqual(verdict["stage_reached"], "Stage 2 (DL)")


if __name__ == "__main__":
    unittest.main()
