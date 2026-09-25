"""Checks the live feature vector against the uploaded model artifacts."""

import json
import math
import sys
import unittest
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from flow_aggregator import Flow, FlowAggregator


class TestUpdatedFeatureContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        artifact_dir = ROOT / "updated_models" / "five_attack"
        cls.ml_features = joblib.load(
            artifact_dir / "ml" / "stage1_feature_list.joblib"
        )
        with (artifact_dir / "dl" / "metadata.json").open(encoding="utf-8") as source:
            cls.dl_features = json.load(source)["feature_names"]

    def test_saved_schemas_and_packet_statistics(self):
        flow = Flow("10.0.0.1", "10.0.0.2", 1234, 80, "TCP", 100.0)
        flow.add_packet(12, 100.0, True, {"S": True, "P": True},
                        header_len=40, win_size=512, payload_len=12)
        flow.add_packet(30, 100.01, False, {"U": True},
                        header_len=28, win_size=256, payload_len=30)
        flow.add_packet(18, 100.02, True, {"P": True},
                        header_len=20, win_size=512, payload_len=18)

        features = flow.extract_features()
        self.assertEqual(len(self.ml_features), 20)
        self.assertEqual(len(self.dl_features), 57)
        self.assertFalse(set(self.ml_features) - features.keys())
        self.assertFalse(set(self.dl_features) - features.keys())
        self.assertTrue(all(math.isfinite(float(features[name])) for name in self.dl_features))

        self.assertEqual(features["Total Backward Packets"], 1)
        self.assertEqual(features["Fwd Packets Length Total"], 30)
        self.assertEqual(features["Bwd Packets Length Total"], 30)
        self.assertEqual(features["Fwd PSH Flags"], 2)
        self.assertEqual(features["SYN Flag Count"], 1)
        self.assertEqual(features["URG Flag Count"], 1)
        self.assertEqual(features["Init Fwd Win Bytes"], 512)
        self.assertEqual(features["Init Bwd Win Bytes"], 256)
        self.assertEqual(features["Fwd Act Data Packets"], 2)
        self.assertEqual(features["Fwd Seg Size Min"], 20)
        self.assertEqual(features["Subflow Fwd Packets"], 2)
        self.assertEqual(features["Subflow Bwd Bytes"], 30)
        self.assertEqual(features["Avg Fwd Segment Size"], 15)
        self.assertEqual(features["Avg Bwd Segment Size"], 30)

        # The uploaded Parquet data counts the first packet twice for
        # overall length statistics: [12, 12, 30, 18].
        self.assertEqual(features["Packet Length Mean"], 18)
        self.assertEqual(features["Packet Length Variance"], 72)
        self.assertEqual(features["Avg Packet Size"], 24)
        self.assertAlmostEqual(features["Active Mean"], 20_000, places=3)


    def test_active_and_idle_intervals(self):
        flow = Flow("a", "b", 1, 2, "TCP", 100.0)
        for timestamp in (100.0, 101.0, 107.5, 109.0, 116.0, 117.5):
            flow.add_packet(1, timestamp, True)

        features = flow.extract_features()
        self.assertGreater(features["Active Std"], 0)
        self.assertGreater(features["Idle Std"], 0)
        self.assertAlmostEqual(features["Active Min"], 1_000_000)
        self.assertAlmostEqual(features["Active Max"], 1_500_000)
        self.assertAlmostEqual(features["Idle Min"], 6_500_000)
        self.assertAlmostEqual(features["Idle Max"], 7_000_000)

    def test_updated_aggregator_emits_complete_schema_on_timeout(self):
        aggregator = FlowAggregator(micro_batch_timeout_sec=0.1)
        self.assertIsNone(aggregator.process_raw_packet(
            "a", "b", 1, 2, "TCP", 12, timestamp=100.0,
        ))
        features = aggregator.process_raw_packet(
            "a", "b", 1, 2, "TCP", 18, timestamp=100.11,
        )
        self.assertIsNotNone(features)
        self.assertFalse(set(self.ml_features) - features.keys())
        self.assertFalse(set(self.dl_features) - features.keys())


if __name__ == "__main__":
    unittest.main()
