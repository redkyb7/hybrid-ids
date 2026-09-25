"""Dashboard keeps recent chart counts separate from historical totals."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frontend import app as dashboard


class TestDashboardTelemetry(unittest.TestCase):
    def test_recent_chart_and_optional_model_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "telemetry.db"
            with sqlite3.connect(db_path) as connection:
                connection.execute("""
                    CREATE TABLE logs (
                        id INTEGER PRIMARY KEY, timestamp TEXT, source_ip TEXT,
                        destination_ip TEXT, protocol TEXT, attack_type TEXT,
                        latency_ms INTEGER
                    )
                """)
                connection.executemany(
                    "INSERT INTO logs VALUES (?, 'time', 'client', 'victim', 'TCP', ?, 2)",
                    [(1, "Normal Traffic"), (2, "Normal Traffic"),
                     (3, "Normal Traffic"), (4, "DoS")],
                )
            metrics_path = Path(directory) / "evaluation_metrics.json"
            metrics_path.write_text(json.dumps({"macro_f1": 0.7736512441454672}))
            with patch.object(dashboard, "DATABASE_PATH", db_path), patch.object(
                dashboard, "DL_METRICS_PATH", metrics_path
            ):
                result = dashboard.read_telemetry(limit=2)
            self.assertEqual(result["threat_mix"], {"DoS": 1, "Normal Traffic": 1})
            self.assertEqual(result["kpis"]["total_flows"], 4)
            self.assertEqual(result["kpis"]["total_attacks"], 1)
            self.assertEqual(result["kpis"]["dl_macro_f1"], 0.7736512441454672)
            self.assertIsNone(result["logs"][0]["stage1_attack_probability"])

    def test_historical_rule_alerts_use_the_saved_model_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "telemetry.db"
            with sqlite3.connect(db_path) as connection:
                connection.execute("""CREATE TABLE logs (
                    id INTEGER PRIMARY KEY, timestamp TEXT, source_ip TEXT,
                    destination_ip TEXT, protocol TEXT, attack_type TEXT,
                    confidence REAL, latency_ms INTEGER,
                    model_attack_type TEXT, model_verdict TEXT,
                    model_confidence REAL, detection_source TEXT, rule_id TEXT
                )""")
                connection.executemany("""
                    INSERT INTO logs VALUES (
                        ?, 'time', 'client', 'victim', 'TCP', ?, ?, 2,
                        ?, ?, ?, ?, ?
                    )
                """, [
                    (1, "Web Attack", None, "Normal Traffic", "BENIGN", 0.93,
                     "rule", "http_probe"),
                    (2, "DoS", 0.82, "DoS", "MALICIOUS", 0.82,
                     "model", None),
                ])
            with patch.object(dashboard, "DATABASE_PATH", db_path), patch.object(
                dashboard, "DL_METRICS_PATH", None
            ):
                result = dashboard.read_telemetry(limit=2)
            self.assertEqual(result["threat_mix"], {"DoS": 1, "Normal Traffic": 1})
            self.assertEqual(result["kpis"]["total_attacks"], 1)
            self.assertEqual(result["logs"][1]["attack_type"], "Normal Traffic")
            self.assertEqual(result["logs"][1]["confidence"], 0.93)
            self.assertEqual(result["latest_alert"]["attack_type"], "DoS")


if __name__ == "__main__":
    unittest.main()
