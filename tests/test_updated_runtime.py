"""Run real uploaded artifacts from packets through the live SQLite path."""

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scapy.all import IP, TCP, UDP, Raw, wrpcap


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from live_capture import LiveCaptureDaemon


class TestUpdatedRuntime(unittest.TestCase):
    def test_pcap_replay_finishes_and_flushes_pending_flows(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "ids.db")
            pcap_path = str(Path(directory) / "traffic.pcap")
            packets = [
                IP(src="10.0.0.1", dst="10.0.0.2") /
                TCP(sport=40000, dport=80, flags="S"),
                IP(src="10.0.0.2", dst="10.0.0.1") /
                TCP(sport=80, dport=40000, flags="SA"),
                IP(src="10.0.0.1", dst="10.0.0.2") /
                TCP(sport=40000, dport=80, flags="FA"),
                IP(src="10.0.0.3", dst="10.0.0.2") /
                UDP(sport=53000, dport=53) / Raw(load=b"hello"),
            ]
            for index, packet in enumerate(packets):
                packet.time = 1_700_000_000.0 + index * 0.02
            wrpcap(pcap_path, packets)

            with patch.dict(os.environ, {"IDS_STAGE1_THRESHOLD": "0"}):
                daemon = LiveCaptureDaemon(pcap_file=pcap_path, db_path=db_path)
            try:
                daemon.start()
                rows = daemon.db_conn.execute(
                    "SELECT verdict, stage_reached FROM logs ORDER BY id"
                ).fetchall()
                self.assertEqual(len(rows), 2)
                self.assertEqual(daemon.stats["classification_errors"], 0)
                self.assertEqual(daemon.stats["flows_analyzed"], 2)
                self.assertTrue(all(stage == "Stage 2 (DL)"
                                    for _, stage in rows))
            finally:
                daemon.db_conn.close()

    def test_packet_to_stage2_database_and_strict_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "ids.db")
            with patch.dict(os.environ, {"IDS_STAGE1_THRESHOLD": "0"}):
                daemon = LiveCaptureDaemon(db_path=db_path)

            try:
                self.assertEqual(len(daemon.hybrid_engine.stage2_features), 57)
                packets = [
                    IP(src="10.0.0.1", dst="10.0.0.2") /
                    TCP(sport=40000, dport=80, flags="S", window=4096),
                    IP(src="10.0.0.2", dst="10.0.0.1") /
                    TCP(sport=80, dport=40000, flags="SA", window=8192),
                    IP(src="10.0.0.1", dst="10.0.0.2") /
                    TCP(sport=40000, dport=80, flags="PA") / Raw(load=b"GET /"),
                    IP(src="10.0.0.1", dst="10.0.0.2") /
                    TCP(sport=40000, dport=80, flags="FA"),
                ]
                features = None
                for index, packet in enumerate(packets):
                    packet.time = 100.0 + index * 0.02
                    result = daemon.flow_aggregator.process_scapy_packet(packet)
                    if index < 3:
                        self.assertIsNone(result)
                    else:
                        features = result
                self.assertIsNotNone(features)

                daemon._record_flow_verdict(features)
                row = daemon.db_conn.execute(
                    "SELECT verdict, attack_type, stage_reached FROM logs"
                ).fetchone()
                self.assertEqual(row[2], "Stage 2 (DL)")
                self.assertIn(row[0], ("BENIGN", "MALICIOUS"))
                self.assertEqual(daemon.stats["flows_analyzed"], 1)

                invalid_features = dict(features)
                del invalid_features["Flow Duration"]
                with self.assertRaisesRegex(ValueError, "Missing model features"):
                    daemon._record_flow_verdict(invalid_features)
                count = daemon.db_conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
                self.assertEqual(count, 1)

                daemon.hybrid_engine.stage1_threshold = 1.0
                with patch.object(
                    daemon.hybrid_engine.stage2_classifier,
                    "predict_detailed",
                    side_effect=AssertionError("Stage 2 should not run"),
                ):
                    gated = daemon.hybrid_engine.classify_flow(features)
                self.assertEqual(gated["stage_reached"], "Stage 1 (ML)")
                self.assertEqual(gated["verdict"], "BENIGN")
            finally:
                daemon.db_conn.close()

    def test_existing_database_schema_is_extended(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "old.db")
            with sqlite3.connect(db_path) as connection:
                connection.execute("""CREATE TABLE logs (
                    id INTEGER PRIMARY KEY, timestamp TEXT, source_ip TEXT,
                    destination_ip TEXT, protocol TEXT, attack_type TEXT,
                    latency_ms INTEGER, source_port INTEGER,
                    destination_port INTEGER, confidence REAL
                )""")
            with patch.dict(os.environ, {"IDS_STAGE1_THRESHOLD": "0"}):
                daemon = LiveCaptureDaemon(db_path=db_path)
            try:
                columns = {
                    row[1] for row in daemon.db_conn.execute("PRAGMA table_info(logs)")
                }
                self.assertIn("verdict", columns)
                self.assertIn("stage_reached", columns)
            finally:
                daemon.db_conn.close()


if __name__ == "__main__":
    unittest.main()
