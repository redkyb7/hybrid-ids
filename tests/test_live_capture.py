"""
SentinelFlow IDS - Unit & Integration Tests for LiveCaptureDaemon
=================================================================
Verifies Phase 3 pipeline:
  1. Synthetic Scapy packet stream ingestion & PCAP replay
  2. End-to-end classification through Stage 1 & Stage 2 models
  3. SQLite database persistence with WAL concurrency
  4. Telemetry stats tracking and graceful shutdown
"""

import json
import os
import sys
import time
import sqlite3
import tempfile
import unittest

# Ensure backend modules are on the import path.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import scapy.all as scapy
from live_capture import LiveCaptureDaemon


class TestLiveCaptureDaemon(unittest.TestCase):

    def setUp(self):
        # Create temporary database and pcap file
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_ids_logs.db")
        self.pcap_path = os.path.join(self.tmp_dir, "synthetic_traffic.pcap")
        self.daemons = []

        # Generate synthetic PCAP with Scapy packets
        self._generate_synthetic_pcap()

    def tearDown(self):
        # Allow open database file handles to finalize
        for daemon in self.daemons:
            daemon.db_conn.close()
        time.sleep(0.2)
        for path in [self.pcap_path, self.db_path, f"{self.db_path}-wal", f"{self.db_path}-shm"]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        try:
            os.rmdir(self.tmp_dir)
        except Exception:
            pass

    def _generate_synthetic_pcap(self):
        """Generates synthetic TCP and UDP flows in a pcap file."""
        packets = []

        # Flow 1: Normal HTTP conversation between 192.168.100.101 and 192.168.100.10
        t = time.time()
        # SYN
        packets.append(scapy.IP(src="192.168.100.101", dst="192.168.100.10")/scapy.TCP(sport=50001, dport=80, flags="S", seq=1000))
        # SYN-ACK
        packets.append(scapy.IP(src="192.168.100.10", dst="192.168.100.101")/scapy.TCP(sport=80, dport=50001, flags="SA", seq=2000, ack=1001))
        # ACK
        packets.append(scapy.IP(src="192.168.100.101", dst="192.168.100.10")/scapy.TCP(sport=50001, dport=80, flags="A", seq=1001, ack=2001))
        # HTTP GET Request
        packets.append(scapy.IP(src="192.168.100.101", dst="192.168.100.10")/scapy.TCP(sport=50001, dport=80, flags="PA", seq=1001, ack=2001)/b"GET / HTTP/1.1\r\nHost: 192.168.100.10\r\n\r\n")
        # HTTP 200 OK Response
        packets.append(scapy.IP(src="192.168.100.10", dst="192.168.100.101")/scapy.TCP(sport=80, dport=50001, flags="PA", seq=2001, ack=1050)/b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\nHello World!")
        # FIN
        packets.append(scapy.IP(src="192.168.100.101", dst="192.168.100.10")/scapy.TCP(sport=50001, dport=80, flags="FA", seq=1050, ack=2060))

        # Flow 2: Rapid Port Scan SYN probe from 192.168.100.66 to port 22
        packets.append(scapy.IP(src="192.168.100.66", dst="192.168.100.10")/scapy.TCP(sport=61000, dport=22, flags="S", seq=500))
        packets.append(scapy.IP(src="192.168.100.10", dst="192.168.100.66")/scapy.TCP(sport=22, dport=61000, flags="RA", seq=0, ack=501))

        # Flow 3: Port Scan SYN probe to port 3306 (closed)
        packets.append(scapy.IP(src="192.168.100.66", dst="192.168.100.10")/scapy.TCP(sport=61001, dport=3306, flags="S", seq=600))
        packets.append(scapy.IP(src="192.168.100.10", dst="192.168.100.66")/scapy.TCP(sport=3306, dport=61001, flags="RA", seq=0, ack=601))

        # Write to PCAP
        scapy.wrpcap(self.pcap_path, packets)

    def test_pcap_ingestion_and_sqlite_persistence(self):
        """Verifies daemon ingests PCAP packets, classifies them, and writes logs to SQLite."""
        daemon = LiveCaptureDaemon(
            pcap_file=self.pcap_path,
            db_path=self.db_path,
            log_features=True,
        )
        self.daemons.append(daemon)

        # Run daemon in PCAP mode
        daemon.start(duration_sec=1)

        # Verify statistics
        self.assertGreaterEqual(daemon.stats["packets_sniffed"], 8)
        self.assertGreaterEqual(daemon.stats["flows_analyzed"], 2)

        # Query Database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, source_ip, destination_ip, protocol, attack_type, latency_ms FROM logs")
        rows = cursor.fetchall()
        cursor.execute("""
            SELECT attack_type, verdict, confidence,
                   model_attack_type, model_verdict, model_confidence,
                   stage1_attack_probability, stage1_features_json,
                   flow_features_json, flow_start_epoch, flow_end_epoch
            FROM logs LIMIT 1
        """)
        (attack_type, verdict, confidence, model_label, model_verdict,
         model_confidence, score, feature_json, all_features_json,
         flow_start_epoch, flow_end_epoch) = cursor.fetchone()
        conn.close()

        self.assertGreaterEqual(len(rows), 2, "Database must have at least 2 logged flow verdicts")
        self.assertEqual(attack_type, model_label)
        self.assertEqual(verdict, model_verdict)
        self.assertEqual(confidence, model_confidence)
        self.assertIn(model_verdict, {"BENIGN", "MALICIOUS"})
        self.assertGreaterEqual(score, 0)
        self.assertEqual(len(json.loads(feature_json)), len(daemon.hybrid_engine.stage1_features))
        self.assertEqual(len(json.loads(all_features_json)), len(daemon.hybrid_engine.stage2_features))
        self.assertNotIn("payload_sample", feature_json)
        self.assertNotIn("payload_sample", all_features_json)
        self.assertLessEqual(flow_start_epoch, flow_end_epoch)

        for row in rows:
            row_id, timestamp, src_ip, dst_ip, proto, attack_type, latency_ms = row
            self.assertIsNotNone(timestamp)
            self.assertIn(src_ip, ["192.168.100.101", "192.168.100.66", "192.168.100.10"])
            self.assertEqual(dst_ip, "192.168.100.10")
            self.assertEqual(proto, "TCP")
            self.assertIsInstance(latency_ms, int)
            self.assertGreaterEqual(latency_ms, 0)
            self.assertIn(attack_type, [
                "Normal Traffic", "Port Scan", "DoS", "DDoS", "Brute Force",
                "Web Attack", "Botnet", "Infiltration", "Unknown Attack"
            ])

        print(f"\n[TEST PASS] Successfully verified LiveCaptureDaemon SQLite persistence:")
        print(f"  - Total Flows Logged in DB: {len(rows)}")
        for r in rows:
            print(f"    • [{r[1]}] {r[2]} -> {r[3]} ({r[4]}) | {r[5]} ({r[6]}ms)")

    def test_concurrent_wal_mode_reading(self):
        """Verifies external readers can query SQLite simultaneously while daemon is active without locks."""
        daemon = LiveCaptureDaemon(
            pcap_file=self.pcap_path,
            db_path=self.db_path
        )
        self.daemons.append(daemon)

        # Connect a concurrent reader before daemon starts
        reader_conn = sqlite3.connect(self.db_path)
        reader_cursor = reader_conn.cursor()

        daemon.start(duration_sec=1)

        # Reader query during/after daemon execution
        reader_cursor.execute("SELECT COUNT(*) FROM logs")
        count = reader_cursor.fetchone()[0]
        reader_conn.close()

        self.assertGreaterEqual(count, 2, "Concurrent reader must successfully count rows without locking error")

    def test_stage1_benign_flow_stays_benign_despite_http_probe_payload(self):
        daemon = LiveCaptureDaemon(
            db_path=self.db_path,
            log_features=False,
        )
        self.daemons.append(daemon)
        daemon.hybrid_engine.classify_flow = lambda _flow: {
            "verdict": "BENIGN",
            "attack_type": "Normal Traffic",
            "confidence": 0.98,
            "stage_reached": "Stage 1 (ML)",
            "stage1_attack_probability": 0.02,
        }
        daemon._record_flow_verdict({
            "source_ip": "192.168.100.66",
            "destination_ip": "192.168.100.10",
            "protocol": "TCP",
            "Source Port": 51000,
            "Destination Port": 80,
            "flow_end_ts": time.time(),
            "payload_sample": "GET /search?q=%3Cscript%3Ealert(1) HTTP/1.1\r\n",
        })
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute("""
                SELECT attack_type, verdict, confidence,
                       model_attack_type, model_verdict,
                       stage1_attack_probability
                FROM logs
            """).fetchone()
        self.assertEqual(row, (
            "Normal Traffic", "BENIGN", 0.98,
            "Normal Traffic", "BENIGN", 0.02,
        ))


if __name__ == "__main__":
    unittest.main()
