"""Behavior and HTTP checks that the numeric flow models cannot observe."""

import os
import sys
import unittest


BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from supplemental_detector import SupplementalDetector


def flow(port, source_port, timestamp, payload="", forward=1, backward=1):
    return {
        "source_ip": "192.168.100.66",
        "destination_ip": "192.168.100.10",
        "Source Port": source_port,
        "Destination Port": port,
        "protocol": "TCP",
        "flow_end_ts": timestamp,
        "SYN Flag Count": 1,
        "Total Fwd Packets": forward,
        "Total Backward Packets": backward,
        "payload_sample": payload,
    }


class TestSupplementalDetector(unittest.TestCase):
    def test_scan_requires_distinct_ports_and_deduplicates_snapshots(self):
        detector = SupplementalDetector()
        for index in range(5):
            item = flow(20 + index, 50000 + index, 100 + index)
            self.assertIsNone(detector.inspect(item))
            self.assertIsNone(detector.inspect(item))
        self.assertEqual(
            detector.inspect(flow(25, 50005, 105)),
            {"attack_type": "Port Scan", "rule_id": "multi_port_scan"},
        )
        # A normal HTTP connection after the scan is not itself a scan probe.
        self.assertIsNone(detector.inspect(flow(80, 51000, 106, forward=5, backward=5)))
        # Old port observations must expire even on a repeated connection snapshot.
        self.assertIsNone(detector.inspect(flow(25, 50005, 120)))

    def test_syn_burst_and_window_expiry(self):
        detector = SupplementalDetector()
        for index in range(19):
            self.assertIsNone(detector.inspect(flow(80, 60000 + index, 100 + index * 0.1)))
        self.assertEqual(
            detector.inspect(flow(80, 60019, 102)),
            {"attack_type": "DoS", "rule_id": "syn_burst"},
        )
        self.assertIsNone(detector.inspect(flow(80, 62000, 110)))

    def test_login_rate_counts_connections_even_after_early_snapshot(self):
        detector = SupplementalDetector()
        for index in range(5):
            item = flow(80, 50000 + index, 100 + index, forward=5, backward=5)
            self.assertIsNone(detector.inspect(item))
            item["payload_sample"] = "POST /login HTTP/1.1\r\nHost: victim\r\n\r\n"
            result = detector.inspect(item)
            if index < 4:
                self.assertIsNone(result)
            else:
                self.assertEqual(result, {"attack_type": "Brute Force", "rule_id": "login_rate"})

    def test_web_probes_and_botnet_header_without_benign_matches(self):
        detector = SupplementalDetector()
        benign = flow(80, 50001, 100, "GET /api/status HTTP/1.1\r\nUser-Agent: Mozilla/5.0\r\n", 5, 5)
        self.assertIsNone(detector.inspect(benign))
        sql = flow(80, 50002, 101, "GET /search?q=%27+OR+1%3D1+-- HTTP/1.1\r\n", 5, 5)
        self.assertEqual(detector.inspect(sql)["attack_type"], "Web Attack")
        xss = flow(80, 50003, 102, "GET /search?q=%3Cscript%3Ealert(1) HTTP/1.1\r\n", 5, 5)
        self.assertEqual(detector.inspect(xss)["attack_type"], "Web Attack")
        beacon = flow(80, 50004, 103, "GET /api/status HTTP/1.1\r\nUser-Agent: Mirai/Botnet-Client-v1.4\r\n", 5, 5)
        self.assertEqual(
            detector.inspect(beacon),
            {"attack_type": "Botnet", "rule_id": "beacon_header"},
        )


if __name__ == "__main__":
    unittest.main()
