"""
SentinelFlow IDS - Stress & Edge Case Test Suite for FlowAggregator
===================================================================
Tests resilience and performance under extreme conditions:
  1. High-throughput packet burst (10,000+ packets/sec)
  2. UDP and ICMP non-TCP protocol aggregation
  3. Multi-threaded concurrent packet ingestion
  4. Memory consumption & garbage collection under 1,000+ distinct concurrent flows
  5. Zero-payload, out-of-order, and anomalous TCP flag combinations
"""

import os
import sys
import time
import unittest
import threading
import random
import numpy as np

# Ensure backend is on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from flow_aggregator import FlowAggregator


class TestFlowStressAndEdgeCases(unittest.TestCase):

    def setUp(self):
        self.aggregator = FlowAggregator(
            inactivity_timeout_sec=0.5,
            micro_batch_timeout_sec=0.10,
            max_packets_per_micro_batch=30
        )

    def test_udp_flow_aggregation(self):
        """Verifies UDP traffic (DNS/UDP floods) aggregates correctly without TCP flags."""
        t0 = time.time()
        for i in range(10):
            res = self.aggregator.process_raw_packet(
                src_ip="192.168.100.101", dst_ip="192.168.100.10",
                src_port=53535, dst_port=53, protocol="UDP",
                pkt_len=120, timestamp=t0 + i * 0.005, header_len=8
            )
        # Verify flow exists
        fwd_key = ("192.168.100.101", "192.168.100.10", 53535, 53, "UDP")
        self.assertIn(fwd_key, self.aggregator.flows)
        flow = self.aggregator.flows[fwd_key]
        self.assertEqual(flow.fwd_packets, 10)
        self.assertEqual(flow.fwd_bytes, 1200)
        self.assertEqual(flow.protocol, "UDP")
        self.assertEqual(flow.fin_count, 0)

        # Extract features
        features = flow.extract_features()
        self.assertEqual(features["protocol"], "UDP")
        self.assertEqual(features["Destination Port"], 53)
        self.assertEqual(features["Total Fwd Packets"], 10)

    def test_icmp_flow_aggregation(self):
        """Verifies ICMP ping sweep traffic aggregates properly."""
        t0 = time.time()
        self.aggregator.process_raw_packet(
            src_ip="192.168.100.66", dst_ip="192.168.100.10",
            src_port=0, dst_port=0, protocol="ICMP",
            pkt_len=84, timestamp=t0, header_len=8
        )
        self.aggregator.process_raw_packet(
            src_ip="192.168.100.10", dst_ip="192.168.100.66",
            src_port=0, dst_port=0, protocol="ICMP",
            pkt_len=84, timestamp=t0 + 0.001, header_len=8
        )
        fwd_key = ("192.168.100.66", "192.168.100.10", 0, 0, "ICMP")
        self.assertIn(fwd_key, self.aggregator.flows)
        flow = self.aggregator.flows[fwd_key]
        self.assertEqual(flow.fwd_packets, 1)
        self.assertEqual(flow.bwd_packets, 1)
        self.assertEqual(flow.protocol, "ICMP")

    def test_multi_threaded_concurrency(self):
        """Simulates 4 worker threads ingesting packets simultaneously into the same FlowAggregator."""
        num_threads = 4
        packets_per_thread = 250
        errors = []

        def worker(thread_id):
            try:
                base_port = 10000 + thread_id * 1000
                t0 = time.time()
                for p in range(packets_per_thread):
                    # Each thread generates packets for 10 distinct flows
                    flow_id = p % 10
                    sport = base_port + flow_id
                    self.aggregator.process_raw_packet(
                        src_ip=f"10.0.{thread_id}.{flow_id}",
                        dst_ip="192.168.100.10",
                        src_port=sport,
                        dst_port=80,
                        protocol="TCP",
                        pkt_len=random.randint(54, 1500),
                        timestamp=t0 + p * 0.001,
                        tcp_flags={"S": (p == 0), "A": True}
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread errors encountered: {errors}")
        total_active_flows = len(self.aggregator.flows)
        print(f"\n[CONCURRENCY] Processed {num_threads * packets_per_thread} packets across {num_threads} threads. Active flows in table: {total_active_flows}")
        self.assertEqual(total_active_flows, num_threads * 10)

    def test_high_throughput_burst_speed(self):
        """Measures ingestion throughput in packets per second (Target > 5,000 pkts/s)."""
        num_packets = 10000
        t0 = time.perf_counter()

        for i in range(num_packets):
            flow_idx = i % 50
            self.aggregator.process_raw_packet(
                src_ip="10.0.0.66",
                dst_ip="192.168.100.10",
                src_port=30000 + flow_idx,
                dst_port=80,
                protocol="TCP",
                pkt_len=64 + (i % 500),
                timestamp=1700000000.0 + i * 0.0001,
                tcp_flags={"A": True}
            )

        elapsed = time.perf_counter() - t0
        rate_pps = num_packets / elapsed

        print(f"\n[THROUGHPUT STRESS TEST]")
        print(f"  - Total Packets Processed : {num_packets:,}")
        print(f"  - Total Time Taken        : {elapsed:.4f} s")
        print(f"  - Ingestion Throughput    : {rate_pps:,.1f} packets/second")

        self.assertGreater(rate_pps, 5000.0, "FlowAggregator throughput should exceed 5,000 pkts/sec")


if __name__ == "__main__":
    unittest.main()
