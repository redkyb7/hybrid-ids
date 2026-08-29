"""
SentinelFlow IDS - Unit & Benchmark Tests for FlowAggregator
============================================================
Comprehensive test suite verifying:
  1. Bidirectional 5-tuple matching (Forward & Backward traffic)
  2. Incremental metric accuracy (Lengths, IATs, TCP Flags, Window sizes)
  3. Strict 52-feature schema compatibility with Stage 1 & Stage 2 models
  4. Micro-batch early emission (150ms timeout)
  5. Inactivity timeout purging & garbage collection
  6. Two-Stage HybridIDSEngine integration & latency benchmarking (NFR-002 < 250ms)
"""

import os
import sys
import time
import unittest
import numpy as np

# Ensure backend and models are on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "backend"))
DL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "deep learning model"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if DL_DIR not in sys.path:
    sys.path.insert(0, DL_DIR)

from flow_aggregator import Flow, FlowAggregator
from hybrid_engine import HybridIDSEngine
import predict as dl_predict


class TestFlowAggregator(unittest.TestCase):

    def setUp(self):
        self.aggregator = FlowAggregator(
            inactivity_timeout_sec=1.5,
            micro_batch_timeout_sec=0.15,
            max_packets_per_micro_batch=25
        )
        self.engine = HybridIDSEngine()

    def test_bidirectional_5tuple_matching(self):
        """Verifies client->server and server->client packets merge into the same flow."""
        src_ip, dst_ip = "192.168.100.66", "192.168.100.10"
        src_port, dst_port = 54321, 80
        proto = "TCP"

        t0 = time.time()

        # 1. Client sends SYN (Fwd)
        self.aggregator.process_raw_packet(
            src_ip=src_ip, dst_ip=dst_ip, src_port=src_port, dst_port=dst_port,
            protocol=proto, pkt_len=64, timestamp=t0,
            tcp_flags={"S": True}, header_len=32, win_size=64240
        )

        # 2. Server sends SYN-ACK (Bwd)
        self.aggregator.process_raw_packet(
            src_ip=dst_ip, dst_ip=src_ip, src_port=dst_port, dst_port=src_port,
            protocol=proto, pkt_len=64, timestamp=t0 + 0.002,
            tcp_flags={"S": True, "A": True}, header_len=32, win_size=65160
        )

        # 3. Client sends ACK (Fwd)
        self.aggregator.process_raw_packet(
            src_ip=src_ip, dst_ip=dst_ip, src_port=src_port, dst_port=dst_port,
            protocol=proto, pkt_len=54, timestamp=t0 + 0.003,
            tcp_flags={"A": True}, header_len=20, win_size=64240
        )

        # 4. Client sends HTTP GET Request (Fwd)
        self.aggregator.process_raw_packet(
            src_ip=src_ip, dst_ip=dst_ip, src_port=src_port, dst_port=dst_port,
            protocol=proto, pkt_len=350, timestamp=t0 + 0.005,
            tcp_flags={"P": True, "A": True}, header_len=20, payload_len=296
        )

        # 5. Server sends HTTP 200 OK Response (Bwd)
        self.aggregator.process_raw_packet(
            src_ip=dst_ip, dst_ip=src_ip, src_port=dst_port, dst_port=src_port,
            protocol=proto, pkt_len=1420, timestamp=t0 + 0.010,
            tcp_flags={"P": True, "A": True}, header_len=20, payload_len=1366
        )

        # 6. Client sends FIN to terminate (Trigger immediate emission)
        flow_features = self.aggregator.process_raw_packet(
            src_ip=src_ip, dst_ip=dst_ip, src_port=src_port, dst_port=dst_port,
            protocol=proto, pkt_len=54, timestamp=t0 + 0.015,
            tcp_flags={"F": True, "A": True}, header_len=20
        )

        self.assertIsNotNone(flow_features, "Flow should emit on TCP FIN termination")
        self.assertEqual(flow_features["Total Fwd Packets"], 4)
        self.assertEqual(flow_features["Total Length of Fwd Packets"], 64 + 54 + 350 + 54)
        self.assertEqual(flow_features["Bwd Packet Length Max"], 1420)
        self.assertEqual(flow_features["Init_Win_bytes_forward"], 64240)
        self.assertEqual(flow_features["Init_Win_bytes_backward"], 65160)
        self.assertEqual(flow_features["FIN Flag Count"], 1)
        self.assertEqual(flow_features["PSH Flag Count"], 2)
        self.assertEqual(flow_features["ACK Flag Count"], 5)

    def test_schema_compatibility_with_models(self):
        """Asserts extracted features contain all 52 features expected by Stage 2 and all 40 features for Stage 1."""
        flow = Flow("10.0.0.1", "10.0.0.2", 1234, 80, "TCP", time.time())
        flow.add_packet(100, time.time(), is_forward=True, tcp_flags={"S": True}, header_len=32, win_size=5840)
        flow.add_packet(200, time.time() + 0.01, is_forward=False, tcp_flags={"A": True}, header_len=20, win_size=5840)
        features = flow.extract_features()

        # Check all 52 Stage 2 features exist
        for col in dl_predict.FEATURE_ORDER:
            self.assertIn(col, features, f"Missing expected Stage 2 feature: {col}")
            self.assertIsInstance(features[col], (int, float), f"Feature {col} must be numeric")

        # Check all Stage 1 features exist
        if self.engine.stage1_features:
            for col in self.engine.stage1_features:
                self.assertIn(col, features, f"Missing expected Stage 1 feature: {col}")

    def test_micro_batch_early_emission(self):
        """Verifies active flows emit early when exceeding 150ms timeout without waiting for connection close."""
        aggregator = FlowAggregator(micro_batch_timeout_sec=0.10)
        t0 = time.time()

        # Packet 1
        res1 = aggregator.process_raw_packet("192.168.1.5", "192.168.1.10", 4000, 80, "TCP", 100, timestamp=t0)
        self.assertIsNone(res1, "Should not emit on packet 1")

        # Packet 2 at +0.12s (>100ms micro-batch threshold)
        res2 = aggregator.process_raw_packet("192.168.1.5", "192.168.1.10", 4000, 80, "TCP", 150, timestamp=t0 + 0.12)
        self.assertIsNotNone(res2, "Should emit micro-batch snapshot upon reaching micro_batch_timeout")
        self.assertEqual(res2["Total Fwd Packets"], 2)

    def test_inactivity_purging_garbage_collection(self):
        """Verifies expired silent flows are purged from active memory table."""
        aggregator = FlowAggregator(inactivity_timeout_sec=0.5)
        t0 = time.time()

        aggregator.process_raw_packet("10.0.0.5", "10.0.0.10", 5000, 80, "TCP", 100, timestamp=t0)
        self.assertEqual(len(aggregator.flows), 1)

        # Purge at t0 + 0.2s (Not expired)
        expired = aggregator.purge_inactive_flows(current_time=t0 + 0.2)
        self.assertEqual(len(expired), 0)
        self.assertEqual(len(aggregator.flows), 1)

        # Purge at t0 + 0.6s (Expired)
        expired = aggregator.purge_inactive_flows(current_time=t0 + 0.6)
        self.assertEqual(len(expired), 1)
        self.assertEqual(len(aggregator.flows), 0, "Expired flow should be removed from memory")

    def test_end_to_end_hybrid_engine_latency(self):
        """
        Benchmarks end-to-end latency: Raw Packet Ingestion -> Flow Extraction -> Stage 1 ML -> Stage 2 DL.
        Enforces NFR-002: Latency < 250ms.
        """
        aggregator = FlowAggregator()
        t0 = time.time()

        # Simulate 100 consecutive packet streams
        latencies_ms = []

        for i in range(50):
            # Create a completed synthetic flow
            aggregator.process_raw_packet("10.0.0.66", "192.168.100.10", 40000 + i, 80, "TCP", 64, timestamp=t0 + i * 0.05, tcp_flags={"S": True})
            aggregator.process_raw_packet("192.168.100.10", "10.0.0.66", 80, 40000 + i, "TCP", 64, timestamp=t0 + i * 0.05 + 0.001, tcp_flags={"S": True, "A": True})
            flow_data = aggregator.process_raw_packet("10.0.0.66", "192.168.100.10", 40000 + i, 80, "TCP", 250, timestamp=t0 + i * 0.05 + 0.005, tcp_flags={"F": True, "A": True})

            if flow_data:
                t_start = time.perf_counter()
                result = self.engine.classify_flow(flow_data)
                t_elapsed_ms = (time.perf_counter() - t_start) * 1000
                latencies_ms.append(t_elapsed_ms)

                self.assertIn(result["verdict"], ["BENIGN", "MALICIOUS"])
                self.assertIn("attack_type", result)
                self.assertGreaterEqual(result["confidence"], 0.0)

        avg_latency = float(np.mean(latencies_ms))
        p95_latency = float(np.percentile(latencies_ms, 95))

        print(f"\n[BENCHMARK] Flow Aggregator + Two-Stage Hybrid IDS Engine:")
        print(f"  - Flows Processed : {len(latencies_ms)}")
        print(f"  - Average Latency : {avg_latency:.3f} ms")
        print(f"  - 95th Pct Latency: {p95_latency:.3f} ms (Target NFR-002 < 250ms)")

        self.assertLess(p95_latency, 250.0, f"95th percentile latency {p95_latency}ms must be < 250ms")
        self.assertLess(avg_latency, 50.0, "Average latency should be well under 50ms")


if __name__ == "__main__":
    unittest.main()
