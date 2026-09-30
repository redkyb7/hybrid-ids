"""Verify that TCP teardown ACKs do not become new IDS flow snapshots."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from flow_aggregator import FlowAggregator  # noqa: E402


CLIENT = ("127.0.0.1", "127.0.0.1", 55321, 8080)
SERVER = ("127.0.0.1", "127.0.0.1", 8080, 55321)


def packet(aggregator: FlowAggregator, direction: tuple, timestamp: float,
           flags: dict, payload: int = 0):
    source_ip, destination_ip, source_port, destination_port = direction
    return aggregator.process_raw_packet(
        source_ip, destination_ip, source_port, destination_port,
        "TCP", payload, timestamp=timestamp, tcp_flags=flags,
        header_len=20, payload_len=payload,
    )


def closed_flow(aggregator: FlowAggregator) -> None:
    packet(aggregator, CLIENT, 100.000, {"S": True})
    packet(aggregator, SERVER, 100.001, {"S": True, "A": True})
    packet(aggregator, CLIENT, 100.002, {"A": True})
    packet(aggregator, CLIENT, 100.003, {"P": True, "A": True}, 88)
    first_fin = packet(aggregator, CLIENT, 100.004, {"F": True, "A": True})
    second_fin = packet(aggregator, SERVER, 100.005, {"F": True, "A": True})
    assert first_fin is not None and second_fin is not None


class FlowTeardownAckTests(unittest.TestCase):
    def test_repeated_fin_from_one_endpoint_keeps_flow_open(self):
        aggregator = FlowAggregator(micro_batch_timeout_sec=10)
        packet(aggregator, CLIENT, 100.000, {"S": True})
        packet(aggregator, CLIENT, 100.001, {"F": True, "A": True})
        packet(aggregator, CLIENT, 100.002, {"F": True, "A": True})
        self.assertEqual(len(aggregator.flows), 1)
        self.assertEqual(aggregator.recently_closed_ack, {})
        packet(aggregator, SERVER, 100.003, {"P": True, "A": True}, 95)
        final = packet(aggregator, SERVER, 100.004, {"F": True, "A": True})
        self.assertIsNotNone(final)
        self.assertEqual(final["FIN Flag Count"], 3)
        self.assertEqual(final["Total Backward Packets"], 2)
        self.assertEqual(aggregator.flows, {})

    def test_final_ack_is_suppressed_after_two_fins(self):
        aggregator = FlowAggregator(micro_batch_timeout_sec=10)
        closed_flow(aggregator)
        self.assertIsNone(packet(aggregator, CLIENT, 100.006, {"A": True}))
        self.assertEqual(aggregator.flows, {})
        self.assertEqual(aggregator.flush_active_flows(), [])

    def test_server_fin_retransmission_waits_for_client_fin(self):
        aggregator = FlowAggregator(micro_batch_timeout_sec=10)
        packet(aggregator, CLIENT, 100.000, {"S": True})
        packet(aggregator, SERVER, 100.001, {"F": True, "A": True})
        packet(aggregator, SERVER, 100.002, {"F": True, "A": True})
        self.assertEqual(len(aggregator.flows), 1)
        final = packet(aggregator, CLIENT, 100.003, {"F": True, "A": True})
        self.assertEqual(final["FIN Flag Count"], 3)
        self.assertIsNone(packet(aggregator, SERVER, 100.004, {"A": True}))
        self.assertEqual(aggregator.flows, {})

    def test_new_syn_with_same_tuple_starts_new_flow(self):
        aggregator = FlowAggregator(micro_batch_timeout_sec=10)
        closed_flow(aggregator)
        packet(aggregator, CLIENT, 100.100, {"S": True})
        remaining = aggregator.flush_active_flows()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["SYN Flag Count"], 1)

    def test_ack_after_grace_is_not_suppressed(self):
        aggregator = FlowAggregator(micro_batch_timeout_sec=10)
        closed_flow(aggregator)
        packet(aggregator, CLIENT, 102.000, {"A": True})
        remaining = aggregator.flush_active_flows()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["Total Fwd Packets"], 1)

    def test_rst_does_not_suppress_later_ack(self):
        aggregator = FlowAggregator(micro_batch_timeout_sec=10)
        packet(aggregator, CLIENT, 100.000, {"S": True})
        packet(aggregator, SERVER, 100.001, {"R": True, "A": True})
        packet(aggregator, CLIENT, 100.002, {"A": True})
        remaining = aggregator.flush_active_flows()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["Total Fwd Packets"], 1)


if __name__ == "__main__":
    unittest.main()
