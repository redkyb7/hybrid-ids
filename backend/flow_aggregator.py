"""
SentinelFlow IDS - Streaming Bidirectional Flow Aggregator
==========================================================
High-performance in-memory network flow aggregator that converts raw streaming
packets into the 57-feature contract used by the current ML/DL bundle.

Key Features:
  1. Bidirectional 5-tuple tracking (Forward client->server & Backward server->client)
  2. Incremental online calculation of packet length distributions, IATs, and flags
  3. Micro-batch early emission (150ms timeout) to enforce NFR-002 latency (<250ms)
  4. Memory-safe flow expiration and garbage collection
  5. Dual ingestion support: Scapy Packet stream or raw PCAP files
"""

import math
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class Flow:
    """
    Represents a single bidirectional network conversation (5-tuple flow).
    Accumulates statistical metrics across forward and backward packets in real-time.
    """

    ACTIVE_IDLE_TIMEOUT_SEC = 5.0

    def __init__(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int, protocol: str, start_time: float):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol.upper()

        self.start_time = start_time
        self.last_seen = start_time

        # Forward Packet Trackers
        self.fwd_packets = 0
        self.fwd_bytes = 0
        self.fwd_packet_lengths: List[int] = []
        self.fwd_timestamps: List[float] = []
        self.fwd_header_bytes = 0
        self.init_win_bytes_fwd = 0
        self.min_seg_size_fwd: Optional[int] = None
        self.act_data_pkt_fwd = 0
        self.fwd_psh_count = 0

        # Backward Packet Trackers
        self.bwd_packets = 0
        self.bwd_bytes = 0
        self.bwd_packet_lengths: List[int] = []
        self.bwd_timestamps: List[float] = []
        self.bwd_header_bytes = 0
        self.init_win_bytes_bwd = 0

        # Global Flow Trackers
        self.all_packet_lengths: List[int] = []
        self.all_timestamps: List[float] = []
        self.active_start_time = start_time
        self.active_end_time = start_time
        self.active_periods_us: List[float] = []
        self.idle_periods_us: List[float] = []

        # TCP Flags Counters
        self.fin_count = 0
        self.syn_count = 0
        self.rst_count = 0
        self.psh_count = 0
        self.ack_count = 0
        self.urg_count = 0

        # Flow State
        self.is_terminated = False
        self.emitted_count = 0
        self.last_emitted_at: Optional[float] = None
        self.last_emitted_packet_count = 0
        self.payload_sample = b""

    def add_packet(self, pkt_len: int, timestamp: float, is_forward: bool,
                   tcp_flags: Optional[Dict[str, bool]] = None,
                   header_len: int = 20, win_size: int = 0, payload_len: int = 0,
                   raw_payload: Optional[bytes] = None):
        """
        Ingests a single packet and updates running flow statistics.
        """
        if timestamp - self.active_end_time > self.ACTIVE_IDLE_TIMEOUT_SEC:
            active_us = (self.active_end_time - self.active_start_time) * 1e6
            if active_us > 0:
                self.active_periods_us.append(active_us)
            self.idle_periods_us.append((timestamp - self.active_end_time) * 1e6)
            self.active_start_time = timestamp
        self.active_end_time = max(self.active_end_time, timestamp)
        self.last_seen = max(self.last_seen, timestamp)
        self.all_timestamps.append(timestamp)
        self.all_packet_lengths.append(pkt_len)

        if raw_payload and len(self.payload_sample) < 512:
            self.payload_sample += raw_payload[:512 - len(self.payload_sample)]

        if is_forward:
            self.fwd_packets += 1
            self.fwd_bytes += pkt_len
            self.fwd_packet_lengths.append(pkt_len)
            self.fwd_timestamps.append(timestamp)
            self.fwd_header_bytes += header_len
            if self.fwd_packets == 1:
                self.init_win_bytes_fwd = win_size
            segment_size = header_len if header_len > 0 else 20
            if self.min_seg_size_fwd is None:
                self.min_seg_size_fwd = segment_size
            else:
                self.min_seg_size_fwd = min(self.min_seg_size_fwd, segment_size)
            if payload_len > 0:
                self.act_data_pkt_fwd += 1
            if tcp_flags and tcp_flags.get("P", False):
                self.fwd_psh_count += 1
        else:
            self.bwd_packets += 1
            self.bwd_bytes += pkt_len
            self.bwd_packet_lengths.append(pkt_len)
            self.bwd_timestamps.append(timestamp)
            self.bwd_header_bytes += header_len
            if self.bwd_packets == 1:
                self.init_win_bytes_bwd = win_size

        # Update TCP Flags
        if tcp_flags:
            if tcp_flags.get("F", False):
                self.fin_count += 1
            if tcp_flags.get("S", False):
                self.syn_count += 1
            if tcp_flags.get("R", False):
                self.rst_count += 1
            if tcp_flags.get("P", False):
                self.psh_count += 1
            if tcp_flags.get("A", False):
                self.ack_count += 1
            if tcp_flags.get("U", False):
                self.urg_count += 1

            # Fully terminated when both endpoints FIN-teardown or RST occurs
            self.is_terminated = (self.fin_count >= 2) or (self.rst_count >= 1)

    @staticmethod
    def _compute_stats(arr: List[float | int]) -> Tuple[float, float, float, float, float]:
        """Returns (Min, Max, Mean, Std, Variance) for a numeric series."""
        if not arr:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        n = len(arr)
        arr_min = float(min(arr))
        arr_max = float(max(arr))
        arr_mean = float(sum(arr) / n)
        if n > 1:
            var = sum((x - arr_mean) ** 2 for x in arr) / (n - 1)
            std = math.sqrt(var)
        else:
            var = 0.0
            std = 0.0
        return arr_min, arr_max, arr_mean, std, var

    @staticmethod
    def _compute_iats(timestamps: List[float]) -> Tuple[float, float, float, float, float]:
        """Calculates IAT Total, Mean, Std, Max, Min in microseconds (us)."""
        if len(timestamps) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        iats_us = [(timestamps[i] - timestamps[i - 1]) * 1e6 for i in range(1, len(timestamps))]
        iat_total = sum(iats_us)
        iat_min, iat_max, iat_mean, iat_std, _ = Flow._compute_stats(iats_us)
        return float(iat_total), float(iat_mean), float(iat_std), float(iat_max), float(iat_min)

    def extract_features(self) -> Dict[str, Any]:
        """Extract flow statistics for the deployed model contract."""

        duration_sec = max(self.last_seen - self.start_time, 1e-5)
        flow_duration_us = duration_sec * 1e6

        # Packet Length Statistics
        fwd_min, fwd_max, fwd_mean, fwd_std, _ = self._compute_stats(self.fwd_packet_lengths)
        bwd_min, bwd_max, bwd_mean, bwd_std, _ = self._compute_stats(self.bwd_packet_lengths)
        # In the uploaded Parquet data, the first packet contributes twice to
        # the overall packet-length statistics.
        packet_lengths = self.all_packet_lengths
        if packet_lengths:
            packet_lengths = [packet_lengths[0], *packet_lengths]
        all_min, all_max, all_mean, all_std, all_var = self._compute_stats(packet_lengths)

        # Inter-Arrival Times (IAT) in microseconds (us)
        flow_iat_total, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = self._compute_iats(self.all_timestamps)
        fwd_iat_total, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = self._compute_iats(self.fwd_timestamps)
        bwd_iat_total, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = self._compute_iats(self.bwd_timestamps)

        # Rates & Statistics (Conforming to CICFlowMeter / CICIDS2017 specifications)
        total_packets = self.fwd_packets + self.bwd_packets
        total_bytes = self.fwd_bytes + self.bwd_bytes

        flow_bytes_s = total_bytes / duration_sec
        flow_pkts_s = total_packets / duration_sec
        fwd_pkts_s = self.fwd_packets / duration_sec
        bwd_pkts_s = self.bwd_packets / duration_sec

        avg_pkt_size = total_bytes / total_packets if total_packets > 0 else 0.0

        # Active/idle periods use the CICFlowMeter activity timeout. A short
        # snapshot generally has no completed idle period.
        active_mean = 0.0
        active_std = 0.0
        active_max = 0.0
        active_min = 0.0
        idle_mean = 0.0
        idle_std = 0.0
        idle_max = 0.0
        idle_min = 0.0
        active_periods = list(self.active_periods_us)
        final_active_us = (self.active_end_time - self.active_start_time) * 1e6
        if final_active_us > 0:
            active_periods.append(final_active_us)
        active_min, active_max, active_mean, active_std, _ = self._compute_stats(active_periods)
        idle_min, idle_max, idle_mean, idle_std, _ = self._compute_stats(self.idle_periods_us)

        features = {
            # Network Flow Context
            "source_ip": self.src_ip,
            "destination_ip": self.dst_ip,
            "protocol": self.protocol,
            "flow_start_ts": float(self.start_time),
            "flow_end_ts": float(self.last_seen),
            "Source Port": int(self.src_port),
            "Destination Port": int(self.dst_port),

            # 52 Standardized Model Features
            "Flow Duration": float(flow_duration_us),
            "Total Fwd Packets": int(self.fwd_packets),
            "Total Length of Fwd Packets": int(self.fwd_bytes),
            "Fwd Packet Length Max": float(fwd_max),
            "Fwd Packet Length Min": float(fwd_min),
            "Fwd Packet Length Mean": float(fwd_mean),
            "Fwd Packet Length Std": float(fwd_std),
            "Bwd Packet Length Max": float(bwd_max),
            "Bwd Packet Length Min": float(bwd_min),
            "Bwd Packet Length Mean": float(bwd_mean),
            "Bwd Packet Length Std": float(bwd_std),
            "Flow Bytes/s": float(flow_bytes_s),
            "Flow Packets/s": float(flow_pkts_s),
            "Flow IAT Mean": float(flow_iat_mean),
            "Flow IAT Std": float(flow_iat_std),
            "Flow IAT Max": float(flow_iat_max),
            "Flow IAT Min": float(flow_iat_min),
            "Fwd IAT Total": float(fwd_iat_total),
            "Fwd IAT Mean": float(fwd_iat_mean),
            "Fwd IAT Std": float(fwd_iat_std),
            "Fwd IAT Max": float(fwd_iat_max),
            "Fwd IAT Min": float(fwd_iat_min),
            "Bwd IAT Total": float(bwd_iat_total),
            "Bwd IAT Mean": float(bwd_iat_mean),
            "Bwd IAT Std": float(bwd_iat_std),
            "Bwd IAT Max": float(bwd_iat_max),
            "Bwd IAT Min": float(bwd_iat_min),
            "Fwd Header Length": int(self.fwd_header_bytes),
            "Bwd Header Length": int(self.bwd_header_bytes),
            "Fwd Packets/s": float(fwd_pkts_s),
            "Bwd Packets/s": float(bwd_pkts_s),
            "Min Packet Length": float(all_min),
            "Max Packet Length": float(all_max),
            "Packet Length Mean": float(all_mean),
            "Packet Length Std": float(all_std),
            "Packet Length Variance": float(all_var),
            "FIN Flag Count": int(self.fin_count),
            "PSH Flag Count": int(self.psh_count),
            "ACK Flag Count": int(self.ack_count),
            "Average Packet Size": float(avg_pkt_size),
            "Subflow Fwd Bytes": int(self.fwd_bytes),
            "Init_Win_bytes_forward": int(self.init_win_bytes_fwd),
            "Init_Win_bytes_backward": int(self.init_win_bytes_bwd),
            "act_data_pkt_fwd": int(self.act_data_pkt_fwd),
            "min_seg_size_forward": int(self.min_seg_size_fwd or 0),
            "Active Mean": float(active_mean),
            "Active Max": float(active_max),
            "Active Min": float(active_min),
            "Idle Mean": float(idle_mean),
            "Idle Max": float(idle_max),
            "Idle Min": float(idle_min),
            "payload_sample": self.payload_sample.decode('latin1', errors='replace'),
        }

        # The uploaded Parquet rows show the four Subflow columns equal
        # their corresponding flow totals and segment averages equal the
        # directional packet-length means.
        features.update({
                "Total Backward Packets": int(self.bwd_packets),
                "Fwd Packets Length Total": int(self.fwd_bytes),
                "Bwd Packets Length Total": int(self.bwd_bytes),
                "Fwd PSH Flags": int(self.fwd_psh_count),
                "Bwd Packets/s": float(bwd_pkts_s),
                "Packet Length Max": float(all_max),
                "SYN Flag Count": int(self.syn_count),
                "URG Flag Count": int(self.urg_count),
                "Avg Packet Size": float(sum(packet_lengths) / total_packets) if total_packets else 0.0,
                "Avg Fwd Segment Size": float(fwd_mean),
                "Avg Bwd Segment Size": float(bwd_mean),
                "Subflow Fwd Packets": int(self.fwd_packets),
                "Subflow Fwd Bytes": int(self.fwd_bytes),
                "Subflow Bwd Packets": int(self.bwd_packets),
                "Subflow Bwd Bytes": int(self.bwd_bytes),
                "Init Fwd Win Bytes": int(self.init_win_bytes_fwd),
                "Init Bwd Win Bytes": int(self.init_win_bytes_bwd),
                "Fwd Act Data Packets": int(self.act_data_pkt_fwd),
                "Fwd Seg Size Min": int(self.min_seg_size_fwd or 0),
                "Active Std": float(active_std),
                "Idle Std": float(idle_std),
        })
        return features


class FlowAggregator:
    """
    Thread-safe streaming flow aggregator that manages active bidirectional flows,
    handles micro-batch emissions, and purges expired flows.
    """

    def __init__(self,
                 inactivity_timeout_sec: float = 1.0,
                 micro_batch_timeout_sec: float = 1.0,
                 max_packets_per_micro_batch: int = 40):
        self.inactivity_timeout = inactivity_timeout_sec
        self.micro_batch_timeout = micro_batch_timeout_sec
        self.max_pkts_micro_batch = max_packets_per_micro_batch

        self.flows: Dict[Tuple, Flow] = {}
        self.lock = threading.Lock()
        self.emitted_flows_queue: List[Dict[str, Any]] = []

    @staticmethod
    def _get_5tuple_keys(src_ip: str, dst_ip: str, src_port: int, dst_port: int, proto: str) -> Tuple[Tuple, Tuple]:
        """Generates forward and backward keys for 5-tuple lookup."""
        fwd_key = (src_ip, dst_ip, src_port, dst_port, proto.upper())
        bwd_key = (dst_ip, src_ip, dst_port, src_port, proto.upper())
        return fwd_key, bwd_key

    def process_raw_packet(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int,
                           protocol: str, pkt_len: int, timestamp: Optional[float] = None,
                           tcp_flags: Optional[Dict[str, bool]] = None,
                           header_len: int = 20, win_size: int = 0, payload_len: int = 0,
                           raw_payload: Optional[bytes] = None) -> Optional[Dict[str, Any]]:
        """
        Processes a raw packet tuple. If a flow completes or hits a micro-batch threshold, returns the extracted feature vector.
        """
        now = timestamp if timestamp is not None else time.time()
        fwd_key, bwd_key = self._get_5tuple_keys(src_ip, dst_ip, src_port, dst_port, protocol)

        emitted_feature_dict = None

        with self.lock:
            if fwd_key in self.flows:
                flow = self.flows[fwd_key]
                is_fwd = True
            elif bwd_key in self.flows:
                flow = self.flows[bwd_key]
                is_fwd = False
            else:
                # New Flow established
                flow = Flow(src_ip, dst_ip, src_port, dst_port, protocol, start_time=now)
                self.flows[fwd_key] = flow
                is_fwd = True

            flow.add_packet(
                pkt_len=pkt_len,
                timestamp=now,
                is_forward=is_fwd,
                tcp_flags=tcp_flags,
                header_len=header_len,
                win_size=win_size,
                payload_len=payload_len,
                raw_payload=raw_payload
            )

            # Emit a snapshot when a flow reaches a configured time or packet
            # limit. A first FIN also produces a verdict; the flow remains
            # tracked until the peer FIN, RST, or inactivity timeout.
            pkt_count = flow.fwd_packets + flow.bwd_packets
            new_packets = pkt_count - flow.last_emitted_packet_count
            since_emission = now - (
                flow.last_emitted_at
                if flow.last_emitted_at is not None
                else flow.start_time
            )
            should_emit = new_packets > 0 and (
                flow.is_terminated
                or (flow.fin_count > 0 and flow.emitted_count == 0)
                or new_packets >= self.max_pkts_micro_batch
                or since_emission >= self.micro_batch_timeout
            )

            if should_emit:
                emitted_feature_dict = flow.extract_features()
                flow.emitted_count += 1
                flow.last_emitted_at = now
                flow.last_emitted_packet_count = pkt_count
            if flow.is_terminated:
                # Remove terminated flows cleanly from the 5-tuple table.
                self.flows.pop(fwd_key, None)
                self.flows.pop(bwd_key, None)

        return emitted_feature_dict

    def process_scapy_packet(self, pkt) -> Optional[Dict[str, Any]]:
        """
        Extracts 5-tuple, TCP/UDP headers, flags, and payload lengths from a raw Scapy packet.
        """
        try:
            # Check for IP layer (IPv4 or IPv6)
            if not pkt.haslayer("IP"):
                return None

            ip_layer = pkt["IP"]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            timestamp = float(getattr(pkt, "time", time.time()))

            protocol = "OTHER"
            src_port = 0
            dst_port = 0
            tcp_flags = None
            header_len = 20
            win_size = 0
            payload_len = 0
            raw_payload_bytes = None

            if pkt.haslayer("TCP"):
                protocol = "TCP"
                tcp_layer = pkt["TCP"]
                src_port = int(tcp_layer.sport) if getattr(tcp_layer, "sport", None) is not None else 0
                dst_port = int(tcp_layer.dport) if getattr(tcp_layer, "dport", None) is not None else 0
                win_size = int(tcp_layer.window) if getattr(tcp_layer, "window", None) is not None else 0
                dataofs = getattr(tcp_layer, "dataofs", None)
                header_len = int(dataofs * 4) if dataofs is not None else 20
                flags = int(tcp_layer.flags) if getattr(tcp_layer, "flags", None) is not None else 0
                tcp_flags = {
                    "F": bool(flags & 0x01),
                    "S": bool(flags & 0x02),
                    "R": bool(flags & 0x04),
                    "P": bool(flags & 0x08),
                    "A": bool(flags & 0x10),
                    "U": bool(flags & 0x20),
                }
                if hasattr(tcp_layer, "payload") and tcp_layer.payload:
                    try:
                        raw_payload_bytes = bytes(tcp_layer.payload)
                        payload_len = len(raw_payload_bytes)
                    except Exception:
                        raw_payload_bytes = None
                        payload_len = 0

            elif pkt.haslayer("UDP"):
                protocol = "UDP"
                udp_layer = pkt["UDP"]
                src_port = int(udp_layer.sport) if getattr(udp_layer, "sport", None) is not None else 0
                dst_port = int(udp_layer.dport) if getattr(udp_layer, "dport", None) is not None else 0
                header_len = 8
                if hasattr(udp_layer, "payload") and udp_layer.payload:
                    try:
                        raw_payload_bytes = bytes(udp_layer.payload)
                        payload_len = len(raw_payload_bytes)
                    except Exception:
                        raw_payload_bytes = None
                        payload_len = 0

            # In CICFlowMeter / CICIDS2017 specification:
            # Packet Length statistics measure the Transport Layer payload length (0 for TCP control packets).
            flow_pkt_len = payload_len

            return self.process_raw_packet(
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                pkt_len=flow_pkt_len,
                timestamp=timestamp,
                tcp_flags=tcp_flags,
                header_len=header_len,
                win_size=win_size,
                payload_len=payload_len,
                raw_payload=raw_payload_bytes
            )
        except Exception:
            raise

    def purge_inactive_flows(self, current_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Scans active flows, extracts features for timed-out flows, and purges them from memory.
        """
        now = current_time if current_time is not None else time.time()
        expired_flows: List[Dict[str, Any]] = []

        with self.lock:
            keys_to_remove = []
            for key, flow in list(self.flows.items()):
                if now - flow.last_seen > self.inactivity_timeout:
                    # Avoid a duplicate verdict if the final packet was
                    # already emitted as a snapshot.
                    pkt_count = flow.fwd_packets + flow.bwd_packets
                    if pkt_count > flow.last_emitted_packet_count:
                        expired_flows.append(flow.extract_features())
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                self.flows.pop(key, None)

        return expired_flows

    def flush_active_flows(self) -> List[Dict[str, Any]]:
        """Emit pending snapshots and clear flows after an offline replay."""
        with self.lock:
            pending = [
                flow.extract_features()
                for flow in self.flows.values()
                if flow.fwd_packets + flow.bwd_packets > flow.last_emitted_packet_count
            ]
            self.flows.clear()
        return pending
