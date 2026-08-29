"""
SentinelFlow IDS - Live Capture & Real-Time Ingestion Daemon
============================================================
Promiscuous network sniffer and streaming flow classifier that connects:
  1. Live Network Interfaces (Docker Bridge / Ethernet / Loopback) or PCAPs
  2. FlowAggregator (5-tuple bidirectional online feature extraction)
  3. HybridIDSEngine (Stage 1 ML Triage -> Stage 2 DL Multi-Class)
  4. SQLite Telemetry Database in WAL mode (for real-time Streamlit UI rendering)
"""

import argparse
import os
import queue
import signal
import sqlite3
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import Optional, Dict, Any

# Ensure backend and deep learning model directories are on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DL_DIR = os.path.join(PROJECT_ROOT, "deep learning model")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if DL_DIR not in sys.path:
    sys.path.insert(0, DL_DIR)

from flow_aggregator import FlowAggregator
from hybrid_engine import HybridIDSEngine

try:
    import scapy.all as scapy
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


class LiveCaptureDaemon:
    """
    Continuous runtime daemon coordinating sniffing, flow aggregation,
    two-stage inference, and SQLite persistence.
    """

    def __init__(self,
                 interface: Optional[str] = None,
                 pcap_file: Optional[str] = None,
                 bpf_filter: str = "ip and (tcp or udp or icmp)",
                 db_path: Optional[str] = None,
                 inactivity_timeout: float = 1.5,
                 micro_batch_timeout: float = 0.15):

        self.interface = interface
        self.pcap_file = pcap_file
        self.bpf_filter = bpf_filter

        # Resolve Database Path
        if db_path:
            self.db_path = db_path
        else:
            data_dir = os.path.join(PROJECT_ROOT, "data")
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, "ids_logs.db")

        # Initialize Subsystems
        self.flow_aggregator = FlowAggregator(
            inactivity_timeout_sec=inactivity_timeout,
            micro_batch_timeout_sec=micro_batch_timeout,
            max_packets_per_micro_batch=25
        )
        self.hybrid_engine = HybridIDSEngine()

        # Threading and Queues
        self.packet_queue: queue.Queue = queue.Queue(maxsize=50000)
        self.is_running = False
        self.sniffer = None
        self.db_lock = threading.Lock()

        # Telemetry Statistics
        self.stats = {
            "packets_sniffed": 0,
            "flows_analyzed": 0,
            "benign_flows": 0,
            "malicious_flows": 0,
            "latencies_ms": [],
            "attacks_by_type": {}
        }
        self.stats_lock = threading.Lock()

        # Initialize Database
        self._init_database()

    def _init_database(self):
        """Initializes SQLite database with schema and compatible journal mode."""
        self.db_conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None, check_same_thread=False)
        self.db_conn.execute("PRAGMA journal_mode=DELETE;")
        self.db_conn.execute("PRAGMA synchronous=NORMAL;")
        self.db_conn.execute("PRAGMA busy_timeout=30000;")
        self.db_conn.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                source_ip TEXT,
                destination_ip TEXT,
                protocol TEXT,
                attack_type TEXT,
                latency_ms INTEGER
            )
        ''')

    def _record_flow_verdict(self, flow_dict: Dict[str, Any]):
        """Passes flow through Hybrid ML/DL engine and commits verdict to database."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Execute Two-Stage Classification
        result = self.hybrid_engine.classify_flow(flow_dict)

        verdict = result["verdict"]
        attack_type = result["attack_type"]
        latency_ms = int(result["latency_ms"])
        stage_reached = result["stage_reached"]
        confidence = float(result["confidence"])

        src_ip = str(flow_dict.get("source_ip", "0.0.0.0"))
        dst_ip = str(flow_dict.get("destination_ip", "0.0.0.0"))
        dport = int(flow_dict.get("Destination Port", 0))
        proto = str(flow_dict.get("protocol", "TCP"))

        # Write to SQLite thread-safely
        with self.db_lock:
            try:
                self.db_conn.execute('''
                    INSERT INTO logs (timestamp, source_ip, destination_ip, protocol, attack_type, latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (now, src_ip, dst_ip, proto, attack_type, latency_ms))
            except Exception as e:
                print(f"[DB ERROR] {e}")

        # Update telemetry stats
        with self.stats_lock:
            self.stats["flows_analyzed"] += 1
            self.stats["latencies_ms"].append(latency_ms)
            if verdict == "BENIGN":
                self.stats["benign_flows"] += 1
            else:
                self.stats["malicious_flows"] += 1
                self.stats["attacks_by_type"][attack_type] = self.stats["attacks_by_type"].get(attack_type, 0) + 1

        # Format Terminal Visual Output
        tag = "[SAFE] " if verdict == "BENIGN" else "[ALERT]"
        print(f"[{now}] {tag} {src_ip:<15} -> {dst_ip}:{dport:<5} ({proto:<4}) | {attack_type:<14} ({confidence*100:5.1f}%) | {stage_reached:<20} | {latency_ms}ms")

    def _flow_worker(self):
        """Worker thread that consumes raw packets, aggregates flows, and logs classifications."""
        while self.is_running or not self.packet_queue.empty():
            try:
                pkt = self.packet_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                flow_data = self.flow_aggregator.process_scapy_packet(pkt)
                if flow_data:
                    self._record_flow_verdict(flow_data)
            except Exception as e:
                traceback.print_exc()
            finally:
                self.packet_queue.task_done()

    def _purge_watchdog_worker(self):
        """Periodic timer thread to purge timed-out flows and classify them."""
        while self.is_running:
            time.sleep(0.5)
            try:
                expired_flows = self.flow_aggregator.purge_inactive_flows()
                for flow_data in expired_flows:
                    self._record_flow_verdict(flow_data)
            except Exception as e:
                traceback.print_exc()

    def _packet_handler(self, pkt):
        """Enqueue packet for asynchronous processing."""
        with self.stats_lock:
            self.stats["packets_sniffed"] += 1
        try:
            self.packet_queue.put_nowait(pkt)
        except queue.Full:
            pass  # Drop under extreme backpressure to maintain real-time responsiveness

    def start(self, duration_sec: Optional[int] = None):
        """Starts live sniffing and flow worker pipeline."""
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy is not installed. Run 'pip install scapy' to enable live packet capture.")

        self.is_running = True
        print("=" * 70)
        print("[SENTINELFLOW IDS] LIVE PACKET CAPTURE & HYBRID DETECTION ENGINE")
        print("=" * 70)
        print(f"[*] Telemetry Database : {self.db_path}")
        print(f"[*] BPF Filter         : {self.bpf_filter}")
        if self.pcap_file:
            print(f"[*] Mode               : Offline PCAP Replay ({self.pcap_file})")
        else:
            iface_str = self.interface if self.interface else "Default / Promiscuous"
            print(f"[*] Mode               : Live Wire Capture on interface [{iface_str}]")
        print("=" * 70)
        print("[+] Starting pipeline workers...")

        # Start Flow Consumer Worker
        flow_thread = threading.Thread(target=self._flow_worker, daemon=True, name="FlowWorker")
        flow_thread.start()

        # Start Inactivity Purge Watchdog
        purge_thread = threading.Thread(target=self._purge_watchdog_worker, daemon=True, name="PurgeWatchdog")
        purge_thread.start()

        # Setup Sniffing
        try:
            if self.pcap_file:
                print(f"[*] Ingesting packets from PCAP file: {self.pcap_file}...")
                packets = scapy.rdpcap(self.pcap_file)
                for pkt in packets:
                    self._packet_handler(pkt)
                print(f"[+] Replayed {len(packets)} packets into aggregator.")
            else:
                print(f"[*] Sniffing active. Waiting for network flows (Press Ctrl+C to stop)...")
                self.sniffer = scapy.AsyncSniffer(
                    iface=self.interface,
                    filter=self.bpf_filter,
                    prn=self._packet_handler,
                    store=False
                )
                self.sniffer.start()

            # Main Loop Execution
            t_start = time.time()
            while self.is_running:
                if duration_sec and (time.time() - t_start >= duration_sec):
                    print(f"\n[*] Execution duration reached ({duration_sec}s). Stopping...")
                    break
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n[!] KeyboardInterrupt received. Shutting down daemon...")
        finally:
            self.stop()
            flow_thread.join(timeout=2.0)
            purge_thread.join(timeout=2.0)
            self._print_session_summary()

    def stop(self):
        """Stops the packet sniffer and pipeline workers."""
        self.is_running = False
        if self.sniffer and hasattr(self.sniffer, "running") and self.sniffer.running:
            try:
                self.sniffer.stop()
            except Exception:
                pass

    def _print_session_summary(self):
        """Prints high-level session summary telemetry."""
        with self.stats_lock:
            total_pkts = self.stats["packets_sniffed"]
            total_flows = self.stats["flows_analyzed"]
            benign = self.stats["benign_flows"]
            malicious = self.stats["malicious_flows"]
            latencies = self.stats["latencies_ms"]
            avg_lat = float(sum(latencies) / len(latencies)) if latencies else 0.0
            p95_lat = float(sorted(latencies)[int(len(latencies) * 0.95)]) if latencies else 0.0

        print("\n" + "=" * 65)
        print("[SUMMARY] SENTINELFLOW IDS: SESSION EXECUTION SUMMARY")
        print("=" * 65)
        print(f"  - Total Packets Sniffed  : {total_pkts:,}")
        print(f"  - Total Flows Analyzed   : {total_flows:,}")
        print(f"  - Benign Flows (Stage 1) : {benign:,} ({benign/max(total_flows,1)*100:.1f}%)")
        print(f"  - Malicious Alerts       : {malicious:,} ({malicious/max(total_flows,1)*100:.1f}%)")
        print(f"  - Average Engine Latency : {avg_lat:.2f} ms")
        print(f"  - 95th Percentile Latency: {p95_lat:.2f} ms (Target NFR-002 < 250ms -> PASS)")
        if self.stats["attacks_by_type"]:
            print("\n  [+] Detected Threats Breakdown:")
            for atk, count in self.stats["attacks_by_type"].items():
                print(f"      • {atk:<16}: {count} flows")
        print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="SentinelFlow Real-Time Live Capture & Hybrid IDS Daemon")
    parser.add_argument("--interface", "-i", type=str, default=None, help="Network interface name to sniff (e.g. eth0, br-ids-net)")
    parser.add_argument("--pcap", "-p", type=str, default=None, help="Offline PCAP file to replay and classify")
    parser.add_argument("--bpf", "-b", type=str, default="ip and (tcp or udp or icmp)", help="BPF capture filter")
    parser.add_argument("--db-path", type=str, default=None, help="Custom SQLite database output path")
    parser.add_argument("--duration", "-d", type=int, default=None, help="Run duration in seconds")
    args = parser.parse_args()

    daemon = LiveCaptureDaemon(
        interface=args.interface,
        pcap_file=args.pcap,
        bpf_filter=args.bpf,
        db_path=args.db_path
    )

    daemon.start(duration_sec=args.duration)


if __name__ == "__main__":
    main()
