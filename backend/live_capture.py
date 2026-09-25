"""
SentinelFlow IDS - Live Capture & Real-Time Ingestion Daemon
============================================================
Promiscuous network sniffer and streaming flow classifier that connects:
  1. Live Network Interfaces (Docker Bridge / Ethernet / Loopback) or PCAPs
  2. FlowAggregator (5-tuple bidirectional online feature extraction)
  3. HybridIDSEngine (Stage 1 ML Triage -> Stage 2 DL Multi-Class)
  4. SQLite Telemetry Database (for real-time FastAPI dashboard rendering)
"""

import argparse
import json
import math
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

# Ensure backend modules are on sys.path when launched as a script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

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
                 micro_batch_timeout: float = 0.15,
                 log_features: Optional[bool] = None):

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

        self.hybrid_engine = HybridIDSEngine()
        self.log_features = (
            os.environ.get("IDS_LOG_FEATURES", "0") == "1"
            if log_features is None else log_features
        )
        self.flow_aggregator = FlowAggregator(
            inactivity_timeout_sec=inactivity_timeout,
            micro_batch_timeout_sec=micro_batch_timeout,
            max_packets_per_micro_batch=25,
        )

        # Threading and Queues
        self.packet_queue: queue.Queue = queue.Queue(maxsize=50000)
        self.is_running = False
        self.sniffer = None
        self.db_lock = threading.Lock()

        # Telemetry Statistics
        self.stats = {
            "packets_sniffed": 0,
            "packets_dropped": 0,
            "flows_analyzed": 0,
            "benign_flows": 0,
            "malicious_flows": 0,
            "classification_errors": 0,
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
                latency_ms INTEGER,
                source_port INTEGER,
                destination_port INTEGER,
                confidence REAL,
                verdict TEXT,
                stage_reached TEXT,
                model_attack_type TEXT,
                model_verdict TEXT,
                model_confidence REAL,
                stage1_attack_probability REAL,
                stage1_features_json TEXT,
                flow_features_json TEXT,
                flow_start_epoch REAL,
                flow_end_epoch REAL
            )
        ''')
        columns = {row[1] for row in self.db_conn.execute("PRAGMA table_info(logs)")}
        for column, definition in {
            "source_port": "INTEGER",
            "destination_port": "INTEGER",
            "confidence": "REAL",
            "verdict": "TEXT",
            "stage_reached": "TEXT",
            "model_attack_type": "TEXT",
            "model_verdict": "TEXT",
            "model_confidence": "REAL",
            "stage1_attack_probability": "REAL",
            "stage1_features_json": "TEXT",
            "flow_features_json": "TEXT",
            "flow_start_epoch": "REAL",
            "flow_end_epoch": "REAL",
        }.items():
            if column not in columns:
                self.db_conn.execute(f"ALTER TABLE logs ADD COLUMN {column} {definition}")

    def _record_flow_verdict(self, flow_dict: Dict[str, Any]):
        """Passes flow through Hybrid ML/DL engine and commits verdict to database."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        started = time.perf_counter()

        result = self.hybrid_engine.classify_flow(flow_dict)
        verdict = result["verdict"]
        attack_type = result["attack_type"]
        confidence = float(result["confidence"])
        latency_ms = max(1.0, (time.perf_counter() - started) * 1000)
        stage_reached = result["stage_reached"]
        feature_json = None
        all_feature_json = None
        if self.log_features:
            feature_json = json.dumps({
                name: float(flow_dict[name])
                for name in self.hybrid_engine.stage1_features
            }, allow_nan=False, separators=(",", ":"))
            all_feature_json = json.dumps({
                name: float(flow_dict[name])
                for name in sorted(set(self.hybrid_engine.stage1_features)
                                   | set(self.hybrid_engine.stage2_features))
            }, allow_nan=False, separators=(",", ":"))

        src_ip = str(flow_dict.get("source_ip", "0.0.0.0"))
        dst_ip = str(flow_dict.get("destination_ip", "0.0.0.0"))
        sport = int(flow_dict.get("Source Port", 0))
        dport = int(flow_dict.get("Destination Port", 0))
        proto = str(flow_dict.get("protocol", "TCP"))
        flow_start_epoch = (
            float(flow_dict["flow_start_ts"]) if "flow_start_ts" in flow_dict else None
        )
        flow_end_epoch = (
            float(flow_dict["flow_end_ts"]) if "flow_end_ts" in flow_dict else None
        )

        # Write to SQLite thread-safely
        with self.db_lock:
            self.db_conn.execute('''
                INSERT INTO logs (
                    timestamp, source_ip, destination_ip, protocol, attack_type,
                    latency_ms, source_port, destination_port, confidence,
                    verdict, stage_reached, model_attack_type,
                    model_verdict, model_confidence,
                    stage1_attack_probability, stage1_features_json,
                    flow_features_json, flow_start_epoch, flow_end_epoch
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (now, src_ip, dst_ip, proto, attack_type, math.ceil(latency_ms), sport, dport,
                  confidence, verdict, stage_reached, attack_type, verdict, confidence,
                  float(result["stage1_attack_probability"]), feature_json,
                  all_feature_json, flow_start_epoch, flow_end_epoch))

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
        score = f"{confidence*100:5.1f}%"
        print(f"[{now}] {tag} {src_ip:<15} -> {dst_ip}:{dport:<5} ({proto:<4}) | {attack_type:<14} ({score}) | {stage_reached:<20} | model | {latency_ms:.2f}ms")

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
            except Exception:
                with self.stats_lock:
                    self.stats["classification_errors"] += 1
                traceback.print_exc()
            finally:
                self.packet_queue.task_done()

    def _purge_watchdog_worker(self):
        """Periodic timer thread to purge timed-out flows and classify them."""
        while self.is_running:
            time.sleep(0.5)
            try:
                expired_flows = self.flow_aggregator.purge_inactive_flows()
            except Exception:
                with self.stats_lock:
                    self.stats["classification_errors"] += 1
                traceback.print_exc()
                continue
            for flow_data in expired_flows:
                try:
                    self._record_flow_verdict(flow_data)
                except Exception:
                    with self.stats_lock:
                        self.stats["classification_errors"] += 1
                    traceback.print_exc()

    def _packet_handler(self, pkt):
        """Enqueue packet for asynchronous processing."""
        with self.stats_lock:
            self.stats["packets_sniffed"] += 1
        try:
            self.packet_queue.put_nowait(pkt)
        except queue.Full:
            with self.stats_lock:
                self.stats["packets_dropped"] += 1

    def start(self, duration_sec: Optional[int] = None):
        """Starts live sniffing and flow worker pipeline."""
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy is not installed. Run 'pip install scapy' to enable live packet capture.")

        self.is_running = True
        print("=" * 70)
        print("[SENTINELFLOW IDS] LIVE PACKET CAPTURE & HYBRID DETECTION ENGINE")
        print("=" * 70)
        print(f"[*] Telemetry Database : {self.db_path}")
        print(f"[*] Model Artifacts    : {self.hybrid_engine.artifact_root}")
        print(f"[*] Feature Logging   : {'enabled' if self.log_features else 'disabled'}")
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

        # Offline PCAP timestamps may be years old, so wall-clock expiration
        # cannot run during replay.
        purge_thread = None
        if not self.pcap_file:
            purge_thread = threading.Thread(target=self._purge_watchdog_worker,
                                            daemon=True, name="PurgeWatchdog")
            purge_thread.start()

        # Setup Sniffing
        try:
            if self.pcap_file:
                print(f"[*] Ingesting packets from PCAP file: {self.pcap_file}...")
                packets = scapy.rdpcap(self.pcap_file)
                for pkt in packets:
                    self._packet_handler(pkt)
                print(f"[+] Replayed {len(packets)} packets into aggregator.")
                self.packet_queue.join()
                for flow_data in self.flow_aggregator.flush_active_flows():
                    self._record_flow_verdict(flow_data)
                self.is_running = False
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
            if purge_thread is not None:
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
            dropped_pkts = self.stats["packets_dropped"]
            total_flows = self.stats["flows_analyzed"]
            benign = self.stats["benign_flows"]
            malicious = self.stats["malicious_flows"]
            classification_errors = self.stats["classification_errors"]
            latencies = self.stats["latencies_ms"]
            avg_lat = float(sum(latencies) / len(latencies)) if latencies else 0.0
            p95_lat = float(sorted(latencies)[int(len(latencies) * 0.95)]) if latencies else 0.0

        print("\n" + "=" * 65)
        print("[SUMMARY] SENTINELFLOW IDS: SESSION EXECUTION SUMMARY")
        print("=" * 65)
        print(f"  - Total Packets Sniffed  : {total_pkts:,}")
        print(f"  - Packets Queue Dropped  : {dropped_pkts:,}")
        print(f"  - Total Flows Analyzed   : {total_flows:,}")
        print(f"  - Benign Verdicts        : {benign:,} ({benign/max(total_flows,1)*100:.1f}%)")
        print(f"  - Malicious Alerts       : {malicious:,} ({malicious/max(total_flows,1)*100:.1f}%)")
        print(f"  - Average Engine Latency : {avg_lat:.2f} ms")
        status = "PASS" if latencies and p95_lat < 250 else "FAIL" if latencies else "NOT MEASURED"
        print(f"  - 95th Percentile Latency: {p95_lat:.2f} ms (Target NFR-002 < 250ms -> {status})")
        print(f"  - Classification Errors : {classification_errors:,}")
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
        db_path=args.db_path,
    )

    daemon.start(duration_sec=args.duration)


if __name__ == "__main__":
    main()
