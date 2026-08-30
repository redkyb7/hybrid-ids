"""
SentinelFlow IDS - SOC-Grade Real-Time Telemetry & Attack Campaign Simulator
=============================================================================
Simulates realistic enterprise network traffic dynamics including:
  1. Normal Baseline Operations (Steady state corporate web/DNS traffic)
  2. Port Scan Reconnaissance Sweeps (Sequential probe bursts)
  3. Distributed Denial of Service Surges (Multi-source volumetric floods)
  4. Authentication Brute Force Campaigns (Repeated SSH/FTP credential attacks)
  5. Web Exploitation Waves (Application-layer XSS / SQL Injection)

Feeds authentic flow statistics through the Two-Stage Hybrid ML/DL Engine
(Stage 1 ML Filter -> Stage 2 DL Classifier) and logs telemetry to SQLite.
"""

import os
import pickle
import random
import sqlite3
import time
from datetime import datetime
from hybrid_engine import HybridIDSEngine

# Resolve Database Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "ids_logs.db")

# Initialize SQLite database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Ensure table schema exists
cursor.execute('''
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
conn.commit()

# Load Simulation Traffic Pool
POOL_PATH = os.path.join(BASE_DIR, "simulation_traffic_pool.pkl")
traffic_pool = {}
if os.path.exists(POOL_PATH):
    with open(POOL_PATH, "rb") as f:
        traffic_pool = pickle.load(f)
    total_flows = sum(len(v) for v in traffic_pool.values())
    print(f"[+] Loaded simulation traffic pool ({total_flows} authentic flows across {len(traffic_pool)} classes)")
else:
    print("[!] Warning: simulation_traffic_pool.pkl not found.")

# Initialize Hybrid ML/DL Engine
print("[*] Initializing SentinelFlow Two-Stage Hybrid ML/DL Engine...")
engine = HybridIDSEngine()

# ── Simulated Network Architecture ──────────────────────────────────────────
VICTIM_SERVER_IP = "192.168.1.10" # Web, SSH, and Database Server
INTERNAL_CLIENTS = ["192.168.1.101", "192.168.1.102", "192.168.1.115", "192.168.1.140"]
KALI_ATTACKER_IP = "10.0.0.66"    # Internal testbed attacker
BOTNET_NODE_IPS  = ["45.33.32.156", "185.220.101.5", "194.26.29.112", "91.240.118.232"]
COMMON_PORTS     = [80, 443, 22, 21, 53, 8080, 3306]

# ── Campaign State Machine ──────────────────────────────────────────────────
CAMPAIGN_TYPES = [
    "NORMAL_BASELINE",
    "PORT_SCAN_SWEEP",
    "DDOS_SURGE",
    "DOS_VOLUMETRIC",
    "BRUTE_FORCE_WAVE",
    "WEB_EXPLOITATION",
    "BOTNET_C2_ACTIVITY"
]

CAMPAIGN_WEIGHTS = [0.45, 0.12, 0.12, 0.10, 0.08, 0.07, 0.06]

current_campaign = "NORMAL_BASELINE"
campaign_remaining_ticks = 10
scan_port_index = 0
scan_ports_list = [21, 22, 23, 25, 53, 80, 110, 143, 443, 3306, 8080, 8443]

print(f"[+] Connected to database: {DB_PATH}")
print("[+] Starting SOC-Grade Real-Time Telemetry Stream with Campaign Engine...")
print("    Press Ctrl+C to stop.")

try:
    while True:
        # Check Campaign State Transition
        if campaign_remaining_ticks <= 0:
            current_campaign = random.choices(CAMPAIGN_TYPES, weights=CAMPAIGN_WEIGHTS)[0]
            campaign_remaining_ticks = random.randint(6, 14)
            print("\n" + "=" * 65)
            print(f"[!] NETWORK CAMPAIGN SHIFT -> [{current_campaign}] ({campaign_remaining_ticks} events scheduled)")
            print("=" * 65)

        campaign_remaining_ticks -= 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ----------------------------------------------------------------------
        # Configure Flow Based on Active Campaign
        # ----------------------------------------------------------------------
        if current_campaign == "NORMAL_BASELINE":
            flow_category = "Normal Traffic"
            src_ip = random.choice(INTERNAL_CLIENTS)
            dst_ip = VICTIM_SERVER_IP
            proto = random.choice(["TCP", "TCP", "UDP"])
            port = random.choice([80, 443, 53])
            sleep_interval = random.uniform(1.2, 2.0)

        elif current_campaign == "PORT_SCAN_SWEEP":
            flow_category = "Port Scan"
            src_ip = KALI_ATTACKER_IP
            dst_ip = VICTIM_SERVER_IP
            proto = "TCP"
            port = scan_ports_list[scan_port_index % len(scan_ports_list)]
            scan_port_index += 1
            sleep_interval = random.uniform(0.3, 0.6) # Fast scan burst

        elif current_campaign == "DDOS_SURGE":
            flow_category = "DDoS"
            src_ip = random.choice(BOTNET_NODE_IPS)
            dst_ip = VICTIM_SERVER_IP
            proto = random.choice(["TCP", "UDP"])
            port = random.choice([80, 443])
            sleep_interval = random.uniform(0.2, 0.5) # Flood burst

        elif current_campaign == "DOS_VOLUMETRIC":
            flow_category = "DoS"
            src_ip = KALI_ATTACKER_IP
            dst_ip = VICTIM_SERVER_IP
            proto = "TCP"
            port = 80
            sleep_interval = random.uniform(0.4, 0.7)

        elif current_campaign == "BRUTE_FORCE_WAVE":
            flow_category = "Brute Force"
            src_ip = KALI_ATTACKER_IP
            dst_ip = VICTIM_SERVER_IP
            proto = "TCP"
            port = random.choice([22, 21]) # SSH / FTP
            sleep_interval = random.uniform(0.5, 0.9)

        elif current_campaign == "WEB_EXPLOITATION":
            flow_category = "Web Attack"
            src_ip = random.choice(BOTNET_NODE_IPS)
            dst_ip = VICTIM_SERVER_IP
            proto = "TCP"
            port = random.choice([80, 8080])
            sleep_interval = random.uniform(0.8, 1.4)

        elif current_campaign == "BOTNET_C2_ACTIVITY":
            flow_category = "Botnet"
            src_ip = random.choice(BOTNET_NODE_IPS)
            dst_ip = VICTIM_SERVER_IP
            proto = "TCP"
            port = random.choice([8080, 443])
            sleep_interval = random.uniform(0.7, 1.2)

        # ----------------------------------------------------------------------
        # Draw Authentic Flow Features from Pool
        # ----------------------------------------------------------------------
        samples = traffic_pool.get(flow_category, [])
        if not samples:
            samples = traffic_pool.get("Normal Traffic", [{}])

        raw_flow = dict(random.choice(samples))

        # Enrich with active network attributes
        raw_flow["source_ip"] = src_ip
        raw_flow["destination_ip"] = dst_ip
        raw_flow["protocol"] = proto
        raw_flow["Destination Port"] = port

        # ----------------------------------------------------------------------
        # Execute Real Two-Stage Hybrid Classification
        # ----------------------------------------------------------------------
        result = engine.classify_flow(raw_flow)

        attack_type = result["attack_type"]
        latency_ms = int(result["latency_ms"])
        stage_reached = result["stage_reached"]
        confidence = result["confidence"]

        # Insert live telemetry into SQLite database
        cursor.execute('''
            INSERT INTO logs (timestamp, source_ip, destination_ip, protocol, attack_type, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (now, src_ip, dst_ip, proto, attack_type, latency_ms))
        conn.commit()

        # Terminal Visual Output
        if attack_type == "Normal Traffic":
            tag = "[SAFE]"
        else:
            tag = "[ALERT]"

        print(f"[{now}] {tag} {src_ip:<15} -> {dst_ip}:{port:<5} ({proto:<4}) | {attack_type:<14} ({confidence*100:5.1f}%) | {stage_reached} | {latency_ms}ms")

        time.sleep(sleep_interval)

except KeyboardInterrupt:
    print("\n[!] Simulator stopped by user.")
    conn.close()