"""
SentinelFlow IDS - Real-Time Telemetry & Hybrid Inference Simulator
===================================================================
Generates real-time network flow packets, feeds them through the
Two-Stage Hybrid ML/DL Engine (Stage 1 ML -> Stage 2 DL), and writes
live telemetry with latency profiling into the SQLite database.
"""

import os
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

# Initialize Hybrid ML/DL Engine
print("[*] Initializing SentinelFlow Two-Stage Hybrid ML/DL Engine...")
engine = HybridIDSEngine()

protocols = ['TCP', 'UDP', 'ICMP']
ips = ['192.168.1.105', '10.0.0.210', '172.16.0.45', '192.168.0.88', '45.33.32.156']
dst_ip = '192.168.1.1'
target_ports = [80, 443, 22, 21, 8080, 53]

print(f"[+] Connected to database: {DB_PATH}")
print("[+] Simulating real-time hybrid ML/DL intrusion detection stream...")
print("    Press Ctrl+C to stop.")

try:
    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        src = random.choice(ips)
        proto = random.choice(protocols)
        port = random.choice(target_ports)

        # Generate realistic CICFlowMeter flow dictionary
        raw_flow = {
            "source_ip": src,
            "destination_ip": dst_ip,
            "protocol": proto,
            "Destination Port": port,
            "Flow Duration": random.randint(1000, 500000),
            "Total Fwd Packets": random.randint(1, 150),
            "Total Length of Fwd Packets": random.randint(40, 65000),
            "Flow Bytes/s": random.uniform(100.0, 500000.0),
            "Flow Packets/s": random.uniform(1.0, 2500.0),
            "Flow IAT Mean": random.uniform(10.0, 1000.0),
            "FIN Flag Count": random.choice([0, 1]),
            "SYN Flag Count": random.choice([0, 1]),
            "RST Flag Count": random.choice([0, 0, 0, 1]),
            "PSH Flag Count": random.choice([0, 1]),
            "ACK Flag Count": random.choice([0, 1, 1]),
        }

        # -------------------------------------------------------------
        # Execute Real Two-Stage Hybrid Classification
        # -------------------------------------------------------------
        result = engine.classify_flow(raw_flow)

        attack_type = result["attack_type"]
        latency_ms = int(result["latency_ms"])
        stage_reached = result["stage_reached"]
        confidence = result["confidence"]

        # Insert live telemetry into SQLite database
        cursor.execute('''
            INSERT INTO logs (timestamp, source_ip, destination_ip, protocol, attack_type, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (now, src, dst_ip, proto, attack_type, latency_ms))
        conn.commit()

        # Terminal output
        tag = "[SAFE]" if attack_type == "Normal Traffic" else "[ALERT]"
        print(f"[{now}] {tag} {src} -> {dst_ip}:{port} ({proto}) | {attack_type} ({confidence*100:.1f}%) | {stage_reached} | {latency_ms}ms")

        time.sleep(2) # Stream interval

except KeyboardInterrupt:
    print("\n[!] Simulator stopped.")
    conn.close()