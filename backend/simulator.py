import sqlite3
import random
import time
from datetime import datetime

# Initialize SQLite database
conn = sqlite3.connect('../data/ids_logs.db')
cursor = conn.cursor()

# Create table matching project requirements
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

protocols = ['TCP', 'UDP', 'ICMP']
attacks = ['Normal Traffic', 'Port Scan', 'DDoS', 'Botnet', 'Brute Force']
ips = ['192.168.1.10', '10.0.0.200', '172.16.0.45', '192.168.0.88']

print("Simulating live backend detection data... Press Ctrl+C to stop.")

try:
    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        src = random.choice(ips)
        dst = '192.168.1.1'
        proto = random.choice(protocols)
        # 60% chance of normal traffic, 40% chance of attack
        attack = random.choices(attacks, weights=[0.6, 0.15, 0.1, 0.1, 0.05])[0]
        latency = random.randint(180, 240) # Under 250ms target

        cursor.execute('''
            INSERT INTO logs (timestamp, source_ip, destination_ip, protocol, attack_type, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (now, src, dst, proto, attack, latency))
        conn.commit()
        
        time.sleep(2) # Insert new log every 2 seconds
except KeyboardInterrupt:
    print("Simulator stopped.")
    conn.close()