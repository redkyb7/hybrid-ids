# SentinelFlow Isolated Virtual Testbed (Docker Compose)

This testbed establishes an isolated virtual network (`192.168.100.0/24`) for live traffic generation, realistic attack execution, and packet capture benchmarking.

---

## 1. Network Topology

| Container Name | Role | IP Address | Services / Tools |
| :--- | :--- | :--- | :--- |
| **`ids-victim`** | Target Server | `192.168.100.10` | Nginx/Flask (Port 80), OpenSSH (Port 22, user `admin:password123`), SQLi search endpoint |
| **`ids-benign-client`** | Normal Client | `192.168.100.101` | Automated HTTP/API requests generating legitimate background traffic |
| **`ids-attacker`** | Attacker Suite | `192.168.100.66` | `nmap` (Port Scans), `hping3` (DoS SYN Flood), `hydra` (Brute Force), Web attack automation |

---

## 2. Quick Start & Execution

### Prerequisites
Make sure **Docker Desktop** is running on your machine.

### Start the Testbed
From the `hybrid-ids/` directory:

```bash
# Build and start all three nodes in background
docker compose up -d --build
```

### Check Running Containers
```bash
docker compose ps
```

### View Live Attack & Traffic Logs
```bash
# Follow logs from the attacker node
docker compose logs -f attacker

# Follow logs from the victim web application
docker compose logs -f victim

# Follow logs from the benign client generator
docker compose logs -f benign_client
```

---

## 3. Running Specific Attack Campaigns Interactively

You can also execute specific attack scenarios on demand using `docker compose exec`:

```bash
# 1. Run Port Scan only (Nmap)
docker compose exec attacker python /app/attack_campaigns.py --attack scan

# 2. Run Volumetric DoS SYN Flood
docker compose exec attacker python /app/attack_campaigns.py --attack dos

# 3. Run Credential Brute-Force (HTTP login & SSH)
docker compose exec attacker python /app/attack_campaigns.py --attack bruteforce

# 4. Run Web Exploitation (SQL Injection & XSS)
docker compose exec attacker python /app/attack_campaigns.py --attack web

# 5. Run Botnet C2 Beaconing
docker compose exec attacker python /app/attack_campaigns.py --attack botnet

# 6. Run all campaigns once in sequence
docker compose exec attacker python /app/attack_campaigns.py --attack all
```

---

## 4. Stopping the Testbed

```bash
docker compose down
```
