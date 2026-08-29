"""
SentinelFlow Testbed - Benign Client Traffic Generator
======================================================
Simulates continuous, realistic background user traffic to the victim web server:
  - Standard HTML page browsing
  - Periodic JSON API status queries
  - Legitimate data fetching
"""

import random
import time
import requests

TARGET_URL = "http://192.168.100.10"

ENDPOINTS = [
    ("/", "GET", 0.45),
    ("/api/status", "GET", 0.35),
    ("/api/data", "GET", 0.20),
]


def generate_benign_traffic():
    print("=" * 65)
    print("🌐 SENTINELFLOW BENIGN CLIENT NODE")
    print(f"Generating realistic enterprise web traffic to {TARGET_URL}")
    print("=" * 65)

    headers_pool = [
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0"},
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15"},
        {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0"},
    ]

    while True:
        # Choose endpoint according to weights
        endpoint_choice = random.choices(
            [e[0] for e in ENDPOINTS],
            weights=[e[2] for e in ENDPOINTS]
        )[0]

        url = f"{TARGET_URL}{endpoint_choice}"
        headers = random.choice(headers_pool)

        try:
            r = requests.get(url, headers=headers, timeout=3)
            print(f"[{time.strftime('%H:%M:%S')}] [BENIGN GET] {url:<25} -> HTTP {r.status_code} ({len(r.content)} bytes)")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] [BENIGN] Connection waiting: {e}")

        # Sleep with normal user think-time (1.0 - 2.5s)
        time.sleep(random.uniform(1.0, 2.5))


if __name__ == "__main__":
    generate_benign_traffic()
