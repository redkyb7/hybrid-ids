"""Generate a bounded, manifest-labeled benign HTTP campaign in the lab."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import socket
import time
from uuid import uuid4

import requests


VICTIM_IP = "192.168.100.10"
ENDPOINTS = ("/", "/api/status", "/api/data")
SCENARIOS = {
    "mixed": (ENDPOINTS, 24, (0.15, 0.35)),
    "api": (("/api/status", "/api/data"), 30, (0.25, 0.60)),
    "burst": (ENDPOINTS, 40, (0.02, 0.08)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="mixed")
    parser.add_argument("--count", type=int)
    parser.add_argument("--worker-id", default="benign-client")
    parser.add_argument("--start-at-epoch", type=float)
    parser.add_argument("--header-pad", type=int, default=0,
                        help="Extra fixed HTTP request-header bytes for controlled benign probes")
    parser.add_argument("--output-dir", type=Path, default=Path("/app/campaigns"))
    args = parser.parse_args()
    endpoints, default_count, pause_range = SCENARIOS[args.scenario]
    count = args.count if args.count is not None else default_count
    if not 1 <= count <= 60:
        parser.error("count must be 1..60")
    if not 0 <= args.header_pad <= 512:
        parser.error("header-pad must be 0..512")
    if not args.campaign_id or len(args.campaign_id) > 64 or not all(
        char.isascii() and (char.isalnum() or char in "_-") for char in args.campaign_id
    ):
        parser.error("campaign ID must use 1..64 letters, digits, _ or -")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.worker_id):
        parser.error("worker ID must use 1..64 letters, digits, _ or -")
    if args.start_at_epoch is not None:
        delay = args.start_at_epoch - time.time()
        if not 0 <= delay <= 30:
            parser.error("start-at-epoch must be within the next 30 seconds")
        time.sleep(delay)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect((VICTIM_IP, 80))
        source_ip = probe.getsockname()[0]

    rng = random.Random(args.seed)
    headers = {"User-Agent": "SentinelFlow-Lab-Benign/1.0"}
    if args.header_pad:
        headers["X-Benign-Pad"] = "A" * args.header_pad
    started = time.time()
    successes = 0
    events = []
    for _ in range(count):
        endpoint = rng.choice(endpoints)
        action_start = time.time()
        try:
            response = requests.get(
                f"http://{VICTIM_IP}{endpoint}",
                headers=headers,
                timeout=3,
            )
            success = response.status_code == 200
            detail = f"HTTP {response.status_code}"
            response_bytes = len(response.content)
        except requests.RequestException as error:
            success = False
            detail = type(error).__name__
            response_bytes = 0
        successes += int(success)
        events.append({
            "action": f"GET {endpoint}",
            "started_epoch": round(action_start, 3),
            "ended_epoch": round(time.time(), 3),
            "success": success,
            "response_bytes": response_bytes,
            "detail": detail,
        })
        time.sleep(rng.uniform(*pause_range))
    ended = time.time()
    status = "completed" if successes == count else "partial"
    utc = lambda value: datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="milliseconds")
    manifest = {
        "schema_version": 1,
        "campaign_id": args.campaign_id,
        "worker_id": args.worker_id,
        "mode": "benign_http" if args.scenario == "mixed" else f"benign_http_{args.scenario}",
        "class_label": "Benign",
        "reference_label": None,
        "scenario_label": f"lab-benign-http-{args.scenario}",
        "seed": args.seed,
        "header_pad_bytes": args.header_pad,
        "target_ip": VICTIM_IP,
        "source_ip": source_ip,
        "started_epoch": round(started, 3),
        "ended_epoch": round(ended, 3),
        "started_utc": utc(started),
        "ended_utc": utc(ended),
        "status": status,
        "actions_attempted": count,
        "actions_succeeded": successes,
        "events": events,
        "errors": [] if status == "completed" else [f"{count - successes} HTTP requests failed"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"{args.campaign_id}-{args.worker_id}.json"
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(f"[{args.campaign_id}] benign_http_{args.scenario}: {successes}/{count} succeeded from {source_ip}")
    print(f"Manifest: {destination}")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
