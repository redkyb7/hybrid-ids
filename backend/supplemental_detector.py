"""Small, explicit live-traffic checks for signals absent from CIC flow features.

These rules supplement the ML -> DL verdict. They do not use source IP allowlists
or claim a calibrated probability. Flow snapshots from the same TCP connection
are deduplicated for rate checks.
"""

from collections import defaultdict, deque
import re
import threading
import time
from urllib.parse import unquote_plus


class SupplementalDetector:
    SCAN_WINDOW_SECONDS = 10.0
    SCAN_DISTINCT_PORTS = 6
    SYN_WINDOW_SECONDS = 5.0
    SYN_DISTINCT_CONNECTIONS = 20
    LOGIN_WINDOW_SECONDS = 20.0
    LOGIN_DISTINCT_CONNECTIONS = 5
    MAX_WINDOW_SECONDS = 30.0

    _SQL_PROBE = re.compile(
        r"\bunion\s+select\b|\bor\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+|--|\border\s+by\s+\d+",
        re.IGNORECASE,
    )
    _XSS_PROBE = re.compile(
        r"<\s*script\b|<\s*img\b|<\s*svg\b|\bonerror\s*=|javascript\s*:",
        re.IGNORECASE,
    )

    def __init__(self):
        self._lock = threading.Lock()
        self._probe_seen = {}
        self._login_seen = {}
        self._scan = defaultdict(deque)
        self._syn = defaultdict(deque)
        self._login = defaultdict(deque)
        self._checks = 0

    @staticmethod
    def _prune(events, cutoff):
        while events and events[0][0] < cutoff:
            events.popleft()

    def _sweep(self, now):
        cutoff = now - self.MAX_WINDOW_SECONDS
        for seen in (self._probe_seen, self._login_seen):
            for key, seen_at in list(seen.items()):
                if seen_at < cutoff:
                    del seen[key]
        for buckets in (self._scan, self._syn, self._login):
            for key, events in list(buckets.items()):
                self._prune(events, cutoff)
                if not events:
                    del buckets[key]

    @staticmethod
    def _http_request(payload):
        first_line = payload.split("\r\n", 1)[0]
        parts = first_line.split(" ", 2)
        if len(parts) < 3 or not parts[2].startswith("HTTP/"):
            return None, None
        return parts[0].upper(), unquote_plus(parts[1])

    def inspect(self, flow):
        """Return an alert label/rule ID or None for one emitted flow snapshot."""
        now = float(flow.get("flow_end_ts") or time.time())
        source = str(flow.get("source_ip", ""))
        destination = str(flow.get("destination_ip", ""))
        source_port = int(flow.get("Source Port", 0))
        destination_port = int(flow.get("Destination Port", 0))
        protocol = str(flow.get("protocol", "")).upper()
        payload = str(flow.get("payload_sample", ""))
        method, target = self._http_request(payload)
        connection = (source, destination, source_port, destination_port, protocol)

        with self._lock:
            self._checks += 1
            if self._checks % 512 == 0:
                self._sweep(now)

            # Payload checks are per connection and work even if a later
            # snapshot contains the first complete request line.
            if method == "GET" and target.startswith("/search"):
                if self._SQL_PROBE.search(target) or self._XSS_PROBE.search(target):
                    return {"attack_type": "Web Attack", "rule_id": "http_probe"}
            if method == "GET" and (
                "user-agent: mirai/" in payload.lower()
                or "x-bot-id:" in payload.lower()
            ):
                return {"attack_type": "Botnet", "rule_id": "beacon_header"}

            if method == "POST" and target == "/login":
                events = self._login[(source, destination)]
                self._prune(events, now - self.LOGIN_WINDOW_SECONDS)
                if connection not in self._login_seen:
                    self._login_seen[connection] = now
                    events.append((now, connection))
                if len(events) >= self.LOGIN_DISTINCT_CONNECTIONS:
                    return {"attack_type": "Brute Force", "rule_id": "login_rate"}

            probe_like = (
                protocol == "TCP"
                and int(flow.get("SYN Flag Count", 0)) >= 1
                and int(flow.get("Total Fwd Packets", 0)) <= 2
                and int(flow.get("Total Backward Packets", 0)) <= 2
            )
            if probe_like:
                scan_events = self._scan[(source, destination)]
                self._prune(scan_events, now - self.SCAN_WINDOW_SECONDS)
                syn_events = self._syn[(source, destination, destination_port)]
                self._prune(syn_events, now - self.SYN_WINDOW_SECONDS)
                if connection not in self._probe_seen:
                    self._probe_seen[connection] = now
                    scan_events.append((now, destination_port))
                    syn_events.append((now, connection))

            syn_events = self._syn.get((source, destination, destination_port), ())
            if probe_like and len(syn_events) >= self.SYN_DISTINCT_CONNECTIONS:
                return {"attack_type": "DoS", "rule_id": "syn_burst"}

            scan_events = self._scan.get((source, destination), ())
            if probe_like and len({port for _, port in scan_events}) >= self.SCAN_DISTINCT_PORTS:
                return {"attack_type": "Port Scan", "rule_id": "multi_port_scan"}

        return None
