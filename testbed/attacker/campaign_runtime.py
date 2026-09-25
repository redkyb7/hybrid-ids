"""Bounded execution and ground-truth logging for the isolated IDS testbed."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import socket
import threading
import time
from typing import Any
from uuid import uuid4


VICTIM_IP = "192.168.100.10"
VICTIM_HTTP = f"http://{VICTIM_IP}"
VICTIM_UDP_PORT = 9999
MAX_ACTIONS = 120
MAX_SECONDS = 120
MAX_OUTBOUND_BYTES = 131_072
MAX_CONCURRENCY = 8
CAMPAIGN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PROFILE_PATH = Path(__file__).with_name("profiles.json")
_SCRIPT_PARENTS = Path(__file__).resolve().parents
DEFAULT_LOG_DIR = (
    _SCRIPT_PARENTS[2] / "data" / "campaigns"
    if len(_SCRIPT_PARENTS) > 2 else _SCRIPT_PARENTS[0] / "campaigns"
)


def utc_text(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="milliseconds")


def source_ip() -> str:
    """Read the interface address selected for the fixed victim, without sending."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((VICTIM_IP, 80))
            return str(probe.getsockname()[0])
    except OSError:
        return "unknown"


@dataclass(frozen=True)
class Profile:
    mode: str
    class_label: str
    reference_label: str | None
    scenario_label: str
    max_seconds: int
    max_actions: int
    max_outbound_bytes: int
    parameters: dict[str, Any]


def load_profiles(path: Path = PROFILE_PATH) -> dict[str, Profile]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported campaign profile schema")
    profiles = {}
    allowed_classes = {
        "Botnet", "Bruteforce", "DDoS", "DoS", "Infiltration", "Portscan", "Webattack"
    }
    for mode, values in raw["profiles"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", mode):
            raise ValueError(f"invalid mode: {mode}")
        class_label = values["class_label"]
        if class_label not in allowed_classes:
            raise ValueError(f"invalid broad class for {mode}: {class_label}")
        reference_label = values["reference_label"]
        if reference_label is not None and not isinstance(reference_label, str):
            raise ValueError(f"invalid reference label for {mode}")
        scenario_label = values["scenario_label"]
        if not isinstance(scenario_label, str) or not scenario_label.startswith("lab-"):
            raise ValueError(f"invalid scenario label for {mode}")
        limits = (
            ("max_seconds", MAX_SECONDS),
            ("max_actions", MAX_ACTIONS),
            ("max_outbound_bytes", MAX_OUTBOUND_BYTES),
        )
        for key, maximum in limits:
            value = values[key]
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{mode}.{key} must be 1..{maximum}")
        parameters = values["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError(f"invalid parameters for {mode}")
        for key, value in parameters.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite parameter {mode}.{key}")
        integer_bounds = {
            "count": (1, values["max_actions"]),
            "concurrency": (1, MAX_CONCURRENCY),
            "connections": (1, values["max_actions"]),
            "response_bytes": (0, 2048),
            "delay_ms": (0, 2000),
            "flows": (1, values["max_actions"]),
            "packets_per_flow": (1, values["max_actions"]),
            "payload_bytes": (1, 512),
            "canary_bytes": (1, 4096),
            "interval_ms": (0, 1000),
            "interval_microseconds": (1, 1_000_000),
        }
        for key, (minimum, maximum) in integer_bounds.items():
            if key in parameters:
                value = parameters[key]
                if type(value) is not int or not minimum <= value <= maximum:
                    raise ValueError(f"{mode}.{key} must be {minimum}..{maximum}")
        for key in ("interval_seconds", "fragment_interval_seconds", "duration_seconds"):
            if key in parameters:
                value = parameters[key]
                minimum = 0 if key == "interval_seconds" else 0.001
                if type(value) not in (int, float) or not minimum <= value <= values["max_seconds"]:
                    raise ValueError(f"{mode}.{key} must be {minimum}..{values['max_seconds']}")
        if mode == "botnet" and parameters.get("delay_ms", 0) > 500:
            raise ValueError("botnet response delay exceeds C2 fixture limit")
        if mode == "scan":
            ports = parameters.get("ports")
            if not isinstance(ports, list) or len(ports) > values["max_actions"]:
                raise ValueError("scan ports exceed action budget")
            if not ports or any(type(port) is not int or not 1 <= port <= 65535 for port in ports):
                raise ValueError("invalid scan port")
        if mode == "ddos_udp":
            packets = int(parameters["flows"]) * int(parameters["packets_per_flow"])
            if packets > values["max_actions"]:
                raise ValueError("UDP packets exceed action budget")
            if packets * int(parameters["payload_bytes"]) > values["max_outbound_bytes"]:
                raise ValueError("UDP bytes exceed outbound budget")
        profiles[mode] = Profile(
            mode=mode,
            class_label=class_label,
            reference_label=reference_label,
            scenario_label=scenario_label,
            max_seconds=values["max_seconds"],
            max_actions=values["max_actions"],
            max_outbound_bytes=values["max_outbound_bytes"],
            parameters=parameters,
        )
    return profiles


class BudgetExceeded(RuntimeError):
    """A campaign reached a hard action, byte or time limit."""


class Campaign:
    def __init__(
        self,
        profile: Profile,
        campaign_id: str,
        seed: int,
        worker_id: str,
        log_dir: Path | None = None,
    ) -> None:
        if not CAMPAIGN_ID_PATTERN.fullmatch(campaign_id):
            raise ValueError("campaign ID must use 1..64 letters, digits, _ or -")
        if not CAMPAIGN_ID_PATTERN.fullmatch(worker_id):
            raise ValueError("worker ID must use 1..64 letters, digits, _ or -")
        self.profile = profile
        self.campaign_id = campaign_id
        self.seed = seed
        self.worker_id = worker_id
        self.log_dir = Path(log_dir or os.environ.get("IDS_CAMPAIGN_LOG_DIR", DEFAULT_LOG_DIR))
        self.started_epoch = time.time()
        self.started_monotonic = time.monotonic()
        self.deadline = self.started_monotonic + profile.max_seconds
        self.source_ip = source_ip()
        self.actions_attempted = 0
        self.actions_succeeded = 0
        self.estimated_outbound_bytes = 0
        self.application_response_bytes = 0
        self.events: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._lock = threading.Lock()

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def reserve(self, *, actions: int = 1, estimated_outbound_bytes: int = 0) -> None:
        if actions < 1 or estimated_outbound_bytes < 0:
            raise ValueError("invalid action reservation")
        with self._lock:
            if self.remaining_seconds() <= 0:
                raise BudgetExceeded("campaign deadline reached")
            if self.actions_attempted + actions > self.profile.max_actions:
                raise BudgetExceeded("action cap reached")
            if self.estimated_outbound_bytes + estimated_outbound_bytes > self.profile.max_outbound_bytes:
                raise BudgetExceeded("estimated outbound byte cap reached")
            self.actions_attempted += actions
            self.estimated_outbound_bytes += estimated_outbound_bytes

    def record(
        self,
        action: str,
        started_epoch: float,
        *,
        success: bool,
        actions_completed: int = 1,
        response_bytes: int = 0,
        detail: str = "",
    ) -> None:
        if actions_completed < 1:
            raise ValueError("actions_completed must be positive")
        ended_epoch = time.time()
        event = {
            "action": action,
            "started_epoch": round(started_epoch, 3),
            "ended_epoch": round(ended_epoch, 3),
            "success": success,
            "actions_completed": actions_completed if success else 0,
            "response_bytes": response_bytes,
            "detail": detail[:120],
        }
        with self._lock:
            if success:
                self.actions_succeeded += actions_completed
            elif len(self.errors) < 10:
                self.errors.append(f"{action}: {detail[:120]}")
            self.application_response_bytes += response_bytes
            self.events.append(event)

    def note_error(self, message: str) -> None:
        with self._lock:
            if len(self.errors) < 10:
                self.errors.append(message[:200])

    def pause(self, seconds: float) -> None:
        remaining = self.remaining_seconds()
        if remaining > 0 and seconds > 0:
            time.sleep(min(seconds, remaining))

    def write_manifest(self, *, fatal_error: str | None = None) -> Path:
        ended_epoch = time.time()
        if fatal_error and len(self.errors) < 10:
            self.errors.append(fatal_error[:200])
        status = "failed" if fatal_error or not self.actions_succeeded else (
            "partial" if self.errors else "completed"
        )
        manifest = {
            "schema_version": 1,
            "campaign_id": self.campaign_id,
            "worker_id": self.worker_id,
            "mode": self.profile.mode,
            "class_label": self.profile.class_label,
            "reference_label": self.profile.reference_label,
            "scenario_label": self.profile.scenario_label,
            "seed": self.seed,
            "target_ip": VICTIM_IP,
            "source_ip": self.source_ip,
            "started_epoch": round(self.started_epoch, 3),
            "ended_epoch": round(ended_epoch, 3),
            "started_utc": utc_text(self.started_epoch),
            "ended_utc": utc_text(ended_epoch),
            "status": status,
            "actions_attempted": self.actions_attempted,
            "actions_succeeded": self.actions_succeeded,
            "estimated_outbound_bytes": self.estimated_outbound_bytes,
            "application_response_bytes": self.application_response_bytes,
            "limits": {
                "max_seconds": self.profile.max_seconds,
                "max_actions": self.profile.max_actions,
                "max_outbound_bytes": self.profile.max_outbound_bytes,
            },
            "events": self.events,
            "errors": self.errors,
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        destination = self.log_dir / f"{self.campaign_id}-{self.worker_id}.json"
        temporary = self.log_dir / f".{destination.name}.{uuid4().hex}.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        return destination


def new_campaign_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"
