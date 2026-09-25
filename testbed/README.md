# Isolated Docker testbed

The lab uses a private `192.168.100.0/24` bridge. Its victim is
`192.168.100.10`; the normal client is `.101`, the main attacker is `.66`, and
the two optional DDoS workers are `.67` and `.68`. The monitor captures traffic
in the victim's network namespace and runs the deployed ML → DL pipeline.
Only model verdicts produce alerts. The victim exposes HTTP on host
`127.0.0.1:8080` and SSH on `127.0.0.1:2222`. The UDP sink is reachable only
inside the bridge.

## Start

From the repository root, with Docker Desktop running:

```powershell
# Turn on numeric flow-vector logging for feature comparisons.
$env:IDS_LOG_FEATURES = '1'
docker compose up -d --build
docker compose ps
docker compose logs --tail 30 monitor
```

The default attacker runs at most 20 bounded campaigns or 300 seconds in one
container run. The normal client continues to generate background traffic.
Each attack campaign writes a ground-truth manifest under `data/campaigns/`.
These local manifests and evaluations are ignored by Git.

For controlled measurements, start only the victim and monitor. This keeps
the normal generator and continuous attacker out of the evaluation window:

```powershell
$env:IDS_LOG_FEATURES = '1'
docker compose up -d --build victim
docker compose up -d --no-deps --build monitor
```

## Run one labeled campaign

The PowerShell helper checks that the victim and monitor are running, pauses
any running traffic generators, runs one bounded campaign, and restores their
prior running state. Its target is fixed to the lab victim.

```powershell
.\scripts\run_testbed_campaign.ps1 -Attack ssh_bruteforce -Seed 42
```

Available modes:

| Broad class | Modes |
| --- | --- |
| Portscan | `scan` |
| Botnet | `botnet` |
| Bruteforce | `ssh_bruteforce` |
| Webattack | `web_login`, `web_sqli`, `web_xss` |
| DoS | `dos_http`, `dos_slow`, `dos_syn` |
| DDoS | `ddos_http_loic`, `ddos_http_hoic`, `ddos_udp` |
| Infiltration | `infiltration` |

The DDoS helper modes launch both opt-in workers with a shared campaign ID
and distinct source IPs. Other modes use a temporary attacker container.
`web_login` is labeled **Webattack**, since failed HTTP form submissions are
different from the source dataset's SSH-heavy **Bruteforce** class.
The infiltration fixture accesses only a synthetic in-memory canary and sends
it back to a lab-only endpoint. Profile limits are in
[`attacker/profiles.json`](attacker/profiles.json).

The helper prints the generated campaign ID. After it finishes, evaluate that
run against the monitor's SQLite records:

```powershell
uv tool run --from duckdb python scripts/evaluate_attack_campaigns.py --campaign-id 20260925T082613Z-85c7c3f1
```

The JSON result is written to `data/campaigns/<ID>-evaluation.json`. It reports
source IPs, action and connection counts, Stage 2 reach, model alerts,
correct model classes and latency. With `IDS_LOG_FEATURES=1`,
it also compares the fullest captured snapshot per connection with the raw CIC
subtype's 10th to 90th percentile band. A snapshot is a model verdict, not a
unique connection.

See [`ATTACKER_IMPLEMENTATION_AND_EVALUATION.md`](../ATTACKER_IMPLEMENTATION_AND_EVALUATION.md)
for historical rule-assisted lab results, two model-only follow-up campaigns,
and limitations. Rerun the remaining campaigns to measure the current monitor.

## Inspect and stop

```powershell
docker compose logs --tail 100 monitor
docker compose logs --tail 100 victim
docker compose stop
```

`docker compose stop` preserves containers, SQLite telemetry and campaign
manifests. Use `docker compose down` only when you also want to remove the lab
containers and network.
