# Docker testbed: running and controlled testing

Run every command below from the **repository root**, in PowerShell, with Docker Desktop running Linux containers.

## Network and current models

| Component | Address / purpose |
| --- | --- |
| Victim | `192.168.100.10`; HTTP, SSH and a lab UDP sink |
| Benign client | `192.168.100.101` |
| Main attacker | `192.168.100.66` |
| DDoS workers | `192.168.100.67` and `192.168.100.68` |
| Victim HTTP / SSH on Windows | `http://localhost:8080` / `localhost:2222` |
| Dashboard | `http://localhost:8000` |

The monitor captures the victim's network namespace. The deployed pipeline is **Random Forest → residual MLP**, from `updated_models/five_attack/`, with ML attack threshold **0.10** and DL confidence threshold **0.30**. Only model verdicts produce alerts. Stage 1 can stop a flow as benign; Stage 2 can name an attack, return benign, or return Unknown Attack.

The dashboard's **0.8372 macro F1 is the standalone offline DL score**. It does not score the current campaign. See [the model comparison](../MODEL_IMPLEMENTATION_COMPARISON.md) for pipeline validation and known false alarms.

## 1. Automatic demonstration

```powershell
docker compose up -d --build
docker compose ps
docker compose logs --tail 30 monitor
```

This starts the background benign client and the bounded automatic attacker, alongside the victim, monitor and dashboard. The attacker exits after at most 20 campaigns or 300 seconds. Its automatic rotation covers scan, Botnet, SSH brute force and HTTP DoS. Use the individual helper below for two-source DDoS and other variants.

## 2. Prepare a controlled test

Stop background generators **before** testing so they cannot resume between individual campaigns. If optional DDoS workers were previously started, stop them too:

```powershell
docker compose stop attacker benign_client
docker compose --profile ddos stop ddos_worker_1 ddos_worker_2

# Use the deployed bundle and its saved threshold.
Remove-Item Env:IDS_ARTIFACT_ROOT -ErrorAction SilentlyContinue
Remove-Item Env:IDS_STAGE1_THRESHOLD -ErrorAction SilentlyContinue
$env:IDS_LOG_FEATURES = '1'

docker compose up -d --build victim
docker compose up -d --no-deps --build --force-recreate monitor dashboard
docker compose ps
docker compose logs --tail 30 monitor
```

Expected running services: **victim, monitor, dashboard**. Continuous attacker/client and DDoS workers should be stopped or absent. `--no-deps` prevents the monitor's dependencies from starting the continuous generators. Recreating the monitor also reconnects it if the victim was recreated.

In a second terminal, follow new model decisions:

```powershell
docker compose logs --since 1m -f monitor
```

Open the dashboard. Historical totals remain in the database, so use new timestamps and the campaign's source/destination to identify this run. A single connection can produce multiple cumulative snapshots.

## 3. Run one attack type at a time

The helper checks victim/monitor readiness, builds the relevant traffic image and runs a bounded campaign against the fixed lab victim. It prints a **campaign ID** and saves independent activity manifests under `data/campaigns/`.

Run these commands individually. Inspect each result before moving to the next test:

```powershell
# Port scan
.\scripts\run_testbed_campaign.ps1 -Attack scan_sparse -Seed 101

# Botnet-like repeated traffic
.\scripts\run_testbed_campaign.ps1 -Attack botnet_slow -Seed 102

# SSH brute force
.\scripts\run_testbed_campaign.ps1 -Attack ssh_bruteforce_slow -Seed 103

# HTTP DoS
.\scripts\run_testbed_campaign.ps1 -Attack dos_http -Seed 104

# Distributed UDP DDoS: two separate source containers
.\scripts\run_testbed_campaign.ps1 -Attack ddos_udp -Seed 105
```

The helper waits for each campaign and then allows three seconds for processing. Allow another five seconds between runs if you are comparing isolated windows:

```powershell
Start-Sleep -Seconds 5
```

DDoS modes launch both workers with a common campaign ID/start time and distinct source IPs. Both must finish successfully. One busy attacker container alone does not establish a distributed attack.

The helper temporarily pauses running continuous generators and restores their prior running state afterwards. In the controlled setup above they remain stopped. After a demonstration, the helper can restart a previously running attacker with a new bounded campaign budget; use the controlled setup for isolated measurements.

### All supported individual modes

| Intended activity | Valid `-Attack` values |
| --- | --- |
| Portscan | `scan`, `scan_connect`, `scan_sparse` |
| Botnet | `botnet`, `botnet_fast`, `botnet_slow` |
| Bruteforce | `ssh_bruteforce`, `ssh_bruteforce_fast`, `ssh_bruteforce_slow` |
| DoS | `dos_http`, `dos_slow`, `dos_syn` |
| DDoS | `ddos_udp`, `ddos_http_loic`, `ddos_http_hoic` |
| Benign HTTP | `benign_http`, `benign_http_api`, `benign_http_burst` |
| Benign slow HTTP | `benign_slow_http` |
| Other Attack probes | `web_login`, `web_sqli`, `web_xss`, `infiltration` |

Web probes and infiltration are outside the five named attack classes. Their source labels map to **Other Attack** in training. HTTP login failures are web activity; they are not the SSH brute-force fixture. Profile limits and parameters are in [attacker/profiles.json](attacker/profiles.json).

## 4. Test legitimate traffic and matched slow requests

First establish a benign control:

```powershell
.\scripts\run_testbed_campaign.ps1 -Attack benign_http -Seed 201
.\scripts\run_testbed_campaign.ps1 -Attack benign_http_api -Seed 202
.\scripts\run_testbed_campaign.ps1 -Attack benign_http_burst -Seed 203
```

Successful legitimate requests provide a benign activity label. Inspect whether the model incorrectly alerts; a successful campaign does not guarantee a correct model classification.

Then run a matched slow attack/control pair, with the same settings:

```powershell
.\scripts\run_testbed_campaign.ps1 -Attack dos_slow -Seed 301 -SlowDuration 12 -SlowInterval 0.6 -SlowConnections 4 -SlowPadding 32 -SlowJitter 0.15
Start-Sleep -Seconds 5
.\scripts\run_testbed_campaign.ps1 -Attack benign_slow_http -Seed 301 -SlowDuration 12 -SlowInterval 0.6 -SlowConnections 4 -SlowPadding 32 -SlowJitter 0.15
```

Both use the same request prefix and timed header fragments. The attack leaves headers unfinished; the benign client completes them and verifies a successful response. The benign slow handler runs from the attacker image/IP to avoid making client identity the distinction between classes.

| Slow parameter | Accepted range |
| --- | --- |
| `-SlowDuration` | 2–25 seconds |
| `-SlowInterval` | 0.2–2 seconds |
| `-SlowConnections` | 1–8 |
| `-SlowPadding` | 0–128 bytes |
| `-SlowJitter` | 0–0.2; 0.15 means 15% |

These overrides apply to `dos_slow` and `benign_slow_http`. Combinations exceeding the outbound budget are rejected. Seeds control generated choices; operating-system scheduling and actual packet timing can still vary.

**Known deployment limitation:** previous validation falsely named 97/101 legitimate slow HTTP snapshots as DoS, affecting all five tested legitimate slow connections. Check early alerts as well as the fullest/latest snapshot. A later benign result does not erase earlier false alarms. These small bounded unfinished-header campaigns do not independently prove service denial.

## 5. Inspect the evidence for an individual run

Copy the printed campaign ID, then inspect all its worker manifests:

```powershell
$campaignId = 'PASTE_PRINTED_CAMPAIGN_ID'
Get-ChildItem -Path .\data\campaigns -Filter "$campaignId-*.json" |
    ForEach-Object { Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json } |
    Select-Object campaign_id, worker_id, mode, class_label, source_ip, target_ip, status, actions_attempted, actions_succeeded, started_utc, ended_utc
```

Inspect the full JSON `events` and `errors` if the status is partial/failed. For DDoS, verify two worker manifests and both source addresses. Manifests describe generated activity independently of model predictions; they do not automatically annotate every database row.

For a saved log window, record the start immediately before the helper call:

```powershell
$testStartedUtc = (Get-Date).ToUniversalTime().ToString('o')
.\scripts\run_testbed_campaign.ps1 -Attack scan_sparse -Seed 401

docker compose logs --since $testStartedUtc monitor |
    Set-Content -Path .\data\individual-scan-monitor.log -Encoding utf8
```

This time window includes monitor traffic, including incidental traffic. Check the manifest's time range, protocol and source/destination before assigning ground truth. Manifests use UTC timestamps; account for timezone differences when comparing dashboard times.

Record at least:

- Campaign mode/ID, seed, parameters, worker success and source addresses.
- Stage 1 stops versus Stage 2 reach, and named/benign/Unknown outcomes.
- Missed attacks and false alarms on independently verified benign activity.
- Snapshot counts separately from unique connections; early and mature outcomes.
- Latency, classification errors and any reported packet queue drops.

A manual dashboard check is a functional test. To report macro F1 or recall, independently label and align the scored snapshots/connections, account for unlabeled traffic, and preserve the exact model hashes. The historical collection/audit/scoring utilities were removed during cleanup; this helper generates traffic and manifests, **not an automatic F1 report or labeled Parquet export**. Current model provenance is in [the deployment manifest](../updated_models/deployment_manifest.json).

## 6. Optional Windows / phone packet capture

Install Wireshark with Npcap on Windows and list current interface numbers:

```powershell
& 'C:\Program Files\Wireshark\dumpcap.exe' -D
```

For an authorized phone accessing this victim over the same Wi-Fi, configure the Windows machine's current IPv4 address, then reconnect the monitor:

```powershell
$env:IDS_HTTP_BIND = 'YOUR_WINDOWS_IPV4'
docker compose up -d victim
docker compose up -d --no-deps --force-recreate monitor
```

Browse to `http://YOUR_WINDOWS_IPV4:8080` on the phone. In Windows PowerShell, start the bounded capture before performing the activity:

```powershell
.\scripts\capture_real_pcap.ps1 -Interface 5 -DurationSeconds 60 -CaptureFilter 'host PHONE_IPV4 and tcp port 8080' -OutputPath data/pcaps/phone-control-01.pcapng
```

Replace `5` with the current Wi-Fi interface number and `PHONE_IPV4` with the verified phone address. Use a new filename each run. The capture helper saves a PCAP, hash, adapter and time metadata; it does not label activity or score the models. Successful request records/packet evidence are needed for benign labels. The Windows published-port/proxy path can differ from the victim-side capture.

## 7. Finish, troubleshoot and retrain

For a controlled session, leave generators stopped after testing. To resume an automatic demonstration:

```powershell
docker compose up -d --no-deps --build attacker benign_client
```

To disable optional numeric feature logging afterwards:

```powershell
Remove-Item Env:IDS_LOG_FEATURES -ErrorAction SilentlyContinue
docker compose up -d --no-deps --force-recreate monitor
```

If traffic is absent, inspect `docker compose ps` and monitor logs. Check that `eth0` exists in the monitor:

```powershell
docker compose exec -T monitor cat /sys/class/net/eth0/operstate
```

Recreate the monitor whenever the victim is recreated. If an individual campaign fails, inspect its manifests and worker logs before treating the run as a valid labeled sample. DDoS worker logs are available with:

```powershell
docker compose --profile ddos logs --tail 30 ddos_worker_1 ddos_worker_2
```

Local manifests, numeric features, PCAPs and evaluation databases are excluded from Git. Follow [the Colab guide](../colab_five_attack_retrain/README.md) for retraining from the existing verified dataset. Future model replacements should be checked against fresh attack and benign traffic, including slow clients, after freezing model choices.

After explicitly installing a verified replacement bundle, reload its consumers:

```powershell
docker compose up -d --no-deps --force-recreate monitor dashboard
```

Stop the full testbed with `docker compose down`.
