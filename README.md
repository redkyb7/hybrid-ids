# SentinelFlow Hybrid IDS

SentinelFlow runs a Docker testbed with a victim web server, a benign traffic generator, an attacker, an IDS monitor, and a dashboard. The monitor captures traffic to the victim, classifies flows, and writes results to `data/ids_logs.db`. The dashboard reads that database and the selected DL model's offline evaluation metrics.

## How it works

```text
Victim-side packet capture (Scapy)
  → Bidirectional flow aggregation and numeric features
  → Stage 1: Random Forest binary filter
      ├─ Attack probability < 0.10 → Normal Traffic
      └─ Attack probability ≥ 0.10 → Stage 2: residual MLP
          → Benign / named attack / Other Attack / Unknown Attack
  → SQLite telemetry → FastAPI API → browser dashboard
```

The monitor shares the victim's network namespace. It observes traffic at the
victim rather than capturing every connection across the Docker network.
A connection can generate multiple cumulative flow snapshots; database record
counts therefore do not represent unique TCP connections. Detection records
contain model decisions; the system does not automatically block traffic.

## Prerequisites

- Docker Desktop running with Linux containers (macOS or Windows), or Docker Engine with Compose on Linux.
- The deployed artifacts under `updated_models/five_attack/ml/` and `dl/`.
- Host ports 8000, 8080 and 2222 available.

Docker installs the runtime dependencies; host Python and a GPU are not required
for the Docker demonstration. Python is needed separately for local training,
archive preparation or running the Python test suite. Controlled campaign helpers
in `scripts/*.ps1` use PowerShell.

Run the commands below from the repository root, where `docker-compose.yml` is located.

## Start the Docker testbed

```sh
docker compose up -d --build
docker compose ps
```

The first build may take several minutes because the monitor image installs machine learning dependencies. `docker compose ps` should initially show `victim`, `benign_client`, `attacker`, `monitor`, and `dashboard` as running.

The attacker starts bounded simulated campaigns automatically, then exits after
20 campaigns or 300 seconds. The victim web app is available at
<http://localhost:8080>, and the IDS dashboard is at <http://localhost:8000>.
The victim's SSH port is mapped to `localhost:2222` for testbed use. See
[the isolated testbed guide](testbed/README.md) to run one labeled scenario,
including a two-source DDoS run.

To watch live IDS results:

```powershell
docker compose logs -f monitor
```

Press `Ctrl+C` to stop following logs; the containers continue running. To inspect other services, replace `monitor` with `victim`, `attacker`, or `benign_client`.

## Models used by Docker

The monitor loads **Random Forest ML → residual MLP DL** from
`updated_models/five_attack/{ml,dl}`. Stage 1 uses the saved **0.10** attack
threshold. Flows below it stop as benign; the rest reach Stage 2. DL uses
its saved **0.30** confidence threshold and can return a named attack,
Benign, or Unknown Attack. Only model verdicts produce alerts.

The dashboard displays DL standalone classification macro F1 **0.8372**.
The combined pipeline achieved **0.8650 macro F1** on 432 previously used live
validation snapshots. It detected all 308 attack snapshots but falsely named
97 of 101 legitimate slow HTTP snapshots as DoS. This limitation remains in
the selected deployment; these results are not an unseen final live test.
See [the model comparison](MODEL_IMPLEMENTATION_COMPARISON.md) and
[deployment provenance](updated_models/deployment_manifest.json).

Missing or incompatible artifacts stop the monitor with an error. Inspect
`docker compose logs monitor` to diagnose startup failures.

## Retrain for five named attack types

The training scope is now DDoS, DoS, Portscan, Bruteforce, and Botnet.
Stage 1 remains binary across **all** attack rows. Stage 2 keeps Benign and
these five attack labels, and combines the source Infiltration and Webattack
rows into `Other Attack`. Build the self-contained Colab upload archive with
`python scripts/build_five_attack_colab_zip.py`, then follow
[the Colab retraining guide](colab_five_attack_retrain/README.md). The resulting
`five_attack_colab_slow_retrain_checked.zip` includes the CIC Parquet and verified
live partitions: 879 training and 432 validation snapshots, including matched
slow-header attacks and legitimate slow clients. Extract into the fresh Colab
folder specified in the guide and verify those counts before training. The
exporter rejects models trained against a different live split plan.

Only `updated_models/five_attack/` is retained as the deployed bundle.
Uploaded model ZIPs and extracted candidates are removed after installation.
The current Colab archive and datasets remain local and excluded from Git.
A future upload must be evaluated and explicitly installed before it changes
Docker's selected bundle.

The displayed DL macro F1 does not measure Stage 1 gating or live traffic.
The monitor now records only ML/DL verdicts. The dashboard presents the saved
model verdict for older rows that were previously overridden by rules; the
database itself is preserved. Set `IDS_LOG_FEATURES=1` to
store the numeric Stage 1 and Stage 2 flow features per snapshot for a
diagnostic run; it is off by default and does not store packet payloads.
Recreate the monitor after changing this setting with
`docker compose up -d --no-deps monitor`.

Set `IDS_ARTIFACT_ROOT` before running Compose to select another complete
bundle mounted inside the monitor and dashboard containers. The dashboard
displays `--` unless that bundle has a matching
`dl/evaluation_metrics.json` report.
Clear any `IDS_STAGE1_THRESHOLD` override to use the threshold belonging to
the selected bundle.

## Optional phone access to the victim

By default HTTP is published only on localhost. For an authorized phone on
the same Wi-Fi, set your current Windows IPv4 address before starting Compose:

```powershell
$env:IDS_HTTP_BIND = 'YOUR_WINDOWS_IPV4'
docker compose up -d victim
docker compose up -d --no-deps --force-recreate monitor
```

Browse to `http://YOUR_WINDOWS_IPV4:8080` on the phone. Clear the variable to
restore the localhost default on the next victim recreation.

## Dashboard and investigation workflow

Open <http://localhost:8000>. The FastAPI dashboard reads the shared SQLite
database; it refreshes every **5 seconds** by default. Refresh frequency is
configurable, with pause/resume and manual refresh controls.

- **Overview:** historical totals, latest alert, classification breakdown and
  latency views. Select a class or latency range to investigate matching records.
- **Records and investigation:** search and filter detections, sort records,
  inspect source/destination addresses and ports, protocol, confidence, inference
  latency, pipeline stage and Stage 1 attack score when recorded.
- **Review workflow:** save records for later review, mark individual detections
  reviewed/unreviewed and inspect related records from the same source.
- **Workspace preferences:** light/dark themes, page or single-page layout and
  browser-local profile/preferences.

Bookmarks, review marks and workspace preferences use browser local storage.
They are local to that browser and are not shared accounts or server-side review
records. Clearing browser storage removes them.

**Review priority** orders investigation using repeated traffic patterns,
recency and browser-local review status. It is a frontend workflow aid, not an
additional model verdict or validated incident severity score. Marking a record
reviewed does not resolve an incident or change the saved detection.

Historical threat/flow totals include all rows in `data/ids_logs.db`.
Classification charts and average inference latency use the loaded window,
which defaults to the latest **500 snapshots**. Latency describes model engine
processing, not packet-to-browser delay. The displayed **0.8372 DL macro F1**
is the selected model's standalone offline score, not live campaign accuracy.
Missing evaluation metrics appear as `--`.

For older rows with rule-overridden labels, the API presents their saved model
verdict when available without rewriting those historical labels.

### API endpoints

| Endpoint | Response |
| --- | --- |
| `GET /` | Dashboard HTML |
| `GET /api/telemetry?limit=500` | Latest records, KPIs, classification counts and latest alert; limit accepts 1–2500 |
| `GET /api/health` | API status and whether the database file exists (`ready` or `waiting`) |

The health endpoint checks database file presence; inspect telemetry and monitor
logs to confirm that detections are being written.

## Controlled traffic campaigns

For isolated tests, stop background traffic first:

```sh
docker compose stop attacker benign_client
docker compose --profile ddos stop ddos_worker_1 ddos_worker_2
```

Then run individual campaigns from PowerShell, for example:

```powershell
./scripts/run_testbed_campaign.ps1 -Attack scan_sparse -Seed 101
./scripts/run_testbed_campaign.ps1 -Attack benign_http -Seed 201
./scripts/run_testbed_campaign.ps1 -Attack ddos_udp -Seed 105
```

Supported activities include Portscan, Botnet, SSH Bruteforce, HTTP/slow/SYN
DoS, two-source DDoS, ordinary and slow benign HTTP, and Other Attack probes.
DDoS campaigns use distinct worker containers with a common campaign ID.
Activity manifests are saved under `data/campaigns/`; they describe generated
traffic independently of model predictions. A successful campaign does not
establish correct classification. Follow the
[testbed guide](testbed/README.md) for preparation, all modes, matched slow
attack/control runs and evidence collection.

## Configuration

Set environment variables in the terminal running Compose, then recreate the
affected services.

| Variable | Default | Purpose |
| --- | --- | --- |
| `IDS_ARTIFACT_ROOT` | `/app/updated_models/five_attack` | Complete model bundle at a path accessible inside both consumers |
| `IDS_STAGE1_THRESHOLD` | Selected bundle's saved threshold (currently 0.10) | Override binary attack gating |
| `IDS_LOG_FEATURES` | `0` | Set to `1` to save numeric model features per snapshot; no packet payloads |
| `IDS_HTTP_BIND` | `127.0.0.1` | Host address for the victim's published HTTP port |

For example, enable diagnostic feature logging in macOS/Linux shells:

```sh
export IDS_LOG_FEATURES=1
docker compose up -d --no-deps --force-recreate monitor
```

Or in PowerShell:

```powershell
$env:IDS_LOG_FEATURES = '1'
docker compose up -d --no-deps --force-recreate monitor
```

After selecting another installed bundle, recreate both `monitor` and
`dashboard`. Clear `IDS_STAGE1_THRESHOLD` to use the selected bundle's own
threshold. Recreate the monitor whenever the victim is recreated so it joins
the current victim network namespace.

## Project layout

```text
backend/                    Packet capture, flow aggregation and hybrid inference
frontend/                   FastAPI telemetry service and dashboard template
testbed/                    Victim, benign client, attacker and monitor images
updated_models/five_attack/ Deployed ML/DL artifacts and evaluation reports
colab_five_attack_retrain/   Colab workflow and candidate bundle exporter
deep learning model/        DL preprocessing, training, evaluation and inference
scripts/                    Campaign, capture and model bundle utilities
tests/                      Runtime, feature, dashboard and testbed checks
data/                       Local SQLite telemetry and campaign evidence
docker-compose.yml          Service definitions and lab networking
```

## Stop or restart

Stop the Docker testbed with:

```powershell
docker compose down
```

To start the testbed again without rebuilding unchanged images, run `docker compose up -d`.

## Troubleshooting

If the dashboard has no records, check monitor startup and shared database
access. Both monitor and dashboard mount `./data`; initial database creation
can take time while models load. Stopping containers preserves this local data.

After changing dashboard source, rebuild its image:

```sh
docker compose up -d --no-deps --build dashboard
```

If a container is restarting, check its logs:

```powershell
docker compose ps
docker compose logs --tail=100 victim monitor dashboard
```

If you see `exec /app/entrypoint.sh: no such file or directory` for `victim`, rebuild it with `docker compose up -d --build victim`. The victim Dockerfile normalizes Windows line endings in its entrypoint during the build.

See [testbed/README.md](testbed/README.md) for campaign commands and
[MODEL_IMPLEMENTATION_COMPARISON.md](MODEL_IMPLEMENTATION_COMPARISON.md)
for the original 1D-CNN comparison, current deployment results and known
slow-client false alarms.
