# SentinelFlow Hybrid IDS

SentinelFlow runs a Docker testbed with a victim web server, a benign traffic generator, an attacker, an IDS monitor, and a dashboard. The monitor captures traffic to the victim, classifies flows, and writes results to `data/ids_logs.db`. The dashboard reads that database and the selected DL model's offline evaluation metrics.

## Prerequisites

- Docker Desktop running with Linux containers (or Docker Engine with Compose on Linux).

Run the commands below from the repository root, where `docker-compose.yml` is located.

## Start the Docker testbed

```powershell
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

## Open the dashboard

The dashboard starts with `docker compose up -d --build`; no host Python setup
is needed. Open <http://localhost:8000>. It displays live detections and the
selected DL model's offline classification macro F1 when its report is present.
Threat and flow totals include historical
rows already in `data/ids_logs.db`; the classification chart uses the latest
500 snapshots so new detections are visible without historical dilution.

If the dashboard cannot read telemetry, check `docker compose logs dashboard`
and `docker compose logs monitor`. Both containers mount the same `./data`
directory.

## Stop or restart

Stop the Docker testbed with:

```powershell
docker compose down
```

To start the testbed again without rebuilding unchanged images, run `docker compose up -d`.

## Troubleshooting

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
