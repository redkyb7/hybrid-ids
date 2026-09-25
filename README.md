# SentinelFlow Hybrid IDS

SentinelFlow runs a Docker testbed with a victim web server, a benign traffic generator, an attacker, an IDS monitor, and a dashboard. The monitor captures traffic to the victim, classifies flows, and writes results to `data/ids_logs.db`. The dashboard reads that database and, when available, a combined holdout report for the selected model bundle.

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

The monitor defaults to the earlier uploaded XGBoost ML and residual-MLP DL
models in `updated_models/extracted/{ml,dl}`. Stage 1 uses the engine's 0.10
attack threshold because this ML bundle has no saved threshold file. Stage 2
uses its saved 0.70 confidence threshold. The 0.10 Stage 1 threshold was not
evaluated in the original Colab report. A flow below that threshold stops at
Stage 1; the supplemental rules can still raise an alert. If an artifact is
missing or has the wrong schema, the monitor exits and reports the error in
`docker compose logs monitor`.

The monitor image includes the Python, scikit-learn, TensorFlow, XGBoost, and
pandas dependencies needed by these models.

The deployed ML and DL models were evaluated on different test splits, so
there is no measured combined attack F1 for this pairing. The dashboard shows
the DL model's standalone classification macro F1 (`0.7737`) from
`updated_models/extracted/dl/evaluation_metrics.json`. This is an offline
eight-class test-set result, not a combined pipeline or live-traffic score.
Their standalone scores cannot be combined into one pipeline score.

The current attacker evaluation covers 13 bounded scenarios across seven
attack classes using the deployed bundle. It records Stage 2 reach, model
classifications, rule alerts, and feature comparisons with CIC data. See
[ATTACKER_IMPLEMENTATION_AND_EVALUATION.md](ATTACKER_IMPLEMENTATION_AND_EVALUATION.md)
for the measured results and limitations.

Rule alerts are labeled separately from ML/DL verdicts in the dashboard's log
details. The displayed DL macro F1 does not measure Stage 1 gating or the
supplemental checks. Set
`IDS_SUPPLEMENTAL_RULES=0` before
recreating the monitor to disable those checks. Set `IDS_LOG_FEATURES=1` to
store the numeric Stage 1 and Stage 2 flow features per snapshot for a
diagnostic run; it is off by default and does not store packet payloads.
Recreate the monitor after
changing either setting with `docker compose up -d --no-deps monitor`.

Set `IDS_ARTIFACT_ROOT` before running Compose to select another complete
bundle mounted inside the monitor and dashboard containers. The dashboard
displays `--` unless that bundle has a matching
`dl/evaluation_metrics.json` report.
Clear any `IDS_STAGE1_THRESHOLD` override to use the threshold belonging to
the selected bundle.

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
[ATTACKER_IMPLEMENTATION_AND_EVALUATION.md](ATTACKER_IMPLEMENTATION_AND_EVALUATION.md)
for the measured seven-class live evaluation.
