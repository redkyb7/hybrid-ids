# Shared-split training and evaluation

The separately trained ML and DL models run through the Docker IDS path, but
their reported scores use different train/test splits. Run both stages again
with one split before claiming a combined detection score or selecting a live
threshold. The script below writes a separate bundle and keeps the earlier
uploaded artifacts intact.

## Run in Colab

In a Colab runtime with enough RAM for the 9.17 million-row Parquet file,
enable a GPU if available. From `/content/hybrid-ids`, run:

```python
%cd /content/hybrid-ids
!pip install -q -r "deep learning model/requirements.txt" xgboost joblib
!python scripts/train_shared_models.py --data clean_data/cic-collection.parquet --output updated_models/shared
```

The run hashes the dataset and reads the Parquet file in batches. It groups
rows by 32 fixed ML candidate features, retains one seeded random 57-feature
representative per unambiguous group, and removes groups with conflicting
labels. It then makes one stratified 70/15/15 split; selects the final ML
features using training rows only; trains Random Forest, XGBoost, and the
residual MLP; and chooses the Stage 1 model plus both thresholds using
validation combined attack F1. It evaluates the chosen two-stage rule once on
test rows. Low-confidence DL predictions after the Stage 1 gate count as
`Unknown Attack` alerts, matching the live engine.

A bounded scan of the actual Parquet file found about 8.84 million distinct
57-feature vectors but only 1.465 million distinct 32-feature candidate
vectors. The shared report therefore describes **unique ML behavior groups**,
with one sampled full DL vector each. It does not estimate accuracy weighted
by the original 9.17 million flow frequencies, and its DL result should not
be compared directly with the uploaded raw-flow DL report. The chosen 32
columns and representative policy are recorded in `shared_evaluation.json`.
The same scan found 15,666 groups with conflicting labels; removing them
leaves about 1,449,364 groups. Rare classes become small: roughly 827 Botnet,
968 Bruteforce, 76 Portscan, and 437 Webattack groups before splitting. Treat
their per-class test estimates as uncertain and inspect them individually.

The output directory contains `ml/`, `dl/`, `split_row_ids.npz`, and
`shared_evaluation.json`. Review these report fields before deployment:

- `combined_attack_recall`, `combined_false_positive_rate`, and
  `combined_attack_f1` on the shared test set.
- `by_true_class.*.stage1_forwarded_fraction`: attacks lost at the Stage 1
  gate cannot be recovered by DL.
- `by_true_class.*.combined_alert_fraction` and
  `by_true_class.*.correct_class_fraction`, particularly for Infiltration,
  Portscan, and Webattack.
- `unknown_after_gate_count`: alerts without a reliable attack category.
- `policy`: the model and thresholds selected on validation. Test metrics are
  not used to choose them.

The saved `split_row_ids.npz` contains original Parquet row indices for each
split. `shared_evaluation.json` records the dataset hash and package versions.
The script refuses to overwrite a nonempty output directory. Use a fresh
output path for a second experiment.

To download the complete bundle from Colab:

```python
!zip -q -r shared-models.zip updated_models/shared
from google.colab import files
files.download("shared-models.zip")
```

Place the `updated_models/shared` folder from that ZIP into this repository.
The updated engine reads its saved Stage 1 threshold automatically. The DL
predictor reads its saved confidence threshold.

## Uploaded shared run (24 September 2026)

The Colab bundle is extracted at [`updated_models/shared/`](updated_models/shared/).
Its report records the same SHA-256 hash as the local
`clean_data/cic-collection.parquet`. Validation selected **Random Forest**
with a Stage 1 threshold of **0.65** and a DL confidence threshold of **0.90**.
The saved threshold files agree with the report.

| Shared test measure | Reported value |
| --- | ---: |
| Test groups | 217,405 |
| Combined attack F1 | 0.9675 |
| Combined attack recall | 0.9404 |
| False-positive rate | 0.0504% (96 of 190,302 benign groups) |
| Flows passed to Stage 2 | 25,593 |
| Low-confidence `Unknown Attack` after the gate | 157 |
| Raw DL multiclass macro F1 | 0.6977 |

The combined F1 is a **binary alert** score. The raw DL macro F1 describes
multiclass predictions and is below the 0.70 target used in earlier model
reports. The per-class test results show why the combined score alone is not
sufficient for choosing a live model:

| True class | Test support | Forwarded by Stage 1 | Alerted by combined rule | Correct class |
| --- | ---: | ---: | ---: | ---: |
| Botnet | 124 | 86.29% | 86.29% | 81.45% |
| Bruteforce | 145 | 97.24% | 97.24% | 96.55% |
| DDoS | 10,795 | 99.63% | 99.61% | 98.86% |
| DoS | 14,349 | 99.56% | 99.51% | 99.17% |
| Infiltration | 1,613 | 9.67% | 9.67% | 8.74% |
| Portscan | 12 | 50.00% | 50.00% | 8.33% |
| Webattack | 65 | 70.77% | 70.77% | 67.69% |

These are groups after deduplication by the 32 ML candidate features, with one
representative DL vector per group. They do not measure performance weighted
by the original flow frequencies. Portscan has only 12 test groups, so its
percentage is especially uncertain. The Stage 1 gate misses most Infiltration
groups; Stage 2 cannot recover flows it never receives.

The bundle loads in the monitor image and all 20 tests pass with
`IDS_ARTIFACT_ROOT=/app/updated_models/shared`. That verifies artifact and
runtime compatibility, not the Colab accuracy figures. The monitor image now
uses Python 3.13, scikit-learn 1.6.1, TensorFlow 2.20.0, XGBoost 3.4.1, and
pandas 2.2.3, matching the versions recorded in the shared run. Select this
bundle with `IDS_ARTIFACT_ROOT=/app/updated_models/shared`. Its Infiltration
gap remains a limitation when this bundle is used.

## Check the bundle locally

From PowerShell in the repository root, with Docker Desktop running:

```powershell
docker compose config --quiet
docker run --rm --entrypoint python `
  -e TF_CPP_MIN_LOG_LEVEL=2 `
  -e IDS_ARTIFACT_ROOT=/app/updated_models/shared `
  -v "${PWD}:/app:ro" -w /app hybrid-ids-monitor:latest `
  -m unittest discover -s tests -p test_updated_runtime.py -q
```

The command above checks the shared bundle explicitly. The engine defaults to
`updated_models/extracted`; set `IDS_ARTIFACT_ROOT` to test a different bundle.

The inference benchmark is separate from detection accuracy:

```powershell
docker run --rm --entrypoint python `
  -e IDS_ARTIFACT_ROOT=/app/updated_models/shared `
  -v "${PWD}:/app:ro" -w /app hybrid-ids-monitor:latest `
  scripts/benchmark_updated_runtime.py --runs 100
```

To start the monitor and dashboard with the shared bundle in PowerShell:

```powershell
Remove-Item Env:IDS_STAGE1_THRESHOLD -ErrorAction SilentlyContinue
$env:IDS_ARTIFACT_ROOT = '/app/updated_models/shared'
docker compose up -d --build
docker compose ps
```

The Compose default selects `updated_models/extracted`. The override above
selects the shared bundle, and the dashboard reads its matching
`shared_evaluation.json` automatically. A different
`IDS_STAGE1_THRESHOLD` environment value overrides the saved threshold.

The displayed F1 is the offline shared holdout result, not live accuracy. The
dashboard shows `--` when the selected model bundle has no matching report.

## Current limits

The uploaded DL test array reproduced its archived raw-argmax score in the
monitor image, but its 0.70 confidence policy and the combined Stage 1 gate
were not part of that archived score. The current synthetic packet replay
proves model execution and SQLite persistence; it does not estimate real
traffic accuracy. The live aggregator emits short snapshots, while many
training features came from completed CIC flows, so captured-flow validation
is still required before treating the offline report as live performance.
