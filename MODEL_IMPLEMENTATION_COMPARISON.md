# Original models versus deployed retrained models

Updated October 1, 2026. The corrected slow-traffic retrain is now selected at `updated_models/five_attack/`. Original source and its saved report remain in Git at commit `4f426d6c2a007ee3d7c62e570419bfe8bc3cfc52`. Current artifact hashes and validation results are recorded in `updated_models/deployment_manifest.json`.

## Implementation comparison

| Aspect | Original models | Current deployment |
| --- | --- | --- |
| Dataset | `cicids2017_cleaned.csv` | `cic-collection.parquet` plus independently labeled live traffic |
| Target | `Attack Type` | `ClassLabel`; fine-grained `Label` excluded from inputs |
| ML task | Binary benign/attack triage | Same; all source attack categories remain attacks |
| ML candidates | Random Forest and XGBoost | Same families; selected **Random Forest** |
| ML preprocessing | Numeric filtering, correlation pruning and deduplication before stratified splitting | Retains CIC preprocessing; adds campaign-separated live training rows |
| ML selection | Original `ml_model.py` selects on test macro F1 | Validation selection using the minimum CIC/live binary macro F1 |
| ML adaptation | CICIDS2017 only | 1,036,726 deduplicated CIC training rows plus 879 live rows, live sample weight **10** |
| ML deployed input / gate | Historical schema depends on training path | Saved **20 features**; attack threshold **0.10** |
| DL architecture | **1D-CNN:** Conv1D 32/64/128, pooling, dense 128/64, softmax | **Residual MLP:** flattened inputs, dense 256 stem, residual blocks 256/128, dense 64, softmax |
| DL saved input | 49 features; historical comments/predictor schema were inconsistent | Saved and validated **57 features** |
| DL outputs | Normal, Botnet, Brute Force, DDoS, DoS, Port Scan, Web Attack | Benign, Botnet, Bruteforce, DDoS, DoS, Portscan, **Other Attack** |
| Out-of-scope source attacks | Original canonical mapping | Infiltration/Webattack grouped into Other Attack, never Benign |
| DL preprocessing | Training-fitted RobustScaler | RobustScaler plus saved training-derived percentile clipping |
| DL schedule defaults | 15 epochs, batch 2,048, Adam learning rate 0.001 | 30 epochs, batch 512, Adam learning rate 0.0003 |
| DL imbalance/adaptation | Softened balanced class weights | Configurable softened/clipped weights and live training repeated **10 times** |
| DL checkpoint | Validation loss | Minimum CIC/live validation multiclass macro F1 |
| DL confidence threshold | Historical predictor default 0.60 | Saved **0.30**, tuned using minimum-domain runtime binary attack F1 |
| Runtime | Historical integration had schema/predictor mismatches | Strict validated **ML → DL**, no supplemental classification rules |

Flow statistics are tabular variables, so an MLP is a sensible architecture. Neighboring feature columns do not represent packet time; convolution over them is not temporal packet analysis. Architecture alone does not prove improved generalization.

## How the retrain was made

The CIC collection remains the main training reference. Bounded campaigns and successful benign requests were independently labeled using activity records and packet evidence. Cumulative flow snapshots were extracted through the runtime aggregator, with campaigns and actual connections separated between live training and validation.

The live split contains **879 training / 432 validation snapshots**. Validation supports are Benign 124, Botnet 20, Bruteforce 15, DDoS 20, DoS 241, Portscan 12. ML weights live training by 10; DL repeats it 10 times. Validation guides model/checkpoint/threshold selection and is not fitted as training data. DL repetition also affects training-derived clipping/scaling statistics.

Both exported models record the checked split-plan hash `52185fbfa64c32687419fd7b7ed4431db93ee2c97ac5c4e1dcb98daa65ba34fb`. The exporter rejects stale plan hashes/counts. Model, scaler, feature schema, labels, thresholds and metadata are deployed together. All **432 labeled replay vectors exactly match** the exported live validation vectors, including all 57 features, campaign and label with multiplicities preserved.

The deployed DL metadata records 7,342,854 training samples, 917,190 validation samples and 916,759 test samples; training/validation counts include the live additions/repetitions.

## Offline DL results

| Saved standalone evaluation | Original 1D-CNN | Deployed residual MLP |
| --- | ---: | ---: |
| Dataset | CICIDS2017 CSV | CIC collection Parquet |
| Test samples | 252,076 | 916,759 |
| Classes | 7, including Web Attack | 7, including Other Attack |
| Accuracy | 0.9414 | 0.9806 |
| **Classification macro F1** | **0.5393** | **0.8372** |
| Macro precision | 0.7449 | 0.8202 |
| Macro recall | 0.5045 | 0.8629 |
| Weighted F1 | 0.9236 | 0.9798 |

These saved reports use raw class predictions and are not combined hybrid results. This is **not a controlled architecture comparison**: datasets, labels, preprocessing, sample populations and training changed. The CIC DL row split retains duplicate/related traffic, so its score does not establish independent live performance.

A verified original-CSV ML evaluation report is unavailable here. The later Parquet XGBoost console macro F1 0.9564 is not an original-CSV baseline. Current Random Forest metadata records **0.6108 live / 0.6552 CIC binary validation macro F1** at gate 0.10. Binary triage and multiclass DL scores measure different tasks.

## Combined live pipeline validation

The previous deployment was September 25 **XGBoost + residual MLP**, not the original 1D-CNN. Both bundles were replayed on the same ten previously used validation captures:

| Metric, 432 labeled snapshots | Previous deployment | Current deployment |
| --- | ---: | ---: |
| Six-class pipeline macro F1 | 0.2452 | **0.8650** |
| Exact classification accuracy | 33.33% | 77.55% |
| Attack snapshots alerted | 30/308 (9.74%) | **308/308 (100%)** |
| Benign false alerts | 0/124 | **97/124 (78.23%)** |
| Unknown Attack outputs | 10 | 0 |
| Mean / p95 CPU replay latency | 4.60 / 16 ms | 39.33 / 48 ms |

Current deployment correctly names Botnet 20/20, Bruteforce 15/15, DDoS 20/20, DoS 241/241 and Portscan 12/12. However, **97/101 legitimate slow HTTP snapshots are falsely named DoS**, with false alerts on **all five legitimate slow TCP connections**. Four eventually end with a benign result through ML gating. Ordinary phone traffic has **0/23 false alerts across ten verified connections**.

An earlier adapted candidate scored 0.6767 macro F1 on the original six-session/230-snapshot subset; this deployment scores 1.0000 on that same subset. The historical comparison uses preserved, aligned predictions. Adding slow benign controls exposes errors absent from that smaller subset.

## Deployment decision and limitations

The user selected this retrain for deployment on October 1. It improves recognition of these controlled attacks, with an unresolved tendency to classify legitimate slow clients as DoS. All 97 routed benign false alerts are DL DoS winners, not Unknown fallbacks. Raising DL confidence threshold alone can turn routed predictions into Unknown Attack alerts and does not guarantee fewer false alarms.

- These captures participated in validation/model selection: **no unseen final live-test claim is made**.
- Snapshots are correlated; 308 attack snapshots are not 308 independent attack trials.
- Early slow benign and attack requests have nearly identical unfinished prefixes. Campaign intent can differ before the current flow statistics reveal the distinction. No exact cross-label live feature-vector collisions were found, which does not establish early separability.
- Exact live replay/export equality does not establish CICFlowMeter feature equivalence.
- Bounded unfinished-header attempts alone do not prove victim service denial.
- Replay latency is not a concurrent production throughput benchmark.
- The dashboard displays **standalone DL macro F1 0.8372**, not pipeline validation macro F1 0.8650.

Only the deployed bundle is retained. Random Forest was losslessly compressed from 107,431,561 to **32,080,751 bytes** for GitHub. Every tree state is preserved; predictions and gate decisions match on all 432 validation samples with a fixed reduction order. The manifest records both hashes. Raw captures, training data and diagnostic databases remain local and are excluded from commits.
