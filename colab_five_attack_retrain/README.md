# Expanded five-attack Colab retraining

Upload `five_attack_colab_slow_retrain_checked.zip` to Drive. It includes the original CIC dataset, corrected live snapshots, and the training scripts. Use a fresh Colab runtime/extraction for this bundle to avoid mixing older files.

The live splits contain **879 training snapshots** and **432 validation snapshots**, including verified slow-header attacks and matched legitimate slow clients. The phone validation session stays out of training. Previously inspected attack holdouts are now validation data; collect a fresh final evaluation after choosing a candidate.

## Run in Colab

Choose a high RAM GPU runtime, then run:

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import zipfile, json
run_folder = Path('/content/slow_retrain_run')
assert not run_folder.exists(), 'Choose a fresh folder or a fresh runtime'
with zipfile.ZipFile('/content/drive/MyDrive/five_attack_colab_slow_retrain_checked.zip') as archive:
    archive.extractall(run_folder)
%cd /content/slow_retrain_run/five_attack_retrain
plan = json.loads(Path('lab_data/training_split_plan.json').read_text())
assert plan['splits']['train']['usable_rows'] == 879
assert plan['splits']['validation']['usable_rows'] == 432
print('Verified: 879 training / 432 validation live snapshots')
!pip install -r requirements.txt
```

Restart the runtime after installation, remount Drive, and return to `/content/slow_retrain_run/five_attack_retrain`. Reuse that verified folder; do not extract again over older files.

Start with this **experimental** weight/repetition setting:

```python
%cd /content/slow_retrain_run/five_attack_retrain
!python ml-model-updated.py --full --lab-plan lab_data/training_split_plan.json --lab-weight 10
!python "deep learning model/train.py" --lab-plan lab_data/training_split_plan.json --lab-repeat 10
!python "deep learning model/evaluate.py"
!python export_bundle.py --output candidate_slow_w10_r10.zip
!cp candidate_slow_w10_r10.zip /content/drive/MyDrive/
```

Upload `candidate_slow_w10_r10.zip` back into the repository's `updated_models` folder. Keep the training output and evaluation report. Future uploads are candidate artifacts until explicitly installed. The corrected October 1 weight/repetition-10 run is currently deployed; its slow benign false alarms are documented in `MODEL_IMPLEMENTATION_COMPARISON.md`.

## What changed

- Training combines CIC with independently labeled live traffic, including varied benign clients and all five attack classes.
- The loader verifies hashes, feature order, and separate campaigns and connections between live training and validation.
- ML model selection and DL checkpoint selection use the **lower** of CIC and live validation macro F1, reporting both separately. Test data does not select models.
- CIC DL validation covers seven outputs: Benign, Botnet, Bruteforce, DDoS, DoS, Portscan, Other Attack. Live validation covers the six observed classes. Infiltration and Webattack remain CIC attacks mapped to Other Attack.
- DL threshold tuning matches runtime: a low-confidence benign prediction becomes an Unknown Attack alert. It selects using the lower domain binary attack F1.
- Candidate exports check that ML and DL used the same live split plan and CIC dataset. The archive records dataset, training source, and extractor hashes.

## Experimental limits and next evaluation

The slow pairs share a source IP, request prefix, fragment format and timing settings. Only the legitimate client completes the headers and verifies HTTP 200. Early snapshots can overlap across classes before that difference becomes visible. These are campaign-context labels, not proof that each early snapshot uniquely identifies an attack. Evaluate early and mature predictions separately; adding this data does not guarantee successful slow-DoS classification.

Weight/repetition 10 is a starting setting, not a proven optimum. Planned comparisons are 1, 10, and 100 using these same splits. Save a distinct candidate ZIP before each subsequent run; exports refuse to overwrite files. DL repetition also changes the training data used to fit clipping and scaling, so its effect is not purely a change in loss weighting.

The CIC reference remains a row-based split; ML correlation pruning still happens before splitting. These limitations remain. The CIC standalone DL macro F1 is not the combined pipeline score, and previously inspected live validation is not an untouched final test.

After choosing a candidate using validation, evaluate fresh mixed traffic with frozen models. Report attack recall by class, benign false alerts by client and actual TCP connection, Unknown Attack alerts, Other Attack behavior, and pipeline latency. For future replacements, keep the current deployment until that comparison is reviewed.

Do not upload saved `X_test.npy` or `y_test.npy` as deployment artifacts. Raw PCAPs and final evaluation captures are excluded from this training archive.
