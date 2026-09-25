# Five attack class retraining bundle

This archive trains the two stage IDS on the CIC collection Parquet dataset.

| Stage | Target |
| --- | --- |
| ML | Benign versus **every** attack row |
| DL | Benign, Botnet, Bruteforce, DDoS, DoS, Portscan, Other Attack |

The DL target maps source `Infiltration` and `Webattack` rows to `Other Attack` before splitting. Those rows remain attacks. The five named classes are the project focus. This is still a per-flow classifier; changing labels alone does not fix the live Portscan feature mismatch.

## Run in Colab

Upload `five_attack_colab_retrain.zip` to your Drive. Choose a GPU runtime with enough RAM for 9.17 million flow rows. In a Colab notebook, run:

```python
from google.colab import drive
drive.mount('/content/drive')
!unzip -q /content/drive/MyDrive/five_attack_colab_retrain.zip -d /content
%cd /content/five_attack_retrain
!pip install -r requirements.txt
```

Restart the Colab runtime after installing packages, then remount Drive and return to `/content/five_attack_retrain`. If a full run runs out of RAM, use a high RAM runtime. The ML script's default sample is useful for a smoke test, but the final run should use `--full`.

```python
%cd /content/five_attack_retrain
!python ml-model-updated.py --full
!python "deep learning model/train.py"
!python "deep learning model/evaluate.py"
!python export_bundle.py
!cp five_attack_model_bundle.zip /content/drive/MyDrive/
```

The training programs save the ML model under `updated_models/experiments/standalone_ml/` and the DL model under `deep learning model/saved_model/`. `export_bundle.py` checks that the DL output has seven classes and includes only deployment artifacts and the DL evaluation report in `five_attack_model_bundle.zip`. It excludes the large saved test arrays.

Download that resulting ZIP from Drive and extract its `ml/` and `dl/` directories into the repo's `updated_models/five_attack/` directory. Docker Compose now selects this bundle by default. If you downloaded the separate raw `standalone_ml*.zip` and `saved_model*.zip` archives, place them under `updated_models/five_attack/` and run `python scripts/install_five_attack_archives.py` from the repo root to extract only deployment files. Recreate the monitor and dashboard afterward. The Stage 1 script does not calibrate a decision threshold; the runtime's current 0.10 fallback is an experiment setting and must be evaluated with the new pairing. The DL macro F1 in the report is a standalone score, not a combined pipeline score.

Do not upload `X_test.npy` or `y_test.npy` as deployment artifacts. Keep them in Colab only if you need to rerun the standalone evaluation.
