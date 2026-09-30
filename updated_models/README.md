# Deployed models

Docker loads **`five_attack/ml/` and `five_attack/dl/`**. This is the corrected October 1 retrain: Random Forest binary triage followed by a residual MLP. ML attack threshold is 0.10; DL confidence threshold is 0.30.

`deployment_manifest.json` records artifact hashes, training provenance, lossless ML compression verification, and live validation results. See [the model comparison](../MODEL_IMPLEMENTATION_COMPARISON.md) for the original CICIDS2017 / 1D-CNN comparison and current limitations.

Only one deployed bundle is retained. Raw model uploads and extracted candidate copies are removed after installation. Training datasets, PCAPs, evaluation databases, and Colab ZIPs stay local and are excluded from Git.
