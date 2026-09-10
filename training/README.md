# Model Training & Pipeline Documentation

This directory contains the training pipelines for the **SentinelFlow Two-Stage Hybrid IDS**.

---

## 1. Directory Structure

```
training/
├── train_stage1_binary.py       # Stage 1 Binary Triage Model Training (Random Forest)
├── train_stage2_multiclass.py   # Stage 2 Multi-Class Attack Classifier Training (Random Forest)
└── README.md                    # Pipeline documentation & experiment guide
```

---

## 2. Training Scripts Overview

### A. Stage 1 Binary Triage (`train_stage1_binary.py`)
- **Objective:** Ultra-fast binary filtering of benign traffic ($< 35\text{ ms}$) to discard $>80-99\%$ of non-threatening network flows before escalating to Stage 2.
- **Dataset:** 2.52 Million flows from `clean_data/cicids2017_cleaned.csv`.
- **Features:** 48 Pure Behavioral Features (strictly port-invariant and window-size invariant).
- **Holdout ROC-AUC:** `0.99981` (Holdout Accuracy: `99.75%`).
- **Generated Artifacts:**
  - `backend/models/stage1_binary_filter.joblib`
  - `backend/models/stage1_feature_list.joblib`

### B. Stage 2 Multi-Class Attack Classifier (`train_stage2_multiclass.py`)
- **Objective:** Deep threat dissection and multi-class categorization across all attack families.
- **Dataset:** 425,694 attack flows from `clean_data/cicids2017_cleaned.csv`.
- **Classes (6):** `DoS`, `DDoS`, `Port Scan`, `Brute Force`, `Web Attack`, `Botnet`.
- **Features:** Identical 48 behavioral features matching Stage 1.
- **Holdout Macro F1:** `98.78%` across 85,139 holdout flows.
- **Generated Artifacts:**
  - `backend/models/stage2_attack_classifier.joblib`
  - `backend/models/stage2_attack_classes.joblib`

---

## 3. How to Retrain Models

Execute directly from the repository root:

```bash
# Retrain Stage 1 Binary Filter
python training/train_stage1_binary.py

# Retrain Stage 2 Multi-Class Classifier
python training/train_stage2_multiclass.py
```
