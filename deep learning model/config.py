# config.py

import os


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(BASE_DIR, "..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "clean_data",
    "cicids2017_cleaned.csv",
)

SAVED_MODEL_DIR = os.path.join(
    BASE_DIR,
    "saved_model",
)

os.makedirs(
    SAVED_MODEL_DIR,
    exist_ok=True,
)


# ============================================================
# SAVED ARTIFACTS
# ============================================================

MODEL_SAVE_PATH = os.path.join(
    SAVED_MODEL_DIR,
    "cnn_nids.keras",
)

SCALER_SAVE_PATH = os.path.join(
    SAVED_MODEL_DIR,
    "scaler.pkl",
)

LABEL_ENCODER_SAVE_PATH = os.path.join(
    SAVED_MODEL_DIR,
    "label_encoder.pkl",
)

FEATURE_NAMES_SAVE_PATH = os.path.join(
    SAVED_MODEL_DIR,
    "feature_names.pkl",
)

X_TEST_SAVE_PATH = os.path.join(
    SAVED_MODEL_DIR,
    "X_test.npy",
)

Y_TEST_SAVE_PATH = os.path.join(
    SAVED_MODEL_DIR,
    "y_test.npy",
)


# ============================================================
# DATASET
# ============================================================

LABEL_COLUMN = "Attack Type"

LABEL_MAP = {

    # Normal
    "Normal Traffic": "Normal",
    "BENIGN": "Normal",
    "Normal": "Normal",

    # DoS
    "DoS": "DoS",

    # DDoS
    "DDoS": "DDoS",

    # Port Scan
    "Port Scanning": "Port Scan",
    "Port Scan": "Port Scan",

    # Brute Force
    "Brute Force": "Brute Force",

    # Web Attack
    "Web Attacks": "Web Attack",
    "Web Attack": "Web Attack",

    # Botnet
    "Bots": "Botnet",
    "Botnet": "Botnet",
}


# ============================================================
# DATA SPLITTING
# ============================================================

# Final split:
# 80% training
# 10% validation
# 10% testing

HOLDOUT_SIZE = 0.20
HOLDOUT_TEST_RATIO = 0.50

RANDOM_STATE = 42


# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================

# Maximum epochs.
# Early stopping will normally finish before this.
EPOCHS = 15

# Large batch is suitable for 2M+ flows.
BATCH_SIZE = 2048

# Initial Adam learning rate.
LEARNING_RATE = 1e-3

# Allow slightly more recovery time than previous run.
EARLY_STOPPING_PATIENCE = 4

# Reduce LR before early stopping.
LR_REDUCTION_PATIENCE = 2

MIN_LEARNING_RATE = 1e-6


# ============================================================
# CLASS WEIGHT CONFIGURATION
# ============================================================

# "balanced" raw class weights were too extreme:
#
# Botnet       ~184
# Web Attack   ~168
#
# Square-root compression reduces extreme weights while
# still giving minority classes additional importance.

USE_SOFTENED_CLASS_WEIGHTS = True


# ============================================================
# TF.DATA
# ============================================================

SHUFFLE_BUFFER_SIZE = 100_000