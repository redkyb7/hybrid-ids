# config.py
import os

# ── Dynamic Base Paths ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "clean_data", "cicids2017_cleaned.csv")
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(SAVED_MODEL_DIR, "cnn_nids.keras")
SCALER_SAVE_PATH = os.path.join(SAVED_MODEL_DIR, "scaler.pkl")
LABEL_ENCODER_SAVE_PATH = os.path.join(SAVED_MODEL_DIR, "label_encoder.pkl")

# ── Dataset ────────────────────────────────────────────────────────────────
LABEL_COLUMN = "Attack Type"

# Map your dataset's labels to canonical class names
LABEL_MAP = {
    "Normal Traffic": "Normal",
    "BENIGN": "Normal",
    "DoS": "DoS",
    "DDoS": "DDoS",
    "Port Scanning": "Port Scan",
    "Port Scan": "Port Scan",
    "Brute Force": "Brute Force",
    "Web Attacks": "Web Attack",
    "Web Attack": "Web Attack",
    "Bots": "Botnet",
    "Botnet": "Botnet",
}

# ── Preprocessing ──────────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Model Hyperparameters ──────────────────────────────────────────────────
EPOCHS = 30
BATCH_SIZE = 2048       # Large batch for CICIDS2017 scale
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5
