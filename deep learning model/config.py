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
    "cic-collection.parquet",
)

SAVED_MODEL_DIR = os.path.join(
    BASE_DIR,
    "saved_model",
)

# ============================================================
# SAVED ARTIFACTS
# ============================================================

MODEL_SAVE_PATH = os.path.join(
    SAVED_MODEL_DIR,
    "nids_model.keras",
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

METADATA_SAVE_PATH = os.path.join(
    SAVED_MODEL_DIR,
    "metadata.json",
)

THRESHOLDS_SAVE_PATH = os.path.join(
    SAVED_MODEL_DIR,
    "thresholds.json",
)

X_TEST_SAVE_PATH = os.path.join(
    SAVED_MODEL_DIR,
    "X_test.npy",
)

Y_TEST_SAVE_PATH = os.path.join(
    SAVED_MODEL_DIR,
    "y_test.npy",
)

CLASSIFICATION_REPORT_SAVE_PATH = os.path.join(
    SAVED_MODEL_DIR,
    "classification_report.json",
)

METRICS_SAVE_PATH = os.path.join(
    SAVED_MODEL_DIR,
    "evaluation_metrics.json",
)

CONFUSION_MATRIX_SAVE_PATH = os.path.join(
    SAVED_MODEL_DIR,
    "confusion_matrix.png",
)

NORMALIZED_CONFUSION_MATRIX_SAVE_PATH = os.path.join(
    SAVED_MODEL_DIR,
    "confusion_matrix_normalized.png",
)


# ============================================================
# DATASET
# ============================================================

# Your Parquet dataset has:
#
# - Label: 33 fine-grained attack labels
# - ClassLabel: 8 canonical attack categories
#
# We train on the already curated ClassLabel target.
LABEL_COLUMN = "ClassLabel"

# The fine-grained label is retained in the Parquet source, but excluded
# because it directly reveals a more specific version of the target label.
LEAKAGE_COLUMNS = [
    "Label",
]

# Your inspected Parquet data has 57 numeric feature columns and two
# categorical label columns. No extra feature exclusion is needed.
EXCLUDE_FEATURES = LEAKAGE_COLUMNS

# These are the exact expected target categories in your data.
EXPECTED_CLASSES = [
    "Benign",
    "Botnet",
    "Bruteforce",
    "DDoS",
    "DoS",
    "Infiltration",
    "Portscan",
    "Webattack",
]

# Retain strict validation so accidental target/schema changes are detected.
STRICT_CLASS_VALIDATION = True

# Exact duplicate removal is expensive on a 9.1M-row dataset.
# Set True only if you have plenty of RAM and want to remove duplicates.
REMOVE_DUPLICATES = False


# ============================================================
# DATA SPLITTING
# ============================================================

# 80% train, 10% validation, 10% test
HOLDOUT_SIZE = 0.20
HOLDOUT_TEST_RATIO = 0.50

RANDOM_STATE = 42


# ============================================================
# PREPROCESSING
# ============================================================

# Your inspection showed no NaN or infinity values. Clipping is still
# beneficial for heavily skewed flow-statistics values.
USE_PERCENTILE_CLIPPING = True

LOWER_CLIP_PERCENTILE = 0.5
UPPER_CLIP_PERCENTILE = 99.5


# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================

EPOCHS = 30

# Start at 512. If macOS runs out of memory, reduce to 256 or 128.
BATCH_SIZE = 512

LEARNING_RATE = 3e-4

EARLY_STOPPING_PATIENCE = 6

LR_REDUCTION_PATIENCE = 2

MIN_LEARNING_RATE = 1e-6

L2_REGULARIZATION = 1e-5

DROPOUT_INPUT = 0.30
DROPOUT_BLOCK_1 = 0.25
DROPOUT_BLOCK_2 = 0.20


# ============================================================
# LOSS / CLASS IMBALANCE
# ============================================================

USE_SOFTENED_CLASS_WEIGHTS = True

# Square-root compression of balanced class weights.
CLASS_WEIGHT_POWER = 0.5

MIN_CLASS_WEIGHT = 0.25
MAX_CLASS_WEIGHT = 10.0

# Leave False for the first training run.
# Test focal loss later as a separate experiment.
USE_FOCAL_LOSS = False

FOCAL_GAMMA = 2.0


# ============================================================
# INFERENCE
# ============================================================

DEFAULT_UNKNOWN_THRESHOLD = 0.60

TUNE_UNKNOWN_THRESHOLD = True

THRESHOLD_GRID = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


# ============================================================
# TF.DATA / REPRODUCIBILITY
# ============================================================

SHUFFLE_BUFFER_SIZE = 100_000

# Keep False unless deterministic repeatability is required.
ENABLE_DETERMINISTIC_OPS = False
