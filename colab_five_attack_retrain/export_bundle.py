"""Package the newly trained ML and DL deployment artifacts."""

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
ML = ROOT / "updated_models" / "experiments" / "standalone_ml"
DL = ROOT / "deep learning model" / "saved_model"
EXPECTED_CLASSES = {
    "Benign", "Botnet", "Bruteforce", "DDoS", "DoS", "Portscan", "Other Attack"
}
FILES = {
    "ml/stage1_binary_filter.joblib": ML / "stage1_binary_filter.joblib",
    "ml/stage1_feature_list.joblib": ML / "stage1_feature_list.joblib",
    "dl/nids_model.keras": DL / "nids_model.keras",
    "dl/scaler.pkl": DL / "scaler.pkl",
    "dl/label_encoder.pkl": DL / "label_encoder.pkl",
    "dl/feature_names.pkl": DL / "feature_names.pkl",
    "dl/metadata.json": DL / "metadata.json",
    "dl/thresholds.json": DL / "thresholds.json",
    "dl/evaluation_metrics.json": DL / "evaluation_metrics.json",
    "dl/classification_report.json": DL / "classification_report.json",
}


def main() -> None:
    missing = [str(path) for path in FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Train and evaluate both stages first:\n" + "\n".join(missing))

    metadata = json.loads((DL / "metadata.json").read_text(encoding="utf-8"))
    if set(metadata.get("classes", [])) != EXPECTED_CLASSES:
        raise ValueError(f"Expected seven DL classes, got {metadata.get('classes')}")

    destination = ROOT / "five_attack_model_bundle.zip"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in FILES.items():
            archive.write(path, arcname=name)
    print(f"Created {destination}")


if __name__ == "__main__":
    main()
