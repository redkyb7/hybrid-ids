"""Package the newly trained ML and DL deployment artifacts."""

import json
import argparse
import hashlib
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
    "ml/stage1_threshold.json": ML / "stage1_threshold.json",
    "ml/training_metadata.json": ML / "training_metadata.json",
    "dl/nids_model.keras": DL / "nids_model.keras",
    "dl/scaler.pkl": DL / "scaler.pkl",
    "dl/label_encoder.pkl": DL / "label_encoder.pkl",
    "dl/feature_names.pkl": DL / "feature_names.pkl",
    "dl/metadata.json": DL / "metadata.json",
    "dl/thresholds.json": DL / "thresholds.json",
    "dl/evaluation_metrics.json": DL / "evaluation_metrics.json",
    "dl/classification_report.json": DL / "classification_report.json",
}


def verify_current_plan(ml_metadata, dl_metadata, plan_path):
    """Reject mutually matching old artifacts left in a newer extraction."""
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    expected = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    adaptation = dl_metadata.get('lab_adaptation', {})
    if ml_metadata.get('lab_plan_sha256') != expected or adaptation.get('lab_plan_sha256') != expected:
        raise ValueError('Artifacts were not trained with the currently extracted live plan. Retrain both stages in a fresh folder.')
    for split, ml_key, dl_key in (
        ('train', 'lab_train_rows', 'training_snapshots'),
        ('validation', 'lab_validation_rows', 'validation_snapshots')):
        rows = plan['splits'][split]['usable_rows']
        if ml_metadata.get(ml_key) != rows or adaptation.get(dl_key) != rows:
            raise ValueError('Artifact live row counts differ from the current plan')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='five_attack_model_bundle.zip')
    args = parser.parse_args()
    missing = [str(path) for path in FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Train and evaluate both stages first:\n" + "\n".join(missing))

    metadata = json.loads((DL / "metadata.json").read_text(encoding="utf-8"))
    if set(metadata.get("classes", [])) != EXPECTED_CLASSES:
        raise ValueError(f"Expected seven DL classes, got {metadata.get('classes')}")
    ml_metadata = json.loads((ML / "training_metadata.json").read_text(encoding="utf-8"))
    ml_plan = ml_metadata.get("lab_plan_sha256")
    dl_plan = metadata.get("lab_adaptation", {}).get("lab_plan_sha256")
    if not ml_plan or ml_plan != dl_plan:
        raise ValueError("ML and DL artifacts must use the same lab training plan")
    verify_current_plan(ml_metadata, metadata, ROOT / 'lab_data/training_split_plan.json')
    cic_hash = ml_metadata.get('cic_dataset_sha256')
    if not cic_hash or cic_hash != metadata.get('lab_adaptation', {}).get('cic_dataset_sha256'):
        raise ValueError('ML and DL must record the same CIC dataset hash')
    threshold = json.loads((ML / "stage1_threshold.json").read_text(encoding="utf-8"))
    if threshold.get("attack_threshold") != ml_metadata.get("stage1_threshold"):
        raise ValueError("Stage 1 threshold differs from its training metadata")

    destination = (ROOT / args.output).resolve()
    if not destination.is_relative_to(ROOT.resolve()) or destination.suffix != '.zip':
        raise ValueError('Output must be a ZIP inside the extracted training folder')
    if destination.exists():
        raise FileExistsError('Preserve the existing candidate ZIP or choose a new output name')
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in FILES.items():
            archive.write(path, arcname=name)
    print(f"Created {destination}")


if __name__ == "__main__":
    main()
