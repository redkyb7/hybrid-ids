"""Install the raw Colab ML/DL ZIP outputs as a Docker artifact bundle."""

import json
from pathlib import Path
from shutil import copyfileobj
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1] / "updated_models" / "five_attack"
EXPECTED_CLASSES = {
    "Benign", "Botnet", "Bruteforce", "DDoS", "DoS", "Portscan", "Other Attack"
}
ML_FILES = ("stage1_binary_filter.joblib", "stage1_feature_list.joblib")
DL_FILES = (
    "nids_model.keras", "scaler.pkl", "label_encoder.pkl", "feature_names.pkl",
    "metadata.json", "thresholds.json", "evaluation_metrics.json",
    "classification_report.json",
)


def one_archive(pattern: str) -> Path:
    matches = sorted(ROOT.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} in {ROOT}; found {len(matches)}")
    return matches[0]


def extract_selected(archive_path: Path, prefix: str, names: tuple[str, ...],
                     destination: Path) -> None:
    with ZipFile(archive_path) as archive:
        required = {f"{prefix}/{name}" for name in names}
        missing = required - set(archive.namelist())
        if missing:
            raise ValueError(f"{archive_path.name} lacks {sorted(missing)}")
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            with archive.open(f"{prefix}/{name}") as source, \
                    (destination / name).open("wb") as target:
                copyfileobj(source, target)
            print(f"Installed {destination.name}/{name}")


def main() -> None:
    ml_archive = one_archive("standalone_ml*.zip")
    dl_archive = one_archive("saved_model*.zip")
    with ZipFile(dl_archive) as archive:
        metadata = json.load(archive.open("saved_model/metadata.json"))
        if set(metadata.get("classes", [])) != EXPECTED_CLASSES:
            raise ValueError(f"Unexpected DL classes: {metadata.get('classes')}")
        if metadata.get("num_features") != 57:
            raise ValueError("Expected 57 DL features")
    extract_selected(ml_archive, "standalone_ml", ML_FILES, ROOT / "ml")
    extract_selected(dl_archive, "saved_model", DL_FILES, ROOT / "dl")
    print("Five-attack bundle installed. Run the runtime smoke test before deployment.")


if __name__ == "__main__":
    main()
