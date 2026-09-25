"""Create a self-contained Colab archive without copying the 1 GB dataset."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "five_attack_colab_retrain.zip"
PREFIX = "five_attack_retrain"
SOURCE_FILES = [
    "ml-model-updated.py",
    "colab_five_attack_retrain/README.md",
    "colab_five_attack_retrain/requirements.txt",
    "colab_five_attack_retrain/export_bundle.py",
    "deep learning model/config.py",
    "deep learning model/preprocess.py",
    "deep learning model/model.py",
    "deep learning model/train.py",
    "deep learning model/evaluate.py",
    "deep learning model/predict.py",
]
DATASET = ROOT / "clean_data" / "cic-collection.parquet"


def main() -> None:
    if not DATASET.is_file():
        raise FileNotFoundError(DATASET)
    missing = [name for name in SOURCE_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing training files: {missing}")

    with ZipFile(OUTPUT, "w", allowZip64=True) as archive:
        for name in SOURCE_FILES:
            relative = name.removeprefix("colab_five_attack_retrain/")
            archive.write(ROOT / name, f"{PREFIX}/{relative}", compress_type=ZIP_DEFLATED)
        # Parquet is already compressed. Store it directly to save build time.
        archive.write(DATASET, f"{PREFIX}/clean_data/cic-collection.parquet",
                      compress_type=ZIP_STORED)
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
