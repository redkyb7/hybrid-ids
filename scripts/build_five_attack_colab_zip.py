"""Create a CIC plus lab-adaptation Colab archive without the final holdout."""

import hashlib
import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "five_attack_colab_slow_retrain_checked.zip"
PREFIX = "five_attack_retrain"
SOURCE_FILES = [
    "ml-model-updated.py",
    "lab_training_data.py",
    "colab_five_attack_retrain/README.md",
    "colab_five_attack_retrain/requirements.txt",
    "colab_five_attack_retrain/export_bundle.py",
    "deep learning model/config.py",
    "deep learning model/preprocess.py",
    "deep learning model/model.py",
    "deep learning model/train.py",
    "deep learning model/evaluate.py",
    "deep learning model/predict.py",
    "deep learning model/validation_policy.py",
]
DATASET = ROOT / "clean_data" / "cic-collection.parquet"
LAB_PLAN = ROOT / "data/labeled_testbed/expanded-slow-live-20260930/training_split_plan.json"
LAB_SPLITS = LAB_PLAN.parent


def main() -> None:
    global OUTPUT, LAB_PLAN, LAB_SPLITS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab-plan', type=Path, default=LAB_PLAN)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    LAB_PLAN, OUTPUT = args.lab_plan.resolve(), args.output.resolve()
    LAB_SPLITS = LAB_PLAN.parent
    if not LAB_PLAN.is_relative_to(ROOT / 'data/labeled_testbed') or not OUTPUT.is_relative_to(ROOT) or OUTPUT.suffix != '.zip':
        raise ValueError('Plan must be under data/labeled_testbed and output a repository ZIP')
    if not DATASET.is_file():
        raise FileNotFoundError(DATASET)
    missing = [name for name in SOURCE_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing training files: {missing}")
    split_plan = json.loads(LAB_PLAN.read_text(encoding="utf-8"))
    if set(split_plan.get("splits", {})) != {"train", "validation"}:
        raise ValueError("Expected training/validation only; final evaluation stays outside")
    if OUTPUT.exists():
        raise FileExistsError(f"Preserve the prior bundle: {OUTPUT}")
    training_plan = {
        "schema_version": 1,
        "feature_schema_sha256": split_plan["feature_schema_sha256"],
        "features": split_plan["features"],
        "splits": {},
    }
    for name in ("train", "validation"):
        source = LAB_SPLITS / f"{name}.parquet"
        stats = split_plan["splits"][name]
        if hashlib.sha256(source.read_bytes()).hexdigest() != stats["parquet_sha256"]:
            raise ValueError(f"Lab {name} Parquet hash differs from split plan")
        training_plan["splits"][name] = {
            "parquet": f"{name}.parquet",
            "parquet_sha256": stats["parquet_sha256"],
            "usable_rows": stats["usable_rows"],
            "campaign_ids": stats["campaign_ids"],
            "connection_ids": stats["connection_ids"],
        }

    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Remove the incomplete prior build: {temporary}")
    with ZipFile(temporary, "w", allowZip64=True) as archive:
        for name in SOURCE_FILES:
            relative = name.removeprefix("colab_five_attack_retrain/")
            archive.write(ROOT / name, f"{PREFIX}/{relative}", compress_type=ZIP_DEFLATED)
        # Parquet is already compressed. Store it directly to save build time.
        archive.write(DATASET, f"{PREFIX}/clean_data/cic-collection.parquet",
                      compress_type=ZIP_STORED)
        for name in ("train", "validation"):
            archive.write(LAB_SPLITS / f"{name}.parquet",
                          f"{PREFIX}/lab_data/{name}.parquet",
                          compress_type=ZIP_STORED)
        archive.writestr(
            f"{PREFIX}/lab_data/training_split_plan.json",
            json.dumps(training_plan, indent=2) + "\n",
            compress_type=ZIP_DEFLATED,
        )
        archive.writestr(f"{PREFIX}/bundle_manifest.json", json.dumps({
            "schema_version": 1,
            "cic_dataset_sha256": file_digest(DATASET),
            "extractor_sha256": split_plan["extractor_sha256"],
            "live_source_plan_sha256": file_digest(LAB_PLAN),
            "sources_sha256": {name: file_digest(ROOT / name) for name in SOURCE_FILES},
            "final_evaluation_status": split_plan["final_evaluation_status"],
        }, indent=2) + "\n")
    temporary.replace(OUTPUT)
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
