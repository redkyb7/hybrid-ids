"""Ensure exporting a new dataset cannot silently reuse matching old models."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export_bundle', ROOT / 'colab_five_attack_retrain/export_bundle.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class ExportPlanTest(unittest.TestCase):
    def test_old_pair_rejected_and_current_pair_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'plan.json'
            path.write_text(json.dumps({'splits': {'train': {'usable_rows': 879}, 'validation': {'usable_rows': 432}}}))
            current = hashlib.sha256(path.read_bytes()).hexdigest()
            ml = {'lab_plan_sha256': 'old', 'lab_train_rows': 513, 'lab_validation_rows': 230}
            dl = {'lab_adaptation': {'lab_plan_sha256': 'old', 'training_snapshots': 513, 'validation_snapshots': 230}}
            with self.assertRaisesRegex(ValueError, 'currently extracted live plan'):
                export.verify_current_plan(ml, dl, path)
            ml.update(lab_plan_sha256=current, lab_train_rows=879, lab_validation_rows=432)
            dl['lab_adaptation'].update(lab_plan_sha256=current, training_snapshots=879, validation_snapshots=432)
            export.verify_current_plan(ml, dl, path)
            dl['lab_adaptation']['training_snapshots'] = 513
            with self.assertRaisesRegex(ValueError, 'row counts'):
                export.verify_current_plan(ml, dl, path)


if __name__ == '__main__':
    unittest.main()
