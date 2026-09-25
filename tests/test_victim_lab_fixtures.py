"""Exercise bounded HTTP fixtures and synthetic canary access."""

import hashlib
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("victim_lab_app", ROOT / "testbed/victim/app.py")
victim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(victim)


class TestVictimLabFixtures(unittest.TestCase):
    def setUp(self):
        self.client = victim.app.test_client()

    def test_bounded_response_size_and_delay(self):
        response = self.client.get("/lab/load?size=935&delay_ms=0")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 935)
        self.assertEqual(self.client.get("/lab/load?size=4096").status_code, 400)
        self.assertEqual(self.client.get("/lab/c2?size=129").status_code, 200)

    def test_canary_requires_lab_token_and_accepts_only_synthetic_data(self):
        self.assertEqual(self.client.get("/lab/canary").status_code, 401)
        bad = self.client.post("/lab/auth", data={"username": "lab", "password": "wrong"})
        self.assertEqual(bad.status_code, 401)
        good = self.client.post(
            "/lab/auth", data={"username": "lab", "password": "lab-only-password"}
        )
        token = good.get_json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        canary = self.client.get("/lab/canary?size=1024", headers=headers)
        self.assertEqual(len(canary.data), 1024)
        self.assertEqual(self.client.get("/lab/canary?size=5000", headers=headers).status_code, 400)
        rejected = self.client.post("/lab/collect", data=b"real-data", headers=headers)
        self.assertEqual(rejected.status_code, 400)
        accepted = self.client.post("/lab/collect", data=canary.data, headers=headers)
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(
            accepted.get_json()["sha256"], hashlib.sha256(canary.data).hexdigest()
        )


if __name__ == "__main__":
    unittest.main()
