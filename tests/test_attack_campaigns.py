"""Checks for lab-only targets, label mapping, budgets and UDP action limits."""

import json
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "testbed" / "attacker"))
import attack_campaigns
from campaign_runtime import BudgetExceeded, Campaign, Profile, VICTIM_IP, load_profiles


class TestAttackCampaigns(unittest.TestCase):
    def test_profiles_cover_every_broad_attack_and_correct_login_mapping(self):
        profiles = load_profiles()
        self.assertEqual(set(profiles), set(attack_campaigns.HANDLERS))
        self.assertEqual(
            {value.class_label for value in profiles.values()},
            {"Botnet", "Bruteforce", "DDoS", "DoS", "Infiltration", "Portscan", "Webattack"},
        )
        self.assertEqual(profiles["ssh_bruteforce"].reference_label, "Bruteforce-SSH")
        self.assertEqual(profiles["web_login"].reference_label, "Webattack-bruteforce")
        self.assertNotIn("password123", attack_campaigns.INVALID_PASSWORDS)
        self.assertEqual(
            {profiles[mode].class_label for mode in attack_campaigns.ALL_LOCAL_MODES},
            {"Botnet", "Bruteforce", "DoS", "Portscan"},
        )

    def test_profile_rejects_zero_or_negative_action_parameters(self):
        source = json.loads((ROOT / "testbed/attacker/profiles.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profiles.json"
            for mode, key, invalid in (
                ("botnet", "count", 0),
                ("ddos_udp", "packets_per_flow", -1),
                ("dos_slow", "fragment_interval_seconds", 0),
            ):
                changed = json.loads(json.dumps(source))
                changed["profiles"][mode]["parameters"][key] = invalid
                profile_path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(mode=mode, key=key), self.assertRaises(ValueError):
                    load_profiles(profile_path)

    def test_budget_and_manifest(self):
        profile = Profile("test", "Portscan", "Portscan", "lab-test", 5, 2, 100, {})
        with tempfile.TemporaryDirectory() as directory:
            campaign = Campaign(profile, "test-1", 42, "worker-1", Path(directory))
            campaign.reserve(actions=2, estimated_outbound_bytes=50)
            with self.assertRaises(BudgetExceeded):
                campaign.reserve(actions=1)
            campaign.record("two_ports", time.time(), success=True, actions_completed=2)
            manifest = json.loads(campaign.write_manifest().read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["actions_attempted"], 2)
            self.assertEqual(manifest["actions_succeeded"], 2)
            self.assertEqual(manifest["events"][0]["actions_completed"], 2)
            self.assertEqual(manifest["target_ip"], VICTIM_IP)

    def test_http_action_cannot_leave_fixed_victim(self):
        profile = Profile("test", "Webattack", None, "lab-test", 5, 1, 500, {})
        with tempfile.TemporaryDirectory() as directory:
            campaign = Campaign(profile, "test-2", 42, "worker-1", Path(directory))
            with self.assertRaises(ValueError):
                attack_campaigns.http_action(campaign, "GET", "//other.example", "probe")
            fake = SimpleNamespace(status_code=200, content=b"ok")
            with patch.object(attack_campaigns.requests, "request", return_value=fake) as send:
                response = attack_campaigns.http_action(campaign, "GET", "/lab/c2", "probe")
            self.assertIs(response, fake)
            self.assertEqual(send.call_args.args[1], f"http://{VICTIM_IP}/lab/c2")
            self.assertEqual(campaign.actions_succeeded, 1)

    def test_udp_sends_only_budgeted_packets_to_lab_sink(self):
        profiles = load_profiles()
        with tempfile.TemporaryDirectory() as directory:
            campaign = Campaign(profiles["ddos_udp"], "test-3", 42, "worker-1", Path(directory))
            campaign.pause = lambda seconds: None
            destinations = []

            class FakeSocket:
                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    return False

                def sendto(self, payload, destination):
                    destinations.append((len(payload), destination))

            with patch.object(attack_campaigns.socket, "socket", return_value=FakeSocket()):
                attack_campaigns.ddos_udp(campaign, random.Random(42))
            self.assertEqual(campaign.actions_attempted, 40)
            self.assertEqual(campaign.actions_succeeded, 40)
            self.assertEqual(set(destinations), {(389, (VICTIM_IP, 9999))})


if __name__ == "__main__":
    unittest.main()
