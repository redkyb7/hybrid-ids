# Attacker implementation and live testbed evaluation

25 September 2026

> Historical baseline: the 13-scenario measurements below used the former
> rule-assisted monitor. The current monitor records only ML/DL verdicts.
> A two-campaign model-only follow-up appears at the end of this report.

## What was implemented

The attack node now has 13 explicit, bounded scenarios covering the seven CIC
attack classes. [`testbed/attacker/profiles.json`](testbed/attacker/profiles.json)
records each class, source subtype, scenario name, action limit, time limit,
estimated outbound-byte limit, and parameters. The campaign runner fixes the
only target at `192.168.100.10`, applies the limits at runtime, and writes an
action-level JSON manifest with seed, times, worker and source IP. The default
continuous attacker also ends after 20 campaigns or 300 seconds; Compose does
not restart it into another run.

The victim now provides small lab-only response-size/delay, C2 check-in,
failed-login, synthetic canary and non-amplifying UDP fixtures. The optional
Compose DDoS profile starts two real containers at `.67` and `.68` with one
campaign ID. No source IP spoofing or external destination is used. SSH
failures represent **Bruteforce**; HTTP form failures are labeled
**Webattack-bruteforce**. The canary is generated in memory and can only be
sent to the local collection fixture.

[`scripts/run_testbed_campaign.ps1`](scripts/run_testbed_campaign.ps1) runs one
isolated scenario and restores the prior generator state.
[`scripts/evaluate_attack_campaigns.py`](scripts/evaluate_attack_campaigns.py)
joins manifests to monitor SQLite verdicts and distinguishes ML/DL detections
from supplemental rules. With `IDS_LOG_FEATURES=1`, the monitor records the
numeric flow vector and first/last packet times for comparison. The evaluator
uses the fullest snapshot per 5-tuple for feature comparison: some closed TCP
connections produced a later, one-packet teardown record. Every verdict still
counts toward the snapshot and alert totals.

## Controlled measurements

Victim and monitor were running; the continuous attacker and benign generator
were stopped during each attack run. The monitor used the deployed model bundle
in `updated_models/extracted` with supplemental rules enabled and full numeric
feature logging. Every run below completed its configured actions and produced
victim-bound monitor records. The DDoS runs each showed **two distinct source
IPs** in captured traffic. Counts are per distinct 5-tuple except `Actions`,
`Snapshots`, and the latency percentile. A connection is counted as reaching
Stage 2, receiving a correct model class, or receiving an alert if **any** of
its snapshots did so. `Rule only` means the alert came from a supplemental
rule with a benign model verdict. `Feature band` is the number of 26 selected
flow-feature medians inside the raw source subtype's approximate 10th–90th
percentile band; it is a diagnostic, not a similarity score.

| Scenario | Actions | Sources | Connections / snapshots | Stage 2 connections | Correct model class | Rule only | Final alert connections | Feature band | p95 engine ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `scan` | 12/12 ports | 1 | 12 / 12 | 10 | 0 | 7 | 7 | 21/26 | 40.25 |
| `botnet` | 8/8 check-ins | 1 | 8 / 24 | 0 | 0 | 8 | 8 | 5/26 | 3.85 |
| `ssh_bruteforce` | 5/5 rejected logins | 1 | 5 / 12 | 0 | 0 | 0 | 0 | 4/26 | 3.45 |
| `web_login` | 8/8 failed forms | 1 | 8 / 24 | 0 | 0 | 4 | 4 | 17/26 | 3.00 |
| `web_sqli` | 5/5 probes | 1 | 5 / 15 | 0 | 0 | 5 | 5 | 17/26 | 4.00 |
| `web_xss` | 3/3 probes | 1 | 3 / 9 | 0 | 0 | 3 | 3 | 18/26 | 3.00 |
| `dos_http` | 40/40 requests | 1 | 40 / 120 | 0 | 0 | 0 | 0 | 19/26 | 4.05 |
| `dos_slow` | 4/4 connections | 1 | 4 / 140 | 4 | 0 | 0 | 0 | 10/26 | 17.05 |
| `dos_syn` | 60/60 SYNs | 1 | 60 / 60 | 0 | 0 | 41 | 41 | n/a | 3.00 |
| `ddos_http_loic` | 24/24 requests | 2 | 24 / 66 | 0 | 0 | 0 | 0 | 12/26 | 4.00 |
| `ddos_http_hoic` | 40/40 requests | 2 | 40 / 120 | 0 | 0 | 0 | 0 | 17/26 | 4.00 |
| `ddos_udp` | 80/80 datagrams | 2 | 20 / 20 | 20 | **20 DDoS** | 0 | 20 | 19/26 | 15.15 |
| `infiltration` | 4/4 canary steps | 1 | 4 / 12 | 0 | 0 | 0 | 0 | 18/26 | 5.70 |

A separate **35-second benign-only** interval from `.101` recorded 18
connections and 54 snapshots, with 0 Stage 2 snapshots, 0 model alerts, 0
rule alerts and 0 final alerts; p95 engine latency was 4 ms. This is a short
smoke baseline, not a false-positive estimate for broader benign traffic. Its
SQLite records are IDs 53404–53457 (2026-09-25 08:31:48–08:32:25 UTC).
Likewise, the deliberately selected attack scenarios and repeated snapshots
do not support a population precision, recall, macro F1, or combined F1 score.
The reported latency measures model/rule processing after a flow snapshot is
emitted, not the time from the attack's first packet to an alert.

### What the feature measurements say

- **UDP DDoS** reproduced the key one-way flow shape: live median 4 forward,
  0 backward packets and 389-byte maximum packet, matching the CIC DDoS-UDP
  medians. It was the only scenario in this run correctly classified by the
  deployed ML → DL path (20 of 20 observed connections).
- **LOIC-like HTTP DDoS** was tuned from a 0.20-second to a 1.50-second lab
  response. Its final live median duration was 1.50 seconds, within the CIC
  subtype's 1.02–34.24-second reference band. Live packet counts were still
  6/5 versus about 3/3 in CIC, and all flows stopped at Stage 1.
- **HOIC-like HTTP DDoS** matched the subtype duration closely (12.5 versus
  11.7 ms) and the 935-byte maximum packet. It still had 6/5 versus 3/4
  forward/backward packets and no Stage 2 reach.
- **HTTP DoS** produced 935-byte response packets, equal to the CIC DoS-Hulk
  median maximum. Its median duration was 22 versus 203 ms, and its initial
  forward TCP window was 64,240 versus 225 bytes in the source rows. Stage 1
  marked the captured snapshots benign.
- **Botnet check-ins** matched the narrow subtype's duration (11.7 versus
  10.9 ms), but ordinary HTTP produced 6/5 rather than 3/4 packets, and its
  initial TCP window was 64,240 versus 8,192. Supplemental rules supplied
  every alert in this scenario.
- **SSH brute force** produced real rejected OpenSSH authentications, but the
  fullest captured flow median was 9/9 packets versus 22/22 in the CIC SSH
  subtype. It had no model or rule alert. The first exploratory run attempted
  eight logins and encountered OpenSSH's connection limit; the final profile
  caps the run at five, and the recorded five-login run completed.
- **Slow headers** kept four connections open about 20 seconds. The raw CIC
  Slowloris subtype has a median around 102 seconds with sparse 3/2 packet
  flows; the live median was roughly 21 seconds and 36/34 packets. The monitor
  emitted many interim snapshots, and Stage 2 judged all reached flows benign.
- **Port scan** reached Stage 2 on 10 of 12 connections, but DL classified
  them benign. Seven connections received supplemental rule alerts. The
  dataset's zero SYN flags for Portscan do not describe the live Nmap SYN
  scan; suppressing real SYNs would falsify the packet evidence.
- **SQLi, XSS and infiltration** depend on payload content or a multi-step
  sequence. The raw Parquet has neither packet payload nor flow order. Their
  scenario labels are therefore ground truth from the manifest, not something
  one numeric flow can prove. SQLi/XSS received only rule alerts; the
  synthetic infiltration sequence was not detected.

Across many TCP modes, the live initial window and SYN counts differ sharply
from the raw CIC rows. The monitor also produces early cumulative snapshots
and can create a trailing one-packet teardown record for a completed 5-tuple.
The evaluation preserves those verdicts and uses the fullest snapshot only for
feature-shape comparison. These are capture/extractor and deployment-domain
differences; increasing traffic intensity or claiming the scenario labels as
model predictions would not resolve them.

## Reproduce and inspect

See [`testbed/README.md`](testbed/README.md) for startup commands and mode
names. For example:

```powershell
$env:IDS_LOG_FEATURES = '1'
docker compose up -d --build victim
docker compose up -d --no-deps --build monitor
.\scripts\run_testbed_campaign.ps1 -Attack ddos_udp -Seed 42
uv tool run --from duckdb python scripts/evaluate_attack_campaigns.py --campaign-id 20260925T082613Z-85c7c3f1
```

Replace the example campaign ID with the one printed by the new run. Local
manifests and evaluation JSON files are in `data/campaigns/`; the CIC reference
statistics are in [`analysis/attack_dataset_profile.json`](analysis/attack_dataset_profile.json).
The selected campaign IDs for this report are:

| Mode | Campaign ID | Mode | Campaign ID |
| --- | --- | --- | --- |
| `scan` | `20260925T083710Z-56604175` | `botnet` | `20260925T083716Z-8070132c` |
| `ssh_bruteforce` | `20260925T083724Z-ae2bc9dc` | `web_login` | `20260925T083744Z-9c230567` |
| `web_sqli` | `20260925T083702Z-a17f1008` | `web_xss` | `20260925T083751Z-8bcd934f` |
| `dos_http` | `20260925T083758Z-157324d5` | `dos_slow` | `20260925T082402Z-200b466d` |
| `dos_syn` | `20260925T082430Z-e70404e9` | `infiltration` | `20260925T083803Z-f56a9dcf` |
| `ddos_http_loic` | `20260925T083544Z-c615b9c6` | `ddos_http_hoic` | `20260925T082558Z-8b74625e` |
| `ddos_udp` | `20260925T082613Z-85c7c3f1` | | |

## Next engineering decision

The attack node and ground-truth coverage are in place. The strongest next
experiment is to inspect the full TCP packet traces alongside the CIC feature
extractor contract, then test a completed-flow representation for Stage 1.
Retest the SSH, Botnet and HTTP scenarios on that same representation before
changing the models or claiming a hybrid detection gain. Content- and
sequence-dependent attacks may need an explicit payload or sequence detector;
flow dimensions alone did not identify the synthetic infiltration run.

## Model-only follow-up (25 September 2026)

After removing supplemental alerts, the monitor and dashboard were restarted
with the same extracted ML/DL artifacts. Two isolated campaigns completed with
the background generators stopped. Feature logging was off, so this check
measures verdicts and Stage 2 reach, not feature similarity. These are bounded
lab campaigns rather than a population F1 evaluation.

| Scenario | Campaign ID | Actions | Connections | Stage 2 | Correct model class | Model alerts | p95 engine ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `scan` | `20260925T095340Z-2cab5cb4` | 12/12 | 12 | 10 | 0 | 0 | 48.70 |
| `ddos_udp` | `20260925T095412Z-7ef9e71f` | 80/80 | 20, two sources | 20 | 20 DDoS | 20 | 18.65 |

The scan result confirms that a Stage 1 or Stage 2 benign verdict is no longer
overridden by a rule. It also exposes a real detection gap for this scan. The
UDP run confirms that Stage 1 still forwards attack candidates to DL and that
DL alerts reach the dashboard. The remaining 11 scenarios require a new
model-only run before their current detection rates can be reported.
