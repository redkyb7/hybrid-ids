# Controlled live testbed evaluation — 24 September 2026

## Setup

The running Docker testbed used the shared model bundle in
`updated_models/shared/`. Its saved Stage 1 Random Forest attack threshold was
`0.65`. The victim and monitor remained running. The automatic attacker and
benign generator were stopped; then a short benign-only interval and each of
the five attacker campaigns were run separately. The two automatic generators
were restored to their previous running state afterward.

The attacker scripts targeted only the isolated Docker victim at
`192.168.100.10`. Each measurement used a distinct SQLite `logs.id` range so
earlier dashboard history did not affect it. The rows for the benign baseline
all had source `192.168.100.101`; the rows for every attack campaign all had
source `192.168.100.66`. The monitor remained up with no classification errors
found in its logs during the test.

For Stage 1 Normal verdicts, the database stores `confidence = 1 - Stage 1
attack probability`, rounded to four decimal places. The probabilities below
were reconstructed from that value. A row is one emitted flow *snapshot*;
several rows may come from the same TCP connection.

## Observed verdicts

| Isolated run | SQLite row IDs | New snapshots | Alerts | Stage 1 attack probability range |
| --- | ---: | ---: | ---: | ---: |
| Benign HTTP baseline | 47717–47767 | 51 | 0 | 0.0030–0.0477 |
| Nmap port scan | 47768–47779 | 12 | 0 | 0.1543–0.1752 |
| 100-SYN DoS script | 47780–47877 | 98 | 0 | 0.1645 |
| HTTP password guesses | 47878–47907 | 30 | 0 | 0.0121–0.0199 |
| SQL injection and XSS probes | 47908–47931 | 24 | 0 | 0.0052–0.0699 |
| Botnet-style HTTP beacons | 47932–47955 | 24 | 0 | 0.0123–0.0199 |

All 188 attacker-origin snapshots were labeled `Normal Traffic` at `Stage 1
(ML)`; none reached Stage 2. The attack scripts completed successfully: Nmap
scanned the intended ports, the DoS script sent its SYN burst, the web and
botnet requests received HTTP 200 responses, and the password-guessing script
made ten attempts, with the final one accepted by the intentionally vulnerable
testbed. The database recorded 98 snapshots for the 100-SYN burst; a snapshot
count is not a packet count.

## Threshold check

The following is a **Stage 1 gate counterfactual** using the rounded scores
stored for these Normal verdicts. It shows how many snapshots would have
passed the gate at each hypothetical threshold. It does not predict Stage 2
labels or establish a suitable production threshold.

| Gate threshold | Benign passed / 51 | Scan passed / 12 | DoS passed / 98 | Brute force passed / 30 | Web passed / 24 | Botnet passed / 24 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.65 (current) | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.10 | 0 | 12 | 98 | 0 | 0 | 0 |
| 0.02 | 13 | 12 | 98 | 0 | 5 | 0 |
| 0.01 | 41 | 12 | 98 | 30 | 18 | 24 |

Lowering the threshold might recover scans and the SYN burst, but the
HTTP-based attacks overlap the benign baseline. A threshold alone cannot
separate those examples reliably.

## What this establishes

This controlled run confirms that the deployed pipeline missed all five
simulated campaign types in the live testbed at its current threshold. The
large Normal slice on the dashboard therefore includes attack-script traffic,
not just benign traffic. The shared holdout's `0.9675` combined binary attack
F1 was measured on CIC data and does not represent live testbed detection.

These are short, synthetic campaigns and the database does not retain packet
payloads, ground-truth event IDs, or the complete feature vectors. The table
therefore reports observed verdicts, not a statistically representative live
F1 score or a precise connection-level false-negative rate.

## Follow-up implementation and re-test — 25 September 2026

The monitor now runs the original ML → DL inference and then applies separate,
explicit checks for multi-port scans, short SYN bursts, repeated HTTP login
attempts, SQL injection/XSS request patterns, and botnet-style HTTP headers.
These checks do not use the attacker IP as a signal. A rule can raise an alert
when the model says Normal; the database retains the model's original label,
confidence, Stage 1 attack probability, rule ID, and final detection source.
Rule alerts have no fabricated probability. Optional `IDS_LOG_FEATURES=1`
stores the 21 numeric Stage 1 inputs as JSON, without packet payloads. It was
enabled for this run and returned to its default off state afterward.

The automatic generators were paused again. A benign-only interval lasted
about two minutes; each attack campaign then ran by itself. SQLite row ranges
served as campaign labels. The benign window contained only
`192.168.100.101`; every attack window contained only `192.168.100.66`.

| Isolated run | SQLite row IDs | Connections alerted / seen | Snapshots alerted / seen | Model alerts | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Benign HTTP baseline | 54368–54580 | 0 / 71 | 0 / 213 | 0 | 32 ms |
| Nmap port scan | 54581–54592 | 7 / 12 | 7 / 12 | 0 | 42 ms |
| 100-SYN DoS script | 54593–54691 | 80 / 99 | 80 / 99 | 0 | 50 ms |
| HTTP password guesses | 54692–54721 | 6 / 10 | 12 / 30 | 0 | 36 ms |
| SQL injection and XSS probes | 54722–54745 | 8 / 8 | 16 / 24 | 0 | 29 ms |
| Botnet-style HTTP beacons | 54746–54769 | 8 / 8 | 16 / 24 | 0 | 39 ms |

Every campaign produced alerts, while the model still labeled all 189 attack
snapshots Normal at Stage 1. The rate checks start alerting only once their
threshold is reached, which explains the early Normal snapshots. The HTTP
checks alerted at least once on every web probe and beacon connection. There
were no alerts in the two-minute benign sample. All five Docker services were
restored, and the monitor logs showed no classification errors. The dashboard
now shows the latest 500 snapshots in its classification chart; historical
totals remain separate. Its `0.9675` F1 remains the **offline ML/DL-only**
holdout result and excludes the live rules.

### Stage 1 input comparison

The same logged windows were compared with the local CIC Parquet. The analysis
scanned all 9,167,581 rows in batches and selected deterministic samples by
broad class: Benign 7,215, Portscan 2,255, DoS 1,588, Bruteforce 2,058,
Webattack 2,995, and Botnet 2,905. This is a sample of the **raw CIC data**,
not the exact deduplicated training split. The complete 21-feature summaries
are in [LIVE_TESTBED_FEATURE_COMPARISON.json](LIVE_TESTBED_FEATURE_COMPARISON.json);
[`scripts/compare_live_features.py`](scripts/compare_live_features.py) can
regenerate them from labeled SQLite windows.

| Compared classes | CIC sample median | Live median |
| --- | ---: | ---: |
| Benign `Init Fwd Win Bytes` | 508 | 64,240 |
| Portscan `SYN Flag Count` | 0 | 1 |
| DoS `Total Fwd Packets` | 4 | 2 |
| Bruteforce `Total Fwd Packets` | 22 | 7 |
| Bruteforce `Fwd Header Length` | 712 | 232 |
| Webattack `Packet Length Max` | 0 | 2,138 |
| Botnet `Init Fwd Win Bytes` | 8,192 | 64,240 |

The large differences are consistent with the live model scores being low.
They may reflect the short snapshot window, the testbed's operating systems
and services, and differences in feature construction. The CIC Bruteforce
class is predominantly SSH/FTP traffic, whereas this test uses HTTP login
requests. The comparison does not prove that one extractor calculation alone
is responsible.

### Remaining limits

This is a controlled demonstration on a small, synthetic testbed. It is not a
general live F1 estimate. The rate checks can alert on legitimate bursts, and
the HTTP patterns only inspect cleartext traffic with these visible strings.
A longer and more varied benign capture, independent attacks, and held-out
live feature vectors are needed before choosing a new model or threshold.
