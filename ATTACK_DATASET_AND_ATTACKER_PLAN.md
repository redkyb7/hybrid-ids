# CIC attack-flow findings and attacker-node implementation plan

25 September 2026

**Implementation status:** The bounded attack scenarios and live measurements
described by this plan are now recorded in
[`ATTACKER_IMPLEMENTATION_AND_EVALUATION.md`](ATTACKER_IMPLEMENTATION_AND_EVALUATION.md).
The gap and implementation sections below describe the design baseline.

## Data inspected and extracted

The local source is `clean_data/cic-collection.parquet` (9,167,581 rows, 59
columns). `ClassLabel` contains seven attack classes plus Benign; `Label`
contains 32 attack subtypes. The script
[`scripts/profile_attack_dataset.py`](scripts/profile_attack_dataset.py)
extracted **all 1,981,392 attack rows** into
`analysis/attack_rows/ClassLabel=<class>/Label=<subtype>/*.parquet`. It verified
the exported row count of every subtype against the source. It also wrote 2,000
seeded sample rows per broad class, 100 per subtype (or every row for smaller
subtypes), and 2,000 benign rows to `analysis/attack_row_samples/*.csv`.

The [full numerical profile](analysis/attack_dataset_profile.json) contains
whole-dataset 10th/50th/90th percentiles and zero fractions for 26 flow
features, plus single-feature rank AUCs from the 2,000-row class samples.
[Representative complete rows](analysis/attack_row_examples.json) preserve one
sampled row per subtype for review in Git. The large Parquet and CSV exports are
local files ignored by Git; the script regenerates them:

```powershell
uv tool run --from duckdb python scripts/profile_attack_dataset.py
# If the Parquet exports already exist and only the summaries need refreshing:
uv tool run --from duckdb python scripts/profile_attack_dataset.py --reuse-exports
```

These are profiles of the **raw source Parquet**, not the exact deduplicated ML
training split or a DL holdout split. Percentiles are approximate; row counts
and zero fractions are exact. Each row is a network flow. The file has no IPs,
ports, protocol, packet payload, absolute time, or order between flows, so it
cannot by itself establish distribution across senders, scanned port counts,
HTTP exploit strings, beacon intervals, or an infiltration sequence.

## Broad-class measurements

Durations are shown in milliseconds here; the Parquet stores microseconds.
Packet sizes and directional totals are bytes. Values below are approximate
medians across **all** rows in a broad class, so mixed DoS and DDoS subtypes
need the subtype detail that follows.

| Class | Rows | Duration | Fwd / bwd packets | Fwd / bwd bytes | Max packet | Packets/s | Reading |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Benign | 7,186,189 | 260 ms | 3 / 2 | 98 / 241 | 178 | 21 | Reference only; broad spread. |
| Botnet | 145,968 | 10.9 ms | 3 / 4 | 326 / 129 | 326 | 642 | Short, highly uniform exchanges in this dataset. |
| Bruteforce | 103,244 | 376 ms | 22 / 22 | 1,928 / 2,665 | 976 | 117 | Dominated by SSH authentication sessions. |
| DDoS | 1,234,729 | 1,392 ms | 3 / 0 | 20 / 0 | 516 | 5 | Mixture of HTTP, UDP, SYN and slow variants; median is not a single attack recipe. |
| DoS | 397,344 | 5,425 ms | 4 / 4 | 318 / 935 | 935 | 1.1 | Short HTTP floods and long, sparse connections are mixed. |
| Infiltration | 94,857 | 247 ms | 3 / 2 | 77 / 154 | 130 | 20 | Close to the benign center on these measures. |
| Portscan | 2,255 | 0.64 ms | 2 / 1 | 0 / 2 | 6 | 5,280 | Mostly tiny, short exchanges; multiport context is absent from a row. |
| Webattack | 2,995 | 5,476 ms | 3 / 1 | 0 / 0 | 0 | 0.76 | Zero recorded payload is common and must not be copied as an attack mechanism. |

For a view of **actual extracted rows**, these seven examples were selected
from the subtype samples by proximity to six subtype medians. They are not
synthetic target vectors or claims that one flow proves its attack label.

| Source `Label` | Duration (µs) | Fwd / bwd packets | Fwd / bwd bytes | Max packet (B) |
| --- | ---: | ---: | ---: | ---: |
| Botnet | 10,908 | 3 / 4 | 326 / 129 | 326 |
| Bruteforce-SSH | 375,389 | 22 / 22 | 1,928 / 2,665 | 976 |
| DDoS-NTP | 1,706 | 44 / 0 | 19,360 / 0 | 440 |
| DoS-Hulk | 44,452 | 3 / 4 | 287 / 935 | 935 |
| Infiltration | 231,303 | 2 / 2 | 76 / 204 | 102 |
| Portscan | 424 | 1 / 1 | 0 / 6 | 6 |
| Webattack-XSS | 5,407,328 | 3 / 1 | 0 / 0 | 0 |

### What distinguishes the labels in these rows

- **Botnet:** the central 80% of rows has exactly 3 forward and 4 backward
  packets, 326 forward bytes and 129 backward bytes, with flow duration about
  9.7–13.3 ms. It is a narrow flow template; the rows do **not** show periodic
  beaconing or a real command-and-control topology. In the 2,000-row sample,
  `Flow Bytes/s` separates it from other attacks with AUC 0.85.
- **Bruteforce:** 97,260 of 103,244 rows are `Bruteforce-SSH`; the rest are
  `Bruteforce-FTP`. SSH's median is 22 packets each direction over 375 ms,
  while FTP's median is 9/15 packets over 8.7 s. Forward/backward packet and
  header counts strongly separate the SSH-heavy class from other attacks
  (sample AUC about 0.94–0.96). The current attacker's HTTP form guessing is
  semantically closer to `Webattack-bruteforce`, not this broad class.
- **DDoS:** the dominant `DDoS-LOIC-HTTP` subtype (575,364 rows) has a median
  3/3 packets and 1.75 s duration. `DDoS-HOIC` (198,861) is about 3/4 packets
  and 11.7 ms. `DDoS-NTP` (121,328) has about 43 forward packets of 440 bytes
  and essentially no backward packets within roughly 1.1 ms. Other UDP
  variants are often one-way. These are multiple traffic profiles; the flow
  table cannot prove that any individual row came from multiple sources.
- **DoS:** `DoS-Hulk` accounts for 318,740 of 397,344 rows and has a median
  4/5 packets, 203 ms duration, and about 935-byte largest packet. By
  contrast, `DoS-Slowloris` has a median 102 s duration, 3/2 packets and
  0.2 packets/s. A short SYN burst is not a substitute for either HTTP
  subtype. Longer `Flow IAT Mean` distinguishes the broad DoS sample from
  other attacks (AUC 0.74), although its subtypes vary widely.
- **Infiltration:** median duration, packet counts, and packet size are close
  to benign (247 vs 260 ms; 3/2 packets in both). No single one of the 26
  sampled features separates it strongly from benign: the largest observed
  AUC departure from 0.5 is about 0.105. A single flow's dimensions do not
  define an intrusion; this needs scenario-level ground truth.
- **Portscan:** central behavior is a tiny, short connection with 2/1 packets,
  zero median forward payload and about 6-byte maximum packet. Across the
  sample, short duration and small packet lengths distinguish it from many
  other attacks. The Parquet does not include destination ports or time
  ordering, which are needed to recognize a *scan*. Notably, `SYN Flag Count`
  is zero in all Portscan rows, whereas the current live SYN scan emits SYN
  flags. We should investigate the extractor difference, not suppress SYNs.
- **Webattack:** 2,020 rows are HTTP brute force, 876 XSS, and only 99 SQLi.
  `Packet Length Max` is zero in 75.3% of broad Webattack rows (84.0% of XSS
  rows), despite payload-based labels. This is a measurement or dataset
  artifact from the point of view of an HTTP probe. SQLi/XSS contents are
  absent from the Parquet, and the deployed DL model predicts the broad
  `Webattack` category rather than these subtypes.

The other source subtypes are retained in the exported folders and JSON
profile: DDoS also contains generic DDoS, DNS, Ddossim, LDAP, MSSQL, NetBIOS,
SNMP, Slowloris, Syn, TFTP, UDP and UDPLag; DoS also contains Goldeneye,
Heartbleed (only 11 rows), Rudy, Slowbody, Slowheaders, Slowhttptest and
Slowread. For example, some DNS/LDAP/MSSQL rows have 1–2 µs durations and
large packet rates; copying those exact values into a lab is not a meaningful
or necessarily feasible implementation goal.

## Testbed gap before implementation

[`testbed/attacker/attack_campaigns.py`](testbed/attacker/attack_campaigns.py)
currently offers scan, SYN burst, HTTP login guesses, SQLi/XSS requests and
HTTP requests with botnet-themed headers. Its docstring advertises DDoS but
there is no distinct DDoS campaign or multiple sender node. It has no
infiltration scenario and no SSH/FTP brute-force campaign. The victim already
exposes HTTP and SSH, but no FTP service, local C2 fixture, or canary scenario.
The [Compose topology](docker-compose.yml) has one attacker IP.

The [live comparison](LIVE_TESTBED_EVALUATION.md) previously found all 189
isolated attack-origin snapshots benign at Stage 1 of the ML → DL pipeline;
supplemental rules supplied the alerts. Its logged feature comparison also
showed material CIC/live differences, including 22 vs 7 median forward
packets for Bruteforce, 0 vs 1 for Portscan SYN flag, and 0 vs 2,138 for
Webattack maximum packet length. The [flow aggregator](backend/flow_aggregator.py)
emits cumulative snapshots after roughly 150 ms or 25 new packets and expires
flows after about one second of inactivity. A long or sparse CIC flow may
therefore produce several live verdicts or be split. Attack generation alone
cannot guarantee that Stage 1 will pass it to DL.

## Implementation plan

### 1. Give every campaign a bounded, reproducible contract

Refactor `testbed/attacker/attack_campaigns.py` so a CLI command chooses an
explicit **broad class + subtype scenario**. Put initial parameters in
`testbed/attacker/profiles.json`: target service, request or packet budget,
duration, concurrency, size range, delay range, random seed and expected
dataset label. Use the source medians and 10th–90th percentiles as
*comparison bands*, not numbers to inject directly into the monitor. Keep
`192.168.100.10` as the only allowed target and reject arbitrary external
addresses. Enforce a global deadline and total byte/connection caps, even
for continuous mode.

Each run should write a small manifest to a mounted local volume with campaign
ID, seed, scenario label, worker IP, start/end times and actual request/packet
counts. The default continuous loop should call these same bounded functions;
isolated evaluation should use one chosen campaign at a time.

### 2. Implement one primary scenario for every broad class

| Broad class / proposed CLI mode | Attacker-node behavior and observed flow target | Required support |
| --- | --- | --- |
| `Portscan / scan` | Keep a 12-port SYN/connect sweep. Aim for small, mostly payload-free connections and record the number of distinct target ports in 10 s. Do not target the dataset's zero SYN count. | Existing Nmap and victim ports suffice. |
| `Botnet / botnet` | Send 8–20 small check-ins per run to a lab-only status/C2 endpoint, with seeded payload sizes and intervals. Check whether resulting flows approach the dataset's 3/4 packets and short, few-hundred-byte exchange. The interval is a **scenario choice**, not a feature inferred from Parquet. | Existing HTTP endpoint can serve first; a mock `/lab/c2` endpoint can return fixed small replies if size control is needed. |
| `Bruteforce / ssh_bruteforce` | Make a capped series of failed SSH logins against the victim's SSH service. Measure full authentication exchanges toward the SSH subtype's roughly 20–24 packets per direction. Add `ftp_bruteforce` only after a local FTP fixture exists. Rename the current HTTP form loop to `web_login` so its ground truth is `Webattack-bruteforce`. | SSH already exists; attacker image has Hydra. FTP needs a new lab-only service. |
| `DoS / dos_http` and `dos_slow` | Use bounded concurrent HTTP requests for Hulk-like short exchanges, then a separate few-connection slow-header/body mode for long, sparse flows. Start with 30–60 requests over ≤5 s for `dos_http`; start slow mode at ≤8 connections and ≤30 s, extending toward 100 s only after checking victim resources and flow expiration. | Victim HTTP response size/delay fixtures; monitor may need a separate completed-flow record for slow modes. |
| `DDoS / ddos_http` and `ddos_udp` | Run coordinated, capped HTTP bursts from at least two distinct attacker container IPs, with LOIC-like slower and HOIC-like shorter exchanges. A later UDP mode can send bounded 389–440-byte packets to a local sink without reflection or amplification. Compare per-flow measurements and cross-source campaign counts separately. | On-demand Compose worker services plus a local UDP sink if UDP mode is included. One attacker IP cannot establish a distributed attack. |
| `Infiltration / infiltration` | Execute a labeled multi-step lab scenario: authenticate to a mock account, access a synthetic canary, then transfer a small bounded canary blob to a local lab endpoint. Compare constituent flows to benign, while scoring the **sequence** by run ID; do not expect one CIC-like flow vector to prove infiltration. | Victim canary fixture and local collection endpoint; no real credentials or data. |
| `Webattack / web_sqli`, `web_xss`, `web_login` | Preserve the existing lab-only search probes; run SQLi, XSS and failed HTTP login attempts as separate scenarios. Vary request and response sizes within plausible HTTP behavior, and verify payload-based supplemental alerts independently of the ML/DL verdict. Do not aim to create zero-length HTTP payloads just because many dataset rows have zero. | Existing `/search` and `/login` suffice for first pass. |

The current SYN burst may remain as an extra `dos_syn` scenario, but it should
not stand in for all DoS or for DDoS. Do not describe an HTTP login campaign as
the dataset's SSH/FTP `Bruteforce` class.

### 3. Add only the supporting topology and fixtures needed by the scenarios

- Update [`docker-compose.yml`](docker-compose.yml) with opt-in attacker
  workers (`profiles: [ddos]`) on the existing testbed network, each with a
  distinct address. Keep their entrypoints idle until a coordinated run and
  enforce one shared campaign budget. Avoid source-IP spoofing.
- Extend [`testbed/victim/app.py`](testbed/victim/app.py) with small, fixed
  response-size/delay fixtures and synthetic canary endpoints. If adding UDP
  or FTP, provide minimal lab-only services; do not use third-party systems
  or real amplification services.
- Keep the attacker's target allowlist and limits in code as well as the
  Compose configuration. Record any size or timing band that the container
  stack cannot reproduce rather than changing model inputs to fake a match.

### 4. Measure packets, flow features, and model outcomes separately

Run each campaign with the automatic benign and attacker loops paused, leaving
victim and monitor active as in [`testbed/README.md`](testbed/README.md).
Record a benign-only interval. Enable `IDS_LOG_FEATURES=1` for the measurement
run and collect packet counts, per-source IPs, destination ports, connection
counts, full emitted feature vectors, Stage 1 scores, Stage 2 reach/class,
supplemental rule IDs and final labels. Use the campaign manifests to join
telemetry to ground truth. A snapshot is not one connection or one packet.

For each scenario, compare live medians and 10th–90th percentiles with the
corresponding subtype profile for controllable features: packet counts,
directional bytes, packet size, duration and interarrival time. Analyze flow
completion separately from the monitor's early snapshots. Track what portion
of flows reach DL, the DL broad-class confusion matrix, final alerts due to
model versus rules, benign false positives and p95 latency. Report results by
scenario and source worker; never credit a rule-only alert as a DL success.

If packet capture confirms the intended traffic but live flow features remain
far from the training feature definitions, address the
`backend/flow_aggregator.py` / `backend/live_capture.py` contract (including
long-flow completion and idle timeout) before tuning attack intensity or
retraining. Retest the Stage 1 gate after that change. Infiltration and
content-based Webattack may require sequence/payload evidence beyond these
57 numeric flow inputs; a low ML/DL score for them is an evaluation finding,
not a reason to manufacture impossible feature values.

### 5. Acceptance checks and delivery order

1. Add profile schema validation, target allowlist, global caps, campaign
   manifests and one-command isolated execution; test deterministic seeds and
   budget enforcement.
2. Correct label semantics and implement SSH brute force, HTTP DoS, separate
   web probes and botnet check-ins. Smoke-test each against the victim and
   verify actual packets/requests and recorded subtype labels.
3. Add opt-in multi-source DDoS workers and the bounded infiltration fixture.
   Verify distinct source IPs for DDoS and local-only canary movement for
   Infiltration. Add slow/UDP/FTP variants only with matching lab services.
4. Re-run benign and every primary broad-class scenario in isolation. Publish
   a table of observed feature ranges, Stage 1 pass rates, Stage 2 outcomes,
   rule-only alerts, false positives and latency. State any subtype whose
   source profile cannot be reproduced or whose model classification fails.

Success means all seven broad classes have a **runnable, bounded, labeled lab
scenario** and measured traffic evidence. It does not mean the current models
will automatically recognize all seven: that claim requires the measured
ML → DL results from step 4.
