# Benchmarks

Fovea's public engineering target is a median guided-test error of at most
`0.06` display-normalized units and at most `3°` at a typical 60 cm laptop
viewing distance. Both thresholds must pass; neither is a medical or assistive
technology certification.

No maintainer-verified live reports have been published yet. The required runs
on at least three distinct machines remain an owner action because they require
real cameras, measured displays, controlled distances, and human review. Do not
replace this notice with simulated or unverified numbers.

| Machine | Camera | Resolution | Lighting | Glasses | 50 cm median | 60 cm median | 75 cm median | Jitter p95 | Drift delta | Latency p95 | Report |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| Pending live run 1 | — | — | — | — | — | — | — | — | — | — | — |
| Pending live run 2 | — | — | — | — | — | — | — | — | — | — | — |
| Pending live run 3 | — | — | — | — | — | — | — | — | — | — | — |

Run the procedure in [bench/PROTOCOL.md](bench/PROTOCOL.md), review the JSON for
completeness, commit the report under `bench/results/`, and replace one pending
row with its measured values. Keep failed runs: transparent regressions are more
useful than a table containing only favorable hardware.
