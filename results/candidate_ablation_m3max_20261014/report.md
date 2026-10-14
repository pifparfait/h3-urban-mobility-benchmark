# Candidate-architecture ablation

Continuous-exact is a cross-representation reference, not behavioral truth.
Primary latencies are uninstrumented; component fractions are from a separate diagnostic replay.
Continuous radius-cap retains the full radius query and selects 24 nearest eligible POIs by exact distance (ID tie-break).
Continuous kNN requests 24 nearest POIs from the same per-layer BallTree, then applies the original exact radius; it does not enumerate all within-radius POIs.
H3 candidate ordering and H9 fallback are unchanged from the production implementation.
The scored-candidate sample uses a separate identically seeded independent workload; H3 cached-best fallback may return a destination without scoring a candidate in the query.

## cambridge_ma

Largest measured workload: 1,000,000 independent queries; 5 paired repetitions.
Q1. C-exact→C-radius-cap24: 0.94×; -9.3% of C-exact→H3-24 latency reduction (difference-of-medians descriptive estimate).
Q2. H3-24/C-kNN24: 2.89×; H3-24/C-radius-cap24: 3.09×. These are architecture contrasts, not isolated H3-index effects.
Q3. Scored candidates, mean: C-exact 68.8; H3-24 1.1. See distribution CSV/figure.
Q4. Phase fractions are in component_timing.csv. H3 distance_time includes scoring; residual includes Python orchestration and instrumentation.
Q5. Destination agreement: C-radius-cap24 98.2%, C-kNN24 97.2%, H3-24 73.7%.
Q6. H3 K=24→48 agreement 73.7%→73.7%; median latency 21.40→21.51 µs.

## Cross-city interpretation

Q7. Compare the city-specific controlled ratios above; unavailable cities are not inferred.
Q8. No single A–D mechanism claim is assigned without all study areas. Interpret the controlled contrasts above directly.
Missing inputs: {}

## Limitations

The query population is fixed by --agents and may differ from the manuscript's N-agent sweeps. kNN distance filtering preserves the maximum radius but truncates accessible candidates. H3's cached H9 best-POI fallback can choose a POI outside the continuous radius; this production behavior is intentionally retained. Diagnostic timing includes trace overhead and is only approximate. Bootstrap intervals describe measured runs, not machine-independent performance.
