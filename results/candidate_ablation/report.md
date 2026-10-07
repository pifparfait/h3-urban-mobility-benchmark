# Candidate-architecture ablation

All timing figures are uninstrumented complete decisions (excluding common OSM routing). Continuous-exact is a cross-representation reference, not behavioral truth. The diagnostic workload comprises 10,000 independent seeded queries per city and architecture.

The continuous radius-cap strategy performs the original exhaustive BallTree radius query, exact-distance filter, and full-radius accessibility calculation, then deterministically retains the nearest 24 eligible POIs (ID tie-break) before shared destination scoring. The continuous kNN strategy requests 24 neighbors from the same semantic-layer BallTree, applies the original exact maximum radius, and uses the same score. The H3 variants preserve cached H9 accessibility and fallback; only the H10 primary cap differs. Backend execution order was randomized by repetition.

## Original-benchmark reproduction check

The unchanged `full_agent_decision` path at 10,000 queries (3 repetitions) yielded 63.07 µs for Continuous and 23.23 µs for H3 (2.71× ratio of medians). The manuscript's one-million-query Cambridge values are 62.96 and 22.01 µs (2.86×); population and hardware/software conditions differ, but the ordering and magnitude are consistent. The paired ablation results below are the primary new measurements, not this check.

## cambridge_ma: N=1,000,000 independent decisions

| Architecture | Median µs | IQR µs | Paired speedup vs C-exact | Exact destination agreement | Mean scored candidates |
|---|---:|---:|---:|---:|---:|
| continuous_exact | 63.33 | 63.32–63.36 | 1.00× | 100.0% | 68.84 |
| continuous_radius_cap24 | 69.26 | 68.87–69.31 | 0.92× | 98.1% | 21.43 |
| continuous_knn24 | 62.65 | 62.65–62.75 | 1.01× | 97.2% | 21.43 |
| h3_cap24 | 23.36 | 23.32–23.59 | 2.71× | 73.5% | 1.07 |
| h3_cap48 | 23.21 | 23.01–23.49 | 2.73× | 73.5% | 1.07 |

### Explicit interpretation

**Q1 — Candidate-scoring restriction after radius retrieval.** C-exact→C-radius-cap24 gives 0.92× paired speedup. The cap accounts for -14.8% of the 39.98 µs architecture latency reduction on a difference-of-medians basis; a negative value means the cap variant is slower, not that restricting scoring intrinsically slows computation.

**Q2 — Bounded-index contrast.** H3-24 is 2.68× faster than C-kNN24, and 2.95× faster than C-radius-cap24 (paired medians). These are architecture contrasts, not an isolated H3-index effect: actual scored candidate volumes and cached context still differ.

**Q3 — Candidate distributions.** C-exact scores mean 68.84 (median 52, P95 180) versus H3-24 mean 1.07 (median 0, P95 5). C-radius-cap24 hits its cap in 81.0% of sampled decisions; H3-24 in 0.1%. H3 cached-best/explicit fallback frequency is 58.2%; no-destination frequency is 0.0%.

**Q4 — continuous_exact dominant measured phase:** lookup_time_s ≈55% of the separately instrumented replay.
**Q4 — continuous_radius_cap24 dominant measured phase:** lookup_time_s ≈51% of the separately instrumented replay.
**Q4 — continuous_knn24 dominant measured phase:** lookup_time_s ≈58% of the separately instrumented replay.
**Q4 — h3_cap24 dominant measured phase:** distance_time_s ≈46% of the separately instrumented replay.
**Q4 — h3_cap48 dominant measured phase:** distance_time_s ≈45% of the separately instrumented replay.
These phase boundaries are coarse; residual Python/measurement time is listed in component_timing.csv. Fine-grained timers are excluded from headline latencies.

**Q5 — Fidelity.** Destination identity agreement is 98.1% for radius-cap24, 97.2% for kNN24, 73.5% for H3-24, and 73.5% for H3-48. Separation, within-distance fractions, trip-distance Pearson/NMAE, and class substitution are in fidelity_summary.csv. Separation and trip-distance metrics condition on both variants returning a destination.

**Q6 — H3 cap sensitivity.** K=24→48 changes agreement 73.5%→73.5% and median latency 23.36→23.21 µs (0.99× runtime). No penalty was detected; the small measured decrease should not be interpreted as an intrinsic benefit of a larger cap.

### Proposed manuscript text

In Cambridge, Massachusetts, the H3-based complete-decision architecture reduced median latency from 63.33 to 23.36 µs per independent query (2.71× paired speedup; N=1,000,000, five paired repetitions). Applying a 24-candidate scoring cap *after* the continuous radius query yielded 69.26 µs, while bounded BallTree kNN24 yielded 62.65 µs. Thus simply reducing the post-radius score count did not reproduce the architecture gain in this city; the bounded continuous-index contrast likewise retained a substantial latency gap. This gap must not be assigned to H3 indexing alone because the H3 path also uses cached H9 context and a much smaller realized local candidate set.

The H3 primary cap of 24 was rarely reached in the sampled decisions (0.1%). Increasing it to 48 changed destination identity agreement from 73.5% to 73.5%, with median latency 23.36→23.21 µs. These are cross-representation agreement measures against the original continuous implementation, not behavioral accuracy against observed travel choices. Cached H9/explicit fallback served 58.2% of sampled H3-24 decisions. The actual scored-candidate distributions and H9 cached-best use, rather than nominal K alone, govern interpretation of the mechanism.

## Cross-city conclusion and reviewer response

**Q7.** Available study areas: cambridge_ma. Missing required datasets: {'madrid': ['data/processed/madrid_osm_pois.csv', 'data/processed/madrid_population_h3_r9.csv'], 'new_york_ny': ['data/new_york_ny/processed/new_york_osm_pois.csv', 'data/new_york_ny/processed/new_york_population_h3_r9.csv']}. Cross-city consistency cannot be tested.

**Q8.** None of A–D is established across all three cities by the available evidence. In the available city, the results reject a simple 'post-radius scoring cap explains most of the speedup' account, but do not isolate H3 indexing from precomputation, cached context, or the smaller realized candidate set.

**Proposed reviewer response.** We added a paired architecture ablation that retains the original exhaustive-radius continuous reference, introduces a continuous variant that caps scoring only after identical radius retrieval, compares a bounded BallTree kNN24 architecture with the existing H3-24 path, and tests H3-48 sensitivity. The same seeded independent queries, POIs, semantic layers, behavioral parameters, and exact-coordinate score are replayed across all configurations. We report five repeated uninstrumented complete-decision timings, realized candidate-count distributions, cross-representation destination agreement, and separately instrumented phase profiles. The results distinguish the complete H3-based architecture gain from any claim that H3 indexing alone causes it. Only Cambridge inputs were available locally, so the revised paper should not generalize this ablation to Madrid or New York until those datasets are run.

## Limitations

The measured one-million-query run samples a fixed 10,000-agent world, unlike a one-million-agent population sweep; queries remain independent. H3 cached-best selection can return a destination beyond the continuous radius, as in the existing implementation. The nominal K=24 variants do not equalize actual scored-candidate counts. No paired temporal/OD output experiment was run because the existing simulation runner does not parameterize these ablation variants. Component fractions are approximate and include instrumentation overhead in the residual. Small five-run bootstrap intervals characterize this machine, not cross-machine uncertainty.

## Figures and machine-readable data

For each available city, `figures/{city}_latency`, `_speedup`, `_candidates`, and `_frontier` are supplied as 300-dpi PNG and editable SVG. The source tables are `timings_raw.csv`, `timings_summary.csv`, `candidate_counts_raw.csv`, `candidate_counts_summary.csv`, `fidelity_summary.csv`, and `component_timing.csv`; run provenance is in `metadata.json`.
