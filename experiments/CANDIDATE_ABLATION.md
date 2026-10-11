# Candidate-architecture ablation

From the repository root:

```bash
PYTHONPATH=src python3 -m pytest -q tests/test_candidate_ablation.py tests/test_architecture_contract.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 experiments/08_candidate_ablation.py \
  --cities madrid cambridge_ma new_york_ny \
  --workloads 1000000 100000 10000 500000 \
  --agents 10000 --repeats 5 --diagnostic-queries 10000
```

The delivered Cambridge results use the priority 1M and 100K workloads:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 experiments/08_candidate_ablation.py --cities madrid cambridge_ma new_york_ny \
  --workloads 1000000 100000 --agents 10000 --repeats 5 --diagnostic-queries 10000
python3 analysis/09_finalize_candidate_ablation.py
```

Results go to `results/candidate_ablation/` by default. The runner skips a city
only when its POI or population CSV is absent and records both paths in
`metadata.json`. Each available city uses one fixed, seeded query workload per
N; all architectures replay it unchanged. Backend order is independently
shuffled for each measured repetition. A 1,000-query untimed warm-up precedes
timing. The separate 10,000-query diagnostic replay is used for candidate
counts, destination agreement, and approximate component timing. This latter
replay is instrumented and never supplies headline latency.

For publication text, extended candidate-count percentiles, and SVG figures,
run the deterministic CSV-only finalizer after timing:

```bash
python3 analysis/09_finalize_candidate_ablation.py
```

If workload sizes were checkpointed into separate result directories, supply
the additional directories with `--merge-timings`; only complete paired timing
rows are merged. The first directory supplies the common diagnostic replay.

The default `--agents 10000` fixes the world population across workload sizes,
so `--workloads 1000000` means one million *independent decisions* sampled from
10,000 agents, not one million evolving agents. To match the manuscript's
one-million-agent population sweep more closely, use `--agents 1000000` (with
sufficient memory and the same datasets). Results from different agent
populations must not be conflated. The independent query generator and backend
decision paths are the same as `experiments/01_architecture_benchmark.py`.

`continuous_radius_cap24` performs the original complete BallTree radius query
and exact radius filtering before selecting 24 nearest eligible POIs by exact
distance (POI ID resolves ties). `continuous_knn24` requests 24 neighbors
directly from the per-semantic-layer BallTree and then applies the original
exact radius. It does not relax the maximum destination distance. Both use the
shared `destination_score` function unchanged. Continuous weighted
accessibility in the radius-cap variant remains the exhaustive-radius value;
the kNN variant aggregates only its bounded set, whereas H3 continues to use
its cached H9 accessibility state. Thus this is a
decision-architecture ablation, not a claim of identical context outputs.

The search radius is `Config.search_radius_m` (2,000 m by default); semantic
layers are sampled from active schedule-layer IDs by `make_query_set` with
seed `Config.seed + 7001`. World/agent sampling uses `Config.seed`. Destination
choice is deterministic `np.argmax` on the shared score, so an exact score tie
uses the first candidate in each backend's retrieval order. The new
radius-cap selection has an explicit distance/POI-ID tie-break; BallTree kNN
ties at the K boundary may differ from radius retrieval, which is reflected
in destination agreement rather than hidden.

H3 caps its *collected* H10 candidates before distance eligibility filtering;
the default H10 neighborhood is ring 1 with cap 24, and the explicit bounded
H9 ring-1 fallback cap is 48. The K=48 variant changes only the primary cap;
its cached H9 best-POI path may return a destination without scoring a local
candidate, and its H9 fallback may select a POI outside the Continuous search
radius. These existing behaviors are preserved, not harmonized away. Therefore
nominal K=24 is not an identical actual scored-candidate distribution: inspect
`candidate_counts_summary.csv` before attributing a residual speedup.

`component_timing.csv` is approximate. Continuous `lookup_time_s` includes
query-coordinate construction plus BallTree retrieval; `distance_time_s`
includes exact distance, radius filtering, cap selection, and the
full-radius accessibility computation in the radius-cap variant;
`decision_time_s` covers scoring and argmax. H3 `lookup_time_s` covers H12/H5/H8/H9
context; `candidate_collection_s` covers H10 bucket collection;
`distance_time_s` covers local exact distance, radius filtering, scoring and
argmax; `fallback_s` covers the bounded H9 search/evaluation. Residual time is
Python orchestration and measurement overhead. Component fractions must not be
used as causal speedups because timers perturb microsecond paths.
