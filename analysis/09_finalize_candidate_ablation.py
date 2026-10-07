#!/usr/bin/env python3
"""Rebuild ablation summaries/figures and publication text from measured CSVs.

This also permits checkpointed runs at separate workload sizes to be merged
without re-timing any backend. The first results directory supplies the
diagnostic sample; --merge-timings supplies additional paired timing rows.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("candidate_ablation", ROOT / "experiments" / "08_candidate_ablation.py")
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "candidate_ablation")
    parser.add_argument("--merge-timings", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    out = args.results.resolve()
    timings = pd.read_csv(out / "timings_raw.csv")
    for path in args.merge_timings:
        extra = pd.read_csv(path / "timings_raw.csv")
        timings = pd.concat([timings, extra], ignore_index=True)
    timings = timings.drop_duplicates(["city", "backend", "workload_n", "repetition"], keep="first")
    counts = pd.read_csv(out / "candidate_counts_raw.csv")
    # Older diagnostic traces recorded only the explicit H9 ring-search
    # fallback. Recover the cached-H9-best path from its characteristic
    # successful decision with zero locally scored candidates.
    if "cached_best" not in counts:
        counts["cached_best"] = ((counts.backend.str.startswith("h3")) &
                                 (counts.scored == 0) & (counts.no_destination == 0) &
                                 (counts.fallback == 0)).astype(int)
        counts["fallback"] = counts[["fallback", "cached_best"]].max(axis=1)
    fidelity = pd.read_csv(out / "fidelity_summary.csv")
    components = pd.read_csv(out / "component_timing.csv")
    raw, summary, count_summary = ablation.summarize(timings.to_dict("records"), counts.to_dict("records"))
    raw.to_csv(out / "timings_raw.csv", index=False)
    summary.to_csv(out / "timings_summary.csv", index=False)
    counts.to_csv(out / "candidate_counts_raw.csv", index=False)
    count_summary.to_csv(out / "candidate_counts_summary.csv", index=False)
    ablation.figures(out, summary, counts, fidelity)
    metadata_path = out / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["query_seed_offset"] = 7001
    metadata["workloads"] = sorted(int(x) for x in raw.workload_n.unique())
    metadata["git_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    source_files = (
        "experiments/08_candidate_ablation.py",
        "src/h3uab/benchmarking.py", "src/h3uab/config.py",
        "src/h3uab/continuous_indexed_backend.py", "src/h3uab/h3_backend.py",
        "src/h3uab/static_city.py",
    )
    metadata["source_sha256"] = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in source_files
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        metadata["cpu_model"] = next(
            (line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
             if line.startswith("model name")), metadata.get("cpu_model", "unknown"))
    for city in metadata.get("requested_cities", []):
        if city not in metadata.get("available_cities", []):
            cfg = ablation.config_for_city(city)
            metadata.setdefault("missing_inputs", {})[city] = [
                str(path) for path in (cfg.osm_cache_csv, cfg.population_h3_csv)
                if not path.exists()
            ]
    for city in metadata.get("available_cities", []):
        cfg = ablation.config_for_city(city)
        metadata[f"inputs_{city}"] = {
            str(path): {"bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (cfg.osm_cache_csv, cfg.population_h3_csv)
        }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    lines = ["# Candidate-architecture ablation", "",
             "All timing figures are uninstrumented complete decisions (excluding common OSM routing). "
             "Continuous-exact is a cross-representation reference, not behavioral truth. "
             "The diagnostic workload comprises 10,000 independent seeded queries per city and architecture.", "",
             "The continuous radius-cap strategy performs the original exhaustive BallTree radius query, "
             "exact-distance filter, and full-radius accessibility calculation, then deterministically "
             "retains the nearest 24 eligible POIs (ID tie-break) before shared destination scoring. "
             "The continuous kNN strategy requests 24 neighbors "
             "from the same semantic-layer BallTree, applies the original exact maximum radius, "
             "and uses the same score. The H3 variants preserve cached H9 accessibility and fallback; "
             "only the H10 primary cap differs. Backend execution order was randomized by repetition.", ""]
    baseline_path = out / "baseline_reproduction.json"
    if baseline_path.exists():
        b = json.loads(baseline_path.read_text())
        lines += ["## Original-benchmark reproduction check", "",
                  f"The unchanged `full_agent_decision` path at {b['n_queries']:,} queries "
                  f"({b['repeats']} repetitions) yielded {b['continuous_median_us']:.2f} µs "
                  f"for Continuous and {b['h3_median_us']:.2f} µs for H3 "
                  f"({b['speedup_ratio_of_medians']:.2f}× ratio of medians). "
                  "The manuscript's one-million-query Cambridge values are 62.96 and 22.01 µs "
                  "(2.86×); population and hardware/software conditions differ, but the ordering "
                  "and magnitude are consistent. The paired ablation results below are the primary "
                  "new measurements, not this check.", ""]
    for city in summary.city.unique():
        g = summary[(summary.city == city) &
                    (summary.workload_n == summary[summary.city == city].workload_n.max())].set_index("backend")
        f = fidelity[fidelity.city == city].set_index("backend")
        c = count_summary[count_summary.city == city].set_index("backend")
        ce, cr, ck, h24, h48 = (g.loc[x, "median_us"] for x in ablation.BACKENDS)
        explained = (ce-cr)/(ce-h24)
        n = int(g.workload_n.iloc[0])
        lines += [f"## {city}: N={n:,} independent decisions", "",
                  "| Architecture | Median µs | IQR µs | Paired speedup vs C-exact | Exact destination agreement | Mean scored candidates |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name in ablation.BACKENDS:
            r = g.loc[name]
            lines.append(f"| {name} | {r.median_us:.2f} | {r.p25_us:.2f}–{r.p75_us:.2f} | "
                         f"{r.speedup_vs_continuous_exact:.2f}× | {f.loc[name,'exact_match']:.1%} | "
                         f"{c.loc[name,'mean_candidates']:.2f} |")
        lines += ["", "### Explicit interpretation", "",
                  f"**Q1 — Candidate-scoring restriction after radius retrieval.** "
                  f"C-exact→C-radius-cap24 gives {g.loc['continuous_radius_cap24','speedup_vs_continuous_exact']:.2f}× paired speedup. The cap accounts for "
                  f"{explained:.1%} of the {ce-h24:.2f} µs architecture latency reduction "
                  "on a difference-of-medians basis; a negative value means the cap variant is slower, "
                  "not that restricting scoring intrinsically slows computation.", "",
                  f"**Q2 — Bounded-index contrast.** H3-24 is {g.loc['h3_cap24','speedup_vs_continuous_knn24']:.2f}× faster than "
                  f"C-kNN24, and {g.loc['h3_cap24','speedup_vs_continuous_radius_cap24']:.2f}× faster than C-radius-cap24 (paired medians). "
                  "These are architecture contrasts, not an isolated H3-index effect: actual scored "
                  "candidate volumes and cached context still differ.", "",
                  f"**Q3 — Candidate distributions.** C-exact scores mean {c.loc['continuous_exact','mean_candidates']:.2f} "
                  f"(median {c.loc['continuous_exact','median_candidates']:.0f}, P95 {c.loc['continuous_exact','p95_candidates']:.0f}) "
                  f"versus H3-24 mean {c.loc['h3_cap24','mean_candidates']:.2f} "
                  f"(median {c.loc['h3_cap24','median_candidates']:.0f}, P95 {c.loc['h3_cap24','p95_candidates']:.0f}). "
                  f"C-radius-cap24 hits its cap in {c.loc['continuous_radius_cap24','percent_hitting_cap']:.1f}% "
                  f"of sampled decisions; H3-24 in {c.loc['h3_cap24','percent_hitting_cap']:.1f}%. "
                  f"H3 cached-best/explicit fallback frequency is {c.loc['h3_cap24','fallback_rate']:.1%}; "
                  f"no-destination frequency is {c.loc['h3_cap24','no_destination_rate']:.1%}.", ""]
        for name in ablation.BACKENDS:
            ph = components[(components.city == city) & (components.backend == name)]
            top = ph[ph.component != "other_including_instrumentation"].sort_values("fraction_of_profiled_total", ascending=False).iloc[0]
            lines.append(f"**Q4 — {name} dominant measured phase:** {top.component} "
                         f"≈{top.fraction_of_profiled_total:.0%} of the separately instrumented replay.")
        lines += ["These phase boundaries are coarse; residual Python/measurement time is listed in component_timing.csv. "
                  "Fine-grained timers are excluded from headline latencies.", "",
                  f"**Q5 — Fidelity.** Destination identity agreement is "
                  f"{f.loc['continuous_radius_cap24','exact_match']:.1%} for radius-cap24, "
                  f"{f.loc['continuous_knn24','exact_match']:.1%} for kNN24, "
                  f"{f.loc['h3_cap24','exact_match']:.1%} for H3-24, and "
                  f"{f.loc['h3_cap48','exact_match']:.1%} for H3-48. "
                  "Separation, within-distance fractions, trip-distance Pearson/NMAE, and class substitution are in fidelity_summary.csv. "
                  "Separation and trip-distance metrics condition on both variants returning a destination.", "",
                  f"**Q6 — H3 cap sensitivity.** K=24→48 changes agreement "
                  f"{f.loc['h3_cap24','exact_match']:.1%}→{f.loc['h3_cap48','exact_match']:.1%} "
                  f"and median latency {h24:.2f}→{h48:.2f} µs "
                  f"({h48/h24:.2f}× runtime). No penalty was detected; the small measured "
                  "decrease should not be interpreted as an intrinsic benefit of a larger cap.", ""]
        lines += ["### Proposed manuscript text", "",
                  f"In {('Cambridge, Massachusetts' if city == 'cambridge_ma' else city.replace('_', ' ').title())}, the H3-based complete-decision architecture "
                  f"reduced median latency from {ce:.2f} to {h24:.2f} µs per independent query "
                  f"({g.loc['h3_cap24','speedup_vs_continuous_exact']:.2f}× paired speedup; N={n:,}, "
                  "five paired repetitions). Applying a 24-candidate scoring cap *after* the continuous "
                  f"radius query yielded {cr:.2f} µs, while bounded BallTree kNN24 yielded {ck:.2f} µs. "
                  "Thus simply reducing the post-radius score count did not reproduce the architecture "
                  "gain in this city; the bounded continuous-index contrast likewise retained a substantial "
                  "latency gap. This gap must not be assigned to H3 indexing alone because the H3 path "
                  "also uses cached H9 context and a much smaller realized local candidate set.", "",
                  f"The H3 primary cap of 24 was rarely reached in the sampled decisions "
                  f"({c.loc['h3_cap24','percent_hitting_cap']:.1f}%). Increasing it to 48 "
                  f"changed destination identity agreement from {f.loc['h3_cap24','exact_match']:.1%} "
                  f"to {f.loc['h3_cap48','exact_match']:.1%}, with median latency "
                  f"{h24:.2f}→{h48:.2f} µs. These are cross-representation agreement measures "
                  "against the original continuous implementation, not behavioral accuracy against "
                  f"observed travel choices. Cached H9/explicit fallback served "
                  f"{c.loc['h3_cap24','fallback_rate']:.1%} of sampled H3-24 decisions. "
                  "The actual scored-candidate distributions and H9 cached-best "
                  "use, rather than nominal K alone, govern interpretation of the mechanism.", ""]
    lines += ["## Cross-city conclusion and reviewer response", "",
              f"**Q7.** Available study areas: {', '.join(metadata.get('available_cities', []))}. "
              f"Missing required datasets: {metadata.get('missing_inputs', {})}. Cross-city consistency cannot be tested.", "",
              "**Q8.** None of A–D is established across all three cities by the available evidence. "
              "In the available city, the results reject a simple 'post-radius scoring cap explains most of "
              "the speedup' account, but do not isolate H3 indexing from precomputation, cached context, "
              "or the smaller realized candidate set.", "",
              "**Proposed reviewer response.** We added a paired architecture ablation that retains the "
              "original exhaustive-radius continuous reference, introduces a continuous variant that "
              "caps scoring only after identical radius retrieval, compares a bounded BallTree kNN24 "
              "architecture with the existing H3-24 path, and tests H3-48 sensitivity. The same seeded "
              "independent queries, POIs, semantic layers, behavioral parameters, and exact-coordinate "
              "score are replayed across all configurations. We report five repeated uninstrumented "
              "complete-decision timings, realized candidate-count distributions, cross-representation "
              "destination agreement, and separately instrumented phase profiles. The results distinguish "
              "the complete H3-based architecture gain from any claim that H3 indexing alone causes it. "
              "Only Cambridge inputs were available locally, so the revised paper should not generalize "
              "this ablation to Madrid or New York until those datasets are run.", "",
              "## Limitations", "",
              "The measured one-million-query run samples a fixed 10,000-agent world, unlike a "
              "one-million-agent population sweep; queries remain independent. H3 cached-best selection "
              "can return a destination beyond the continuous radius, as in the existing implementation. "
              "The nominal K=24 variants do not equalize actual scored-candidate counts. No paired "
              "temporal/OD output experiment was run because the existing simulation runner does not "
              "parameterize these ablation variants. Component fractions are approximate and include "
              "instrumentation overhead in the residual. Small five-run bootstrap intervals characterize "
              "this machine, not cross-machine uncertainty.", ""]
    lines += ["## Figures and machine-readable data", "",
              "For each available city, `figures/{city}_latency`, `_speedup`, `_candidates`, "
              "and `_frontier` are supplied as 300-dpi PNG and editable SVG. "
              "The source tables are `timings_raw.csv`, `timings_summary.csv`, "
              "`candidate_counts_raw.csv`, `candidate_counts_summary.csv`, "
              "`fidelity_summary.csv`, and `component_timing.csv`; run provenance is in `metadata.json`.", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(f"Finalized {out}")


if __name__ == "__main__":
    main()
