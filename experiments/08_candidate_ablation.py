#!/usr/bin/env python3
"""Paired, independent-query architecture ablation (routing excluded)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from h3uab.benchmarking import make_query_set
from h3uab.city_config import config_for_city
from h3uab.continuous_indexed_backend import ContinuousIndexedBackend
from h3uab.geometry import haversine_m
from h3uab.h3_backend import MultiLayerH3Backend
from h3uab.robust_metrics import nmae
from h3uab.world import build_world

BACKENDS = ("continuous_exact", "continuous_radius_cap24", "continuous_knn24",
            "h3_cap24", "h3_cap48")
PLOT_LABELS = {"continuous_exact": "C-exact", "continuous_radius_cap24": "C-radius-cap24",
               "continuous_knn24": "C-kNN24", "h3_cap24": "H3-24", "h3_cap48": "H3-48"}
FIELDS = ("returned", "requested", "eligible", "retained", "scored", "buckets",
          "raw_pois", "fallback", "cached_best", "fallback_inspected", "fallback_scored")


def environment(seed):
    def version(name):
        try:
            module = __import__(name)
            return getattr(module, "__version__", "unknown")
        except ImportError:
            return None
    cpuinfo = Path("/proc/cpuinfo")
    cpu_model = (next((x.split(":", 1)[1].strip() for x in cpuinfo.read_text().splitlines()
                       if x.startswith("model name")), platform.processor() or "unknown")
                 if cpuinfo.exists() else platform.processor() or "unknown")
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                os=platform.platform(), cpu_model=cpu_model,
                logical_cores=psutil.cpu_count(), physical_cores=psutil.cpu_count(logical=False),
                ram_bytes=psutil.virtual_memory().total, python=sys.version,
                versions={x: version(x) for x in ("numpy", "pandas", "scipy", "sklearn", "h3",
                                                  "matplotlib", "geopandas", "shapely", "rasterio")},
                threads={x: os.environ.get(x) for x in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                       "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}, seed=seed)


def prepare(world, cfg, name):
    if name.startswith("continuous"):
        strategy = {"continuous_exact": "radius_exact", "continuous_radius_cap24": "radius_cap",
                    "continuous_knn24": "knn"}[name]
        local_cfg = replace(cfg, baseline=replace(cfg.baseline, candidate_strategy=strategy,
                                                   candidate_cap=None if strategy == "radius_exact" else 24))
        backend = ContinuousIndexedBackend()
    else:
        cap = int(name.removeprefix("h3_cap"))
        local_cfg = replace(cfg, destination_refinement=replace(cfg.destination_refinement,
                                                                 max_candidates=cap))
        backend = MultiLayerH3Backend()
    backend.prepare(world, local_cfg)
    return backend


def replay(world, backend, q, limit=None, diagnostics=False):
    n = len(q.agent_ids) if limit is None else min(limit, len(q.agent_ids))
    ids = np.empty(n, dtype=np.int64) if diagnostics else None
    trips = np.empty(n, dtype=float) if diagnostics else None
    rows = []
    checksum = 0
    for i in range(n):
        aid = int(q.agent_ids[i])
        trace = {} if diagnostics else None
        if diagnostics:
            backend.trace = trace
        d = backend.decide(aid, float(world.agent_lat[aid]), float(world.agent_lng[aid]),
                           str(q.layer_codes[i]))
        checksum += int(d.destination_id)
        if diagnostics:
            ids[i] = d.destination_id
            trips[i] = d.straight_distance_m if d.destination_id >= 0 else np.nan
            rows.append(dict(query=i, **{k: trace.get(k, 0) for k in FIELDS},
                             collection_time_s=trace.get("collection_time_s", 0.0),
                             fallback_time_s=trace.get("fallback_time_s", 0.0),
                             no_destination=int(d.destination_id < 0)))
    if diagnostics:
        backend.trace = None
    return checksum, ids, trips, rows


def stats(a):
    x = np.asarray(a, dtype=float)
    return dict(mean=float(np.mean(x)), median=float(np.median(x)), std=float(np.std(x)),
                p25=float(np.percentile(x, 25)), p75=float(np.percentile(x, 75)),
                p90=float(np.percentile(x, 90)), p95=float(np.percentile(x, 95)),
                p99=float(np.percentile(x, 99)), max=float(np.max(x)))


def fidelity(world, reference, alternative):
    ref_ids, ref_trips = reference
    alt_ids, alt_trips = alternative
    both = (ref_ids >= 0) & (alt_ids >= 0)
    separation = np.full(len(ref_ids), np.nan)
    separation[both] = haversine_m(world.amenity_lat[ref_ids[both]],
                                   world.amenity_lng[ref_ids[both]],
                                   world.amenity_lat[alt_ids[both]],
                                   world.amenity_lng[alt_ids[both]])
    valid = separation[np.isfinite(separation)]
    denominator = max(int(both.sum()), 1)
    r = np.corrcoef(ref_trips[both], alt_trips[both])[0, 1] if both.sum() > 1 and np.std(ref_trips[both]) and np.std(alt_trips[both]) else np.nan
    return dict(exact_match=float(np.mean(ref_ids == alt_ids)),
                separation_mean_m=float(np.mean(valid)) if len(valid) else np.nan,
                separation_median_m=float(np.median(valid)) if len(valid) else np.nan,
                separation_p90_m=float(np.percentile(valid, 90)) if len(valid) else np.nan,
                separation_p95_m=float(np.percentile(valid, 95)) if len(valid) else np.nan,
                **{f"within_{d}m": float(np.sum(valid <= d) / denominator) for d in (100, 250, 500, 1000)},
                trip_distance_r=float(r),
                trip_distance_nmae=nmae(ref_trips[both], alt_trips[both]) if both.sum() else np.nan,
                compared_destinations=int(both.sum()), reference_only=int(np.sum((ref_ids >= 0) & (alt_ids < 0))),
                variant_only=int(np.sum((ref_ids < 0) & (alt_ids >= 0))),
                semantic_class_difference=float(np.mean(world.amenity_layer_code[ref_ids[both]] !=
                                                         world.amenity_layer_code[alt_ids[both]])) if both.sum() else np.nan)


def summarize(timings, counts):
    td = pd.DataFrame(timings)
    for ref in ("continuous_exact", "continuous_radius_cap24", "continuous_knn24"):
        key = td[td.backend == ref].set_index(["city", "workload_n", "repetition"])["microseconds_per_decision"]
        td[f"speedup_vs_{ref}"] = [key.loc[(r.city, r.workload_n, r.repetition)] / r.microseconds_per_decision
                                      for r in td.itertuples()]
    summaries = []
    for (city, name, n), g in td.groupby(["city", "backend", "workload_n"], sort=False):
        x = g.microseconds_per_decision.to_numpy()
        rng = np.random.default_rng(109)
        boot = np.median(rng.choice(x, size=(2000, len(x)), replace=True), axis=1)
        paired = g.speedup_vs_continuous_exact.to_numpy()
        paired_boot = np.median(rng.choice(paired, size=(2000, len(paired)), replace=True), axis=1)
        summaries.append(dict(city=city, backend=name, workload_n=n, median_us=np.median(x),
                              p25_us=np.percentile(x, 25), p75_us=np.percentile(x, 75),
                              min_us=np.min(x), max_us=np.max(x),
                              speedup_vs_continuous_exact=np.median(g.speedup_vs_continuous_exact),
                              speedup_p25=np.percentile(paired, 25),
                              speedup_p75=np.percentile(paired, 75),
                              speedup_min=np.min(paired), speedup_max=np.max(paired),
                              speedup_vs_continuous_radius_cap24=np.median(g.speedup_vs_continuous_radius_cap24),
                              speedup_vs_continuous_knn24=np.median(g.speedup_vs_continuous_knn24),
                              ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
                              speedup_ci_low=np.percentile(paired_boot, 2.5),
                              speedup_ci_high=np.percentile(paired_boot, 97.5)))
    cd = pd.DataFrame(counts)
    count_summary = []
    for (city, name), g in cd.groupby(["city", "backend"], sort=False):
        s = stats(g.scored)
        cap = 24 if name.endswith("24") else (48 if name.endswith("48") else None)
        detailed = {f"{field}_{stat}": value for field in FIELDS
                    for stat, value in stats(g[field]).items()}
        count_summary.append(dict(city=city, backend=name, mean_candidates=s["mean"],
                                  median_candidates=s["median"], std_candidates=s["std"],
                                  p25_candidates=s["p25"], p75_candidates=s["p75"],
                                  p90_candidates=s["p90"], p95_candidates=s["p95"],
                                  p99_candidates=s["p99"], max_candidates=s["max"],
                                  percent_hitting_cap=100 * np.mean(g.retained >= cap) if cap else np.nan,
                                  fallback_rate=np.mean(g.fallback), no_destination_rate=np.mean(g.no_destination),
                                  **detailed))
    return td, pd.DataFrame(summaries), pd.DataFrame(count_summary)


def figures(out, summary, counts, fidelity_df):
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    def save(fig, stem):
        fig.tight_layout()
        fig.savefig(figdir / f"{stem}.png", dpi=300)
        svg_path = figdir / f"{stem}.svg"
        fig.savefig(svg_path)
        # Matplotlib's multiline SVG paths include trailing spaces; keep
        # publication artifacts clean for git diff --check.
        svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
        plt.close(fig)
    for city in summary.city.unique():
        g = summary[(summary.city == city) & (summary.workload_n == summary[summary.city == city].workload_n.max())].set_index("backend").reindex(BACKENDS)
        if g.median_us.isna().all():
            continue
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(range(len(g)), g.median_us, color=["#566573", "#5d8aa8", "#79a6ad", "#c06c42", "#d59573"])
        ax.errorbar(range(len(g)), g.median_us, yerr=[g.median_us-g.ci_low, g.ci_high-g.median_us], fmt="none", color="black", capsize=4)
        ax.set_xticks(range(len(g)), [PLOT_LABELS[x] for x in g.index], rotation=15, ha="right")
        ax.set_ylabel("Decision latency (µs; median, bootstrap 95% CI)")
        ax.set_title(city)
        save(fig, f"{city}_latency")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(range(1, len(g)), g.speedup_vs_continuous_exact.iloc[1:], color=["#5d8aa8", "#79a6ad", "#c06c42", "#d59573"])
        ax.errorbar(range(1, len(g)), g.speedup_vs_continuous_exact.iloc[1:],
                    yerr=[g.speedup_vs_continuous_exact.iloc[1:] - g.speedup_ci_low.iloc[1:],
                          g.speedup_ci_high.iloc[1:] - g.speedup_vs_continuous_exact.iloc[1:]],
                    fmt="none", color="black", capsize=4)
        ax.axhline(1, color="black", linewidth=.8)
        ax.set_xticks(range(1, len(g)), [PLOT_LABELS[x] for x in g.index[1:]], rotation=15, ha="right")
        ax.set_ylabel("Median paired speedup vs C-exact")
        ax.set_title(f"{city}: H3-24 speedup over C-kNN24 = {g.loc['h3_cap24','speedup_vs_continuous_knn24']:.2f}×")
        save(fig, f"{city}_speedup")
        c = counts[(counts.city == city) & counts.backend.isin(BACKENDS[:4])]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.boxplot([c[c.backend == name].scored for name in BACKENDS[:4]], showfliers=False,
                   tick_labels=[PLOT_LABELS[x] for x in BACKENDS[:4]])
        ax.set_ylabel("Candidates actually scored (sampled queries)")
        ax.tick_params(axis="x", rotation=15)
        save(fig, f"{city}_candidates")
        f = fidelity_df[fidelity_df.city == city].set_index("backend")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        label_offsets = {"continuous_exact": (-8, -17),
                         "continuous_radius_cap24": (-8, 10),
                         "continuous_knn24": (-8, -17),
                         "h3_cap24": (8, 10), "h3_cap48": (8, -17)}
        for i, name in enumerate(BACKENDS):
            ax.scatter(g.loc[name, "median_us"], f.loc[name, "exact_match"], s=70)
            ax.annotate(PLOT_LABELS[name], (g.loc[name, "median_us"], f.loc[name, "exact_match"]),
                        xytext=label_offsets[name], textcoords="offset points", fontsize=8,
                        ha="right" if name.startswith("continuous") else "left")
        ax.set_xlabel("Median decision latency (µs)"); ax.set_ylabel("Destination identity agreement with C-exact")
        latency_span = max(float(g.median_us.max() - g.median_us.min()), 1.0)
        ax.set_xlim(max(0, float(g.median_us.min()) - .1 * latency_span),
                    float(g.median_us.max()) + .1 * latency_span)
        ax.set_ylim(max(0, float(f.exact_match.min()) - .05), 1.035)
        save(fig, f"{city}_frontier")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", nargs="+", default=["madrid", "cambridge_ma", "new_york_ny"])
    p.add_argument("--workloads", nargs="+", type=int, default=[1_000_000, 100_000, 10_000, 500_000])
    p.add_argument("--agents", type=int, default=10_000, help="Fixed world population; queries are independent draws")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--diagnostic-queries", type=int, default=10_000)
    p.add_argument("--output", type=Path, default=ROOT / "results" / "candidate_ablation")
    args = p.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    metadata = environment(42)
    metadata.update(agents=args.agents, repeats=args.repeats, workloads=args.workloads,
                    requested_cities=args.cities, available_cities=[], missing_inputs={},
                    scope="independent complete decisions; no routing or movement",
                    query_seed_offset=7001,
                    diagnostic_queries=args.diagnostic_queries)
    timings, counts, fidelity_rows, components = [], [], [], []
    for city in args.cities:
        cfg = config_for_city(city, n_agents=args.agents)
        required = [cfg.osm_cache_csv, cfg.population_h3_csv]
        missing = [str(x) for x in required if not x.exists()]
        if missing:
            metadata["missing_inputs"][city] = missing
            print(f"SKIP {city}: missing {missing}", flush=True)
            continue
        print(f"BUILD {city}", flush=True)
        world, world_meta = build_world(cfg)
        metadata["available_cities"].append(city)
        metadata[f"world_{city}"] = world_meta
        metadata[f"inputs_{city}"] = {
            str(path): {"bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in required
        }
        backends = {name: prepare(world, cfg, name) for name in BACKENDS}
        q_diag = make_query_set(world, cfg, args.diagnostic_queries)
        results = {}
        for name, backend in backends.items():
            before = {k: getattr(backend.profile, k, 0) for k in ("lookup_time_s", "distance_time_s", "decision_time_s")}
            t0 = perf_counter()
            _, dest, trips, rows = replay(world, backend, q_diag, diagnostics=True)
            total = perf_counter() - t0
            results[name] = (dest, trips)
            counts.extend(dict(city=city, backend=name, **row) for row in rows)
            phase = {k: getattr(backend.profile, k, 0) - before[k] for k in before}
            if name.startswith("h3"):
                phase["candidate_collection_s"] = sum(r["collection_time_s"] for r in rows)
                phase["fallback_s"] = sum(r["fallback_time_s"] for r in rows)
            accounted = sum(phase.values())
            components.extend(dict(city=city, backend=name, component=k, seconds=v,
                                   fraction_of_profiled_total=v/total, profiled_total_seconds=total)
                              for k, v in [*phase.items(), ("other_including_instrumentation", max(0, total-accounted))])
        for name in BACKENDS:
            fidelity_rows.append(dict(city=city, backend=name, diagnostic_n=len(q_diag.agent_ids),
                                      **fidelity(world, results["continuous_exact"], results[name])))
        for n in args.workloads:
            q = make_query_set(world, cfg, n)
            for backend in backends.values():
                replay(world, backend, q, limit=min(1000, n))
            for repetition in range(args.repeats):
                order = np.random.default_rng(cfg.seed + 10000 + repetition).permutation(BACKENDS)
                for name in order:
                    backend = backends[name]
                    t0 = perf_counter(); checksum, _, _, _ = replay(world, backend, q)
                    elapsed = perf_counter() - t0
                    row = dict(city=city, backend=name,
                               candidate_strategy=(backend.candidate_strategy if name.startswith("continuous") else "h10_buckets"),
                               candidate_cap=(backend.candidate_cap if name.startswith("continuous") else backend.cfg.destination_refinement.max_candidates),
                               fallback_cap=(backend.cfg.destination_refinement.fallback_max_candidates if name.startswith("h3") else None),
                               workload_n=n, repetition=repetition, total_seconds=elapsed,
                               microseconds_per_decision=elapsed/n*1e6, throughput=n/elapsed,
                               seed=cfg.seed, checksum=checksum)
                    timings.append(row)
                    print(f"{city} N={n} rep={repetition} {name}: {row['microseconds_per_decision']:.2f} µs", flush=True)
            td, ts, cs = summarize(timings, counts)
            td.to_csv(out / "timings_raw.csv", index=False)
            ts.to_csv(out / "timings_summary.csv", index=False)
            pd.DataFrame(counts).to_csv(out / "candidate_counts_raw.csv", index=False)
            cs.to_csv(out / "candidate_counts_summary.csv", index=False)
            pd.DataFrame(fidelity_rows).to_csv(out / "fidelity_summary.csv", index=False)
            pd.DataFrame(components).to_csv(out / "component_timing.csv", index=False)
            (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    if timings:
        td, ts, cs = summarize(timings, counts)
        fd = pd.DataFrame(fidelity_rows)
        figures(out, ts, pd.DataFrame(counts), fd)
        lines = ["# Candidate-architecture ablation", "", "Continuous-exact is a cross-representation reference, not behavioral truth.",
                 "Primary latencies are uninstrumented; component fractions are from a separate diagnostic replay.",
                 "Continuous radius-cap retains the full radius query and selects 24 nearest eligible POIs by exact distance (ID tie-break).",
                 "Continuous kNN requests 24 nearest POIs from the same per-layer BallTree, then applies the original exact radius; it does not enumerate all within-radius POIs.",
                 "H3 candidate ordering and H9 fallback are unchanged from the production implementation.",
                 "The scored-candidate sample uses a separate identically seeded independent workload; H3 cached-best fallback may return a destination without scoring a candidate in the query.", ""]
        for city in ts.city.unique():
            g = ts[(ts.city == city) & (ts.workload_n == ts[ts.city == city].workload_n.max())].set_index("backend")
            f = fd[fd.city == city].set_index("backend")
            c = cs[cs.city == city].set_index("backend")
            ce, cr, ck, h24, h48 = (g.loc[x, "median_us"] for x in BACKENDS)
            explained = (ce-cr)/(ce-h24) if ce != h24 else np.nan
            lines += [f"## {city}", "", f"Largest measured workload: {int(g.workload_n.iloc[0]):,} independent queries; {args.repeats} paired repetitions.",
                      f"Q1. C-exact→C-radius-cap24: {ce/cr:.2f}×; {explained:.1%} of C-exact→H3-24 latency reduction (difference-of-medians descriptive estimate).",
                      f"Q2. H3-24/C-kNN24: {ck/h24:.2f}×; H3-24/C-radius-cap24: {cr/h24:.2f}×. These are architecture contrasts, not isolated H3-index effects.",
                      f"Q3. Scored candidates, mean: C-exact {c.loc['continuous_exact','mean_candidates']:.1f}; H3-24 {c.loc['h3_cap24','mean_candidates']:.1f}. See distribution CSV/figure.",
                      "Q4. Phase fractions are in component_timing.csv. H3 distance_time includes scoring; residual includes Python orchestration and instrumentation.",
                      f"Q5. Destination agreement: C-radius-cap24 {f.loc['continuous_radius_cap24','exact_match']:.1%}, C-kNN24 {f.loc['continuous_knn24','exact_match']:.1%}, H3-24 {f.loc['h3_cap24','exact_match']:.1%}.",
                      f"Q6. H3 K=24→48 agreement {f.loc['h3_cap24','exact_match']:.1%}→{f.loc['h3_cap48','exact_match']:.1%}; median latency {h24:.2f}→{h48:.2f} µs.", ""]
        lines += ["## Cross-city interpretation", "", "Q7. Compare the city-specific controlled ratios above; unavailable cities are not inferred.",
                  "Q8. No single A–D mechanism claim is assigned without all study areas. Interpret the controlled contrasts above directly.",
                  f"Missing inputs: {metadata['missing_inputs']}", "",
                  "## Limitations", "", "The query population is fixed by --agents and may differ from the manuscript's N-agent sweeps. kNN distance filtering preserves the maximum radius but truncates accessible candidates. H3's cached H9 best-POI fallback can choose a POI outside the continuous radius; this production behavior is intentionally retained. Diagnostic timing includes trace overhead and is only approximate. Bootstrap intervals describe measured runs, not machine-independent performance."]
        (out / "report.md").write_text("\n".join(lines) + "\n")
    # City skips can occur after the last measured city; persist them too.
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    main()
