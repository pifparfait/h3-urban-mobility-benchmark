from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import gc
import tracemalloc

import h3
import numpy as np
import pandas as pd

from h3uab.geometry import haversine_m
from h3uab.semantic_layers import id_to_code


@dataclass(frozen=True)
class QuerySet:
    agent_ids: np.ndarray
    layer_codes: np.ndarray


def make_query_set(world, cfg, n_queries: int, seed_offset: int = 7001) -> QuerySet:
    rng = np.random.default_rng(cfg.seed + seed_offset)
    active_ids = np.asarray(
        sorted({
            int(x)
            for x in world.profile_schedule_layer_ids.ravel()
            if int(x) > 0
        }),
        dtype=np.int64,
    )
    agent_ids = rng.integers(0, len(world.agent_lat), size=int(n_queries))
    layer_ids = rng.choice(active_ids, size=int(n_queries))
    layer_codes = np.asarray(
        [id_to_code(int(x)) for x in layer_ids],
        dtype=object,
    )
    return QuerySet(agent_ids=agent_ids, layer_codes=layer_codes)


def _measure(fn, repeats=5, warmup=1):
    for _ in range(int(warmup)):
        fn()
    gc.collect()
    samples = []
    for _ in range(int(repeats)):
        t0 = perf_counter()
        fn()
        samples.append(perf_counter() - t0)
    a = np.asarray(samples, dtype=float)
    return {
        "median_s": float(np.median(a)),
        "mean_s": float(np.mean(a)),
        "std_s": float(np.std(a)),
        "min_s": float(np.min(a)),
        "max_s": float(np.max(a)),
    }


def _row(operation, backend, n, stats):
    med = stats["median_s"]
    return {
        "operation": operation,
        "backend": backend,
        "queries": int(n),
        **stats,
        "median_us_per_query": 1e6 * med / max(int(n), 1),
        "queries_per_second": int(n) / med if med > 0 else np.nan,
    }


def benchmark_pure_lookup(world, h3_backend, queries, repeats=5):
    """
    Pure state-addressing benchmark.

    Continuous:
      coordinate -> BallTree radius lookup.

    H3:
      already-maintained H12 agent cell -> H9 accessibility dictionary lookup.

    H12 is NOT a fidelity layer. It is only the maintained agent address used
    to reach the semantic H9 state in O(1).
    """
    ids = queries.agent_ids
    layers = queries.layer_codes

    def continuous():
        total = 0
        for aid, layer in zip(ids, layers):
            tree = h3_backend._benchmark_continuous_trees[layer]
            q = np.radians([[world.agent_lat[aid], world.agent_lng[aid]]])
            total += len(tree.query_radius(
                q,
                r=h3_backend._benchmark_radius_rad,
                return_distance=False,
            )[0])
        return total

    def h3_lookup():
        total = 0.0
        for aid, layer in zip(ids, layers):
            h12 = h3_backend.dynamic.position(int(aid))
            state = h3_backend.city.get_accessibility(h12, str(layer))
            if state is not None:
                total += state.weighted_accessibility
        return total

    rows = [
        _row("pure_spatial_lookup", "continuous_indexed", len(ids),
             _measure(continuous, repeats=repeats)),
        _row("pure_spatial_lookup", "h3", len(ids),
             _measure(h3_lookup, repeats=repeats)),
    ]
    return pd.DataFrame(rows)


def benchmark_context_retrieval(world, indexed_backend, h3_backend, queries, cfg, repeats=5):
    """
    Retrieve the information needed for an urban decision, without scoring.

    Continuous-indexed:
      BallTree candidates + exact distances + accessibility aggregation.

    H3:
      H5 macro + H8 amenity aggregate + H9 accessibility state +
      bounded H10 candidate bucket retrieval.
    """
    ids = queries.agent_ids
    layers = queries.layer_codes

    def continuous():
        acc = 0.0
        for aid, layer in zip(ids, layers):
            tree = indexed_backend.trees[str(layer)]
            poi_ids = indexed_backend.by_layer[str(layer)]
            q = np.radians([[world.agent_lat[aid], world.agent_lng[aid]]])
            local = tree.query_radius(
                q,
                r=cfg.search_radius_m / 6_371_008.8,
                return_distance=False,
            )[0]
            cand = poi_ids[local]
            if cand.size:
                d = haversine_m(
                    float(world.agent_lat[aid]),
                    float(world.agent_lng[aid]),
                    world.amenity_lat[cand],
                    world.amenity_lng[cand],
                )
                keep = d <= cfg.search_radius_m
                cand, d = cand[keep], d[keep]
                if cand.size:
                    acc += float(np.sum(
                        world.amenity_attractiveness[cand]
                        * np.exp(-d / cfg.accessibility_decay_m)
                    ))
        return acc

    def h3_context():
        acc = 0.0
        for aid, layer in zip(ids, layers):
            h12 = h3_backend.dynamic.position(int(aid))
            macro = h3_backend.city.get_macro(h12)
            amenity = h3_backend.city.get_amenity_aggregate(
                h12, str(layer), cfg.h3.amenity_functional_res
            )
            access = h3_backend.city.get_accessibility(h12, str(layer))
            bucket = h3_backend.city.get_poi_bucket(h12, str(layer))
            if macro is not None:
                acc += macro.population * 0.0
            if amenity is not None:
                acc += amenity.attractiveness_sum * 0.0
            if access is not None:
                acc += access.weighted_accessibility
            acc += len(bucket) * 0.0
        return acc

    return pd.DataFrame([
        _row("urban_context_retrieval", "continuous_indexed", len(ids),
             _measure(continuous, repeats=repeats)),
        _row("urban_context_retrieval", "h3", len(ids),
             _measure(h3_context, repeats=repeats)),
    ])


def benchmark_accessibility_query(world, indexed_backend, h3_backend, queries, cfg, repeats=5):
    ids = queries.agent_ids
    layers = queries.layer_codes

    def continuous():
        out = 0.0
        for aid, layer in zip(ids, layers):
            tree = indexed_backend.trees[str(layer)]
            poi_ids = indexed_backend.by_layer[str(layer)]
            q = np.radians([[world.agent_lat[aid], world.agent_lng[aid]]])
            local = tree.query_radius(
                q,
                r=cfg.search_radius_m / 6_371_008.8,
                return_distance=False,
            )[0]
            cand = poi_ids[local]
            if cand.size:
                d = haversine_m(
                    float(world.agent_lat[aid]),
                    float(world.agent_lng[aid]),
                    world.amenity_lat[cand],
                    world.amenity_lng[cand],
                )
                keep = d <= cfg.search_radius_m
                out += float(np.sum(
                    world.amenity_attractiveness[cand[keep]]
                    * np.exp(-d[keep] / cfg.accessibility_decay_m)
                ))
        return out

    def h3_access():
        out = 0.0
        for aid, layer in zip(ids, layers):
            h12 = h3_backend.dynamic.position(int(aid))
            state = h3_backend.city.get_accessibility(h12, str(layer))
            if state is not None:
                out += state.weighted_accessibility
        return out

    return pd.DataFrame([
        _row("accessibility_query", "continuous_indexed", len(ids),
             _measure(continuous, repeats=repeats)),
        _row("accessibility_query", "h3", len(ids),
             _measure(h3_access, repeats=repeats)),
    ])


def benchmark_candidate_retrieval(world, indexed_backend, h3_backend, queries, cfg, repeats=5):
    ids = queries.agent_ids
    layers = queries.layer_codes

    def continuous():
        total = 0
        for aid, layer in zip(ids, layers):
            tree = indexed_backend.trees[str(layer)]
            q = np.radians([[world.agent_lat[aid], world.agent_lng[aid]]])
            total += len(tree.query_radius(
                q,
                r=cfg.search_radius_m / 6_371_008.8,
                return_distance=False,
            )[0])
        return total

    def h3_candidates():
        total = 0
        for aid, layer in zip(ids, layers):
            h12 = h3_backend.dynamic.position(int(aid))
            total += len(h3_backend.city.get_poi_bucket(h12, str(layer)))
        return total

    return pd.DataFrame([
        _row("candidate_retrieval", "continuous_indexed", len(ids),
             _measure(continuous, repeats=repeats)),
        _row("candidate_retrieval", "h3", len(ids),
             _measure(h3_candidates, repeats=repeats)),
    ])


def benchmark_full_decision(world, indexed_backend, h3_backend, queries, repeats=5):
    ids = queries.agent_ids
    layers = queries.layer_codes

    def run(backend):
        checksum = 0
        for aid, layer in zip(ids, layers):
            d = backend.decide(
                int(aid),
                float(world.agent_lat[aid]),
                float(world.agent_lng[aid]),
                str(layer),
            )
            checksum += int(d.destination_id)
        return checksum

    return pd.DataFrame([
        _row("full_agent_decision", "continuous_indexed", len(ids),
             _measure(lambda: run(indexed_backend), repeats=repeats)),
        _row("full_agent_decision", "h3", len(ids),
             _measure(lambda: run(h3_backend), repeats=repeats)),
    ])


def add_speedups(df):
    rows = []
    for operation, g in df.groupby("operation", sort=False):
        c = g[g.backend == "continuous_indexed"]
        h = g[g.backend == "h3"]
        if c.empty or h.empty:
            continue
        c = c.iloc[0]
        h = h.iloc[0]
        rows.append({
            "operation": operation,
            "queries": int(c["queries"]),
            "continuous_median_s": float(c["median_s"]),
            "h3_median_s": float(h["median_s"]),
            "continuous_us_per_query": float(c["median_us_per_query"]),
            "h3_us_per_query": float(h["median_us_per_query"]),
            "continuous_qps": float(c["queries_per_second"]),
            "h3_qps": float(h["queries_per_second"]),
            "speedup_h3_over_continuous": (
                float(c["median_s"]) / float(h["median_s"])
                if float(h["median_s"]) > 0 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def prepare_pure_lookup_helpers(indexed_backend, h3_backend, cfg):
    # Stored on H3 backend only to keep the pure benchmark call compact.
    h3_backend._benchmark_continuous_trees = indexed_backend.trees
    h3_backend._benchmark_radius_rad = cfg.search_radius_m / 6_371_008.8
