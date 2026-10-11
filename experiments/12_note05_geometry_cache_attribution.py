#!/usr/bin/env python3
"""Note 05 Script 12: fail-closed geometry and H9-cache attribution.

Read-only on existing experiment data. No simulation or static-cache rebuild.
Run from repository root, with original trace directories intact.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import h3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from h3uab.behavior import destination_score
from h3uab.city_config import config_for_city
from h3uab.geometry import haversine_m
from h3uab.h3_utils import radius_to_k
from h3uab.semantic_layers import id_to_code
from h3uab.static_city import get_h10_candidate_poi_ids, load_or_build_static_city
from h3uab.world import build_world

CITIES = {
    'madrid': ('madrid_traced', 'madrid'),
    'cambridge_ma': ('cambridge_traced', 'cambridge'),
    'new_york_ny': ('nyc_traced', 'nyc'),
}
KEY = ['agent_id', 'step']


def require(ok, reason, row=None, **details):
    if not ok:
        ident = '' if row is None else f' agent={int(row.agent_id)} step={int(row.step)}'
        raise AssertionError(reason + ident + ' ' + json.dumps(details, default=str))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def bucket_ids(index, cell, layer):
    if (cell, layer) in index:
        value = index[(cell, layer)]
    elif cell in index:
        entry = index[cell]
        value = entry.get(layer) if isinstance(entry, dict) else entry
    else:
        value = None
    if value is None:
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(v) for v in value]


def recover(index, cells, layer, cap):
    raw, full = [], []
    for cell in cells:
        raw.extend(bucket_ids(index, cell, layer))
        if len(raw) >= cap:
            break
    for cell in cells:
        full.extend(bucket_ids(index, cell, layer))
    return tuple(dict.fromkeys(raw))[:cap], tuple(dict.fromkeys(full))


def score(ids, lat, lng, world, cfg, radius=True):
    ids = np.asarray(ids, dtype=np.int64)
    require(np.all((ids >= 0) & (ids < len(world.amenity_lat))), 'INVALID POI IDS')
    if not ids.size:
        return -1, np.empty(0, dtype=np.int64), np.empty(0)
    dist = np.asarray(haversine_m(float(lat), float(lng), world.amenity_lat[ids], world.amenity_lng[ids]))
    if radius:
        keep = dist <= cfg.search_radius_m
        ids, dist = ids[keep], dist[keep]
    if not ids.size:
        return -1, ids, dist
    values = destination_score(dist, world.amenity_attractiveness[ids], cfg.beta_distance, cfg.gamma_attractiveness)
    return int(ids[int(np.argmax(values))]), ids, dist


def summary_dist(values):
    if not values:
        return {'n': 0, 'p50_m': None, 'p90_m': None, 'p99_m': None, 'max_m': None}
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'p50_m': float(np.quantile(a, .5)), 'p90_m': float(np.quantile(a, .9)),
            'p99_m': float(np.quantile(a, .99)), 'max_m': float(a.max())}


def process(city_name, agents, base):
    trace_name, cf_name = CITIES[city_name]
    ep_path = base / 'note05_paired_trace' / trace_name / f'diagnostics_n{agents}' / 'paired_trip_endpoints.csv'
    trace_path = base / 'note05_paired_trace' / trace_name / 'note05_trace.csv'
    cf_path = base / 'note05_counterfactual' / cf_name / 'per_decision.csv'
    for path in (ep_path, trace_path, cf_path):
        require(path.is_file(), 'MISSING INPUT', path=str(path))
    ep, trace, cf = (pd.read_csv(p) for p in (ep_path, trace_path, cf_path))
    require(len(ep) == agents * 5, 'UNEXPECTED ENDPOINT ROW COUNT', city=city_name, n=len(ep))
    for name, df, cols in [('endpoints', ep, KEY), ('trace', trace, KEY + ['backend']), ('cf', cf, KEY)]:
        require(not df.duplicated(cols).any(), 'DUPLICATE KEYS', table=name)
    htrace = trace.loc[trace.backend.eq('h3'), KEY + ['route', 'destination_id']]
    require(len(htrace) == len(ep), 'INCOMPLETE H3 TRACE')
    x = ep.merge(htrace, on=KEY, validate='one_to_one', suffixes=('', '_trace'))
    require(len(x) == len(ep), 'MISSING TRACE ROWS')
    require(x.h3_destination_id.astype('int64').eq(x.destination_id.astype('int64')).all(), 'TRACE/PUBLISHED DESTINATION MISMATCH')
    require(x.route.isin(['local_h10', 'cached_best_h9', 'bounded_fallback_h9', 'none']).all(), 'UNKNOWN ROUTE')
    x['shared'] = x.continuous_origin_lat.eq(x.h3_origin_lat) & x.continuous_origin_lng.eq(x.h3_origin_lng)
    require(x.loc[x.step.eq(1), 'shared'].all(), 'STEP1 ORIGIN INVARIANT')
    cfg = config_for_city(city_name, n_agents=agents)
    require(cfg.search_radius_m == 2000 and cfg.h3.amenity_functional_res == 8 and
            cfg.h3.accessibility_res == 9 and cfg.h3.amenity_index_res == 10 and
            int(cfg.destination_refinement.h10_k_ring) == 1 and
            int(cfg.destination_refinement.max_candidates) == 24 and
            radius_to_k(cfg.search_radius_m, 8) == 4, 'CONFIGURATION DRIFT')
    require(not cfg.rebuild_static_city and Path(cfg.static_city_cache).is_file(),
            'STATIC CACHE MISSING OR REBUILD ENABLED', path=str(cfg.static_city_cache))
    world, _ = build_world(cfg)
    static, _, mode = load_or_build_static_city(world, cfg)
    require(mode == 'loaded', 'STATIC CACHE NOT LOADED')
    npoi = len(world.amenity_lat)
    layers = defaultdict(list)
    h8_index = defaultdict(list)
    poi_h10 = []
    for pid in range(npoi):
        layer = str(world.amenity_layer_code[pid])
        layers[layer].append(pid)
        h10 = h3.latlng_to_cell(float(world.amenity_lat[pid]), float(world.amenity_lng[pid]), 10)
        poi_h10.append(h10)
        h8_index[(h3.cell_to_parent(h10, 8), layer)].append(pid)
    # Historical H10 buckets must agree with the rebuilt POI-to-cell assignment.
    for (cell, layer), ids in h8_index.items():
        require(all(h3.cell_to_parent(poi_h10[p], 8) == cell for p in ids), 'H8 INDEX INVARIANT')
    rows_local, rows_cache = [], []
    selected = x.loc[x.shared & x.route.isin(['local_h10', 'cached_best_h9'])].copy()
    local_cf = cf.set_index(KEY)
    require(len(cf) == int(x.route.eq('local_h10').sum()), 'COUNTERFACTUAL COVERAGE')
    require(set(map(tuple, cf[KEY].to_numpy())) == set(map(tuple, x.loc[x.route.eq('local_h10'), KEY].to_numpy())),
            'COUNTERFACTUAL KEYS DO NOT MATCH LOCAL ROUTE')
    cache_memo = {}
    for j, row in enumerate(selected.itertuples(index=False), 1):
        aid, step = int(row.agent_id), int(row.step)
        lat, lng = float(row.h3_origin_lat), float(row.h3_origin_lng)
        layer_id = int(world.profile_schedule_layer_ids[int(world.agent_profile_type[aid]), step])
        require(layer_id > 0, 'INACTIVE SCHEDULE', row)
        layer = id_to_code(layer_id)
        dC, dH = int(row.continuous_destination_id), int(row.h3_destination_id)
        full_radius = score(layers[layer], lat, lng, world, cfg)[1]
        full_radius_set = set(map(int, full_radius))
        # Control applies only to valid Continuous destinations; -1 means no destination.
        if dC >= 0:
            require(dC in full_radius_set, 'CONTROL FAIL: CONTINUOUS DESTINATION NOT IN C_R', row,
                    dC=dC, layer=layer, n_CR=len(full_radius_set))
        if row.route == 'local_h10':
            cfr = local_cf.loc[(aid, step)]
            h12 = h3.latlng_to_cell(lat, lng, 12)
            center_h10 = h3.cell_to_parent(h12, 10)
            cells = list(h3.grid_disk(center_h10, 1))
            c24, cfull = recover(static.poi_index_h10, cells, layer, 24)
            require(c24 == tuple(get_h10_candidate_poi_ids(static, h12, layer, cfg)),
                    'H10 C24 BUCKET WALK REPRODUCTION FAIL', row)
            d24, ids24, _ = score(c24, lat, lng, world, cfg)
            dfull, idsfull, _ = score(cfull, lat, lng, world, cfg)
            require(d24 == dH == int(cfr.d24) and dfull == int(cfr.d_full),
                    'CONTROL FAIL: D24/DFULL REPRODUCTION', row, d24=d24, dH=dH, dfull=dfull)
            full_set, set24 = set(map(int, idsfull)), set(map(int, ids24))
            if dC < 0:
                category, geom_dist = 'continuous_none', None
            elif dC not in full_set:
                if poi_h10[dC] not in cells:
                    category = 'geometry_outside_h10_ring'
                    geom_dist = float(haversine_m(lat, lng, float(world.amenity_lat[dC]), float(world.amenity_lng[dC])))
                else:
                    require(False, 'CONTROL FAIL: INSIDE H10 RING BUT ABSENT FROM C_FULL', row, dC=dC, layer=layer)
            elif dC not in set24:
                category, geom_dist = 'truncation_excluded', None
            else:
                category, geom_dist = 'inside_C24', None
            rows_local.append(dict(city=city_name, agent_id=aid, step=step, layer=layer,
                category=category, dC=dC, d24=d24, d_full=dfull, recovered_no_cap=int(dfull == dC and d24 != dC),
                shared_disagreement=int(dC != dH), continuous_distance_m=geom_dist,
                n_CR=len(full_radius_set), n_Cfull=len(full_set), n_C24=len(set24)))
        else:
            h12 = h3.latlng_to_cell(lat, lng, 12)
            h9 = h3.cell_to_parent(h12, cfg.h3.accessibility_res)
            cache_key = (h9, layer)
            if cache_key not in cache_memo:
                center_lat, center_lng = h3.cell_to_latlng(h9)
                center_h8 = h3.latlng_to_cell(center_lat, center_lng, 8)
                bucket_cells = list(h3.grid_disk(center_h8, 4))
                bucket_ids_all = []
                for h8 in bucket_cells:
                    bucket_ids_all.extend(h8_index.get((h8, layer), []))
                dcenter, fixed, _ = score(bucket_ids_all, center_lat, center_lng, world, cfg)
                state = static.accessibility_h9.get((h9, layer))
                if state is None:
                    candidate = static.accessibility_h9.get(h9)
                    state = candidate.get(layer) if isinstance(candidate, dict) else candidate
                require(state is not None, 'CONTROL FAIL: MISSING H9 STATE', row, h9=h9, layer=layer)
                require(dcenter == int(state.best_poi_id), 'CONTROL FAIL: RECONSTRUCTED H9 BEST != PERSISTED',
                        row, dcenter=dcenter, persisted=int(state.best_poi_id), h9=h9, layer=layer)
                require(len(fixed) == int(state.candidate_count), 'CONTROL FAIL: H9 CANDIDATE COUNT',
                        row, reconstructed=len(fixed), persisted=int(state.candidate_count))
                cache_memo[cache_key] = (dcenter, tuple(map(int, fixed)), tuple(bucket_ids_all))
            dcenter, fixed, bucket_ids_all = cache_memo[cache_key]
            require(dcenter == dH, 'CONTROL FAIL: RECONSTRUCTED CACHE != PUBLISHED', row,
                    dcenter=dcenter, published=dH)
            # Same center-filtered candidate set, re-scored at agent without refiltering.
            d_fixed, _, _ = score(fixed, lat, lng, world, cfg, radius=False)
            # Same H8 buckets, now filtered to agent's R; isolates radius-filter change.
            d_refilter, agent_ids, _ = score(bucket_ids_all, lat, lng, world, cfg, radius=True)
            # Difference from full R candidate set isolates H8 bucket coverage.
            d_CR, _, _ = score(layers[layer], lat, lng, world, cfg)
            # Continuous's spatial-index order may break exact-score ties differently
            # from POI-ID order. Require score optimality, not identical tie identity.
            cr_ids = np.asarray(full_radius, dtype=np.int64)
            if dC < 0:
                require(cr_ids.size == 0 and d_CR == -1,
                        'CONTROL FAIL: CONTINUOUS NONE WITH ELIGIBLE CANDIDATES', row,
                        dC=dC, d_CR=d_CR, n_CR=len(cr_ids))
                cr_tie_count, cr_score_gap, cr_identity_tie = 0, None, False
            else:
                require(dC in full_radius_set,
                        'CONTROL FAIL: CONTINUOUS DESTINATION OUTSIDE C_R', row, dC=dC)
                cr_dist = np.asarray(haversine_m(lat, lng,
                    world.amenity_lat[cr_ids], world.amenity_lng[cr_ids]))
                cr_scores = np.asarray(destination_score(cr_dist,
                    world.amenity_attractiveness[cr_ids],
                    cfg.beta_distance, cfg.gamma_attractiveness))
                cr_best = float(np.max(cr_scores))
                cr_published = float(cr_scores[np.flatnonzero(cr_ids == dC)[0]])
                cr_score_gap = cr_best - cr_published
                require(np.isfinite(cr_score_gap) and abs(cr_score_gap) <= 1e-12,
                        'CONTROL FAIL: CONTINUOUS DESTINATION SUBOPTIMAL', row,
                        dC=dC, d_CR=d_CR, score_gap=cr_score_gap)
                cr_tie_count = int(np.count_nonzero(np.isclose(
                    cr_scores, cr_best, rtol=0, atol=1e-12)))
                cr_identity_tie = bool(dC != d_CR)
                require(not cr_identity_tie or cr_tie_count >= 2,
                        'CONTROL FAIL: DIFFERENT ID WITHOUT SCORE TIE', row,
                        dC=dC, d_CR=d_CR, tie_count=cr_tie_count)
            rows_cache.append(dict(city=city_name, agent_id=aid, step=step, layer=layer,
                cr_tie_count=cr_tie_count, cr_score_gap=cr_score_gap,
                cr_identity_tie=int(cr_identity_tie),
                dC=dC, d_cache=dcenter, d_agent_fixed=d_fixed, d_agent_refilter=d_refilter,
                d_CR=d_CR, shared_disagreement=int(dC != dH),
                cache_agrees=int(dcenter == dC), fixed_agrees=int(d_fixed == dC),
                refilter_agrees=int(d_refilter == dC),
                changed_fixed=int(d_fixed != dcenter), changed_refilter=int(d_refilter != d_fixed),
                n_center_candidates=len(fixed), n_agent_candidates=len(agent_ids),
                continuous_in_H8_buckets=int(dC in bucket_ids_all) if dC >= 0 else None,
                continuous_in_center_filtered=int(dC in fixed) if dC >= 0 else None))
        if j % 10000 == 0:
            print(city_name, 'CHECKED', j, '/', len(selected), flush=True)
    local = pd.DataFrame(rows_local)
    cached = pd.DataFrame(rows_cache)
    require(len(local) == int((x.shared & x.route.eq('local_h10')).sum()), 'LOCAL SHARED COVERAGE')
    require(len(cached) == int((x.shared & x.route.eq('cached_best_h9')).sum()), 'CACHE SHARED COVERAGE')
    if not local.empty:
        require(int(local.recovered_no_cap.sum()) == int(((local.d_full == local.dC) & (local.d24 != local.dC)).sum()),
                'RECOVERY INVARIANT')
    result = {
        'city': city_name, 'baseline_commit': '329add9e59fdd6f6660cc8d69b693f50175f65ea',
        'input_sha256': {str(p): sha(p) for p in (ep_path, trace_path, cf_path)},
        'total_decisions': len(x), 'shared_origins': int(x.shared.sum()),
        'local_h10_shared': len(local), 'cached_best_h9_shared': len(cached),
        'local_category_all_shared': local.category.value_counts().to_dict(),
        'local_category_disagreements': local.loc[local.shared_disagreement.eq(1), 'category'].value_counts().to_dict(),
        'geometry_continuous_distance_m': summary_dist(local.loc[local.category.eq('geometry_outside_h10_ring'), 'continuous_distance_m'].dropna().tolist()),
        'local_recovered_no_cap': int(local.recovered_no_cap.sum()) if len(local) else 0,
        'cache': {
            'published_agreement': int(cached.cache_agrees.sum()) if len(cached) else 0,
            'continuous_full_R_identity_ties': int(cached.cr_identity_tie.sum()) if len(cached) else 0,
            'continuous_full_R_tied_maxima_decisions': int((cached.cr_tie_count > 1).sum()) if len(cached) else 0,
            'fixed_agreement': int(cached.fixed_agrees.sum()) if len(cached) else 0,
            'refilter_agreement': int(cached.refilter_agrees.sum()) if len(cached) else 0,
            'changed_center_to_fixed': int(cached.changed_fixed.sum()) if len(cached) else 0,
            'changed_fixed_to_refilter': int(cached.changed_refilter.sum()) if len(cached) else 0,
            'continuous_outside_H8_buckets': int((cached.continuous_in_H8_buckets == 0).sum()) if len(cached) else 0,
            'continuous_in_H8_but_outside_center_radius': int(((cached.continuous_in_H8_buckets == 1) & (cached.continuous_in_center_filtered == 0)).sum()) if len(cached) else 0,
        },
    }
    return local, cached, result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--agents', type=int, default=10000)
    ap.add_argument('--base', type=Path, default=Path('outputs'))
    ap.add_argument('--output', type=Path, default=Path('outputs/note05_geometry_cache_attribution'))
    args = ap.parse_args()
    require(not args.output.exists(), 'REFUSING TO OVERWRITE OUTPUT', path=str(args.output))
    require(Path('experiments/10_note05_truncation_counterfactual.py').is_file(), 'NOT RUNNING FROM REPO ROOT')
    reports, locals_, caches = [], [], []
    for city in CITIES:
        print('\nCITY', city, flush=True)
        local, cache, report = process(city, args.agents, args.base)
        reports.append(report)
        locals_.append(local)
        caches.append(cache)
        print(json.dumps(report, indent=2), flush=True)
    # All cities pass before writing any output.
    args.output.mkdir(parents=True, exist_ok=False)
    pd.concat(locals_, ignore_index=True).to_csv(args.output / 'local_h10_geometry.csv', index=False)
    pd.concat(caches, ignore_index=True).to_csv(args.output / 'cached_h9_counterfactual.csv', index=False)
    (args.output / 'summary.json').write_text(json.dumps(reports, indent=2) + '\n')
    print('\nNOTE05 SCRIPT12: ALL THREE CITIES PASS')
    print('OUTPUT', args.output)


if __name__ == '__main__':
    main()
