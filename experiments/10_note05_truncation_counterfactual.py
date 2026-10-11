#!/usr/bin/env python3
"""Note 05: fail-closed, read-only H10 cap counterfactual on paired endpoints.

Run from repository root with src/ importable. No simulations or agent movement.
Reproduces the exact bounded bucket walk before evaluating counterfactuals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from h3uab.behavior import destination_score
from h3uab.city_config import config_for_city
from h3uab.geometry import haversine_m
from h3uab.semantic_layers import id_to_code
from h3uab.static_city import get_h10_candidate_poi_ids, load_or_build_static_city
from h3uab.world import build_world


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def bucket_ids(index, cell, layer):
    # Match static_city.get_h10_candidate_poi_ids, including historical index layouts.
    if (cell, layer) in index:
        value = index[(cell, layer)]
    elif cell in index:
        bucket = index[cell]
        value = bucket.get(layer) if isinstance(bucket, dict) else bucket
    else:
        value = None
    if value is None:
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    try:
        return [int(v) for v in value]
    except TypeError:
        return []


def recover(index, cells, layer, cap):
    """Exact original bucket-level early stop; full visits all buckets."""
    raw = []
    visited = 0
    for cell in cells:
        visited += 1
        raw.extend(bucket_ids(index, cell, layer))
        if len(raw) >= cap:
            break
    original = tuple(dict.fromkeys(raw))[:cap]
    full_raw = []
    for cell in cells:
        full_raw.extend(bucket_ids(index, cell, layer))
    full = tuple(dict.fromkeys(full_raw))
    return original, full, visited, len(raw), len(full_raw)


def scored(ids, lat, lng, world, cfg):
    """Replicate _evaluate_candidates: valid IDs, exact radius, np.argmax first tie."""
    arr = np.asarray(ids, dtype=np.int64)
    arr = arr[(arr >= 0) & (arr < len(world.amenity_lat))]
    if not len(arr):
        return -1, float('nan'), 0, np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)
    d = np.asarray(haversine_m(float(lat), float(lng),
                               world.amenity_lat[arr], world.amenity_lng[arr]))
    keep = d <= cfg.search_radius_m
    arr, d = arr[keep], d[keep]
    if not len(arr):
        return -1, float('nan'), 0, arr, d, np.empty(0)
    s = np.asarray(destination_score(d, world.amenity_attractiveness[arr],
                                     cfg.beta_distance, cfg.gamma_attractiveness))
    best = int(np.argmax(s))
    return int(arr[best]), float(s[best]), len(arr), arr, d, s


def first_failure(reason, row, **detail):
    raise AssertionError(f'{reason} agent_id={int(row.agent_id)} step={int(row.step)} '
                         + json.dumps(detail, default=str, ensure_ascii=False))


def summarize(x, output):
    def counts(v):
        return {'count': int(v.sum()), 'fraction_local_h10': float(v.mean())}
    eff = x[x.effective_exclusion]
    a = eff[eff.A_id]
    ct = pd.crosstab(eff.A_id, eff.G, dropna=False).reindex(
        index=[False, True], columns=[False, True], fill_value=0)
    ct.index.name = 'A_id'; ct.columns.name = 'G'
    ct.to_csv(output / 'A_by_G_effective.csv')
    print('\nA × G ON T_EFFECTIVE\n' + ct.to_string())
    print('\nGLOBAL RATES (ALL LOCAL_H10)')
    for name in ('early_stop', 'cap_reached', 'raw_gt_retained',
                 'effective_exclusion', 'A_id', 'A_loss', 'B', 'G', 'tie_only'):
        print(f'{name:24s} {int(x[name].sum()):7d} / {len(x):7d} = {x[name].mean():.6%}')
    print('\nCONDITIONAL RECOVERY')
    print('P(G=1 | A_id=1, T_effective) =',
          float(a.G.mean()) if len(a) else None, 'n=', len(a))
    loss = eff[eff.A_loss]
    print('P(G=1 | A_loss=1, T_effective) =',
          float(loss.G.mean()) if len(loss) else None, 'n=', len(loss))
    magnitude = {}
    for col in ('loss_equiv_m', 'separation_m'):
        values = x.loc[x.A_id, col]
        magnitude[col] = {key: (float(values.quantile(q)) if len(values) else None)
                          for key, q in [('p50', .5), ('p90', .9), ('p99', .99)]}
        magnitude[col]['max'] = float(values.max()) if len(values) else None
        print(col, magnitude[col])
    result = {
        'local_h10': len(x), 'T': int(x.T_flag.sum()),
        'T_effective': int(x.effective_exclusion.sum()),
        'rates': {k: counts(x[k]) for k in
                  ('early_stop', 'cap_reached', 'effective_exclusion',
                   'A_id', 'A_loss', 'B', 'G', 'tie_only')},
        'A_by_G_effective': {f'A={i},G={j}': int(ct.loc[i, j])
                             for i in (False, True) for j in (False, True)},
        'P_G_given_A_id_effective': float(a.G.mean()) if len(a) else None,
        'P_G_given_A_loss_effective': float(loss.G.mean()) if len(loss) else None,
        'magnitude_on_A_id': magnitude,
    }
    (output / 'summary.json').write_text(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--city', choices=('madrid', 'cambridge_ma', 'new_york_ny'),
                    default='madrid')
    ap.add_argument('--agents', type=int, default=10000)
    ap.add_argument('--traced', type=Path, required=True,
                    help='Independent Note05 traced run directory')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--score-tolerance', type=float, default=1e-10,
                    help='Absolute score tolerance for distinguishing ties; score units')
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Output directory already exists; refusing overwrite')
    if args.score_tolerance < 0:
        ap.error('score tolerance must be nonnegative')
    cap = 24
    cfg = config_for_city(args.city, n_agents=args.agents)
    if int(cfg.destination_refinement.max_candidates) != cap or int(cfg.destination_refinement.h10_k_ring) != 1:
        raise AssertionError('This analysis requires the validated H10 cap=24, k=1')
    endpoints_path = args.traced / f'diagnostics_n{args.agents}' / 'paired_trip_endpoints.csv'
    trace_path = args.traced / 'note05_trace.csv'
    for p in (endpoints_path, trace_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    endpoints = pd.read_csv(endpoints_path)
    trace = pd.read_csv(trace_path)
    if endpoints.duplicated(['agent_id', 'step']).any() or trace.duplicated(['agent_id','step','backend']).any():
        raise AssertionError('Duplicate decision keys')
    local = trace[(trace.backend == 'h3') & (trace.route == 'local_h10')].copy()
    if args.city == 'madrid' and len(local) != 41862:
        raise AssertionError(f'Madrid expected 41862 local_h10 decisions, got {len(local)}')
    joined = local.merge(endpoints, on=['agent_id', 'step'], validate='one_to_one',
                         suffixes=('_trace', '_published'))
    if len(joined) != len(local):
        raise AssertionError('Missing paired endpoints')
    world, meta = build_world(cfg)
    city, _, mode = load_or_build_static_city(world, cfg)
    print('CITY:', args.city, 'STATIC_CITY:', mode, 'POIS:', len(world.amenity_lat))
    print('INPUT_SHA256:', json.dumps({str(p): sha256(p) for p in (endpoints_path, trace_path)}))
    print('LOCAL_H10:', len(joined), 'CAP:', cap, 'RADIUS:', cfg.search_radius_m)
    rows = []
    for j, row in enumerate(joined.itertuples(index=False)):
        aid, step = int(row.agent_id), int(row.step)
        origin_lat, origin_lng = float(row.h3_origin_lat), float(row.h3_origin_lng)
        # Use observed origin; H12 cell is the dynamic state's coordinate index.
        h12 = h3.latlng_to_cell(origin_lat, origin_lng, 12)
        layer_id = int(world.profile_schedule_layer_ids[int(world.agent_profile_type[aid]), step])
        if layer_id <= 0:
            first_failure('Missing active schedule layer', row, layer_id=layer_id)
        layer = id_to_code(layer_id)
        center = h3.cell_to_parent(h12, cfg.h3.amenity_index_res)
        cells = list(h3.grid_disk(center, int(cfg.destination_refinement.h10_k_ring)))
        original, full, visited, raw, full_raw = recover(city.poi_index_h10, cells, layer, cap)
        # Independent reference: exact production retrieval, including original bucket stop.
        actual = get_h10_candidate_poi_ids(city, h12, layer, cfg)
        if original != tuple(actual):
            first_failure('C24 PREFIX / BUCKET WALK mismatch', row,
                          recovered=original, backend=actual, layer=layer)
        if not set(original).issubset(set(full)):
            first_failure('C24 not subset of C_full', row)
        if not (1 <= visited <= len(cells)):
            first_failure('Invalid bucket count', row, visited=visited)
        if 'h10_buckets' in joined and pd.notna(row.h10_buckets) and visited != int(row.h10_buckets):
            first_failure('Visited bucket count differs from recorded trace', row,
                          reconstructed=visited, traced=row.h10_buckets)
        if 'h10_raw' in joined and pd.notna(row.h10_raw) and raw != int(row.h10_raw):
            first_failure('Raw candidate count differs from trace', row,
                          reconstructed=raw, traced=row.h10_raw)
        if 'h10_retained' in joined and pd.notna(row.h10_retained) and len(original) != int(row.h10_retained):
            first_failure('Retained candidate count differs from trace', row,
                          reconstructed=len(original), traced=row.h10_retained)
        d24, s24, n24, ids24, dist24, _ = scored(original, origin_lat, origin_lng, world, cfg)
        observed_id = int(row.h3_destination_id)
        if d24 != observed_id or d24 != int(row.destination_id):
            first_failure('d24 reproduction failed', row, layer=layer,
                          reconstructed=d24, endpoint=observed_id, trace=row.destination_id)
        if 'h10_scored' in joined and pd.notna(row.h10_scored) and n24 != int(row.h10_scored):
            first_failure('Scored count mismatch', row, reconstructed=n24, trace=row.h10_scored)
        dfull, sfull, _, idsfull, distfull, _ = scored(full, origin_lat, origin_lng, world, cfg)
        if dfull < 0:
            first_failure('No eligible full candidate for local_h10', row)
        nearest = sorted(zip(distfull.tolist(), idsfull.tolist()), key=lambda pair: (pair[0], pair[1]))[:cap]
        dnear, snear, _, _, _, _ = scored([p for _, p in nearest], origin_lat, origin_lng, world, cfg)
        if dnear < 0:
            first_failure('No eligible near24 candidate', row)
        early = visited < len(cells)
        # T is the procedural exposure, while T_effective proves an actual exclusion.
        T = early or len(full) > cap
        effective = bool(set(idsfull.tolist()) - set(ids24.tolist()))
        A = d24 != dfull
        B = d24 != dnear
        G = dnear == dfull
        loss_score = sfull - s24
        if loss_score < -args.score_tolerance:
            first_failure('Negative score loss', row, loss_score=loss_score,
                          d24=d24, dfull=dfull)
        if not effective and A:
            first_failure('A=1 outside T_effective', row, d24=d24, dfull=dfull)
        if effective and not T:
            first_failure('Effective exclusion without T', row)
        loss_m = max(0.0, loss_score) / float(cfg.beta_distance)
        sep = float(haversine_m(float(world.amenity_lat[d24]), float(world.amenity_lng[d24]),
                                float(world.amenity_lat[dfull]), float(world.amenity_lng[dfull])))
        rows.append(dict(agent_id=aid, step=step, layer=layer, route='local_h10',
                         n_cells=len(cells), visited_cells=visited, raw_ids=raw,
                         full_raw_ids=full_raw, retained_ids=len(original),
                         full_unique_ids=len(full), scored_24=n24, scored_full=len(idsfull),
                         early_stop=early, cap_reached=n24 == cap,
                         raw_gt_retained=raw > len(original), T_flag=T,
                         effective_exclusion=effective, d24=d24, d_full=dfull,
                         d_near24=dnear, A_id=A, B=B, G=G,
                         loss_equiv_m=loss_m, separation_m=sep,
                         A_loss=loss_score > args.score_tolerance,
                         tie_only=A and abs(loss_score) <= args.score_tolerance))
        if (j+1) % 10000 == 0:
            print('VERIFIED:', j+1, '/', len(joined), flush=True)
    result = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.output / 'per_decision.csv', index=False)
    summarize(result, args.output)
    print('\nNOTE05 TRUNCATION COUNTERFACTUAL:', args.city.upper(), 'PASS')
    print('OUTPUT:', args.output)

if __name__ == '__main__':
    main()
