#!/usr/bin/env python3
"""Note05: read-only cross-backend attribution and route diagnostics.

Run from repository root. Inputs are immutable. No simulation is executed.
The counterfactual only covers local_h10; other H3 routes remain as published.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from h3uab.city_config import config_for_city
from h3uab.semantic_layers import id_to_code
from h3uab.world import build_world

CITIES = {
    'madrid': ('madrid_traced', 'madrid', 41862, 4),
    'cambridge_ma': ('cambridge_traced', 'cambridge', 32718, 7),
    'new_york_ny': ('nyc_traced', 'nyc', 28390, 14),
}
KEY = ['agent_id', 'step']
ROUTES = ['local_h10', 'cached_best_h9', 'bounded_fallback_h9', 'none']

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def require(ok, msg):
    if not ok:
        raise AssertionError(msg)

def ids_equal(a, b):
    return a.notna() & b.notna() & a.eq(b)

def origin_equal(x):
    return (x.continuous_origin_lat.eq(x.h3_origin_lat) &
            x.continuous_origin_lng.eq(x.h3_origin_lng))

def status(original, alternative, reference):
    before = ids_equal(original, reference)
    after = ids_equal(alternative, reference)
    return np.select([~before & after, before & ~after],
                     ['recover', 'worsen'], default='neither')

def process(city, agents, base):
    traced_name, cf_name, expected_h10, expected_excess = CITIES[city]
    ep_path = base / 'note05_paired_trace' / traced_name / f'diagnostics_n{agents}' / 'paired_trip_endpoints.csv'
    trace_path = base / 'note05_paired_trace' / traced_name / 'note05_trace.csv'
    cf_path = base / 'note05_counterfactual' / cf_name / 'per_decision.csv'
    for p in (ep_path, trace_path, cf_path):
        require(p.is_file(), f'Missing input: {p}')
    ep, trace, cf = (pd.read_csv(p) for p in (ep_path, trace_path, cf_path))
    require(len(ep) == 5 * agents, f'{city}: unexpected paired row count')
    for label, data, key in [('endpoints', ep, KEY), ('trace', trace, KEY + ['backend']), ('counterfactual', cf, KEY)]:
        require(not data.duplicated(key).any(), f'{city}: duplicate {label} keys')
    for col in ['purpose_id','continuous_origin_lat','continuous_origin_lng',
                'h3_origin_lat','h3_origin_lng','continuous_destination_id','h3_destination_id']:
        require(col in ep, f'{city}: missing endpoint field {col}')
    for col in ['layer','d24','d_full','d_near24','A_id','B']:
        require(col in cf, f'{city}: missing counterfactual field {col}')
    require(set(trace.backend.dropna().unique()) == {'continuous','h3'}, f'{city}: trace backend mismatch')
    h3trace = trace.loc[trace.backend.eq('h3'), KEY + ['route']].copy()
    require(len(h3trace) == len(ep), f'{city}: incomplete H3 trace')
    x = ep.merge(h3trace, on=KEY, validate='one_to_one', indicator=True)
    require(x._merge.eq('both').all(), f'{city}: missing trace rows')
    x = x.drop(columns='_merge')
    require(x.route.isin(ROUTES).all(), f'{city}: unknown H3 route')
    require(x.step.between(1, 5).all(), f'{city}: unexpected active step')
    require(x[KEY].notna().all().all(), f'{city}: missing decision key')
    x['shared'] = origin_equal(x)
    require(x.loc[x.step.eq(1), 'shared'].all(), f'{city}: step 1 has unequal origins')
    require(x.continuous_destination_id.notna().all() and x.h3_destination_id.notna().all(),
            f'{city}: missing destination IDs; inspect no-destination encoding')
    require(len(cf) == expected_h10, f'{city}: local_h10 count mismatch')
    require(cf.route.eq('local_h10').all(), f'{city}: counterfactual contains nonlocal routes')
    require(cf[KEY].notna().all().all(), f'{city}: missing CF keys')
    require(cf.d24.notna().all() and cf.d_full.notna().all() and cf.d_near24.notna().all(),
            f'{city}: missing CF destinations')
    local = x.loc[x.route.eq('local_h10')].merge(cf, on=KEY, validate='one_to_one', indicator=True)
    require(len(local) == len(cf) and local._merge.eq('both').all(), f'{city}: CF alignment mismatch')
    require(local.d24.astype('int64').eq(local.h3_destination_id.astype('int64')).all(),
            f'{city}: published H3 != d24')
    require(local.A_id.astype(bool).eq(local.d24.ne(local.d_full)).all(), f'{city}: A flag mismatch')
    require(local.B.astype(bool).eq(local.d24.ne(local.d_near24)).all(), f'{city}: B flag mismatch')
    # Independent layer check: use the original profile schedule, not the CF reconstruction.
    cfg = config_for_city(city, n_agents=agents)
    world, _ = build_world(cfg)
    aid = local.agent_id.to_numpy(dtype='int64')
    steps = local.step.to_numpy(dtype='int64')
    scheduled = world.profile_schedule_layer_ids[world.agent_profile_type[aid], steps]
    require(np.all(scheduled > 0), f'{city}: inactive scheduled layer in local_h10')
    expected_layers = np.asarray([id_to_code(int(v)) for v in scheduled])
    require(np.array_equal(expected_layers, local.layer.astype(str).to_numpy()),
            f'{city}: counterfactual layer differs from paired agent schedule')
    # Validate purpose-to-layer mapping, only when identifiable from observed schedule.
    all_aid = x.agent_id.to_numpy(dtype='int64')
    all_step = x.step.to_numpy(dtype='int64')
    all_layers = world.profile_schedule_layer_ids[world.agent_profile_type[all_aid], all_step]
    require(np.all(all_layers > 0), f'{city}: inactive schedule in paired endpoints')
    x['scheduled_layer'] = [id_to_code(int(v)) for v in all_layers]
    purpose_map = x.groupby('purpose_id').scheduled_layer.nunique(dropna=False)
    require((purpose_map == 1).all(),
            f'{city}: purpose_id is not uniquely mapped to layer: {purpose_map.to_dict()}; explicit mapping required')
    inferred = x.groupby('purpose_id').scheduled_layer.first().to_dict()
    require(local.apply(lambda r: inferred[r.purpose_id] == r.layer, axis=1).all(),
            f'{city}: purpose_id / counterfactual layer mismatch')
    # Local counterfactual is a substitution on recorded decisions, not a rerun.
    sub = cf[KEY + ['d_full','d_near24']].copy()
    x = x.merge(sub, on=KEY, how='left', validate='one_to_one')
    require(x.d_full.notna().eq(x.route.eq('local_h10')).all(), f'{city}: full substitution coverage mismatch')
    for label, col in [('published','h3_destination_id'), ('full','d_full'), ('near24','d_near24')]:
        x['dest_' + label] = x[col].fillna(x.h3_destination_id).astype('int64')
        x['agree_' + label] = ids_equal(x['dest_' + label], x.continuous_destination_id)
    route_counts = x.groupby(['step','route'], dropna=False).size().rename('n').reset_index()
    global_routes = x.groupby('route').size().reindex(ROUTES, fill_value=0)
    step1_routes = x.loc[x.step.eq(1)].groupby('route').size().reindex(ROUTES, fill_value=0)
    route_table = pd.DataFrame({'route':ROUTES,'global_n':global_routes.values,
        'global_pct':100*global_routes.values/len(x),'step1_n':step1_routes.values,
        'step1_pct':100*step1_routes.values/agents})
    step1 = x.loc[x.step.eq(1)].copy()
    step1_route = step1.groupby('route', sort=False).agg(
        decisions=('agent_id','size'), agreement_n=('agree_published','sum'))
    step1_route['agreement_pct'] = 100 * step1_route.agreement_n / step1_route.decisions
    step1_route = step1_route.reindex(ROUTES).fillna(0).reset_index()
    shared = x.loc[x.shared].copy()
    require(len(shared) > 0, f'{city}: no shared origins')
    disagreements = int((~shared.agree_published).sum())
    require(disagreements > 0, f'{city}: zero original disagreements; undefined relative denominator')
    local_shared = shared.loc[shared.route.eq('local_h10')].copy()
    affected = local_shared.loc[local_shared.dest_published.ne(local_shared.dest_full)]
    report = {'city':city, 'input_sha256':{str(p):digest(p) for p in (ep_path,trace_path,cf_path)},
        'decisions':len(x), 'shared_origins':len(shared),
        'original_disagreements_shared':disagreements,
        'local_h10_shared':len(local_shared), 'A_on_shared':len(affected),
        'purpose_layer_mapping':{str(k):v for k,v in inferred.items()},
        'variants':{}, 'affected_A_full':{}}
    for variant in ('published','full','near24'):
        n = int(shared['agree_'+variant].sum())
        report['variants'][variant] = {'agreement_n':n,'agreement_pct':100*n/len(shared),
            'disagreement_n':len(shared)-n}
    for variant in ('full','near24'):
        changed = local_shared.loc[local_shared.dest_published.ne(local_shared['dest_'+variant])].copy()
        cats = pd.Series(status(changed.dest_published, changed['dest_'+variant],
                                changed.continuous_destination_id)).value_counts()
        recover, worsen, neither = (int(cats.get(k,0)) for k in ('recover','worsen','neither'))
        require(recover+worsen+neither == len(changed), f'{city}: classification incomplete')
        net = recover-worsen
        require(report['variants'][variant]['agreement_n'] - report['variants']['published']['agreement_n'] == net,
                f'{city}: net agreement mismatch {variant}')
        report['variants'][variant].update({'changed_on_shared':len(changed),
            'recover':recover,'worsen':worsen,'neither':neither,
            'net_gain_n':net,'delta_shared_pp':100*net/len(shared),
            'delta_disagreements_pct':100*net/disagreements})
    cats = pd.Series(status(affected.dest_published, affected.dest_full,
                            affected.continuous_destination_id)).value_counts()
    report['affected_A_full'] = {k:int(cats.get(k,0)) for k in ('recover','worsen','neither')}
    require(sum(report['affected_A_full'].values()) == len(affected), f'{city}: A partition mismatch')
    print('\n' + '='*76)
    print(city.upper(), 'ATTRIBUTION: PASS')
    print('ROUTES (global / step1)')
    print(route_table.to_string(index=False, float_format=lambda v:f'{v:.4f}'))
    print('STEP1 ROUTE AGREEMENT')
    print(step1_route.to_string(index=False, float_format=lambda v:f'{v:.4f}'))
    print('SHARED ORIGINS:',len(shared),'ORIGINAL DISAGREEMENTS:',disagreements)
    print('A ON SHARED:',len(affected),'FULL A CATEGORIES:',report['affected_A_full'])
    print('VARIANTS:',json.dumps(report['variants'], indent=2))
    return report, route_table.assign(city=city), step1_route.assign(city=city)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--agents',type=int,default=10000)
    ap.add_argument('--base',type=Path,default=Path('outputs'))
    ap.add_argument('--output',type=Path,default=Path('outputs/note05_continuous_attribution'))
    args = ap.parse_args()
    require(not args.output.exists(), f'Output already exists; refusing overwrite: {args.output}')
    summaries, routes, step1 = [], [], []
    for city in CITIES:
        s, r, t = process(city,args.agents,args.base)
        summaries.append(s); routes.append(r); step1.append(t)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2))
    pd.concat(routes,ignore_index=True).to_csv(args.output/'routes_global_and_step1.csv',index=False)
    pd.concat(step1,ignore_index=True).to_csv(args.output/'step1_route_agreement.csv',index=False)
    print('\nNOTE05 CONTINUOUS ATTRIBUTION: ALL THREE CITIES PASS')
    print('OUTPUT:',args.output)

if __name__=='__main__':
    main()
