#!/usr/bin/env python3
"""Read-only audit of Note05 residual identity ties and fallback/none routes.
No simulation, no cache rebuilding, no changes to published artifacts.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from h3uab.city_config import config_for_city
from h3uab.world import build_world
from h3uab.geometry import haversine_m
from h3uab.behavior import destination_score

CITIES={'madrid':('madrid_traced','madrid'), 'cambridge_ma':('cambridge_traced','cambridge'), 'new_york_ny':('nyc_traced','nyc')}
KEY=['agent_id','step']
TOL=1e-12

def check(condition,message):
    if not bool(condition): raise AssertionError(message)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def score(pid,lat,lng,world,cfg):
    if pid<0: return None
    d=float(haversine_m(lat,lng,float(world.amenity_lat[pid]),float(world.amenity_lng[pid])))
    s=float(np.asarray(destination_score(np.asarray([d]),np.asarray([world.amenity_attractiveness[pid]]),cfg.beta_distance,cfg.gamma_attractiveness))[0])
    return s

def metadata(world,pid):
    out={}
    if pid<0: return out
    for field in ('amenity_osm_id','amenity_osm_ids','amenity_source_id','amenity_source_ids','amenity_osm_type','amenity_layer_code'):
        arr=getattr(world,field,None)
        if arr is not None:
            try:
                v=arr[pid]
                out[field]=v.item() if hasattr(v,'item') else str(v)
            except (TypeError,IndexError,KeyError): pass
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--agents',type=int,default=10000)
    ap.add_argument('--base',type=Path,default=Path('outputs'))
    ap.add_argument('--script12',type=Path,default=Path('outputs/note05_geometry_cache_attribution'))
    ap.add_argument('--output',type=Path,default=Path('outputs/note05_final_tie_route_audit'))
    a=ap.parse_args()
    check(not a.output.exists(),f'REFUSE OVERWRITE: {a.output}')
    check(a.script12.is_dir(),'Script12 output directory missing')
    local=pd.read_csv(a.script12/'local_h10_geometry.csv')
    cache=pd.read_csv(a.script12/'cached_h9_counterfactual.csv')
    results=[]; ties=[]; route_rows=[]; inputs={}
    for city,(traced,_) in CITIES.items():
        print('AUDIT',city,flush=True)
        ep_path=a.base/'note05_paired_trace'/traced/f'diagnostics_n{a.agents}'/'paired_trip_endpoints.csv'
        trace_path=a.base/'note05_paired_trace'/traced/'note05_trace.csv'
        inputs[str(ep_path)]=sha(ep_path);inputs[str(trace_path)]=sha(trace_path)
        ep=pd.read_csv(ep_path)
        t=pd.read_csv(trace_path)
        t=t.loc[t.backend.eq('h3'),KEY+['route','destination_id']]
        x=ep.merge(t,on=KEY,validate='one_to_one')
        check(len(x)==a.agents*5,f'{city} endpoint coverage')
        check((x.h3_destination_id.astype(int)==x.destination_id.astype(int)).all(),f'{city} trace mismatch')
        x['shared']=x.continuous_origin_lat.eq(x.h3_origin_lat)&x.continuous_origin_lng.eq(x.h3_origin_lng)
        x=x.loc[x.shared].copy()
        cfg=config_for_city(city,n_agents=a.agents)
        world,_=build_world(cfg)
        lc=local.loc[local.city.eq(city)].copy()
        cc=cache.loc[cache.city.eq(city)].copy()
        check(len(lc)==int(x.route.eq('local_h10').sum()),f'{city} local coverage')
        check(len(cc)==int(x.route.eq('cached_best_h9').sum()),f'{city} cache coverage')
        check(not lc.duplicated(KEY).any() and not cc.duplicated(KEY).any(),f'{city} duplicate keys')
        all_dis=int((x.continuous_destination_id!=x.h3_destination_id).sum())
        route_count={}
        for route,g in x.groupby('route'):
            mism=g.loc[g.continuous_destination_id.ne(g.h3_destination_id)]
            route_count[route]={'n':len(g),'disagreements':len(mism),'agreements':len(g)-len(mism)}
            for r in mism.itertuples(index=False):
                route_rows.append(dict(city=city,route=route,agent_id=int(r.agent_id),step=int(r.step),dC=int(r.continuous_destination_id),dH=int(r.h3_destination_id)))
        ldis=lc.loc[lc.shared_disagreement.eq(1)]
        local_counts=ldis.category.value_counts().to_dict()
        check(int(ldis.shape[0])==route_count.get('local_h10',{}).get('disagreements',-1),f'{city} local mismatch')
        check(int(cc.shared_disagreement.sum())==route_count.get('cached_best_h9',{}).get('disagreements',-1),f'{city} cache mismatch')
        tie_candidates=[]
        for r in lc.loc[lc.category.eq('inside_C24') & lc.shared_disagreement.eq(1)].itertuples(index=False):
            tie_candidates.append(('local_h10',r,int(r.dC),int(r.d24)))
        for r in cc.loc[cc.refilter_agrees.eq(0)].itertuples(index=False):
            tie_candidates.append(('cached_best_h9_refilter',r,int(r.dC),int(r.d_agent_refilter)))
        n_exact=0;n_score_equiv=0;n_same_coords_attr=0;n_osm_same=0;n_osm_known=0
        for route,r,p,q in tie_candidates:
            check(p>=0 and q>=0,f'{city} tie audit negative destination {route} {r.agent_id}/{r.step}')
            er=x.loc[(x.agent_id==r.agent_id)&(x.step==r.step)]
            check(len(er)==1,f'{city} missing endpoint')
            e=er.iloc[0]
            lat,lng=float(e.h3_origin_lat),float(e.h3_origin_lng)
            sp,sq=score(p,lat,lng,world,cfg),score(q,lat,lng,world,cfg)
            gap=abs(sp-sq)
            equal_exact=(sp==sq)
            equivalent=gap<=TOL
            same_coord=bool(float(world.amenity_lat[p])==float(world.amenity_lat[q]) and float(world.amenity_lng[p])==float(world.amenity_lng[q]))
            same_attr=bool(float(world.amenity_attractiveness[p])==float(world.amenity_attractiveness[q]))
            mp,mq=metadata(world,p),metadata(world,q)
            osm_field=next((f for f in ('amenity_osm_id','amenity_osm_ids','amenity_source_id','amenity_source_ids') if f in mp and f in mq),None)
            osm_same=(str(mp[osm_field])==str(mq[osm_field])) if osm_field else None
            n_exact+=int(equal_exact);n_score_equiv+=int(equivalent);n_same_coords_attr+=int(same_coord and same_attr)
            n_osm_known+=int(osm_same is not None);n_osm_same+=int(osm_same is True)
            ties.append(dict(city=city,route=route,agent_id=int(r.agent_id),step=int(r.step),poi_continuous=p,poi_comparator=q,
                score_continuous=sp,score_comparator=sq,abs_score_gap=gap,exact_score_tie=equal_exact,score_equivalent_1e12=equivalent,
                same_coordinates=same_coord,same_attractiveness=same_attr,
                lat_continuous=float(world.amenity_lat[p]),lng_continuous=float(world.amenity_lng[p]),
                lat_comparator=float(world.amenity_lat[q]),lng_comparator=float(world.amenity_lng[q]),
                attractiveness_continuous=float(world.amenity_attractiveness[p]),attractiveness_comparator=float(world.amenity_attractiveness[q]),
                layer_continuous=str(world.amenity_layer_code[p]),layer_comparator=str(world.amenity_layer_code[q]),
                osm_field=osm_field,osm_id_continuous=mp.get(osm_field) if osm_field else None,
                osm_id_comparator=mq.get(osm_field) if osm_field else None,osm_id_equal=osm_same))
        none_step1=x.loc[x.route.eq('none')&x.step.eq(1)]
        none_both=int(((none_step1.continuous_destination_id==-1)&(none_step1.h3_destination_id==-1)).sum())
        other_dis=int(sum(v['disagreements'] for k,v in route_count.items() if k not in ('local_h10','cached_best_h9')))
        check(all_dis==int(ldis.shape[0])+int(cc.shared_disagreement.sum())+other_dis,f'{city} disagreement partition')
        report=dict(city=city,shared_origins=len(x),disagreements=all_dis,route_counts=route_count,
            local_disagreement_categories=local_counts,cache_disagreements=int(cc.shared_disagreement.sum()),other_route_disagreements=other_dis,
            tie_candidates=len(tie_candidates),exact_score_ties=n_exact,score_equivalent_1e12=n_score_equiv,
            same_coordinates_and_attractiveness=n_same_coords_attr,source_ids_available=n_osm_known,same_source_ids=n_osm_same,
            none_step1_total=len(none_step1),none_step1_both_minus_one=none_both,
            none_step1_other=len(none_step1)-none_both)
        results.append(report)
        print(json.dumps(report,indent=2),flush=True)
    check(sum(r['disagreements'] for r in results)==21042,'Expected total disagreement 21042 drift')
    a.output.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(ties).to_csv(a.output/'identity_tie_pairs.csv',index=False)
    pd.DataFrame(route_rows).to_csv(a.output/'all_route_disagreements.csv',index=False)
    (a.output/'summary.json').write_text(json.dumps({'baseline_commit':'329add9e59fdd6f6660cc8d69b693f50175f65ea',
        'input_sha256':inputs,'cities':results},indent=2)+'\n')
    print('AUDIT COMPLETE',a.output)

if __name__=='__main__':main()
