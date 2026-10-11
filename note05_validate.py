#!/usr/bin/env python3
"""Read-only, fail-closed Note05 Madrid reproduction validator."""
import argparse, json, hashlib
from pathlib import Path
import pandas as pd

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def endpoints(root,agents):
    p=Path(root)/f'diagnostics_n{agents}'/'paired_trip_endpoints.csv'
    if not p.is_file(): raise FileNotFoundError(p)
    return p,pd.read_csv(p)

def compare(a,b,an,bn):
    if len(a)!=len(b): raise AssertionError(f'{an}/{bn}: row count {len(a)} != {len(b)}')
    key=['agent_id','step']
    for col in key:
        if col not in a or col not in b: raise AssertionError(f'missing key {col}')
    if a.duplicated(key).any() or b.duplicated(key).any(): raise AssertionError('duplicate decision key')
    aa=a.sort_values(key,kind='stable').reset_index(drop=True)
    bb=b.sort_values(key,kind='stable').reset_index(drop=True)
    for col in key:
        neq=~(aa[col].eq(bb[col]) | (aa[col].isna() & bb[col].isna()))
        if neq.any():
            i=int(neq.idxmax());raise AssertionError(f'{an}/{bn}: first key divergence row={i}: {aa.loc[i,key].to_dict()} vs {bb.loc[i,key].to_dict()}')
    cols=[c for c in aa.columns if c in bb.columns]
    # All common endpoint fields: exact numeric values, including NaNs, and all identifiers.
    for i in range(len(aa)):
        left=aa.iloc[i]; right=bb.iloc[i]
        for col in cols:
            x,y=left[col],right[col]
            if (pd.isna(x) and pd.isna(y)) or x==y: continue
            backend=('continuous' if 'continuous' in col.lower() or col.lower().endswith('_c') else
                     'h3' if 'h3' in col.lower() or col.lower().endswith('_h') else 'unknown')
            raise AssertionError(f'{an}/{bn}: first divergence agent_id={left.agent_id} step={left.step} backend={backend} field={col} {x!r} != {y!r}')
    print(f'PASS {an} == {bn}: {len(aa)} decisions, {len(cols)} shared columns')
    return len(aa)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--published',required=True)
    ap.add_argument('--plain',required=True)
    ap.add_argument('--traced',required=True)
    ap.add_argument('--agents',type=int,default=10000)
    args=ap.parse_args()
    p,a=endpoints(args.published,args.agents)
    q,b=endpoints(args.plain,args.agents)
    r,c=endpoints(args.traced,args.agents)
    print('INPUTS',json.dumps({str(x):sha(x) for x in (p,q,r)},indent=2))
    compare(a,b,'published','plain')
    compare(b,c,'plain','traced')
    compare(a,c,'published','traced')
    trace=Path(args.traced)/'note05_trace.csv'
    if not trace.is_file():raise FileNotFoundError(trace)
    t=pd.read_csv(trace)
    if len(t)!=2*len(c):raise AssertionError(f'trace row count {len(t)} != {2*len(c)}')
    if t.duplicated(['agent_id','step','backend']).any():raise AssertionError('duplicate trace keys')
    for backend in ('continuous','h3'):
        s=t[t.backend==backend]
        if len(s)!=len(c):raise AssertionError(f'{backend} trace rows {len(s)} != {len(c)}')
        print(f'\n{backend.upper()} ROUTES')
        print(s.groupby('route',dropna=False).agg(decisions=('route','size'),scored_p50=('scored','median'),scored_p90=('scored',lambda x:x.quantile(.9)),scored_p99=('scored',lambda x:x.quantile(.99))).to_string())
        s=s.copy();s['beyond_radius']=s.straight_distance_m.gt(2000)
        print('BEYOND 2000m BY ROUTE')
        print(s.groupby('route',dropna=False)['beyond_radius'].sum().to_string())
        print('TOTAL',int(s.beyond_radius.sum()))
        if backend == 'h3':
            expected_by_city = {
                'madrid': 4,
                'cambridge_ma': 7,
                'new_york_ny': 14,
            }
            published_path = str(args.published)
            city = next(
                (name for name in ('cambridge_ma', 'new_york_ny')
                 if name in published_path),
                'madrid'
            )
            expected = expected_by_city[city]
            actual = int(s.beyond_radius.sum())
            if actual != expected:
                raise AssertionError(
                    f'{city} H3 >R: expected {expected}, observed {actual}'
                )
        if backend=='continuous' and int(s.beyond_radius.sum())!=0:raise AssertionError('Madrid Continuous >R != 0 expected')
    print('\nMADRID NOTE05 VALIDATION: PASS')
if __name__=='__main__':main()
