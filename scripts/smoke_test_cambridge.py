#!/usr/bin/env python3
"""Fast validation of Cambridge Phase 1 assets before any benchmark run."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.cambridge_ma import CAMBRIDGE


def main():
    checks = [
        CAMBRIDGE.boundary_path,
        CAMBRIDGE.graph_path,
        CAMBRIDGE.poi_path,
        CAMBRIDGE.population_h3_path,
        CAMBRIDGE.worldpop_clipped_path,
    ]

    print("=" * 88)
    print("CAMBRIDGE PHASE 1 — SMOKE TEST")
    print("=" * 88)

    for p in checks:
        if not p.exists():
            raise FileNotFoundError(p)
        print(f"[OK] {p.relative_to(ROOT)}")

    pois = pd.read_csv(CAMBRIDGE.poi_path)
    pop = pd.read_csv(CAMBRIDGE.population_h3_path)
    G = nx.read_graphml(CAMBRIDGE.graph_path)

    required_poi = {"poi_id", "lat", "lng", "category", "subcategory"}
    required_pop = {"h3", "population", "lat", "lng"}

    missing_poi = required_poi - set(pois.columns)
    missing_pop = required_pop - set(pop.columns)

    if missing_poi:
        raise ValueError(f"POI schema missing: {sorted(missing_poi)}")
    if missing_pop:
        raise ValueError(f"Population schema missing: {sorted(missing_pop)}")

    if len(pois) == 0 or len(pop) == 0 or len(G.nodes) == 0:
        raise ValueError("One or more Cambridge assets are empty.")

    print()
    print(f"OSM graph nodes      : {len(G.nodes):,}")
    print(f"OSM graph edges      : {len(G.edges):,}")
    print(f"POIs                 : {len(pois):,}")
    print(f"H3 population cells  : {len(pop):,}")
    print(f"Population sum       : {pop['population'].sum():,.1f}")
    print(f"Kendall zoom         : {CAMBRIDGE.kendall_zoom_bounds}")
    print(f"Volpe deep zoom      : {CAMBRIDGE.volpe_deep_zoom_bounds}")
    print()
    print("SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
