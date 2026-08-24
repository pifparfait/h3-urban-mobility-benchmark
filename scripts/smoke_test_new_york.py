#!/usr/bin/env python3
"""Fast validation of New York City Phase 1 assets before benchmark runs."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

from configs.new_york_ny import NEW_YORK


def main():
    checks = [
        NEW_YORK.boundary_path,
        NEW_YORK.graph_path,
        NEW_YORK.poi_path,
        NEW_YORK.population_h3_path,
        NEW_YORK.worldpop_clipped_path,
    ]

    print("=" * 88)
    print("NEW YORK CITY PHASE 1 — SMOKE TEST")
    print("=" * 88)

    for p in checks:
        if not p.exists():
            raise FileNotFoundError(p)

        print(
            f"[OK] {p.relative_to(ROOT)}"
        )

    pois = pd.read_csv(
        NEW_YORK.poi_path
    )

    pop = pd.read_csv(
        NEW_YORK.population_h3_path
    )

    G = nx.read_graphml(
        NEW_YORK.graph_path
    )

    required_poi = {
        "poi_id",
        "lat",
        "lng",
        "category",
        "subcategory",
    }

    required_pop = {
        "h3",
        "population",
        "lat",
        "lng",
    }

    missing_poi = (
        required_poi
        - set(pois.columns)
    )

    missing_pop = (
        required_pop
        - set(pop.columns)
    )

    if missing_poi:
        raise ValueError(
            f"POI schema missing: "
            f"{sorted(missing_poi)}"
        )

    if missing_pop:
        raise ValueError(
            f"Population schema missing: "
            f"{sorted(missing_pop)}"
        )

    if (
        len(pois) == 0
        or len(pop) == 0
        or len(G.nodes) == 0
    ):
        raise ValueError(
            "One or more New York City assets are empty."
        )

    print()
    print(
        f"OSM graph nodes      : "
        f"{len(G.nodes):,}"
    )
    print(
        f"OSM graph edges      : "
        f"{len(G.edges):,}"
    )
    print(
        f"POIs                 : "
        f"{len(pois):,}"
    )
    print(
        f"H3 population cells  : "
        f"{len(pop):,}"
    )
    print(
        f"Population sum       : "
        f"{pop['population'].sum():,.1f}"
    )
    print(
        f"Manhattan zoom       : "
        f"{NEW_YORK.manhattan_zoom_bounds}"
    )
    print(
        f"Midtown deep zoom    : "
        f"{NEW_YORK.midtown_deep_zoom_bounds}"
    )
    print()
    print(
        "SMOKE TEST: PASS"
    )


if __name__ == "__main__":
    main()
