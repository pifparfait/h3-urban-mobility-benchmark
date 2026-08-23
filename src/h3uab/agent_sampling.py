from __future__ import annotations

from pathlib import Path
from typing import Dict

import h3
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


def load_population_h3(cfg):
    path = Path(cfg.population_h3_csv)

    if not path.exists():
        raise FileNotFoundError(
            f"Real H3 population file not found: {path}\n"
            "Create it first with: python prepare_worldpop.py"
        )

    df = pd.read_csv(path)

    required = {
        cfg.population_h3_id_col,
        cfg.population_col,
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}."
        )

    work = df[
        [cfg.population_h3_id_col, cfg.population_col]
    ].copy()
    work.columns = ["h3_id", "population"]

    work["h3_id"] = work["h3_id"].astype(str)
    work["population"] = pd.to_numeric(
        work["population"],
        errors="coerce",
    )
    work = work.dropna()
    work = work[work["population"] > 0].copy()

    if work.empty:
        raise ValueError("No positive WorldPop cells found.")

    resolutions = [
        h3.get_resolution(cell)
        for cell in work["h3_id"]
    ]

    metadata = {
        "agent_sampling_mode": "worldpop_h3_weighted",
        "population_source_path": str(path),
        "population_cells_used": int(len(work)),
        "population_total_weight": float(work["population"].sum()),
        "population_h3_resolution_min": int(min(resolutions)),
        "population_h3_resolution_max": int(max(resolutions)),
    }

    return work, metadata


def _cell_geometry(cell):
    boundary = h3.cell_to_boundary(cell)
    poly = Polygon(
        [(lng, lat) for lat, lng in boundary]
    )
    return poly, poly.bounds


def _sample_inside_cell(cell, rng, cache):
    item = cache.get(cell)
    if item is None:
        item = _cell_geometry(cell)
        cache[cell] = item

    poly, (minx, miny, maxx, maxy) = item

    for _ in range(64):
        lng = rng.uniform(minx, maxx)
        lat = rng.uniform(miny, maxy)
        if poly.covers(Point(lng, lat)):
            return float(lat), float(lng)

    lat, lng = h3.cell_to_latlng(cell)
    return float(lat), float(lng)


def sample_agents(cfg, rng):
    pop, metadata = load_population_h3(cfg)

    weights = pop["population"].to_numpy(
        dtype=float,
        copy=True,
    )
    total = weights.sum()
    if total <= 0:
        raise ValueError("Population weights sum to <= 0.")

    weights = weights / total

    selected_idx = rng.choice(
        len(pop),
        size=cfg.n_agents,
        replace=True,
        p=weights,
    )

    selected_cells = (
        pop.iloc[selected_idx]["h3_id"]
        .astype(str)
        .to_numpy()
    )

    lats = np.empty(cfg.n_agents, dtype=float)
    lngs = np.empty(cfg.n_agents, dtype=float)

    cache: Dict[str, tuple] = {}

    for i, cell in enumerate(selected_cells):
        lat, lng = _sample_inside_cell(
            cell,
            rng,
            cache,
        )
        lats[i] = lat
        lngs[i] = lng

    metadata["sampled_unique_population_cells"] = int(
        len(set(selected_cells.tolist()))
    )

    return lats, lngs, metadata
