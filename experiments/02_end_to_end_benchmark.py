#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h3uab.config import Config
from h3uab.world import build_world
from h3uab.continuous_indexed_backend import ContinuousIndexedBackend
from h3uab.h3_backend import MultiLayerH3Backend
from h3uab.semantic_layers import id_to_code


def active_steps(world):
    return [
        s for s in range(world.profile_schedule_layer_ids.shape[1])
        if np.any(world.profile_schedule_layer_ids[:, s] > 0)
    ]


def run_backend(world, backend, cfg):
    backend.prepare(world, cfg)
    lat = world.agent_lat.copy()
    lng = world.agent_lng.copy()
    decisions = 0
    t0 = perf_counter()

    for step in active_steps(world):
        for p in np.unique(world.agent_profile_type):
            layer_id = int(world.profile_schedule_layer_ids[int(p), int(step)])
            if layer_id <= 0:
                continue
            layer = id_to_code(layer_id)
            ids = np.flatnonzero(world.agent_profile_type == p)
            for aid in ids:
                d = backend.decide(
                    int(aid), float(lat[aid]), float(lng[aid]), layer
                )
                decisions += 1
                if d.destination_id >= 0:
                    lat[aid] = world.amenity_lat[d.destination_id]
                    lng[aid] = world.amenity_lng[d.destination_id]
                    if backend.name == "h3_multilayer":
                        backend.dynamic.move_to_coords(
                            int(aid), float(lat[aid]), float(lng[aid])
                        )
    elapsed = perf_counter() - t0
    return elapsed, decisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--agents", nargs="+", type=int,
        default=[10_000, 100_000, 500_000, 1_000_000]
    )
    args = ap.parse_args()

    rows = []
    for n in args.agents:
        print(f"\nN={n:,}")
        cfg = replace(Config(), n_agents=n)
        world, _ = build_world(cfg)

        tc, dc = run_backend(world, ContinuousIndexedBackend(), cfg)
        th, dh = run_backend(world, MultiLayerH3Backend(), cfg)
        assert dc == dh

        row = {
            "agents": n,
            "decisions": dc,
            "continuous_s": tc,
            "h3_s": th,
            "speedup_h3_over_continuous": tc / th,
            "continuous_decisions_per_s": dc / tc,
            "h3_decisions_per_s": dh / th,
        }
        rows.append(row)
        print(row)

        out = ROOT / "outputs/end_to_end"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "end_to_end_summary.csv", index=False)


if __name__ == "__main__":
    main()
