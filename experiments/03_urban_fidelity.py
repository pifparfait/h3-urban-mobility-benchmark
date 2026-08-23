#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h3uab.config import Config
from h3uab.world import build_world
from h3uab.continuous_indexed_backend import ContinuousIndexedBackend
from h3uab.h3_backend import MultiLayerH3Backend
from h3uab.semantic_layers import id_to_code
from h3uab.fidelity import UrbanFidelity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=10_000)
    args = ap.parse_args()

    cfg = replace(
        Config(),
        n_agents=args.agents,
    )

    world, _ = build_world(cfg)

    continuous = ContinuousIndexedBackend()
    h3_backend = MultiLayerH3Backend()

    continuous.prepare(world, cfg)
    h3_backend.prepare(world, cfg)

    lat_c = world.agent_lat.copy()
    lng_c = world.agent_lng.copy()
    lat_h = world.agent_lat.copy()
    lng_h = world.agent_lng.copy()

    # Genuine urban-analysis scales only.
    # H11/H12 remain localization/visualization layers and never enter
    # the fidelity comparison.
    fidelity = UrbanFidelity(
        world,
        resolutions=(7, 8, 9),
    )

    steps = [
        s
        for s in range(
            world.profile_schedule_layer_ids.shape[1]
        )
        if np.any(
            world.profile_schedule_layer_ids[:, s] > 0
        )
    ]

    for step in steps:
        print(f"step {step}")

        for profile_id in np.unique(
            world.agent_profile_type
        ):
            layer_id = int(
                world.profile_schedule_layer_ids[
                    int(profile_id),
                    int(step),
                ]
            )

            if layer_id <= 0:
                continue

            layer = id_to_code(layer_id)

            agent_ids = np.flatnonzero(
                world.agent_profile_type
                == profile_id
            )

            for aid_raw in agent_ids:
                aid = int(aid_raw)

                old_c = (
                    float(lat_c[aid]),
                    float(lng_c[aid]),
                )

                old_h = (
                    float(lat_h[aid]),
                    float(lng_h[aid]),
                )

                decision_c = continuous.decide(
                    aid,
                    old_c[0],
                    old_c[1],
                    layer,
                )

                decision_h = h3_backend.decide(
                    aid,
                    old_h[0],
                    old_h[1],
                    layer,
                )

                if decision_c.destination_id >= 0:
                    lat_c[aid] = world.amenity_lat[
                        decision_c.destination_id
                    ]
                    lng_c[aid] = world.amenity_lng[
                        decision_c.destination_id
                    ]

                if decision_h.destination_id >= 0:
                    lat_h[aid] = world.amenity_lat[
                        decision_h.destination_id
                    ]
                    lng_h[aid] = world.amenity_lng[
                        decision_h.destination_id
                    ]

                    h3_backend.dynamic.move_to_coords(
                        aid,
                        float(lat_h[aid]),
                        float(lng_h[aid]),
                    )

                new_c = (
                    float(lat_c[aid]),
                    float(lng_c[aid]),
                )

                new_h = (
                    float(lat_h[aid]),
                    float(lng_h[aid]),
                )

                fidelity.update(
                    old_c,
                    old_h,
                    new_c,
                    new_h,
                    decision_c.destination_id,
                    decision_h.destination_id,
                    decision_c.weighted_accessibility,
                    decision_h.weighted_accessibility,
                    decision_c.straight_distance_m,
                    decision_h.straight_distance_m,
                    agent_id=aid,
                    step=step,
                )

    fidelity.finalize_occupancy(
        lat_c,
        lng_c,
        lat_h,
        lng_h,
    )

    summary = fidelity.summary()

    output_dir = (
        ROOT
        / "outputs"
        / "urban_fidelity"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / f"urban_fidelity_n{args.agents}.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    diagnostics_dir = (
        output_dir
        / f"diagnostics_n{args.agents}"
    )

    fidelity.export_diagnostics(
        diagnostics_dir
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(f"[saved] {summary_path}")
    print(f"[saved] {diagnostics_dir}")
    print(
        f"[saved] "
        f"{diagnostics_dir / 'paired_trip_endpoints.csv'}"
    )


if __name__ == "__main__":
    main()
