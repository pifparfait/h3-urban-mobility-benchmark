#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(
        ROOT
        / "src"
    ),
)

from h3uab.city_config import config_for_city
from h3uab.world import build_world
from h3uab.continuous_indexed_backend import (
    ContinuousIndexedBackend,
)
from h3uab.h3_backend import (
    MultiLayerH3Backend,
)


def _check_path(label, path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{label} missing: {path}"
        )

    print(
        f"[OK] {label:<24}: "
        f"{path}"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--city",
        default="cambridge_ma",
        choices=[
            "madrid",
            "cambridge_ma",
        ],
    )

    ap.add_argument(
        "--agents",
        type=int,
        default=1_000,
    )

    args = ap.parse_args()

    cfg = config_for_city(
        args.city,
        n_agents=args.agents,
    )

    # Architecture smoke test: routing remains OFF.
    # This is intentional and matches the architecture benchmark contract.
    if cfg.routing.enabled:
        raise RuntimeError(
            "City smoke test expects routing.enabled=False."
        )

    print(
        "="
        * 100
    )
    print(
        "00 — MULTI-CITY CORE SMOKE TEST"
    )
    print(
        "="
        * 100
    )
    print(
        f"City              : "
        f"{cfg.city_name}"
    )
    print(
        f"Agents            : "
        f"{cfg.n_agents:,}"
    )
    print(
        "Fidelity scales   : "
        "H7/H8/H9"
    )
    print(
        "H11/H12           : "
        "localization/visualization only"
    )
    print(
        "Routing           : "
        "disabled for architecture smoke test"
    )
    print()

    _check_path(
        "OSM POIs",
        cfg.osm_cache_csv,
    )

    _check_path(
        "population H3",
        cfg.population_h3_csv,
    )

    if cfg.city_name == "cambridge_ma":
        _check_path(
            "Cambridge OSM graph",
            cfg.routing.graphml_path,
        )

    print()
    print(
        "[1/4] Building world..."
    )

    t0 = time.perf_counter()

    world, metadata = build_world(
        cfg
    )

    build_world_s = (
        time.perf_counter()
        - t0
    )

    print(
        f"      world build time : "
        f"{build_world_s:.3f}s"
    )
    print(
        f"      agents           : "
        f"{len(world.agent_lat):,}"
    )
    print(
        f"      normalized POIs  : "
        f"{len(world.amenity_lat):,}"
    )
    print(
        f"      active layers    : "
        f"{world.active_layer_codes}"
    )

    if len(
        world.agent_lat
    ) != cfg.n_agents:
        raise AssertionError(
            "Agent count mismatch."
        )

    if len(
        world.amenity_lat
    ) == 0:
        raise AssertionError(
            "No normalized POIs."
        )

    for values, label in (
        (
            world.agent_lat,
            "agent latitudes",
        ),
        (
            world.agent_lng,
            "agent longitudes",
        ),
        (
            world.amenity_lat,
            "POI latitudes",
        ),
        (
            world.amenity_lng,
            "POI longitudes",
        ),
    ):
        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise AssertionError(
                f"Non-finite {label}."
            )

    print()
    print(
        "[2/4] Preparing "
        "ContinuousIndexedBackend..."
    )

    continuous = (
        ContinuousIndexedBackend()
    )

    t0 = time.perf_counter()

    continuous.prepare(
        world,
        cfg,
    )

    continuous_prepare_s = (
        time.perf_counter()
        - t0
    )

    print(
        f"      prepare time     : "
        f"{continuous_prepare_s:.3f}s"
    )

    print()
    print(
        "[3/4] Preparing "
        "MultiLayerH3Backend..."
    )

    h3_backend = (
        MultiLayerH3Backend()
    )

    t0 = time.perf_counter()

    h3_backend.prepare(
        world,
        cfg,
    )

    h3_prepare_s = (
        time.perf_counter()
        - t0
    )

    print(
        f"      prepare time     : "
        f"{h3_prepare_s:.3f}s"
    )
    print(
        f"      static city cache: "
        f"{cfg.static_city_cache}"
    )

    print()
    print(
        "[4/4] Minimal paired "
        "decision sanity check..."
    )

    valid_steps = [
        s
        for s in range(
            world
            .profile_schedule_layer_ids
            .shape[1]
        )
        if np.any(
            world
            .profile_schedule_layer_ids[
                :,
                s,
            ]
            > 0
        )
    ]

    if not valid_steps:
        raise RuntimeError(
            "No active schedule step."
        )

    step = int(
        valid_steps[0]
    )

    tested = 0
    n_valid_c = 0
    n_valid_h = 0

    max_test = min(
        100,
        cfg.n_agents,
    )

    for aid in range(
        max_test
    ):
        profile_id = int(
            world
            .agent_profile_type[
                aid
            ]
        )

        layer_id = int(
            world
            .profile_schedule_layer_ids[
                profile_id,
                step,
            ]
        )

        if layer_id <= 0:
            continue

        # Avoid importing routing or changing physical infrastructure.
        # Decision test uses exactly the same starting coordinates.
        from h3uab.semantic_layers import (
            id_to_code,
        )

        layer = id_to_code(
            layer_id
        )

        lat = float(
            world.agent_lat[
                aid
            ]
        )
        lng = float(
            world.agent_lng[
                aid
            ]
        )

        dc = continuous.decide(
            aid,
            lat,
            lng,
            layer,
        )

        dh = h3_backend.decide(
            aid,
            lat,
            lng,
            layer,
        )

        tested += 1
        n_valid_c += int(
            dc.destination_id
            >= 0
        )
        n_valid_h += int(
            dh.destination_id
            >= 0
        )

    if tested == 0:
        raise RuntimeError(
            "No decisions were tested."
        )

    print(
        f"      decisions tested : "
        f"{tested:,}"
    )
    print(
        f"      continuous valid : "
        f"{n_valid_c:,}"
    )
    print(
        f"      H3 valid         : "
        f"{n_valid_h:,}"
    )

    if (
        n_valid_c == 0
        or n_valid_h == 0
    ):
        raise RuntimeError(
            "One backend produced no valid "
            "destinations in the smoke sample."
        )

    print()
    print(
        "-"
        * 100
    )
    print(
        "CITY CORE SMOKE TEST: PASS"
    )
    print(
        "-"
        * 100
    )
    print(
        f"City                 : "
        f"{cfg.city_name}"
    )
    print(
        f"Population cells     : "
        f"{metadata.get('population_cells_used')}"
    )
    print(
        f"Population weight    : "
        f"{metadata.get('population_total_weight')}"
    )
    print(
        f"Normalized POIs      : "
        f"{metadata.get('n_real_osm_pois')}"
    )
    print(
        f"Static H3 cache      : "
        f"{cfg.static_city_cache}"
    )
    print()
    print(
        "Routing was NOT benchmarked "
        "or discretized in this test."
    )


if __name__ == "__main__":
    main()
