#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h3uab.city_config import config_for_city
from h3uab.world import build_world
from h3uab.continuous_indexed_backend import ContinuousIndexedBackend
from h3uab.h3_backend import MultiLayerH3Backend
from h3uab.semantic_layers import id_to_code
from h3uab.fidelity import UrbanFidelity


def _output_dir_for_city(city_name: str) -> Path:
    """
    Keep Madrid backward-compatible with the already validated repository
    layout while storing all new cross-city results in city-specific folders.

    Madrid:
        outputs/urban_fidelity/

    Cambridge:
        outputs/cambridge_ma/urban_fidelity/

    This guarantees that Cambridge never overwrites Madrid results.
    """
    if city_name == "madrid":
        return ROOT / "outputs" / "urban_fidelity"

    return (
        ROOT
        / "outputs"
        / city_name
        / "urban_fidelity"
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Paired Continuous-indexed vs H3-indexed urban fidelity "
            "experiment for Madrid or Cambridge."
        )
    )

    ap.add_argument(
        "--city",
        type=str,
        default="madrid",
        choices=[
            "madrid",
            "cambridge_ma",
            "new_york_ny",
        ],
        help=(
            "City configuration. Default remains Madrid for "
            "backward compatibility."
        ),
    )

    ap.add_argument(
        "--agents",
        type=int,
        default=10_000,
        help="Number of synthetic agents.",
    )

    args = ap.parse_args()

    cfg = config_for_city(
        args.city,
        n_agents=args.agents,
    )

    print(
        "="
        * 100
    )
    print(
        "URBAN FIDELITY — MULTI-CITY VALIDATION"
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
        "Fidelity scales   : H7, H8, H9"
    )
    print(
        "H11/H12           : localization/visualization only"
    )
    print(
        "Routing           : not part of this fidelity experiment"
    )
    print()

    world, metadata = build_world(
        cfg
    )

    print(
        f"[WORLD] normalized POIs : "
        f"{metadata.get('n_real_osm_pois'):,}"
    )
    print(
        f"[WORLD] population cells: "
        f"{metadata.get('population_cells_used')}"
    )
    print(
        f"[WORLD] population weight: "
        f"{metadata.get('population_total_weight')}"
    )
    print(
        f"[WORLD] active layers   : "
        f"{tuple(world.active_layer_codes)}"
    )
    print()

    continuous = ContinuousIndexedBackend()
    h3_backend = MultiLayerH3Backend()

    continuous.prepare(
        world,
        cfg,
    )

    h3_backend.prepare(
        world,
        cfg,
    )

    # Paired initial state:
    # both architectures begin from exactly the same coordinates.
    lat_c = world.agent_lat.copy()
    lng_c = world.agent_lng.copy()

    lat_h = world.agent_lat.copy()
    lng_h = world.agent_lng.copy()

    # Genuine urban-analysis scales only.
    #
    # IMMUTABLE SCIENTIFIC CONTRACT:
    #   H7/H8/H9 = urban fidelity analysis
    #   H11/H12  = localization/visualization only
    #
    # H11/H12 never enter this comparison.
    fidelity = UrbanFidelity(
        world,
        resolutions=(
            7,
            8,
            9,
        ),
    )

    steps = [
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

    print(
        f"[RUN] active schedule steps: "
        f"{steps}"
    )
    print()

    for step in steps:
        print(
            f"step {step}"
        )

        for profile_id in np.unique(
            world.agent_profile_type
        ):
            layer_id = int(
                world
                .profile_schedule_layer_ids[
                    int(profile_id),
                    int(step),
                ]
            )

            if layer_id <= 0:
                continue

            layer = id_to_code(
                layer_id
            )

            agent_ids = np.flatnonzero(
                world.agent_profile_type
                == profile_id
            )

            for aid_raw in agent_ids:
                aid = int(
                    aid_raw
                )

                old_c = (
                    float(
                        lat_c[
                            aid
                        ]
                    ),
                    float(
                        lng_c[
                            aid
                        ]
                    ),
                )

                old_h = (
                    float(
                        lat_h[
                            aid
                        ]
                    ),
                    float(
                        lng_h[
                            aid
                        ]
                    ),
                )

                decision_c = continuous.decide(
                    aid,
                    old_c[
                        0
                    ],
                    old_c[
                        1
                    ],
                    layer,
                )

                decision_h = h3_backend.decide(
                    aid,
                    old_h[
                        0
                    ],
                    old_h[
                        1
                    ],
                    layer,
                )

                if (
                    decision_c.destination_id
                    >= 0
                ):
                    lat_c[
                        aid
                    ] = world.amenity_lat[
                        decision_c.destination_id
                    ]

                    lng_c[
                        aid
                    ] = world.amenity_lng[
                        decision_c.destination_id
                    ]

                if (
                    decision_h.destination_id
                    >= 0
                ):
                    lat_h[
                        aid
                    ] = world.amenity_lat[
                        decision_h.destination_id
                    ]

                    lng_h[
                        aid
                    ] = world.amenity_lng[
                        decision_h.destination_id
                    ]

                    # H3 dynamic-state maintenance is architecture-specific.
                    # Physical OSM routing is intentionally not involved here.
                    h3_backend.dynamic.move_to_coords(
                        aid,
                        float(
                            lat_h[
                                aid
                            ]
                        ),
                        float(
                            lng_h[
                                aid
                            ]
                        ),
                    )

                new_c = (
                    float(
                        lat_c[
                            aid
                        ]
                    ),
                    float(
                        lng_c[
                            aid
                        ]
                    ),
                )

                new_h = (
                    float(
                        lat_h[
                            aid
                        ]
                    ),
                    float(
                        lng_h[
                            aid
                        ]
                    ),
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

    output_dir = _output_dir_for_city(
        cfg.city_name
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

    diagnostics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fidelity.export_diagnostics(
        diagnostics_dir
    )

    print()
    print(
        "="
        * 100
    )
    print(
        "URBAN FIDELITY SUMMARY"
    )
    print(
        "="
        * 100
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"[saved] "
        f"{summary_path}"
    )
    print(
        f"[saved] "
        f"{diagnostics_dir}"
    )
    print(
        f"[saved] "
        f"{diagnostics_dir / 'paired_trip_endpoints.csv'}"
    )

    print()
    print(
        "Experiment completed without "
        "modifying Madrid outputs."
        if cfg.city_name != "madrid"
        else
        "Madrid backward-compatible output layout preserved."
    )


if __name__ == "__main__":
    main()
