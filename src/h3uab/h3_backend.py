from __future__ import annotations

import time

import numpy as np

from h3uab.behavior import destination_score
from h3uab.dynamic_state import DynamicAgentState
from h3uab.geometry import haversine_m
from h3uab.models import SpatialDecision
from h3uab.spatial_base import SpatialBackend
from h3uab.static_city import (
    get_h10_candidate_poi_ids,
    get_h9_fallback_candidate_poi_ids,
    load_or_build_static_city,
)


class MultiLayerH3Backend(
    SpatialBackend
):
    """
    V9.8 multiresolution H3 urban-knowledge backend.

    Primary state:
        H3@9 accessibility

    Bounded refinement:
        H3@10 local candidate POIs
        -> exact current-agent distances
        -> local best candidate

    Fallback:
        neighboring H9 best-POI states

    H11/H12 are observation/index layers derived from continuous coordinates.
    They do not define or approximate the physical OSM route.
    """

    name = "h3_multilayer"

    def __init__(
        self,
    ):
        super().__init__()

        self.refinement_decisions = 0
        self.refinement_candidate_total = 0

        self.fallback_decisions = 0
        self.fallback_successes = 0

    def prepare(
        self,
        world,
        cfg,
    ):
        t0 = (
            time.perf_counter()
        )

        self.world = world
        self.cfg = cfg

        cfg.h3.validate()

        t_static = (
            time.perf_counter()
        )

        (
            self.city,
            static_time,
            mode,
        ) = (
            load_or_build_static_city(
                world,
                cfg,
            )
        )

        self.static_city_mode = (
            mode
        )

        self.profile.static_city_load_time_s = (
            time.perf_counter()
            - t_static
        )

        self.dynamic = DynamicAgentState(
            world,
            cfg.h3,
        )

        self.prepare_routing(
            world,
            cfg,
        )

        elapsed = (
            time.perf_counter()
            - t0
        )

        self.profile.initialization_time_s = (
            elapsed
        )

        return elapsed

    def _evaluate_candidates(
        self,
        candidate_ids,
        agent_lat,
        agent_lng,
        diagnostics=None,
    ):
        if diagnostics is not None:
            diagnostics["within_radius"] = 0
            diagnostics["scored"] = 0
        if not candidate_ids:
            return None

        ids = np.asarray(
            candidate_ids,
            dtype=np.int64,
        )

        # Keep only valid POI ids.
        ids = ids[
            (
                ids >= 0
            )
            & (
                ids
                < len(
                    self.world.amenity_lat
                )
            )
        ]

        if ids.size == 0:
            return None

        distances = haversine_m(
            float(
                agent_lat
            ),
            float(
                agent_lng
            ),
            self.world.amenity_lat[
                ids
            ],
            self.world.amenity_lng[
                ids
            ],
        )

        within = (
            distances
            <= self.cfg.search_radius_m
        )

        ids = ids[
            within
        ]

        distances = distances[
            within
        ]

        if diagnostics is not None:
            diagnostics["within_radius"] = int(ids.size)
            diagnostics["scored"] = int(ids.size)
        if ids.size == 0:
            return None

        attractiveness = (
            self.world.amenity_attractiveness[
                ids
            ]
        )

        scores = destination_score(
            distances,
            attractiveness,
            self.cfg.beta_distance,
            self.cfg.gamma_attractiveness,
        )

        best = int(
            np.argmax(
                scores
            )
        )

        return (
            int(
                ids[
                    best
                ]
            ),
            float(
                distances[
                    best
                ]
            ),
            int(
                ids.size
            ),
        )

    def decide(
        self,
        agent_idx,
        agent_lat,
        agent_lng,
        layer_code,
    ):
        t_lookup = (
            time.perf_counter()
        )

        h12 = (
            self.dynamic.position(
                agent_idx
            )
        )

        macro = (
            self.city.get_macro(
                h12
            )
        )

        amenity_context = (
            self.city.get_amenity_aggregate(
                h12,
                layer_code,
                self.cfg.h3.amenity_functional_res,
            )
        )

        access = (
            self.city.get_accessibility(
                h12,
                layer_code,
            )
        )

        _ = macro
        _ = amenity_context

        self.profile.lookup_time_s += (
            time.perf_counter()
            - t_lookup
        )

        trace = getattr(self, "trace", None)
        if trace is not None:
            trace.clear()
            trace["route"] = "none"
            trace["scored"] = 0

        # ----------------------------------------------------------
        # V9.4 H10 local refinement
        # ----------------------------------------------------------
        if (
            self.cfg.destination_refinement.enabled
        ):
            candidates = (
                get_h10_candidate_poi_ids(
                    self.city,
                    h12,
                    layer_code,
                    self.cfg,
                    diagnostics=trace.setdefault("h10", {}) if trace is not None else None,
                )
            )

            if candidates:
                t_refine = (
                    time.perf_counter()
                )

                evaluated = (
                    self._evaluate_candidates(
                        candidates,
                        agent_lat,
                        agent_lng,
                        diagnostics=trace["h10"] if trace is not None else None,
                    )
                )

                self.profile.distance_time_s += (
                    time.perf_counter()
                    - t_refine
                )

                self.refinement_decisions += 1

                self.refinement_candidate_total += (
                    len(
                        candidates
                    )
                )

                if evaluated is not None:
                    (
                        destination_id,
                        exact_distance,
                        local_count,
                    ) = evaluated

                    weighted_accessibility = (
                        access.weighted_accessibility
                        if access is not None
                        else 0.0
                    )

                    candidate_count = (
                        access.candidate_count
                        if access is not None
                        else local_count
                    )

                    if trace is not None:
                        trace["route"] = "local_h10"
                        trace["scored"] = trace["h10"]["scored"]
                    return SpatialDecision(
                        destination_id=(
                            destination_id
                        ),

                        straight_distance_m=(
                            exact_distance
                        ),

                        candidate_count=(
                            candidate_count
                        ),

                        weighted_accessibility=(
                            weighted_accessibility
                        ),
                    )

        # ----------------------------------------------------------
        # Original H9 best-POI state if available
        # ----------------------------------------------------------
        if (
            access is not None
            and access.best_poi_id >= 0
        ):
            destination_id = int(
                access.best_poi_id
            )

            td = (
                time.perf_counter()
            )

            exact_straight_distance = float(
                haversine_m(
                    float(
                        agent_lat
                    ),
                    float(
                        agent_lng
                    ),
                    float(
                        self.world.amenity_lat[
                            destination_id
                        ]
                    ),
                    float(
                        self.world.amenity_lng[
                            destination_id
                        ]
                    ),
                )
            )

            self.profile.distance_time_s += (
                time.perf_counter()
                - td
            )

            if trace is not None:
                trace["route"] = "cached_best_h9"
            return SpatialDecision(
                destination_id=(
                    destination_id
                ),

                straight_distance_m=(
                    exact_straight_distance
                ),

                candidate_count=(
                    access.candidate_count
                ),

                weighted_accessibility=(
                    access.weighted_accessibility
                ),
            )

        # ----------------------------------------------------------
        # V9.4 bounded H9 fallback
        # ----------------------------------------------------------
        if (
            self.cfg.destination_refinement.enable_fallback
        ):
            self.fallback_decisions += 1

            fallback_ids = (
                get_h9_fallback_candidate_poi_ids(
                    self.city,
                    h12,
                    layer_code,
                    self.cfg,
                    diagnostics=trace.setdefault("fallback_h9", {}) if trace is not None else None,
                )
            )

            if fallback_ids:
                evaluated = (
                    self._evaluate_candidates(
                        fallback_ids,
                        agent_lat,
                        agent_lng,
                        diagnostics=trace["fallback_h9"] if trace is not None else None,
                    )
                )

                if evaluated is not None:
                    (
                        destination_id,
                        exact_distance,
                        local_count,
                    ) = evaluated

                    self.fallback_successes += 1
                    if trace is not None:
                        trace["route"] = "bounded_fallback_h9"
                        trace["scored"] = trace["fallback_h9"]["scored"]

                    return SpatialDecision(
                        destination_id=(
                            destination_id
                        ),

                        straight_distance_m=(
                            exact_distance
                        ),

                        candidate_count=(
                            local_count
                        ),

                        weighted_accessibility=(
                            0.0
                        ),
                    )

        return SpatialDecision(
            -1,
            0.0,
            0,
            0.0,
        )

    def move_agent(
        self,
        agent_idx,
        destination_id,
        old_lat,
        old_lng,
        new_lat,
        new_lng,
    ):
        if destination_id < 0:
            return

        t0 = (
            time.perf_counter()
        )

        self.dynamic.move_to_coords(
            agent_idx,
            new_lat,
            new_lng,
        )

        self.profile.mobility_flow_updates += 1
        self.profile.occupancy_updates += 1

        self.profile.move_update_time_s += (
            time.perf_counter()
            - t0
        )
