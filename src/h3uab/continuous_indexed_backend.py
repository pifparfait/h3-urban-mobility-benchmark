from __future__ import annotations

import time

import numpy as np
from sklearn.neighbors import BallTree

from h3uab.behavior import destination_score
from h3uab.geometry import haversine_m
from h3uab.models import SpatialDecision
from h3uab.spatial_base import SpatialBackend


EARTH_RADIUS_M = 6_371_008.8


class ContinuousIndexedBackend(SpatialBackend):
    """
    Continuous spatial reference with a standard metric spatial index.

    Coordinates remain continuous.
    POIs remain real.
    Routing remains the same exact OSM router used by H3.

    Query complexity:
        BallTree neighborhood search ~ O(log P_D + k)
        exact scoring over returned candidates O(k)

    This is the primary scientific Continuous baseline in V9.8 because it
    avoids comparing H3 only against an intentionally naive full POI scan.
    """

    name = "continuous_indexed"

    def prepare(self, world, cfg):
        t0 = time.perf_counter()

        self.world = world
        self.cfg = cfg

        self.by_layer = {}
        self.trees = {}

        for layer_code in world.active_layer_codes:
            ids = np.flatnonzero(
                world.amenity_layer_code
                == layer_code
            ).astype(
                np.int64,
                copy=False,
            )

            self.by_layer[
                layer_code
            ] = ids

            if ids.size == 0:
                self.trees[
                    layer_code
                ] = None
                continue

            # BallTree haversine expects [lat, lon] in radians.
            coords_rad = np.radians(
                np.column_stack(
                    (
                        world.amenity_lat[
                            ids
                        ],
                        world.amenity_lng[
                            ids
                        ],
                    )
                )
            )

            self.trees[
                layer_code
            ] = BallTree(
                coords_rad,
                metric="haversine",
                leaf_size=int(
                    cfg.baseline.indexed_leaf_size
                ),
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

    def decide(
        self,
        agent_idx,
        agent_lat,
        agent_lng,
        layer_code,
    ):
        trace = getattr(self, "trace", None)
        if trace is not None:
            trace.clear()
            trace.update(route="none", balltree_returned=0, within_radius=0, scored=0)
        ids = self.by_layer.get(
            layer_code
        )

        tree = self.trees.get(
            layer_code
        )

        if (
            ids is None
            or ids.size == 0
            or tree is None
        ):
            return SpatialDecision(
                -1,
                0.0,
                0,
                0.0,
            )

        tq = time.perf_counter()

        query = np.radians(
            [
                [
                    float(
                        agent_lat
                    ),
                    float(
                        agent_lng
                    ),
                ]
            ]
        )

        local_idx = tree.query_radius(
            query,
            r=(
                float(
                    self.cfg.search_radius_m
                )
                / EARTH_RADIUS_M
            ),
            return_distance=False,
        )[
            0
        ]

        self.profile.lookup_time_s += (
            time.perf_counter()
            - tq
        )

        if trace is not None:
            trace["balltree_returned"] = int(local_idx.size)
        if local_idx.size == 0:
            return SpatialDecision(
                -1,
                0.0,
                0,
                0.0,
            )

        candidates = ids[
            local_idx
        ]

        td = time.perf_counter()

        distances = haversine_m(
            float(
                agent_lat
            ),
            float(
                agent_lng
            ),
            self.world.amenity_lat[
                candidates
            ],
            self.world.amenity_lng[
                candidates
            ],
        )

        # Numerical safety: exact same radius rule as naive Continuous.
        keep = (
            distances
            <= float(
                self.cfg.search_radius_m
            )
        )

        candidates = candidates[
            keep
        ]

        distances = distances[
            keep
        ]

        if trace is not None:
            trace["within_radius"] = int(candidates.size)
        self.profile.distance_time_s += (
            time.perf_counter()
            - td
        )

        if candidates.size == 0:
            return SpatialDecision(
                -1,
                0.0,
                0,
                0.0,
            )

        attractiveness = (
            self.world.amenity_attractiveness[
                candidates
            ]
        )

        accessibility = float(
            np.sum(
                attractiveness
                * np.exp(
                    -distances
                    / self.cfg.accessibility_decay_m
                )
            )
        )

        t_dec = time.perf_counter()

        if trace is not None:
            trace["scored"] = int(candidates.size)
            trace["route"] = "radius_search"
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

        self.profile.decision_time_s += (
            time.perf_counter()
            - t_dec
        )

        return SpatialDecision(
            destination_id=int(
                candidates[
                    best
                ]
            ),
            straight_distance_m=float(
                distances[
                    best
                ]
            ),
            candidate_count=int(
                candidates.size
            ),
            weighted_accessibility=(
                accessibility
            ),
        )

    def move_agent(
        self,
        *args,
        **kwargs,
    ):
        # Continuous physical state is maintained by engine.py.
        return
