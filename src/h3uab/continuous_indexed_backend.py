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
        self.candidate_strategy = cfg.baseline.candidate_strategy
        self.candidate_cap = cfg.baseline.candidate_cap
        if self.candidate_strategy not in {"radius_exact", "radius_cap", "knn"}:
            raise ValueError(f"Unknown candidate strategy: {self.candidate_strategy}")
        if self.candidate_strategy != "radius_exact" and (self.candidate_cap is None or self.candidate_cap < 1):
            raise ValueError("Bounded continuous strategies require candidate_cap >= 1")

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

        if self.candidate_strategy == "knn":
            local_idx = tree.query(
                query, k=min(int(self.candidate_cap), len(ids)),
                return_distance=False,
            )[0]
        else:
            local_idx = tree.query_radius(
                query,
                r=float(self.cfg.search_radius_m) / EARTH_RADIUS_M,
                return_distance=False,
            )[0]
        returned_count = int(local_idx.size)

        self.profile.lookup_time_s += (
            time.perf_counter()
            - tq
        )

        if local_idx.size == 0:
            trace = getattr(self, "trace", None)
            if trace is not None:
                trace.update(returned=0, eligible=0, retained=0, scored=0,
                             requested=(int(self.candidate_cap) if self.candidate_strategy == "knn" else 0))
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
        eligible_count = int(candidates.size)

        # The radius-cap ablation changes destination scoring only. Keep the
        # exhaustive baseline's accessibility aggregate over every eligible
        # POI, even when fewer destinations are subsequently scored.
        full_accessibility = None
        if self.candidate_strategy == "radius_cap" and candidates.size > self.candidate_cap:
            full_accessibility = float(np.sum(
                self.world.amenity_attractiveness[candidates]
                * np.exp(-distances / self.cfg.accessibility_decay_m)
            ))

        # Radius-cap retains the original exhaustive query and exact-radius
        # filtering. Stable distance/POI-ID order makes ties reproducible.
        if self.candidate_strategy == "radius_cap" and candidates.size > self.candidate_cap:
            selected = np.lexsort((candidates, distances))[:self.candidate_cap]
            candidates = candidates[selected]
            distances = distances[selected]

        trace = getattr(self, "trace", None)
        if trace is not None:
            trace.update(returned=returned_count, eligible=eligible_count,
                         retained=int(candidates.size), scored=int(candidates.size),
                         requested=(int(self.candidate_cap) if self.candidate_strategy == "knn" else 0))

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

        accessibility = (
            full_accessibility
            if full_accessibility is not None
            else float(np.sum(
                attractiveness
                * np.exp(-distances / self.cfg.accessibility_decay_m)
            ))
        )

        t_dec = time.perf_counter()

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
