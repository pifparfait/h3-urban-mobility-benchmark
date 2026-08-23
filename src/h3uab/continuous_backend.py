import time
import numpy as np

from h3uab.behavior import destination_score
from h3uab.geometry import haversine_m
from h3uab.models import SpatialDecision
from h3uab.spatial_base import SpatialBackend


class ContinuousBackend(SpatialBackend):
    """
    Continuous destination reference: O(P_D) per decision.

    Physical routing is NOT different between models. Continuous and H3 use
    the same real OSM graph, H3@12 snapping interface and shortest-path rule.
    """

    name = "continuous"

    def prepare(self, world, cfg):
        t0 = time.perf_counter()
        self.world = world
        self.cfg = cfg

        self.by_layer = {
            layer_code: np.flatnonzero(world.amenity_layer_code == layer_code)
            for layer_code in world.active_layer_codes
        }

        self.prepare_routing(
            world,
            cfg,
        )

        elapsed = time.perf_counter() - t0
        self.profile.initialization_time_s = elapsed
        return elapsed

    def decide(self, agent_idx, agent_lat, agent_lng, layer_code):
        t0 = time.perf_counter()
        ids = self.by_layer.get(layer_code, np.empty(0, dtype=np.int64))
        self.profile.lookup_time_s += time.perf_counter() - t0

        if ids.size == 0:
            return SpatialDecision(-1, 0.0, 0, 0.0)

        td = time.perf_counter()
        distances = haversine_m(
            agent_lat,
            agent_lng,
            self.world.amenity_lat[ids],
            self.world.amenity_lng[ids],
        )
        self.profile.distance_time_s += time.perf_counter() - td

        keep = distances <= self.cfg.search_radius_m
        candidates = ids[keep]
        distances = distances[keep]

        if candidates.size == 0:
            return SpatialDecision(-1, 0.0, 0, 0.0)

        attractiveness = self.world.amenity_attractiveness[candidates]
        accessibility = float(
            np.sum(
                attractiveness
                * np.exp(-distances / self.cfg.accessibility_decay_m)
            )
        )

        t_dec = time.perf_counter()
        scores = destination_score(
            distances,
            attractiveness,
            self.cfg.beta_distance,
            self.cfg.gamma_attractiveness,
        )
        best = int(np.argmax(scores))
        self.profile.decision_time_s += time.perf_counter() - t_dec

        return SpatialDecision(
            destination_id=int(candidates[best]),
            straight_distance_m=float(distances[best]),
            candidate_count=int(candidates.size),
            weighted_accessibility=accessibility,
        )

    def move_agent(self, *args, **kwargs):
        return
