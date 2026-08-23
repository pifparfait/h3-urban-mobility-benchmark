from abc import ABC, abstractmethod

from h3uab.models import BackendProfile


class SpatialBackend(ABC):
    name = "base"

    def __init__(self):
        self.profile = BackendProfile()

        self.road_graph = None
        self.road_index = None
        self.routing_oracle = None
        self.router = None

        self.poi_osm_nodes = None
        self.initial_agent_osm_nodes = None

    def prepare_routing(self, world, cfg):
        if not cfg.routing.enabled:
            self.road_graph = None
            self.road_index = None
            self.routing_oracle = None
            self.router = None
            self.poi_osm_nodes = None
            self.initial_agent_osm_nodes = None
            return 0.0

        from h3uab.road_network import (
            RoadRouter,
            build_initial_agent_osm_nodes,
            build_poi_osm_node_array,
            load_graph_index_oracle,
        )

        (
            G,
            index,
            oracle,
            load_time,
        ) = load_graph_index_oracle(cfg)

        self.road_graph = G
        self.road_index = index
        self.routing_oracle = oracle

        self.poi_osm_nodes = build_poi_osm_node_array(
            G,
            oracle,
        )

        (
            self.initial_agent_osm_nodes,
            n_unique_h12,
            initial_snap_time,
        ) = build_initial_agent_osm_nodes(
            G,
            world,
            cfg,
        )

        self.profile.road_graph_load_time_s = load_time
        self.profile.routing_initial_snap_setup_time_s = (
            initial_snap_time
        )
        self.profile.routing_initial_unique_h12_cells = (
            n_unique_h12
        )

        self.router = RoadRouter(
            G,
            index,
            oracle,
            cfg,
        )

        print(f"  [{self.name}] routing setup:")
        print(
            f"      POI→OSM nodes                : "
            f"{len(self.poi_osm_nodes):,}"
        )
        print(
            f"      initial unique H3@12 cells   : "
            f"{n_unique_h12:,}"
        )
        print(
            f"      initial batch snapping time  : "
            f"{initial_snap_time:.3f}s"
        )
        print(
            "      online nearest_nodes target  : 0"
        )

        return load_time + initial_snap_time

    def route_batch_nodes(
        self,
        origin_nodes,
        destination_nodes,
        progress_label="routing",
        step_timing=None,
    ):
        if self.router is None:
            return None

        result = self.router.route_batch_nodes(
            origin_nodes,
            destination_nodes,
            progress_label=progress_label,
            step_timing=step_timing,
        )

        self.profile.route_time_s = self.router.route_time_s
        self.profile.route_cache_hits = self.router.cache_hits
        self.profile.route_cache_misses = self.router.cache_misses
        self.profile.routable_trips = self.router.routable_trips
        self.profile.unroutable_trips = self.router.unroutable_trips

        self.profile.traversed_osm_edges = int(
            sum(
                self.router.edge_flows.values()
            )
        )

        self.profile.traversed_h11_cells = int(
            sum(
                self.router.h11_route_flows.values()
            )
        )

        self.profile.online_nearest_node_calls = (
            self.router.online_nearest_node_calls
        )

        return result

    def edge_flows(self):
        if self.router is None:
            return {}

        return dict(
            self.router.edge_flows
        )

    def h11_route_flows(self):
        if self.router is None:
            return {}

        return dict(
            self.router.h11_route_flows
        )

    @abstractmethod
    def prepare(self, world, cfg):
        ...

    @abstractmethod
    def decide(
        self,
        agent_idx,
        agent_lat,
        agent_lng,
        layer_code,
    ):
        ...

    @abstractmethod
    def move_agent(
        self,
        agent_idx,
        destination_id,
        old_lat,
        old_lng,
        new_lat,
        new_lng,
    ):
        ...