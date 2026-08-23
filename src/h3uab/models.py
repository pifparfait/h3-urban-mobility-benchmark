from dataclasses import dataclass
import numpy as np


@dataclass
class World:
    agent_lat: np.ndarray
    agent_lng: np.ndarray

    agent_profile_type: np.ndarray
    profile_schedule_purpose_ids: np.ndarray
    profile_schedule_layer_ids: np.ndarray

    amenity_lat: np.ndarray
    amenity_lng: np.ndarray
    amenity_layer_code: np.ndarray
    amenity_attractiveness: np.ndarray
    amenity_osm_id: np.ndarray
    amenity_name: np.ndarray

    active_layer_codes: tuple[str, ...]


@dataclass(frozen=True)
class AmenityAggregate:
    count: int
    attractiveness_sum: float


@dataclass(frozen=True)
class AccessibilityState:
    candidate_count: int
    weighted_accessibility: float
    best_poi_id: int
    best_distance_m: float


@dataclass(frozen=True)
class MacroState:
    population: float
    poi_count: int


@dataclass
class SpatialDecision:
    destination_id: int
    straight_distance_m: float
    candidate_count: int
    weighted_accessibility: float


@dataclass
class BackendProfile:
    initialization_time_s: float = 0.0

    lookup_time_s: float = 0.0
    distance_time_s: float = 0.0
    decision_time_s: float = 0.0
    move_update_time_s: float = 0.0

    static_city_load_time_s: float = 0.0

    road_graph_load_time_s: float = 0.0
    routing_initial_snap_setup_time_s: float = 0.0
    routing_initial_unique_h12_cells: int = 0

    route_time_s: float = 0.0
    route_cache_hits: int = 0
    route_cache_misses: int = 0
    routable_trips: int = 0
    unroutable_trips: int = 0
    traversed_osm_edges: int = 0
    traversed_h11_cells: int = 0

    online_nearest_node_calls: int = 0

    mobility_flow_updates: int = 0
    occupancy_updates: int = 0


@dataclass
class SimulationResult:
    backend_name: str
    runtime_s: float
    initialization_time_s: float
    profile: BackendProfile

    chosen_destinations: np.ndarray

    straight_distances_m: np.ndarray
    network_distances_m: np.ndarray
    travel_times_s: np.ndarray
    route_edge_counts: np.ndarray
    route_node_counts: np.ndarray
    route_routable: np.ndarray

    candidate_counts: np.ndarray
    accessibility_scores: np.ndarray

    agent_ids: np.ndarray
    steps: np.ndarray

    profile_types: np.ndarray
    activity_purpose_ids: np.ndarray
    layer_ids: np.ndarray

    lat_before: np.ndarray
    lng_before: np.ndarray
    lat_after: np.ndarray
    lng_after: np.ndarray

    final_agent_lat: np.ndarray
    final_agent_lng: np.ndarray
    final_agent_osm_nodes: np.ndarray

    edge_flows: dict
    h11_route_flows: dict
