from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class H3Config:
    macro_res: int = 5
    amenity_coarse_res: int = 7
    amenity_functional_res: int = 8
    accessibility_res: int = 9
    amenity_index_res: int = 10

    # IMMUTABLE ARCHITECTURAL RULE:
    # H11/H12 are agent localization / visualization layers only.
    interaction_res: int = 11
    position_res: int = 12

    def validate(self):
        semantic = (
            self.macro_res,
            self.amenity_coarse_res,
            self.amenity_functional_res,
            self.accessibility_res,
            self.amenity_index_res,
        )
        if tuple(sorted(semantic)) != semantic:
            raise ValueError("Semantic H3 resolutions must be increasing.")
        if self.interaction_res != 11 or self.position_res != 12:
            raise ValueError(
                "H11/H12 are frozen as localization/visualization layers."
            )


@dataclass(frozen=True)
class RoutingConfig:
    # Routing is deliberately excluded from architecture speedup.
    # When enabled for physical mobility, BOTH backends use the SAME OSM graph.
    enabled: bool = False
    network_type: str = "walk"
    route_weight: str = "length"
    walking_speed_m_s: float = 1.4

    graphml_path: Path = Path("data/cache/madrid_walk_graph.graphml")
    h3_index_path: Path = Path("data/cache/madrid_walk_graph_h3_index.pkl")
    routing_matrix_dir: Path = Path("data/cache/routing_matrix")
    routing_matrix_distance_file: str = "distance_matrix.float32.dat"
    routing_matrix_nodes_file: str = "destination_node_ids.npy"
    routing_matrix_poi_index_file: str = "poi_to_matrix_index.npy"
    routing_matrix_metadata_file: str = "metadata.json"
    routing_matrix_next_hop_file: str = "next_hop.int32.dat"
    routing_matrix_batch_size: int = 32
    route_cache_max_entries: int = 250_000
    edge_h11_sample_spacing_m: float = 15.0
    allow_unroutable_movement: bool = True
    routing_is_common_physical_infrastructure: bool = True
    use_compressed_next_hop: bool = False


@dataclass(frozen=True)
class DestinationRefinementConfig:
    enabled: bool = True
    max_candidates: int = 24
    h10_k_ring: int = 1
    enable_fallback: bool = True
    fallback_h9_k_ring: int = 1
    fallback_max_candidates: int = 48


@dataclass(frozen=True)
class BaselineConfig:
    primary_continuous: str = "indexed"
    run_naive_dynamic: bool = False
    indexed_leaf_size: int = 40


@dataclass(frozen=True)
class Config:
    seed: int = 42

    lat_min: float = 40.38
    lat_max: float = 40.50
    lng_min: float = -3.75
    lng_max: float = -3.55

    n_agents: int = 10_000
    n_steps: int = 8

    search_radius_m: float = 2_000.0
    accessibility_decay_m: float = 1_000.0
    beta_distance: float = 0.001
    gamma_attractiveness: float = 0.75

    # Existing validated Madrid assets are linked/copied here by scripts/link_data.py
    osm_cache_csv: Path = Path("data/processed/madrid_osm_pois.csv")
    population_h3_csv: Path = Path("data/processed/madrid_population_h3_r9.csv")
    population_h3_id_col: str = "h3_id"
    population_col: str = "population"
    static_city_cache: Path = Path("data/cache/madrid_static_city_h3.pkl")
    rebuild_static_city: bool = False

    worldpop_spain_tif: Path = Path(
        "data/raw/worldpop/esp_ppp_2020_UNadj_constrained.tif"
    )
    worldpop_madrid_tif: Path = Path(
        "data/processed/worldpop/madrid_worldpop_2020.tif"
    )
    worldpop_url: str = (
        "https://data.worldpop.org/GIS/Population/"
        "Global_2000_2020_Constrained/2020/BSGM/ESP/"
        "esp_ppp_2020_UNadj_constrained.tif"
    )

    relative_error_floor_absolute: float = 1e-6
    relative_error_floor_fraction_of_positive_median: float = 0.01

    h3: H3Config = field(default_factory=H3Config)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    destination_refinement: DestinationRefinementConfig = field(
        default_factory=DestinationRefinementConfig
    )
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
