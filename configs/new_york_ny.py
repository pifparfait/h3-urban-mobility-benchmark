"""
New York City cross-city validation configuration.

This file contains city-specific constants only. The scientific protocol
remains identical to Madrid and Cambridge:
- Continuous-indexed vs hierarchical H3-indexed architecture
- H7/H8/H9 fidelity evaluation
- H11/H12 localization/visualization only
- shared physical OSM walking network

The Manhattan/Midtown bounds below are visualization windows only; they do
not define the simulation domain.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class NewYorkConfig:
    city_key: str = "new_york_ny"
    place_query: str = "New York City, New York, USA"
    country_iso3: str = "USA"

    population_year: int = 2020
    population_h3_resolution: int = 9
    fidelity_resolutions: tuple[int, ...] = (7, 8, 9)
    localization_h3_resolution: int = 12

    # Small buffer around the official NYC municipal boundary.
    graph_buffer_m: float = 500.0

    # ------------------------------------------------------------------
    # Reproducible visualization windows.
    # Order: (west, east, south, north)
    # ------------------------------------------------------------------

    # Manhattan context window.
    manhattan_zoom_bounds: tuple[float, float, float, float] = (
        -74.0200,
        -73.9300,
        40.7000,
        40.8800,
    )

    # Midtown / Times Square street-level inspection window.
    midtown_deep_zoom_bounds: tuple[float, float, float, float] = (
        -74.0000,
        -73.9700,
        40.7450,
        40.7700,
    )

    @property
    def data_dir(self) -> Path:
        return ROOT / "data" / self.city_key

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def outputs_dir(self) -> Path:
        return ROOT / "outputs" / self.city_key

    @property
    def graph_path(self) -> Path:
        return self.cache_dir / "new_york_walk_graph.graphml"

    @property
    def poi_path(self) -> Path:
        return self.processed_dir / "new_york_osm_pois.csv"

    @property
    def population_h3_path(self) -> Path:
        return self.processed_dir / "new_york_population_h3_r9.csv"

    @property
    def worldpop_raw_path(self) -> Path:
        return (
            self.raw_dir
            / "worldpop"
            / "usa_ppp_2020_UNadj_constrained.tif"
        )

    @property
    def worldpop_clipped_path(self) -> Path:
        return (
            self.processed_dir
            / "worldpop"
            / "new_york_worldpop_2020.tif"
        )

    @property
    def boundary_path(self) -> Path:
        return self.processed_dir / "new_york_boundary.geojson"


NEW_YORK = NewYorkConfig()
