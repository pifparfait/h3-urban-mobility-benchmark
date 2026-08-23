"""
Cambridge, MA cross-city validation configuration.

This file deliberately contains only city-specific constants. The scientific
benchmark remains unchanged: Continuous-indexed vs H3-indexed, H7/H8/H9
fidelity evaluation, and a shared physical OSM walking network.

Kendall/Volpe bounds are visualization windows, not simulation boundaries.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CambridgeConfig:
    city_key: str = "cambridge_ma"
    place_query: str = "Cambridge, Massachusetts, USA"
    country_iso3: str = "USA"

    # Same population vintage used by the Madrid benchmark.
    population_year: int = 2020

    # H3 resolution used for the population surface / agent sampling.
    population_h3_resolution: int = 9

    # Fidelity scales: keep identical to Madrid.
    fidelity_resolutions: tuple[int, ...] = (7, 8, 9)

    # H11/H12 remain localization/visualization helpers only.
    localization_h3_resolution: int = 12

    # Small graph buffer avoids artificial clipping at Cambridge's municipal edge.
    graph_buffer_m: float = 500.0

    # ------------------------------------------------------------------
    # Reproducible visualization windows.
    # Order everywhere in this project:
    #     (west, east, south, north)
    # ------------------------------------------------------------------

    # Kendall Square / MIT / East Cambridge local context.
    kendall_zoom_bounds: tuple[float, float, float, float] = (
        -71.1030,
        -71.0735,
        42.3525,
        42.3745,
    )

    # Volpe / Kendall deep street-level inspection window.
    # This is intentionally a display window only.
    volpe_deep_zoom_bounds: tuple[float, float, float, float] = (
        -71.0925,
        -71.0810,
        42.3590,
        42.3675,
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
        return self.cache_dir / "cambridge_walk_graph.graphml"

    @property
    def poi_path(self) -> Path:
        return self.processed_dir / "cambridge_osm_pois.csv"

    @property
    def population_h3_path(self) -> Path:
        return self.processed_dir / "cambridge_population_h3_r9.csv"

    @property
    def worldpop_raw_path(self) -> Path:
        return self.raw_dir / "worldpop" / "usa_ppp_2020_UNadj_constrained.tif"

    @property
    def worldpop_clipped_path(self) -> Path:
        return self.processed_dir / "worldpop" / "cambridge_worldpop_2020.tif"

    @property
    def boundary_path(self) -> Path:
        return self.processed_dir / "cambridge_boundary.geojson"


CAMBRIDGE = CambridgeConfig()
