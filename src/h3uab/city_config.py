from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from h3uab.config import Config, RoutingConfig


SUPPORTED_CITIES = (
    "madrid",
    "cambridge_ma",
)


def normalize_city_name(city: str) -> str:
    value = str(city).strip().lower()

    aliases = {
        "madrid": "madrid",
        "madrid_es": "madrid",
        "madrid_spain": "madrid",
        "cambridge": "cambridge_ma",
        "cambridge_ma": "cambridge_ma",
        "cambridge_massachusetts": "cambridge_ma",
        "kendall": "cambridge_ma",
        "kendall_square": "cambridge_ma",
        "volpe": "cambridge_ma",
    }

    if value not in aliases:
        raise ValueError(
            f"Unsupported city {city!r}. "
            f"Supported: {', '.join(SUPPORTED_CITIES)}"
        )

    return aliases[value]


def _madrid_config(n_agents: int | None = None) -> Config:
    cfg = Config()

    if n_agents is not None:
        cfg = replace(
            cfg,
            n_agents=int(n_agents),
        )

    return cfg


def _cambridge_config(n_agents: int | None = None) -> Config:
    """
    Cambridge configuration using exactly the same scientific architecture
    as Madrid.

    IMPORTANT
    ---------
    - H7/H8/H9 remain the urban-fidelity analysis scales.
    - H11/H12 remain localization / visualization only.
    - routing.enabled defaults to False for architecture benchmarks.
    - when routing is enabled, both backends use the same Cambridge OSM graph.
    """

    base = Config()

    routing = replace(
        base.routing,
        graphml_path=Path(
            "data/cambridge_ma/cache/cambridge_walk_graph.graphml"
        ),
        h3_index_path=Path(
            "data/cambridge_ma/cache/"
            "cambridge_walk_graph_h3_index.pkl"
        ),
        routing_matrix_dir=Path(
            "data/cambridge_ma/cache/routing_matrix"
        ),
    )

    cfg = replace(
        base,

        city_name="cambridge_ma",

        # Bounds contain Cambridge plus the small OSM graph buffer.
        lat_min=42.3470,
        lat_max=42.4100,
        lng_min=-71.1690,
        lng_max=-71.0570,

        n_agents=(
            int(n_agents)
            if n_agents is not None
            else base.n_agents
        ),

        osm_cache_csv=Path(
            "data/cambridge_ma/processed/"
            "cambridge_osm_pois.csv"
        ),

        population_h3_csv=Path(
            "data/cambridge_ma/processed/"
            "cambridge_population_h3_r9.csv"
        ),

        # Cambridge Phase 1 builder writes the H3 identifier as "h3".
        population_h3_id_col="h3",

        static_city_cache=Path(
            "data/cambridge_ma/cache/"
            "cambridge_static_city_h3.pkl"
        ),

        worldpop_country_tif=Path(
            "data/cambridge_ma/raw/worldpop/"
            "usa_ppp_2020_UNadj_constrained.tif"
        ),

        worldpop_city_tif=Path(
            "data/cambridge_ma/processed/worldpop/"
            "cambridge_worldpop_2020.tif"
        ),

        worldpop_url=(
            "https://data.worldpop.org/GIS/Population/"
            "Global_2000_2020_Constrained/2020/BSGM/USA/"
            "usa_ppp_2020_UNadj_constrained.tif"
        ),

        routing=routing,
    )

    cfg.h3.validate()
    return cfg


def config_for_city(
    city: str,
    *,
    n_agents: int | None = None,
) -> Config:
    city = normalize_city_name(city)

    if city == "madrid":
        return _madrid_config(
            n_agents=n_agents,
        )

    if city == "cambridge_ma":
        return _cambridge_config(
            n_agents=n_agents,
        )

    raise AssertionError(
        f"Unhandled normalized city: {city}"
    )
