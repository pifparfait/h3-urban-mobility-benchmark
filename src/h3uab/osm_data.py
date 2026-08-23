from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from h3uab.semantic_layers import (
    LEGACY_CATEGORY_TO_CODE,
    SEMANTIC_LAYERS,
    layer_name,
)


RAW_OSM_CATEGORY_COLUMNS = {
    "amenity",
    "shop",
    "leisure",
    "tourism",
    "office",
    "healthcare",
    "public_transport",
}


def deterministic_attractiveness(osm_id: str) -> float:
    """
    Stable pseudo-attractiveness in [0.5, 1.0].

    It is NOT claimed to be a real rating; it is a controlled behavior
    parameter shared by both backends.
    """
    digest = hashlib.blake2b(
        str(osm_id).encode("utf-8"),
        digest_size=8,
    ).digest()

    value = int.from_bytes(
        digest,
        "big",
    ) / (2**64 - 1)

    return 0.5 + 0.5 * value


def _clean_tag(value) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip().lower()

    if text in {
        "",
        "nan",
        "none",
        "<na>",
    }:
        return ""

    return text


def classify_from_tags(row) -> str | None:
    """
    Best-effort mapping from common OSM tags to the frozen D3-D25 taxonomy.

    The taxonomy itself is NOT changed for Cambridge. This preserves
    cross-city comparability with Madrid.
    """
    amenity = _clean_tag(
        row.get("amenity", "")
    )
    shop = _clean_tag(
        row.get("shop", "")
    )
    leisure = _clean_tag(
        row.get("leisure", "")
    )
    tourism = _clean_tag(
        row.get("tourism", "")
    )
    office = _clean_tag(
        row.get("office", "")
    )
    healthcare = _clean_tag(
        row.get("healthcare", "")
    )
    building = _clean_tag(
        row.get("building", "")
    )
    landuse = _clean_tag(
        row.get("landuse", "")
    )

    # ------------------------------------------------------------------
    # Food / hospitality
    # ------------------------------------------------------------------
    if amenity == "restaurant":
        return "D21"

    if amenity in {
        "cafe",
        "ice_cream",
    }:
        return "D22"

    if amenity in {
        "bar",
        "biergarten",
        "nightclub",
    }:
        return "D23"

    if amenity == "pub":
        return "D24"

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    if amenity == "pharmacy":
        return "D17"

    if healthcare == "pharmacy":
        return "D17"

    if amenity in {
        "hospital",
        "clinic",
        "doctors",
        "dentist",
    }:
        return "D8"

    if healthcare in {
        "hospital",
        "clinic",
        "doctor",
        "doctors",
        "dentist",
        "physiotherapist",
        "psychotherapist",
        "optometrist",
        "laboratory",
        "rehabilitation",
        "alternative",
    }:
        return "D8"

    # ------------------------------------------------------------------
    # Education / childcare / training
    # ------------------------------------------------------------------
    if amenity in {
        "school",
        "college",
        "university",
        "library",
        "music_school",
        "language_school",
    }:
        return "D5"

    if amenity in {
        "kindergarten",
        "childcare",
    }:
        return "D19"

    if amenity in {
        "driving_school",
        "training",
        "cooking_school",
    }:
        return "D18"

    # ------------------------------------------------------------------
    # Religious / government
    # ------------------------------------------------------------------
    if amenity == "place_of_worship":
        return "D7"

    if amenity in {
        "townhall",
        "courthouse",
        "police",
        "fire_station",
        "ranger_station",
        "social_facility",
    }:
        return "D9"

    # ------------------------------------------------------------------
    # Grocery and retail
    # ------------------------------------------------------------------
    if shop in {
        "supermarket",
        "convenience",
        "greengrocer",
        "grocery",
        "butcher",
        "bakery",
        "deli",
        "seafood",
        "cheese",
        "farm",
    }:
        return "D15"

    if amenity == "marketplace":
        return "D15"

    if shop:
        return "D3"

    # ------------------------------------------------------------------
    # Recreation / green / culture
    # ------------------------------------------------------------------
    if leisure in {
        "fitness_centre",
        "sports_centre",
        "stadium",
        "pitch",
        "track",
        "swimming_pool",
        "fitness_station",
        "dance",
        "ice_rink",
        "golf_course",
    }:
        return "D16"

    if leisure in {
        "park",
        "garden",
        "nature_reserve",
        "dog_park",
        "playground",
    }:
        return "D25"

    if tourism in {
        "museum",
        "gallery",
        "attraction",
        "artwork",
        "viewpoint",
        "picnic_site",
    }:
        return "D16"

    if tourism in {
        "hotel",
        "guest_house",
        "hostel",
        "motel",
        "apartment",
    }:
        return "D4"

    # ------------------------------------------------------------------
    # Offices / commercial
    # ------------------------------------------------------------------
    if office == "coworking":
        return "D20"

    if office in {
        "company",
        "corporate",
        "headquarters",
    }:
        return "D14"

    if office:
        return "D4"

    if building in {
        "apartments",
        "residential",
        "house",
        "detached",
    }:
        return "D6"

    if landuse in {
        "commercial",
        "retail",
    }:
        return "D4"

    # Public transport is deliberately NOT mapped here because the frozen
    # D3-D25 taxonomy used in the Madrid benchmark has no transit layer.
    # Adding one only for Cambridge would break cross-city comparability.

    return None


def classify_from_category_subcategory(
    row,
) -> str | None:
    """
    Adapter for the Cambridge Phase-1 schema:

        category      = OSM tag key
        subcategory   = OSM tag value

    Example:
        category='amenity', subcategory='restaurant'
        -> virtual amenity='restaurant'
        -> D21

    This adapter lets Cambridge reuse the exact same semantic taxonomy and
    downstream model as Madrid without changing the experimental architecture.
    """
    category = _clean_tag(
        row.get(
            "category",
            "",
        )
    )

    subcategory = _clean_tag(
        row.get(
            "subcategory",
            "",
        )
    )

    if (
        category not in RAW_OSM_CATEGORY_COLUMNS
        or not subcategory
    ):
        return None

    virtual = {
        "amenity": "",
        "shop": "",
        "leisure": "",
        "tourism": "",
        "office": "",
        "healthcare": "",
        "public_transport": "",
        "building": "",
        "landuse": "",
    }

    virtual[
        category
    ] = subcategory

    return classify_from_tags(
        virtual
    )


def _looks_like_tag_pair_schema(
    df: pd.DataFrame,
) -> bool:
    """
    Distinguish Cambridge's:
        category='amenity', subcategory='restaurant'

    from Madrid's legacy:
        category='food'
    """
    if (
        "category" not in df.columns
        or "subcategory" not in df.columns
    ):
        return False

    categories = set(
        df[
            "category"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    if not categories:
        return False

    overlap = (
        categories
        & RAW_OSM_CATEGORY_COLUMNS
    )

    return len(
        overlap
    ) > 0


def normalize_osm_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize supported OSM cache schemas.

    A) Legacy Madrid cache
       osm_id,name,lat,lng,category,category_id
       with category values such as:
           retail, education, health, food, leisure

    B) Rich OSM cache
       osm_id,name,lat,lng,layer_code
       plus optional raw OSM tag columns.

    C) Cambridge Phase-1 tag-pair cache
       osm_type,osm_id,name,category,subcategory,lat,lng
       where category is the OSM key and subcategory its value.

    All three are projected onto the SAME frozen D3-D25 semantic taxonomy.
    """
    df = df.copy()

    required_geo = {
        "osm_id",
        "lat",
        "lng",
    }

    missing = (
        required_geo
        .difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            f"OSM CSV is missing columns "
            f"{sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )

    if "name" not in df.columns:
        df[
            "name"
        ] = ""

    if "layer_code" not in df.columns:
        # --------------------------------------------------------------
        # Cambridge schema: category stores the OSM tag KEY.
        # --------------------------------------------------------------
        if _looks_like_tag_pair_schema(
            df
        ):
            df[
                "layer_code"
            ] = df.apply(
                classify_from_category_subcategory,
                axis=1,
            )

        # --------------------------------------------------------------
        # Madrid legacy schema: category stores a coarse semantic class.
        # --------------------------------------------------------------
        elif "category" in df.columns:
            df[
                "layer_code"
            ] = (
                df[
                    "category"
                ]
                .astype(str)
                .str.lower()
                .map(
                    LEGACY_CATEGORY_TO_CODE
                )
            )

        # --------------------------------------------------------------
        # Rich raw-tag schema.
        # --------------------------------------------------------------
        else:
            df[
                "layer_code"
            ] = df.apply(
                classify_from_tags,
                axis=1,
            )

    # If layer_code exists but some rows are still missing, try every
    # compatible classifier in a conservative order.
    missing_code = (
        df[
            "layer_code"
        ]
        .isna()
    )

    if missing_code.any():
        if _looks_like_tag_pair_schema(
            df.loc[
                missing_code
            ]
        ):
            df.loc[
                missing_code,
                "layer_code",
            ] = (
                df.loc[
                    missing_code
                ]
                .apply(
                    classify_from_category_subcategory,
                    axis=1,
                )
            )

    missing_code = (
        df[
            "layer_code"
        ]
        .isna()
    )

    if missing_code.any():
        df.loc[
            missing_code,
            "layer_code",
        ] = (
            df.loc[
                missing_code
            ]
            .apply(
                classify_from_tags,
                axis=1,
            )
        )

    df = df[
        df[
            "layer_code"
        ].isin(
            set(
                SEMANTIC_LAYERS.keys()
            )
        )
    ].copy()

    df[
        "layer_name"
    ] = df[
        "layer_code"
    ].map(
        lambda x: layer_name(
            str(
                x
            )
        )
    )

    if "attractiveness" not in df.columns:
        df[
            "attractiveness"
        ] = [
            deterministic_attractiveness(
                x
            )
            for x in df[
                "osm_id"
            ].astype(str)
        ]

    df[
        "lat"
    ] = pd.to_numeric(
        df[
            "lat"
        ],
        errors="coerce",
    )

    df[
        "lng"
    ] = pd.to_numeric(
        df[
            "lng"
        ],
        errors="coerce",
    )

    df[
        "attractiveness"
    ] = pd.to_numeric(
        df[
            "attractiveness"
        ],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "lat",
            "lng",
            "layer_code",
            "attractiveness",
        ]
    )

    return df.reset_index(
        drop=True
    )


def load_osm_pois(
    cfg,
) -> pd.DataFrame:
    """
    Load the city-specific OSM cache configured by cfg.osm_cache_csv.
    """
    path = Path(
        cfg.osm_cache_csv
    )

    if not path.exists():
        raise FileNotFoundError(
            f"OSM cache not found: {path}"
        )

    return normalize_osm_dataframe(
        pd.read_csv(
            path
        )
    )
