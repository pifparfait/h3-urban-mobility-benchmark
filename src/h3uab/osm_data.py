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


def deterministic_attractiveness(osm_id: str) -> float:
    """
    Stable pseudo-attractiveness in [0.5, 1.0].
    It is NOT claimed to be a real rating; it is a controlled behavior parameter
    shared by both backends.
    """
    digest = hashlib.blake2b(
        str(osm_id).encode("utf-8"),
        digest_size=8,
    ).digest()
    value = int.from_bytes(digest, "big") / (2**64 - 1)
    return 0.5 + 0.5 * value


def classify_from_tags(row) -> str | None:
    """
    Best-effort mapping from common OSM tags to D3-D25.

    This is only used when a richer OSM cache is available. Existing legacy
    5-category caches are normalized separately and remain fully usable.
    """
    amenity = str(row.get("amenity", "") or "").lower()
    shop = str(row.get("shop", "") or "").lower()
    leisure = str(row.get("leisure", "") or "").lower()
    office = str(row.get("office", "") or "").lower()
    building = str(row.get("building", "") or "").lower()
    landuse = str(row.get("landuse", "") or "").lower()

    if amenity == "restaurant":
        return "D21"
    if amenity == "cafe":
        return "D22"
    if amenity == "bar":
        return "D23"
    if amenity == "pub":
        return "D24"
    if amenity == "pharmacy":
        return "D17"
    if amenity in {"hospital", "clinic", "doctors", "dentist"}:
        return "D8"
    if amenity in {"school", "college", "university", "library"}:
        return "D5"
    if amenity == "kindergarten":
        return "D19"
    if amenity in {"place_of_worship"}:
        return "D7"
    if amenity in {"townhall", "courthouse", "police", "fire_station"}:
        return "D9"

    if shop in {"supermarket", "convenience", "greengrocer", "marketplace"}:
        return "D15"
    if shop and shop != "nan":
        return "D3"

    if leisure in {"fitness_centre", "sports_centre", "stadium", "pitch"}:
        return "D16"
    if leisure in {"park", "garden"}:
        return "D25"

    if office in {"coworking"}:
        return "D20"
    if office in {"company", "corporate", "headquarters"}:
        return "D14"
    if office and office != "nan":
        return "D4"

    if building in {"apartments", "residential", "house", "detached"}:
        return "D6"
    if landuse in {"commercial", "retail"}:
        return "D4"

    return None


def normalize_osm_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize either:
      A) old repository cache:
         osm_id,name,lat,lng,category,category_id
      B) richer cache:
         osm_id,name,lat,lng,layer_code and optional raw OSM tags.
    """
    df = df.copy()

    required_geo = {"osm_id", "lat", "lng"}
    missing = required_geo.difference(df.columns)
    if missing:
        raise ValueError(
            f"OSM CSV is missing columns {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )

    if "name" not in df.columns:
        df["name"] = ""

    if "layer_code" not in df.columns:
        if "category" in df.columns:
            df["layer_code"] = (
                df["category"]
                .astype(str)
                .str.lower()
                .map(LEGACY_CATEGORY_TO_CODE)
            )
        else:
            df["layer_code"] = df.apply(
                classify_from_tags,
                axis=1,
            )

    # If layer_code exists but some rows are missing, try raw tags.
    missing_code = df["layer_code"].isna()
    if missing_code.any():
        df.loc[missing_code, "layer_code"] = (
            df.loc[missing_code]
            .apply(classify_from_tags, axis=1)
        )

    df = df[
        df["layer_code"].isin(
            set(SEMANTIC_LAYERS.keys())
        )
    ].copy()

    df["layer_name"] = df["layer_code"].map(
        lambda x: layer_name(str(x))
    )

    if "attractiveness" not in df.columns:
        df["attractiveness"] = [
            deterministic_attractiveness(x)
            for x in df["osm_id"].astype(str)
        ]

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df["attractiveness"] = pd.to_numeric(
        df["attractiveness"],
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

    return df.reset_index(drop=True)


def load_osm_pois(cfg) -> pd.DataFrame:
    """
    Load the real Madrid OSM cache already used by the project.
    """
    path = Path(cfg.osm_cache_csv)

    if not path.exists():
        raise FileNotFoundError(
            f"OSM cache not found: {path}\n"
            "Use the existing madrid_osm_pois.csv from the previous repository."
        )

    return normalize_osm_dataframe(
        pd.read_csv(path)
    )
