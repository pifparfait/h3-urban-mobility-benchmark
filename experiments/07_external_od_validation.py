#!/usr/bin/env python3
"""
07_external_od_validation.py
============================

External OD validation for the H3 urban simulation benchmark.

Purpose
-------
Compare three OD representations on a common H3 lattice:

    1. Observed OD (e.g. LEHD/LODES JT00)
    2. Continuous-backend simulated OD
    3. H3-backend simulated OD

The experiment is explicitly a *non-degradation* validation:
it asks whether the H3 architecture preserves the agreement with observed
mobility achieved by the continuous baseline.

For each H3 resolution (default: 7, 8, 9), the script computes:
    - Pearson correlation
    - Spearman correlation
    - absolute-flow NMAE
    - Jensen-Shannon divergence
    - cosine similarity
    - total-flow ratio
    - support overlap
    - normalized OD-share MAE / L1 / total-variation distance
    - normalized OD-share JSD and cosine similarity
    - explicit H3-minus-Continuous non-degradation gaps

The normalized OD-share metrics separate spatial-structure disagreement from
differences in total trip mass between observed and simulated datasets.

Cambridge / New York LODES
----------------------------
Expected observed file: LEHD LODES OD JT00 CSV, with:
    w_geocode, h_geocode, S000

Expected TIGER block file:
    tl_2020_25_tabblock20.zip
containing 2020 Massachusetts census blocks.

Important
---------
LODES JT00 is commuting (home -> work), not general multipurpose mobility.
Therefore simulated inputs used for this validation should contain only
work/commuting trips when comparing against LODES.


Madrid / INE mobile-phone mobility
----------------------------------
Expected observed data:
    - INE November 2019 OD flow matrix:
        Flujos+15_O-D_M1_NOV.csv
      or the outer datos_disponibles.zip containing Noviembre 2019.zip.
    - INE mobility-area polygons:
        celdas_marzo_2020.shp
      or shapefiles_celdas_marzo2020.zip.

Madrid mobility areas are larger than U.S. Census Blocks. Therefore the
Madrid adapter does NOT assign each area to a single H3 centroid. It clips
the INE mobility polygons to the exact simulation boundary, builds an
area-to-H3 fractional-area crosswalk at H7/H8/H9, and projects OD flows via:

    F_H3 = W^T F_INE W

where W is the area-to-H3 allocation matrix. This preserves total OD mass
(up to numerical tolerance) while avoiding artificial centroid assignment.

Important source comparability note
-----------------------------------
LODES is home-to-work commuting. The INE November-2019 M1 matrix represents
residence-to-usual/daytime mobility inferred from mobile-phone locations and
is not a pure work-only trip survey. By default the paired simulation loader
uses work-purpose OD as the closest controlled proxy. This source mismatch
must be reported as a limitation; the architectural comparison remains
differential because Continuous and H3 use the identical behavioural model.

This script does NOT claim to validate the behavioral model absolutely.
It measures whether replacing continuous spatial computation with H3
materially changes agreement with the same observed OD reference.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy import sparse
except ImportError as exc:
    raise SystemExit(
        "scipy is required for Madrid area-weighted OD projection. "
        "Install it with: pip install scipy"
    ) from exc

try:
    from pyproj import Transformer
except ImportError as exc:
    raise SystemExit(
        "pyproj is required for Madrid area-weighted projection. "
        "Install it with: pip install pyproj"
    ) from exc

try:
    from shapely.geometry import Polygon, mapping
    from shapely.ops import transform as shapely_transform
except ImportError as exc:
    raise SystemExit(
        "shapely is required for Madrid area-weighted projection. "
        "Install it with: pip install shapely"
    ) from exc

try:
    import geopandas as gpd
except ImportError as exc:
    raise SystemExit(
        "geopandas is required. Install it with: pip install geopandas"
    ) from exc

try:
    import h3
except ImportError as exc:
    raise SystemExit(
        "h3 is required. Install it with: pip install h3"
    ) from exc


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_RESOLUTIONS = (7, 8, 9)
EPS = 1e-12

# Cambridge city FIPS in Massachusetts:
# state 25, county Middlesex 017, place Cambridge 11000.
CAMBRIDGE_PLACEFP = "11000"


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_geoid(series: pd.Series, width: int = 15) -> pd.Series:
    """
    Normalize Census block GEOIDs.

    LODES geocodes may be parsed as integers, strings, or floats by pandas.
    Census block GEOIDs are 15 digits.
    """
    s = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    return s.str.zfill(width)


def latlng_to_h3(lat: float, lon: float, resolution: int) -> str:
    """Compatibility wrapper for h3-py v3/v4."""
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(float(lat), float(lon), int(resolution))
    return h3.geo_to_h3(float(lat), float(lon), int(resolution))


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) <= EPS or np.std(y) <= EPS:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(x.corr(y, method="spearman"))


def nmae(pred: np.ndarray, obs: np.ndarray) -> float:
    """
    Mean absolute error normalized by mean observed flow.

    NMAE = mean(|pred-obs|) / mean(obs)
    """
    denom = float(np.mean(obs))
    if denom <= EPS:
        return np.nan
    return float(np.mean(np.abs(pred - obs)) / denom)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= EPS:
        return np.nan
    return float(np.dot(x, y) / denom)


def js_divergence(x: np.ndarray, y: np.ndarray) -> float:
    """
    Jensen-Shannon divergence using natural logarithms.
    Returns [0, ln(2)].
    """
    sx = float(x.sum())
    sy = float(y.sum())
    if sx <= EPS or sy <= EPS:
        return np.nan

    p = x.astype(float) / sx
    q = y.astype(float) / sy
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def normalize_distribution(
    values: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """
    Normalize a non-negative vector to unit mass.

    This is used to compare OD *structure* independently of differences in
    total trip counts between observed and simulated datasets.
    """
    x = np.asarray(values, dtype=float)

    if np.any(~np.isfinite(x)):
        raise ValueError(f"{label} contains non-finite values.")
    if np.any(x < -EPS):
        raise ValueError(f"{label} contains negative flow values.")

    x = np.clip(x, 0.0, None)
    total = float(x.sum())

    if total <= EPS:
        raise ValueError(f"{label} has zero total flow; cannot normalize.")

    p = x / total

    if not np.isclose(float(p.sum()), 1.0, rtol=0.0, atol=1e-10):
        raise AssertionError(
            f"{label} normalization failed: shares sum to {p.sum():.16f}"
        )

    return p


def distribution_metrics(
    observed_flow: np.ndarray,
    predicted_flow: np.ndarray,
    *,
    label: str,
) -> dict:
    """
    Compare normalized OD distributions.

    Metrics
    -------
    share_mae
        Mean absolute difference between OD shares.
    share_l1
        L1 distance between the two normalized distributions.
    total_variation_distance
        0.5 * L1; bounded in [0, 1].
    share_nmae
        MAE normalized by the mean observed share. Since the observed
        distribution sums to 1, this equals the L1 distance numerically.
        It is retained for continuity with the paper's NMAE terminology.
    jsd_normalized
        Jensen-Shannon divergence between unit-mass OD distributions.
    cosine_normalized
        Cosine similarity between unit-mass OD distributions.
    """
    obs_share = normalize_distribution(
        observed_flow,
        label=f"{label}: observed",
    )
    pred_share = normalize_distribution(
        predicted_flow,
        label=f"{label}: predicted",
    )

    abs_diff = np.abs(pred_share - obs_share)
    share_mae = float(np.mean(abs_diff))
    share_l1 = float(np.sum(abs_diff))
    tvd = 0.5 * share_l1

    mean_obs_share = float(np.mean(obs_share))
    share_nmae = (
        float(share_mae / mean_obs_share)
        if mean_obs_share > EPS
        else np.nan
    )

    # Conservation / metric-range invariants.
    if share_l1 < -EPS or share_l1 > 2.0 + 1e-10:
        raise AssertionError(
            f"{label}: normalized L1 outside [0,2]: {share_l1}"
        )
    if tvd < -EPS or tvd > 1.0 + 1e-10:
        raise AssertionError(
            f"{label}: TVD outside [0,1]: {tvd}"
        )

    return {
        "share_mae": share_mae,
        "share_l1": share_l1,
        "share_nmae": share_nmae,
        "total_variation_distance": tvd,
        "jsd_normalized": js_divergence(obs_share, pred_share),
        "cosine_normalized": cosine_similarity(obs_share, pred_share),
    }


def relative_gap(
    h3_value: float,
    continuous_value: float,
) -> float:
    """
    Relative H3-vs-Continuous gap for error metrics.

    Positive values mean H3 has larger error than Continuous.
    Negative values mean H3 has smaller error.
    """
    if (
        not np.isfinite(h3_value)
        or not np.isfinite(continuous_value)
        or abs(continuous_value) <= EPS
    ):
        return np.nan

    return float(
        (h3_value - continuous_value)
        / abs(continuous_value)
    )



def parse_resolutions(values: Iterable[int]) -> list[int]:
    out = sorted(set(int(v) for v in values))
    for r in out:
        if not 0 <= r <= 15:
            raise ValueError(f"Invalid H3 resolution: {r}")
    return out


# =============================================================================
# TIGER / CENSUS BLOCK GEOMETRY
# =============================================================================

def load_tiger_blocks(tiger_path: Path) -> gpd.GeoDataFrame:
    """
    Load Census TIGER/Line tabblock20 from .zip, .shp, or directory.
    """
    if not tiger_path.exists():
        raise FileNotFoundError(tiger_path)

    print(f"[load] TIGER blocks: {tiger_path}")

    if tiger_path.suffix.lower() == ".zip":
        source = f"zip://{tiger_path.resolve()}"
        blocks = gpd.read_file(source)
    else:
        blocks = gpd.read_file(tiger_path)

    geoid_candidates = ["GEOID20", "GEOID", "BLOCKID20"]
    geoid_col = next((c for c in geoid_candidates if c in blocks.columns), None)
    if geoid_col is None:
        raise ValueError(
            f"Could not find Census block GEOID column. "
            f"Available columns: {list(blocks.columns)}"
        )

    blocks = blocks.rename(columns={geoid_col: "block_geoid"}).copy()
    blocks["block_geoid"] = normalize_geoid(blocks["block_geoid"])

    if blocks.crs is None:
        raise ValueError("TIGER block file has no CRS.")

    # Representative points are safer than raw centroids for irregular polygons.
    # Compute in a projected CRS to avoid geographic-centroid warnings.
    projected = blocks.to_crs(3857)
    rep = projected.geometry.representative_point()
    rep_wgs84 = gpd.GeoSeries(rep, crs=3857).to_crs(4326)

    blocks["lon"] = rep_wgs84.x.to_numpy()
    blocks["lat"] = rep_wgs84.y.to_numpy()

    keep = ["block_geoid", "lat", "lon", "geometry"]
    for c in ["STATEFP20", "COUNTYFP20", "TRACTCE20", "BLOCKCE20"]:
        if c in blocks.columns:
            keep.append(c)

    return blocks[keep].copy()


def build_block_h3_crosswalk(
    blocks: gpd.GeoDataFrame,
    resolutions: list[int],
) -> pd.DataFrame:
    """
    Build block GEOID -> H3 cell crosswalk using a representative point.
    """
    out = blocks[["block_geoid", "lat", "lon"]].copy()

    for r in resolutions:
        print(f"[crosswalk] Census blocks -> H3 r={r}")
        out[f"h3_{r}"] = [
            latlng_to_h3(lat, lon, r)
            for lat, lon in zip(out["lat"], out["lon"])
        ]

    return out


# =============================================================================
# OBSERVED LODES OD
# =============================================================================

def load_lodes_jt00(path: Path) -> pd.DataFrame:
    """
    Load LEHD/LODES OD JT00.

    Required:
        w_geocode : workplace census block
        h_geocode : home census block
        S000      : total jobs/workers
    """
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"[load] LODES JT00: {path}")
    df = pd.read_csv(
        path,
        dtype={"w_geocode": "string", "h_geocode": "string"},
        low_memory=False,
    )

    required = {"w_geocode", "h_geocode", "S000"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"LODES file missing columns {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )

    df = df[["w_geocode", "h_geocode", "S000"]].copy()
    df["w_geocode"] = normalize_geoid(df["w_geocode"])
    df["h_geocode"] = normalize_geoid(df["h_geocode"])
    df["S000"] = pd.to_numeric(df["S000"], errors="coerce").fillna(0.0)
    df = df[df["S000"] > 0].copy()

    return df


def attach_block_coordinates(
    od: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Attach H3 cells to home and workplace block GEOIDs.
    """
    home = crosswalk.add_prefix("home_")
    work = crosswalk.add_prefix("work_")

    merged = od.merge(
        home,
        left_on="h_geocode",
        right_on="home_block_geoid",
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        work,
        left_on="w_geocode",
        right_on="work_block_geoid",
        how="left",
        validate="many_to_one",
    )

    home_ok = merged["home_lat"].notna()
    work_ok = merged["work_lat"].notna()
    both_ok = home_ok & work_ok

    stats = {
        "lodes_rows": int(len(od)),
        "unique_home_blocks": int(od["h_geocode"].nunique()),
        "unique_work_blocks": int(od["w_geocode"].nunique()),
        "home_geoid_match_pct": float(100 * home_ok.mean()),
        "work_geoid_match_pct": float(100 * work_ok.mean()),
        "both_geoid_match_pct": float(100 * both_ok.mean()),
        "observed_workers_total": float(od["S000"].sum()),
        "observed_workers_matched": float(merged.loc[both_ok, "S000"].sum()),
    }

    return merged, stats


def aggregate_observed_h3(
    od_joined: pd.DataFrame,
    resolution: int,
    domain_cells: set[str] | None = None,
    domain_rule: str = "either",
) -> pd.DataFrame:
    """
    Aggregate LODES block-to-block commuting flows to H3 OD.

    LODES direction:
        home -> workplace

    domain_rule:
        both   : both origin and destination must be inside domain_cells
        origin : origin must be inside
        dest   : destination must be inside
        either : at least one endpoint must be inside
        none   : no domain filtering
    """
    o = f"home_h3_{resolution}"
    d = f"work_h3_{resolution}"

    x = od_joined[[o, d, "S000"]].dropna().copy()
    x = x.rename(columns={o: "origin_h3", d: "destination_h3", "S000": "flow"})

    if domain_cells is not None and domain_rule != "none":
        in_o = x["origin_h3"].isin(domain_cells)
        in_d = x["destination_h3"].isin(domain_cells)

        if domain_rule == "both":
            mask = in_o & in_d
        elif domain_rule == "origin":
            mask = in_o
        elif domain_rule == "dest":
            mask = in_d
        elif domain_rule == "either":
            mask = in_o | in_d
        else:
            raise ValueError(f"Unknown domain rule: {domain_rule}")

        x = x.loc[mask].copy()

    out = (
        x.groupby(["origin_h3", "destination_h3"], as_index=False)["flow"]
        .sum()
        .sort_values("flow", ascending=False)
        .reset_index(drop=True)
    )
    return out



# =============================================================================
# MADRID / INE MOBILITY OD
# =============================================================================

MADRID_METRIC_CRS = "EPSG:25830"


def _clean_ine_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip INE CSV whitespace and remove trailing empty columns."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    empty_cols = [
        c for c in out.columns
        if c == "" or c.lower().startswith("unnamed")
    ]
    if empty_cols:
        out = out.drop(columns=empty_cols)
    return out


def _read_ine_flow_csv_bytes(raw: bytes) -> pd.DataFrame:
    """
    Read Flujos+15_O-D_M1_NOV.csv from bytes.

    The released file is semicolon-separated and commonly Latin-1 encoded.
    """
    last_error = None
    for encoding in ("latin1", "cp1252", "utf-8"):
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                sep=";",
                encoding=encoding,
                dtype=str,
                low_memory=False,
            )
            df = _clean_ine_columns(df)
            if {
                "CELDA_ORIGEN",
                "CELDA_DESTINO",
                "FLUJO",
            }.issubset(df.columns):
                return df
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_error = exc

    raise ValueError(
        "Could not parse the INE November-2019 OD CSV. "
        f"Last error: {last_error}"
    )


def load_madrid_ine_od(path: Path) -> pd.DataFrame:
    """
    Load the INE November-2019 mobility OD matrix.

    Supported inputs
    ----------------
    1. Direct Flujos+15_O-D_M1_NOV.csv
    2. Noviembre 2019.zip
    3. Outer datos_disponibles.zip containing Noviembre 2019.zip

    Returns canonical columns:
        origin_area, destination_area, flow
    """
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"[load] Madrid INE OD: {path}")

    raw_df = None

    if path.suffix.lower() == ".csv":
        raw = path.read_bytes()
        raw_df = _read_ine_flow_csv_bytes(raw)

    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as outer:
            members = [
                n for n in outer.namelist()
                if "__MACOSX" not in n
            ]

            direct_csv = next(
                (
                    n for n in members
                    if n.lower().endswith(".csv")
                    and "flujos+15_o-d_m1_nov" in n.lower()
                ),
                None,
            )

            if direct_csv is not None:
                raw_df = _read_ine_flow_csv_bytes(
                    outer.read(direct_csv)
                )
            else:
                nested_name = next(
                    (
                        n for n in members
                        if n.lower().endswith(".zip")
                        and "noviembre 2019" in n.lower()
                    ),
                    None,
                )

                if nested_name is None:
                    raise ValueError(
                        "ZIP does not contain Flujos+15_O-D_M1_NOV.csv "
                        "or a nested 'Noviembre 2019.zip'."
                    )

                nested_bytes = outer.read(nested_name)
                with zipfile.ZipFile(
                    io.BytesIO(nested_bytes)
                ) as nested:
                    nested_members = [
                        n for n in nested.namelist()
                        if "__MACOSX" not in n
                    ]
                    flow_name = next(
                        (
                            n for n in nested_members
                            if n.lower().endswith(".csv")
                            and "flujos+15_o-d_m1_nov" in n.lower()
                        ),
                        None,
                    )

                    if flow_name is None:
                        raise ValueError(
                            "Nested November ZIP does not contain "
                            "Flujos+15_O-D_M1_NOV.csv."
                        )

                    raw_df = _read_ine_flow_csv_bytes(
                        nested.read(flow_name)
                    )
    else:
        raise ValueError(
            "Madrid observed OD must be a .csv or .zip file."
        )

    out = raw_df[
        ["CELDA_ORIGEN", "CELDA_DESTINO", "FLUJO"]
    ].copy()

    out = out.rename(
        columns={
            "CELDA_ORIGEN": "origin_area",
            "CELDA_DESTINO": "destination_area",
            "FLUJO": "flow",
        }
    )

    out["origin_area"] = (
        out["origin_area"].astype(str).str.strip()
    )
    out["destination_area"] = (
        out["destination_area"].astype(str).str.strip()
    )

    # Decimal commas are defensive; FLUJO is normally integer.
    out["flow"] = pd.to_numeric(
        out["flow"]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )

    out = out.dropna(
        subset=["origin_area", "destination_area", "flow"]
    )
    out = out[out["flow"] > 0].copy()

    # Consolidate duplicate OD records if present.
    out = (
        out.groupby(
            ["origin_area", "destination_area"],
            as_index=False,
        )["flow"]
        .sum()
    )

    return out


def load_madrid_mobility_areas(path: Path) -> gpd.GeoDataFrame:
    """
    Load INE mobility-area polygons.

    Supported:
        - direct celdas_marzo_2020.shp
        - shapefiles_celdas_marzo2020.zip

    Returns:
        area_id, area_name, geometry
    """
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"[load] Madrid INE mobility areas: {path}")

    if path.suffix.lower() == ".shp":
        areas = gpd.read_file(path)

    elif path.suffix.lower() == ".zip":
        # pyogrio/GDAL zip handling differs by environment. Extracting to a
        # temporary directory is more reproducible and avoids __MACOSX files.
        with tempfile.TemporaryDirectory(
            prefix="madrid_ine_areas_"
        ) as tmp:
            tmp_path = Path(tmp)
            with zipfile.ZipFile(path) as zf:
                zf.extractall(tmp_path)

            shp_candidates = [
                p for p in tmp_path.rglob("*.shp")
                if "__MACOSX" not in str(p)
                and not p.name.startswith("._")
            ]

            preferred = [
                p for p in shp_candidates
                if "celdas_marzo_2020" in p.name.lower()
            ]

            candidates = preferred or shp_candidates

            if not candidates:
                raise ValueError(
                    "No valid .shp found inside Madrid mobility-area ZIP."
                )

            areas = gpd.read_file(candidates[0])
    else:
        raise ValueError(
            "Madrid mobility areas must be a .shp or .zip file."
        )

    required = {"ID_GRUPO", "geometry"}
    missing = required - set(areas.columns)
    if missing:
        raise ValueError(
            f"Madrid mobility-area file missing {sorted(missing)}. "
            f"Found: {list(areas.columns)}"
        )

    if areas.crs is None:
        raise ValueError(
            "Madrid mobility-area file has no CRS."
        )

    out = areas.copy()
    out["area_id"] = (
        out["ID_GRUPO"].astype(str).str.strip()
    )

    if "NOMBRE_CEL" in out.columns:
        out["area_name"] = out["NOMBRE_CEL"].astype(str)
    else:
        out["area_name"] = ""

    out = out[
        ["area_id", "area_name", "geometry"]
    ].copy()

    if out["area_id"].duplicated().any():
        # Dissolve duplicate geometries under the same official area ID.
        out = (
            out.dissolve(
                by="area_id",
                as_index=False,
                aggfunc={"area_name": "first"},
            )
            [["area_id", "area_name", "geometry"]]
        )

    return out


def clip_madrid_areas_to_boundary(
    areas: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict]:
    """
    Clip INE mobility areas to the exact Madrid simulation boundary.

    The clipped polygons define the source areas used by the H3 allocation
    matrix. Tiny/empty geometric intersections are removed.
    """
    areas_wgs = areas.to_crs(4326)
    boundary_wgs = boundary.to_crs(4326)
    city_geom = boundary_wgs.geometry.iloc[0]

    clipped = areas_wgs.copy()
    clipped["geometry"] = clipped.geometry.intersection(
        city_geom
    )
    clipped = clipped[
        (~clipped.geometry.is_empty)
        & clipped.geometry.notna()
    ].copy()

    # Quantify retained physical area in a metric CRS.
    original_metric = areas_wgs.to_crs(
        MADRID_METRIC_CRS
    )
    clipped_metric = clipped.to_crs(
        MADRID_METRIC_CRS
    )

    original_area_lookup = dict(
        zip(
            original_metric["area_id"],
            original_metric.geometry.area,
        )
    )

    clipped["clipped_area_m2"] = (
        clipped_metric.geometry.area.to_numpy()
    )
    clipped["original_area_m2"] = clipped[
        "area_id"
    ].map(original_area_lookup)

    clipped["boundary_area_fraction"] = (
        clipped["clipped_area_m2"]
        / clipped["original_area_m2"].clip(lower=EPS)
    )

    # Exclude numerical slivers.
    clipped = clipped[
        clipped["clipped_area_m2"] > 1.0
    ].copy()

    stats = {
        "source_area_count": int(len(areas)),
        "areas_intersecting_boundary": int(len(clipped)),
        "clipped_area_total_km2": float(
            clipped["clipped_area_m2"].sum() / 1e6
        ),
        "area_ids": sorted(
            clipped["area_id"].astype(str).tolist()
        ),
    }

    return clipped, stats


def _geometry_h3_polyfill(
    geom,
    resolution: int,
) -> set[str]:
    """
    Return candidate H3 cells for a Polygon/MultiPolygon geometry.

    h3.geo_to_cells follows the GeoJSON interface in h3-py v4. Boundary
    neighbors are added later so that fractional edge intersections are not
    lost.
    """
    cells: set[str] = set()

    if geom is None or geom.is_empty:
        return cells

    parts = (
        list(geom.geoms)
        if geom.geom_type == "MultiPolygon"
        else [geom]
    )

    for part in parts:
        if part.is_empty:
            continue

        if hasattr(h3, "geo_to_cells"):
            try:
                cells.update(
                    h3.geo_to_cells(
                        mapping(part),
                        resolution,
                    )
                )
                continue
            except Exception:
                pass

        # Robust fallback: representative-point seed. This is used only if
        # geo_to_cells is unavailable/fails; the boundary expansion below
        # still provides neighboring candidates.
        rp = part.representative_point()
        cells.add(
            latlng_to_h3(
                rp.y,
                rp.x,
                resolution,
            )
        )

    if not cells:
        rp = geom.representative_point()
        cells.add(
            latlng_to_h3(
                rp.y,
                rp.x,
                resolution,
            )
        )

    expanded = set(cells)
    for cell in list(cells):
        if hasattr(h3, "grid_disk"):
            expanded.update(h3.grid_disk(cell, 1))
        else:  # h3-py v3
            expanded.update(h3.k_ring(cell, 1))

    return expanded


def _h3_cell_polygon_wgs84(cell: str) -> Polygon:
    boundary = h3.cell_to_boundary(cell)
    # h3 returns (lat, lng); shapely expects (x=lng, y=lat).
    return Polygon(
        [(float(lng), float(lat)) for lat, lng in boundary]
    )


def build_madrid_area_h3_weights(
    clipped_areas: gpd.GeoDataFrame,
    resolution: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Build fractional area weights W(area -> H3).

    For each INE mobility area clipped to the simulation boundary:

        w_ih = Area(A_i ∩ H_h) / sum_h Area(A_i ∩ H_h)

    The final per-area weights are explicitly renormalized to sum to 1.
    """
    transformer = Transformer.from_crs(
        4326,
        MADRID_METRIC_CRS,
        always_xy=True,
    )

    cell_metric_cache: dict[str, object] = {}
    rows = []

    for idx, rec in clipped_areas.iterrows():
        area_id = str(rec["area_id"])
        geom_wgs = rec.geometry

        if geom_wgs is None or geom_wgs.is_empty:
            continue

        geom_metric = shapely_transform(
            transformer.transform,
            geom_wgs,
        )

        candidates = _geometry_h3_polyfill(
            geom_wgs,
            resolution,
        )

        local = []
        for cell in candidates:
            if cell not in cell_metric_cache:
                cell_wgs = _h3_cell_polygon_wgs84(cell)
                cell_metric_cache[cell] = shapely_transform(
                    transformer.transform,
                    cell_wgs,
                )

            inter = geom_metric.intersection(
                cell_metric_cache[cell]
            )
            area_m2 = float(inter.area)

            if area_m2 > 1e-6:
                local.append(
                    (cell, area_m2)
                )

        if not local:
            raise RuntimeError(
                f"No H3 intersections found for Madrid area "
                f"{area_id} at H{resolution}."
            )

        total = float(sum(a for _, a in local))
        if total <= EPS:
            raise RuntimeError(
                f"Zero allocated area for {area_id} at H{resolution}."
            )

        for cell, area_m2 in local:
            rows.append(
                {
                    "area_id": area_id,
                    "h3_id": str(cell),
                    "intersection_area_m2": area_m2,
                    "weight": area_m2 / total,
                }
            )

    weights = pd.DataFrame(rows)

    if weights.empty:
        raise RuntimeError(
            f"Empty Madrid area->H3 crosswalk at H{resolution}."
        )

    sums = weights.groupby("area_id")["weight"].sum()

    max_abs_error = float(
        np.max(np.abs(sums.to_numpy() - 1.0))
    )

    if max_abs_error > 1e-9:
        raise AssertionError(
            f"Madrid H{resolution} area-weight conservation failed: "
            f"max |sum(w)-1|={max_abs_error}"
        )

    stats = {
        "resolution": int(resolution),
        "area_count": int(weights["area_id"].nunique()),
        "h3_cell_count": int(weights["h3_id"].nunique()),
        "allocation_rows": int(len(weights)),
        "max_area_weight_sum_error": max_abs_error,
        "min_cells_per_area": int(
            weights.groupby("area_id").size().min()
        ),
        "max_cells_per_area": int(
            weights.groupby("area_id").size().max()
        ),
        "median_cells_per_area": float(
            weights.groupby("area_id").size().median()
        ),
    }

    return weights, stats


def project_madrid_od_to_h3(
    od: pd.DataFrame,
    weights: pd.DataFrame,
    resolution: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Project INE area OD to H3 using sparse matrix multiplication:

        F_H3 = W^T F_AREA W

    where rows of W sum exactly to 1.

    Only OD pairs whose origin and destination mobility areas intersect the
    simulation boundary are retained (domain-rule='both').
    """
    area_ids = sorted(
        weights["area_id"].astype(str).unique()
    )
    area_index = {
        area_id: i
        for i, area_id in enumerate(area_ids)
    }

    cells = sorted(
        weights["h3_id"].astype(str).unique()
    )
    cell_index = {
        cell: i
        for i, cell in enumerate(cells)
    }

    x = od[
        od["origin_area"].isin(area_index)
        & od["destination_area"].isin(area_index)
    ].copy()

    if x.empty:
        raise RuntimeError(
            "No Madrid INE OD pairs remain after boundary-area filtering."
        )

    f_rows = x["origin_area"].map(area_index).to_numpy()
    f_cols = x["destination_area"].map(area_index).to_numpy()
    f_data = x["flow"].to_numpy(dtype=float)

    F = sparse.coo_matrix(
        (f_data, (f_rows, f_cols)),
        shape=(len(area_ids), len(area_ids)),
        dtype=float,
    ).tocsr()
    F.sum_duplicates()

    w_rows = (
        weights["area_id"].map(area_index).to_numpy()
    )
    w_cols = (
        weights["h3_id"].map(cell_index).to_numpy()
    )
    w_data = weights["weight"].to_numpy(dtype=float)

    W = sparse.coo_matrix(
        (w_data, (w_rows, w_cols)),
        shape=(len(area_ids), len(cells)),
        dtype=float,
    ).tocsr()
    W.sum_duplicates()

    H = (W.T @ F @ W).tocoo()
    H.sum_duplicates()

    mask = H.data > EPS

    out = pd.DataFrame(
        {
            "origin_h3": np.asarray(cells, dtype=object)[
                H.row[mask]
            ],
            "destination_h3": np.asarray(cells, dtype=object)[
                H.col[mask]
            ],
            "flow": H.data[mask],
        }
    )

    out = out.sort_values(
        "flow",
        ascending=False,
    ).reset_index(drop=True)

    input_flow = float(F.sum())
    output_flow = float(out["flow"].sum())

    if not np.isclose(
        input_flow,
        output_flow,
        rtol=1e-9,
        atol=max(1e-6, 1e-10 * input_flow),
    ):
        raise AssertionError(
            f"Madrid H{resolution} OD mass conservation failed: "
            f"input={input_flow}, output={output_flow}"
        )

    stats = {
        "resolution": int(resolution),
        "retained_mobility_areas": int(len(area_ids)),
        "retained_area_od_pairs": int(F.nnz),
        "observed_flow_before_boundary_filter": float(
            od["flow"].sum()
        ),
        "observed_flow_internal_boundary": input_flow,
        "projected_h3_flow": output_flow,
        "projected_h3_od_pairs": int(len(out)),
        "projected_h3_cells": int(len(cells)),
    }

    return out, stats


def print_madrid_projection_report(
    area_stats: dict,
    projection_stats_by_r: dict[int, dict],
) -> None:
    print("\n" + "=" * 96)
    print("MADRID INE AREA-WEIGHTED PROJECTION")
    print("=" * 96)
    print(
        f"INE mobility areas in source       : "
        f"{area_stats['source_area_count']:,}"
    )
    print(
        f"Areas intersecting Madrid boundary : "
        f"{area_stats['areas_intersecting_boundary']:,}"
    )
    print(
        f"Clipped area represented           : "
        f"{area_stats['clipped_area_total_km2']:,.2f} km²"
    )

    for r, stats in projection_stats_by_r.items():
        print(
            f"H{r}: area OD={stats['retained_area_od_pairs']:,} | "
            f"H3 OD={stats['projected_h3_od_pairs']:,} | "
            f"flow={stats['projected_h3_flow']:,.0f}"
        )


# =============================================================================
# SIMULATED OD
# =============================================================================

def _detect_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not detect {label} column. Tried {candidates}. "
        f"Available: {list(df.columns)}"
    )


def load_simulated_od(
    path: Path,
    resolution: int,
    backend: str,
    purpose: str | None = "work",
) -> pd.DataFrame:
    """
    Load simulated OD in one of two common forms:

    A) already aggregated:
       origin_h3, destination_h3, flow

    B) trip-level:
       origin_lat/origin_lon + destination_lat/destination_lon
       (or common aliases), optionally purpose/activity/trip_purpose.

    If H3 columns already exist with resolution-specific names, they are used.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"[load] {backend} simulated OD: {path}")
    df = pd.read_csv(path, low_memory=False)

    # Optional commuting filter.
    purpose_cols = [
        c for c in ["purpose", "trip_purpose", "activity", "activity_type"]
        if c in df.columns
    ]
    if purpose and purpose_cols:
        pc = purpose_cols[0]
        before = len(df)
        df = df[
            df[pc].astype(str).str.lower().str.strip() == purpose.lower().strip()
        ].copy()
        print(
            f"[filter] {backend}: {pc}={purpose!r}: "
            f"{before:,} -> {len(df):,} rows"
        )

    # Already aggregated H3 OD.
    if {"origin_h3", "destination_h3"}.issubset(df.columns):
        flow_col = "flow" if "flow" in df.columns else None
        if flow_col is None:
            df["flow"] = 1.0
            flow_col = "flow"

        out = df[["origin_h3", "destination_h3", flow_col]].copy()
        out = out.rename(columns={flow_col: "flow"})
        return (
            out.groupby(["origin_h3", "destination_h3"], as_index=False)["flow"]
            .sum()
        )

    # Resolution-specific H3 columns.
    h3_pairs = [
        (f"origin_h3_{resolution}", f"destination_h3_{resolution}"),
        (f"origin_h{resolution}", f"destination_h{resolution}"),
        (f"o_h3_{resolution}", f"d_h3_{resolution}"),
    ]
    for oc, dc in h3_pairs:
        if oc in df.columns and dc in df.columns:
            flow_col = "flow" if "flow" in df.columns else None
            if flow_col is None:
                df["flow"] = 1.0
                flow_col = "flow"

            out = df[[oc, dc, flow_col]].copy()
            out.columns = ["origin_h3", "destination_h3", "flow"]
            return (
                out.groupby(["origin_h3", "destination_h3"], as_index=False)["flow"]
                .sum()
            )

    # Trip-level coordinates.
    olat = _detect_column(
        df,
        ["origin_lat", "o_lat", "home_lat", "start_lat", "lat_origin"],
        "origin latitude",
    )
    olon = _detect_column(
        df,
        ["origin_lon", "origin_lng", "o_lon", "o_lng",
         "home_lon", "home_lng", "start_lon", "start_lng", "lon_origin"],
        "origin longitude",
    )
    dlat = _detect_column(
        df,
        ["destination_lat", "d_lat", "work_lat", "end_lat", "lat_destination"],
        "destination latitude",
    )
    dlon = _detect_column(
        df,
        ["destination_lon", "destination_lng", "d_lon", "d_lng",
         "work_lon", "work_lng", "end_lon", "end_lng", "lon_destination"],
        "destination longitude",
    )

    x = df[[olat, olon, dlat, dlon]].dropna().copy()

    x["origin_h3"] = [
        latlng_to_h3(a, b, resolution)
        for a, b in zip(x[olat], x[olon])
    ]
    x["destination_h3"] = [
        latlng_to_h3(a, b, resolution)
        for a, b in zip(x[dlat], x[dlon])
    ]
    x["flow"] = 1.0

    return (
        x.groupby(["origin_h3", "destination_h3"], as_index=False)["flow"]
        .sum()
    )



def load_paired_simulated_od(
    directory: Path,
    resolution: int,
    kind: str = "work",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load paired work-only OD diagnostics exported by UrbanFidelity.

    Expected file:
        work_od_flows_H{resolution}.csv

    Required columns:
        origin_h3
        destination_h3
        continuous_count
        h3_count

    Returns
    -------
    continuous_od, h3_od
        Two canonical DataFrames with:
            origin_h3, destination_h3, flow
    """
    filename = (
        f"work_od_flows_H{resolution}.csv"
        if kind == "work"
        else f"od_flows_H{resolution}.csv"
    )
    path = directory / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Paired simulated OD file not found: {path}"
        )

    print(
        f"[load] paired simulated {kind} OD H{resolution}: {path}"
    )

    df = pd.read_csv(path, low_memory=False)

    required = {
        "origin_h3",
        "destination_h3",
        "continuous_count",
        "h3_count",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )

    keys = ["origin_h3", "destination_h3"]

    continuous = df[
        keys + ["continuous_count"]
    ].copy()
    continuous = continuous.rename(
        columns={"continuous_count": "flow"}
    )

    h3_backend = df[
        keys + ["h3_count"]
    ].copy()
    h3_backend = h3_backend.rename(
        columns={"h3_count": "flow"}
    )

    for x in (continuous, h3_backend):
        x["origin_h3"] = x["origin_h3"].astype(str)
        x["destination_h3"] = x["destination_h3"].astype(str)
        x["flow"] = pd.to_numeric(
            x["flow"],
            errors="coerce",
        ).fillna(0.0)

    continuous = (
        continuous.groupby(
            keys,
            as_index=False,
        )["flow"]
        .sum()
    )

    h3_backend = (
        h3_backend.groupby(
            keys,
            as_index=False,
        )["flow"]
        .sum()
    )

    return continuous, h3_backend


def paired_internal_metrics(
    continuous: pd.DataFrame,
    h3_backend: pd.DataFrame,
) -> dict:
    """
    Compute Continuous <-> H3 agreement on their own common OD support.

    This is retained as an internal architectural control alongside
    the two observed-reference comparisons.
    """
    keys = ["origin_h3", "destination_h3"]

    c = continuous.rename(
        columns={"flow": "continuous_flow"}
    )
    h = h3_backend.rename(
        columns={"flow": "h3_flow"}
    )

    aligned = c.merge(
        h,
        on=keys,
        how="outer",
    ).fillna(0.0)

    x = aligned["continuous_flow"].to_numpy(dtype=float)
    y = aligned["h3_flow"].to_numpy(dtype=float)

    return {
        "continuous_h3_pearson": safe_pearson(x, y),
        "continuous_h3_spearman": safe_spearman(
            pd.Series(x),
            pd.Series(y),
        ),
        "continuous_h3_nmae": nmae(y, x),
        "continuous_h3_jsd": js_divergence(y, x),
        "continuous_h3_cosine": cosine_similarity(x, y),
    }


# =============================================================================
# THREE-WAY COMPARISON
# =============================================================================

def align_three_way(
    observed: pd.DataFrame,
    continuous: pd.DataFrame,
    h3_backend: pd.DataFrame,
) -> pd.DataFrame:
    """
    Outer-align all OD pairs so missing pairs are represented by zero flow.
    """
    obs = observed.rename(columns={"flow": "observed_flow"})
    con = continuous.rename(columns={"flow": "continuous_flow"})
    hx = h3_backend.rename(columns={"flow": "h3_flow"})

    keys = ["origin_h3", "destination_h3"]

    out = obs.merge(con, on=keys, how="outer")
    out = out.merge(hx, on=keys, how="outer")

    for c in ["observed_flow", "continuous_flow", "h3_flow"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out


def metric_bundle(
    aligned: pd.DataFrame,
    prediction_col: str,
) -> dict:
    obs = aligned["observed_flow"].to_numpy(dtype=float)
    pred = aligned[prediction_col].to_numpy(dtype=float)

    support_obs = obs > 0
    support_pred = pred > 0
    union = support_obs | support_pred
    intersection = support_obs & support_pred

    dist = distribution_metrics(
        obs,
        pred,
        label=prediction_col,
    )

    result = {
        "pearson": safe_pearson(obs, pred),
        "spearman": safe_spearman(
            pd.Series(obs), pd.Series(pred)
        ),

        # Absolute-flow error. This retains differences in total trip mass.
        "nmae": nmae(pred, obs),

        # JSD already normalizes internally, but we retain the legacy field
        # and also expose explicit normalized-distribution diagnostics below.
        "jsd": js_divergence(pred, obs),
        "cosine_similarity": cosine_similarity(obs, pred),

        "observed_total_flow": float(obs.sum()),
        "predicted_total_flow": float(pred.sum()),
        "total_flow_ratio": (
            float(pred.sum() / obs.sum()) if obs.sum() > EPS else np.nan
        ),

        "observed_support_pairs": int(support_obs.sum()),
        "predicted_support_pairs": int(support_pred.sum()),
        "support_jaccard": (
            float(intersection.sum() / union.sum()) if union.sum() else np.nan
        ),
    }

    result.update(dist)
    return result


def compare_resolution(
    observed: pd.DataFrame,
    continuous: pd.DataFrame,
    h3_backend: pd.DataFrame,
    resolution: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = align_three_way(observed, continuous, h3_backend)

    mc = metric_bundle(aligned, "continuous_flow")
    mh = metric_bundle(aligned, "h3_flow")

    row = {
        "h3_resolution": resolution,

        "observed_od_pairs": int((aligned["observed_flow"] > 0).sum()),
        "observed_total_flow": float(aligned["observed_flow"].sum()),

        "continuous_pearson": mc["pearson"],
        "h3_pearson": mh["pearson"],
        "pearson_gap_h3_minus_continuous": mh["pearson"] - mc["pearson"],

        "continuous_spearman": mc["spearman"],
        "h3_spearman": mh["spearman"],
        "spearman_gap_h3_minus_continuous": mh["spearman"] - mc["spearman"],

        # Absolute-flow diagnostics: sensitive to total trip mass.
        "continuous_nmae": mc["nmae"],
        "h3_nmae": mh["nmae"],
        "nmae_absolute_gap": abs(mh["nmae"] - mc["nmae"]),
        "nmae_signed_gap_h3_minus_continuous": mh["nmae"] - mc["nmae"],
        "nmae_relative_gap_rho": (
            abs(mh["nmae"] - mc["nmae"]) / mc["nmae"]
            if np.isfinite(mc["nmae"]) and abs(mc["nmae"]) > EPS
            else np.nan
        ),

        "continuous_jsd": mc["jsd"],
        "h3_jsd": mh["jsd"],
        "jsd_absolute_gap": abs(mh["jsd"] - mc["jsd"]),
        "jsd_signed_gap_h3_minus_continuous": mh["jsd"] - mc["jsd"],
        "jsd_relative_gap_h3_minus_continuous": relative_gap(
            mh["jsd"], mc["jsd"]
        ),

        "continuous_cosine": mc["cosine_similarity"],
        "h3_cosine": mh["cosine_similarity"],
        "cosine_gap_h3_minus_continuous": (
            mh["cosine_similarity"] - mc["cosine_similarity"]
        ),

        # Unit-mass OD-share diagnostics: isolate spatial structure from
        # differences in observed/simulated total trip counts.
        "continuous_share_mae": mc["share_mae"],
        "h3_share_mae": mh["share_mae"],
        "share_mae_gap_h3_minus_continuous": (
            mh["share_mae"] - mc["share_mae"]
        ),

        "continuous_share_l1": mc["share_l1"],
        "h3_share_l1": mh["share_l1"],
        "share_l1_gap_h3_minus_continuous": (
            mh["share_l1"] - mc["share_l1"]
        ),

        "continuous_share_nmae": mc["share_nmae"],
        "h3_share_nmae": mh["share_nmae"],
        "share_nmae_gap_h3_minus_continuous": (
            mh["share_nmae"] - mc["share_nmae"]
        ),
        "share_nmae_relative_gap_h3_minus_continuous": relative_gap(
            mh["share_nmae"],
            mc["share_nmae"],
        ),

        "continuous_tvd": mc["total_variation_distance"],
        "h3_tvd": mh["total_variation_distance"],
        "tvd_gap_h3_minus_continuous": (
            mh["total_variation_distance"]
            - mc["total_variation_distance"]
        ),
        "tvd_relative_gap_h3_minus_continuous": relative_gap(
            mh["total_variation_distance"],
            mc["total_variation_distance"],
        ),

        "continuous_normalized_jsd": mc["jsd_normalized"],
        "h3_normalized_jsd": mh["jsd_normalized"],
        "normalized_jsd_gap_h3_minus_continuous": (
            mh["jsd_normalized"] - mc["jsd_normalized"]
        ),
        "normalized_jsd_relative_gap_h3_minus_continuous": relative_gap(
            mh["jsd_normalized"],
            mc["jsd_normalized"],
        ),

        "continuous_normalized_cosine": mc["cosine_normalized"],
        "h3_normalized_cosine": mh["cosine_normalized"],
        "normalized_cosine_gap_h3_minus_continuous": (
            mh["cosine_normalized"] - mc["cosine_normalized"]
        ),

        "continuous_total_flow_ratio": mc["total_flow_ratio"],
        "h3_total_flow_ratio": mh["total_flow_ratio"],

        "continuous_support_jaccard": mc["support_jaccard"],
        "h3_support_jaccard": mh["support_jaccard"],
    }

    return pd.DataFrame([row]), aligned


# =============================================================================
# DOMAIN HANDLING
# =============================================================================

def load_domain_cells(path: Path | None, resolution: int) -> set[str] | None:
    """
    Optional simulation-domain H3 list.

    Accepted CSV columns:
        h3
        h3_id
        h3_index
        cell
        h3_7 / h3_8 / h3_9
    """
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    candidates = [
        f"h3_{resolution}",
        "h3",
        "h3_id",
        "h3_index",
        "cell",
    ]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        raise ValueError(
            f"Could not find H3 domain column in {path}. "
            f"Available: {list(df.columns)}"
        )

    return set(df[col].dropna().astype(str))



def load_boundary_geometry(path: Path) -> gpd.GeoDataFrame:
    """
    Load the simulation-domain boundary and normalize it to EPSG:4326.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"[load] simulation boundary: {path}")
    boundary = gpd.read_file(path)

    if boundary.empty:
        raise ValueError(f"Boundary file is empty: {path}")
    if boundary.crs is None:
        raise ValueError(f"Boundary file has no CRS: {path}")

    boundary = boundary.to_crs(4326)
    boundary = boundary[["geometry"]].copy()

    # Dissolve all parts into a single city/domain geometry.
    geom = boundary.geometry.union_all()
    return gpd.GeoDataFrame({"geometry": [geom]}, crs=4326)


def filter_od_to_boundary(
    od_joined: pd.DataFrame,
    boundary: gpd.GeoDataFrame,
    rule: str = "both",
) -> tuple[pd.DataFrame, dict]:
    """
    Filter LODES OD rows by whether representative home/work block points
    fall inside the exact simulation boundary.

    rule:
        both   : home AND workplace inside boundary
        origin : home inside boundary
        dest   : workplace inside boundary
        either : home OR workplace inside boundary
        none   : no filtering
    """
    if rule == "none":
        return od_joined.copy(), {
            "domain_rule": "none",
            "rows_before": int(len(od_joined)),
            "rows_after": int(len(od_joined)),
            "flow_before": float(od_joined["S000"].sum()),
            "flow_after": float(od_joined["S000"].sum()),
        }

    geom = boundary.geometry.iloc[0]

    # Build point GeoSeries once for vectorized spatial predicates.
    home_points = gpd.GeoSeries(
        gpd.points_from_xy(
            od_joined["home_lon"],
            od_joined["home_lat"],
        ),
        crs=4326,
        index=od_joined.index,
    )

    work_points = gpd.GeoSeries(
        gpd.points_from_xy(
            od_joined["work_lon"],
            od_joined["work_lat"],
        ),
        crs=4326,
        index=od_joined.index,
    )

    # covers() includes points on the polygon boundary.
    home_inside = home_points.apply(geom.covers)
    work_inside = work_points.apply(geom.covers)

    if rule == "both":
        mask = home_inside & work_inside
    elif rule == "origin":
        mask = home_inside
    elif rule == "dest":
        mask = work_inside
    elif rule == "either":
        mask = home_inside | work_inside
    else:
        raise ValueError(f"Unknown boundary rule: {rule}")

    filtered = od_joined.loc[mask].copy()

    stats = {
        "domain_rule": rule,
        "rows_before": int(len(od_joined)),
        "rows_after": int(len(filtered)),
        "rows_retained_pct": float(100.0 * len(filtered) / len(od_joined))
        if len(od_joined) else np.nan,
        "flow_before": float(od_joined["S000"].sum()),
        "flow_after": float(filtered["S000"].sum()),
        "flow_retained_pct": float(
            100.0 * filtered["S000"].sum() / od_joined["S000"].sum()
        )
        if od_joined["S000"].sum() > 0
        else np.nan,
        "home_inside_rows": int(home_inside.sum()),
        "work_inside_rows": int(work_inside.sum()),
        "either_inside_rows": int((home_inside | work_inside).sum()),
        "both_inside_rows": int((home_inside & work_inside).sum()),
    }

    return filtered, stats


def print_domain_report(stats: dict) -> None:
    print("\n" + "=" * 96)
    print("SIMULATION-DOMAIN FILTER")
    print("=" * 96)
    print(f"Rule                           : {stats['domain_rule']}")
    print(f"OD rows before                 : {stats['rows_before']:,}")
    print(f"OD rows after                  : {stats['rows_after']:,}")
    if "rows_retained_pct" in stats:
        print(f"Rows retained                  : {stats['rows_retained_pct']:.3f}%")
    print(f"S000 flow before               : {stats['flow_before']:,.0f}")
    print(f"S000 flow after                : {stats['flow_after']:,.0f}")
    if "flow_retained_pct" in stats:
        print(f"Flow retained                  : {stats['flow_retained_pct']:.3f}%")
    if "home_inside_rows" in stats:
        print(f"Rows with home in domain       : {stats['home_inside_rows']:,}")
        print(f"Rows with workplace in domain  : {stats['work_inside_rows']:,}")
        print(f"Rows touching domain           : {stats['either_inside_rows']:,}")
        print(f"Rows fully internal            : {stats['both_inside_rows']:,}")


# =============================================================================
# SANITY REPORT
# =============================================================================

def print_join_report(stats: dict) -> None:
    print("\n" + "=" * 96)
    print("OBSERVED OD / CENSUS BLOCK JOIN")
    print("=" * 96)
    print(f"LODES OD rows                  : {stats['lodes_rows']:,}")
    print(f"Unique home blocks             : {stats['unique_home_blocks']:,}")
    print(f"Unique workplace blocks        : {stats['unique_work_blocks']:,}")
    print(f"Matched home GEOIDs            : {stats['home_geoid_match_pct']:.3f}%")
    print(f"Matched workplace GEOIDs       : {stats['work_geoid_match_pct']:.3f}%")
    print(f"Matched both endpoints         : {stats['both_geoid_match_pct']:.3f}%")
    print(f"Observed workers / S000        : {stats['observed_workers_total']:,.0f}")
    print(f"Workers with matched endpoints : {stats['observed_workers_matched']:,.0f}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-way external OD validation: observed vs Continuous vs H3."
    )

    parser.add_argument(
        "--city",
        default="cambridge_ma",
        choices=["cambridge_ma", "new_york_ny", "madrid"],
    )
    parser.add_argument(
        "--observed-od",
        type=Path,
        required=True,
        help=(
            "Observed OD source. Cambridge/NYC: LODES JT00 CSV/CSV.GZ. "
            "Madrid: Flujos+15_O-D_M1_NOV.csv, Noviembre 2019.zip, "
            "or outer datos_disponibles.zip."
        ),
    )
    parser.add_argument(
        "--tiger-blocks",
        type=Path,
        help="TIGER/Line tabblock20 .zip/.shp. Required for LODES.",
    )
    parser.add_argument(
        "--madrid-areas",
        type=Path,
        default=None,
        help=(
            "Madrid INE mobility-area geometry: celdas_marzo_2020.shp "
            "or shapefiles_celdas_marzo2020.zip."
        ),
    )
    parser.add_argument(
        "--paired-sim-od-kind",
        choices=["work", "all"],
        default="work",
        help=(
            "Which paired UrbanFidelity OD export to compare. "
            "'work' uses work_od_flows_H*.csv (default); "
            "'all' uses od_flows_H*.csv. For LODES use 'work'. "
            "Madrid INE M1 is not strictly work-only, so the choice must "
            "be reported explicitly."
        ),
    )
    parser.add_argument(
        "--continuous-od",
        type=Path,
        help="Continuous-backend commuting OD CSV.",
    )
    parser.add_argument(
        "--h3-od",
        type=Path,
        help="H3-backend commuting OD CSV.",
    )
    parser.add_argument(
        "--paired-sim-od-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing paired work_od_flows_H7/H8/H9.csv "
            "exported by experiments/03_urban_fidelity.py. "
            "When supplied, this is preferred over --continuous-od/--h3-od."
        ),
    )
    parser.add_argument(
        "--domain-cells",
        type=Path,
        default=None,
        help="Optional CSV defining simulation-domain H3 cells.",
    )
    parser.add_argument(
        "--domain-boundary",
        type=Path,
        default=None,
        help=(
            "Optional GeoJSON/Shapefile defining the exact simulation domain. "
            "For LODES, representative home/work Census Block points are "
            "filtered against this boundary before H3 aggregation."
        ),
    )
    parser.add_argument(
        "--domain-rule",
        choices=["both", "origin", "dest", "either", "none"],
        default="both",
        help=(
            "Observed-flow inclusion rule relative to the simulation domain. "
            "Default 'both' retains only internal OD pairs, which is the "
            "primary like-for-like comparison with a city-bounded simulation."
        ),
    )
    parser.add_argument(
        "--purpose",
        default="work",
        help="Simulated trip purpose to compare against commuting OD.",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/external_od_validation"),
    )
    parser.add_argument(
        "--observed-only",
        action="store_true",
        help=(
            "Build/validate observed H7/H8/H9 OD only. "
            "Does not require simulated OD files."
        ),
    )

    args = parser.parse_args()
    resolutions = parse_resolutions(args.resolutions)
    outdir = ensure_dir(args.output_dir / args.city)

    if args.city in {"cambridge_ma", "new_york_ny"} and args.tiger_blocks is None:
        parser.error("--tiger-blocks is required for LODES validation.")

    if args.city == "madrid":
        if args.madrid_areas is None:
            parser.error("--madrid-areas is required for Madrid validation.")
        if args.domain_boundary is None:
            parser.error(
                "--domain-boundary is required for Madrid area-weighted validation."
            )
        if args.domain_rule != "both":
            parser.error(
                "Madrid area-weighted adapter currently freezes the primary "
                "protocol to --domain-rule both."
            )

    # -------------------------------------------------------------------------
    # 1. OBSERVED OD
    # -------------------------------------------------------------------------
    if args.city in {"cambridge_ma", "new_york_ny"}:
        blocks = load_tiger_blocks(args.tiger_blocks)
        crosswalk = build_block_h3_crosswalk(blocks, resolutions)
        crosswalk.to_csv(outdir / "census_block_to_h3_crosswalk.csv", index=False)

        observed_raw = load_lodes_jt00(args.observed_od)
        observed_joined, join_stats = attach_block_coordinates(
            observed_raw, crosswalk
        )
        print_join_report(join_stats)

        (outdir / "join_quality.json").write_text(
            json.dumps(join_stats, indent=2),
            encoding="utf-8",
        )

        # Optional exact simulation-domain filter.
        if args.domain_boundary is not None:
            boundary = load_boundary_geometry(args.domain_boundary)
            observed_joined, domain_stats = filter_od_to_boundary(
                observed_joined,
                boundary,
                rule=args.domain_rule,
            )
            print_domain_report(domain_stats)
            (outdir / "domain_filter_quality.json").write_text(
                json.dumps(domain_stats, indent=2),
                encoding="utf-8",
            )
        else:
            boundary = None
            domain_stats = None
    else:
        # Madrid / INE November-2019 mobile-phone mobility.
        observed_raw = load_madrid_ine_od(
            args.observed_od
        )
        madrid_areas = load_madrid_mobility_areas(
            args.madrid_areas
        )
        boundary = load_boundary_geometry(
            args.domain_boundary
        )
        madrid_areas_clipped, madrid_area_stats = (
            clip_madrid_areas_to_boundary(
                madrid_areas,
                boundary,
            )
        )

        (outdir / "madrid_area_boundary_quality.json").write_text(
            json.dumps(
                madrid_area_stats,
                indent=2,
            ),
            encoding="utf-8",
        )

    # -------------------------------------------------------------------------
    # 2. AGGREGATE OBSERVED OD TO H7/H8/H9
    # -------------------------------------------------------------------------
    observed_by_r: dict[int, pd.DataFrame] = {}

    print("\n" + "=" * 96)
    print("OBSERVED OD AGGREGATION")
    print("=" * 96)

    madrid_projection_stats_by_r: dict[int, dict] = {}

    for r in resolutions:
        if args.city == "madrid":
            weights_r, weight_stats = (
                build_madrid_area_h3_weights(
                    madrid_areas_clipped,
                    resolution=r,
                )
            )

            weights_path = (
                outdir
                / f"madrid_area_to_h3_weights_H{r}.csv"
            )
            weights_r.to_csv(
                weights_path,
                index=False,
            )

            obs_r, projection_stats = (
                project_madrid_od_to_h3(
                    observed_raw,
                    weights_r,
                    resolution=r,
                )
            )

            projection_stats.update(
                {
                    "weight_area_count": weight_stats["area_count"],
                    "weight_h3_cell_count": weight_stats["h3_cell_count"],
                    "weight_allocation_rows": weight_stats["allocation_rows"],
                    "weight_max_sum_error": weight_stats[
                        "max_area_weight_sum_error"
                    ],
                    "weight_min_cells_per_area": weight_stats[
                        "min_cells_per_area"
                    ],
                    "weight_max_cells_per_area": weight_stats[
                        "max_cells_per_area"
                    ],
                    "weight_median_cells_per_area": weight_stats[
                        "median_cells_per_area"
                    ],
                }
            )
            madrid_projection_stats_by_r[r] = projection_stats

            (
                outdir
                / f"madrid_projection_quality_H{r}.json"
            ).write_text(
                json.dumps(
                    projection_stats,
                    indent=2,
                ),
                encoding="utf-8",
            )

        else:
            domain_cells = load_domain_cells(
                args.domain_cells,
                r,
            )

            obs_r = aggregate_observed_h3(
                observed_joined,
                resolution=r,
                domain_cells=domain_cells,
                domain_rule=(
                    args.domain_rule
                    if domain_cells
                    else "none"
                ),
            )

        observed_by_r[r] = obs_r

        path = outdir / f"observed_od_h{r}.csv"
        obs_r.to_csv(path, index=False)

        print(
            f"H{r}: {len(obs_r):,} OD pairs | "
            f"flow={obs_r['flow'].sum():,.0f} | saved={path}"
        )

    if args.city == "madrid":
        print_madrid_projection_report(
            madrid_area_stats,
            madrid_projection_stats_by_r,
        )

    if args.observed_only:
        print("\n[done] Observed-only phase completed.")
        if args.domain_boundary is None:
            print(
                "WARNING: no --domain-boundary was supplied; observed OD may "
                "cover the full source geography rather than the simulation domain."
            )
        print("Next: inspect domain-filtered OD, then provide Continuous and H3 commuting OD exports.")
        return

    # -------------------------------------------------------------------------
    # 3. THREE-WAY EXTERNAL VALIDATION
    # -------------------------------------------------------------------------
    if args.paired_sim_od_dir is None:
        if args.continuous_od is None or args.h3_od is None:
            parser.error(
                "Provide either --paired-sim-od-dir or both "
                "--continuous-od and --h3-od unless --observed-only is used."
            )

    result_rows = []

    print("\n" + "=" * 96)
    print("THREE-WAY EXTERNAL OD VALIDATION")
    print("=" * 96)

    for r in resolutions:
        if args.paired_sim_od_dir is not None:
            continuous_r, h3_r = load_paired_simulated_od(
                args.paired_sim_od_dir,
                resolution=r,
                kind=args.paired_sim_od_kind,
            )
        else:
            continuous_r = load_simulated_od(
                args.continuous_od,
                resolution=r,
                backend="Continuous",
                purpose=args.purpose,
            )
            h3_r = load_simulated_od(
                args.h3_od,
                resolution=r,
                backend="H3",
                purpose=args.purpose,
            )

        summary_r, aligned_r = compare_resolution(
            observed_by_r[r],
            continuous_r,
            h3_r,
            resolution=r,
        )

        internal = paired_internal_metrics(
            continuous_r,
            h3_r,
        )

        for key, value in internal.items():
            summary_r[key] = value

        summary_r["city"] = args.city
        summary_r["domain_rule"] = args.domain_rule
        summary_r["paired_sim_od_kind"] = args.paired_sim_od_kind

        if args.city == "madrid":
            summary_r["observed_source"] = (
                "INE_M1_mobile_mobility_Nov2019"
            )
            summary_r["comparison_scope"] = (
                "internal_Madrid_daily_destination_vs_"
                f"{args.paired_sim_od_kind}_simulation"
            )
        else:
            summary_r["observed_source"] = "LODES_JT00_2023"
            summary_r["comparison_scope"] = (
                "internal_home_to_work_commuting"
            )

        # Preserve transparent counts for interpretation.
        summary_r["continuous_total_simulated_flow"] = float(
            continuous_r["flow"].sum()
        )
        summary_r["h3_total_simulated_flow"] = float(
            h3_r["flow"].sum()
        )
        summary_r["continuous_od_pairs"] = int(len(continuous_r))
        summary_r["h3_od_pairs"] = int(len(h3_r))

        result_rows.append(summary_r)

        # Persist both raw flow and unit-mass OD shares for complete
        # auditability of absolute-volume vs spatial-structure comparisons.
        for raw_col, share_col in [
            ("observed_flow", "observed_share"),
            ("continuous_flow", "continuous_share"),
            ("h3_flow", "h3_share"),
        ]:
            values = aligned_r[raw_col].to_numpy(dtype=float)
            aligned_r[share_col] = normalize_distribution(
                values,
                label=f"H{r} {raw_col}",
            )

        # Conservation checks.
        for share_col in [
            "observed_share",
            "continuous_share",
            "h3_share",
        ]:
            if not np.isclose(
                float(aligned_r[share_col].sum()),
                1.0,
                rtol=0.0,
                atol=1e-10,
            ):
                raise AssertionError(
                    f"H{r}: {share_col} does not sum to 1."
                )

        aligned_r.to_csv(
            outdir / f"three_way_od_h{r}.csv",
            index=False,
        )

    summary = pd.concat(result_rows, ignore_index=True)
    summary_path = outdir / "external_od_validation_summary.csv"
    summary.to_csv(summary_path, index=False)

    paper_cols = [
        "h3_resolution",

        # Empirical agreement on raw OD flows.
        "continuous_pearson",
        "h3_pearson",
        "pearson_gap_h3_minus_continuous",

        # Absolute-flow error, retained for transparency.
        "continuous_nmae",
        "h3_nmae",
        "nmae_relative_gap_rho",

        # Unit-mass OD structure: preferred external non-degradation metrics.
        "continuous_tvd",
        "h3_tvd",
        "tvd_gap_h3_minus_continuous",
        "tvd_relative_gap_h3_minus_continuous",

        "continuous_normalized_jsd",
        "h3_normalized_jsd",
        "normalized_jsd_gap_h3_minus_continuous",

        "continuous_normalized_cosine",
        "h3_normalized_cosine",
        "normalized_cosine_gap_h3_minus_continuous",

        # Internal architectural agreement control.
        "continuous_h3_pearson",
    ]

    paper_table = summary[paper_cols].copy()
    paper_table_path = outdir / "TABLE_external_od_validation.csv"
    paper_table.to_csv(
        paper_table_path,
        index=False,
    )

    structure_cols = [
        "h3_resolution",
        "continuous_tvd",
        "h3_tvd",
        "tvd_gap_h3_minus_continuous",
        "tvd_relative_gap_h3_minus_continuous",
        "continuous_normalized_jsd",
        "h3_normalized_jsd",
        "normalized_jsd_gap_h3_minus_continuous",
        "continuous_normalized_cosine",
        "h3_normalized_cosine",
        "normalized_cosine_gap_h3_minus_continuous",
    ]
    structure_table = summary[structure_cols].copy()
    structure_table_path = (
        outdir / "TABLE_external_od_normalized_structure.csv"
    )
    structure_table.to_csv(
        structure_table_path,
        index=False,
    )

    cols = [
        "h3_resolution",

        "continuous_pearson",
        "h3_pearson",
        "pearson_gap_h3_minus_continuous",

        "continuous_spearman",
        "h3_spearman",

        "continuous_nmae",
        "h3_nmae",
        "nmae_relative_gap_rho",

        "continuous_tvd",
        "h3_tvd",
        "tvd_gap_h3_minus_continuous",
        "tvd_relative_gap_h3_minus_continuous",

        "continuous_normalized_jsd",
        "h3_normalized_jsd",
        "normalized_jsd_gap_h3_minus_continuous",

        "continuous_normalized_cosine",
        "h3_normalized_cosine",

        "continuous_h3_pearson",
        "continuous_h3_nmae",

        "observed_od_pairs",
        "continuous_od_pairs",
        "h3_od_pairs",

        "observed_total_flow",
        "continuous_total_simulated_flow",
        "h3_total_simulated_flow",
    ]

    print("\n" + summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 96)
    print("FILES SAVED")
    print("=" * 96)
    print(f"[saved] {summary_path.resolve()}")
    print(f"[saved] {paper_table_path.resolve()}")
    print(f"[saved] {structure_table_path.resolve()}")
    for r in resolutions:
        print(f"[saved] {(outdir / f'observed_od_h{r}.csv').resolve()}")
        print(f"[saved] {(outdir / f'three_way_od_h{r}.csv').resolve()}")

    print(
        "\nInterpretation: Continuous and H3 are compared against the SAME "
        "observed reference under an identical spatial protocol. Raw-flow "
        "metrics retain total-volume differences; normalized OD-share metrics "
        "isolate spatial structure. For error metrics (TVD/JSD/NMAE), an "
        "H3-minus-Continuous gap near zero supports empirical non-degradation "
        "by the H3 spatial representation. Negative gaps indicate slightly "
        "lower H3 error; positive gaps indicate additional H3 error. "
        "For Madrid, the INE M1 source is daytime/usual mobility rather than "
        "pure work commuting; this source-to-purpose mismatch must be reported "
        "explicitly. This experiment does not, by itself, establish absolute "
        "behavioral realism."
    )


if __name__ == "__main__":
    main()
