#!/usr/bin/env python3
"""
Build the New York City data layer for the H3 urban benchmark.

Outputs
-------
data/new_york_ny/
├── raw/worldpop/usa_ppp_2020_UNadj_constrained.tif
├── processed/new_york_boundary.geojson
├── processed/new_york_osm_pois.csv
├── processed/new_york_population_h3_r9.csv
├── processed/worldpop/new_york_worldpop_2020.tif
└── cache/new_york_walk_graph.graphml

Important
---------
This script builds city inputs only. It does NOT change the benchmark,
behavioral model, H3 fidelity resolutions, or routing assumptions.

Population methodology intentionally mirrors the Madrid WorldPop approach:
a 2020 constrained UN-adjusted WorldPop raster is clipped to Cambridge and
aggregated to H3 resolution 9 by raster-cell centers.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.new_york_ny import NEW_YORK


# WorldPop 2020 constrained, UN-adjusted population raster for the USA.
# Kept as a CLI-overridable constant so the exact source can be pinned in
# README / data provenance if WorldPop reorganizes its download hierarchy.
DEFAULT_WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/"
    "Global_2000_2020_Constrained/2020/BSGM/USA/"
    "usa_ppp_2020_UNadj_constrained.tif"
)


POI_TAGS = {
    "amenity": True,
    "shop": True,
    "leisure": True,
    "tourism": True,
    "office": True,
    "healthcare": True,
    "public_transport": True,
}


def ensure_dirs() -> None:
    for p in (
        NEW_YORK.raw_dir / "worldpop",
        NEW_YORK.processed_dir / "worldpop",
        NEW_YORK.cache_dir,
        NEW_YORK.outputs_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)


def build_boundary() -> gpd.GeoDataFrame:
    print("[1/5] Resolving New York City municipal boundary...")
    boundary = ox.geocode_to_gdf(NEW_YORK.place_query)

    if boundary.empty:
        raise RuntimeError("OSM/Nominatim returned no Cambridge boundary.")

    boundary = boundary.to_crs(4326)
    boundary = boundary[["geometry"]].copy()
    boundary["city_key"] = NEW_YORK.city_key
    boundary.to_file(NEW_YORK.boundary_path, driver="GeoJSON")

    print(f"      saved: {NEW_YORK.boundary_path}")
    return boundary


def buffered_polygon_wgs84(boundary: gpd.GeoDataFrame):
    local = boundary.to_crs(boundary.estimate_utm_crs())
    buffered = local.geometry.union_all().buffer(NEW_YORK.graph_buffer_m)
    return gpd.GeoSeries([buffered], crs=local.crs).to_crs(4326).iloc[0]


def build_walk_graph(boundary: gpd.GeoDataFrame, force: bool = False) -> None:
    path = NEW_YORK.graph_path
    if path.exists() and not force:
        print(f"[2/5] Walk graph exists, keeping: {path}")
        return

    print("[2/5] Downloading buffered New York City OSM walking graph...")
    polygon = buffered_polygon_wgs84(boundary)

    G = ox.graph_from_polygon(
        polygon,
        network_type="walk",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )

    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    ox.save_graphml(G, path)
    print(
        f"      nodes={len(G.nodes):,} edges={len(G.edges):,}\n"
        f"      saved: {path}"
    )


def _first_non_null(row, columns):
    for c in columns:
        if c in row.index:
            value = row[c]
            if pd.notna(value):
                return value
    return None


def _resolve_osm_identity_columns(features_reset: pd.DataFrame):
    """
    OSMnx version compatibility.

    Depending on OSMnx version, features_from_polygon() may reset its
    MultiIndex into:
        element / id
    rather than:
        element_type / osmid

    The previous New York City builder assumed the latter, causing most rows to
    receive (None, None) as OSM identity and then collapse during
    drop_duplicates() to roughly one row per subcategory.
    """
    type_candidates = (
        "element",
        "element_type",
        "type",
    )
    id_candidates = (
        "id",
        "osmid",
        "osm_id",
    )

    type_col = next(
        (c for c in type_candidates if c in features_reset.columns),
        None,
    )
    id_col = next(
        (c for c in id_candidates if c in features_reset.columns),
        None,
    )

    if type_col is None or id_col is None:
        raise ValueError(
            "Could not identify OSM feature identity columns after "
            f"reset_index(). First columns: {list(features_reset.columns[:15])}"
        )

    return type_col, id_col


def build_pois(
    boundary: gpd.GeoDataFrame,
    force: bool = False,
) -> None:
    path = NEW_YORK.poi_path

    if path.exists() and not force:
        print(f"[3/5] POI file exists, keeping: {path}")
        return

    print("[3/5] Downloading New York City OSM POIs...")
    polygon = boundary.geometry.union_all()

    features = ox.features_from_polygon(
        polygon,
        POI_TAGS,
    )

    if features.empty:
        raise RuntimeError(
            "No OSM POIs returned for Cambridge."
        )

    print(
        f"      raw OSM features returned: "
        f"{len(features):,}"
    )

    features = features.reset_index()

    type_col, id_col = _resolve_osm_identity_columns(
        features
    )

    print(
        f"      OSM identity columns: "
        f"{type_col!r}, {id_col!r}"
    )

    # Use representative points so points, lines and polygons share
    # one reproducible lat/lng schema.
    metric_crs = features.estimate_utm_crs()
    metric = features.to_crs(metric_crs)

    representative = (
        metric.geometry
        .representative_point()
    )

    representative = gpd.GeoSeries(
        representative,
        crs=metric_crs,
    ).to_crs(4326)

    rows = []

    for i, row in features.iterrows():
        # One OSM object can carry more than one requested semantic tag.
        # Preserve each tag independently instead of keeping only the first.
        for category in POI_TAGS:
            if category not in row.index:
                continue

            category_value = row[category]

            if pd.isna(category_value):
                continue

            # Some OSM values can be lists/sets after parsing.
            if isinstance(
                category_value,
                (list, tuple, set),
            ):
                subcategories = [
                    str(x)
                    for x in category_value
                    if pd.notna(x)
                ]
            else:
                subcategories = [
                    str(category_value)
                ]

            name = _first_non_null(
                row,
                [
                    "name",
                    "brand",
                    "operator",
                    "official_name",
                ],
            )

            for subcategory in subcategories:
                rows.append(
                    {
                        "osm_type": str(
                            row[type_col]
                        ),
                        "osm_id": str(
                            row[id_col]
                        ),
                        "name": name,
                        "category": category,
                        "subcategory": subcategory,
                        "lat": float(
                            representative.iloc[i].y
                        ),
                        "lng": float(
                            representative.iloc[i].x
                        ),
                    }
                )

    pois = pd.DataFrame(rows)

    if pois.empty:
        raise RuntimeError(
            "POI processing produced zero rows."
        )

    # Exact semantic duplicates only.
    # Different OSM objects with the same subcategory MUST remain distinct.
    before = len(pois)

    pois = pois.drop_duplicates(
        subset=[
            "osm_type",
            "osm_id",
            "category",
            "subcategory",
        ]
    ).reset_index(drop=True)

    pois.insert(
        0,
        "poi_id",
        np.arange(
            len(pois),
            dtype=np.int64,
        ),
    )

    pois.to_csv(
        path,
        index=False,
    )

    print(
        f"      semantic rows before dedupe: "
        f"{before:,}"
    )
    print(
        f"      POIs after exact dedupe      : "
        f"{len(pois):,}"
    )

    print("      category counts:")
    counts = (
        pois["category"]
        .value_counts()
    )

    for category, count in counts.items():
        print(
            f"        {category:18s}: "
            f"{count:,}"
        )

    print(f"      saved: {path}")


def reuse_existing_usa_worldpop_if_available() -> bool:
    """
    Avoid downloading the ~USA-wide WorldPop raster again if Cambridge has
    already downloaded the identical 2020 source.

    A symlink is created inside data/new_york_ny/raw/worldpop so the NYC
    pipeline remains self-contained at the path level while sharing the same
    physical source file.
    """
    target = NEW_YORK.worldpop_raw_path

    if target.exists():
        return True

    candidates = [
        ROOT
        / "data"
        / "cambridge_ma"
        / "raw"
        / "worldpop"
        / "usa_ppp_2020_UNadj_constrained.tif",
        ROOT
        / "data"
        / "raw"
        / "worldpop"
        / "usa_ppp_2020_UNadj_constrained.tif",
    ]

    for source in candidates:
        if source.exists():
            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            try:
                target.symlink_to(
                    source.resolve()
                )
                print(
                    "      reusing existing USA WorldPop raster via symlink:"
                )
                print(
                    f"      {target} -> {source.resolve()}"
                )
            except OSError:
                # Filesystems that disallow symlinks can still use a direct
                # copy; this is a fallback, not the preferred path.
                import shutil

                shutil.copy2(
                    source,
                    target,
                )
                print(
                    "      reused existing USA WorldPop raster via copy:"
                )
                print(
                    f"      {source} -> {target}"
                )

            return True

    return False


def download_file(url: str, path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        print(f"      WorldPop exists, keeping: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")

    print(f"      downloading WorldPop:\n      {url}")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    tmp.replace(path)
    print(f"      saved: {path}")


def clip_worldpop(boundary: gpd.GeoDataFrame, force: bool = False) -> None:
    src_path = NEW_YORK.worldpop_raw_path
    dst_path = NEW_YORK.worldpop_clipped_path

    if dst_path.exists() and not force:
        print(f"      clipped raster exists, keeping: {dst_path}")
        return

    with rasterio.open(src_path) as src:
        boundary_src = boundary.to_crs(src.crs)
        shapes = [mapping(boundary_src.geometry.union_all())]

        data, transform = mask(
            src,
            shapes,
            crop=True,
            filled=True,
            nodata=src.nodata,
        )

        meta = src.meta.copy()
        meta.update(
            {
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": transform,
            }
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **meta) as dst:
            dst.write(data)

    print(f"      saved clipped raster: {dst_path}")


def latlng_to_h3(lat: float, lng: float, resolution: int) -> str:
    # h3-py v4
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lng, resolution)

    # Compatibility with h3-py v3.
    return h3.geo_to_h3(lat, lng, resolution)


def h3_center(cell: str):
    if hasattr(h3, "cell_to_latlng"):
        return h3.cell_to_latlng(cell)
    return h3.h3_to_geo(cell)


def aggregate_worldpop_to_h3(force: bool = False) -> None:
    out_path = NEW_YORK.population_h3_path

    if out_path.exists() and not force:
        print(f"      population H3 exists, keeping: {out_path}")
        return

    print(
        "[5/5] Aggregating clipped WorldPop raster to "
        f"H3 r{NEW_YORK.population_h3_resolution}..."
    )

    totals: dict[str, float] = {}

    with rasterio.open(NEW_YORK.worldpop_clipped_path) as src:
        band = src.read(1, masked=True)

        rows, cols = np.where(~band.mask)
        values = np.asarray(band.data[rows, cols], dtype=float)

        positive = np.isfinite(values) & (values > 0.0)
        rows = rows[positive]
        cols = cols[positive]
        values = values[positive]

        xs, ys = rasterio.transform.xy(
            src.transform,
            rows,
            cols,
            offset="center",
        )

        if src.crs.to_epsg() != 4326:
            pts = gpd.GeoSeries(
                gpd.points_from_xy(xs, ys),
                crs=src.crs,
            ).to_crs(4326)
            lngs = pts.x.to_numpy()
            lats = pts.y.to_numpy()
        else:
            lngs = np.asarray(xs, dtype=float)
            lats = np.asarray(ys, dtype=float)

        for lat, lng, population in zip(lats, lngs, values):
            cell = latlng_to_h3(
                float(lat),
                float(lng),
                NEW_YORK.population_h3_resolution,
            )
            totals[cell] = totals.get(cell, 0.0) + float(population)

    records = []
    for cell, population in totals.items():
        lat, lng = h3_center(cell)
        records.append(
            {
                "h3": cell,
                "population": population,
                "lat": float(lat),
                "lng": float(lng),
            }
        )

    df = pd.DataFrame(records).sort_values(
        "population",
        ascending=False,
    ).reset_index(drop=True)

    df.to_csv(out_path, index=False)

    print(f"      H3 cells={len(df):,}")
    print(f"      population sum={df['population'].sum():,.1f}")
    print(f"      saved: {out_path}")


def validate_outputs() -> None:
    required = [
        NEW_YORK.boundary_path,
        NEW_YORK.graph_path,
        NEW_YORK.poi_path,
        NEW_YORK.worldpop_clipped_path,
        NEW_YORK.population_h3_path,
    ]

    print("\n" + "=" * 88)
    print("NEW_YORK DATA BUILD SUMMARY")
    print("=" * 88)

    failed = False
    for path in required:
        ok = path.exists() and path.stat().st_size > 0
        print(f"[{'OK' if ok else 'FAIL':4}] {path.relative_to(ROOT)}")
        failed |= not ok

    if failed:
        raise RuntimeError("New York City build incomplete.")

    pois = pd.read_csv(NEW_YORK.poi_path)
    pop = pd.read_csv(NEW_YORK.population_h3_path)

    print(f"\nPOIs              : {len(pois):,}")
    print(f"H3 population cells: {len(pop):,}")
    print(f"Population sum    : {pop['population'].sum():,.1f}")
    print(f"Kendall zoom      : {NEW_YORK.manhattan_zoom_bounds}")
    print(f"Midtown deep zoom : {NEW_YORK.midtown_deep_zoom_bounds}")
    print("\nNew York City Phase 1 data layer: READY")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worldpop-url",
        default=DEFAULT_WORLDPOP_URL,
        help="Exact WorldPop GeoTIFF URL to use.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild/redownload existing Cambridge assets.",
    )
    parser.add_argument(
        "--skip-worldpop-download",
        action="store_true",
        help=(
            "Do not download WorldPop. Useful if the raster was placed "
            "manually at the configured raw path."
        ),
    )
    parser.add_argument(
        "--rebuild-pois",
        action="store_true",
        help=(
            "Rebuild only the New York City OSM POI table. "
            "Existing graph and WorldPop assets are preserved."
        ),
    )
    args = parser.parse_args()

    ensure_dirs()

    print("=" * 88)
    print("NEW_YORK, MA — CROSS-CITY BENCHMARK DATA BUILD")
    print("=" * 88)
    print(f"City              : {NEW_YORK.place_query}")
    print(f"Population year   : {NEW_YORK.population_year}")
    print(
        "Fidelity scales   : "
        + ", ".join(f"H{r}" for r in NEW_YORK.fidelity_resolutions)
    )
    print("Routing           : shared physical OSM walking network")
    print()

    boundary = build_boundary()
    build_walk_graph(boundary, force=args.force)
    build_pois(boundary, force=(args.force or args.rebuild_pois))

    print("[4/5] Preparing New York City WorldPop raster...")
    if not args.force:
        reuse_existing_usa_worldpop_if_available()

    if not args.skip_worldpop_download:
        download_file(
            args.worldpop_url,
            NEW_YORK.worldpop_raw_path,
            force=args.force,
        )
    elif not NEW_YORK.worldpop_raw_path.exists():
        raise FileNotFoundError(
            "WorldPop download skipped but raster is missing:\n"
            f"{NEW_YORK.worldpop_raw_path}"
        )

    clip_worldpop(boundary, force=args.force)
    aggregate_worldpop_to_h3(force=args.force)
    validate_outputs()


if __name__ == "__main__":
    main()
