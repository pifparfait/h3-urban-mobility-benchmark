from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import pickle
import time

import h3
import numpy as np
import pandas as pd

from h3uab.behavior import destination_score
from h3uab.geometry import haversine_m
from h3uab.h3_utils import cells_covering_aoi, parent, radius_to_k
from h3uab.models import (
    AccessibilityState,
    AmenityAggregate,
    MacroState,
)


@dataclass
class StaticCityState:
    """
    Immutable urban state.

    Nothing at H3@11/H3@12 is precomputed here.
    """

    h3_config: object
    active_layer_codes: tuple[str, ...]

    # H3@5
    macro_h5: dict

    # H3@7 and H3@8
    amenities_h7: dict
    amenities_h8: dict

    # H3@9
    accessibility_h9: dict

    # H3@10
    poi_index_h10: dict

    # Metadata
    build_time_s: float
    n_pois: int

    def get_macro(self, h12_cell: str) -> MacroState | None:
        h5 = parent(
            h12_cell,
            self.h3_config.macro_res,
        )
        return self.macro_h5.get(h5)

    def get_amenity_aggregate(
        self,
        h12_cell: str,
        layer_code: str,
        resolution: int,
    ) -> AmenityAggregate | None:
        if resolution == self.h3_config.amenity_coarse_res:
            table = self.amenities_h7
        elif resolution == self.h3_config.amenity_functional_res:
            table = self.amenities_h8
        else:
            raise ValueError(
                "Amenity aggregates are defined only at the configured "
                "H3@7/H3@8 semantic resolutions."
            )

        cell = parent(
            h12_cell,
            resolution,
        )
        return table.get(
            (cell, layer_code)
        )

    def get_accessibility(
        self,
        h12_cell: str,
        layer_code: str,
    ) -> AccessibilityState | None:
        h9 = parent(
            h12_cell,
            self.h3_config.accessibility_res,
        )
        return self.accessibility_h9.get(
            (h9, layer_code)
        )

    def get_poi_bucket(
        self,
        h12_cell: str,
        layer_code: str,
    ) -> tuple[int, ...]:
        h10 = parent(
            h12_cell,
            self.h3_config.amenity_index_res,
        )
        return self.poi_index_h10.get(
            (h10, layer_code),
            (),
        )


def _aggregate_worldpop_to_h5(cfg):
    df = pd.read_csv(cfg.population_h3_csv)
    hcol = cfg.population_h3_id_col
    pcol = cfg.population_col

    sums = defaultdict(float)

    for cell, population in zip(
        df[hcol].astype(str),
        pd.to_numeric(df[pcol], errors="coerce").fillna(0.0),
    ):
        if population <= 0:
            continue

        h5 = parent(
            cell,
            cfg.h3.macro_res,
        )
        sums[h5] += float(population)

    return sums


def build_static_city(world, cfg) -> StaticCityState:
    """
    Build all STATIC information exactly once.

    Static layers:
      H3@5  macro population + POI count
      H3@7  coarse amenity semantic aggregates
      H3@8  functional amenity semantic aggregates
      H3@9  accessibility state
      H3@10 fine POI bucket index

    H3@11/H3@12 are intentionally absent.
    """
    t0 = time.perf_counter()
    hcfg = cfg.h3
    hcfg.validate()

    active_layers = tuple(world.active_layer_codes)

    # ------------------------------------------------------------------
    # Fine H3@10 POI index.
    # ------------------------------------------------------------------
    poi_index_mut = defaultdict(list)

    # H3@7/H3@8 semantic amenity aggregation.
    count7 = defaultdict(int)
    attr7 = defaultdict(float)
    count8 = defaultdict(int)
    attr8 = defaultdict(float)

    # Macro POI counts H3@5.
    poi_count_h5 = defaultdict(int)

    # H3@8 lookup used by H3@9 accessibility precompute.
    poi_ids_by_h8_layer = defaultdict(list)

    for poi_id, (
        lat,
        lng,
        layer_code,
        attractiveness,
    ) in enumerate(
        zip(
            world.amenity_lat,
            world.amenity_lng,
            world.amenity_layer_code,
            world.amenity_attractiveness,
        )
    ):
        h10 = h3.latlng_to_cell(
            float(lat),
            float(lng),
            hcfg.amenity_index_res,
        )
        h8 = h3.cell_to_parent(
            h10,
            hcfg.amenity_functional_res,
        )
        h7 = h3.cell_to_parent(
            h10,
            hcfg.amenity_coarse_res,
        )
        h5 = h3.cell_to_parent(
            h10,
            hcfg.macro_res,
        )

        key10 = (h10, str(layer_code))
        key8 = (h8, str(layer_code))
        key7 = (h7, str(layer_code))

        poi_index_mut[key10].append(poi_id)

        count7[key7] += 1
        attr7[key7] += float(attractiveness)

        count8[key8] += 1
        attr8[key8] += float(attractiveness)

        poi_count_h5[h5] += 1
        poi_ids_by_h8_layer[key8].append(poi_id)

    poi_index_h10 = {
        key: tuple(ids)
        for key, ids in poi_index_mut.items()
    }

    amenities_h7 = {
        key: AmenityAggregate(
            count=count7[key],
            attractiveness_sum=attr7[key],
        )
        for key in count7
    }

    amenities_h8 = {
        key: AmenityAggregate(
            count=count8[key],
            attractiveness_sum=attr8[key],
        )
        for key in count8
    }

    # ------------------------------------------------------------------
    # H3@5 macro city state.
    # ------------------------------------------------------------------
    population_h5 = _aggregate_worldpop_to_h5(cfg)

    macro_cells = set(population_h5) | set(poi_count_h5)

    macro_h5 = {
        cell: MacroState(
            population=float(population_h5.get(cell, 0.0)),
            poi_count=int(poi_count_h5.get(cell, 0)),
        )
        for cell in macro_cells
    }

    # ------------------------------------------------------------------
    # H3@9 accessibility state.
    # This is the only global precomputed spatial-decision layer.
    # ~2k cells in the current Madrid AOI, not H3@11/H3@12.
    # ------------------------------------------------------------------
    accessibility_h9 = {}

    h9_cells = cells_covering_aoi(
        cfg,
        hcfg.accessibility_res,
    )

    # Search candidate POIs using H3@8 buckets.
    k8 = radius_to_k(
        cfg.search_radius_m,
        hcfg.amenity_functional_res,
    )

    for h9 in h9_cells:
        origin_lat, origin_lng = h3.cell_to_latlng(h9)

        center_h8 = h3.latlng_to_cell(
            origin_lat,
            origin_lng,
            hcfg.amenity_functional_res,
        )

        neighborhood_h8 = h3.grid_disk(
            center_h8,
            k8,
        )

        for layer_code in active_layers:
            ids = []

            for h8 in neighborhood_h8:
                vals = poi_ids_by_h8_layer.get(
                    (h8, layer_code)
                )
                if vals:
                    ids.extend(vals)

            if not ids:
                accessibility_h9[
                    (h9, layer_code)
                ] = AccessibilityState(
                    candidate_count=0,
                    weighted_accessibility=0.0,
                    best_poi_id=-1,
                    best_distance_m=0.0,
                )
                continue

            ids_arr = np.asarray(
                ids,
                dtype=np.int64,
            )

            distances = haversine_m(
                origin_lat,
                origin_lng,
                world.amenity_lat[ids_arr],
                world.amenity_lng[ids_arr],
            )

            keep = (
                distances
                <= cfg.search_radius_m
            )

            ids_arr = ids_arr[keep]
            distances = distances[keep]

            if ids_arr.size == 0:
                accessibility_h9[
                    (h9, layer_code)
                ] = AccessibilityState(
                    candidate_count=0,
                    weighted_accessibility=0.0,
                    best_poi_id=-1,
                    best_distance_m=0.0,
                )
                continue

            attractiveness = (
                world.amenity_attractiveness[ids_arr]
            )

            weighted_accessibility = float(
                np.sum(
                    attractiveness
                    * np.exp(
                        -distances
                        / cfg.accessibility_decay_m
                    )
                )
            )

            scores = destination_score(
                distances,
                attractiveness,
                cfg.beta_distance,
                cfg.gamma_attractiveness,
            )

            best = int(np.argmax(scores))

            accessibility_h9[
                (h9, layer_code)
            ] = AccessibilityState(
                candidate_count=int(ids_arr.size),
                weighted_accessibility=weighted_accessibility,
                best_poi_id=int(ids_arr[best]),
                best_distance_m=float(distances[best]),
            )

    return StaticCityState(
        h3_config=hcfg,
        active_layer_codes=active_layers,

        macro_h5=macro_h5,
        amenities_h7=amenities_h7,
        amenities_h8=amenities_h8,
        accessibility_h9=accessibility_h9,
        poi_index_h10=poi_index_h10,

        build_time_s=(
            time.perf_counter() - t0
        ),
        n_pois=len(world.amenity_lat),
    )


def save_static_city(city, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        pickle.dump(
            city,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_static_city(path):
    """
    Load a StaticCity cache produced by either:

        1. the original flat-module Madrid repository, or
        2. the clean h3uab package.

    The original cache may contain pickle references such as:

        static_city.StaticCityState
        config.H3Config
        semantic_layers....
        models....

    while the clean repository packages those modules as:

        h3uab.static_city
        h3uab.config
        h3uab.semantic_layers
        h3uab.models

    This function temporarily registers compatibility aliases in
    sys.modules while unpickling.

    IMPORTANT
    ---------
    This does NOT rebuild or modify the static-city cache.
    It only resolves historical Python module names.
    """

    import importlib
    import sys

    path = Path(path)

    legacy_modules = {
        "static_city": "h3uab.static_city",
        "config": "h3uab.config",
        "models": "h3uab.models",
        "semantic_layers": "h3uab.semantic_layers",
        "geometry": "h3uab.geometry",
        "h3_utils": "h3uab.h3_utils",
    }

    previous_modules = {}

    try:
        for legacy_name, package_name in legacy_modules.items():
            previous_modules[legacy_name] = sys.modules.get(
                legacy_name
            )

            module = importlib.import_module(
                package_name
            )

            sys.modules[legacy_name] = module

        with path.open("rb") as f:
            city = pickle.load(f)

    finally:
        for legacy_name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(
                    legacy_name,
                    None,
                )
            else:
                sys.modules[legacy_name] = previous_module

    return city


def load_or_build_static_city(world, cfg):
    path = Path(cfg.static_city_cache)

    if (
        path.exists()
        and not cfg.rebuild_static_city
    ):
        t0 = time.perf_counter()
        city = load_static_city(path)
        return city, time.perf_counter() - t0, "loaded"

    city = build_static_city(
        world,
        cfg,
    )
    save_static_city(
        city,
        path,
    )

    return city, city.build_time_s, "built"


# ============================================================================
# V9.4 bounded local destination refinement helpers
# ============================================================================

def get_h10_candidate_poi_ids(
    city,
    h12_cell,
    layer_code,
    cfg,
    diagnostics=None,
):
    """
    Return a bounded candidate tuple from H3@10 buckets around the agent.

    Expected poi_index_h10 keys:
        (h10_cell, layer_code) -> iterable of POI ids

    If the historical cache uses another compatible dict layout, this helper
    tries both common conventions.
    """
    center_h10 = h3.cell_to_parent(
        h12_cell,
        cfg.h3.amenity_index_res,
    )

    cells = h3.grid_disk(
        center_h10,
        int(
            cfg.destination_refinement.h10_k_ring
        ),
    )

    candidate_ids = []

    index = city.poi_index_h10
    if diagnostics is not None:
        diagnostics["buckets_inspected"] = 0
        diagnostics["raw_ids"] = 0

    for cell in cells:
        if diagnostics is not None:
            diagnostics["buckets_inspected"] += 1
        value = None

        if (
            cell,
            layer_code,
        ) in index:
            value = index[
                (
                    cell,
                    layer_code,
                )
            ]
        elif cell in index:
            bucket = index[
                cell
            ]

            if isinstance(
                bucket,
                dict,
            ):
                value = bucket.get(
                    layer_code
                )
            else:
                value = bucket

        if value is None:
            continue

        if isinstance(
            value,
            (
                int,
                np.integer,
            ),
        ):
            candidate_ids.append(
                int(
                    value
                )
            )
        else:
            try:
                candidate_ids.extend(
                    int(
                        x
                    )
                    for x
                    in value
                )
            except TypeError:
                pass

        if diagnostics is not None:
            diagnostics["raw_ids"] = len(candidate_ids)
        if len(
            candidate_ids
        ) >= int(
            cfg.destination_refinement.max_candidates
        ):
            break

    if not candidate_ids:
        if diagnostics is not None:
            diagnostics["retained_ids"] = 0
        return ()

    # Stable deduplication + hard cap.
    result = tuple(
        dict.fromkeys(
            candidate_ids
        )
    )[
        :
        int(
            cfg.destination_refinement.max_candidates
        )
    ]
    if diagnostics is not None:
        diagnostics["retained_ids"] = len(result)
    return result


def get_h9_fallback_candidate_poi_ids(
    city,
    h12_cell,
    layer_code,
    cfg,
    diagnostics=None,
):
    """
    Fallback for H9 missing states.

    Search neighboring H9 cells and collect their best POI states. This keeps
    the fallback bounded rather than reverting to a full continuous scan.
    """
    center_h9 = h3.cell_to_parent(
        h12_cell,
        cfg.h3.accessibility_res,
    )

    cells = h3.grid_disk(
        center_h9,
        int(
            cfg.destination_refinement.fallback_h9_k_ring
        ),
    )

    ids = []
    if diagnostics is not None:
        diagnostics["cells_inspected"] = 0
        diagnostics["cells_with_poi"] = 0

    for cell in cells:
        if diagnostics is not None:
            diagnostics["cells_inspected"] += 1
        state = city.accessibility_h9.get(
            (
                cell,
                layer_code,
            )
        )

        if state is None:
            state = city.accessibility_h9.get(
                cell
            )

            if isinstance(
                state,
                dict,
            ):
                state = state.get(
                    layer_code
                )

        if (
            state is not None
            and getattr(
                state,
                "best_poi_id",
                -1,
            ) >= 0
        ):
            ids.append(
                int(
                    state.best_poi_id
                )
            )
            if diagnostics is not None:
                diagnostics["cells_with_poi"] += 1

        if len(
            ids
        ) >= int(
            cfg.destination_refinement.fallback_max_candidates
        ):
            break

    result = tuple(
        dict.fromkeys(
            ids
        )
    )
    if diagnostics is not None:
        diagnostics["raw_ids"] = len(ids)
        diagnostics["retained_ids"] = len(result)
    return result
