from __future__ import annotations

import math
import h3


def aoi_poly(cfg):
    return h3.LatLngPoly(
        [
            [cfg.lat_min, cfg.lng_min],
            [cfg.lat_min, cfg.lng_max],
            [cfg.lat_max, cfg.lng_max],
            [cfg.lat_max, cfg.lng_min],
        ]
    )


def cells_covering_aoi(cfg, resolution):
    return list(
        h3.h3shape_to_cells(
            aoi_poly(cfg),
            resolution,
        )
    )


def average_edge_length_m(resolution):
    if hasattr(h3, "average_hexagon_edge_length"):
        return float(
            h3.average_hexagon_edge_length(
                resolution,
                unit="m",
            )
        )
    raise RuntimeError(
        "Expected h3>=4.1 with average_hexagon_edge_length()."
    )


def radius_to_k(radius_m, resolution):
    """
    Conservative ring distance for a fixed physical radius.
    """
    edge = average_edge_length_m(resolution)
    center_spacing = max(math.sqrt(3.0) * edge, 1.0)
    return max(
        1,
        int(math.ceil(radius_m / center_spacing)) + 1,
    )


def parent(cell: str, resolution: int) -> str:
    current = h3.get_resolution(cell)
    if current == resolution:
        return cell
    if current < resolution:
        raise ValueError(
            f"Cannot get parent at finer resolution {resolution} "
            f"from cell resolution {current}."
        )
    return h3.cell_to_parent(cell, resolution)
