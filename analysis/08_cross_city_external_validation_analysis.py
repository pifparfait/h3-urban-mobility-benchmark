#!/usr/bin/env python3
"""
08_cross_city_external_validation_analysis_v2.py
================================================

Version 2 of the cross-city external OD validation synthesis.

CHANGES FROM V1
---------------
The primary spatial validation figure no longer draws straight origin-
destination chords.

Instead, H7 OD pairs are routed over the SAME OSM walking graph used for the
physical-network visualization layer. The figure therefore shows actual
serpentine street-network paths rather than abstract Euclidean links.

The figure design is:

Rows:
    1. New York City
    2. Cambridge
    3. Madrid

Columns:
    1. Observed OD structure
    2. Continuous residual vs observed
    3. H3 residual vs observed

The two residual panels use the same signed residual definition:

    residual = simulated_share - observed_share

and are plotted independently but on the same network and with a symmetric
per-city color scale so their visual similarity can be assessed directly.

This follows the methodological decision that the figure should illustrate
the residual structures, while quantitative non-degradation is established by
the cross-city table and ΔTVD / ΔJSD metrics.

No experiment is re-run here.

INPUTS
------
Primary external-validation summaries:
    outputs/external_od_validation/<city>/external_od_validation_summary.csv

H7 aligned OD:
    outputs/external_od_validation/<city>/three_way_od_h7.csv

Routing graphs are auto-detected from the project cache/configured data tree.
If a graph cannot be found, pass an explicit path with:
    --nyc-graph
    --cambridge-graph
    --madrid-graph

OUTPUTS
-------
outputs/cross_city_external_validation/
    cross_city_external_validation_full.csv
    TABLE_external_validation_cross_city.csv
    TABLE_external_validation_H7.csv
    TABLE_madrid_external_validation_sensitivity.csv
    paper_figures/
        FIG_EV1_cross_city_H7_external_validation_routed.png
        FIG_EV2_cross_city_external_validation_metrics.png

IMPORTANT
---------
This script is analysis/visualization only. Experiment 07 remains frozen.
"""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import h3
except ImportError as exc:
    raise SystemExit(
        "h3 is required. Install with: pip install h3"
    ) from exc

try:
    import networkx as nx
except ImportError as exc:
    raise SystemExit(
        "networkx is required. Install with: pip install networkx"
    ) from exc

try:
    import osmnx as ox
except ImportError as exc:
    raise SystemExit(
        "osmnx is required. Install with: pip install osmnx"
    ) from exc


# =============================================================================
# CONSTANTS
# =============================================================================

CITY_ORDER = [
    "new_york_ny",
    "cambridge_ma",
    "madrid",
]

CITY_LABELS = {
    "new_york_ny": "New York City",
    "cambridge_ma": "Cambridge",
    "madrid": "Madrid",
}

PRIMARY_ROOT = Path("outputs/external_od_validation")
SENSITIVITY_ROOT = Path("outputs/external_od_validation_sensitivity")
DEFAULT_OUTPUT = Path("outputs/cross_city_external_validation")

H7 = 7
EPS = 1e-12

DEFAULT_GRAPH_CANDIDATES = {
    "madrid": [
        Path("data/cache/madrid_walk_graph.graphml"),
        Path("data/madrid/cache/madrid_walk_graph.graphml"),
    ],
    "cambridge_ma": [
        Path("data/cambridge_ma/cache/cambridge_walk_graph.graphml"),
        Path("data/cambridge_ma/cache/cambridge_graph.graphml"),
        Path("data/cache/cambridge_walk_graph.graphml"),
    ],
    "new_york_ny": [
        Path("data/new_york_ny/cache/new_york_walk_graph.graphml"),
        Path("data/new_york_ny/cache/new_york_graph.graphml"),
        Path("data/cache/new_york_walk_graph.graphml"),
    ],
}

# Local zooms, consistent with the visual logic used in the earlier physical
# network figure. If None, bounds come from the plotted routes.
ZOOM_BOUNDS = {
    # Manhattan-focused local view.
    "new_york_ny": (-74.03, -73.93, 40.69, 40.82),

    # Cambridge / Kendall / Volpe local view.
    "cambridge_ma": (-71.125, -71.055, 42.345, 42.385),

    # Central Madrid local view.
    "madrid": (-3.735, -3.625, 40.395, 40.465),
}


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_city_summary(root: Path, city: str) -> pd.DataFrame:
    path = root / city / "external_od_validation_summary.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "h3_resolution",
        "continuous_pearson",
        "h3_pearson",
        "pearson_gap_h3_minus_continuous",
        "continuous_tvd",
        "h3_tvd",
        "tvd_gap_h3_minus_continuous",
        "tvd_relative_gap_h3_minus_continuous",
        "continuous_normalized_jsd",
        "h3_normalized_jsd",
        "normalized_jsd_gap_h3_minus_continuous",
        "continuous_h3_pearson",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} missing required columns: {sorted(missing)}"
        )

    out = df.copy()
    out["city_key"] = city
    out["city"] = CITY_LABELS[city]

    return out


def load_three_way_h7(root: Path, city: str) -> pd.DataFrame:
    path = root / city / "three_way_od_h7.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "origin_h3",
        "destination_h3",
        "observed_flow",
        "continuous_flow",
        "h3_flow",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} missing required columns: {sorted(missing)}"
        )

    for c in [
        "observed_flow",
        "continuous_flow",
        "h3_flow",
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        ).fillna(0.0)

    for flow_col, share_col in [
        ("observed_flow", "observed_share"),
        ("continuous_flow", "continuous_share"),
        ("h3_flow", "h3_share"),
    ]:
        total = float(df[flow_col].sum())

        if total <= EPS:
            raise ValueError(
                f"{city} H7 has zero total in {flow_col}"
            )

        df[share_col] = (
            df[flow_col] / total
        )

    df["continuous_residual"] = (
        df["continuous_share"]
        - df["observed_share"]
    )

    df["h3_residual"] = (
        df["h3_share"]
        - df["observed_share"]
    )

    return df


def h3_cell_centroid(
    cell: str,
) -> tuple[float, float]:
    lat, lng = h3.cell_to_latlng(str(cell))
    return float(lat), float(lng)


# =============================================================================
# GRAPH LOADING
# =============================================================================

def autodetect_graph(
    city: str,
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return explicit

    for candidate in DEFAULT_GRAPH_CANDIDATES[city]:
        if candidate.exists():
            return candidate

    # Broader fallback search.
    roots = [
        Path("data"),
        Path("outputs"),
    ]

    city_tokens = {
        "new_york_ny": ["new_york", "nyc"],
        "cambridge_ma": ["cambridge"],
        "madrid": ["madrid"],
    }[city]

    matches = []

    for root in roots:
        if not root.exists():
            continue

        for p in root.rglob("*.graphml"):
            low = str(p).lower()

            if any(
                token in low
                for token in city_tokens
            ):
                matches.append(p)

    if not matches:
        raise FileNotFoundError(
            f"Could not locate a GraphML routing graph for {city}. "
            f"Pass it explicitly with the corresponding --*-graph argument."
        )

    # Prefer files with 'walk' in name, then shortest path depth.
    matches.sort(
        key=lambda p: (
            0 if "walk" in p.name.lower() else 1,
            len(p.parts),
            len(str(p)),
        )
    )

    return matches[0]


def load_graph(path: Path):
    print(f"[load] routing graph: {path}")
    G = ox.load_graphml(path)

    if not isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        G = nx.MultiDiGraph(G)

    return G


# =============================================================================
# ROUTING
# =============================================================================

def nearest_graph_node(
    G,
    lat: float,
    lng: float,
):
    return ox.distance.nearest_nodes(
        G,
        X=float(lng),
        Y=float(lat),
    )


def route_nodes_to_xy(
    G,
    nodes: list,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []

    for node in nodes:
        data = G.nodes[node]
        xs.append(float(data["x"]))
        ys.append(float(data["y"]))

    return np.asarray(xs), np.asarray(ys)


def shortest_path_weight_name(
    G,
) -> str | None:
    # Prefer physical length when present.
    for _, _, data in G.edges(data=True):
        if "length" in data:
            return "length"
        break
    return None


def build_h7_node_lookup(
    G,
    od_df: pd.DataFrame,
) -> dict[str, object]:
    cells = pd.Index(
        pd.unique(
            pd.concat(
                [
                    od_df["origin_h3"].astype(str),
                    od_df["destination_h3"].astype(str),
                ],
                ignore_index=True,
            )
        )
    )

    lookup = {}

    for cell in cells:
        lat, lng = h3_cell_centroid(cell)
        lookup[cell] = nearest_graph_node(
            G,
            lat,
            lng,
        )

    return lookup


def route_top_pairs(
    G,
    df: pd.DataFrame,
    *,
    top_n: int,
    zoom_bounds: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    """
    Route a balanced subset of important H7 OD pairs.

    Ranking uses the maximum of observed, Continuous, and H3 shares so no
    architecture is privileged.
    """
    x = df.copy()

    x["plot_rank"] = x[
        [
            "observed_share",
            "continuous_share",
            "h3_share",
        ]
    ].max(axis=1)

    x = x.sort_values(
        "plot_rank",
        ascending=False,
    )

    # To keep routing tractable and the figure readable.
    x = x.head(max(top_n * 4, top_n))

    node_lookup = build_h7_node_lookup(
        G,
        x,
    )

    weight = shortest_path_weight_name(G)

    routed_rows = []

    for _, row in x.iterrows():
        o = str(row["origin_h3"])
        d = str(row["destination_h3"])

        try:
            nodes = nx.shortest_path(
                G,
                source=node_lookup[o],
                target=node_lookup[d],
                weight=weight,
            )
        except (
            nx.NetworkXNoPath,
            nx.NodeNotFound,
        ):
            continue

        xs, ys = route_nodes_to_xy(
            G,
            nodes,
        )

        if len(xs) < 2:
            continue

        if zoom_bounds is not None:
            west, east, south, north = zoom_bounds

            # Keep routes that touch the local visualization window.
            inside = (
                (xs >= west)
                & (xs <= east)
                & (ys >= south)
                & (ys <= north)
            )

            if not inside.any():
                continue

        rec = row.to_dict()
        rec["route_x"] = xs
        rec["route_y"] = ys

        routed_rows.append(rec)

        if len(routed_rows) >= top_n:
            break

    if not routed_rows:
        raise RuntimeError(
            "No routable OD pairs remained for plotting."
        )

    return pd.DataFrame(routed_rows)


# =============================================================================
# NETWORK DRAWING
# =============================================================================

def draw_base_graph(
    ax,
    G,
    *,
    zoom_bounds: tuple[float, float, float, float] | None,
) -> None:
    """
    Lightweight OSM network background.
    """
    for u, v, data in G.edges(data=True):
        xu = G.nodes[u].get("x")
        yu = G.nodes[u].get("y")
        xv = G.nodes[v].get("x")
        yv = G.nodes[v].get("y")

        if None in (xu, yu, xv, yv):
            continue

        if zoom_bounds is not None:
            west, east, south, north = zoom_bounds

            if (
                max(xu, xv) < west
                or min(xu, xv) > east
                or max(yu, yv) < south
                or min(yu, yv) > north
            ):
                continue

        ax.plot(
            [xu, xv],
            [yu, yv],
            linewidth=0.25,
            alpha=0.25,
        )


def draw_routed_metric(
    ax,
    G,
    routed: pd.DataFrame,
    *,
    value_col: str,
    signed: bool,
    title: str,
    zoom_bounds: tuple[float, float, float, float] | None,
    shared_abs_scale: float | None = None,
) -> None:
    draw_base_graph(
        ax,
        G,
        zoom_bounds=zoom_bounds,
    )

    values = routed[value_col].to_numpy(dtype=float)

    if signed:
        vmax = (
            float(shared_abs_scale)
            if shared_abs_scale is not None
            else float(np.nanmax(np.abs(values)))
        )
        vmax = max(vmax, EPS)

        cmap = plt.get_cmap("coolwarm")
        norm = plt.Normalize(
            -vmax,
            vmax,
        )

        for _, row in routed.iterrows():
            v = float(row[value_col])

            # Width driven by residual magnitude.
            width = (
                0.6
                + 6.0 * abs(v) / vmax
            )

            ax.plot(
                row["route_x"],
                row["route_y"],
                linewidth=width,
                alpha=0.82,
                color=cmap(
                    norm(v)
                ),
            )

        sm = plt.cm.ScalarMappable(
            norm=norm,
            cmap=cmap,
        )
        sm.set_array([])

        plt.colorbar(
            sm,
            ax=ax,
            fraction=0.035,
            pad=0.015,
        )

    else:
        vmax = float(
            np.nanmax(values)
        )
        vmax = max(
            vmax,
            EPS,
        )

        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(
            0.0,
            vmax,
        )

        for _, row in routed.iterrows():
            v = float(row[value_col])

            width = (
                0.6
                + 7.0 * v / vmax
            )

            ax.plot(
                row["route_x"],
                row["route_y"],
                linewidth=width,
                alpha=0.86,
                color=cmap(
                    norm(v)
                ),
            )

        sm = plt.cm.ScalarMappable(
            norm=norm,
            cmap=cmap,
        )
        sm.set_array([])

        plt.colorbar(
            sm,
            ax=ax,
            fraction=0.035,
            pad=0.015,
        )

    if zoom_bounds is not None:
        west, east, south, north = zoom_bounds
        ax.set_xlim(
            west,
            east,
        )
        ax.set_ylim(
            south,
            north,
        )

    ax.set_title(
        title,
        fontsize=10,
    )

    ax.set_xlabel(
        "Longitude",
        fontsize=8,
    )
    ax.set_ylabel(
        "Latitude",
        fontsize=8,
    )

    ax.tick_params(
        labelsize=7,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )


# =============================================================================
# TABLES
# =============================================================================

def build_cross_city_summary(
    primary_root: Path,
) -> pd.DataFrame:
    frames = []

    for city in CITY_ORDER:
        frames.append(
            load_city_summary(
                primary_root,
                city,
            )
        )

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    out["city_order"] = out["city_key"].map(
        {
            c: i
            for i, c in enumerate(CITY_ORDER)
        }
    )

    out = out.sort_values(
        [
            "city_order",
            "h3_resolution",
        ]
    ).drop(
        columns=[
            "city_order"
        ]
    )

    return out.reset_index(
        drop=True
    )


def build_paper_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "city",
        "h3_resolution",
        "continuous_pearson",
        "h3_pearson",
        "pearson_gap_h3_minus_continuous",
        "continuous_tvd",
        "h3_tvd",
        "tvd_gap_h3_minus_continuous",
        "tvd_relative_gap_h3_minus_continuous",
        "continuous_normalized_jsd",
        "h3_normalized_jsd",
        "normalized_jsd_gap_h3_minus_continuous",
        "continuous_h3_pearson",
    ]

    return summary[
        cols
    ].copy()


def build_h7_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    x = summary[
        summary["h3_resolution"] == H7
    ].copy()

    cols = [
        "city",
        "continuous_pearson",
        "h3_pearson",
        "pearson_gap_h3_minus_continuous",
        "continuous_tvd",
        "h3_tvd",
        "tvd_gap_h3_minus_continuous",
        "tvd_relative_gap_h3_minus_continuous",
        "continuous_normalized_jsd",
        "h3_normalized_jsd",
        "normalized_jsd_gap_h3_minus_continuous",
        "continuous_h3_pearson",
    ]

    return x[
        cols
    ].copy()


def build_madrid_sensitivity(
    primary_root: Path,
    sensitivity_root: Path,
) -> pd.DataFrame | None:
    p = (
        primary_root
        / "madrid"
        / "external_od_validation_summary.csv"
    )

    s = (
        sensitivity_root
        / "madrid"
        / "external_od_validation_summary.csv"
    )

    if not p.exists() or not s.exists():
        return None

    work = pd.read_csv(p)
    work[
        "simulation_scope"
    ] = "work"

    all_trips = pd.read_csv(s)
    all_trips[
        "simulation_scope"
    ] = "all"

    cols = [
        "simulation_scope",
        "h3_resolution",
        "continuous_pearson",
        "h3_pearson",
        "pearson_gap_h3_minus_continuous",
        "continuous_tvd",
        "h3_tvd",
        "tvd_gap_h3_minus_continuous",
        "tvd_relative_gap_h3_minus_continuous",
        "continuous_normalized_jsd",
        "h3_normalized_jsd",
        "normalized_jsd_gap_h3_minus_continuous",
        "continuous_h3_pearson",
    ]

    return pd.concat(
        [
            work[cols],
            all_trips[cols],
        ],
        ignore_index=True,
    )


# =============================================================================
# FIGURES
# =============================================================================

def make_h7_routed_validation_figure(
    primary_root: Path,
    output_path: Path,
    graph_paths: dict[str, Path],
    *,
    top_n: int,
) -> None:
    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(16, 14),
    )

    for row_idx, city in enumerate(
        CITY_ORDER
    ):
        df = load_three_way_h7(
            primary_root,
            city,
        )

        G = load_graph(
            graph_paths[city]
        )

        zoom = ZOOM_BOUNDS[
            city
        ]

        routed = route_top_pairs(
            G,
            df,
            top_n=top_n,
            zoom_bounds=zoom,
        )

        # Continuous and H3 residual panels MUST share the same absolute
        # scale within each city.
        residual_abs_scale = float(
            np.nanmax(
                np.abs(
                    np.concatenate(
                        [
                            routed[
                                "continuous_residual"
                            ].to_numpy(
                                dtype=float
                            ),
                            routed[
                                "h3_residual"
                            ].to_numpy(
                                dtype=float
                            ),
                        ]
                    )
                )
            )
        )

        label = CITY_LABELS[
            city
        ]

        draw_routed_metric(
            axes[
                row_idx,
                0,
            ],
            G,
            routed,
            value_col="observed_share",
            signed=False,
            title=(
                f"{label} — observed H7 OD"
            ),
            zoom_bounds=zoom,
        )

        draw_routed_metric(
            axes[
                row_idx,
                1,
            ],
            G,
            routed,
            value_col="continuous_residual",
            signed=True,
            title=(
                f"{label} — Continuous − observed"
            ),
            zoom_bounds=zoom,
            shared_abs_scale=residual_abs_scale,
        )

        draw_routed_metric(
            axes[
                row_idx,
                2,
            ],
            G,
            routed,
            value_col="h3_residual",
            signed=True,
            title=(
                f"{label} — H3 − observed"
            ),
            zoom_bounds=zoom,
            shared_abs_scale=residual_abs_scale,
        )

    fig.suptitle(
        "Observed-OD comparison on routed urban networks (H7)",
        fontsize=15,
    )

    fig.text(
        0.5,
        0.015,
        (
            "Observed OD flows and residual structures are projected onto the "
            "same OSM walking network used for both simulation backends. "
            "Continuous and H3 residual panels use identical signed color "
            "scales within each city; the incremental H3 − Continuous "
            "difference is quantified with ΔTVD and ΔJSD rather "
            "than by visual subtraction."
        ),
        ha="center",
        va="bottom",
        wrap=True,
        fontsize=9,
    )

    fig.tight_layout(
        rect=(
            0,
            0.045,
            1,
            0.97,
        )
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def make_cross_city_metric_figure(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(9, 9),
    )

    for city in CITY_ORDER:
        x = summary[
            summary[
                "city_key"
            ] == city
        ].sort_values(
            "h3_resolution"
        )

        axes[0].plot(
            x[
                "h3_resolution"
            ],
            x[
                "continuous_pearson"
            ],
            marker="o",
            linestyle="-",
            label=(
                f"{CITY_LABELS[city]} — Continuous"
            ),
        )

        axes[0].plot(
            x[
                "h3_resolution"
            ],
            x[
                "h3_pearson"
            ],
            marker="s",
            linestyle="--",
            label=(
                f"{CITY_LABELS[city]} — H3"
            ),
        )

    axes[0].set_xlabel(
        "H3 resolution"
    )
    axes[0].set_ylabel(
        "Pearson vs observed OD"
    )
    axes[0].set_title(
        "Absolute empirical agreement"
    )
    axes[0].set_xticks(
        [
            7,
            8,
            9,
        ]
    )
    axes[0].axhline(
        0.0,
        linewidth=0.8,
    )
    axes[0].legend(
        fontsize=8,
        ncol=2,
    )

    for city in CITY_ORDER:
        x = summary[
            summary[
                "city_key"
            ] == city
        ].sort_values(
            "h3_resolution"
        )

        axes[1].plot(
            x[
                "h3_resolution"
            ],
            x[
                "tvd_gap_h3_minus_continuous"
            ],
            marker="o",
            label=(
                CITY_LABELS[
                    city
                ]
            ),
        )

    axes[1].set_xlabel(
        "H3 resolution"
    )
    axes[1].set_ylabel(
        "ΔTVD = TVD(H3) − TVD(Continuous)"
    )
    axes[1].set_title(
        "Incremental empirical error introduced by H3"
    )
    axes[1].set_xticks(
        [
            7,
            8,
            9,
        ]
    )
    axes[1].axhline(
        0.0,
        linewidth=1.0,
    )
    axes[1].legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# REPORT
# =============================================================================

def print_summary(
    summary: pd.DataFrame,
) -> None:
    cols = [
        "city",
        "h3_resolution",
        "continuous_pearson",
        "h3_pearson",
        "pearson_gap_h3_minus_continuous",
        "continuous_tvd",
        "h3_tvd",
        "tvd_gap_h3_minus_continuous",
        "tvd_relative_gap_h3_minus_continuous",
        "continuous_normalized_jsd",
        "h3_normalized_jsd",
        "continuous_h3_pearson",
    ]

    print(
        "=" * 132
    )
    print(
        "CROSS-CITY EXTERNAL OD VALIDATION SUMMARY"
    )
    print(
        "=" * 132
    )

    print(
        summary[
            cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-city external OD synthesis with routed H7 network "
            "visualization for NYC, Cambridge, and Madrid."
        )
    )

    parser.add_argument(
        "--primary-root",
        type=Path,
        default=PRIMARY_ROOT,
    )

    parser.add_argument(
        "--sensitivity-root",
        type=Path,
        default=SENSITIVITY_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--top-n-h7",
        type=int,
        default=120,
        help=(
            "Maximum number of routed H7 OD pairs plotted per city. "
            "Default 120 keeps the physical network legible."
        ),
    )

    parser.add_argument(
        "--nyc-graph",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--cambridge-graph",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--madrid-graph",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    outdir = ensure_dir(
        args.output_dir
    )

    figdir = ensure_dir(
        outdir
        / "paper_figures"
    )

    graph_paths = {
        "new_york_ny": autodetect_graph(
            "new_york_ny",
            args.nyc_graph,
        ),
        "cambridge_ma": autodetect_graph(
            "cambridge_ma",
            args.cambridge_graph,
        ),
        "madrid": autodetect_graph(
            "madrid",
            args.madrid_graph,
        ),
    }

    print(
        "=" * 132
    )
    print(
        "ROUTING GRAPHS"
    )
    print(
        "=" * 132
    )

    for city in CITY_ORDER:
        print(
            f"{CITY_LABELS[city]:16s}: "
            f"{graph_paths[city]}"
        )

    # -------------------------------------------------------------------------
    # 1. Cross-city summary
    # -------------------------------------------------------------------------
    summary = build_cross_city_summary(
        args.primary_root
    )

    summary_path = (
        outdir
        / "cross_city_external_validation_full.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 2. Tables
    # -------------------------------------------------------------------------
    paper = build_paper_table(
        summary
    )

    paper_path = (
        outdir
        / "TABLE_external_validation_cross_city.csv"
    )

    paper.to_csv(
        paper_path,
        index=False,
    )

    h7 = build_h7_table(
        summary
    )

    h7_path = (
        outdir
        / "TABLE_external_validation_H7.csv"
    )

    h7.to_csv(
        h7_path,
        index=False,
    )

    sensitivity = build_madrid_sensitivity(
        args.primary_root,
        args.sensitivity_root,
    )

    sensitivity_path = None

    if sensitivity is not None:
        sensitivity_path = (
            outdir
            / "TABLE_madrid_external_validation_sensitivity.csv"
        )

        sensitivity.to_csv(
            sensitivity_path,
            index=False,
        )

    # -------------------------------------------------------------------------
    # 3. Figures
    # -------------------------------------------------------------------------
    routed_path = (
        figdir
        / "FIG_EV1_cross_city_H7_external_validation_routed.png"
    )

    make_h7_routed_validation_figure(
        args.primary_root,
        routed_path,
        graph_paths,
        top_n=args.top_n_h7,
    )

    metric_path = (
        figdir
        / "FIG_EV2_cross_city_external_validation_metrics.png"
    )

    make_cross_city_metric_figure(
        summary,
        metric_path,
    )

    # -------------------------------------------------------------------------
    # 4. Console
    # -------------------------------------------------------------------------
    print_summary(
        summary
    )

    print()
    print(
        "=" * 132
    )
    print(
        "H7 PRIMARY CROSS-CITY RESULT"
    )
    print(
        "=" * 132
    )

    print(
        h7.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "=" * 132
    )
    print(
        "FILES SAVED"
    )
    print(
        "=" * 132
    )

    print(
        f"[saved] {summary_path.resolve()}"
    )
    print(
        f"[saved] {paper_path.resolve()}"
    )
    print(
        f"[saved] {h7_path.resolve()}"
    )

    if sensitivity_path is not None:
        print(
            f"[saved] {sensitivity_path.resolve()}"
        )

    print(
        f"[saved] {routed_path.resolve()}"
    )
    print(
        f"[saved] {metric_path.resolve()}"
    )

    print(
        "\nInterpretation: the routed figure visualizes observed and residual "
        "OD structures on the same physical OSM network. Quantitative "
        "non-degradation is established by the near-zero H3-minus-Continuous "
        "TVD/JSD gaps, not by visual subtraction."
    )


if __name__ == "__main__":
    main()
