#!/usr/bin/env python3
"""
FINAL SCIENTIFIC VISUALIZATION
==============================

Purpose
-------
Create the figures needed to demonstrate, in one coherent visual package:

A) Computational scalability / architectural gain
   Continuous-indexed vs H3 from 10K -> 1M.

B) Urban fidelity
   Accessibility + OD + visit density + occupancy at H7/H8/H9.

C) Madrid spatial preservation
   Real Madrid OSM road graph as basemap, with:
       Continuous heatmap
       H3 heatmap
       H3 - Continuous difference
   for visit density and final occupancy.

D) Madrid OD preservation
   Real Madrid OSM road graph as basemap, with:
       Continuous OD
       H3 OD
       OD difference
   using exactly the same edge-flow normalization.

E) Local zoom
   Automatically chooses the highest-activity H8 area and produces a
   Continuous / H3 / Difference zoom.

Architectural rules
-------------------
1. H11/H12 NEVER enter fidelity claims.
2. H11/H12 are localization/visualization only.
3. Routing is NOT discretized.
4. The OSM graph is used here ONLY as a geographic basemap.
5. H7/H8/H9 are analytical aggregation scales applied to real coordinates.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math

import h3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.collections import (
    LineCollection,
    PolyCollection,
)
from matplotlib.colors import TwoSlopeNorm

import osmnx as ox
import networkx as nx
from matplotlib.colors import PowerNorm


ROOT = Path(__file__).resolve().parents[1]

ARCH_DIR = ROOT / "outputs" / "architecture"
FID_DIR = ROOT / "outputs" / "urban_fidelity"
FIG_DIR = ROOT / "outputs" / "paper_figures"
GRAPH_PATH = ROOT / "data" / "cache" / "madrid_walk_graph.graphml"


# =====================================================================
# Helpers
# =====================================================================

def ensure_dirs():
    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def population_from_name(path: Path) -> int:
    digits = "".join(
        ch
        for ch in path.stem
        if ch.isdigit()
    )
    return int(digits)


def load_architecture():
    files = sorted(
        ARCH_DIR.glob(
            "architecture_summary_n*.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No architecture_summary_n*.csv in {ARCH_DIR}"
        )

    frames = [
        pd.read_csv(
            p
        )
        for p in files
    ]

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    return df.sort_values(
        [
            "operation",
            "n_agents",
        ]
    )


def load_largest_fidelity():
    files = sorted(
        FID_DIR.glob(
            "urban_fidelity_n*.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No urban_fidelity_n*.csv in {FID_DIR}"
        )

    path = max(
        files,
        key=population_from_name,
    )

    return (
        pd.read_csv(
            path
        ),
        population_from_name(
            path
        ),
    )


def choose_diagnostics_population(
    requested=None,
):
    dirs = [
        p
        for p in FID_DIR.glob(
            "diagnostics_n*"
        )
        if p.is_dir()
    ]

    if not dirs:
        raise FileNotFoundError(
            "No diagnostics_n* directory found. "
            "Run the updated 03_urban_fidelity.py first."
        )

    if requested is not None:
        target = (
            FID_DIR
            / f"diagnostics_n{int(requested)}"
        )

        if target.exists():
            return target, int(
                requested
            )

    chosen = max(
        dirs,
        key=lambda p: int(
            p.name.split(
                "diagnostics_n"
            )[
                1
            ]
        ),
    )

    n = int(
        chosen.name.split(
            "diagnostics_n"
        )[
            1
        ]
    )

    return chosen, n


def load_graph():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"OSM graph not found: {GRAPH_PATH}"
        )

    print(
        f"[MAP] Loading Madrid OSM graph: "
        f"{GRAPH_PATH}"
    )

    return ox.load_graphml(
        GRAPH_PATH
    )


def graph_segments(G):
    """
    Convert the OSM graph to matplotlib line segments once.

    Geometry is used when available; otherwise node-to-node segments are used.
    """
    segments = []

    for u, v, _, data in G.edges(
        keys=True,
        data=True,
    ):
        geometry = data.get(
            "geometry"
        )

        if geometry is not None:
            try:
                coords = list(
                    geometry.coords
                )

                if len(
                    coords
                ) >= 2:
                    segments.append(
                        np.asarray(
                            coords,
                            dtype=float,
                        )
                    )
                    continue
            except Exception:
                pass

        nu = G.nodes[
            u
        ]

        nv = G.nodes[
            v
        ]

        segments.append(
            np.asarray(
                [
                    [
                        float(
                            nu[
                                "x"
                            ]
                        ),
                        float(
                            nu[
                                "y"
                            ]
                        ),
                    ],
                    [
                        float(
                            nv[
                                "x"
                            ]
                        ),
                        float(
                            nv[
                                "y"
                            ]
                        ),
                    ],
                ],
                dtype=float,
            )
        )

    return segments


def graph_bounds(G):
    xs = np.asarray(
        [
            float(
                data[
                    "x"
                ]
            )
            for _, data
            in G.nodes(
                data=True
            )
        ],
        dtype=float,
    )

    ys = np.asarray(
        [
            float(
                data[
                    "y"
                ]
            )
            for _, data
            in G.nodes(
                data=True
            )
        ],
        dtype=float,
    )

    return (
        xs.min(),
        xs.max(),
        ys.min(),
        ys.max(),
    )


def add_osm_basemap(
    ax,
    segments,
    bounds,
    zoom_bounds=None,
):
    """
    Draw the real OSM road graph beneath the analytical layers.

    OSM here is ONLY background geographic context.
    """
    # Subdued OSM basemap: geographic context only.
    # The simulated OD flows / H3 observables remain visually dominant.
    roads = LineCollection(
        segments,
        linewidths=0.12,
        alpha=0.10,
        zorder=1,
    )

    ax.add_collection(
        roads
    )

    if zoom_bounds is None:
        xmin, xmax, ymin, ymax = bounds
    else:
        xmin, xmax, ymin, ymax = zoom_bounds

    ax.set_xlim(
        xmin,
        xmax,
    )

    ax.set_ylim(
        ymin,
        ymax,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )


def h3_polygons(cells):
    polygons = []

    for cell in cells:
        boundary = h3.cell_to_boundary(
            str(
                cell
            )
        )

        polygons.append(
            [
                (
                    float(
                        lng
                    ),
                    float(
                        lat
                    ),
                )
                for lat, lng
                in boundary
            ]
        )

    return polygons


def percentile_clip(
    values,
    percentile=99.0,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    positive = values[
        values > 0
    ]

    if positive.size == 0:
        return 1.0

    return max(
        float(
            np.percentile(
                positive,
                percentile,
            )
        ),
        1.0,
    )


def add_h3_heatmap(
    ax,
    df,
    polygons,
    column,
    vmin,
    vmax,
    difference=False,
):
    values = df[
        column
    ].to_numpy(
        dtype=float
    )

    if difference:
        collection = PolyCollection(
            polygons,
            array=values,
            edgecolors="none",
            alpha=0.86,
            cmap="coolwarm",
            norm=TwoSlopeNorm(
                vmin=vmin,
                vcenter=0.0,
                vmax=vmax,
            ),
            zorder=3,
        )
    else:
        collection = PolyCollection(
            polygons,
            array=values,
            edgecolors="none",
            alpha=0.78,
            zorder=3,
        )

        collection.set_clim(
            vmin,
            vmax,
        )

    ax.add_collection(
        collection
    )

    return collection


# AUDIT OUTPUT FAMILY: FIG_MAP_visit_density

# =====================================================================
# A. Computational scalability
# =====================================================================

def plot_speedup_scalability(
    architecture,
):
    labels = {
        "pure_spatial_lookup":
            "Urban-state lookup",
        "urban_context_retrieval":
            "Urban-context retrieval",
        "accessibility_query":
            "Accessibility",
        "candidate_retrieval":
            "Candidate retrieval",
        "full_agent_decision":
            "Full decision",
    }

    fig, ax = plt.subplots(
        figsize=(
            9.4,
            5.8,
        )
    )

    for operation, g in architecture.groupby(
        "operation"
    ):
        g = g.sort_values(
            "n_agents"
        )

        ax.plot(
            g[
                "n_agents"
            ],
            g[
                "speedup_h3_over_continuous"
            ],
            marker="o",
            linewidth=2,
            label=labels.get(
                operation,
                operation,
            ),
        )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        "Agents / queries"
    )

    ax.set_ylabel(
        "H3 speedup over Continuous-indexed (×)"
    )

    ax.set_title(
        "Computational scalability of the H3 urban-knowledge architecture"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    path = (
        FIG_DIR
        / "FIG_A_architecture_speedup.png"
    )

    fig.savefig(
        path,
        dpi=240,
    )

    plt.close(
        fig
    )

    return path


def plot_full_decision_scalability(
    architecture,
):
    g = architecture[
        architecture[
            "operation"
        ]
        == "full_agent_decision"
    ].sort_values(
        "n_agents"
    )

    fig, ax = plt.subplots(
        figsize=(
            8.8,
            5.5,
        )
    )

    ax.plot(
        g[
            "n_agents"
        ],
        g[
            "continuous_us_per_query"
        ],
        marker="o",
        linewidth=2,
        label="Continuous-indexed",
    )

    ax.plot(
        g[
            "n_agents"
        ],
        g[
            "h3_us_per_query"
        ],
        marker="o",
        linewidth=2,
        label="H3",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        "Agents / decisions"
    )

    ax.set_ylabel(
        "Median latency (µs / decision)"
    )

    ax.set_title(
        "End-to-end agent-decision latency"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    path = (
        FIG_DIR
        / "FIG_B_full_decision_latency.png"
    )

    fig.savefig(
        path,
        dpi=240,
    )

    plt.close(
        fig
    )

    return path


# =====================================================================
# B. Fidelity summary
# =====================================================================

def plot_fidelity_summary(
    fidelity,
    n_agents,
):
    rows = []

    access = fidelity[
        fidelity[
            "metric"
        ]
        == "accessibility"
    ]

    if not access.empty:
        rows.append(
            {
                "observable":
                    "Accessibility",
                "H7":
                    float(
                        access.iloc[
                            0
                        ][
                            "pearson"
                        ]
                    ),
                "H8":
                    float(
                        access.iloc[
                            0
                        ][
                            "pearson"
                        ]
                    ),
                "H9":
                    float(
                        access.iloc[
                            0
                        ][
                            "pearson"
                        ]
                    ),
            }
        )

    for metric, label in (
        (
            "od_flow",
            "OD flow",
        ),
        (
            "occupancy",
            "Final occupancy",
        ),
    ):
        row = {
            "observable":
                label,
        }

        for scale in (
            "H7",
            "H8",
            "H9",
        ):
            q = fidelity[
                (
                    fidelity[
                        "metric"
                    ]
                    == metric
                )
                & (
                    fidelity[
                        "scale"
                    ]
                    == scale
                )
            ]

            row[
                scale
            ] = (
                float(
                    q.iloc[
                        0
                    ][
                        "pearson"
                    ]
                )
                if not q.empty
                else np.nan
            )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    ).set_index(
        "observable"
    )

    fig, ax = plt.subplots(
        figsize=(
            8.8,
            5.5,
        )
    )

    summary.T.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_xlabel(
        "Urban analysis scale"
    )

    ax.set_ylabel(
        "Pearson correlation"
    )

    ax.set_title(
        f"Urban fidelity — N={n_agents:,}"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    path = (
        FIG_DIR
        / "FIG_C_urban_fidelity.png"
    )

    fig.savefig(
        path,
        dpi=240,
    )

    plt.close(
        fig
    )

    return path


# =====================================================================
# C. Madrid heatmaps
# =====================================================================

def plot_madrid_heatmap_triptych(
    *,
    G,
    segments,
    bounds,
    diagnostics_dir,
    metric_name,
    resolution,
    n_agents,
    zoom_bounds=None,
    suffix="city",
):
    csv_path = (
        diagnostics_dir
        / f"{metric_name}_H{resolution}.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            csv_path
        )

    df = pd.read_csv(
        csv_path
    )

    polygons = h3_polygons(
        df[
            "h3_id"
        ].tolist()
    )

    vmax = percentile_clip(
        np.concatenate(
            [
                df[
                    "continuous_count"
                ].to_numpy(
                    dtype=float
                ),
                df[
                    "h3_count"
                ].to_numpy(
                    dtype=float
                ),
            ]
        ),
        percentile=99.0,
    )

    diff = df[
        "difference"
    ].to_numpy(
        dtype=float
    )

    max_abs = percentile_clip(
        np.abs(
            diff
        ),
        percentile=99.0,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            18,
            6.2,
        ),
        constrained_layout=True,
    )

    titles = [
        "Continuous-indexed",
        "H3-indexed",
        "H3 − Continuous",
    ]

    columns = [
        "continuous_count",
        "h3_count",
        "difference",
    ]

    collections = []

    for ax, title, column in zip(
        axes,
        titles,
        columns,
    ):
        add_osm_basemap(
            ax,
            segments,
            bounds,
            zoom_bounds=zoom_bounds,
        )

        if column == "difference":
            collection = add_h3_heatmap(
                ax,
                df,
                polygons,
                column,
                -max_abs,
                max_abs,
                difference=True,
            )
        else:
            collection = add_h3_heatmap(
                ax,
                df,
                polygons,
                column,
                0.0,
                vmax,
                difference=False,
            )

        collections.append(
            collection
        )

        ax.set_title(
            title
        )

    metric_label = metric_name.replace(
        "_",
        " ",
    )

    if suffix == "city":
        scope_label = "city-wide"
    elif suffix == "manhattan":
        scope_label = "Manhattan zoom"
    elif suffix == "midtown_deep_zoom":
        scope_label = "Midtown deep zoom"
    elif suffix == "kendall":
        scope_label = "Kendall Square zoom"
    elif suffix == "volpe_deep_zoom":
        scope_label = "Volpe Center deep zoom"
    elif suffix in {"street_zoom", "deep_zoom"}:
        scope_label = "deep local zoom"
    else:
        scope_label = "local zoom"

    fig.suptitle(
        (
            f"Madrid {metric_label} — "
            f"H{resolution} aggregation, {scope_label}, N={n_agents:,}"
        ),
        fontsize=15,
    )

    # Same color scale for Continuous and H3.
    fig.colorbar(
        collections[
            0
        ],
        ax=[
            axes[
                0
            ],
            axes[
                1
            ],
        ],
        shrink=0.82,
        label="Count",
    )

    fig.colorbar(
        collections[
            2
        ],
        ax=axes[
            2
        ],
        shrink=0.82,
        label="H3 − Continuous",
    )

    path = (
        FIG_DIR
        / (
            f"FIG_MAP_{metric_name}_"
            f"H{resolution}_{suffix}.png"
        )
    )

    fig.savefig(
        path,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return path


def compute_activity_zoom(
    diagnostics_dir,
    resolution=8,
    expansion=0.018,
):
    """
    Find the strongest common activity location and construct a geographic
    zoom around it.

    The zoom is data-driven, not manually selected.
    """
    path = (
        diagnostics_dir
        / f"visit_density_H{resolution}.csv"
    )

    df = pd.read_csv(
        path
    )

    common_activity = np.minimum(
        df[
            "continuous_count"
        ].to_numpy(
            dtype=float
        ),
        df[
            "h3_count"
        ].to_numpy(
            dtype=float
        ),
    )

    idx = int(
        np.argmax(
            common_activity
        )
    )

    center_lat = float(
        df.iloc[
            idx
        ][
            "lat"
        ]
    )

    center_lng = float(
        df.iloc[
            idx
        ][
            "lng"
        ]
    )

    return (
        center_lng
        - expansion,
        center_lng
        + expansion,
        center_lat
        - expansion,
        center_lat
        + expansion,
    )


# =====================================================================
# D. OD maps
# =====================================================================

def od_line_segments(
    df,
):
    return [
        [
            (
                float(
                    row.origin_lng
                ),
                float(
                    row.origin_lat
                ),
            ),
            (
                float(
                    row.destination_lng
                ),
                float(
                    row.destination_lat
                ),
            ),
        ]
        for row in df.itertuples(
            index=False
        )
    ]


def select_top_od_union(
    df,
    top_k,
):
    """
    Fair visualization selection:
    union of the top-K OD flows from Continuous and H3.

    This prevents selecting links only from one architecture.
    """
    c_idx = set(
        df.nlargest(
            top_k,
            "continuous_count",
        ).index.tolist()
    )

    h_idx = set(
        df.nlargest(
            top_k,
            "h3_count",
        ).index.tolist()
    )

    idx = sorted(
        c_idx
        | h_idx
    )

    return df.loc[
        idx
    ].copy()



def restrict_od_to_zoom(
    df,
    zoom_bounds,
):
    """
    Restrict OD flows geographically BEFORE selecting the top flows.

    Primary rule:
        keep flows whose origin AND destination are inside the zoom.

    Fallback:
        if too few fully-local flows exist, keep flows with at least
        one endpoint inside the zoom.
    """
    if zoom_bounds is None:
        return df.copy(), "city-wide"

    xmin, xmax, ymin, ymax = zoom_bounds

    origin_inside = (
        df["origin_lng"].between(xmin, xmax)
        & df["origin_lat"].between(ymin, ymax)
    )

    destination_inside = (
        df["destination_lng"].between(xmin, xmax)
        & df["destination_lat"].between(ymin, ymax)
    )

    local = df[
        origin_inside
        & destination_inside
    ].copy()

    label = "local OD: both endpoints inside zoom"

    if len(local) < 12:
        local = df[
            origin_inside
            | destination_inside
        ].copy()

        label = "local OD: at least one endpoint inside zoom"

    return local, label


def plot_od_triptych(
    *,
    segments,
    bounds,
    diagnostics_dir,
    resolution,
    n_agents,
    top_k=100,
    zoom_bounds=None,
    suffix="city",
):
    path = (
        diagnostics_dir
        / f"od_flows_H{resolution}.csv"
    )

    df = pd.read_csv(
        path
    )

    df = df[
        (
            df[
                "continuous_count"
            ]
            > 0
        )
        | (
            df[
                "h3_count"
            ]
            > 0
        )
    ].copy()

    # For zoomed OD maps, spatial restriction happens BEFORE top-flow
    # selection. This prevents selecting Madrid-wide top flows that are
    # invisible inside the local zoom.
    df, od_scope_label = restrict_od_to_zoom(
        df,
        zoom_bounds,
    )

    if df.empty:
        raise ValueError(
            f"No OD flows available for the requested H{resolution} zoom."
        )

    df = select_top_od_union(
        df,
        top_k=min(top_k, len(df)),
    )

    line_segments = od_line_segments(
        df
    )

    common_max = max(
        float(
            df[
                "continuous_count"
            ].max()
        ),
        float(
            df[
                "h3_count"
            ].max()
        ),
        1.0,
    )

    diff_max = max(
        float(
            np.abs(
                df[
                    "difference"
                ].to_numpy(
                    dtype=float
                )
            ).max()
        ),
        1.0,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            18,
            6.2,
        ),
        constrained_layout=True,
    )

    for ax in axes:
        add_osm_basemap(
            ax,
            segments,
            bounds,
            zoom_bounds=zoom_bounds,
        )

    # Continuous
    c_values = df[
        "continuous_count"
    ].to_numpy(
        dtype=float
    )

    c_widths = (
        0.65
        + 6.0
        * np.sqrt(
            c_values
            / common_max
        )
    )

    c_lines = LineCollection(
        line_segments,
        linewidths=c_widths,
        alpha=0.82,
        zorder=4,
    )

    c_lines.set_array(
        c_values
    )

    c_lines.set_norm(
        PowerNorm(
            gamma=0.45,
            vmin=0.0,
            vmax=common_max,
        )
    )

    axes[
        0
    ].add_collection(
        c_lines
    )

    axes[
        0
    ].set_title(
        "Continuous-indexed OD"
    )

    # H3
    h_values = df[
        "h3_count"
    ].to_numpy(
        dtype=float
    )

    h_widths = (
        0.65
        + 6.0
        * np.sqrt(
            h_values
            / common_max
        )
    )

    h_lines = LineCollection(
        line_segments,
        linewidths=h_widths,
        alpha=0.82,
        zorder=4,
    )

    h_lines.set_array(
        h_values
    )

    h_lines.set_norm(
        PowerNorm(
            gamma=0.45,
            vmin=0.0,
            vmax=common_max,
        )
    )

    axes[
        1
    ].add_collection(
        h_lines
    )

    axes[
        1
    ].set_title(
        "H3-indexed OD"
    )

    # Difference
    difference = df[
        "difference"
    ].to_numpy(
        dtype=float
    )

    d_widths = (
        0.65
        + 6.0
        * np.sqrt(
            np.abs(
                difference
            )
            / diff_max
        )
    )

    d_lines = LineCollection(
        line_segments,
        linewidths=d_widths,
        alpha=0.86,
        zorder=4,
        cmap="coolwarm",
        norm=TwoSlopeNorm(
            vmin=-diff_max,
            vcenter=0,
            vmax=diff_max,
        ),
    )

    d_lines.set_array(
        difference
    )

    axes[
        2
    ].add_collection(
        d_lines
    )

    axes[
        2
    ].set_title(
        "H3 − Continuous OD"
    )

    if suffix == "city":
        scope_label = "city-wide"
    elif suffix == "manhattan":
        scope_label = "Manhattan zoom"
    elif suffix == "midtown_deep_zoom":
        scope_label = "Midtown deep zoom"
    elif suffix == "kendall":
        scope_label = "Kendall Square zoom"
    elif suffix == "volpe_deep_zoom":
        scope_label = "Volpe Center deep zoom"
    elif suffix in {"street_zoom", "deep_zoom"}:
        scope_label = "deep local zoom"
    else:
        scope_label = "local zoom"

    fig.suptitle(
        (
            f"Madrid OD flows — "
            f"H{resolution} aggregation, {scope_label}, N={n_agents:,}"
        ),
        fontsize=15,
    )

    fig.colorbar(
        c_lines,
        ax=[
            axes[
                0
            ],
            axes[
                1
            ],
        ],
        shrink=0.82,
        label="OD flow count",
    )

    fig.colorbar(
        d_lines,
        ax=axes[
            2
        ],
        shrink=0.82,
        label="H3 − Continuous flow",
    )

    output = (
        FIG_DIR
        / f"FIG_OD_H{resolution}_{suffix}.png"
    )

    fig.savefig(
        output,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output



# =====================================================================
# E. REPRESENTATIVE PHYSICAL OSM ROUTES
# =====================================================================

def _edge_geometry_points(G, u, v):
    """
    Return the geometry of the shortest available directed edge u->v.
    Coordinates are returned as (lng, lat).

    This is purely cartographic reconstruction over the common OSM graph.
    """
    edge_dict = G.get_edge_data(u, v)

    if not edge_dict:
        return np.asarray(
            [
                [float(G.nodes[u]["x"]), float(G.nodes[u]["y"])],
                [float(G.nodes[v]["x"]), float(G.nodes[v]["y"])],
            ],
            dtype=float,
        )

    best = min(
        edge_dict.values(),
        key=lambda d: float(d.get("length", 1e30)),
    )

    geometry = best.get("geometry")

    if geometry is not None:
        try:
            return np.asarray(
                list(geometry.coords),
                dtype=float,
            )
        except Exception:
            pass

    return np.asarray(
        [
            [float(G.nodes[u]["x"]), float(G.nodes[u]["y"])],
            [float(G.nodes[v]["x"]), float(G.nodes[v]["y"])],
        ],
        dtype=float,
    )


def _route_polyline(G, origin_node, destination_node):
    """
    Shortest physical route over the real OSM walk graph.
    """
    try:
        route = nx.shortest_path(
            G,
            source=origin_node,
            target=destination_node,
            weight="length",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    if len(route) < 2:
        return None

    pieces = []

    for u, v in zip(route[:-1], route[1:]):
        pts = _edge_geometry_points(
            G,
            u,
            v,
        )

        if pieces and len(pts):
            # Avoid duplicate edge-join coordinate where possible.
            if np.allclose(pieces[-1][-1], pts[0]):
                pts = pts[1:]

        if len(pts):
            pieces.append(pts)

    if not pieces:
        return None

    return np.vstack(pieces)


def _trip_od_key(row, architecture, resolution):
    prefix = (
        "continuous"
        if architecture == "continuous"
        else "h3"
    )

    return (
        getattr(row, f"{prefix}_origin_h{resolution}"),
        getattr(row, f"{prefix}_destination_h{resolution}"),
    )


def _aggregate_exact_trip_od(
    trips,
    architecture,
    resolution,
    zoom_bounds=None,
):
    prefix = (
        "continuous"
        if architecture == "continuous"
        else "h3"
    )

    olat = f"{prefix}_origin_lat"
    olng = f"{prefix}_origin_lng"
    dlat = f"{prefix}_destination_lat"
    dlng = f"{prefix}_destination_lng"
    oh = f"{prefix}_origin_h{resolution}"
    dh = f"{prefix}_destination_h{resolution}"

    df = trips.copy()

    if zoom_bounds is not None:
        xmin, xmax, ymin, ymax = zoom_bounds

        inside_o = (
            df[olng].between(xmin, xmax)
            & df[olat].between(ymin, ymax)
        )

        inside_d = (
            df[dlng].between(xmin, xmax)
            & df[dlat].between(ymin, ymax)
        )

        local = df[
            inside_o
            & inside_d
        ].copy()

        if len(local) < 12:
            local = df[
                inside_o
                | inside_d
            ].copy()

        df = local

    grouped = []

    for (origin_h, destination_h), g in df.groupby(
        [oh, dh],
        sort=False,
    ):
        # Real representative trip: choose the trip closest to the median
        # endpoints of the group, never an H3 centroid.
        endpoints = g[
            [
                olat,
                olng,
                dlat,
                dlng,
            ]
        ].to_numpy(
            dtype=float
        )

        target = np.median(
            endpoints,
            axis=0,
        )

        idx = int(
            np.argmin(
                np.sum(
                    (
                        endpoints
                        - target
                    )
                    ** 2,
                    axis=1,
                )
            )
        )

        representative = g.iloc[
            idx
        ]

        grouped.append(
            {
                "od_key": (
                    str(origin_h),
                    str(destination_h),
                ),
                "count": int(len(g)),
                "origin_lat": float(representative[olat]),
                "origin_lng": float(representative[olng]),
                "destination_lat": float(representative[dlat]),
                "destination_lng": float(representative[dlng]),
            }
        )

    return pd.DataFrame(grouped)


def _routed_union(
    G,
    diagnostics_dir,
    resolution,
    top_k,
    zoom_bounds=None,
):
    trip_path = (
        diagnostics_dir
        / "paired_trip_endpoints.csv"
    )

    if not trip_path.exists():
        raise FileNotFoundError(
            f"{trip_path} not found. "
            "Re-run 03_urban_fidelity.py once with the routed-OD diagnostics."
        )

    trips = pd.read_csv(
        trip_path
    )

    c = _aggregate_exact_trip_od(
        trips,
        "continuous",
        resolution,
        zoom_bounds=zoom_bounds,
    )

    h = _aggregate_exact_trip_od(
        trips,
        "h3",
        resolution,
        zoom_bounds=zoom_bounds,
    )

    if c.empty or h.empty:
        raise ValueError(
            f"No routed OD groups found for H{resolution}."
        )

    c_map = {
        tuple(row.od_key): row
        for row in c.itertuples(index=False)
    }

    h_map = {
        tuple(row.od_key): row
        for row in h.itertuples(index=False)
    }

    c_top = set(
        tuple(x)
        for x in c.nlargest(
            min(top_k, len(c)),
            "count",
        )[
            "od_key"
        ].tolist()
    )

    h_top = set(
        tuple(x)
        for x in h.nlargest(
            min(top_k, len(h)),
            "count",
        )[
            "od_key"
        ].tolist()
    )

    union_keys = sorted(
        c_top
        | h_top
    )

    rows = []

    for key in union_keys:
        cr = c_map.get(key)
        hr = h_map.get(key)

        rows.append(
            {
                "od_key": key,
                "continuous_count": int(cr.count) if cr is not None else 0,
                "h3_count": int(hr.count) if hr is not None else 0,
                "continuous_origin_lat": float(cr.origin_lat) if cr is not None else np.nan,
                "continuous_origin_lng": float(cr.origin_lng) if cr is not None else np.nan,
                "continuous_destination_lat": float(cr.destination_lat) if cr is not None else np.nan,
                "continuous_destination_lng": float(cr.destination_lng) if cr is not None else np.nan,
                "h3_origin_lat": float(hr.origin_lat) if hr is not None else np.nan,
                "h3_origin_lng": float(hr.origin_lng) if hr is not None else np.nan,
                "h3_destination_lat": float(hr.destination_lat) if hr is not None else np.nan,
                "h3_destination_lng": float(hr.destination_lng) if hr is not None else np.nan,
            }
        )

    union = pd.DataFrame(
        rows
    )

    # Pre-create route columns explicitly as object dtype.
    # Each cell will contain a NumPy array with shape (n_route_points, 2).
    # Without object columns, pandas may try to broadcast the array as if it
    # were a tabular value, which raises a ValueError.
    union["continuous_route"] = pd.Series(
        [None] * len(union),
        dtype=object,
    )

    union["h3_route"] = pd.Series(
        [None] * len(union),
        dtype=object,
    )

    # Batch snap the exact representative endpoints to the common OSM graph.
    for architecture in (
        "continuous",
        "h3",
    ):
        valid = union[
            [
                f"{architecture}_origin_lat",
                f"{architecture}_origin_lng",
                f"{architecture}_destination_lat",
                f"{architecture}_destination_lng",
            ]
        ].notna().all(
            axis=1
        )

        idx = union.index[
            valid
        ]

        if len(idx) == 0:
            continue

        origin_nodes = ox.distance.nearest_nodes(
            G,
            X=union.loc[
                idx,
                f"{architecture}_origin_lng",
            ].to_numpy(
                dtype=float
            ),
            Y=union.loc[
                idx,
                f"{architecture}_origin_lat",
            ].to_numpy(
                dtype=float
            ),
        )

        destination_nodes = ox.distance.nearest_nodes(
            G,
            X=union.loc[
                idx,
                f"{architecture}_destination_lng",
            ].to_numpy(
                dtype=float
            ),
            Y=union.loc[
                idx,
                f"{architecture}_destination_lat",
            ].to_numpy(
                dtype=float
            ),
        )

        for row_index, origin_node, destination_node in zip(
            idx,
            origin_nodes,
            destination_nodes,
        ):
            union.at[
                row_index,
                f"{architecture}_route",
            ] = _route_polyline(
                G,
                origin_node,
                destination_node,
            )

    return union



def compute_street_level_zoom(
    parent_zoom_bounds,
    width_fraction=0.46,
    height_fraction=0.46,
    center_x_fraction=0.50,
    center_y_fraction=0.48,
):
    """
    Create a DEEPER VIEW OF THE EXISTING H8 ROUTED-OD FIGURE.

    Important:
    - This does NOT re-select OD flows.
    - This does NOT recompute a new local OD sample.
    - It simply crops the already selected H8 local routed-flow figure.

    The crop is expressed as fractions of the parent H8 zoom window.
    """
    xmin, xmax, ymin, ymax = parent_zoom_bounds

    width = xmax - xmin
    height = ymax - ymin

    crop_width = width * float(width_fraction)
    crop_height = height * float(height_fraction)

    center_x = xmin + width * float(center_x_fraction)
    center_y = ymin + height * float(center_y_fraction)

    return (
        center_x - crop_width / 2.0,
        center_x + crop_width / 2.0,
        center_y - crop_height / 2.0,
        center_y + crop_height / 2.0,
    )


def plot_routed_od_triptych(
    *,
    G,
    segments,
    bounds,
    diagnostics_dir,
    resolution,
    n_agents,
    top_k=35,
    zoom_bounds=None,
    selection_bounds=None,
    display_bounds=None,
    suffix="city",
):
    # selection_bounds controls WHICH OD routes are selected.
    # display_bounds controls only WHAT PART of those same routes is shown.
    # This distinction is essential for the street-level figure, which is a
    # genuine crop/zoom of FIG_ROUTED_OD_H8_zoom rather than a new OD sample.
    if selection_bounds is None:
        selection_bounds = zoom_bounds

    if display_bounds is None:
        display_bounds = zoom_bounds

    routed = _routed_union(
        G,
        diagnostics_dir,
        resolution,
        top_k,
        zoom_bounds=selection_bounds,
    )

    common_max = max(
        float(routed["continuous_count"].max()),
        float(routed["h3_count"].max()),
        1.0,
    )

    difference = (
        routed["h3_count"]
        - routed["continuous_count"]
    ).to_numpy(
        dtype=float
    )

    diff_max = max(
        float(np.max(np.abs(difference))),
        1.0,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6.2),
        constrained_layout=True,
    )

    for ax in axes:
        add_osm_basemap(
            ax,
            segments,
            bounds,
            zoom_bounds=display_bounds,
        )

    collections = []

    for axis_index, architecture in enumerate(
        ("continuous", "h3")
    ):
        route_segments = []
        values = []

        for row in routed.itertuples(index=False):
            route = getattr(
                row,
                f"{architecture}_route",
                None,
            )

            count = float(
                getattr(
                    row,
                    f"{architecture}_count",
                )
            )

            if (
                isinstance(route, np.ndarray)
                and len(route) >= 2
                and count > 0
            ):
                route_segments.append(route)
                values.append(count)

        values = np.asarray(
            values,
            dtype=float,
        )

        # City-wide H7 should read as metropolitan corridors rather than
        # thick local route blobs. H8 zoom keeps stronger local detail.
        if resolution == 7 and display_bounds is None:
            widths = (
                0.35
                + 2.4
                * np.sqrt(
                    values
                    / common_max
                )
            )
            route_alpha = 0.58
        elif suffix in {"street_zoom", "deep_zoom"}:
            widths = (
                1.0
                + 7.0
                * np.sqrt(
                    values
                    / common_max
                )
            )
            route_alpha = 0.90
        else:
            widths = (
                0.8
                + 6.5
                * np.sqrt(
                    values
                    / common_max
                )
            )
            route_alpha = 0.84

        collection = LineCollection(
            route_segments,
            linewidths=widths,
            alpha=route_alpha,
            zorder=5,
            norm=PowerNorm(
                gamma=0.45,
                vmin=0.0,
                vmax=common_max,
            ),
        )

        collection.set_array(
            values
        )

        axes[
            axis_index
        ].add_collection(
            collection
        )

        axes[
            axis_index
        ].set_title(
            (
                f"Continuous-indexed physical routes (H{resolution} aggregation)"
                if architecture == "continuous"
                else f"H3-indexed physical routes (H{resolution} aggregation)"
            )
        )

        collections.append(
            collection
        )

    # Difference uses the Continuous representative route when available,
    # otherwise the H3 representative route. This panel encodes flow-count
    # difference, not a different physical-routing algorithm.
    diff_routes = []
    diff_values = []

    for row, diff in zip(
        routed.itertuples(index=False),
        difference,
    ):
        route = getattr(
            row,
            "continuous_route",
            None,
        )

        if not isinstance(route, np.ndarray):
            route = getattr(
                row,
                "h3_route",
                None,
            )

        if (
            isinstance(route, np.ndarray)
            and len(route) >= 2
            and abs(diff) > 0
        ):
            diff_routes.append(route)
            diff_values.append(diff)

    diff_values = np.asarray(
        diff_values,
        dtype=float,
    )

    if resolution == 7 and display_bounds is None:
        diff_widths = (
            0.35
            + 2.4
            * np.sqrt(
                np.abs(diff_values)
                / diff_max
            )
        )
        diff_alpha = 0.62
    elif suffix in {"street_zoom", "deep_zoom"}:
        diff_widths = (
            1.0
            + 7.0
            * np.sqrt(
                np.abs(diff_values)
                / diff_max
            )
        )
        diff_alpha = 0.90
    else:
        diff_widths = (
            0.8
            + 6.5
            * np.sqrt(
                np.abs(diff_values)
                / diff_max
            )
        )
        diff_alpha = 0.86

    diff_collection = LineCollection(
        diff_routes,
        linewidths=diff_widths,
        alpha=diff_alpha,
        zorder=5,
        cmap="coolwarm",
        norm=TwoSlopeNorm(
            vmin=-diff_max,
            vcenter=0.0,
            vmax=diff_max,
        ),
    )

    diff_collection.set_array(
        diff_values
    )

    axes[
        2
    ].add_collection(
        diff_collection
    )

    axes[
        2
    ].set_title(
        "OD-induced difference on shared OSM network"
    )

    if display_bounds is None:
        scope = "city-wide"
    elif suffix == "manhattan":
        scope = "Manhattan zoom"
    elif suffix == "midtown_deep_zoom":
        scope = "Midtown deep zoom"
    elif suffix == "kendall":
        scope = "Kendall Square zoom"
    elif suffix == "volpe_deep_zoom":
        scope = "Volpe Center deep zoom"
    elif suffix in {"street_zoom", "deep_zoom"}:
        scope = "deep local zoom"
    else:
        scope = "local zoom"

    fig.suptitle(
        (
            f"Madrid representative physical OSM routes — "
            f"H{resolution} OD aggregation, {scope}, N={n_agents:,}"
        ),
        fontsize=15,
    )

    if collections:
        fig.colorbar(
            collections[0],
            ax=[
                axes[0],
                axes[1],
            ],
            shrink=0.82,
            label="OD group count",
        )

    if len(diff_values):
        fig.colorbar(
            diff_collection,
            ax=axes[2],
            shrink=0.82,
            label="H3 − Continuous OD group count",
        )

    fig.text(
        0.5,
        0.012,
        (
            "Both architectures are projected onto the same OSM walking network; "
            "routing is not part of the architectural comparison."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    output = (
        FIG_DIR
        / f"FIG_ROUTED_OD_H{resolution}_{suffix}.png"
    )

    fig.savefig(
        output,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output



# =====================================================================
# E. Destination ECDF
# =====================================================================

def plot_destination_ecdf(
    diagnostics_dir,
    n_agents,
):
    path = (
        diagnostics_dir
        / "destination_separations.csv"
    )

    df = pd.read_csv(
        path
    )

    x = np.sort(
        df[
            "destination_separation_m"
        ].dropna().to_numpy(
            dtype=float
        )
    )

    y = (
        np.arange(
            1,
            len(
                x
            )
            + 1,
        )
        / len(
            x
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            8.6,
            5.5,
        )
    )

    ax.plot(
        x,
        y,
        linewidth=2,
    )

    for threshold in (
        100,
        250,
        500,
        1000,
    ):
        ax.axvline(
            threshold,
            linestyle="--",
            alpha=0.45,
        )

    p995 = float(
        np.percentile(
            x,
            99.5,
        )
    )

    ax.set_xlim(
        0,
        p995,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_xlabel(
        "Continuous–H3 destination separation (m)"
    )

    ax.set_ylabel(
        "Empirical cumulative probability"
    )

    ax.set_title(
        f"Local destination substitution — N={n_agents:,}"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    output = (
        FIG_DIR
        / "FIG_destination_separation_ECDF.png"
    )

    fig.savefig(
        output,
        dpi=240,
    )

    plt.close(
        fig
    )

    return output


# =====================================================================
# F. Summary CSV
# =====================================================================

def build_headline_table(
    architecture,
    fidelity,
    n_fidelity,
):
    largest_n = int(
        architecture[
            "n_agents"
        ].max()
    )

    arch_max = architecture[
        architecture[
            "n_agents"
        ]
        == largest_n
    ]

    rows = []

    for operation in (
        "pure_spatial_lookup",
        "urban_context_retrieval",
        "accessibility_query",
        "candidate_retrieval",
        "full_agent_decision",
    ):
        q = arch_max[
            arch_max[
                "operation"
            ]
            == operation
        ]

        if not q.empty:
            rows.append(
                {
                    "result_type":
                        "architecture",
                    "observable":
                        operation,
                    "scale":
                        f"N={largest_n}",
                    "primary_metric":
                        "speedup_x",
                    "value":
                        float(
                            q.iloc[
                                0
                            ][
                                "speedup_h3_over_continuous"
                            ]
                        ),
                }
            )

    for metric in (
        "accessibility",
        "od_flow",
        "occupancy",
        "amenity_usage",
    ):
        q = fidelity[
            fidelity[
                "metric"
            ]
            == metric
        ]

        for _, row in q.iterrows():
            rows.append(
                {
                    "result_type":
                        "fidelity",
                    "observable":
                        metric,
                    "scale":
                        str(
                            row[
                                "scale"
                            ]
                        ),
                    "primary_metric":
                        "pearson",
                    "value":
                        float(
                            row[
                                "pearson"
                            ]
                        ),
                }
            )

    output = pd.DataFrame(
        rows
    )

    output[
        "fidelity_population"
    ] = int(
        n_fidelity
    )

    output.to_csv(
        FIG_DIR
        / "headline_results.csv",
        index=False,
    )

    return output


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--diagnostics-agents",
        type=int,
        default=None,
        help=(
            "Population whose diagnostics_n* directory should be used. "
            "Defaults to the largest diagnostics population available."
        ),
    )

    parser.add_argument(
        "--heatmap-resolution",
        type=int,
        default=8,
        choices=[
            7,
            8,
            9,
        ],
    )

    parser.add_argument(
        "--od-resolution",
        type=int,
        default=7,
        choices=[
            7,
            8,
            9,
        ],
    )

    parser.add_argument(
        "--od-top-k",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    ensure_dirs()

    architecture = load_architecture()

    fidelity, n_fidelity_summary = (
        load_largest_fidelity()
    )

    diagnostics_dir, n_diagnostics = (
        choose_diagnostics_population(
            args.diagnostics_agents
        )
    )

    print(
        f"[DATA] Diagnostics population: "
        f"{n_diagnostics:,}"
    )

    print(
        f"[DATA] Fidelity summary population: "
        f"{n_fidelity_summary:,}"
    )

    G = load_graph()

    print(
        "[MAP] Preparing road geometry..."
    )

    segments = graph_segments(
        G
    )

    bounds = graph_bounds(
        G
    )

    outputs = []

    outputs.append(
        plot_speedup_scalability(
            architecture
        )
    )

    outputs.append(
        plot_full_decision_scalability(
            architecture
        )
    )

    outputs.append(
        plot_fidelity_summary(
            fidelity,
            n_fidelity_summary,
        )
    )

    outputs.append(
        plot_madrid_heatmap_triptych(
            G=G,
            segments=segments,
            bounds=bounds,
            diagnostics_dir=diagnostics_dir,
            metric_name="visit_density",
            resolution=args.heatmap_resolution,
            n_agents=n_diagnostics,
            suffix="city",
        )
    )

    outputs.append(
        plot_madrid_heatmap_triptych(
            G=G,
            segments=segments,
            bounds=bounds,
            diagnostics_dir=diagnostics_dir,
            metric_name="occupancy",
            resolution=args.heatmap_resolution,
            n_agents=n_diagnostics,
            suffix="city",
        )
    )

    zoom_bounds = compute_activity_zoom(
        diagnostics_dir,
        resolution=args.heatmap_resolution,
    )

    outputs.append(
        plot_madrid_heatmap_triptych(
            G=G,
            segments=segments,
            bounds=bounds,
            diagnostics_dir=diagnostics_dir,
            metric_name="visit_density",
            resolution=args.heatmap_resolution,
            n_agents=n_diagnostics,
            zoom_bounds=zoom_bounds,
            suffix="zoom",
        )
    )

    outputs.append(
        plot_od_triptych(
            segments=segments,
            bounds=bounds,
            diagnostics_dir=diagnostics_dir,
            resolution=args.od_resolution,
            n_agents=n_diagnostics,
            top_k=args.od_top_k,
            suffix="city",
        )
    )

    outputs.append(
        plot_od_triptych(
            segments=segments,
            bounds=bounds,
            diagnostics_dir=diagnostics_dir,
            resolution=8,
            n_agents=n_diagnostics,
            top_k=min(
                args.od_top_k,
                150,
            ),
            zoom_bounds=zoom_bounds,
            suffix="zoom",
        )
    )

    # Representative physical routes over the real OSM network.
    # These use exact paired trip endpoints exported by the fidelity run.
    outputs.append(
        plot_routed_od_triptych(
            G=G,
            segments=segments,
            bounds=bounds,
            diagnostics_dir=diagnostics_dir,
            resolution=7,
            n_agents=n_diagnostics,
            top_k=35,
            suffix="city",
        )
    )

    outputs.append(
        plot_routed_od_triptych(
            G=G,
            segments=segments,
            bounds=bounds,
            diagnostics_dir=diagnostics_dir,
            resolution=8,
            n_agents=n_diagnostics,
            top_k=30,
            zoom_bounds=zoom_bounds,
            suffix="zoom",
        )
    )

    # TRUE DEEP ZOOM of FIG_ROUTED_OD_H8_zoom:
    # Keep EXACTLY the same H8 local OD selection and the same OSM routes.
    # Only the displayed geographic extent is cropped more deeply.
    # Bounds follow the plotting convention: (xmin, xmax, ymin, ymax).
    deep_zoom_bounds = (
        -3.5835,  # west / xmin
        -3.5755,  # east / xmax
        40.4685,  # south / ymin
        40.4755,  # north / ymax
    )

    outputs.append(
        plot_routed_od_triptych(
            G=G,
            segments=segments,
            bounds=bounds,
            diagnostics_dir=diagnostics_dir,
            resolution=8,
            n_agents=n_diagnostics,
            top_k=30,
            selection_bounds=zoom_bounds,
            display_bounds=deep_zoom_bounds,
            suffix="deep_zoom",
        )
    )

    outputs.append(
        plot_destination_ecdf(
            diagnostics_dir,
            n_diagnostics,
        )
    )

    headline = build_headline_table(
        architecture,
        fidelity,
        n_fidelity_summary,
    )

    print()
    print("=" * 96)
    print(
        "FINAL SCIENTIFIC FIGURES"
    )
    print("=" * 96)

    for path in outputs:
        print(
            f"  {path}"
        )

    print()
    print(
        headline.to_string(
            index=False
        )
    )

    print()
    print(
        f"All figures saved in: {FIG_DIR}"
    )


if __name__ == "__main__":
    main()
