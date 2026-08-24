#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h3uab.city_config import config_for_city
from h3uab.osm_data import load_osm_pois


CITY_SPECS = {
    "madrid": {
        "label": "Madrid",
        "arch_dir": ROOT / "outputs" / "architecture",
        "fid_dir": ROOT / "outputs" / "urban_fidelity",
    },
    "cambridge_ma": {
        "label": "Cambridge",
        "arch_dir": ROOT / "outputs" / "cambridge_ma" / "architecture",
        "fid_dir": ROOT / "outputs" / "cambridge_ma" / "urban_fidelity",
    },
    "new_york_ny": {
        "label": "New York City",
        "arch_dir": ROOT / "outputs" / "new_york_ny" / "architecture",
        "fid_dir": ROOT / "outputs" / "new_york_ny" / "urban_fidelity",
    },
}

OPERATIONS = [
    "pure_spatial_lookup",
    "urban_context_retrieval",
    "accessibility_query",
    "candidate_retrieval",
    "full_agent_decision",
]

OP_LABELS = {
    "pure_spatial_lookup": "Spatial lookup",
    "urban_context_retrieval": "Urban context",
    "accessibility_query": "Accessibility",
    "candidate_retrieval": "Candidate retrieval",
    "full_agent_decision": "Full decision",
}


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_architecture(spec: dict) -> pd.DataFrame:
    path = require(
        spec["arch_dir"] / "architecture_sweep_summary.csv"
    )
    df = pd.read_csv(path)

    required = {
        "n_agents",
        "operation",
        "continuous_us_per_query",
        "h3_us_per_query",
        "speedup_h3_over_continuous",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path} missing columns: {sorted(missing)}"
        )

    return df.copy()


def load_fidelity(spec: dict) -> pd.DataFrame:
    preferred = spec["fid_dir"] / "urban_fidelity_n10000.csv"

    if preferred.exists():
        return pd.read_csv(preferred)

    files = sorted(
        spec["fid_dir"].glob("urban_fidelity_n*.csv"),
        key=lambda p: int(p.stem.split("_n")[-1]),
    )
    if not files:
        raise FileNotFoundError(
            f"No urban_fidelity_n*.csv in {spec['fid_dir']}"
        )

    return pd.read_csv(files[-1])


def fidelity_value(
    df: pd.DataFrame,
    metric: str,
    scale: str,
    column: str,
) -> float:
    row = df[
        (df["metric"].astype(str) == metric)
        & (df["scale"].astype(str) == scale)
    ]

    if row.empty or column not in row.columns:
        return np.nan

    value = pd.to_numeric(
        row.iloc[0][column],
        errors="coerce",
    )
    return float(value) if pd.notna(value) else np.nan


def load_city_scale(city_key: str) -> dict:
    """
    Derive city-scale metadata from the exact same files used by the model.
    """
    cfg = config_for_city(
        city_key,
        n_agents=10_000,
    )

    pois = load_osm_pois(cfg)
    pop = pd.read_csv(cfg.population_h3_csv)

    population = pd.to_numeric(
        pop[cfg.population_col],
        errors="coerce",
    ).fillna(0.0)

    raw_poi_count = np.nan
    try:
        raw_poi_count = len(pd.read_csv(cfg.osm_cache_csv))
    except Exception:
        pass

    return {
        "raw_pois": raw_poi_count,
        "semantic_pois": int(len(pois)),
        "population_h3_cells": int(len(pop)),
        "population_weight": float(population.sum()),
    }


def architecture_at_n(
    df: pd.DataFrame,
    operation: str,
    n_agents: int,
    column: str,
) -> float:
    row = df[
        (pd.to_numeric(df["n_agents"], errors="coerce") == n_agents)
        & (df["operation"].astype(str) == operation)
    ]

    if row.empty:
        return np.nan

    value = pd.to_numeric(
        row.iloc[0][column],
        errors="coerce",
    )
    return float(value) if pd.notna(value) else np.nan


def max_architecture_n(df: pd.DataFrame) -> int:
    return int(
        pd.to_numeric(
            df["n_agents"],
            errors="coerce",
        ).dropna().max()
    )


def build_master_table(
    architectures: dict[str, pd.DataFrame],
    fidelities: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for city_key, spec in CITY_SPECS.items():
        arch = architectures[city_key]
        fid = fidelities[city_key]
        scale = load_city_scale(city_key)

        n_max = max_architecture_n(arch)

        row = {
            "city_key": city_key,
            "city": spec["label"],
            "raw_pois": scale["raw_pois"],
            "semantic_pois": scale["semantic_pois"],
            "population_h3_cells": scale["population_h3_cells"],
            "population_weight": scale["population_weight"],
            "architecture_max_agents": n_max,

            "spatial_lookup_speedup_x": architecture_at_n(
                arch, "pure_spatial_lookup", n_max,
                "speedup_h3_over_continuous",
            ),
            "urban_context_speedup_x": architecture_at_n(
                arch, "urban_context_retrieval", n_max,
                "speedup_h3_over_continuous",
            ),
            "accessibility_speedup_x": architecture_at_n(
                arch, "accessibility_query", n_max,
                "speedup_h3_over_continuous",
            ),
            "candidate_retrieval_speedup_x": architecture_at_n(
                arch, "candidate_retrieval", n_max,
                "speedup_h3_over_continuous",
            ),
            "full_decision_speedup_x": architecture_at_n(
                arch, "full_agent_decision", n_max,
                "speedup_h3_over_continuous",
            ),

            "accessibility_pearson": fidelity_value(
                fid, "accessibility", "decision", "pearson"
            ),
            "destination_exact_match": fidelity_value(
                fid, "destination", "decision", "primary_value"
            ),
            "destination_within_100m": fidelity_value(
                fid, "destination", "decision", "within_100m"
            ),
            "destination_within_250m": fidelity_value(
                fid, "destination", "decision", "within_250m"
            ),
            "destination_within_500m": fidelity_value(
                fid, "destination", "decision", "within_500m"
            ),
            "destination_within_1000m": fidelity_value(
                fid, "destination", "decision", "within_1000m"
            ),
            "trip_distance_pearson": fidelity_value(
                fid, "trip_distance", "decision", "pearson"
            ),

            "od_h7_pearson": fidelity_value(
                fid, "od_flow", "H7", "pearson"
            ),
            "od_h8_pearson": fidelity_value(
                fid, "od_flow", "H8", "pearson"
            ),
            "od_h9_pearson": fidelity_value(
                fid, "od_flow", "H9", "pearson"
            ),

            "occupancy_h7_pearson": fidelity_value(
                fid, "occupancy", "H7", "pearson"
            ),
            "occupancy_h8_pearson": fidelity_value(
                fid, "occupancy", "H8", "pearson"
            ),
            "occupancy_h9_pearson": fidelity_value(
                fid, "occupancy", "H9", "pearson"
            ),

            "amenity_usage_pearson": fidelity_value(
                fid, "amenity_usage", "city", "pearson"
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def build_architecture_long(
    architectures: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    frames = []

    for city_key, spec in CITY_SPECS.items():
        df = architectures[city_key].copy()
        df["city_key"] = city_key
        df["city"] = spec["label"]
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def build_fidelity_long(
    fidelities: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    frames = []

    for city_key, spec in CITY_SPECS.items():
        df = fidelities[city_key].copy()
        df["city_key"] = city_key
        df["city"] = spec["label"]
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def save_fig_cc1_speedup_scaling(
    architecture_long: pd.DataFrame,
    fig_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    for city in architecture_long["city"].drop_duplicates():
        city_df = architecture_long[
            architecture_long["city"] == city
        ]

        for operation in OPERATIONS:
            df = city_df[
                city_df["operation"] == operation
            ].sort_values("n_agents")

            if df.empty:
                continue

            ax.plot(
                df["n_agents"],
                df["speedup_h3_over_continuous"],
                marker="o",
                linewidth=1.6,
                label=f"{city} — {OP_LABELS[operation]}",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Simulated agents / benchmark queries")
    ax.set_ylabel("H3 speedup over continuous-indexed baseline (×)")
    ax.set_title(
        "Cross-city computational scaling of the hierarchical spatial architecture"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()

    path = fig_dir / "FIG_CC1_speedup_scaling.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_fig_cc2_full_decision_scaling(
    architecture_long: pd.DataFrame,
    fig_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    df = architecture_long[
        architecture_long["operation"]
        == "full_agent_decision"
    ].copy()

    for city in df["city"].drop_duplicates():
        city_df = df[
            df["city"] == city
        ].sort_values("n_agents")

        ax.plot(
            city_df["n_agents"],
            city_df["continuous_us_per_query"],
            marker="o",
            linewidth=1.8,
            label=f"{city} — Continuous",
        )

        ax.plot(
            city_df["n_agents"],
            city_df["h3_us_per_query"],
            marker="s",
            linewidth=1.8,
            linestyle="--",
            label=f"{city} — H3",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Simulated agents / benchmark queries")
    ax.set_ylabel("Full agent decision latency (µs/query)")
    ax.set_title("Cross-city full-agent decision latency")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout()

    path = fig_dir / "FIG_CC2_full_decision_scaling.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_fig_cc3_fidelity_resolution(
    master: pd.DataFrame,
    fig_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10.0, 6.3))

    resolutions = np.array([7, 8, 9], dtype=int)

    for row in master.itertuples(index=False):
        od = np.array(
            [
                row.od_h7_pearson,
                row.od_h8_pearson,
                row.od_h9_pearson,
            ],
            dtype=float,
        )

        occ = np.array(
            [
                row.occupancy_h7_pearson,
                row.occupancy_h8_pearson,
                row.occupancy_h9_pearson,
            ],
            dtype=float,
        )

        ax.plot(
            resolutions,
            od,
            marker="o",
            linewidth=2,
            label=f"{row.city} — OD flow",
        )

        ax.plot(
            resolutions,
            occ,
            marker="s",
            linewidth=2,
            linestyle="--",
            label=f"{row.city} — Occupancy",
        )

    ax.set_xticks(resolutions, ["H7", "H8", "H9"])
    ax.set_ylim(0.55, 1.01)
    ax.set_xlabel("Urban analytical resolution")
    ax.set_ylabel("Pearson fidelity")
    ax.set_title(
        "Resolution-dependent preservation of urban dynamics across cities"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout()

    path = fig_dir / "FIG_CC3_fidelity_resolution.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_fig_cc4_accessibility_tradeoff(
    master: pd.DataFrame,
    fig_dir: Path,
) -> Path:
    """
    Accessibility-specific performance–fidelity comparison.

    X = accessibility-query speedup at the maximum architecture population.
    Y = accessibility Pearson fidelity in the controlled N=10,000 run.
    """
    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    for row in master.itertuples(index=False):
        ax.scatter(
            row.accessibility_speedup_x,
            row.accessibility_pearson,
            s=90,
        )

        ax.annotate(
            row.city,
            (
                row.accessibility_speedup_x,
                row.accessibility_pearson,
            ),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_xlabel(
        "Accessibility-query speedup at maximum benchmark population (×)"
    )
    ax.set_ylabel(
        "Accessibility fidelity at N=10,000 (Pearson r)"
    )
    ax.set_title(
        "Cross-city accessibility: computational gain versus fidelity"
    )

    ymin = max(
        0.90,
        float(master["accessibility_pearson"].min()) - 0.02,
    )
    ax.set_ylim(ymin, 1.002)
    ax.grid(alpha=0.25)

    fig.tight_layout()

    path = (
        fig_dir
        / "FIG_CC4_accessibility_speedup_vs_fidelity.png"
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    out_dir = ROOT / "outputs" / "cross_city"
    fig_dir = out_dir / "paper_figures"

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    architectures = {
        key: load_architecture(spec)
        for key, spec in CITY_SPECS.items()
    }

    fidelities = {
        key: load_fidelity(spec)
        for key, spec in CITY_SPECS.items()
    }

    architecture_long = build_architecture_long(
        architectures
    )

    fidelity_long = build_fidelity_long(
        fidelities
    )

    master = build_master_table(
        architectures,
        fidelities,
    )

    architecture_long.to_csv(
        out_dir / "master_architecture_results.csv",
        index=False,
    )

    fidelity_long.to_csv(
        out_dir / "master_fidelity_results.csv",
        index=False,
    )

    master.to_csv(
        out_dir / "master_cross_city_results.csv",
        index=False,
    )

    paper_columns = [
        "city",
        "semantic_pois",
        "population_h3_cells",
        "population_weight",
        "spatial_lookup_speedup_x",
        "urban_context_speedup_x",
        "accessibility_speedup_x",
        "candidate_retrieval_speedup_x",
        "full_decision_speedup_x",
        "accessibility_pearson",
        "destination_exact_match",
        "destination_within_500m",
        "destination_within_1000m",
        "od_h7_pearson",
        "od_h8_pearson",
        "od_h9_pearson",
        "occupancy_h7_pearson",
        "occupancy_h8_pearson",
        "occupancy_h9_pearson",
        "amenity_usage_pearson",
    ]

    paper_table = master[paper_columns].copy()

    paper_table.to_csv(
        out_dir / "TABLE_1_cross_city_headline_results.csv",
        index=False,
    )

    figures = [
        save_fig_cc1_speedup_scaling(
            architecture_long,
            fig_dir,
        ),
        save_fig_cc2_full_decision_scaling(
            architecture_long,
            fig_dir,
        ),
        save_fig_cc3_fidelity_resolution(
            master,
            fig_dir,
        ),
        save_fig_cc4_accessibility_tradeoff(
            master,
            fig_dir,
        ),
    ]

    print("=" * 120)
    print("MASTER CROSS-CITY PAPER ANALYSIS")
    print("=" * 120)
    print()
    print("HEADLINE TABLE")
    print("-" * 120)

    print(
        paper_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Saved tables:")
    print(f"  {out_dir / 'master_cross_city_results.csv'}")
    print(f"  {out_dir / 'master_architecture_results.csv'}")
    print(f"  {out_dir / 'master_fidelity_results.csv'}")
    print(
        f"  {out_dir / 'TABLE_1_cross_city_headline_results.csv'}"
    )

    print()
    print("Saved figures:")
    for path in figures:
        print(f"  {path}")


if __name__ == "__main__":
    main()
