#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import networkx as nx
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
        "e2e_dir": ROOT / "outputs" / "end_to_end",
    },
    "cambridge_ma": {
        "label": "Cambridge",
        "arch_dir": ROOT / "outputs" / "cambridge_ma" / "architecture",
        "fid_dir": ROOT / "outputs" / "cambridge_ma" / "urban_fidelity",
        "e2e_dir": ROOT / "outputs" / "cambridge_ma" / "end_to_end",
    },
    "new_york_ny": {
        "label": "New York City",
        "arch_dir": ROOT / "outputs" / "new_york_ny" / "architecture",
        "fid_dir": ROOT / "outputs" / "new_york_ny" / "urban_fidelity",
        "e2e_dir": ROOT / "outputs" / "new_york_ny" / "end_to_end",
    },
}


PERFORMANCE_OPERATIONS = [
    "pure_spatial_lookup",
    "urban_context_retrieval",
    "accessibility_query",
    "candidate_retrieval",
    "full_agent_decision",
]


PERFORMANCE_LABELS = {
    "pure_spatial_lookup": "Spatial lookup",
    "urban_context_retrieval": "Urban context retrieval",
    "accessibility_query": "Accessibility query",
    "candidate_retrieval": "Candidate retrieval",
    "full_agent_decision": "Full agent decision",
}


E2E_TARGET_POPULATIONS = (10_000, 100_000)
DECISIONS_PER_AGENT = 5.0



def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_architecture(city_key: str) -> pd.DataFrame:
    spec = CITY_SPECS[city_key]
    path = require(
        spec["arch_dir"] / "architecture_sweep_summary.csv"
    )
    df = pd.read_csv(path)

    required = {
        "n_agents",
        "operation",
        "continuous_us_per_query",
        "h3_us_per_query",
        "continuous_qps",
        "h3_qps",
        "speedup_h3_over_continuous",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path} missing required columns: {sorted(missing)}"
        )

    return df.copy()


def load_fidelity(city_key: str) -> pd.DataFrame:
    spec = CITY_SPECS[city_key]

    preferred = (
        spec["fid_dir"]
        / "urban_fidelity_n10000.csv"
    )

    if preferred.exists():
        return pd.read_csv(preferred)

    candidates = sorted(
        spec["fid_dir"].glob(
            "urban_fidelity_n*.csv"
        ),
        key=lambda p: int(
            p.stem.split("_n")[-1]
        ),
    )

    if not candidates:
        raise FileNotFoundError(
            f"No fidelity summary found in {spec['fid_dir']}"
        )

    return pd.read_csv(candidates[-1])



def load_end_to_end(city_key: str) -> pd.DataFrame:
    """
    Load temporal end-to-end summary results.

    Preferred reproducible layout:
        end_to_end_summary_both_n10000.csv
        end_to_end_summary_both_n100000.csv

    Backward-compatible fallback:
        end_to_end_summary_both.csv

    IMPORTANT:
    Older benchmark versions overwrote the same city-level file when a new
    population was run. If only one N is present in the fallback file, this
    function cannot reconstruct the missing N. In that case the script prints
    a warning and Table 4 will contain only the available population.
    """
    e2e_dir = CITY_SPECS[city_key]["e2e_dir"]

    per_n_files = []
    for n in E2E_TARGET_POPULATIONS:
        p = e2e_dir / f"end_to_end_summary_both_n{n}.csv"
        if p.exists():
            per_n_files.append(p)

    frames = []

    if per_n_files:
        for p in per_n_files:
            df = pd.read_csv(p)
            df["_source_file"] = str(p)
            frames.append(df)
    else:
        fallback = e2e_dir / "end_to_end_summary_both.csv"
        if not fallback.exists():
            raise FileNotFoundError(
                f"No end-to-end summary found for {city_key}: {fallback}"
            )
        df = pd.read_csv(fallback)
        df["_source_file"] = str(fallback)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    out["n_agents"] = pd.to_numeric(
        out["n_agents"],
        errors="coerce",
    )

    out = out[
        out["n_agents"].isin(E2E_TARGET_POPULATIONS)
    ].copy()

    out = (
        out.sort_values(["n_agents", "_source_file"])
        .drop_duplicates(subset=["n_agents"], keep="last")
        .reset_index(drop=True)
    )

    available = {
        int(x)
        for x in out["n_agents"].dropna().tolist()
    }
    missing = [
        n for n in E2E_TARGET_POPULATIONS
        if n not in available
    ]

    if missing:
        print(
            f"[WARNING] {CITY_SPECS[city_key]['label']}: "
            f"missing end-to-end populations {missing}. "
            "Older outputs may have been overwritten. "
            "Prefer per-N summary files."
        )

    return out


def _num(row: pd.Series, column: str) -> float:
    if column not in row.index:
        return np.nan
    value = pd.to_numeric(
        row[column],
        errors="coerce",
    )
    return float(value) if pd.notna(value) else np.nan


def build_end_to_end_table(
    end_to_end: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Long-format temporal end-to-end table.

    Break-even decisions are the primary amortization metric.
    Equivalent agents are a secondary interpretation under the controlled
    five-decision-per-agent schedule used in this experiment.
    """
    rows = []

    for city_key, spec in CITY_SPECS.items():
        df = end_to_end[city_key]

        for n in E2E_TARGET_POPULATIONS:
            sub = df[
                pd.to_numeric(
                    df["n_agents"],
                    errors="coerce",
                )
                == n
            ]

            if sub.empty:
                continue

            r = sub.iloc[-1]

            cold_be = _num(
                r,
                "cold_break_even_decisions",
            )
            warm_be = _num(
                r,
                "warm_break_even_decisions",
            )

            rows.append(
                {
                    "city_key": city_key,
                    "city": spec["label"],
                    "n_agents": int(n),
                    "decisions_per_agent":
                        _num(r, "decisions_per_agent")
                        if pd.notna(_num(r, "decisions_per_agent"))
                        else DECISIONS_PER_AGENT,
                    "total_decisions":
                        int(n * DECISIONS_PER_AGENT),

                    "repeats":
                        int(_num(r, "repeats"))
                        if pd.notna(_num(r, "repeats"))
                        else np.nan,

                    # Primary temporal runtime result.
                    "temporal_speedup_paired_median":
                        _num(r, "runtime_speedup_median")
                        if pd.notna(_num(r, "runtime_speedup_median"))
                        else _num(
                            r,
                            "runtime_speedup_h3_over_continuous",
                        ),
                    "temporal_speedup_min":
                        _num(r, "runtime_speedup_min"),
                    "temporal_speedup_max":
                        _num(r, "runtime_speedup_max"),
                    "continuous_runtime_s_median":
                        _num(r, "continuous_runtime_s_median"),
                    "h3_runtime_s_median":
                        _num(r, "h3_runtime_s_median"),

                    # Cold-start preparation / total.
                    "continuous_prepare_s_median":
                        _num(r, "continuous_prepare_s_median"),
                    "h3_cold_prepare_s_median":
                        _num(r, "h3_cold_prepare_s_median"),
                    "cold_total_speedup":
                        _num(
                            r,
                            "cold_total_speedup_profile_h3_over_continuous",
                        ),
                    "cold_break_even_decisions":
                        cold_be,
                    "cold_break_even_decisions_low":
                        _num(r, "cold_break_even_decisions_low"),
                    "cold_break_even_decisions_high":
                        _num(r, "cold_break_even_decisions_high"),
                    "cold_break_even_equivalent_agents":
                        (
                            cold_be / DECISIONS_PER_AGENT
                            if pd.notna(cold_be)
                            else np.nan
                        ),

                    # Warm-cache operational regime.
                    "h3_warm_prepare_s_median":
                        _num(r, "h3_warm_prepare_s_median"),
                    "warm_total_speedup":
                        _num(
                            r,
                            "warm_total_speedup_h3_over_continuous",
                        ),
                    "warm_break_even_decisions":
                        warm_be,
                    "warm_break_even_decisions_low":
                        _num(r, "warm_break_even_decisions_low"),
                    "warm_break_even_decisions_high":
                        _num(r, "warm_break_even_decisions_high"),
                    "warm_break_even_equivalent_agents":
                        (
                            warm_be / DECISIONS_PER_AGENT
                            if pd.notna(warm_be)
                            else np.nan
                        ),

                    "_source_file":
                        str(r.get("_source_file", "")),
                }
            )

    return pd.DataFrame(rows)



def validate_end_to_end_invariants(
    e2e_table: pd.DataFrame,
):
    """
    Fail loudly on arithmetic inconsistencies.

    When temporal, cold-total, and warm-total speedups are all summarized
    as paired-median repeat ratios, adding non-negative preparation cost
    cannot make total speedup exceed the runtime-only speedup.
    """
    if e2e_table.empty:
        return

    checks = [
        (
            "cold_total_speedup",
            "cold_total > temporal",
        ),
        (
            "warm_total_speedup",
            "warm_total > temporal",
        ),
    ]

    for column, label in checks:
        if column not in e2e_table.columns:
            continue

        mask = (
            pd.to_numeric(
                e2e_table[column],
                errors="coerce",
            )
            >
            pd.to_numeric(
                e2e_table[
                    "temporal_speedup_paired_median"
                ],
                errors="coerce",
            )
            + 1e-9
        )

        viol = e2e_table.loc[
            mask,
            [
                "city",
                "n_agents",
                "temporal_speedup_paired_median",
                column,
            ],
        ]

        if not viol.empty:
            raise ValueError(
                f"{label} in end-to-end table:\\n"
                + viol.to_string(index=False)
                + "\\n"
                "This usually indicates inconsistent aggregation "
                "(e.g. paired-median runtime speedup but ratio-of-medians "
                "total speedup) or stale legacy summaries."
            )


def save_end_to_end_paper_table(
    e2e_table: pd.DataFrame,
    out_dir: Path,
):
    """
    Table 4 for the manuscript.

    Decisions are deliberately primary. Equivalent agents are included only
    as a schedule-specific secondary interpretation.
    """
    cols = [
        "city",
        "n_agents",
        "repeats",
        "total_decisions",
        "temporal_speedup_paired_median",
        "temporal_speedup_min",
        "temporal_speedup_max",
        "cold_total_speedup",
        "cold_break_even_decisions",
        "cold_break_even_decisions_low",
        "cold_break_even_decisions_high",
        "cold_break_even_equivalent_agents",
        "warm_total_speedup",
        "warm_break_even_decisions",
        "warm_break_even_decisions_low",
        "warm_break_even_decisions_high",
        "warm_break_even_equivalent_agents",
    ]

    existing = [
        c for c in cols
        if c in e2e_table.columns
    ]

    e2e_table[
        existing
    ].to_csv(
        out_dir
        / "TABLE_4_end_to_end_temporal_amortization.csv",
        index=False,
    )


def plot_temporal_speedup(
    e2e_table: pd.DataFrame,
    out_dir: Path,
):
    """
    Paper figure: paired-median temporal-loop speedup by city and N.
    Error bars show the observed min/max range, not a confidence interval.
    """
    fig_dir = out_dir / "paper_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    cities = list(CITY_SPECS.values())
    city_labels = [x["label"] for x in cities]
    x = np.arange(len(city_labels), dtype=float)
    width = 0.34

    for j, n in enumerate(E2E_TARGET_POPULATIONS):
        vals = []
        lower = []
        upper = []

        for spec in CITY_SPECS.values():
            row = e2e_table[
                (e2e_table["city"] == spec["label"])
                & (e2e_table["n_agents"] == n)
            ]

            if row.empty:
                vals.append(np.nan)
                lower.append(0.0)
                upper.append(0.0)
                continue

            r = row.iloc[0]
            med = float(
                r["temporal_speedup_paired_median"]
            )
            lo = float(
                r["temporal_speedup_min"]
            )
            hi = float(
                r["temporal_speedup_max"]
            )

            vals.append(med)
            lower.append(max(0.0, med - lo))
            upper.append(max(0.0, hi - med))

        offset = (
            -width / 2
            if j == 0
            else width / 2
        )

        ax.bar(
            x + offset,
            vals,
            width,
            label=f"N={n:,}",
            yerr=np.array([lower, upper]),
            capsize=4,
        )

    ax.axhline(
        1.0,
        linewidth=1.0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(city_labels)
    ax.set_ylabel(
        "H3 temporal-loop speedup over Continuous (×)"
    )
    ax.set_title(
        "End-to-End Temporal Decision Performance"
    )
    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.25,
    )
    fig.tight_layout()

    path = (
        fig_dir
        / "FIG_CC5_end_to_end_temporal_speedup.png"
    )
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return path


def plot_break_even(
    e2e_table: pd.DataFrame,
    out_dir: Path,
):
    """
    Paper figure: cold-start and warm-cache break-even in DECISIONS.

    Log scale is essential because city-scale substrate construction produces
    break-even values spanning multiple orders of magnitude.
    """
    fig_dir = out_dir / "paper_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    plot_df = e2e_table[
        e2e_table["n_agents"]
        == max(E2E_TARGET_POPULATIONS)
    ].copy()

    # If 100K is unavailable for a city, fall back to its largest available N.
    rows = []
    for city_key, spec in CITY_SPECS.items():
        sub = e2e_table[
            e2e_table["city"] == spec["label"]
        ].sort_values("n_agents")

        if sub.empty:
            continue

        preferred = sub[
            sub["n_agents"]
            == max(E2E_TARGET_POPULATIONS)
        ]
        rows.append(
            preferred.iloc[0]
            if not preferred.empty
            else sub.iloc[-1]
        )

    if not rows:
        return None

    plot_df = pd.DataFrame(rows)

    x = np.arange(len(plot_df), dtype=float)
    width = 0.34

    cold = pd.to_numeric(
        plot_df["cold_break_even_decisions"],
        errors="coerce",
    ).to_numpy(float)

    warm = pd.to_numeric(
        plot_df["warm_break_even_decisions"],
        errors="coerce",
    ).to_numpy(float)

    ax.bar(
        x - width / 2,
        cold,
        width,
        label="Cold start",
    )
    ax.bar(
        x + width / 2,
        warm,
        width,
        label="Warm cache",
    )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_df["city"].astype(str).tolist()
    )
    ax.set_ylabel(
        "Break-even spatial decisions (log scale)"
    )
    ax.set_title(
        "Amortization of Hierarchical Spatial Preparation"
    )
    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.25,
        which="both",
    )
    fig.tight_layout()

    path = (
        fig_dir
        / "FIG_CC6_break_even_decisions.png"
    )
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return path



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

    if row.empty:
        return np.nan

    if column not in row.columns:
        return np.nan

    value = pd.to_numeric(
        row.iloc[0][column],
        errors="coerce",
    )

    return (
        float(value)
        if pd.notna(value)
        else np.nan
    )


def architecture_value(
    df: pd.DataFrame,
    operation: str,
    n_agents: int,
    column: str,
) -> float:
    row = df[
        (
            pd.to_numeric(
                df["n_agents"],
                errors="coerce",
            )
            == n_agents
        )
        & (
            df["operation"].astype(str)
            == operation
        )
    ]

    if row.empty:
        return np.nan

    value = pd.to_numeric(
        row.iloc[0][column],
        errors="coerce",
    )

    return (
        float(value)
        if pd.notna(value)
        else np.nan
    )


def max_architecture_population(
    df: pd.DataFrame,
) -> int:
    return int(
        pd.to_numeric(
            df["n_agents"],
            errors="coerce",
        )
        .dropna()
        .max()
    )


def load_city_characteristics(
    city_key: str,
) -> dict:
    """
    Read city characteristics from the exact assets used by the model.
    """

    cfg = config_for_city(
        city_key,
        n_agents=10_000,
    )

    raw_pois = pd.read_csv(
        cfg.osm_cache_csv
    )

    semantic_pois = load_osm_pois(
        cfg
    )

    population = pd.read_csv(
        cfg.population_h3_csv
    )

    population_values = pd.to_numeric(
        population[cfg.population_col],
        errors="coerce",
    ).fillna(0.0)

    graph_path = Path(
        cfg.routing.graphml_path
    )

    if not graph_path.exists():
        graph_nodes = np.nan
        graph_edges = np.nan
    else:
        G = nx.read_graphml(
            graph_path
        )
        graph_nodes = int(
            G.number_of_nodes()
        )
        graph_edges = int(
            G.number_of_edges()
        )

    return {
        "raw_pois": int(
            len(raw_pois)
        ),
        "semantic_pois": int(
            len(semantic_pois)
        ),
        "population_h3_cells": int(
            len(population)
        ),
        "represented_population": float(
            population_values.sum()
        ),
        "osm_graph_nodes": graph_nodes,
        "osm_graph_edges": graph_edges,
    }


def build_city_characteristics_table() -> pd.DataFrame:
    rows = []

    for city_key, spec in CITY_SPECS.items():
        c = load_city_characteristics(
            city_key
        )

        rows.append(
            {
                "city_key": city_key,
                "city": spec["label"],
                **c,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_performance_table(
    architecture: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Long-format table with Continuous latency, H3 latency,
    QPS, and H3-over-Continuous speedup at the largest
    completed architecture population for each city.
    """

    rows = []

    for city_key, spec in CITY_SPECS.items():
        df = architecture[city_key]

        n_max = max_architecture_population(
            df
        )

        for operation in PERFORMANCE_OPERATIONS:
            rows.append(
                {
                    "city_key": city_key,
                    "city": spec["label"],
                    "n_agents": n_max,
                    "operation": operation,
                    "operation_label":
                        PERFORMANCE_LABELS[
                            operation
                        ],

                    "continuous_us_per_query":
                        architecture_value(
                            df,
                            operation,
                            n_max,
                            "continuous_us_per_query",
                        ),

                    "h3_us_per_query":
                        architecture_value(
                            df,
                            operation,
                            n_max,
                            "h3_us_per_query",
                        ),

                    "continuous_qps":
                        architecture_value(
                            df,
                            operation,
                            n_max,
                            "continuous_qps",
                        ),

                    "h3_qps":
                        architecture_value(
                            df,
                            operation,
                            n_max,
                            "h3_qps",
                        ),

                    "speedup_h3_over_continuous":
                        architecture_value(
                            df,
                            operation,
                            n_max,
                            "speedup_h3_over_continuous",
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def build_fidelity_table(
    fidelity: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for city_key, spec in CITY_SPECS.items():
        df = fidelity[city_key]

        rows.append(
            {
                "city_key": city_key,
                "city": spec["label"],

                "accessibility_pearson":
                    fidelity_value(
                        df,
                        "accessibility",
                        "decision",
                        "pearson",
                    ),

                "accessibility_nmae":
                    fidelity_value(
                        df,
                        "accessibility",
                        "decision",
                        "nmae",
                    ),

                "destination_exact_match":
                    fidelity_value(
                        df,
                        "destination",
                        "decision",
                        "primary_value",
                    ),

                "destination_within_100m":
                    fidelity_value(
                        df,
                        "destination",
                        "decision",
                        "within_100m",
                    ),

                "destination_within_250m":
                    fidelity_value(
                        df,
                        "destination",
                        "decision",
                        "within_250m",
                    ),

                "destination_within_500m":
                    fidelity_value(
                        df,
                        "destination",
                        "decision",
                        "within_500m",
                    ),

                "destination_within_1000m":
                    fidelity_value(
                        df,
                        "destination",
                        "decision",
                        "within_1000m",
                    ),

                "median_destination_separation_m":
                    fidelity_value(
                        df,
                        "destination",
                        "decision",
                        "median_separation_m",
                    ),

                "p90_destination_separation_m":
                    fidelity_value(
                        df,
                        "destination",
                        "decision",
                        "p90_separation_m",
                    ),

                "trip_distance_pearson":
                    fidelity_value(
                        df,
                        "trip_distance",
                        "decision",
                        "pearson",
                    ),

                "trip_distance_nmae":
                    fidelity_value(
                        df,
                        "trip_distance",
                        "decision",
                        "nmae",
                    ),

                "od_h7_pearson":
                    fidelity_value(
                        df,
                        "od_flow",
                        "H7",
                        "pearson",
                    ),

                "od_h7_nmae":
                    fidelity_value(
                        df,
                        "od_flow",
                        "H7",
                        "nmae",
                    ),

                "od_h7_jsd":
                    fidelity_value(
                        df,
                        "od_flow",
                        "H7",
                        "jsd",
                    ),

                "od_h8_pearson":
                    fidelity_value(
                        df,
                        "od_flow",
                        "H8",
                        "pearson",
                    ),

                "od_h8_nmae":
                    fidelity_value(
                        df,
                        "od_flow",
                        "H8",
                        "nmae",
                    ),

                "od_h8_jsd":
                    fidelity_value(
                        df,
                        "od_flow",
                        "H8",
                        "jsd",
                    ),

                "od_h9_pearson":
                    fidelity_value(
                        df,
                        "od_flow",
                        "H9",
                        "pearson",
                    ),

                "od_h9_nmae":
                    fidelity_value(
                        df,
                        "od_flow",
                        "H9",
                        "nmae",
                    ),

                "od_h9_jsd":
                    fidelity_value(
                        df,
                        "od_flow",
                        "H9",
                        "jsd",
                    ),

                "occupancy_h7_pearson":
                    fidelity_value(
                        df,
                        "occupancy",
                        "H7",
                        "pearson",
                    ),

                "occupancy_h7_nmae":
                    fidelity_value(
                        df,
                        "occupancy",
                        "H7",
                        "nmae",
                    ),

                "occupancy_h7_jsd":
                    fidelity_value(
                        df,
                        "occupancy",
                        "H7",
                        "jsd",
                    ),

                "occupancy_h8_pearson":
                    fidelity_value(
                        df,
                        "occupancy",
                        "H8",
                        "pearson",
                    ),

                "occupancy_h8_nmae":
                    fidelity_value(
                        df,
                        "occupancy",
                        "H8",
                        "nmae",
                    ),

                "occupancy_h8_jsd":
                    fidelity_value(
                        df,
                        "occupancy",
                        "H8",
                        "jsd",
                    ),

                "occupancy_h9_pearson":
                    fidelity_value(
                        df,
                        "occupancy",
                        "H9",
                        "pearson",
                    ),

                "occupancy_h9_nmae":
                    fidelity_value(
                        df,
                        "occupancy",
                        "H9",
                        "nmae",
                    ),

                "occupancy_h9_jsd":
                    fidelity_value(
                        df,
                        "occupancy",
                        "H9",
                        "jsd",
                    ),

                "amenity_usage_pearson":
                    fidelity_value(
                        df,
                        "amenity_usage",
                        "city",
                        "pearson",
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_master_wide(
    city_table: pd.DataFrame,
    performance_table: pd.DataFrame,
    fidelity_table: pd.DataFrame,
    end_to_end_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    One-row-per-city master table.
    """

    perf_wide_parts = []

    for operation in PERFORMANCE_OPERATIONS:
        sub = performance_table[
            performance_table[
                "operation"
            ]
            == operation
        ].copy()

        prefix = operation

        sub = sub[
            [
                "city_key",
                "continuous_us_per_query",
                "h3_us_per_query",
                "continuous_qps",
                "h3_qps",
                "speedup_h3_over_continuous",
            ]
        ].rename(
            columns={
                "continuous_us_per_query":
                    f"{prefix}_continuous_us_per_query",
                "h3_us_per_query":
                    f"{prefix}_h3_us_per_query",
                "continuous_qps":
                    f"{prefix}_continuous_qps",
                "h3_qps":
                    f"{prefix}_h3_qps",
                "speedup_h3_over_continuous":
                    f"{prefix}_speedup_x",
            }
        )

        perf_wide_parts.append(
            sub
        )

    perf_wide = perf_wide_parts[0]

    for sub in perf_wide_parts[1:]:
        perf_wide = perf_wide.merge(
            sub,
            on="city_key",
            how="outer",
        )

    master = (
        city_table
        .merge(
            perf_wide,
            on="city_key",
            how="left",
        )
        .merge(
            fidelity_table.drop(
                columns=["city"],
                errors="ignore",
            ),
            on="city_key",
            how="left",
        )
    )

    if (
        end_to_end_table is not None
        and not end_to_end_table.empty
    ):
        headline_n = max(
            E2E_TARGET_POPULATIONS
        )

        e2e_headline = end_to_end_table[
            end_to_end_table["n_agents"]
            == headline_n
        ].copy()

        if e2e_headline.empty:
            # Graceful fallback to the largest available N per city.
            e2e_headline = (
                end_to_end_table
                .sort_values("n_agents")
                .groupby("city_key", as_index=False)
                .tail(1)
            )

        e2e_headline = e2e_headline[
            [
                "city_key",
                "n_agents",
                "temporal_speedup_paired_median",
                "temporal_speedup_min",
                "temporal_speedup_max",
                "cold_total_speedup",
                "cold_break_even_decisions",
                "cold_break_even_equivalent_agents",
                "warm_total_speedup",
                "warm_break_even_decisions",
            ]
        ].rename(
            columns={
                "n_agents":
                    "temporal_headline_n_agents",
                "temporal_speedup_paired_median":
                    "temporal_speedup_x",
                "temporal_speedup_min":
                    "temporal_speedup_min_x",
                "temporal_speedup_max":
                    "temporal_speedup_max_x",
                "cold_total_speedup":
                    "cold_total_speedup_x",
                "cold_break_even_decisions":
                    "cold_break_even_decisions",
                "cold_break_even_equivalent_agents":
                    "cold_break_even_equivalent_agents",
                "warm_total_speedup":
                    "warm_total_speedup_x",
                "warm_break_even_decisions":
                    "warm_break_even_decisions",
            }
        )

        master = master.merge(
            e2e_headline,
            on="city_key",
            how="left",
        )

    return master


def build_paper_summary(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compact table intended for manuscript drafting.
    """

    cols = [
        "city",
        "represented_population",
        "semantic_pois",
        "population_h3_cells",
        "osm_graph_nodes",
        "osm_graph_edges",

        "pure_spatial_lookup_speedup_x",
        "urban_context_retrieval_speedup_x",
        "accessibility_query_speedup_x",
        "candidate_retrieval_speedup_x",
        "full_agent_decision_speedup_x",

        "temporal_headline_n_agents",
        "temporal_speedup_x",
        "temporal_speedup_min_x",
        "temporal_speedup_max_x",
        "cold_total_speedup_x",
        "cold_break_even_decisions",
        "cold_break_even_equivalent_agents",
        "warm_total_speedup_x",
        "warm_break_even_decisions",

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

    existing = [
        c
        for c in cols
        if c in master.columns
    ]

    return master[
        existing
    ].copy()


def save_transposed_paper_tables(
    city_table: pd.DataFrame,
    performance_table: pd.DataFrame,
    fidelity_table: pd.DataFrame,
    end_to_end_table: pd.DataFrame,
    out_dir: Path,
):
    """
    Generate human-readable metric-by-city tables.
    """

    # --------------------------------------------------------------
    # TABLE 1 — City characteristics
    # --------------------------------------------------------------

    city_display = city_table[
        [
            "city",
            "represented_population",
            "raw_pois",
            "semantic_pois",
            "population_h3_cells",
            "osm_graph_nodes",
            "osm_graph_edges",
        ]
    ].set_index("city").T

    city_display.index.name = "metric"

    city_display.to_csv(
        out_dir
        / "TABLE_1_city_characteristics.csv"
    )

    # --------------------------------------------------------------
    # TABLE 2 — Computational performance @ max N
    # --------------------------------------------------------------

    perf_rows = []

    for operation in PERFORMANCE_OPERATIONS:
        sub = performance_table[
            performance_table["operation"]
            == operation
        ]

        for metric, label in [
            (
                "continuous_us_per_query",
                "Continuous latency (µs/query)",
            ),
            (
                "h3_us_per_query",
                "H3 latency (µs/query)",
            ),
            (
                "speedup_h3_over_continuous",
                "H3 speedup (×)",
            ),
        ]:
            row = {
                "metric":
                    f"{PERFORMANCE_LABELS[operation]} — "
                    f"{label}"
            }

            for _, r in sub.iterrows():
                row[
                    str(r["city"])
                ] = r[metric]

            perf_rows.append(
                row
            )

    pd.DataFrame(
        perf_rows
    ).to_csv(
        out_dir
        / "TABLE_2_computational_performance_1M.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # TABLE 3 — Fidelity @ 10K
    # --------------------------------------------------------------

    fidelity_metrics = [
        (
            "accessibility_pearson",
            "Accessibility Pearson r",
        ),
        (
            "destination_exact_match",
            "Exact destination match",
        ),
        (
            "destination_within_500m",
            "Destination within 500 m",
        ),
        (
            "destination_within_1000m",
            "Destination within 1 km",
        ),
        (
            "trip_distance_pearson",
            "Trip-distance Pearson r",
        ),
        (
            "od_h7_pearson",
            "OD flow Pearson r — H7",
        ),
        (
            "od_h8_pearson",
            "OD flow Pearson r — H8",
        ),
        (
            "od_h9_pearson",
            "OD flow Pearson r — H9",
        ),
        (
            "occupancy_h7_pearson",
            "Occupancy Pearson r — H7",
        ),
        (
            "occupancy_h8_pearson",
            "Occupancy Pearson r — H8",
        ),
        (
            "occupancy_h9_pearson",
            "Occupancy Pearson r — H9",
        ),
        (
            "amenity_usage_pearson",
            "Amenity usage Pearson r",
        ),
    ]

    fidelity_rows = []

    for column, label in fidelity_metrics:
        row = {
            "metric": label
        }

        for _, r in fidelity_table.iterrows():
            row[
                str(r["city"])
            ] = r[column]

        fidelity_rows.append(
            row
        )

    pd.DataFrame(
        fidelity_rows
    ).to_csv(
        out_dir
        / "TABLE_3_fidelity_10K.csv",
        index=False,
    )


    # --------------------------------------------------------------
    # TABLE 4 — Temporal end-to-end performance & amortization
    # --------------------------------------------------------------
    save_end_to_end_paper_table(
        end_to_end_table,
        out_dir,
    )


def print_summary(
    city_table: pd.DataFrame,
    performance_table: pd.DataFrame,
    fidelity_table: pd.DataFrame,
    end_to_end_table: pd.DataFrame,
):
    print("=" * 120)
    print("MASTER CROSS-CITY EXPERIMENTAL SUMMARY")
    print("=" * 120)

    print()
    print("A. CITY CHARACTERISTICS")
    print("-" * 120)

    print(
        city_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print("B. COMPUTATIONAL PERFORMANCE @ MAX BENCHMARK POPULATION")
    print("-" * 120)

    print(
        performance_table[
            [
                "city",
                "n_agents",
                "operation_label",
                "continuous_us_per_query",
                "h3_us_per_query",
                "speedup_h3_over_continuous",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("C. FIDELITY @ N=10,000")
    print("-" * 120)

    headline = fidelity_table[
        [
            "city",
            "accessibility_pearson",
            "destination_exact_match",
            "destination_within_500m",
            "destination_within_1000m",
            "trip_distance_pearson",
            "od_h7_pearson",
            "od_h8_pearson",
            "od_h9_pearson",
            "occupancy_h7_pearson",
            "occupancy_h8_pearson",
            "occupancy_h9_pearson",
            "amenity_usage_pearson",
        ]
    ]

    print(
        headline.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    print()
    print("D. TEMPORAL END-TO-END PERFORMANCE & AMORTIZATION")
    print("-" * 120)

    if end_to_end_table.empty:
        print("[no end-to-end results available]")
    else:
        display_cols = [
            "city",
            "n_agents",
            "repeats",
            "temporal_speedup_paired_median",
            "temporal_speedup_min",
            "temporal_speedup_max",
            "cold_total_speedup",
            "cold_break_even_decisions",
            "cold_break_even_equivalent_agents",
            "warm_total_speedup",
            "warm_break_even_decisions",
        ]

        existing = [
            c for c in display_cols
            if c in end_to_end_table.columns
        ]

        print(
            end_to_end_table[
                existing
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )


def main():
    out_dir = (
        ROOT
        / "outputs"
        / "cross_city"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    architecture = {
        city_key:
            load_architecture(
                city_key
            )
        for city_key
        in CITY_SPECS
    }

    fidelity = {
        city_key:
            load_fidelity(
                city_key
            )
        for city_key
        in CITY_SPECS
    }


    end_to_end = {
        city_key:
            load_end_to_end(
                city_key
            )
        for city_key
        in CITY_SPECS
    }

    city_table = (
        build_city_characteristics_table()
    )

    performance_table = (
        build_performance_table(
            architecture
        )
    )

    fidelity_table = (
        build_fidelity_table(
            fidelity
        )
    )


    end_to_end_table = (
        build_end_to_end_table(
            end_to_end
        )
    )

    validate_end_to_end_invariants(
        end_to_end_table
    )

    master = build_master_wide(
        city_table,
        performance_table,
        fidelity_table,
        end_to_end_table,
    )

    paper_summary = build_paper_summary(
        master
    )

    # Full machine-readable outputs.
    city_table.to_csv(
        out_dir
        / "city_characteristics_long.csv",
        index=False,
    )

    performance_table.to_csv(
        out_dir
        / "computational_performance_1M_long.csv",
        index=False,
    )

    fidelity_table.to_csv(
        out_dir
        / "fidelity_10K_long.csv",
        index=False,
    )


    end_to_end_table.to_csv(
        out_dir
        / "end_to_end_temporal_long.csv",
        index=False,
    )

    master.to_csv(
        out_dir
        / "MASTER_cross_city_full.csv",
        index=False,
    )

    paper_summary.to_csv(
        out_dir
        / "TABLE_PAPER_cross_city_summary.csv",
        index=False,
    )

    save_transposed_paper_tables(
        city_table,
        performance_table,
        fidelity_table,
        end_to_end_table,
        out_dir,
    )

    print_summary(
        city_table,
        performance_table,
        fidelity_table,
        end_to_end_table,
    )

    fig_speedup = plot_temporal_speedup(
        end_to_end_table,
        out_dir,
    )

    fig_break_even = plot_break_even(
        end_to_end_table,
        out_dir,
    )

    print()
    print("=" * 120)
    print("FILES SAVED")
    print("=" * 120)

    for name in [
        "MASTER_cross_city_full.csv",
        "city_characteristics_long.csv",
        "computational_performance_1M_long.csv",
        "fidelity_10K_long.csv",
        "end_to_end_temporal_long.csv",
        "TABLE_1_city_characteristics.csv",
        "TABLE_2_computational_performance_1M.csv",
        "TABLE_3_fidelity_10K.csv",
        "TABLE_4_end_to_end_temporal_amortization.csv",
        "TABLE_PAPER_cross_city_summary.csv",
    ]:
        print(
            f"[saved] {out_dir / name}"
        )


    if fig_speedup is not None:
        print(f"[saved] {fig_speedup}")

    if fig_break_even is not None:
        print(f"[saved] {fig_break_even}")


if __name__ == "__main__":
    main()
