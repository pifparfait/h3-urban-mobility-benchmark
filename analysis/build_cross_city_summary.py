#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CITY_SPECS = {
    "madrid": {
        "label": "Madrid",
        "architecture_dir": ROOT / "outputs" / "architecture",
        "fidelity_dir": ROOT / "outputs" / "urban_fidelity",
    },
    "cambridge_ma": {
        "label": "Cambridge",
        "architecture_dir": ROOT / "outputs" / "cambridge_ma" / "architecture",
        "fidelity_dir": ROOT / "outputs" / "cambridge_ma" / "urban_fidelity",
    },
}

OPERATIONS = [
    "pure_spatial_lookup",
    "urban_context_retrieval",
    "accessibility_query",
    "candidate_retrieval",
    "full_agent_decision",
]

ARCH_SHORT = {
    "pure_spatial_lookup": "spatial_lookup_speedup_x",
    "urban_context_retrieval": "urban_context_speedup_x",
    "accessibility_query": "accessibility_speedup_x",
    "candidate_retrieval": "candidate_retrieval_speedup_x",
    "full_agent_decision": "full_decision_speedup_x",
}

FIDELITY_ROWS = [
    ("accessibility", "decision", "pearson", "accessibility_pearson"),
    ("destination", "decision", "primary_value", "destination_exact_match"),
    ("destination", "decision", "within_100m", "destination_within_100m"),
    ("destination", "decision", "within_250m", "destination_within_250m"),
    ("destination", "decision", "within_500m", "destination_within_500m"),
    ("destination", "decision", "within_1000m", "destination_within_1000m"),
    ("destination", "decision", "median_separation_m", "median_destination_separation_m"),
    ("destination", "decision", "p90_separation_m", "p90_destination_separation_m"),
    ("trip_distance", "decision", "pearson", "trip_distance_pearson"),
    ("od_flow", "H7", "pearson", "od_h7_pearson"),
    ("od_flow", "H8", "pearson", "od_h8_pearson"),
    ("od_flow", "H9", "pearson", "od_h9_pearson"),
    ("occupancy", "H7", "pearson", "occupancy_h7_pearson"),
    ("occupancy", "H8", "pearson", "occupancy_h8_pearson"),
    ("occupancy", "H9", "pearson", "occupancy_h9_pearson"),
    ("amenity_usage", "city", "pearson", "amenity_usage_pearson"),
]


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_architecture(city_key: str, spec: dict) -> pd.DataFrame:
    path = spec["architecture_dir"] / "architecture_sweep_summary.csv"
    require(path)
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
            f"{path} missing architecture columns: {sorted(missing)}"
        )

    df = df.copy()
    df["city_key"] = city_key
    df["city"] = spec["label"]
    return df


def choose_fidelity_file(spec: dict) -> Path:
    preferred = spec["fidelity_dir"] / "urban_fidelity_n10000.csv"
    if preferred.exists():
        return preferred

    candidates = sorted(
        spec["fidelity_dir"].glob("urban_fidelity_n*.csv"),
        key=lambda p: int(p.stem.split("_n")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No urban_fidelity_n*.csv in {spec['fidelity_dir']}"
        )
    return candidates[-1]


def load_fidelity(city_key: str, spec: dict) -> tuple[pd.DataFrame, int]:
    path = choose_fidelity_file(spec)
    df = pd.read_csv(path)

    required = {"metric", "scale"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path} missing fidelity columns: {sorted(missing)}"
        )

    n_agents = int(path.stem.split("_n")[-1])
    df = df.copy()
    df["city_key"] = city_key
    df["city"] = spec["label"]
    df["fidelity_n_agents"] = n_agents
    return df, n_agents


def find_value(
    fidelity: pd.DataFrame,
    metric: str,
    scale: str,
    column: str,
) -> float:
    row = fidelity[
        (fidelity["metric"].astype(str) == metric)
        & (fidelity["scale"].astype(str) == scale)
    ]

    if row.empty or column not in row.columns:
        return np.nan

    value = pd.to_numeric(row.iloc[0][column], errors="coerce")
    return float(value) if pd.notna(value) else np.nan


def build_macro_row(
    city_key: str,
    spec: dict,
    arch: pd.DataFrame,
    fidelity: pd.DataFrame,
    fidelity_n: int,
    n_agents: int,
) -> dict:
    row = {
        "city_key": city_key,
        "city": spec["label"],
        "n_agents": int(n_agents),
        "fidelity_n_agents": int(fidelity_n),
    }

    for op in OPERATIONS:
        op_row = arch[
            (arch["n_agents"].astype(int) == int(n_agents))
            & (arch["operation"].astype(str) == op)
        ]

        col = ARCH_SHORT[op]
        if op_row.empty:
            row[col] = np.nan
            row[col.replace("_speedup_x", "_continuous_us")] = np.nan
            row[col.replace("_speedup_x", "_h3_us")] = np.nan
        else:
            r = op_row.iloc[0]
            row[col] = float(r["speedup_h3_over_continuous"])
            row[col.replace("_speedup_x", "_continuous_us")] = float(
                r["continuous_us_per_query"]
            )
            row[col.replace("_speedup_x", "_h3_us")] = float(
                r["h3_us_per_query"]
            )

    # Fidelity is measured once at the controlled fidelity population
    # (normally N=10,000), so repeat it only on the matching macro row.
    if int(n_agents) == int(fidelity_n):
        for metric, scale, source_col, output_col in FIDELITY_ROWS:
            row[output_col] = find_value(
                fidelity,
                metric,
                scale,
                source_col,
            )
    else:
        for _, _, _, output_col in FIDELITY_ROWS:
            row[output_col] = np.nan

    return row


def build_headline_table(
    architectures: dict[str, pd.DataFrame],
    fidelities: dict[str, pd.DataFrame],
    fidelity_ns: dict[str, int],
) -> pd.DataFrame:
    rows = []

    for city_key, spec in CITY_SPECS.items():
        arch = architectures[city_key]
        fid = fidelities[city_key]
        fidelity_n = fidelity_ns[city_key]

        max_n = int(
            pd.to_numeric(
                arch["n_agents"],
                errors="coerce",
            ).max()
        )

        result = {
            "city": spec["label"],
            "architecture_max_agents": max_n,
            "fidelity_agents": fidelity_n,
        }

        for op in OPERATIONS:
            subset = arch[
                (arch["n_agents"].astype(int) == max_n)
                & (arch["operation"].astype(str) == op)
            ]
            result[ARCH_SHORT[op]] = (
                float(subset.iloc[0]["speedup_h3_over_continuous"])
                if not subset.empty
                else np.nan
            )

        for metric, scale, source_col, output_col in FIDELITY_ROWS:
            result[output_col] = find_value(
                fid,
                metric,
                scale,
                source_col,
            )

        rows.append(result)

    return pd.DataFrame(rows)


def main():
    out_dir = ROOT / "outputs" / "cross_city"
    out_dir.mkdir(parents=True, exist_ok=True)

    architectures = {}
    fidelities = {}
    fidelity_ns = {}

    for city_key, spec in CITY_SPECS.items():
        architectures[city_key] = load_architecture(city_key, spec)
        fidelities[city_key], fidelity_ns[city_key] = load_fidelity(
            city_key,
            spec,
        )

    # 1) Long architecture table
    arch_long = pd.concat(
        [architectures[k] for k in CITY_SPECS],
        ignore_index=True,
    )
    arch_long.to_csv(
        out_dir / "cross_city_architecture_long.csv",
        index=False,
    )

    # 2) Long fidelity table
    fidelity_long = pd.concat(
        [fidelities[k] for k in CITY_SPECS],
        ignore_index=True,
    )
    fidelity_long.to_csv(
        out_dir / "cross_city_fidelity_long.csv",
        index=False,
    )

    # 3) Macro table: one row per city x architecture population
    populations = sorted(
        set(
            int(x)
            for df in architectures.values()
            for x in pd.to_numeric(
                df["n_agents"],
                errors="coerce",
            ).dropna().astype(int).tolist()
        )
    )

    macro_rows = []
    for city_key, spec in CITY_SPECS.items():
        arch = architectures[city_key]
        available_n = set(
            pd.to_numeric(
                arch["n_agents"],
                errors="coerce",
            ).dropna().astype(int)
        )

        for n_agents in populations:
            if n_agents not in available_n:
                continue

            macro_rows.append(
                build_macro_row(
                    city_key,
                    spec,
                    arch,
                    fidelities[city_key],
                    fidelity_ns[city_key],
                    n_agents,
                )
            )

    macro = pd.DataFrame(macro_rows).sort_values(
        ["city", "n_agents"]
    )
    macro.to_csv(
        out_dir / "madrid_cambridge_macro_summary.csv",
        index=False,
    )

    # 4) Headline table: max-N architecture + N=10K fidelity
    headline = build_headline_table(
        architectures,
        fidelities,
        fidelity_ns,
    )
    headline.to_csv(
        out_dir / "madrid_cambridge_headline_summary.csv",
        index=False,
    )

    # 5) Paper-friendly reduced headline table
    paper_cols = [
        "city",
        "architecture_max_agents",
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

    paper = headline[
        [c for c in paper_cols if c in headline.columns]
    ].copy()

    paper.to_csv(
        out_dir / "madrid_cambridge_paper_table.csv",
        index=False,
    )

    print("=" * 120)
    print("MADRID × CAMBRIDGE — CROSS-CITY MACRO SUMMARY")
    print("=" * 120)
    print()
    print("HEADLINE RESULTS")
    print("-" * 120)
    print(
        paper.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print()
    print("Saved:")
    print(f"  {out_dir / 'cross_city_architecture_long.csv'}")
    print(f"  {out_dir / 'cross_city_fidelity_long.csv'}")
    print(f"  {out_dir / 'madrid_cambridge_macro_summary.csv'}")
    print(f"  {out_dir / 'madrid_cambridge_headline_summary.csv'}")
    print(f"  {out_dir / 'madrid_cambridge_paper_table.csv'}")


if __name__ == "__main__":
    main()
