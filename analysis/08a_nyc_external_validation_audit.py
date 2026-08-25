#!/usr/bin/env python3
"""
08a_nyc_external_validation_audit.py
====================================

Audit the unexpectedly low absolute observed-reference agreement for NYC H7
without changing the frozen Experiment 07 protocol.

This script is diagnostic only. It reads the existing NYC external-validation
outputs and asks whether the low Pearson (~0.27 at H7) is primarily associated
with:

    1. support mismatch / sparsity,
    2. long-tail observed OD structure,
    3. poor origin/destination marginal agreement,
    4. possible origin/destination inversion,
    5. genuine pairwise OD mismatch.

It does NOT overwrite Experiment 07 outputs.

Expected input
--------------
outputs/external_od_validation/new_york_ny/three_way_od_h7.csv

Required columns:
    origin_h3
    destination_h3
    observed_flow
    continuous_flow
    h3_flow

Outputs
-------
outputs/external_od_validation/new_york_ny/audit/
    NYC_H7_audit_summary.csv
    NYC_H7_support_audit.csv
    NYC_H7_topflow_audit.csv
    NYC_H7_origin_marginals.csv
    NYC_H7_destination_marginals.csv
    NYC_H7_marginal_summary.csv
    NYC_H7_directionality_audit.csv

Interpretation
--------------
- If Pearson rises strongly on common support or top observed-flow mass, the
  low global correlation is largely a sparsity / support-tail phenomenon.
- If origin and destination marginals correlate well but OD-pair Pearson does
  not, the main mismatch is in pairwise matching rather than city-wide
  production/attraction patterns.
- If transposed observed OD correlates substantially better than the correctly
  oriented matrix, inspect direction conventions for a possible origin/work
  inversion bug.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12

DEFAULT_INPUT = Path(
    "outputs/external_od_validation/new_york_ny/three_way_od_h7.csv"
)
DEFAULT_OUTPUT = Path(
    "outputs/external_od_validation/new_york_ny/audit"
)


# =============================================================================
# Generic metrics
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return np.nan

    if np.std(x) <= EPS or np.std(y) <= EPS:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(x.corr(y, method="spearman"))


def nmae(pred: np.ndarray, obs: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)

    denom = float(np.mean(obs))
    if denom <= EPS:
        return np.nan

    return float(np.mean(np.abs(pred - obs)) / denom)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= EPS:
        return np.nan

    return float(np.dot(x, y) / denom)


def js_divergence(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    sx = float(x.sum())
    sy = float(y.sum())

    if sx <= EPS or sy <= EPS:
        return np.nan

    p = x / sx
    q = y / sy
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(
            np.sum(a[mask] * np.log(a[mask] / b[mask]))
        )

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def total_variation_distance(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    sx = float(x.sum())
    sy = float(y.sum())

    if sx <= EPS or sy <= EPS:
        return np.nan

    p = x / sx
    q = y / sy

    return float(0.5 * np.abs(p - q).sum())


def metric_bundle(
    obs: np.ndarray,
    pred: np.ndarray,
) -> dict:
    return {
        "pearson": safe_pearson(obs, pred),
        "spearman": safe_spearman(
            pd.Series(obs),
            pd.Series(pred),
        ),
        "nmae": nmae(pred, obs),
        "tvd": total_variation_distance(
            obs,
            pred,
        ),
        "jsd": js_divergence(
            obs,
            pred,
        ),
        "cosine": cosine_similarity(
            obs,
            pred,
        ),
    }


# =============================================================================
# Load / validate
# =============================================================================

def load_three_way(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)

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

    out = df[
        [
            "origin_h3",
            "destination_h3",
            "observed_flow",
            "continuous_flow",
            "h3_flow",
        ]
    ].copy()

    for c in [
        "observed_flow",
        "continuous_flow",
        "h3_flow",
    ]:
        out[c] = pd.to_numeric(
            out[c],
            errors="coerce",
        ).fillna(0.0)

        if (out[c] < -EPS).any():
            raise ValueError(
                f"Negative values detected in {c}"
            )

    out["origin_h3"] = (
        out["origin_h3"]
        .astype(str)
        .str.strip()
    )
    out["destination_h3"] = (
        out["destination_h3"]
        .astype(str)
        .str.strip()
    )

    return out


# =============================================================================
# 1. Global / support audit
# =============================================================================

def support_audit(df: pd.DataFrame) -> pd.DataFrame:
    masks = {
        "all_union_pairs": np.ones(
            len(df),
            dtype=bool,
        ),
        "observed_positive": (
            df["observed_flow"].to_numpy() > 0
        ),
        "continuous_positive": (
            df["continuous_flow"].to_numpy() > 0
        ),
        "h3_positive": (
            df["h3_flow"].to_numpy() > 0
        ),
        "observed_and_continuous_positive": (
            (df["observed_flow"].to_numpy() > 0)
            & (df["continuous_flow"].to_numpy() > 0)
        ),
        "observed_and_h3_positive": (
            (df["observed_flow"].to_numpy() > 0)
            & (df["h3_flow"].to_numpy() > 0)
        ),
        "all_three_positive": (
            (df["observed_flow"].to_numpy() > 0)
            & (df["continuous_flow"].to_numpy() > 0)
            & (df["h3_flow"].to_numpy() > 0)
        ),
        "simulated_union_positive": (
            (df["continuous_flow"].to_numpy() > 0)
            | (df["h3_flow"].to_numpy() > 0)
        ),
    }

    rows = []

    obs_total = float(df["observed_flow"].sum())

    for name, mask in masks.items():
        x = df.loc[mask].copy()

        if x.empty:
            continue

        obs = x["observed_flow"].to_numpy(dtype=float)
        cont = x["continuous_flow"].to_numpy(dtype=float)
        h3x = x["h3_flow"].to_numpy(dtype=float)

        mc = metric_bundle(obs, cont)
        mh = metric_bundle(obs, h3x)

        rows.append(
            {
                "support": name,
                "n_pairs": int(len(x)),
                "observed_flow_on_support": float(obs.sum()),
                "observed_flow_share_pct": (
                    100.0 * float(obs.sum()) / obs_total
                    if obs_total > EPS
                    else np.nan
                ),
                "continuous_flow_on_support": float(cont.sum()),
                "h3_flow_on_support": float(h3x.sum()),

                "continuous_pearson": mc["pearson"],
                "h3_pearson": mh["pearson"],
                "pearson_gap_h3_minus_continuous": (
                    mh["pearson"] - mc["pearson"]
                    if np.isfinite(mc["pearson"])
                    and np.isfinite(mh["pearson"])
                    else np.nan
                ),

                "continuous_spearman": mc["spearman"],
                "h3_spearman": mh["spearman"],

                "continuous_tvd": mc["tvd"],
                "h3_tvd": mh["tvd"],

                "continuous_jsd": mc["jsd"],
                "h3_jsd": mh["jsd"],

                "continuous_cosine": mc["cosine"],
                "h3_cosine": mh["cosine"],
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 2. Top observed-flow mass audit
# =============================================================================

def topflow_audit(
    df: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    """
    Evaluate the OD matrices on the pairs needed to explain a given percentage
    of observed H7 flow.

    Example threshold 0.80:
        smallest prefix of OD pairs, sorted by observed flow descending, whose
        cumulative observed flow reaches 80% of all observed H7 flow.
    """
    x = df.sort_values(
        "observed_flow",
        ascending=False,
    ).copy()

    total = float(x["observed_flow"].sum())

    if total <= EPS:
        raise ValueError(
            "Observed H7 flow is zero."
        )

    x["observed_flow_cumsum"] = (
        x["observed_flow"].cumsum()
    )
    x["observed_flow_cumshare"] = (
        x["observed_flow_cumsum"] / total
    )

    rows = []

    for threshold in thresholds:
        if not 0 < threshold <= 1:
            raise ValueError(
                f"Invalid top-flow threshold: {threshold}"
            )

        idx = int(
            np.searchsorted(
                x["observed_flow_cumshare"].to_numpy(),
                threshold,
                side="left",
            )
        )

        subset = x.iloc[
            : min(idx + 1, len(x))
        ].copy()

        obs = subset["observed_flow"].to_numpy(dtype=float)
        cont = subset["continuous_flow"].to_numpy(dtype=float)
        h3x = subset["h3_flow"].to_numpy(dtype=float)

        mc = metric_bundle(obs, cont)
        mh = metric_bundle(obs, h3x)

        rows.append(
            {
                "target_observed_flow_share": threshold,
                "actual_observed_flow_share": (
                    float(obs.sum()) / total
                ),
                "n_pairs": int(len(subset)),
                "continuous_nonzero_pairs": int(
                    (cont > 0).sum()
                ),
                "h3_nonzero_pairs": int(
                    (h3x > 0).sum()
                ),

                "continuous_pearson": mc["pearson"],
                "h3_pearson": mh["pearson"],
                "continuous_spearman": mc["spearman"],
                "h3_spearman": mh["spearman"],

                "continuous_tvd": mc["tvd"],
                "h3_tvd": mh["tvd"],

                "continuous_jsd": mc["jsd"],
                "h3_jsd": mh["jsd"],

                "continuous_cosine": mc["cosine"],
                "h3_cosine": mh["cosine"],
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 3. Marginals
# =============================================================================

def build_marginal_table(
    df: pd.DataFrame,
    key_col: str,
) -> pd.DataFrame:
    out = (
        df.groupby(
            key_col,
            as_index=False,
        )[
            [
                "observed_flow",
                "continuous_flow",
                "h3_flow",
            ]
        ]
        .sum()
    )

    return out


def marginal_summary(
    origin: pd.DataFrame,
    destination: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for label, table in [
        ("origin", origin),
        ("destination", destination),
    ]:
        obs = table["observed_flow"].to_numpy(dtype=float)
        cont = table["continuous_flow"].to_numpy(dtype=float)
        h3x = table["h3_flow"].to_numpy(dtype=float)

        mc = metric_bundle(obs, cont)
        mh = metric_bundle(obs, h3x)

        rows.append(
            {
                "marginal": label,
                "n_cells": int(len(table)),

                "continuous_pearson": mc["pearson"],
                "h3_pearson": mh["pearson"],
                "continuous_spearman": mc["spearman"],
                "h3_spearman": mh["spearman"],

                "continuous_tvd": mc["tvd"],
                "h3_tvd": mh["tvd"],

                "continuous_jsd": mc["jsd"],
                "h3_jsd": mh["jsd"],

                "continuous_cosine": mc["cosine"],
                "h3_cosine": mh["cosine"],
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 4. Direction / transpose audit
# =============================================================================

def transpose_observed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build observed flow with origin/destination swapped and align it back onto
    the original Continuous/H3 support.

    If this transposed observed matrix compares much better with simulations
    than the correctly oriented matrix, inspect a possible O/D inversion bug.
    """
    obs_t = (
        df[
            [
                "origin_h3",
                "destination_h3",
                "observed_flow",
            ]
        ]
        .rename(
            columns={
                "origin_h3": "destination_h3",
                "destination_h3": "origin_h3",
                "observed_flow": "observed_flow_transposed",
            }
        )
    )

    sim = df[
        [
            "origin_h3",
            "destination_h3",
            "continuous_flow",
            "h3_flow",
        ]
    ].copy()

    aligned = sim.merge(
        obs_t,
        on=[
            "origin_h3",
            "destination_h3",
        ],
        how="outer",
    ).fillna(0.0)

    return aligned


def directionality_audit(df: pd.DataFrame) -> pd.DataFrame:
    normal_obs = df["observed_flow"].to_numpy(dtype=float)
    cont = df["continuous_flow"].to_numpy(dtype=float)
    h3x = df["h3_flow"].to_numpy(dtype=float)

    t = transpose_observed(df)

    trans_obs = t[
        "observed_flow_transposed"
    ].to_numpy(dtype=float)
    trans_cont = t[
        "continuous_flow"
    ].to_numpy(dtype=float)
    trans_h3 = t[
        "h3_flow"
    ].to_numpy(dtype=float)

    mc_normal = metric_bundle(
        normal_obs,
        cont,
    )
    mh_normal = metric_bundle(
        normal_obs,
        h3x,
    )

    mc_trans = metric_bundle(
        trans_obs,
        trans_cont,
    )
    mh_trans = metric_bundle(
        trans_obs,
        trans_h3,
    )

    return pd.DataFrame(
        [
            {
                "orientation": "normal_home_to_work",
                "continuous_pearson": mc_normal["pearson"],
                "h3_pearson": mh_normal["pearson"],
                "continuous_tvd": mc_normal["tvd"],
                "h3_tvd": mh_normal["tvd"],
                "continuous_jsd": mc_normal["jsd"],
                "h3_jsd": mh_normal["jsd"],
            },
            {
                "orientation": "observed_transposed",
                "continuous_pearson": mc_trans["pearson"],
                "h3_pearson": mh_trans["pearson"],
                "continuous_tvd": mc_trans["tvd"],
                "h3_tvd": mh_trans["tvd"],
                "continuous_jsd": mc_trans["jsd"],
                "h3_jsd": mh_trans["jsd"],
            },
        ]
    )


# =============================================================================
# 5. Compact summary / diagnosis
# =============================================================================

def build_audit_summary(
    df: pd.DataFrame,
    support: pd.DataFrame,
    topflow: pd.DataFrame,
    marginals: pd.DataFrame,
    direction: pd.DataFrame,
) -> pd.DataFrame:
    global_row = support[
        support["support"] == "all_union_pairs"
    ].iloc[0]

    common_row = support[
        support["support"]
        == "observed_and_continuous_positive"
    ]

    common_pearson = (
        float(common_row.iloc[0]["continuous_pearson"])
        if len(common_row)
        else np.nan
    )

    normal = direction[
        direction["orientation"] == "normal_home_to_work"
    ].iloc[0]

    transposed = direction[
        direction["orientation"] == "observed_transposed"
    ].iloc[0]

    origin = marginals[
        marginals["marginal"] == "origin"
    ].iloc[0]

    destination = marginals[
        marginals["marginal"] == "destination"
    ].iloc[0]

    top90 = topflow[
        np.isclose(
            topflow["target_observed_flow_share"],
            0.90,
        )
    ]

    top90_pearson = (
        float(top90.iloc[0]["continuous_pearson"])
        if len(top90)
        else np.nan
    )

    return pd.DataFrame(
        [
            {
                "global_continuous_pearson": float(
                    global_row["continuous_pearson"]
                ),
                "global_h3_pearson": float(
                    global_row["h3_pearson"]
                ),
                "common_support_continuous_pearson": common_pearson,
                "top90_observed_flow_continuous_pearson": top90_pearson,

                "origin_marginal_continuous_pearson": float(
                    origin["continuous_pearson"]
                ),
                "destination_marginal_continuous_pearson": float(
                    destination["continuous_pearson"]
                ),

                "normal_continuous_pearson": float(
                    normal["continuous_pearson"]
                ),
                "transposed_continuous_pearson": float(
                    transposed["continuous_pearson"]
                ),

                "observed_total_flow": float(
                    df["observed_flow"].sum()
                ),
                "continuous_total_flow": float(
                    df["continuous_flow"].sum()
                ),
                "h3_total_flow": float(
                    df["h3_flow"].sum()
                ),

                "observed_positive_pairs": int(
                    (df["observed_flow"] > 0).sum()
                ),
                "continuous_positive_pairs": int(
                    (df["continuous_flow"] > 0).sum()
                ),
                "h3_positive_pairs": int(
                    (df["h3_flow"] > 0).sum()
                ),
            }
        ]
    )


def print_interpretive_flags(
    summary: pd.DataFrame,
) -> None:
    row = summary.iloc[0]

    global_p = row["global_continuous_pearson"]
    common_p = row["common_support_continuous_pearson"]
    top90_p = row["top90_observed_flow_continuous_pearson"]
    origin_p = row["origin_marginal_continuous_pearson"]
    dest_p = row["destination_marginal_continuous_pearson"]
    normal_p = row["normal_continuous_pearson"]
    trans_p = row["transposed_continuous_pearson"]

    print("\n" + "=" * 104)
    print("DIAGNOSTIC FLAGS")
    print("=" * 104)

    if (
        np.isfinite(common_p)
        and np.isfinite(global_p)
        and common_p > global_p + 0.20
    ):
        print(
            "[SIGNAL] Pearson rises strongly on observed/simulated common "
            "support -> sparsity/support mismatch is a major contributor."
        )
    else:
        print(
            "[INFO] Common-support Pearson does not rise dramatically."
        )

    if (
        np.isfinite(top90_p)
        and np.isfinite(global_p)
        and top90_p > global_p + 0.15
    ):
        print(
            "[SIGNAL] Pearson rises on the top 90% of observed flow -> "
            "the observed long tail materially depresses the global metric."
        )
    else:
        print(
            "[INFO] Top-flow filtering does not dramatically rescue Pearson."
        )

    if (
        np.isfinite(origin_p)
        and np.isfinite(dest_p)
        and min(origin_p, dest_p) > global_p + 0.20
    ):
        print(
            "[SIGNAL] Origin/destination marginals agree substantially better "
            "than pairwise OD -> mismatch is mainly in OD pairing."
        )
    else:
        print(
            "[INFO] Marginals are not uniformly much stronger than OD pairs."
        )

    if (
        np.isfinite(trans_p)
        and np.isfinite(normal_p)
        and trans_p > normal_p + 0.10
    ):
        print(
            "[WARNING] Transposed observed OD correlates materially better "
            "than normal orientation -> inspect origin/destination convention."
        )
    else:
        print(
            "[OK] No strong evidence of an origin/destination inversion."
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the low NYC H7 observed-reference Pearson without "
            "changing Experiment 07."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="NYC three_way_od_h7.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--top-flow-thresholds",
        type=float,
        nargs="+",
        default=[0.50, 0.80, 0.90, 0.95, 0.99],
    )

    args = parser.parse_args()

    outdir = ensure_dir(args.output_dir)

    df = load_three_way(args.input)

    print("=" * 104)
    print("NYC H7 EXTERNAL VALIDATION AUDIT")
    print("=" * 104)
    print(f"Input                         : {args.input}")
    print(f"Union OD pairs                : {len(df):,}")
    print(
        f"Observed positive pairs       : "
        f"{(df['observed_flow'] > 0).sum():,}"
    )
    print(
        f"Continuous positive pairs     : "
        f"{(df['continuous_flow'] > 0).sum():,}"
    )
    print(
        f"H3 positive pairs             : "
        f"{(df['h3_flow'] > 0).sum():,}"
    )
    print(
        f"Observed flow                 : "
        f"{df['observed_flow'].sum():,.0f}"
    )
    print(
        f"Continuous flow               : "
        f"{df['continuous_flow'].sum():,.0f}"
    )
    print(
        f"H3 flow                       : "
        f"{df['h3_flow'].sum():,.0f}"
    )

    # 1. Support
    support = support_audit(df)
    support_path = outdir / "NYC_H7_support_audit.csv"
    support.to_csv(
        support_path,
        index=False,
    )

    # 2. Top-flow
    topflow = topflow_audit(
        df,
        thresholds=args.top_flow_thresholds,
    )
    topflow_path = outdir / "NYC_H7_topflow_audit.csv"
    topflow.to_csv(
        topflow_path,
        index=False,
    )

    # 3. Marginals
    origin = build_marginal_table(
        df,
        "origin_h3",
    )
    destination = build_marginal_table(
        df,
        "destination_h3",
    )

    origin_path = outdir / "NYC_H7_origin_marginals.csv"
    destination_path = outdir / "NYC_H7_destination_marginals.csv"

    origin.to_csv(
        origin_path,
        index=False,
    )
    destination.to_csv(
        destination_path,
        index=False,
    )

    marginal_stats = marginal_summary(
        origin,
        destination,
    )
    marginal_path = outdir / "NYC_H7_marginal_summary.csv"
    marginal_stats.to_csv(
        marginal_path,
        index=False,
    )

    # 4. Directionality
    direction = directionality_audit(df)
    direction_path = outdir / "NYC_H7_directionality_audit.csv"
    direction.to_csv(
        direction_path,
        index=False,
    )

    # 5. Compact summary
    audit_summary = build_audit_summary(
        df,
        support,
        topflow,
        marginal_stats,
        direction,
    )
    audit_summary_path = outdir / "NYC_H7_audit_summary.csv"
    audit_summary.to_csv(
        audit_summary_path,
        index=False,
    )

    # Console
    print("\n" + "=" * 104)
    print("SUPPORT AUDIT")
    print("=" * 104)
    print(
        support[
            [
                "support",
                "n_pairs",
                "observed_flow_share_pct",
                "continuous_pearson",
                "h3_pearson",
                "continuous_tvd",
                "h3_tvd",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 104)
    print("TOP OBSERVED-FLOW AUDIT")
    print("=" * 104)
    print(
        topflow[
            [
                "target_observed_flow_share",
                "actual_observed_flow_share",
                "n_pairs",
                "continuous_nonzero_pairs",
                "continuous_pearson",
                "h3_pearson",
                "continuous_tvd",
                "h3_tvd",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 104)
    print("MARGINAL AUDIT")
    print("=" * 104)
    print(
        marginal_stats.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 104)
    print("DIRECTIONALITY AUDIT")
    print("=" * 104)
    print(
        direction.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print_interpretive_flags(
        audit_summary
    )

    print("\n" + "=" * 104)
    print("FILES SAVED")
    print("=" * 104)
    for path in [
        audit_summary_path,
        support_path,
        topflow_path,
        origin_path,
        destination_path,
        marginal_path,
        direction_path,
    ]:
        print(f"[saved] {path.resolve()}")


if __name__ == "__main__":
    main()
