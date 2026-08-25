#!/usr/bin/env python3
"""
08b_nyc_workplace_destination_audit.py

Diagnostic audit for the low NYC H7 external-validation Pearson.

PURPOSE
-------
The cross-city external OD validation shows substantially lower absolute
OD-pair agreement for New York City than for Cambridge or Madrid:

    NYC H7 Pearson ≈ 0.27

This script investigates two possible explanations WITHOUT changing the
external-validation protocol:

1. Destination-structure mismatch
   --------------------------------
   Does the simulated workplace distribution reproduce the observed
   LODES workplace inflow structure at H7?

   We compare:
       - destination marginals
       - Pearson / Spearman
       - TVD / JSD / cosine similarity
       - top-K workplace-cell overlap
       - concentration in top destinations
       - rank agreement among important destinations

2. Finite-sample / sparsity ceiling
   ---------------------------------
   The observed NYC LODES matrix contains millions of workers, whereas
   the paired simulation contains only ~11k work trips.

   We repeatedly draw synthetic samples of the SAME size as the simulated
   work-trip sample directly from the observed H7 OD distribution.

   Each Monte Carlo sample is then compared with the full observed
   distribution using exactly the same metrics.

   This provides an empirical reference distribution for the agreement
   that can be expected purely from finite sampling.

IMPORTANT
---------
This is a diagnostic audit. It does NOT replace the main external
validation and must not be used to retrospectively select a more
favourable validation protocol.

Expected input:
    outputs/external_od_validation/new_york_ny/three_way_od_h7.csv

Expected columns are detected flexibly.

Outputs:
    outputs/external_od_validation/new_york_ny/audit_08b/

Author: diagnostic analysis for H3 urban benchmark
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_INPUT = Path(
    "outputs/external_od_validation/new_york_ny/three_way_od_h7.csv"
)

DEFAULT_OUTPUT = Path(
    "outputs/external_od_validation/new_york_ny/audit_08b"
)

DEFAULT_MC_REPS = 1000
DEFAULT_SEED = 42

TOP_K_VALUES = (5, 10, 20, 50, 100)


# =============================================================================
# COLUMN DETECTION
# =============================================================================

ORIGIN_CANDIDATES = (
    "origin_h3",
    "origin",
    "o_h3",
    "home_h3",
    "h3_origin",
    "origin_cell",
)

DESTINATION_CANDIDATES = (
    "destination_h3",
    "destination",
    "dest_h3",
    "d_h3",
    "work_h3",
    "h3_destination",
    "destination_cell",
)

OBSERVED_CANDIDATES = (
    "observed_flow",
    "observed",
    "obs_flow",
    "flow_observed",
    "observed_count",
    "obs",
)

CONTINUOUS_CANDIDATES = (
    "continuous_flow",
    "continuous",
    "continuous_count",
    "flow_continuous",
    "cont_flow",
    "cont",
)

H3_CANDIDATES = (
    "h3_flow",
    "h3",
    "h3_count",
    "flow_h3",
)


def find_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    description: str,
) -> str:
    """Find a column using exact and case-insensitive candidate matching."""

    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

        candidate_lower = candidate.lower()

        if candidate_lower in lower_map:
            return lower_map[candidate_lower]

    raise KeyError(
        f"Could not identify {description} column.\n"
        f"Available columns:\n{list(df.columns)}"
    )


# =============================================================================
# BASIC METRICS
# =============================================================================

def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = pd.Series(np.asarray(x, dtype=float))
    y = pd.Series(np.asarray(y, dtype=float))

    if len(x) < 2:
        return np.nan

    if x.nunique() <= 1 or y.nunique() <= 1:
        return np.nan

    return float(x.corr(y, method="spearman"))


def normalize_distribution(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    x = np.where(
        np.isfinite(x),
        x,
        0.0,
    )

    x = np.clip(
        x,
        0.0,
        None,
    )

    total = x.sum()

    if total <= 0:
        return np.zeros_like(x)

    return x / total


def tvd(x: np.ndarray, y: np.ndarray) -> float:
    p = normalize_distribution(x)
    q = normalize_distribution(y)

    return float(
        0.5 * np.abs(p - q).sum()
    )


def jsd(x: np.ndarray, y: np.ndarray) -> float:
    """
    Jensen-Shannon divergence normalized to [0, 1] using log base 2.
    """

    p = normalize_distribution(x)
    q = normalize_distribution(y)

    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0

        if not np.any(mask):
            return 0.0

        return float(
            np.sum(
                a[mask] *
                np.log2(
                    a[mask] / b[mask]
                )
            )
        )

    return float(
        0.5 * kl(p, m) +
        0.5 * kl(q, m)
    )


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    denom = (
        np.linalg.norm(x) *
        np.linalg.norm(y)
    )

    if denom == 0:
        return np.nan

    return float(
        np.dot(x, y) / denom
    )


def metric_bundle(
    observed: np.ndarray,
    simulated: np.ndarray,
) -> dict:
    return {
        "pearson": safe_pearson(
            observed,
            simulated,
        ),
        "spearman": safe_spearman(
            observed,
            simulated,
        ),
        "tvd": tvd(
            observed,
            simulated,
        ),
        "jsd": jsd(
            observed,
            simulated,
        ),
        "cosine": cosine_similarity(
            observed,
            simulated,
        ),
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_three_way_od(path: Path):
    print(f"[load] {path}")

    df = pd.read_csv(path)

    origin_col = find_column(
        df,
        ORIGIN_CANDIDATES,
        "origin H3",
    )

    destination_col = find_column(
        df,
        DESTINATION_CANDIDATES,
        "destination H3",
    )

    observed_col = find_column(
        df,
        OBSERVED_CANDIDATES,
        "observed flow",
    )

    continuous_col = find_column(
        df,
        CONTINUOUS_CANDIDATES,
        "continuous flow",
    )

    h3_col = find_column(
        df,
        H3_CANDIDATES,
        "H3 flow",
    )

    for col in (
        observed_col,
        continuous_col,
        h3_col,
    ):
        df[col] = (
            pd.to_numeric(
                df[col],
                errors="coerce",
            )
            .fillna(0.0)
            .astype(float)
        )

    return (
        df,
        origin_col,
        destination_col,
        observed_col,
        continuous_col,
        h3_col,
    )


# =============================================================================
# DESTINATION MARGINAL AUDIT
# =============================================================================

def build_destination_marginals(
    df: pd.DataFrame,
    destination_col: str,
    observed_col: str,
    continuous_col: str,
    h3_col: str,
) -> pd.DataFrame:

    marginals = (
        df.groupby(
            destination_col,
            as_index=False,
        )[
            [
                observed_col,
                continuous_col,
                h3_col,
            ]
        ]
        .sum()
        .rename(
            columns={
                destination_col: "destination_h3",
                observed_col: "observed_flow",
                continuous_col: "continuous_flow",
                h3_col: "h3_flow",
            }
        )
    )

    observed_total = marginals[
        "observed_flow"
    ].sum()

    continuous_total = marginals[
        "continuous_flow"
    ].sum()

    h3_total = marginals[
        "h3_flow"
    ].sum()

    marginals[
        "observed_share"
    ] = (
        marginals["observed_flow"] /
        observed_total
        if observed_total > 0
        else 0.0
    )

    marginals[
        "continuous_share"
    ] = (
        marginals["continuous_flow"] /
        continuous_total
        if continuous_total > 0
        else 0.0
    )

    marginals[
        "h3_share"
    ] = (
        marginals["h3_flow"] /
        h3_total
        if h3_total > 0
        else 0.0
    )

    marginals[
        "observed_rank"
    ] = (
        marginals["observed_flow"]
        .rank(
            method="min",
            ascending=False,
        )
    )

    marginals[
        "continuous_rank"
    ] = (
        marginals["continuous_flow"]
        .rank(
            method="min",
            ascending=False,
        )
    )

    marginals[
        "h3_rank"
    ] = (
        marginals["h3_flow"]
        .rank(
            method="min",
            ascending=False,
        )
    )

    return marginals


def destination_metric_summary(
    marginals: pd.DataFrame,
) -> pd.DataFrame:

    observed = marginals[
        "observed_flow"
    ].to_numpy()

    continuous = marginals[
        "continuous_flow"
    ].to_numpy()

    h3 = marginals[
        "h3_flow"
    ].to_numpy()

    rows = []

    for name, values in (
        ("continuous", continuous),
        ("h3", h3),
    ):
        metrics = metric_bundle(
            observed,
            values,
        )

        rows.append(
            {
                "representation": name,
                "n_destination_cells": len(
                    marginals
                ),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# TOP-K AUDIT
# =============================================================================

def top_k_set(
    df: pd.DataFrame,
    flow_col: str,
    k: int,
) -> set:

    return set(
        df.nlargest(
            k,
            flow_col,
        )["destination_h3"]
    )


def top_k_overlap(
    a: set,
    b: set,
) -> float:

    if not a:
        return np.nan

    return len(a & b) / len(a)


def top_k_audit(
    marginals: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for k in TOP_K_VALUES:

        actual_k = min(
            k,
            len(marginals),
        )

        observed_top = top_k_set(
            marginals,
            "observed_flow",
            actual_k,
        )

        continuous_top = top_k_set(
            marginals,
            "continuous_flow",
            actual_k,
        )

        h3_top = top_k_set(
            marginals,
            "h3_flow",
            actual_k,
        )

        observed_top_df = marginals[
            marginals[
                "destination_h3"
            ].isin(observed_top)
        ]

        observed_concentration = (
            observed_top_df[
                "observed_share"
            ].sum()
        )

        continuous_mass_on_observed_top = (
            observed_top_df[
                "continuous_share"
            ].sum()
        )

        h3_mass_on_observed_top = (
            observed_top_df[
                "h3_share"
            ].sum()
        )

        rows.append(
            {
                "k": actual_k,

                "continuous_topk_overlap":
                    top_k_overlap(
                        observed_top,
                        continuous_top,
                    ),

                "h3_topk_overlap":
                    top_k_overlap(
                        observed_top,
                        h3_top,
                    ),

                "observed_flow_concentration":
                    observed_concentration,

                "continuous_mass_on_observed_topk":
                    continuous_mass_on_observed_top,

                "h3_mass_on_observed_topk":
                    h3_mass_on_observed_top,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# TOP DESTINATION TABLE
# =============================================================================

def build_top_destination_table(
    marginals: pd.DataFrame,
    n: int = 50,
) -> pd.DataFrame:

    top = (
        marginals
        .sort_values(
            "observed_flow",
            ascending=False,
        )
        .head(n)
        .copy()
    )

    top[
        "continuous_share_ratio_to_observed"
    ] = np.where(
        top["observed_share"] > 0,
        top["continuous_share"] /
        top["observed_share"],
        np.nan,
    )

    top[
        "h3_share_ratio_to_observed"
    ] = np.where(
        top["observed_share"] > 0,
        top["h3_share"] /
        top["observed_share"],
        np.nan,
    )

    return top


# =============================================================================
# MONTE CARLO SAMPLE-SIZE CEILING
# =============================================================================

def multinomial_sample(
    rng: np.random.Generator,
    probabilities: np.ndarray,
    n: int,
) -> np.ndarray:

    return rng.multinomial(
        n,
        probabilities,
    ).astype(float)


def monte_carlo_sampling_ceiling(
    df: pd.DataFrame,
    observed_col: str,
    continuous_col: str,
    h3_col: str,
    reps: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:

    observed = (
        df[observed_col]
        .to_numpy(dtype=float)
    )

    continuous = (
        df[continuous_col]
        .to_numpy(dtype=float)
    )

    h3_values = (
        df[h3_col]
        .to_numpy(dtype=float)
    )

    probabilities = normalize_distribution(
        observed
    )

    if probabilities.sum() <= 0:
        raise RuntimeError(
            "Observed OD distribution has zero total flow."
        )

    n_continuous = int(
        round(
            continuous.sum()
        )
    )

    n_h3 = int(
        round(
            h3_values.sum()
        )
    )

    if n_continuous <= 0:
        raise RuntimeError(
            "Continuous simulated flow is zero."
        )

    if n_h3 <= 0:
        raise RuntimeError(
            "H3 simulated flow is zero."
        )

    if n_continuous != n_h3:
        print(
            "[warning] Continuous and H3 total "
            "simulated flows differ:"
        )
        print(
            f"          Continuous = {n_continuous:,}"
        )
        print(
            f"          H3         = {n_h3:,}"
        )

    # Main reference uses Continuous sample size.
    sample_size = n_continuous

    rng = np.random.default_rng(
        seed
    )

    rows = []

    print()
    print(
        f"[Monte Carlo] repetitions : {reps:,}"
    )
    print(
        f"[Monte Carlo] sample size : {sample_size:,}"
    )
    print(
        f"[Monte Carlo] OD support  : {len(observed):,}"
    )

    for rep in range(reps):

        sample = multinomial_sample(
            rng,
            probabilities,
            sample_size,
        )

        metrics = metric_bundle(
            observed,
            sample,
        )

        rows.append(
            {
                "replicate": rep + 1,
                "sample_size": sample_size,
                "positive_od_pairs": int(
                    np.count_nonzero(sample)
                ),
                **metrics,
            }
        )

        if (
            (rep + 1) % 100 == 0
            or rep == 0
            or rep + 1 == reps
        ):
            print(
                f"  replicate {rep + 1:,}/{reps:,}"
            )

    mc = pd.DataFrame(rows)

    actual_continuous = metric_bundle(
        observed,
        continuous,
    )

    actual_h3 = metric_bundle(
        observed,
        h3_values,
    )

    summary = {
        "sample_size": sample_size,
        "repetitions": reps,

        "mc_pearson_mean":
            float(mc["pearson"].mean()),

        "mc_pearson_std":
            float(mc["pearson"].std()),

        "mc_pearson_p025":
            float(mc["pearson"].quantile(0.025)),

        "mc_pearson_p50":
            float(mc["pearson"].quantile(0.50)),

        "mc_pearson_p975":
            float(mc["pearson"].quantile(0.975)),

        "mc_tvd_mean":
            float(mc["tvd"].mean()),

        "mc_tvd_p025":
            float(mc["tvd"].quantile(0.025)),

        "mc_tvd_p975":
            float(mc["tvd"].quantile(0.975)),

        "actual_continuous_pearson":
            actual_continuous["pearson"],

        "actual_h3_pearson":
            actual_h3["pearson"],

        "actual_continuous_tvd":
            actual_continuous["tvd"],

        "actual_h3_tvd":
            actual_h3["tvd"],
    }

    # Empirical percentile:
    #
    # Fraction of LODES-resampled simulations whose Pearson is
    # <= actual model Pearson.
    #
    # Small value means model agreement is substantially below what
    # sampling alone would produce.

    summary[
        "continuous_pearson_mc_percentile"
    ] = float(
        (
            mc["pearson"] <=
            actual_continuous["pearson"]
        ).mean()
    )

    summary[
        "h3_pearson_mc_percentile"
    ] = float(
        (
            mc["pearson"] <=
            actual_h3["pearson"]
        ).mean()
    )

    # For TVD, lower is better.
    #
    # Fraction of Monte Carlo samples with TVD >= actual model TVD.
    # If this is tiny, actual model TVD is worse than expected from
    # sampling alone.

    summary[
        "continuous_tvd_mc_upper_tail"
    ] = float(
        (
            mc["tvd"] >=
            actual_continuous["tvd"]
        ).mean()
    )

    summary[
        "h3_tvd_mc_upper_tail"
    ] = float(
        (
            mc["tvd"] >=
            actual_h3["tvd"]
        ).mean()
    )

    return mc, summary


# =============================================================================
# DESTINATION-MARGINAL MONTE CARLO
# =============================================================================

def monte_carlo_destination_ceiling(
    df: pd.DataFrame,
    destination_col: str,
    observed_col: str,
    continuous_col: str,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    """
    Same Monte Carlo idea, but evaluated on destination marginals.

    This answers:

        If we draw ~11k trips perfectly from LODES,
        how accurately should the workplace inflow distribution
        itself be recoverable?
    """

    observed_od = (
        df[observed_col]
        .to_numpy(dtype=float)
    )

    probabilities = normalize_distribution(
        observed_od
    )

    sample_size = int(
        round(
            df[continuous_col].sum()
        )
    )

    destinations = (
        df[destination_col]
        .astype(str)
        .to_numpy()
    )

    destination_codes, inverse = np.unique(
        destinations,
        return_inverse=True,
    )

    observed_destination = np.bincount(
        inverse,
        weights=observed_od,
        minlength=len(destination_codes),
    )

    rng = np.random.default_rng(
        seed + 1000
    )

    rows = []

    for rep in range(reps):

        sample_od = rng.multinomial(
            sample_size,
            probabilities,
        )

        sample_destination = np.bincount(
            inverse,
            weights=sample_od,
            minlength=len(destination_codes),
        )

        metrics = metric_bundle(
            observed_destination,
            sample_destination,
        )

        rows.append(
            {
                "replicate": rep + 1,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# INTERPRETATION
# =============================================================================

def classify_result(
    mc_summary: dict,
    destination_summary: pd.DataFrame,
) -> dict:

    mc_median = (
        mc_summary[
            "mc_pearson_p50"
        ]
    )

    actual_cont = (
        mc_summary[
            "actual_continuous_pearson"
        ]
    )

    actual_h3 = (
        mc_summary[
            "actual_h3_pearson"
        ]
    )

    mc_low = (
        mc_summary[
            "mc_pearson_p025"
        ]
    )

    continuous_percentile = (
        mc_summary[
            "continuous_pearson_mc_percentile"
        ]
    )

    h3_percentile = (
        mc_summary[
            "h3_pearson_mc_percentile"
        ]
    )

    dest_cont = float(
        destination_summary.loc[
            destination_summary[
                "representation"
            ] == "continuous",
            "pearson",
        ].iloc[0]
    )

    dest_h3 = float(
        destination_summary.loc[
            destination_summary[
                "representation"
            ] == "h3",
            "pearson",
        ].iloc[0]
    )

    if (
        actual_cont >= mc_low
        and actual_h3 >= mc_low
    ):
        sampling_conclusion = (
            "The observed NYC H7 Pearson lies within the "
            "95% Monte Carlo range expected from finite sampling. "
            "Sample size/sparsity can plausibly explain much of the "
            "low absolute OD-pair correlation."
        )

    elif (
        continuous_percentile < 0.025
        and h3_percentile < 0.025
    ):
        sampling_conclusion = (
            "Both Continuous and H3 Pearson values fall below the "
            "2.5th percentile of the finite-sampling reference. "
            "The low NYC OD-pair correlation therefore cannot be "
            "explained by sample size alone; a genuine behavioural, "
            "destination-choice, boundary, or source-model mismatch "
            "remains."
        )

    else:
        sampling_conclusion = (
            "Finite sampling explains part, but not necessarily all, "
            "of the NYC H7 discrepancy. The result is intermediate "
            "relative to the Monte Carlo reference."
        )

    if (
        dest_cont > 0.8
        and dest_h3 > 0.8
    ):
        destination_conclusion = (
            "Destination marginals are strongly reproduced. "
            "The remaining weakness is primarily in OD pairing "
            "rather than aggregate workplace geography."
        )

    elif (
        dest_cont < 0.6
        and dest_h3 < 0.6
    ):
        destination_conclusion = (
            "Destination marginals remain weak. "
            "This points to a substantive workplace-destination "
            "distribution mismatch, not merely OD-pair sparsity."
        )

    else:
        destination_conclusion = (
            "Destination marginals show intermediate agreement. "
            "Both destination geography and OD pairing may contribute "
            "to the low absolute NYC OD correlation."
        )

    architectural_conclusion = (
        "Continuous and H3 should still be interpreted differentially: "
        "if both remain close to each other across these diagnostics, "
        "the audit does not indicate an H3-specific degradation even "
        "when absolute empirical fit is limited."
    )

    return {
        "mc_median_pearson": mc_median,
        "continuous_actual_pearson": actual_cont,
        "h3_actual_pearson": actual_h3,
        "sampling_conclusion": sampling_conclusion,
        "destination_conclusion": destination_conclusion,
        "architectural_conclusion": architectural_conclusion,
    }


# =============================================================================
# PRINT HELPERS
# =============================================================================

def banner(title: str):
    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def print_dataframe(
    df: pd.DataFrame,
    digits: int = 4,
):
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.float_format",
        lambda x: f"{x:.{digits}f}",
    ):
        print(
            df.to_string(
                index=False
            )
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Audit low NYC H7 external-validation Pearson using "
            "workplace-destination diagnostics and Monte Carlo "
            "finite-sample reference distributions."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "NYC three_way_od_h7.csv generated by "
            "07_external_od_validation.py"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--mc-reps",
        type=int,
        default=DEFAULT_MC_REPS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        df,
        origin_col,
        destination_col,
        observed_col,
        continuous_col,
        h3_col,
    ) = load_three_way_od(
        args.input
    )

    # -------------------------------------------------------------------------
    # BASIC AUDIT
    # -------------------------------------------------------------------------

    banner(
        "NYC H7 WORKPLACE DESTINATION / SAMPLE-SIZE AUDIT"
    )

    print(
        f"Input                       : {args.input}"
    )
    print(
        f"OD union pairs              : {len(df):,}"
    )
    print(
        f"Origin column               : {origin_col}"
    )
    print(
        f"Destination column          : {destination_col}"
    )
    print(
        f"Observed column             : {observed_col}"
    )
    print(
        f"Continuous column           : {continuous_col}"
    )
    print(
        f"H3 column                   : {h3_col}"
    )
    print(
        f"Observed total flow         : {df[observed_col].sum():,.0f}"
    )
    print(
        f"Continuous simulated flow   : {df[continuous_col].sum():,.0f}"
    )
    print(
        f"H3 simulated flow           : {df[h3_col].sum():,.0f}"
    )

    # -------------------------------------------------------------------------
    # DESTINATION MARGINALS
    # -------------------------------------------------------------------------

    marginals = build_destination_marginals(
        df,
        destination_col,
        observed_col,
        continuous_col,
        h3_col,
    )

    destination_summary = (
        destination_metric_summary(
            marginals
        )
    )

    banner(
        "WORKPLACE DESTINATION MARGINAL AUDIT"
    )

    print_dataframe(
        destination_summary
    )

    # -------------------------------------------------------------------------
    # TOP-K
    # -------------------------------------------------------------------------

    topk = top_k_audit(
        marginals
    )

    banner(
        "TOP-K WORKPLACE DESTINATION AUDIT"
    )

    print_dataframe(
        topk
    )

    # -------------------------------------------------------------------------
    # TOP DESTINATIONS
    # -------------------------------------------------------------------------

    top_destinations = (
        build_top_destination_table(
            marginals,
            n=50,
        )
    )

    banner(
        "TOP 20 OBSERVED WORKPLACE DESTINATIONS"
    )

    print_dataframe(
        top_destinations.head(20)
    )

    # -------------------------------------------------------------------------
    # MONTE CARLO OD PAIR CEILING
    # -------------------------------------------------------------------------

    banner(
        "MONTE CARLO FINITE-SAMPLE OD-PAIR REFERENCE"
    )

    mc_od, mc_summary = (
        monte_carlo_sampling_ceiling(
            df=df,
            observed_col=observed_col,
            continuous_col=continuous_col,
            h3_col=h3_col,
            reps=args.mc_reps,
            seed=args.seed,
        )
    )

    mc_summary_df = pd.DataFrame(
        [mc_summary]
    )

    banner(
        "MONTE CARLO OD-PAIR SUMMARY"
    )

    print_dataframe(
        mc_summary_df
    )

    # -------------------------------------------------------------------------
    # DESTINATION MARGINAL MONTE CARLO
    # -------------------------------------------------------------------------

    banner(
        "MONTE CARLO DESTINATION-MARGINAL REFERENCE"
    )

    mc_destination = (
        monte_carlo_destination_ceiling(
            df=df,
            destination_col=destination_col,
            observed_col=observed_col,
            continuous_col=continuous_col,
            reps=args.mc_reps,
            seed=args.seed,
        )
    )

    mc_destination_summary = pd.DataFrame(
        [
            {
                "pearson_mean":
                    mc_destination[
                        "pearson"
                    ].mean(),

                "pearson_p025":
                    mc_destination[
                        "pearson"
                    ].quantile(0.025),

                "pearson_p50":
                    mc_destination[
                        "pearson"
                    ].quantile(0.50),

                "pearson_p975":
                    mc_destination[
                        "pearson"
                    ].quantile(0.975),

                "tvd_mean":
                    mc_destination[
                        "tvd"
                    ].mean(),

                "tvd_p025":
                    mc_destination[
                        "tvd"
                    ].quantile(0.025),

                "tvd_p975":
                    mc_destination[
                        "tvd"
                    ].quantile(0.975),
            }
        ]
    )

    print_dataframe(
        mc_destination_summary
    )

    # -------------------------------------------------------------------------
    # INTERPRETATION
    # -------------------------------------------------------------------------

    interpretation = classify_result(
        mc_summary,
        destination_summary,
    )

    banner(
        "DIAGNOSTIC INTERPRETATION"
    )

    print(
        "[1] FINITE-SAMPLE TEST"
    )
    print(
        interpretation[
            "sampling_conclusion"
        ]
    )

    print()
    print(
        "[2] DESTINATION STRUCTURE"
    )
    print(
        interpretation[
            "destination_conclusion"
        ]
    )

    print()
    print(
        "[3] H3 ARCHITECTURAL INTERPRETATION"
    )
    print(
        interpretation[
            "architectural_conclusion"
        ]
    )

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    files = {
        "NYC_H7_destination_marginals.csv":
            marginals,

        "NYC_H7_destination_metric_summary.csv":
            destination_summary,

        "NYC_H7_destination_topk_audit.csv":
            topk,

        "NYC_H7_top50_workplace_destinations.csv":
            top_destinations,

        "NYC_H7_monte_carlo_od_pair.csv":
            mc_od,

        "NYC_H7_monte_carlo_od_pair_summary.csv":
            mc_summary_df,

        "NYC_H7_monte_carlo_destination.csv":
            mc_destination,

        "NYC_H7_monte_carlo_destination_summary.csv":
            mc_destination_summary,
    }

    banner(
        "FILES SAVED"
    )

    for filename, table in files.items():

        path = (
            args.output_dir /
            filename
        )

        table.to_csv(
            path,
            index=False,
        )

        print(
            f"[saved] {path.resolve()}"
        )

    interpretation_path = (
        args.output_dir /
        "NYC_H7_audit_interpretation.json"
    )

    with interpretation_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            interpretation,
            f,
            indent=2,
        )

    print(
        f"[saved] {interpretation_path.resolve()}"
    )

    banner(
        "DONE"
    )

    print(
        "This audit is diagnostic only. "
        "Do not replace or modify the frozen external-validation "
        "results on the basis of this analysis."
    )


if __name__ == "__main__":
    main()