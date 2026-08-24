#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h3uab.city_config import config_for_city
from h3uab.world import build_world
from h3uab.continuous_indexed_backend import ContinuousIndexedBackend
from h3uab.h3_backend import MultiLayerH3Backend
from h3uab.semantic_layers import id_to_code


CITY_CHOICES = ("madrid", "cambridge_ma", "new_york_ny")
STATIC_MODES = ("cold", "warm", "both")


def active_steps(world):
    return [
        s
        for s in range(world.profile_schedule_layer_ids.shape[1])
        if np.any(world.profile_schedule_layer_ids[:, s] > 0)
    ]


def profile_value(backend, name, default=np.nan):
    profile = getattr(backend, "profile", None)
    return getattr(profile, name, default)


def prepare_backend(backend, world, cfg):
    """Measure backend.prepare() from outside and capture internal profiling."""
    t0 = perf_counter()
    backend.prepare(world, cfg)
    wall_s = perf_counter() - t0

    return {
        "wall_s": float(wall_s),
        "profile_s": float(profile_value(
            backend,
            "initialization_time_s",
            np.nan,
        )),
        "static_city_load_s": float(profile_value(
            backend,
            "static_city_load_time_s",
            np.nan,
        )),
        "static_city_mode": getattr(
            backend,
            "static_city_mode",
            None,
        ),
    }


def run_temporal_loop(world, backend):
    """
    Genuine temporal decision loop.

    Includes:
      - schedule traversal
      - backend.decide(...)
      - destination movement
      - H3 dynamic H12 maintenance via move_to_coords(...)

    Excludes:
      - world construction
      - physical OSM routing (same downstream infrastructure)
    """
    lat = world.agent_lat.copy()
    lng = world.agent_lng.copy()

    decisions = 0
    valid_destinations = 0

    t0 = perf_counter()

    for step in active_steps(world):
        for profile_id in np.unique(world.agent_profile_type):
            layer_id = int(
                world.profile_schedule_layer_ids[
                    int(profile_id),
                    int(step),
                ]
            )

            if layer_id <= 0:
                continue

            layer = id_to_code(layer_id)

            agent_ids = np.flatnonzero(
                world.agent_profile_type == profile_id
            )

            for aid_raw in agent_ids:
                aid = int(aid_raw)

                decision = backend.decide(
                    aid,
                    float(lat[aid]),
                    float(lng[aid]),
                    layer,
                )

                decisions += 1

                if decision.destination_id >= 0:
                    valid_destinations += 1

                    lat[aid] = world.amenity_lat[
                        decision.destination_id
                    ]
                    lng[aid] = world.amenity_lng[
                        decision.destination_id
                    ]

                    # Fair end-to-end accounting:
                    # H3 pays its dynamic localization maintenance cost.
                    if backend.name == "h3_multilayer":
                        backend.dynamic.move_to_coords(
                            aid,
                            float(lat[aid]),
                            float(lng[aid]),
                        )

    runtime_s = perf_counter() - t0

    return {
        "runtime_s": float(runtime_s),
        "decisions": int(decisions),
        "valid_destinations": int(valid_destinations),
        "us_per_decision": (
            runtime_s / decisions * 1e6
            if decisions
            else np.nan
        ),
        "decisions_per_s": (
            decisions / runtime_s
            if runtime_s > 0
            else np.nan
        ),
    }


def ensure_warm_h3_cache(world, cfg):
    """
    Ensure the H3 static-city cache exists before a timed warm prepare.

    The warm-up is intentionally NOT included in the warm preparation time.
    """
    warm_cfg = replace(
        cfg,
        rebuild_static_city=False,
    )

    backend = MultiLayerH3Backend()
    info = prepare_backend(
        backend,
        world,
        warm_cfg,
    )

    return info


def measure_pair(world, cfg, static_city_mode):
    """
    Prepare Continuous and H3, then execute one paired temporal loop.

    Modes:
      cold : force H3 static-city rebuild for the measured prepare.
      warm : ensure cache exists outside timing; measured H3 prepare loads cache.
      both : measure BOTH cold-build and warm-load H3 preparation costs,
             but run the temporal H3 loop once using the warm-loaded backend.

    For 'both', conservative break-even uses cold-build H3 preparation.
    """

    # ----------------------------------------------------------
    # Continuous prepare: always rebuilt in memory.
    # ----------------------------------------------------------
    continuous_backend = ContinuousIndexedBackend()
    c_prepare = prepare_backend(
        continuous_backend,
        world,
        cfg,
    )

    h3_cold_prepare = None
    h3_warm_prepare = None

    # ----------------------------------------------------------
    # H3 prepare regime(s)
    # ----------------------------------------------------------
    if static_city_mode == "cold":
        h3_cfg = replace(
            cfg,
            rebuild_static_city=True,
        )

        h3_backend = MultiLayerH3Backend()
        h3_cold_prepare = prepare_backend(
            h3_backend,
            world,
            h3_cfg,
        )

        if h3_cold_prepare["static_city_mode"] != "built":
            raise RuntimeError(
                "Cold mode requested, but H3 static_city_mode was "
                f"{h3_cold_prepare['static_city_mode']!r}, expected 'built'."
            )

        h3_runtime_backend = h3_backend

    elif static_city_mode == "warm":
        # Untimed warm-up. If a cache already exists, this simply loads it;
        # otherwise it builds it once outside the timed warm measurement.
        ensure_warm_h3_cache(
            world,
            cfg,
        )

        h3_cfg = replace(
            cfg,
            rebuild_static_city=False,
        )

        h3_backend = MultiLayerH3Backend()
        h3_warm_prepare = prepare_backend(
            h3_backend,
            world,
            h3_cfg,
        )

        if h3_warm_prepare["static_city_mode"] != "loaded":
            raise RuntimeError(
                "Warm mode requested, but H3 static_city_mode was "
                f"{h3_warm_prepare['static_city_mode']!r}, expected 'loaded'."
            )

        h3_runtime_backend = h3_backend

    elif static_city_mode == "both":
        # 1) Conservative cold-build preparation.
        cold_cfg = replace(
            cfg,
            rebuild_static_city=True,
        )

        h3_cold_backend = MultiLayerH3Backend()
        h3_cold_prepare = prepare_backend(
            h3_cold_backend,
            world,
            cold_cfg,
        )

        if h3_cold_prepare["static_city_mode"] != "built":
            raise RuntimeError(
                "Both mode: cold H3 prepare did not build the static city."
            )

        # 2) Production warm-load preparation, using the cache just created.
        warm_cfg = replace(
            cfg,
            rebuild_static_city=False,
        )

        h3_warm_backend = MultiLayerH3Backend()
        h3_warm_prepare = prepare_backend(
            h3_warm_backend,
            world,
            warm_cfg,
        )

        if h3_warm_prepare["static_city_mode"] != "loaded":
            raise RuntimeError(
                "Both mode: warm H3 prepare did not load the static city."
            )

        # Runtime should be identical in principle once preparation is complete.
        # Run it once using the warm-loaded backend to avoid doubling the expensive
        # temporal simulation solely to distinguish preparation regimes.
        h3_runtime_backend = h3_warm_backend

    else:
        raise ValueError(
            f"Unknown static_city_mode: {static_city_mode}"
        )

    # ----------------------------------------------------------
    # Paired temporal execution
    # ----------------------------------------------------------
    c_run = run_temporal_loop(
        world,
        continuous_backend,
    )

    h_run = run_temporal_loop(
        world,
        h3_runtime_backend,
    )

    if c_run["decisions"] != h_run["decisions"]:
        raise RuntimeError(
            "Continuous and H3 executed different numbers of decisions: "
            f"{c_run['decisions']} vs {h_run['decisions']}"
        )

    return {
        "continuous_prepare": c_prepare,
        "h3_cold_prepare": h3_cold_prepare,
        "h3_warm_prepare": h3_warm_prepare,
        "continuous_run": c_run,
        "h3_run": h_run,
    }


def median_stats(values):
    a = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "std": (
            float(np.std(a, ddof=1))
            if len(a) > 1
            else 0.0
        ),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def add_stats(row, raw_case, prefix, column):
    stats = median_stats(
        raw_case[column].dropna().to_numpy(float)
    )
    for stat_name, value in stats.items():
        row[f"{prefix}_{stat_name}"] = value


def break_even_decisions(
    continuous_prepare_s,
    h3_prepare_s,
    continuous_us_per_decision,
    h3_us_per_decision,
):
    """
    Break-even for the measured temporal loop:

      T_C = T_prepare,C + N * tau_C
      T_H = T_prepare,H + N * tau_H

    Includes H3 H12 maintenance in tau_H.
    """
    delta_prepare = (
        h3_prepare_s
        - continuous_prepare_s
    )

    delta_runtime_per_decision = (
        continuous_us_per_decision
        - h3_us_per_decision
    ) / 1e6

    if delta_runtime_per_decision <= 0:
        return np.nan

    if delta_prepare <= 0:
        return 0.0

    return (
        delta_prepare
        / delta_runtime_per_decision
    )


def break_even_sensitivity_bounds(
    delta_prepare_s,
    continuous_us_min,
    continuous_us_max,
    h3_us_min,
    h3_us_max,
):
    """
    Sensitivity interval for break-even using observed latency extrema.

    The central break-even remains based on aggregated medians.

    Lower break-even bound:
        largest observed per-decision saving
        = max(C latency) - min(H3 latency)

    Upper break-even bound:
        smallest observed positive per-decision saving
        = min(C latency) - max(H3 latency)

    This is deliberately NOT a confidence interval. It is an observed
    min/max sensitivity range induced by wall-clock dispersion.
    """
    delta_tau_high = (
        float(continuous_us_max)
        - float(h3_us_min)
    ) / 1e6

    delta_tau_low = (
        float(continuous_us_min)
        - float(h3_us_max)
    ) / 1e6

    if delta_prepare_s <= 0:
        return 0.0, 0.0

    lower = (
        delta_prepare_s / delta_tau_high
        if delta_tau_high > 0
        else np.nan
    )

    upper = (
        delta_prepare_s / delta_tau_low
        if delta_tau_low > 0
        else np.nan
    )

    return lower, upper


def summarize_case(raw_case, static_city_mode):
    row = {
        "city": str(raw_case["city"].iloc[0]),
        "n_agents": int(raw_case["n_agents"].iloc[0]),
        "repeats": int(len(raw_case)),
        "static_city_experiment_mode": static_city_mode,
        "decisions": int(
            raw_case["continuous_decisions"].iloc[0]
        ),
    }

    # Runtime distributions.
    for backend in ("continuous", "h3"):
        for metric in (
            "runtime_s",
            "us_per_decision",
            "decisions_per_s",
        ):
            add_stats(
                row,
                raw_case,
                f"{backend}_{metric}",
                f"{backend}_{metric}",
            )

    # Continuous preparation distributions:
    # - wall time: external perf_counter around prepare()
    # - profile time: backend-instrumented prepare cost
    add_stats(
        row,
        raw_case,
        "continuous_prepare_s",
        "continuous_prepare_s",
    )
    add_stats(
        row,
        raw_case,
        "continuous_prepare_profile_s",
        "continuous_prepare_profile_s",
    )

    # Optional cold/warm H3 preparation distributions.
    if raw_case["h3_cold_prepare_s"].notna().any():
        add_stats(
            row,
            raw_case,
            "h3_cold_prepare_s",
            "h3_cold_prepare_s",
        )
        add_stats(
            row,
            raw_case,
            "h3_cold_prepare_profile_s",
            "h3_cold_prepare_profile_s",
        )

    if raw_case["h3_warm_prepare_s"].notna().any():
        add_stats(
            row,
            raw_case,
            "h3_warm_prepare_s",
            "h3_warm_prepare_s",
        )
        add_stats(
            row,
            raw_case,
            "h3_warm_prepare_profile_s",
            "h3_warm_prepare_profile_s",
        )

    # Main temporal-loop speedup.
    #
    # IMPORTANT: repeated runs are paired. Therefore the primary
    # speedup statistic is the MEDIAN OF PAIRED RATIOS,
    #
    #   S_r = T_C,r / T_H3,r
    #
    # rather than ratio(median(T_C), median(T_H3)).
    # We also retain min/max/std to expose wall-clock dispersion.
    speedup_stats = median_stats(
        raw_case[
            "runtime_speedup_h3_over_continuous"
        ].to_numpy(float)
    )

    row["runtime_speedup_h3_over_continuous"] = (
        speedup_stats["median"]
    )
    row["runtime_speedup_median"] = (
        speedup_stats["median"]
    )
    row["runtime_speedup_min"] = (
        speedup_stats["min"]
    )
    row["runtime_speedup_max"] = (
        speedup_stats["max"]
    )
    row["runtime_speedup_mean"] = (
        speedup_stats["mean"]
    )
    row["runtime_speedup_std"] = (
        speedup_stats["std"]
    )

    c_runtime = row["continuous_runtime_s_median"]
    h_runtime = row["h3_runtime_s_median"]

    c_prepare = row["continuous_prepare_s_median"]
    c_tau = row["continuous_us_per_decision_median"]
    h_tau = row["h3_us_per_decision_median"]

    decisions_per_agent = (
        row["decisions"] / row["n_agents"]
        if row["n_agents"] > 0
        else np.nan
    )
    row["decisions_per_agent"] = decisions_per_agent

    # Conservative cold-build break-even.
    #
    # Main reported value uses backend-instrumented preparation time,
    # which captures the in-memory build cost without the one-off
    # serialization-to-disk overhead.
    #
    # We ALSO retain an external wall-clock upper bound that includes
    # serialization and any wrapper overhead. This makes the distinction
    # explicit and auditable.
    if "h3_cold_prepare_s_median" in row:
        h3_cold_prepare_wall = row[
            "h3_cold_prepare_s_median"
        ]
        h3_cold_prepare_profile = row[
            "h3_cold_prepare_profile_s_median"
        ]

        c_prepare_profile = row[
            "continuous_prepare_profile_s_median"
        ]

        # Primary cold-build break-even: instrumented in-memory prepare.
        be_cold_profile = break_even_decisions(
            c_prepare_profile,
            h3_cold_prepare_profile,
            c_tau,
            h_tau,
        )

        row["cold_break_even_decisions"] = be_cold_profile
        row["cold_break_even_agents_equivalent"] = (
            be_cold_profile / decisions_per_agent
            if (
                pd.notna(be_cold_profile)
                and decisions_per_agent > 0
            )
            else np.nan
        )

        # Observed sensitivity range based on min/max per-decision latency.
        # This is not a statistical confidence interval.
        be_cold_low, be_cold_high = (
            break_even_sensitivity_bounds(
                delta_prepare_s=(
                    h3_cold_prepare_profile
                    - c_prepare_profile
                ),
                continuous_us_min=row[
                    "continuous_us_per_decision_min"
                ],
                continuous_us_max=row[
                    "continuous_us_per_decision_max"
                ],
                h3_us_min=row[
                    "h3_us_per_decision_min"
                ],
                h3_us_max=row[
                    "h3_us_per_decision_max"
                ],
            )
        )

        row["cold_break_even_decisions_low"] = (
            be_cold_low
        )
        row["cold_break_even_decisions_high"] = (
            be_cold_high
        )
        row[
            "cold_break_even_agents_equivalent_low"
        ] = (
            be_cold_low / decisions_per_agent
            if (
                pd.notna(be_cold_low)
                and decisions_per_agent > 0
            )
            else np.nan
        )
        row[
            "cold_break_even_agents_equivalent_high"
        ] = (
            be_cold_high / decisions_per_agent
            if (
                pd.notna(be_cold_high)
                and decisions_per_agent > 0
            )
            else np.nan
        )

        # External wall-clock upper bound: includes any persistence/wrapper cost.
        be_cold_wall = break_even_decisions(
            c_prepare,
            h3_cold_prepare_wall,
            c_tau,
            h_tau,
        )

        row["cold_break_even_wall_upper_decisions"] = be_cold_wall
        row["cold_break_even_wall_upper_agents_equivalent"] = (
            be_cold_wall / decisions_per_agent
            if (
                pd.notna(be_cold_wall)
                and decisions_per_agent > 0
            )
            else np.nan
        )

        # Cold-start total times under both interpretations.
        row["continuous_total_cold_profile_s"] = (
            c_prepare_profile + c_runtime
        )
        row["h3_total_cold_profile_s"] = (
            h3_cold_prepare_profile + h_runtime
        )

        # Primary cold-total speedup must use paired repeat ratios,
        # exactly like the temporal-loop speedup.
        if (
            "cold_total_speedup_h3_over_continuous"
            in raw_case.columns
            and raw_case[
                "cold_total_speedup_h3_over_continuous"
            ].notna().any()
        ):
            cold_total_stats = median_stats(
                raw_case[
                    "cold_total_speedup_h3_over_continuous"
                ]
                .dropna()
                .to_numpy(float)
            )

            row[
                "cold_total_speedup_profile_h3_over_continuous"
            ] = cold_total_stats["median"]
            row["cold_total_speedup_median"] = cold_total_stats["median"]
            row["cold_total_speedup_min"] = cold_total_stats["min"]
            row["cold_total_speedup_max"] = cold_total_stats["max"]
            row["cold_total_speedup_std"] = cold_total_stats["std"]
        else:
            # Backward-compatible fallback for legacy raw files.
            # This is ratio-of-medians and should not be used as the
            # manuscript headline once paired raw totals are available.
            row[
                "cold_total_speedup_profile_h3_over_continuous"
            ] = (
                row["continuous_total_cold_profile_s"]
                / row["h3_total_cold_profile_s"]
                if row["h3_total_cold_profile_s"] > 0
                else np.nan
            )

        row["continuous_total_cold_wall_s"] = (
            c_prepare + c_runtime
        )
        row["h3_total_cold_wall_s"] = (
            h3_cold_prepare_wall + h_runtime
        )
        row["cold_total_speedup_wall_h3_over_continuous"] = (
            row[
                "cold_total_speedup_profile_h3_over_continuous"
            ]
        )

    # Warm-load operating regime.
    if "h3_warm_prepare_s_median" in row:
        h3_warm_prepare = row[
            "h3_warm_prepare_s_median"
        ]

        be_warm = break_even_decisions(
            c_prepare,
            h3_warm_prepare,
            c_tau,
            h_tau,
        )

        row["warm_break_even_decisions"] = be_warm
        row["warm_break_even_agents_equivalent"] = (
            be_warm / decisions_per_agent
            if (
                pd.notna(be_warm)
                and decisions_per_agent > 0
            )
            else np.nan
        )

        be_warm_low, be_warm_high = (
            break_even_sensitivity_bounds(
                delta_prepare_s=(
                    h3_warm_prepare
                    - c_prepare
                ),
                continuous_us_min=row[
                    "continuous_us_per_decision_min"
                ],
                continuous_us_max=row[
                    "continuous_us_per_decision_max"
                ],
                h3_us_min=row[
                    "h3_us_per_decision_min"
                ],
                h3_us_max=row[
                    "h3_us_per_decision_max"
                ],
            )
        )

        row["warm_break_even_decisions_low"] = (
            be_warm_low
        )
        row["warm_break_even_decisions_high"] = (
            be_warm_high
        )

        row["continuous_total_warm_s"] = (
            c_prepare + c_runtime
        )
        row["h3_total_warm_s"] = (
            h3_warm_prepare + h_runtime
        )

        if (
            "warm_total_speedup_h3_over_continuous"
            in raw_case.columns
            and raw_case[
                "warm_total_speedup_h3_over_continuous"
            ].notna().any()
        ):
            warm_total_stats = median_stats(
                raw_case[
                    "warm_total_speedup_h3_over_continuous"
                ]
                .dropna()
                .to_numpy(float)
            )

            row[
                "warm_total_speedup_h3_over_continuous"
            ] = warm_total_stats["median"]
            row["warm_total_speedup_median"] = warm_total_stats["median"]
            row["warm_total_speedup_min"] = warm_total_stats["min"]
            row["warm_total_speedup_max"] = warm_total_stats["max"]
            row["warm_total_speedup_std"] = warm_total_stats["std"]
        else:
            row[
                "warm_total_speedup_h3_over_continuous"
            ] = (
                row["continuous_total_warm_s"]
                / row["h3_total_warm_s"]
                if row["h3_total_warm_s"] > 0
                else np.nan
            )

    return row


def output_dir_for_city(city):
    if city == "madrid":
        return ROOT / "outputs" / "end_to_end"

    return (
        ROOT
        / "outputs"
        / city
        / "end_to_end"
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Multi-city temporal end-to-end benchmark with explicit "
            "cold-build and warm-cache H3 preparation regimes."
        )
    )

    ap.add_argument(
        "--city",
        choices=CITY_CHOICES,
        default="madrid",
    )

    ap.add_argument(
        "--agents",
        nargs="+",
        type=int,
        default=[10_000, 100_000],
        help="Agent populations to benchmark.",
    )

    ap.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--static-city-mode",
        choices=STATIC_MODES,
        default="both",
        help=(
            "'cold' forces H3 rebuild; "
            "'warm' measures cache load after untimed warm-up; "
            "'both' measures both preparation regimes and runs the "
            "temporal loop once per repeat."
        ),
    )

    ap.add_argument(
        "--summarize-only",
        action="store_true",
        help=(
            "Do not run simulations. Rebuild the summary CSV from the "
            "existing mode-specific raw CSV using the current statistics."
        ),
    )

    args = ap.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    out_dir = output_dir_for_city(
        args.city
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Output strategy: canonical per-N files + aggregate files
    # ------------------------------------------------------------------
    suffix = args.static_city_mode

    aggregate_raw_path = (
        out_dir
        / f"end_to_end_raw_{suffix}.csv"
    )
    aggregate_summary_path = (
        out_dir
        / f"end_to_end_summary_{suffix}.csv"
    )

    def per_n_raw_path(n_agents: int) -> Path:
        return (
            out_dir
            / f"end_to_end_raw_{suffix}_n{int(n_agents)}.csv"
        )

    def per_n_summary_path(n_agents: int) -> Path:
        return (
            out_dir
            / f"end_to_end_summary_{suffix}_n{int(n_agents)}.csv"
        )

    def discover_per_n_raw_files():
        return sorted(
            out_dir.glob(
                f"end_to_end_raw_{suffix}_n*.csv"
            ),
            key=lambda p: int(
                p.stem.rsplit("_n", 1)[-1]
            ),
        )

    def rebuild_aggregate_outputs():
        """
        Rebuild aggregate raw/summary CSVs from all canonical per-N raw files.
        Running one population can therefore never erase another population.
        """
        files = discover_per_n_raw_files()

        if not files:
            raise FileNotFoundError(
                f"No canonical per-N raw files found in {out_dir}"
            )

        raw_frames = []
        summary_frames = []

        for p in files:
            df_n = pd.read_csv(p)

            if df_n.empty:
                continue

            n_values = (
                pd.to_numeric(
                    df_n["n_agents"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .unique()
            )

            if len(n_values) != 1:
                raise ValueError(
                    f"{p} must contain exactly one N; got "
                    f"{n_values.tolist()}"
                )

            n_value = int(n_values[0])

            raw_frames.append(df_n)

            summary_n = summarize_case(
                df_n.copy(),
                args.static_city_mode,
            )

            summary_df_n = pd.DataFrame(
                [summary_n]
            )

            summary_df_n.to_csv(
                per_n_summary_path(n_value),
                index=False,
            )

            summary_frames.append(
                summary_df_n
            )

        if not raw_frames:
            raise RuntimeError(
                "Canonical per-N files exist but contain no rows."
            )

        aggregate_raw = (
            pd.concat(
                raw_frames,
                ignore_index=True,
            )
            .sort_values(
                ["n_agents", "repeat"]
            )
            .reset_index(drop=True)
        )

        aggregate_summary = (
            pd.concat(
                summary_frames,
                ignore_index=True,
            )
            .sort_values("n_agents")
            .reset_index(drop=True)
        )

        aggregate_raw.to_csv(
            aggregate_raw_path,
            index=False,
        )
        aggregate_summary.to_csv(
            aggregate_summary_path,
            index=False,
        )

        return aggregate_raw, aggregate_summary

    def migrate_legacy_aggregate_if_needed():
        """
        Split an old aggregate raw file into canonical per-N files.

        This preserves whichever populations still exist in outputs produced
        by older versions before any new benchmark is executed.
        Existing canonical per-N files are never overwritten by migration.
        """
        if not aggregate_raw_path.exists():
            return []

        legacy = pd.read_csv(
            aggregate_raw_path
        )

        if legacy.empty:
            return []

        migrated = []

        n_values = sorted(
            pd.to_numeric(
                legacy["n_agents"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        for n_value in n_values:
            dst_raw = per_n_raw_path(
                n_value
            )

            if dst_raw.exists():
                continue

            case = legacy[
                (
                    pd.to_numeric(
                        legacy["n_agents"],
                        errors="coerce",
                    )
                    == n_value
                )
            ].copy()

            if case.empty:
                continue

            case.to_csv(
                dst_raw,
                index=False,
            )

            summary = summarize_case(
                case,
                args.static_city_mode,
            )

            pd.DataFrame(
                [summary]
            ).to_csv(
                per_n_summary_path(n_value),
                index=False,
            )

            migrated.append(n_value)

        return migrated

    migrated_n = migrate_legacy_aggregate_if_needed()

    if migrated_n:
        print(
            "[migration] preserved legacy populations as canonical per-N files: "
            + ", ".join(
                f"{n:,}"
                for n in migrated_n
            )
        )

    if args.summarize_only:
        aggregate_raw, aggregate_summary = (
            rebuild_aggregate_outputs()
        )

        requested_n = {
            int(x)
            for x in args.agents
        }

        display_df = aggregate_summary[
            pd.to_numeric(
                aggregate_summary["n_agents"],
                errors="coerce",
            ).isin(requested_n)
        ].copy()

        if display_df.empty:
            display_df = aggregate_summary.copy()

        print("=" * 112)
        print("SUMMARY REBUILT FROM CANONICAL PER-N RAW DATA")
        print("=" * 112)
        print(f"City                : {args.city}")
        print(f"Static-city mode    : {args.static_city_mode}")
        print(f"Aggregate raw       : {aggregate_raw_path}")
        print(f"Aggregate summary   : {aggregate_summary_path}")
        print()
        print(
            display_df[
                [
                    c
                    for c in (
                        "n_agents",
                        "repeats",
                        "runtime_speedup_median",
                        "runtime_speedup_min",
                        "runtime_speedup_max",
                        "cold_break_even_decisions",
                        "cold_break_even_decisions_low",
                        "cold_break_even_decisions_high",
                    )
                    if c in display_df.columns
                ]
            ].to_string(index=False)
        )
        return

    raw_rows = []
    summary_rows = []

    print("=" * 112)
    print("02 — TEMPORAL END-TO-END BENCHMARK")
    print("=" * 112)
    print(f"City                 : {args.city}")
    print(f"Populations          : {args.agents}")
    print(f"Repeats              : {args.repeats}")
    print(f"H3 static-city mode  : {args.static_city_mode}")
    print("World build          : excluded (common input construction)")
    print("Continuous prepare   : measured (in-memory BallTree construction)")
    print("H3 cold prepare      : forced static-city rebuild when requested")
    print("H3 warm prepare      : cache load after untimed warm-up/build")
    print("Temporal loop        : schedule + decide + movement + H3 H12 maintenance")
    print("Physical routing     : excluded (same OSM infrastructure for both)")
    print()

    for n_agents in args.agents:
        print("-" * 112)
        print(f"N = {n_agents:,}")
        print("-" * 112)

        for rep in range(1, args.repeats + 1):
            print(
                f"  repeat {rep}/{args.repeats}"
            )

            cfg = config_for_city(
                args.city,
                n_agents=n_agents,
            )
            cfg.h3.validate()

            # Shared paired world for this repeat.
            world, _ = build_world(cfg)

            result = measure_pair(
                world,
                cfg,
                args.static_city_mode,
            )

            c_p = result[
                "continuous_prepare"
            ]
            hc_p = result[
                "h3_cold_prepare"
            ]
            hw_p = result[
                "h3_warm_prepare"
            ]
            c = result[
                "continuous_run"
            ]
            h = result[
                "h3_run"
            ]

            row = {
                "city": args.city,
                "n_agents": n_agents,
                "repeat": rep,
                "static_city_experiment_mode":
                    args.static_city_mode,

                "continuous_decisions":
                    c["decisions"],
                "h3_decisions":
                    h["decisions"],

                "continuous_valid_destinations":
                    c["valid_destinations"],
                "h3_valid_destinations":
                    h["valid_destinations"],

                "continuous_prepare_s":
                    c_p["wall_s"],
                "continuous_prepare_profile_s":
                    c_p["profile_s"],

                "h3_cold_prepare_s":
                    (
                        hc_p["wall_s"]
                        if hc_p is not None
                        else np.nan
                    ),
                "h3_cold_prepare_profile_s":
                    (
                        hc_p["profile_s"]
                        if hc_p is not None
                        else np.nan
                    ),
                "h3_cold_static_city_load_s":
                    (
                        hc_p["static_city_load_s"]
                        if hc_p is not None
                        else np.nan
                    ),
                "h3_cold_static_city_mode":
                    (
                        hc_p["static_city_mode"]
                        if hc_p is not None
                        else None
                    ),

                "h3_warm_prepare_s":
                    (
                        hw_p["wall_s"]
                        if hw_p is not None
                        else np.nan
                    ),
                "h3_warm_prepare_profile_s":
                    (
                        hw_p["profile_s"]
                        if hw_p is not None
                        else np.nan
                    ),
                "h3_warm_static_city_load_s":
                    (
                        hw_p["static_city_load_s"]
                        if hw_p is not None
                        else np.nan
                    ),
                "h3_warm_static_city_mode":
                    (
                        hw_p["static_city_mode"]
                        if hw_p is not None
                        else None
                    ),

                "continuous_runtime_s":
                    c["runtime_s"],
                "h3_runtime_s":
                    h["runtime_s"],

                "continuous_us_per_decision":
                    c["us_per_decision"],
                "h3_us_per_decision":
                    h["us_per_decision"],

                "continuous_decisions_per_s":
                    c["decisions_per_s"],
                "h3_decisions_per_s":
                    h["decisions_per_s"],

                "runtime_speedup_h3_over_continuous":
                    (
                        c["runtime_s"]
                        / h["runtime_s"]
                    ),

                # Paired total speedups: preparation + temporal runtime.
                # These must be computed per repeat before aggregation.
                "cold_total_speedup_h3_over_continuous":
                    (
                        (c_p["wall_s"] + c["runtime_s"])
                        / (hc_p["wall_s"] + h["runtime_s"])
                        if hc_p is not None
                        else np.nan
                    ),

                "warm_total_speedup_h3_over_continuous":
                    (
                        (c_p["wall_s"] + c["runtime_s"])
                        / (hw_p["wall_s"] + h["runtime_s"])
                        if hw_p is not None
                        else np.nan
                    ),
            }

            raw_rows.append(row)

            cold_txt = (
                f"{hc_p['wall_s']:.3f}s "
                f"({hc_p['static_city_mode']})"
                if hc_p is not None
                else "n/a"
            )
            warm_txt = (
                f"{hw_p['wall_s']:.3f}s "
                f"({hw_p['static_city_mode']})"
                if hw_p is not None
                else "n/a"
            )

            print(
                "    "
                f"prepare C={c_p['wall_s']:.3f}s | "
                f"H3 cold={cold_txt} | "
                f"H3 warm={warm_txt}"
            )
            print(
                "    "
                f"runtime C/H3="
                f"{c['runtime_s']:.3f}/"
                f"{h['runtime_s']:.3f}s | "
                f"speedup="
                f"{row['runtime_speedup_h3_over_continuous']:.3f}x"
            )

        # --------------------------------------------------------------
        # Canonical per-N checkpoint
        # --------------------------------------------------------------
        raw_df_current_call = pd.DataFrame(
            raw_rows
        )

        case = raw_df_current_call[
            (raw_df_current_call["city"] == args.city)
            & (
                pd.to_numeric(
                    raw_df_current_call["n_agents"],
                    errors="coerce",
                )
                == n_agents
            )
        ].copy()

        summary = summarize_case(
            case,
            args.static_city_mode,
        )

        # Save only the current N to canonical files.
        case.to_csv(
            per_n_raw_path(n_agents),
            index=False,
        )

        pd.DataFrame(
            [summary]
        ).to_csv(
            per_n_summary_path(n_agents),
            index=False,
        )

        # Rebuild aggregate outputs from every canonical N currently stored.
        rebuild_aggregate_outputs()

        print(
            f"  [saved per-N raw] {per_n_raw_path(n_agents)}"
        )
        print(
            f"  [saved per-N summary] {per_n_summary_path(n_agents)}"
        )

        print()
        print(
            "  MEDIAN runtime C/H3 = "
            f"{summary['continuous_runtime_s_median']:.3f}/"
            f"{summary['h3_runtime_s_median']:.3f}s"
        )
        print(
            "  TEMPORAL LOOP SPEEDUP — PAIRED MEDIAN = "
            f"{summary['runtime_speedup_h3_over_continuous']:.3f}x "
            f"[observed range "
            f"{summary['runtime_speedup_min']:.3f}x–"
            f"{summary['runtime_speedup_max']:.3f}x]"
        )

        if "h3_cold_prepare_s_median" in summary:
            print(
                "  COLD-START prepare PROFILE C/H3 = "
                f"{summary['continuous_prepare_profile_s_median']:.3f}/"
                f"{summary['h3_cold_prepare_profile_s_median']:.3f}s"
            )
            print(
                "  COLD-START prepare WALL C/H3 = "
                f"{summary['continuous_prepare_s_median']:.3f}/"
                f"{summary['h3_cold_prepare_s_median']:.3f}s "
                "(wall includes one-off serialization)"
            )
            print(
                "  COLD-START TOTAL SPEEDUP (profile prepare) = "
                f"{summary['cold_total_speedup_profile_h3_over_continuous']:.3f}x"
            )
            print(
                "  COLD-START TOTAL SPEEDUP (wall upper bound) = "
                f"{summary['cold_total_speedup_wall_h3_over_continuous']:.3f}x"
            )
            if pd.notna(
                summary["cold_break_even_decisions"]
            ):
                print(
                    "  COLD-START BREAK-EVEN "
                    "(incl. H12 maintenance) ≈ "
                    f"{summary['cold_break_even_decisions']:,.0f} decisions "
                    f"(~{summary['cold_break_even_agents_equivalent']:,.0f} "
                    "equivalent agents for this schedule)"
                )

                low = summary.get(
                    "cold_break_even_decisions_low",
                    np.nan,
                )
                high = summary.get(
                    "cold_break_even_decisions_high",
                    np.nan,
                )

                if pd.notna(low) and pd.notna(high):
                    print(
                        "  COLD-START BREAK-EVEN — OBSERVED "
                        "LATENCY SENSITIVITY RANGE ≈ "
                        f"{low:,.0f}–{high:,.0f} decisions "
                        "(not a confidence interval)"
                    )
            if pd.notna(
                summary["cold_break_even_wall_upper_decisions"]
            ):
                print(
                    "  COLD-START BREAK-EVEN — WALL UPPER BOUND "
                    "(includes serialization) ≈ "
                    f"{summary['cold_break_even_wall_upper_decisions']:,.0f} decisions"
                )

        if "h3_warm_prepare_s_median" in summary:
            print(
                "  WARM prepare C/H3 = "
                f"{summary['continuous_prepare_s_median']:.3f}/"
                f"{summary['h3_warm_prepare_s_median']:.3f}s"
            )
            print(
                "  WARM TOTAL SPEEDUP = "
                f"{summary['warm_total_speedup_h3_over_continuous']:.3f}x"
            )
            if pd.notna(
                summary["warm_break_even_decisions"]
            ):
                print(
                    "  WARM-CACHE BREAK-EVEN "
                    "(operational regime) ≈ "
                    f"{summary['warm_break_even_decisions']:,.0f} decisions"
                )

                low = summary.get(
                    "warm_break_even_decisions_low",
                    np.nan,
                )
                high = summary.get(
                    "warm_break_even_decisions_high",
                    np.nan,
                )

                if pd.notna(low) and pd.notna(high):
                    print(
                        "  WARM-CACHE BREAK-EVEN — OBSERVED "
                        "LATENCY SENSITIVITY RANGE ≈ "
                        f"{low:,.0f}–{high:,.0f} decisions "
                        "(not a confidence interval)"
                    )

        print()

    print("=" * 112)
    print("END-TO-END BENCHMARK COMPLETE")
    print("=" * 112)
    print(f"[saved aggregate] {aggregate_raw_path}")
    print(f"[saved aggregate] {aggregate_summary_path}")
    print()
    print("Canonical per-N outputs:")
    for p in discover_per_n_raw_files():
        n_value = int(
            p.stem.rsplit("_n", 1)[-1]
        )
        print(f"  [raw]     {p}")
        print(
            f"  [summary] {per_n_summary_path(n_value)}"
        )


if __name__ == "__main__":
    main()
