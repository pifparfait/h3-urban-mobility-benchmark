#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h3uab.city_config import config_for_city
from h3uab.world import build_world
from h3uab.continuous_indexed_backend import ContinuousIndexedBackend
from h3uab.h3_backend import MultiLayerH3Backend
from h3uab.benchmarking import (
    make_query_set,
    prepare_pure_lookup_helpers,
    benchmark_pure_lookup,
    benchmark_context_retrieval,
    benchmark_accessibility_query,
    benchmark_candidate_retrieval,
    benchmark_full_decision,
    add_speedups,
)


def run_one_case(
    *,
    city: str,
    n_agents: int,
    n_queries: int,
    repeats: int,
    output_dir: Path,
):
    cfg = config_for_city(
        city,
        n_agents=int(n_agents),
    )

    cfg.h3.validate()

    print()
    print("=" * 108)
    print(
        f"01 — ARCHITECTURE BENCHMARK | "
        f"city={cfg.city_name} | "
        f"agents={n_agents:,} | queries={n_queries:,} | repeats={repeats}"
    )
    print("=" * 108)
    print(
        "Routing: EXCLUDED "
        "(same real OSM physical infrastructure for both)."
    )
    print(
        "H11/H12: localization/visualization only; "
        "not fidelity layers."
    )
    print()

    world, _ = build_world(cfg)

    continuous = ContinuousIndexedBackend()
    h3_backend = MultiLayerH3Backend()

    continuous.prepare(
        world,
        cfg,
    )

    h3_backend.prepare(
        world,
        cfg,
    )

    prepare_pure_lookup_helpers(
        continuous,
        h3_backend,
        cfg,
    )

    q = make_query_set(
        world,
        cfg,
        int(n_queries),
    )

    frames = [
        benchmark_pure_lookup(
            world,
            h3_backend,
            q,
            repeats,
        ),
        benchmark_context_retrieval(
            world,
            continuous,
            h3_backend,
            q,
            cfg,
            repeats,
        ),
        benchmark_accessibility_query(
            world,
            continuous,
            h3_backend,
            q,
            cfg,
            repeats,
        ),
        benchmark_candidate_retrieval(
            world,
            continuous,
            h3_backend,
            q,
            cfg,
            repeats,
        ),
        benchmark_full_decision(
            world,
            continuous,
            h3_backend,
            q,
            repeats,
        ),
    ]

    raw = pd.concat(
        frames,
        ignore_index=True,
    )

    summary = add_speedups(
        raw
    )

    # ------------------------------------------------------------------
    # Add experiment metadata directly into the CSVs.
    # ------------------------------------------------------------------
    raw.insert(
        0,
        "city",
        str(cfg.city_name),
    )
    raw.insert(
        1,
        "n_agents",
        int(n_agents),
    )
    raw.insert(
        2,
        "n_queries_requested",
        int(n_queries),
    )
    raw.insert(
        3,
        "repeats",
        int(repeats),
    )

    summary.insert(
        0,
        "city",
        str(cfg.city_name),
    )
    summary.insert(
        1,
        "n_agents",
        int(n_agents),
    )
    summary.insert(
        2,
        "n_queries_requested",
        int(n_queries),
    )
    summary.insert(
        3,
        "repeats",
        int(repeats),
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # NEVER overwrite a different population-size result.
    # ------------------------------------------------------------------
    raw_path = (
        output_dir
        / f"architecture_raw_n{int(n_agents)}.csv"
    )

    summary_path = (
        output_dir
        / f"architecture_summary_n{int(n_agents)}.csv"
    )

    raw.to_csv(
        raw_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"[saved] {summary_path}"
    )
    print(
        f"[saved] {raw_path}"
    )

    return raw, summary


def rebuild_aggregate_outputs(
    output_dir: Path,
):
    """
    Rebuild aggregate sweep files from every per-N file currently present.

    This means:
      - interrupted batches do not lose completed cases,
      - runs can be resumed,
      - 10K / 100K / 500K / 1M remain independently reproducible.
    """

    summary_files = sorted(
        output_dir.glob(
            "architecture_summary_n*.csv"
        )
    )

    raw_files = sorted(
        output_dir.glob(
            "architecture_raw_n*.csv"
        )
    )

    if summary_files:
        summary_all = pd.concat(
            [
                pd.read_csv(
                    p
                )
                for p in summary_files
            ],
            ignore_index=True,
        ).sort_values(
            [
                "n_agents",
                "operation",
            ]
        )

        summary_all.to_csv(
            output_dir
            / "architecture_sweep_summary.csv",
            index=False,
        )

    if raw_files:
        raw_all = pd.concat(
            [
                pd.read_csv(
                    p
                )
                for p in raw_files
            ],
            ignore_index=True,
        ).sort_values(
            [
                "n_agents",
                "operation",
                "backend",
            ]
        )

        raw_all.to_csv(
            output_dir
            / "architecture_sweep_raw.csv",
            index=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Continuous-indexed vs H3 architecture benchmark. "
            "Supports one population or a checkpointed population sweep."
        )
    )

    parser.add_argument(
        "--city",
        type=str,
        default="madrid",
        choices=[
            "madrid",
            "cambridge_ma",
            "new_york_ny",
        ],
        help=(
            "City configuration. Madrid remains the default; "
            "Cambridge results are stored separately."
        ),
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        type=int,
        default=[
            10_000,
            100_000,
            500_000,
            1_000_000,
        ],
        help=(
            "One or more population sizes. "
            "Default: 10K 100K 500K 1M."
        ),
    )

    parser.add_argument(
        "--queries",
        type=int,
        default=None,
        help=(
            "Fixed number of queries per population. "
            "If omitted, queries = number of agents for each case."
        ),
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip a population if both per-N output files already exist. "
            "Useful to resume an interrupted sweep."
        ),
    )

    args = parser.parse_args()

    output_dir = (
        ROOT
        / "outputs"
        / "architecture"
        if args.city == "madrid"
        else (
            ROOT
            / "outputs"
            / args.city
            / "architecture"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    requested_agents = [
        int(
            n
        )
        for n in args.agents
    ]

    print("=" * 108)
    print(
        "ARCHITECTURE BENCHMARK — CHECKPOINTED BATCH SWEEP"
    )
    print("=" * 108)
    print(
        f"City       : {args.city}"
    )
    print(
        "Populations:",
        " -> ".join(
            f"{n:,}"
            for n in requested_agents
        ),
    )
    print(
        "Each population is saved independently immediately after completion."
    )
    print()

    for n_agents in requested_agents:
        n_queries = (
            int(
                args.queries
            )
            if args.queries is not None
            else int(
                n_agents
            )
        )

        summary_path = (
            output_dir
            / f"architecture_summary_n{n_agents}.csv"
        )

        raw_path = (
            output_dir
            / f"architecture_raw_n{n_agents}.csv"
        )

        if (
            args.skip_existing
            and summary_path.exists()
            and raw_path.exists()
        ):
            print(
                f"[skip] N={n_agents:,} "
                f"(existing per-N results found)"
            )

            rebuild_aggregate_outputs(
                output_dir
            )

            continue

        run_one_case(
            city=args.city,
            n_agents=n_agents,
            n_queries=n_queries,
            repeats=int(
                args.repeats
            ),
            output_dir=output_dir,
        )

        # Checkpoint aggregate files after EVERY population.
        rebuild_aggregate_outputs(
            output_dir
        )

    rebuild_aggregate_outputs(
        output_dir
    )

    print()
    print("=" * 108)
    print(
        "BATCH COMPLETE"
    )
    print("=" * 108)
    print(
        f"Per-N and aggregate outputs: {output_dir}"
    )
    if args.city != "madrid":
        print(
            "Madrid architecture outputs were not modified."
        )
    print()
    print(
        "Aggregate files:"
    )
    print(
        "  architecture_sweep_summary.csv"
    )
    print(
        "  architecture_sweep_raw.csv"
    )


if __name__ == "__main__":
    main()
