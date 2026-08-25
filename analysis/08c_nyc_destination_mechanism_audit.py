#!/usr/bin/env python3
"""
08c_nyc_destination_mechanism_audit.py
======================================

Diagnostic audit of the NYC workplace-destination mechanism.

This script investigates why NYC H7 destination agreement with LODES is weak,
while Continuous and H3 remain very close to each other.

It compares, at H7:
  1. Observed LODES workplace inflow
  2. Continuous simulated workplace inflow
  3. H3 simulated workplace inflow
  4. POI count per H7 cell
  5. Sum of POI attractiveness per H7 cell
  6. Mean POI attractiveness per H7 cell
  7. Optional work-layer-only versions
  8. Top-K concentration and ranking overlap

The script is diagnostic only and does NOT recalibrate Experiment 07.

Expected input:
    outputs/external_od_validation/new_york_ny/three_way_od_h7.csv

Outputs:
    outputs/external_od_validation/new_york_ny/audit_08c/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import h3
except ImportError as exc:
    raise SystemExit("h3 is required. Install it with: pip install h3") from exc


EPS = 1e-12
H3_RESOLUTION = 7

DEFAULT_THREE_WAY = Path(
    "outputs/external_od_validation/new_york_ny/three_way_od_h7.csv"
)
DEFAULT_OUTPUT = Path(
    "outputs/external_od_validation/new_york_ny/audit_08c"
)
DEFAULT_DATA_ROOT = Path("data/new_york_ny/processed")
TOP_K_VALUES = (5, 10, 20, 50, 100)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) <= EPS or np.std(y) <= EPS:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(x.corr(y, method="spearman"))


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    x = np.clip(x, 0.0, None)
    total = float(x.sum())
    if total <= EPS:
        return np.zeros_like(x)
    return x / total


def tvd(x: np.ndarray, y: np.ndarray) -> float:
    p = normalize(x)
    q = normalize(y)
    return float(0.5 * np.abs(p - q).sum())


def jsd(x: np.ndarray, y: np.ndarray) -> float:
    p = normalize(x)
    q = normalize(y)
    if p.sum() <= EPS or q.sum() <= EPS:
        return np.nan
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= EPS:
        return np.nan
    return float(np.dot(x, y) / denom)


def metric_bundle(observed: np.ndarray, proxy: np.ndarray) -> dict:
    observed = np.asarray(observed, dtype=float)
    proxy = np.asarray(proxy, dtype=float)
    return {
        "pearson": safe_pearson(observed, proxy),
        "spearman": safe_spearman(pd.Series(observed), pd.Series(proxy)),
        "tvd": tvd(observed, proxy),
        "jsd": jsd(observed, proxy),
        "cosine": cosine_similarity(observed, proxy),
    }


def detect_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    label: str,
    *,
    required: bool = True,
) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if required:
        raise ValueError(
            f"Could not detect {label} column. "
            f"Tried: {list(candidates)}. "
            f"Available: {list(df.columns)}"
        )
    return None


def latlng_to_h3(lat: float, lng: float, resolution: int) -> str:
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(float(lat), float(lng), int(resolution))
    return h3.geo_to_h3(float(lat), float(lng), int(resolution))


def load_three_way(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"[load] three-way NYC H7 OD: {path}")
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
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    for c in ["observed_flow", "continuous_flow", "h3_flow"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


def build_destination_marginals(three_way: pd.DataFrame) -> pd.DataFrame:
    return (
        three_way.groupby("destination_h3", as_index=False)[
            ["observed_flow", "continuous_flow", "h3_flow"]
        ]
        .sum()
    )


def detect_poi_file(processed_root: Path) -> Path:
    if not processed_root.exists():
        raise FileNotFoundError(processed_root)

    csvs = list(processed_root.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {processed_root}")

    preferred_tokens = ["osm_pois", "pois", "poi"]
    for token in preferred_tokens:
        candidates = [p for p in csvs if token in p.name.lower()]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_size)

    raise FileNotFoundError(
        "Could not automatically detect NYC POI CSV. "
        f"Available CSVs: {[p.name for p in csvs]}"
    )


def load_pois(
    path: Path,
    *,
    work_layer_codes: set[str] | None,
) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"[load] NYC POIs: {path}")
    df = pd.read_csv(path, low_memory=False)

    lat_col = detect_column(df, ["lat", "latitude", "y"], "POI latitude")
    lng_col = detect_column(df, ["lng", "lon", "longitude", "x"], "POI longitude")

    attractiveness_col = detect_column(
        df,
        ["attractiveness", "amenity_attractiveness", "weight", "score"],
        "POI attractiveness",
        required=False,
    )

    layer_col = detect_column(
        df,
        ["layer_code", "category", "amenity_layer", "layer"],
        "POI layer/category",
        required=False,
    )

    out = pd.DataFrame(
        {
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lng": pd.to_numeric(df[lng_col], errors="coerce"),
        }
    )

    if attractiveness_col is not None:
        out["attractiveness"] = (
            pd.to_numeric(df[attractiveness_col], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
    else:
        out["attractiveness"] = 1.0

    if layer_col is not None:
        out["layer_code"] = (
            df[layer_col].fillna("").astype(str).str.strip()
        )
    else:
        out["layer_code"] = ""

    out = out.dropna(subset=["lat", "lng"]).copy()
    out["h3_7"] = [
        latlng_to_h3(lat, lng, H3_RESOLUTION)
        for lat, lng in zip(out["lat"], out["lng"])
    ]

    out["is_work_layer"] = (
        out["layer_code"].isin(work_layer_codes)
        if work_layer_codes
        else False
    )

    meta = {
        "source_path": str(path),
        "n_pois": int(len(out)),
        "attractiveness_column": (
            attractiveness_col if attractiveness_col is not None
            else "fallback_constant_1"
        ),
        "layer_column": layer_col,
        "work_layer_codes": sorted(work_layer_codes) if work_layer_codes else None,
    }

    return out, meta


def aggregate_poi_proxies(
    pois: pd.DataFrame,
    *,
    work_layer_codes: set[str] | None,
) -> pd.DataFrame:
    all_pois = (
        pois.groupby("h3_7", as_index=False)
        .agg(
            poi_count_all=("h3_7", "size"),
            attractiveness_sum_all=("attractiveness", "sum"),
            attractiveness_mean_all=("attractiveness", "mean"),
        )
        .rename(columns={"h3_7": "destination_h3"})
    )

    if not work_layer_codes:
        return all_pois

    work = pois[pois["is_work_layer"]].copy()

    if work.empty:
        all_pois["poi_count_work"] = 0.0
        all_pois["attractiveness_sum_work"] = 0.0
        all_pois["attractiveness_mean_work"] = 0.0
        return all_pois

    work_agg = (
        work.groupby("h3_7", as_index=False)
        .agg(
            poi_count_work=("h3_7", "size"),
            attractiveness_sum_work=("attractiveness", "sum"),
            attractiveness_mean_work=("attractiveness", "mean"),
        )
        .rename(columns={"h3_7": "destination_h3"})
    )

    return all_pois.merge(work_agg, on="destination_h3", how="outer")


def build_cell_table(
    destinations: pd.DataFrame,
    poi_proxies: pd.DataFrame,
) -> pd.DataFrame:
    out = destinations.merge(
        poi_proxies,
        on="destination_h3",
        how="outer",
    )

    numeric_cols = [c for c in out.columns if c != "destination_h3"]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    for flow_col, share_col in [
        ("observed_flow", "observed_share"),
        ("continuous_flow", "continuous_share"),
        ("h3_flow", "h3_share"),
    ]:
        total = float(out[flow_col].sum())
        out[share_col] = out[flow_col] / total if total > EPS else 0.0

    proxy_cols = [
        c for c in out.columns
        if (
            c.startswith("poi_count")
            or c.startswith("attractiveness_sum")
            or c.startswith("attractiveness_mean")
        )
    ]

    for c in proxy_cols:
        total = float(out[c].sum())
        out[f"{c}_share"] = out[c] / total if total > EPS else 0.0

    for c in ["observed_flow", "continuous_flow", "h3_flow"] + proxy_cols:
        out[f"{c}_rank"] = out[c].rank(
            method="min",
            ascending=False,
        )

    return out


def mechanism_metric_table(cell_table: pd.DataFrame) -> pd.DataFrame:
    observed = cell_table["observed_flow"].to_numpy(dtype=float)

    proxy_columns = ["continuous_flow", "h3_flow"] + [
        c for c in cell_table.columns
        if (
            c.startswith("poi_count")
            or c.startswith("attractiveness_sum")
            or c.startswith("attractiveness_mean")
        )
        and not c.endswith("_share")
        and not c.endswith("_rank")
    ]

    rows = []

    for proxy in proxy_columns:
        values = cell_table[proxy].to_numpy(dtype=float)
        metrics = metric_bundle(observed, values)

        rows.append(
            {
                "proxy": proxy,
                "nonzero_cells": int(np.count_nonzero(values)),
                "proxy_total": float(values.sum()),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def concentration_table(cell_table: pd.DataFrame) -> pd.DataFrame:
    columns = ["observed_flow", "continuous_flow", "h3_flow"]
    columns += [
        c for c in cell_table.columns
        if (
            c.startswith("poi_count")
            or c.startswith("attractiveness_sum")
        )
        and not c.endswith("_share")
        and not c.endswith("_rank")
    ]

    rows = []

    for k in TOP_K_VALUES:
        for col in columns:
            x = cell_table[["destination_h3", col]].sort_values(
                col,
                ascending=False,
            )

            actual_k = min(k, len(x))
            total = float(x[col].sum())

            concentration = (
                float(x.head(actual_k)[col].sum() / total)
                if total > EPS else np.nan
            )

            rows.append(
                {
                    "k": actual_k,
                    "distribution": col,
                    "topk_concentration": concentration,
                }
            )

    return pd.DataFrame(rows)


def rank_overlap_table(cell_table: pd.DataFrame) -> pd.DataFrame:
    proxies = ["continuous_flow", "h3_flow"]
    proxies += [
        c for c in cell_table.columns
        if (
            c.startswith("poi_count")
            or c.startswith("attractiveness_sum")
        )
        and not c.endswith("_share")
        and not c.endswith("_rank")
    ]

    rows = []

    for k in TOP_K_VALUES:
        actual_k = min(k, len(cell_table))

        observed_top = set(
            cell_table.nlargest(actual_k, "observed_flow")["destination_h3"]
        )

        for proxy in proxies:
            proxy_top = set(
                cell_table.nlargest(actual_k, proxy)["destination_h3"]
            )

            overlap = (
                len(observed_top & proxy_top) / actual_k
                if actual_k > 0 else np.nan
            )

            rows.append(
                {
                    "k": actual_k,
                    "proxy": proxy,
                    "topk_overlap_with_lodes": overlap,
                }
            )

    return pd.DataFrame(rows)


def top_hub_table(
    cell_table: pd.DataFrame,
    n: int = 50,
) -> pd.DataFrame:
    x = (
        cell_table
        .sort_values("observed_flow", ascending=False)
        .head(n)
        .copy()
    )

    for prefix in ["continuous", "h3"]:
        x[f"{prefix}_share_ratio_to_observed"] = np.where(
            x["observed_share"] > EPS,
            x[f"{prefix}_share"] / x["observed_share"],
            np.nan,
        )

    return x


def interpretation(
    metrics: pd.DataFrame,
    concentration: pd.DataFrame,
) -> dict:
    def metric(proxy: str, field: str) -> float:
        row = metrics[metrics["proxy"] == proxy]
        if row.empty:
            return np.nan
        return float(row.iloc[0][field])

    cont_p = metric("continuous_flow", "pearson")
    h3_p = metric("h3_flow", "pearson")
    attr_p = metric("attractiveness_sum_all", "pearson")
    count_p = metric("poi_count_all", "pearson")

    observed_top5 = concentration[
        (concentration["k"] == 5)
        & (concentration["distribution"] == "observed_flow")
    ]

    cont_top5 = concentration[
        (concentration["k"] == 5)
        & (concentration["distribution"] == "continuous_flow")
    ]

    obs_c5 = (
        float(observed_top5.iloc[0]["topk_concentration"])
        if len(observed_top5) else np.nan
    )

    cont_c5 = (
        float(cont_top5.iloc[0]["topk_concentration"])
        if len(cont_top5) else np.nan
    )

    conclusions = []

    if (
        np.isfinite(obs_c5)
        and np.isfinite(cont_c5)
        and cont_c5 < 0.6 * obs_c5
    ):
        conclusions.append(
            "The simulated workplace distribution is materially less "
            "concentrated than LODES at the top employment hubs."
        )

    if np.isfinite(attr_p) and attr_p < 0.6:
        conclusions.append(
            "Aggregate POI attractiveness is weakly aligned with observed "
            "employment inflow, suggesting that attractiveness is not an "
            "adequate proxy for employment capacity in NYC."
        )

    if np.isfinite(count_p) and count_p < 0.6:
        conclusions.append(
            "POI density alone is also weakly aligned with LODES workplace "
            "capacity."
        )

    if (
        np.isfinite(cont_p)
        and np.isfinite(h3_p)
        and abs(cont_p - h3_p) < 0.02
    ):
        conclusions.append(
            "Continuous and H3 reproduce essentially the same destination "
            "mismatch, so the evidence does not indicate an H3-specific failure."
        )

    if not conclusions:
        conclusions.append(
            "No single mechanism dominates the audit; inspect the detailed "
            "proxy metrics and top-hub tables."
        )

    return {
        "continuous_destination_pearson": cont_p,
        "h3_destination_pearson": h3_p,
        "poi_count_destination_pearson": count_p,
        "attractiveness_sum_destination_pearson": attr_p,
        "observed_top5_concentration": obs_c5,
        "continuous_top5_concentration": cont_c5,
        "conclusions": conclusions,
    }


def banner(title: str) -> None:
    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def print_table(df: pd.DataFrame, digits: int = 4) -> None:
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.float_format", lambda x: f"{x:.{digits}f}",
    ):
        print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the NYC H7 workplace-destination mechanism against "
            "LODES and POI-based destination proxies."
        )
    )

    parser.add_argument(
        "--three-way",
        type=Path,
        default=DEFAULT_THREE_WAY,
    )

    parser.add_argument(
        "--poi-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit NYC normalized POI CSV. If omitted, "
            "auto-detect under data/new_york_ny/processed/."
        ),
    )

    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--work-layer-codes",
        nargs="*",
        default=None,
        help=(
            "Optional exact POI layer codes considered work-relevant. "
            "Example: --work-layer-codes D3 D8 D21"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    outdir = ensure_dir(args.output_dir)

    work_layer_codes = (
        {str(x).strip() for x in args.work_layer_codes}
        if args.work_layer_codes else None
    )

    three_way = load_three_way(args.three_way)
    destinations = build_destination_marginals(three_way)

    poi_path = (
        args.poi_file
        if args.poi_file is not None
        else detect_poi_file(args.processed_root)
    )

    pois, poi_meta = load_pois(
        poi_path,
        work_layer_codes=work_layer_codes,
    )

    poi_proxies = aggregate_poi_proxies(
        pois,
        work_layer_codes=work_layer_codes,
    )

    cells = build_cell_table(
        destinations,
        poi_proxies,
    )

    metrics = mechanism_metric_table(cells)
    concentration = concentration_table(cells)
    overlap = rank_overlap_table(cells)
    top_hubs = top_hub_table(cells, n=50)
    interp = interpretation(metrics, concentration)

    banner("NYC H7 DESTINATION MECHANISM AUDIT")
    print(f"Three-way OD source          : {args.three_way}")
    print(f"POI source                   : {poi_path}")
    print(f"POIs loaded                  : {len(pois):,}")
    print(f"H7 destination cells         : {len(cells):,}")
    print(f"Observed workplace flow      : {cells['observed_flow'].sum():,.0f}")
    print(f"Continuous workplace flow    : {cells['continuous_flow'].sum():,.0f}")
    print(f"H3 workplace flow            : {cells['h3_flow'].sum():,.0f}")

    if work_layer_codes:
        print(f"Work layer codes             : {sorted(work_layer_codes)}")
        print(f"POIs in work layers          : {int(pois['is_work_layer'].sum()):,}")

    banner("EMPLOYMENT / DESTINATION PROXY METRICS")
    print_table(metrics.sort_values("pearson", ascending=False))

    banner("TOP-K CONCENTRATION")
    print_table(
        concentration[
            concentration["k"].isin([5, 10, 20])
        ]
    )

    banner("TOP-K OVERLAP WITH LODES DESTINATION RANKING")
    print_table(overlap)

    banner("TOP 20 LODES EMPLOYMENT HUBS")
    selected_cols = [
        c for c in [
            "destination_h3",
            "observed_flow",
            "continuous_flow",
            "h3_flow",
            "observed_share",
            "continuous_share",
            "h3_share",
            "poi_count_all",
            "attractiveness_sum_all",
            "observed_flow_rank",
            "continuous_flow_rank",
            "h3_flow_rank",
            "poi_count_all_rank",
            "attractiveness_sum_all_rank",
            "continuous_share_ratio_to_observed",
            "h3_share_ratio_to_observed",
        ]
        if c in top_hubs.columns
    ]
    print_table(top_hubs[selected_cols].head(20))

    banner("DIAGNOSTIC INTERPRETATION")
    for i, conclusion in enumerate(interp["conclusions"], start=1):
        print(f"[{i}] {conclusion}")

    cells_path = outdir / "NYC_H7_destination_mechanism_cells.csv"
    metrics_path = outdir / "NYC_H7_destination_mechanism_metrics.csv"
    hubs_path = outdir / "NYC_H7_destination_top_hubs.csv"
    concentration_path = outdir / "NYC_H7_destination_concentration.csv"
    overlap_path = outdir / "NYC_H7_destination_rank_overlap.csv"
    interpretation_path = (
        outdir / "NYC_H7_destination_mechanism_interpretation.json"
    )
    poi_meta_path = outdir / "NYC_H7_destination_poi_metadata.json"

    cells.to_csv(cells_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    top_hubs.to_csv(hubs_path, index=False)
    concentration.to_csv(concentration_path, index=False)
    overlap.to_csv(overlap_path, index=False)

    interpretation_path.write_text(
        json.dumps(interp, indent=2),
        encoding="utf-8",
    )

    poi_meta_path.write_text(
        json.dumps(poi_meta, indent=2),
        encoding="utf-8",
    )

    banner("FILES SAVED")
    for path in [
        cells_path,
        metrics_path,
        hubs_path,
        concentration_path,
        overlap_path,
        interpretation_path,
        poi_meta_path,
    ]:
        print(f"[saved] {path.resolve()}")

    banner("DONE")
    print("Diagnostic only. Experiment 07 remains frozen and unchanged.")


if __name__ == "__main__":
    main()
