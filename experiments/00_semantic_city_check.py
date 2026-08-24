#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(
        ROOT
        / "src"
    ),
)

from h3uab.city_config import (
    config_for_city,
)
from h3uab.osm_data import (
    load_osm_pois,
)
from h3uab.semantic_layers import (
    SEMANTIC_LAYERS,
)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--city",
        default="cambridge_ma",
        choices=[
            "madrid",
            "cambridge_ma",
            "new_york_ny",
        ],
    )

    args = ap.parse_args()

    cfg = config_for_city(
        args.city,
        n_agents=1_000,
    )

    raw = pd.read_csv(
        cfg.osm_cache_csv
    )

    normalized = load_osm_pois(
        cfg
    )

    print(
        "="
        * 100
    )

    print(
        "00 — CITY SEMANTIC NORMALIZATION CHECK"
    )

    print(
        "="
        * 100
    )

    print(
        f"City                 : "
        f"{cfg.city_name}"
    )

    print(
        f"Raw POI rows         : "
        f"{len(raw):,}"
    )

    print(
        f"Normalized POIs      : "
        f"{len(normalized):,}"
    )

    retention = (
        100.0
        * len(
            normalized
        )
        / max(
            len(
                raw
            ),
            1,
        )
    )

    print(
        f"Semantic retention   : "
        f"{retention:.2f}%"
    )

    active = tuple(
        sorted(
            normalized[
                "layer_code"
            ]
            .astype(str)
            .unique()
        )
    )

    print(
        f"Active layers        : "
        f"{active}"
    )

    print()

    print(
        "LAYER DISTRIBUTION"
    )

    print(
        "-"
        * 100
    )

    counts = (
        normalized[
            "layer_code"
        ]
        .value_counts()
        .to_dict()
    )

    total = max(
        len(
            normalized
        ),
        1,
    )

    for code, name in SEMANTIC_LAYERS.items():
        count = int(
            counts.get(
                code,
                0,
            )
        )

        if count <= 0:
            continue

        pct = (
            100.0
            * count
            / total
        )

        print(
            f"{code:<4} "
            f"{name:<28} "
            f"{count:>7,} "
            f"{pct:>7.2f}%"
        )

    print()

    if "category" in raw.columns:
        print(
            "RAW CATEGORY COUNTS"
        )

        print(
            "-"
            * 100
        )

        print(
            raw[
                "category"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print()

    print(
        "UNMAPPED RAW TAG-PAIRS — TOP 50"
    )

    print(
        "-"
        * 100
    )

    if {
        "category",
        "subcategory",
    }.issubset(
        raw.columns
    ):
        mapped_index = set(
            normalized.index.tolist()
        )

        # Re-run classification row-by-row only for diagnostic reporting
        # while preserving raw DataFrame indices.
        from h3uab.osm_data import (
            classify_from_category_subcategory,
        )

        diag = raw.copy()

        diag[
            "_mapped_code"
        ] = diag.apply(
            classify_from_category_subcategory,
            axis=1,
        )

        unmapped = (
            diag[
                diag[
                    "_mapped_code"
                ].isna()
            ]
            .groupby(
                [
                    "category",
                    "subcategory",
                ],
                dropna=False,
            )
            .size()
            .sort_values(
                ascending=False
            )
            .head(
                50
            )
        )

        if len(
            unmapped
        ):
            print(
                unmapped.to_string()
            )
        else:
            print(
                "None"
            )

    print()

    if len(
        active
    ) <= 1:
        raise RuntimeError(
            "Semantic normalization collapsed "
            "to one or zero active layers."
        )

    if len(
        normalized
    ) < 500:
        raise RuntimeError(
            "Too few normalized POIs for "
            "the city-scale benchmark."
        )

    print(
        "SEMANTIC NORMALIZATION CHECK: PASS"
    )


if __name__ == "__main__":
    main()
