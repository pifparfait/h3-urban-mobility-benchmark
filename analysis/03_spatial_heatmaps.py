#!/usr/bin/env python3
from pathlib import Path
import h3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

diag_dirs = sorted((ROOT / "outputs" / "urban_fidelity").glob("diagnostics_n*"))
if not diag_dirs:
    raise FileNotFoundError(
        "No diagnostics_n* directory. Re-run 03_urban_fidelity.py once "
        "with the updated fidelity.py."
    )

def n_from_dir(p):
    return int(p.name.split("n")[-1])

diag = max(diag_dirs, key=n_from_dir)
n = n_from_dir(diag)

def polygons(cells):
    return [
        [(float(lng), float(lat)) for lat, lng in h3.cell_to_boundary(cell)]
        for cell in cells
    ]

def draw_layer(csv_path, stem, resolution):
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    polys = polygons(df["h3_id"].astype(str).tolist())
    vmax = max(
        float(df["continuous_count"].max()),
        float(df["h3_count"].max()),
        1.0,
    )

    # Continuous and H3 share EXACTLY the same value range.
    for col, label in [
        ("continuous_count", "Continuous-indexed"),
        ("h3_count", "H3"),
    ]:
        fig, ax = plt.subplots(figsize=(8,8))
        coll = PolyCollection(
            polys,
            array=df[col].to_numpy(dtype=float),
            edgecolors="none",
            clim=(0, vmax),
        )
        ax.add_collection(coll)
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{label} {stem} H{resolution} — N={n:,}")
        fig.colorbar(coll, ax=ax, label="Count")
        fig.tight_layout()
        fig.savefig(
            OUT / f"{stem}_H{resolution}_{col}.png",
            dpi=220,
        )
        plt.close(fig)

    diff = df["difference"].to_numpy(dtype=float)
    max_abs = max(float(np.max(np.abs(diff))), 1.0)
    fig, ax = plt.subplots(figsize=(8,8))
    coll = PolyCollection(
        polys,
        array=diff,
        edgecolors="none",
        norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
    )
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"H3 − Continuous {stem} H{resolution} — N={n:,}")
    fig.colorbar(coll, ax=ax, label="Count difference")
    fig.tight_layout()
    fig.savefig(
        OUT / f"{stem}_H{resolution}_difference.png",
        dpi=220,
    )
    plt.close(fig)

for r in (7,8,9):
    draw_layer(diag / f"occupancy_H{r}.csv", "final_occupancy", r)
    draw_layer(diag / f"visit_density_H{r}.csv", "visit_density", r)

print(f"Spatial heatmaps: {OUT}")
