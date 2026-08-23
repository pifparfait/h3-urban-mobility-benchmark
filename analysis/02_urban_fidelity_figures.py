#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

files = sorted((ROOT / "outputs" / "urban_fidelity").glob("urban_fidelity_n*.csv"))
if not files:
    raise FileNotFoundError("No urban_fidelity_n*.csv files found.")

# Use the largest available fidelity population.
def n_from_name(p):
    return int(p.stem.split("n")[-1])

path = max(files, key=n_from_name)
n = n_from_name(path)
df = pd.read_csv(path)

scales = ["H7", "H8", "H9"]
od = [
    float(df[(df.metric=="od_flow") & (df.scale==s)].pearson.iloc[0])
    for s in scales
]
occ = [
    float(df[(df.metric=="occupancy") & (df.scale==s)].pearson.iloc[0])
    for s in scales
]
access = float(df[df.metric=="accessibility"].pearson.iloc[0])

x = np.arange(len(scales))
fig, ax = plt.subplots(figsize=(8,5.3))
ax.bar(x-0.18, od, 0.36, label="OD flow")
ax.bar(x+0.18, occ, 0.36, label="Final occupancy")
ax.axhline(access, linestyle="--", label="Accessibility")
ax.set_xticks(x, scales)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Pearson correlation")
ax.set_title(f"Urban fidelity by analytical scale — N={n:,}")
ax.grid(axis="y", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "urban_fidelity_pearson_H7_H9.png", dpi=220)
plt.close(fig)

od_jsd = [
    float(df[(df.metric=="od_flow") & (df.scale==s)].jsd.iloc[0])
    for s in scales
]
occ_jsd = [
    float(df[(df.metric=="occupancy") & (df.scale==s)].jsd.iloc[0])
    for s in scales
]
fig, ax = plt.subplots(figsize=(8,5.3))
ax.bar(x-0.18, od_jsd, 0.36, label="OD flow")
ax.bar(x+0.18, occ_jsd, 0.36, label="Final occupancy")
ax.set_xticks(x, scales)
ax.set_ylabel("Jensen–Shannon divergence")
ax.set_title(f"Distributional divergence by analytical scale — N={n:,}")
ax.grid(axis="y", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "urban_fidelity_jsd_H7_H9.png", dpi=220)
plt.close(fig)

dest = df[df.metric=="destination"].iloc[0]
cats = ["Exact","≤100 m","≤250 m","≤500 m","≤1 km"]
vals = [
    100*dest["primary_value"],
    100*dest["within_100m"],
    100*dest["within_250m"],
    100*dest["within_500m"],
    100*dest["within_1000m"],
]
fig, ax = plt.subplots(figsize=(8,5.3))
ax.bar(cats, vals)
ax.set_ylim(0,100)
ax.set_ylabel("Paired destination decisions (%)")
ax.set_title(f"Destination spatial equivalence — N={n:,}")
ax.grid(axis="y", alpha=0.25)
for i,v in enumerate(vals):
    ax.text(i, v+1, f"{v:.1f}%", ha="center")
fig.tight_layout()
fig.savefig(OUT / "destination_proximity_thresholds.png", dpi=220)
plt.close(fig)

diag = ROOT / "outputs" / "urban_fidelity" / f"diagnostics_n{n}"
sep_path = diag / "destination_separations.csv"
if sep_path.exists():
    sep = pd.read_csv(sep_path)["destination_separation_m"].dropna().to_numpy(dtype=float)
    sep = np.sort(sep)
    cdf = np.arange(1, len(sep)+1) / len(sep)
    fig, ax = plt.subplots(figsize=(8,5.3))
    ax.plot(sep, cdf)
    ax.set_xlim(left=0)
    ax.set_ylim(0,1)
    ax.set_xlabel("Continuous–H3 destination separation (m)")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title(f"Destination-separation ECDF — N={n:,}")
    ax.grid(alpha=0.3)
    for threshold in [100,250,500,1000]:
        ax.axvline(threshold, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "destination_separation_ecdf.png", dpi=220)
    plt.close(fig)

print(f"Fidelity figures: {OUT}")
