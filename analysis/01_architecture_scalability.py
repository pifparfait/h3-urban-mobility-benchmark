#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

files = sorted((ROOT / "outputs" / "architecture").glob("architecture_summary_n*.csv"))
if not files:
    raise FileNotFoundError("No architecture_summary_n*.csv files found.")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.sort_values(["operation", "n_agents"])

labels = {
    "pure_spatial_lookup": "Urban-state lookup",
    "urban_context_retrieval": "Urban-context retrieval",
    "accessibility_query": "Accessibility query",
    "candidate_retrieval": "Candidate retrieval",
    "full_agent_decision": "Full agent decision",
}

# scaling exponents
rows = []
for op, g in df.groupby("operation"):
    for backend, col in [
        ("Continuous-indexed", "continuous_median_s"),
        ("H3", "h3_median_s"),
    ]:
        x = np.log10(g["queries"].to_numpy(dtype=float))
        y = np.log10(g[col].to_numpy(dtype=float))
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope*x + intercept
        r2 = 1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)
        rows.append({
            "operation": op,
            "backend": backend,
            "scaling_exponent_alpha": slope,
            "loglog_r2": r2,
        })

pd.DataFrame(rows).to_csv(
    ROOT / "outputs" / "architecture_scaling_exponents.csv",
    index=False,
)

# speedup
fig, ax = plt.subplots(figsize=(9, 5.5))
for op, g in df.groupby("operation"):
    g = g.sort_values("n_agents")
    ax.plot(
        g["n_agents"],
        g["speedup_h3_over_continuous"],
        marker="o",
        label=labels.get(op, op),
    )
ax.set_xscale("log")
ax.set_xlabel("Number of agents / queries")
ax.set_ylabel("H3 speedup over Continuous-indexed (×)")
ax.set_title("Architectural speedup from 10K to 1M")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "architecture_speedup_vs_population.png", dpi=220)
plt.close(fig)

# full decision latency
g = df[df.operation == "full_agent_decision"].sort_values("n_agents")
fig, ax = plt.subplots(figsize=(8.5, 5.3))
ax.plot(g["n_agents"], g["continuous_us_per_query"], marker="o", label="Continuous-indexed")
ax.plot(g["n_agents"], g["h3_us_per_query"], marker="o", label="H3")
ax.set_xscale("log")
ax.set_xlabel("Number of agents / decisions")
ax.set_ylabel("Median latency (µs / decision)")
ax.set_title("Full-decision latency")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "full_decision_latency_vs_population.png", dpi=220)
plt.close(fig)

# full decision throughput
fig, ax = plt.subplots(figsize=(8.5, 5.3))
ax.plot(g["n_agents"], g["continuous_qps"], marker="o", label="Continuous-indexed")
ax.plot(g["n_agents"], g["h3_qps"], marker="o", label="H3")
ax.set_xscale("log")
ax.set_xlabel("Number of agents / decisions")
ax.set_ylabel("Decisions per second")
ax.set_title("Full-decision throughput")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "full_decision_throughput_vs_population.png", dpi=220)
plt.close(fig)

print(f"Architecture figures: {OUT}")
