# H3 Urban Architecture Benchmark

A reproducible **multi-city benchmark for scalable urban simulation with H3**, comparing a continuous spatial-indexed backend with a multi-resolution H3 backend while preserving a common physical mobility layer.

> **Research question:** How much computational efficiency can a hierarchical H3 representation provide for large-scale urban agent simulation, and how much urban fidelity is retained across spatial scales?

The current benchmark covers **Madrid, Cambridge (MA), and New York City**, including experiments up to **1,000,000 agents / queries**.

---

## 1. What this repository evaluates

The framework deliberately separates three experimental questions.

### Architectural efficiency
The Continuous and H3 backends are benchmarked on:
- pure spatial lookup;
- urban-context retrieval;
- accessibility queries;
- candidate retrieval;
- full agent decision.

The architecture sweep scales from **10K to 1M agents/queries**.

### Urban fidelity
At `N = 10,000`, paired Continuous/H3 simulations measure:
- accessibility correlation;
- destination spatial equivalence;
- trip-distance fidelity;
- OD-flow similarity at H7/H8/H9;
- final occupancy similarity at H7/H8/H9;
- amenity-usage agreement.

### Physical-network realization
Representative paired origin-destination decisions are projected onto the **same real OpenStreetMap walking graph**. Routing is deliberately *not* part of the H3 speedup claim.

---

## 2. Architectural rules

1. **H7/H8/H9 are urban-analysis and fidelity scales.**
2. **H11/H12 are localization/visualization layers only** and never enter the fidelity claim.
3. **Physical routing is not discretized with H3.**
4. Both backends use the **same OSM physical network** for route visualization.
5. Heatmaps aggregate real agent/destination coordinates at H7/H8/H9.
6. Speedup therefore measures the spatial-computational representation, not a replacement of street-network routing.

---

## 3. Cross-city benchmark

| City | Raw OSM POIs | Semantic POIs | Population H3 cells | Represented population | OSM graph nodes | OSM graph edges |
|---|---:|---:|---:|---:|---:|---:|
| Madrid | 38,949 | 38,949 | 2,037 | 2,337,708 | 122,237 | 373,234 |
| Cambridge, MA | 7,831 | 2,102 | 175 | 103,855 | 20,322 | 54,756 |
| New York City | 305,387 | 126,595 | 6,747 | 8,037,287 | 361,278 | 1,108,754 |

These cities provide substantially different spatial scales, POI densities, network sizes and urban morphologies.

---

## 4. Headline computational results at 1M queries

Speedup = `continuous execution time / H3 execution time`.

| Operation | Madrid | Cambridge | New York City |
|---|---:|---:|---:|
| Spatial lookup | **47.49x** | **29.95x** | **38.97x** |
| Urban context retrieval | **26.60x** | **16.19x** | **18.35x** |
| Accessibility query | **83.70x** | **50.24x** | **43.44x** |
| Candidate retrieval | **42.03x** | **40.22x** | **41.63x** |
| Full agent decision | **4.21x** | **2.86x** | **3.27x** |

The important result is not one maximum number: the H3 architecture remains faster across all three cities and all five computational operations, including the end-to-end agent decision.

### Full-agent decision latency at 1M

| City | Continuous | H3 | Speedup |
|---|---:|---:|---:|
| Madrid | 75.67 us/query | 17.97 us/query | 4.21x |
| Cambridge | 62.96 us/query | 22.01 us/query | 2.86x |
| New York City | 51.39 us/query | 15.72 us/query | 3.27x |

---

## 5. Fidelity validation at 10K agents

| Metric | Madrid | Cambridge | New York City |
|---|---:|---:|---:|
| Accessibility Pearson | 0.9905 | 0.9796 | 0.9917 |
| Exact destination match | 0.6384 | 0.5922 | 0.6485 |
| Destination within 500 m | 0.8834 | 0.7937 | 0.8229 |
| Destination within 1,000 m | 0.9786 | 0.9524 | 0.9306 |
| Trip-distance Pearson | 0.8834 | 0.8265 | 0.8964 |
| OD flow H7 Pearson | 0.9990 | 0.9989 | 0.9945 |
| OD flow H8 Pearson | 0.9891 | 0.9758 | 0.9584 |
| OD flow H9 Pearson | 0.9168 | 0.8969 | 0.8328 |
| Occupancy H7 Pearson | 0.9928 | 0.9972 | 0.9840 |
| Occupancy H8 Pearson | 0.9174 | 0.8452 | 0.8370 |
| Occupancy H9 Pearson | 0.7528 | 0.6446 | 0.6991 |
| Amenity usage Pearson | 1.0000 | 1.0000 | 1.0000 |

The scale effect is clear: **city-scale structure is extremely well preserved at H7, neighborhood structure remains strong at H8, and finer local allocation differences become more visible at H9.**

---

## 6. Main paper figures

The paths below follow the current cross-city output convention.

### Cross-city computational speedup

![Cross-city computational speedup](outputs/cross_city/paper_figures/FIG_CC1_speedup_scaling.png)

### Full-agent decision scaling

![Full-agent decision latency](outputs/cross_city/paper_figures/FIG_CC2_full_decision_scaling.png)

### Resolution-dependent urban fidelity

![Cross-city urban fidelity](outputs/cross_city/paper_figures/FIG_CC3_fidelity_resolution.png)

### Accessibility speedup versus fidelity

![Accessibility speedup versus fidelity](outputs/cross_city/paper_figures/FIG_CC4_accessibility_speedup_vs_fidelity.png)

> If your local analysis script writes these images to a different subdirectory, change only the Markdown image paths; the numerical master tables remain the source of truth.

---

## 7. Physical route visualization

Representative OD routes are projected over the common OSM walking network. Current paper assets include:

```text
FIG_ROUTED_OD_H8_manhattan.png
FIG_ROUTED_OD_H8_kendall.png
FIG_ROUTED_OD_H8_street_zoom.png 
```
### OD Madrid 
![OD Madrid Zoom](outputs/paper_figures/FIG_ROUTED_OD_H8_street_zoom.png)

### OD Cambridge 

![OD Cambridge Zoom](outputs/cambridge_ma/paper_figures/FIG_ROUTED_OD_H8_kendall.png)

### OD New York 

![OD Mantattan Zoom](outputs/new_york_ny/paper_figures/FIG_ROUTED_OD_H8_manhattan.png)


They represent:
- **New York City:** Manhattan / Midtown zoom;
- **Cambridge:** Kendall / Volpe-area zoom;
- **Madrid:** street-level local zoom.

These maps are **not routing-performance comparisons**. They visualize how destination differences produced by the spatial architectures propagate to physically realizable paths over identical routing infrastructure.

---

## 8. Repository structure

```text
madrid_h3_urban_benchmark/
├── analysis/
│   ├── 04_final_madrid_visual_analysis.py
│   ├── 05_cross_city_paper_analysis.py
│   ├── 06_master_cross_city_analysis.py
│   └── run_all_analysis.py
├── configs/
├── data/
│   ├── raw/                  # Madrid raw data; ignored
│   ├── processed/            # Madrid generated data; ignored
│   ├── cache/                # Madrid caches; ignored
│   ├── cambridge_ma/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── cache/
│   └── new_york_ny/
│       ├── raw/
│       ├── processed/
│       └── cache/
├── experiments/
│   ├── 00_city_smoke_test.py
│   ├── 00_semantic_city_check.py
│   ├── 01_architecture_benchmark.py
│   └── 03_urban_fidelity.py
├── outputs/
│   ├── architecture/
│   ├── urban_fidelity/
│   ├── cambridge_ma/
│   ├── new_york_ny/
│   └── cross_city/
├── scripts/
├── src/h3uab/
├── tests/
├── README.md
└── requirements.txt
```

Large raw rasters, GraphML networks and binary caches should remain outside Git tracking.

---

## 9. Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd madrid_h3_urban_benchmark

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 10. Build city data

### Cambridge, Massachusetts

The builder prepares the municipal boundary, OSM walking graph, OSM POIs, WorldPop raster, clipped population raster and H3 r9 population aggregation.

```bash
python scripts/build_cambridge_data.py \
  --worldpop-url "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/USA/usa_ppp_2020_UNadj_constrained.tif"
```

Verify it with:

```bash
python scripts/smoke_test_cambridge.py
```

### New York City

```bash
python scripts/build_new_york_data.py
python scripts/smoke_test_new_york.py
```

A successful build produces the boundary, walking graph, OSM POIs, clipped WorldPop raster and H3 population file.

---

## 11. Semantic normalization

Raw OSM tags are normalized into semantic urban layers (`D3`-`D25`).

| Layer | Meaning |
|---|---|
| D3 | Retail Store |
| D4 | Commercial |
| D5 | Education |
| D8 | Healthcare |
| D14 | Headquarter Office |
| D15 | Grocery-Market |
| D16 | Recreation & Fitness |
| D17 | Pharmacy |
| D19 | Daycare Center |
| D20 | Coworking Office |
| D21 | Restaurant |
| D22 | Cafe |
| D23 | Bar |
| D24 | Pub |
| D25 | Park |

Check normalization with:

```bash
python experiments/00_semantic_city_check.py --city cambridge_ma
python experiments/00_semantic_city_check.py --city new_york_ny
```

---

## 12. Core smoke test

Before a large benchmark:

```bash
python experiments/00_city_smoke_test.py --city cambridge_ma --agents 1000
python experiments/00_city_smoke_test.py --city new_york_ny --agents 1000
```

The smoke test verifies world construction, both computational backends and paired decisions.

---

## 13. Architecture benchmark

One population:

```bash
python experiments/01_architecture_benchmark.py \
  --city new_york_ny \
  --agents 10000 \
  --repeats 3
```

Complete 10K-to-1M sweep:

```bash
python experiments/01_architecture_benchmark.py \
  --city new_york_ny \
  --agents 10000 100000 500000 1000000 \
  --repeats 3
```

The experiment reports Continuous and H3 latency, throughput and speedup. Routing is excluded because it is common physical infrastructure.

---

## 14. Urban-fidelity experiment

```bash
python experiments/03_urban_fidelity.py \
  --city new_york_ny \
  --agents 10000
```

Supported city keys are:

```text
madrid
cambridge_ma
new_york_ny
```

The fidelity run produces a summary and diagnostics including:

```text
destination_separations.csv
paired_trip_endpoints.csv
occupancy_H7.csv
occupancy_H8.csv
occupancy_H9.csv
visit_density_H7.csv
visit_density_H8.csv
visit_density_H9.csv
od_flows_H7.csv
od_flows_H8.csv
od_flows_H9.csv
```

The paired endpoint diagnostic allows representative routes to be realized over the common real OSM walking graph without repeating the architecture sweep.

---

## 15. Master cross-city tables

After completing the three cities:

```bash
python analysis/06_master_cross_city_analysis.py
```

Generated files:

```text
outputs/cross_city/
├── MASTER_cross_city_full.csv
├── city_characteristics_long.csv
├── computational_performance_1M_long.csv
├── fidelity_10K_long.csv
├── TABLE_1_city_characteristics.csv
├── TABLE_2_computational_performance_1M.csv
├── TABLE_3_fidelity_10K.csv
└── TABLE_PAPER_cross_city_summary.csv
```

These are the numerical source for the paper's experimental tables.

---

## 16. Generate figures

For the analysis pipeline:

```bash
python analysis/run_all_analysis.py
```

The visualization convention is intentional:
- Continuous and H3 heatmaps use the **same color scale**;
- difference maps are **zero-centered and symmetric**;
- H7 represents city-scale structure;
- H8 represents neighborhood-scale structure;
- H9 exposes finer local-allocation divergence.

---

## 17. Reproducibility workflow

```text
Build/download municipal data
            |
            v
Validate boundary + OSM graph + POIs + population
            |
            v
Normalize POIs into D3-D25 semantic layers
            |
            v
Run 1K-agent smoke test
            |
            v
Architecture sweep: 10K -> 100K -> 500K -> 1M
            |
            v
Paired urban-fidelity validation at 10K
            |
            v
Generate city visual diagnostics
            |
            v
Generate master cross-city tables
            |
            v
Generate publication figures
```

Expensive raw-data preparation and architecture sweeps therefore do **not** need to be repeated simply to regenerate paper figures.

---

## 18. How to interpret the experiment

### Supported by the benchmark

The experiments test whether hierarchical H3 indexing can reduce the computational cost of repeated urban spatial queries while preserving aggregate urban patterns.

The current three-city results show:
- substantial speedups in every tested spatial operation;
- strong end-to-end full-agent decision acceleration;
- very high H7 OD/occupancy agreement;
- strong H8 fidelity;
- increasing local divergence at H9;
- identical city-level amenity-usage correlation in the paired experiments.

### Not claimed

The benchmark does **not** claim that:
- H3 replaces the physical street network;
- H3 routing is faster than OSM routing;
- H11/H12 provide additional urban-analysis fidelity;
- exact destination identity is always preserved;
- fine-scale H9 allocation is identical to the Continuous backend.

The central claim is a **speed-fidelity trade-off for spatial computation**, evaluated independently from physical routing.

---

## 19. Data provenance and Git policy

The project uses real geospatial infrastructure including OpenStreetMap-derived POIs and walking networks, WorldPop population rasters, municipal boundaries and H3 hierarchical spatial indexing.

Raw and generated geospatial assets are intentionally excluded from Git because some files are hundreds of megabytes or larger. The repository should contain the **builders, algorithms, experiments, compact result tables and publication figures** needed to reproduce and audit the work.

---

## 20. Current experimental status

| Component | Madrid | Cambridge | NYC |
|---|:---:|:---:|:---:|
| City data | Yes | Yes | Yes |
| Semantic normalization | Yes | Yes | Yes |
| OSM walking graph | Yes | Yes | Yes |
| Architecture benchmark to 1M | Yes | Yes | Yes |
| Fidelity validation at 10K | Yes | Yes | Yes |
| Cross-city master tables | Yes | Yes | Yes |
| Paper visual analysis | Yes | Yes | Yes |

The repository is now a **three-city cross-city validation framework**, not a Madrid-only benchmark.

---

## 21. Scientific-analysis package

The former scientific-analysis README described the frozen experiment as three components: architectural efficiency, urban fidelity and spatial visualization. It also established the crucial rules that H11/H12 are localization/visualization only and that real OSM mobility is common infrastructure.

Those principles are now consolidated here so the repository can use **one authoritative README**.

---

## Citation

If you use this repository in academic work, cite the associated paper once its final bibliographic information is available.

## License

Add the repository license here when finalized.
