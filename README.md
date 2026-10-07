# H3 Urban Architecture Benchmark

A reproducible **multi-city benchmark for scalable urban simulation with
H3**, comparing a continuous spatial-indexed backend with a
multi-resolution H3 backend while preserving a common physical mobility
layer and validating architectural non-degradation against observed
mobility.

> **Research question:** How much computational efficiency can a
> hierarchical H3 representation provide for large-scale urban agent
> simulation, how much urban fidelity is retained across spatial scales,
> and does H3 preserve the empirical agreement with observed mobility
> achieved by the Continuous baseline?

The current benchmark covers **Madrid, Cambridge (MA), and New York
City**. Isolated architectural workloads scale to **1,000,000 queries**,
while the temporal end-to-end benchmark evaluates **10,000 and 100,000
agents** under a common five-decision schedule. External OD validation
compares Continuous and H3 outputs against independent observed mobility
references in all three cities.

------------------------------------------------------------------------

## 1. What this repository evaluates

The framework deliberately separates five experimental questions.

### Architectural efficiency

The Continuous and H3 backends are benchmarked on: - pure spatial
lookup; - urban-context retrieval; - accessibility queries; - candidate
retrieval; - full agent decision.

The architecture sweep scales from **10K to 1M agents/queries**.

### Architectural agreement

At `N = 10,000`, paired Continuous/H3 simulations measure architectural
agreement between the Continuous and H3 representations using: -
accessibility correlation; - destination spatial equivalence; -
trip-distance fidelity; - OD-flow similarity at H7/H8/H9; - final
occupancy similarity at H7/H8/H9; - amenity-usage agreement.

### Temporal end-to-end performance and amortization

A separate temporal benchmark evaluates the complete repeated decision
loop at **10K and 100K agents**. The schedule produces **five spatial
decisions per agent** in all three cities. The experiment reports
paired-median temporal-loop speedup, cold- and warm-total speedup, and
break-even in **spatial decisions**.

The repetition protocol is **3 repeats at 10K agents** and **5 repeats
at 100K agents** for each city. Break-even is reported primarily in
decisions because the five-decision schedule is an experimental protocol
rather than a universal property of an agent.

### Physical-network realization

Representative paired origin-destination decisions are projected onto
the **same real OpenStreetMap walking graph**. Routing is deliberately
*not* part of the H3 speedup claim.

### External empirical OD validation

A separate validation layer compares the Continuous and H3
representations against the **same independent observed OD reference**.
Cambridge and New York City use U.S. Census LEHD Origin-Destination
Employment Statistics (LODES) JT00 commuting flows; Madrid uses the INE
mobility-area OD source. Observed flows are projected to H7/H8/H9 under
a common spatial protocol and compared with normalized OD-share metrics.

The primary external-validation claim is deliberately differential: **H3
should not materially degrade the empirical agreement achieved by the
Continuous baseline**. Total variation distance (TVD) and normalized
Jensen-Shannon divergence (JSD) are used because the observed OD
matrices are flow distributions. Absolute behavioral realism is not
inferred from a near-zero H3-minus-Continuous error gap.

------------------------------------------------------------------------

## 2. Architectural rules

1.  **H7/H8/H9 are urban-analysis and fidelity scales.**
2.  **H11/H12 are localization/visualization layers only** and never
    enter the fidelity claim.
3.  **Physical routing is not discretized with H3.**
4.  Both backends use the **same OSM physical network** for route
    visualization.
5.  Heatmaps aggregate real agent/destination coordinates at H7/H8/H9.
6.  Speedup therefore measures the spatial-computational representation,
    not a replacement of street-network routing.

------------------------------------------------------------------------

## 3. Cross-city benchmark

  ------------------------------------------------------------------------------------
  City            Raw OSM   Semantic   Population   Represented  OSM graph   OSM graph
                     POIs       POIs     H3 cells    population      nodes       edges
  ------------ ---------- ---------- ------------ ------------- ---------- -----------
  Madrid           38,949     38,949        2,037     2,337,708    122,237     373,234

  Cambridge,        7,831      2,102          175       103,855     20,322      54,756
  MA                                                                       

  New York        305,387    126,595        6,747     8,037,287    361,278   1,108,754
  City                                                                     
  ------------------------------------------------------------------------------------

These cities provide substantially different spatial scales, POI
densities, network sizes and urban morphologies.

------------------------------------------------------------------------

## 4. Headline computational results at 1M queries

Speedup = `continuous execution time / H3 execution time`.

  Operation                       Madrid    Cambridge   New York City
  ------------------------- ------------ ------------ ---------------
  Spatial lookup              **47.49x**   **29.95x**      **38.97x**
  Urban context retrieval     **26.60x**   **16.19x**      **18.35x**
  Accessibility query         **83.70x**   **50.24x**      **43.44x**
  Candidate retrieval         **42.03x**   **40.22x**      **41.63x**
  Full agent decision          **4.21x**    **2.86x**       **3.27x**

The important result is not one maximum number: the H3 architecture
remains faster across all three cities and all five computational
operations, including the end-to-end agent decision.

### Full-agent decision latency at 1M

  City                  Continuous               H3   Speedup
  --------------- ---------------- ---------------- ---------
  Madrid            75.67 us/query   17.97 us/query     4.21x
  Cambridge         62.96 us/query   22.01 us/query     2.86x
  New York City     51.39 us/query   15.72 us/query     3.27x

### Temporal end-to-end results

The 1M-query architecture benchmark above isolates computational
operations. The temporal benchmark is a separate end-to-end experiment
at **10K and 100K agents**, with five spatial decisions per agent.

  --------------------------------------------------------------------------------
  City            Agents    Temporal  Cold total    Cold BE  Warm total    Warm BE
                             speedup                                    
  ----------- ---------- ----------- ----------- ---------- ----------- ----------
  Madrid             10K   **3.26x**   **1.82x**      17.2K   **3.21x**       0.3K

  Madrid            100K   **3.24x**   **2.98x**      19.2K   **3.20x**       2.6K

  Cambridge          10K   **2.21x**   **1.79x**       9.8K   **2.19x**       0.7K

  Cambridge         100K   **2.22x**   **2.15x**      13.7K   **2.20x**       4.9K

  New York           10K   **2.60x**   **0.28x**     265.9K   **2.06x**       7.3K
  City                                                                  

  New York          100K   **2.54x**   **1.43x**     252.6K   **2.49x**      11.0K
  City                                                                  
  --------------------------------------------------------------------------------

`Temporal speedup` is the paired median of Continuous/H3 runtime ratios.
`Cold total` includes substrate construction; `Warm total` reuses the
precomputed substrate. BE denotes break-even in thousands of spatial
decisions.

The cold-start results deliberately retain the operating regime in which
H3 does not pay off: for New York City at 10K agents, substrate
construction outweighs runtime savings (`0.28x`), whereas at 100K agents
the cold-start total crosses into a net gain (`1.43x`).

------------------------------------------------------------------------

## 5. Architectural agreement at 10K agents

  Metric                         Madrid   Cambridge   New York City
  ---------------------------- -------- ----------- ---------------
  Accessibility Pearson          0.9905      0.9796          0.9917
  Exact destination match        0.6384      0.5922          0.6485
  Destination within 500 m       0.8834      0.7937          0.8229
  Destination within 1,000 m     0.9786      0.9524          0.9306
  Trip-distance Pearson          0.8834      0.8265          0.8964
  OD flow H7 Pearson             0.9990      0.9989          0.9945
  OD flow H8 Pearson             0.9891      0.9758          0.9584
  OD flow H9 Pearson             0.9168      0.8969          0.8328
  Occupancy H7 Pearson           0.9928      0.9972          0.9840
  Occupancy H8 Pearson           0.9174      0.8452          0.8370
  Occupancy H9 Pearson           0.7528      0.6446          0.6991
  Amenity usage Pearson          1.0000      1.0000          1.0000

The scale effect is clear: **city-scale agreement is extremely high at
H7, neighborhood-scale agreement remains strong at H8, and finer local
allocation differences become more visible at H9.** These metrics
quantify internal agreement between the two computational architectures
under the paired protocol. External empirical validation is evaluated
separately in Section 16 against independent observed OD references.
Amenity-usage Pearson = 1.0 is retained as a consistency control rather
than independent evidence of fidelity.

------------------------------------------------------------------------

## 6. External validation against observed mobility

The external-validation experiment compares both computational
representations against the **same observed OD reference**, so the key
architectural quantity is the incremental error introduced by H3:

\[ `\Delta `{=tex}E = E(H3,`\mathrm{Observed}`{=tex}) -
E(`\mathrm{Continuous}`{=tex},`\mathrm{Observed}`{=tex}). \]

For error metrics such as TVD and normalized JSD, a gap near zero
supports empirical **non-degradation**. Negative values indicate
slightly lower H3 error and positive values slightly higher H3 error;
neither sign is interpreted as a general improvement unless supported
independently.

### Primary H7 cross-city result

  ---------------------------------------------------------------------------------------------------------
  City              Pearson   Pearson          TVD   TVD H3      ΔTVD          JSD   JSD H3   Continuous↔H3
                 Continuous        H3   Continuous                      Continuous                  Pearson
  ------------ ------------ --------- ------------ -------- --------- ------------ -------- ---------------
  Cambridge,         0.7741    0.7755       0.5536   0.5304   -0.0232       0.2687   0.2546          0.9984
  MA                                                                                        

  Madrid             0.6801    0.6818       0.7523   0.7533   +0.0010       0.3869   0.3876          0.9990

  New York           0.2744    0.2717       0.9012   0.9020   +0.0008       0.5483   0.5499          0.9893
  City                                                                                      
  ---------------------------------------------------------------------------------------------------------

The absolute observed-reference fit varies substantially across cities,
but the **Continuous-to-H3 change remains small**. Cambridge shows the
largest H7 TVD change and it is in the error-reducing direction; this is
reported as a scale-dependent result rather than evidence that H3 is
intrinsically more accurate.

### Resolution sensitivity

  ------------------------------------------------------------------------------------------------
  City             H3 Pearson C   Pearson   ΔPearson    TVD C   TVD H3      ΔTVD    JSD C   JSD H3
                                       H3                                                 
  ----------- ------- --------- --------- ---------- -------- -------- --------- -------- --------
  Cambridge         7    0.7741    0.7755    +0.0014   0.5536   0.5304   -0.0232   0.2687   0.2546

  Cambridge         8    0.2198    0.2562    +0.0364   0.8196   0.8172   -0.0024   0.4837   0.4778

  Cambridge         9   -0.0006    0.0002    +0.0008   0.9629   0.9682   +0.0053   0.6382   0.6428

  Madrid            7    0.6801    0.6818    +0.0017   0.7523   0.7533   +0.0010   0.3869   0.3876

  Madrid            8    0.2292    0.2304    +0.0012   0.9105   0.9107   +0.0001   0.5768   0.5766

  Madrid            9    0.0320    0.0313    -0.0007   0.9922   0.9923   +0.0001   0.6767   0.6774

  New York          7    0.2744    0.2717    -0.0027   0.9012   0.9020   +0.0008   0.5483   0.5499
  City                                                                                    

  New York          8    0.1485    0.1475    -0.0010   0.9671   0.9673   +0.0002   0.6336   0.6340
  City                                                                                    

  New York          9    0.0447    0.0423    -0.0024   0.9947   0.9948   +0.0001   0.6803   0.6806
  City                                                                                    
  ------------------------------------------------------------------------------------------------

At H9 the absolute OD-pair agreement becomes very weak in all cities.
Critically, the Continuous baseline degrades at the same scale even
though it does not use hierarchical H3 discretization. The
fine-resolution collapse therefore cannot be attributed uniquely to H3.

### NYC audit and support mismatch

New York City has the lowest absolute H7 OD-pair Pearson. A dedicated
audit shows that this is **not an H3-specific failure**: the Continuous
and H3 results remain nearly identical. The observed LODES matrix
contains 31,577 positive H7 OD pairs, whereas the 10K-agent simulation
realizes only hundreds of positive pairs. This support mismatch
depresses absolute pairwise agreement and produces predominantly
negative residuals on high-flow observed corridors.

The audit also shows much stronger origin marginals than OD-pair
agreement, while destination marginals are weaker. This is consistent
with a shared behavioral destination mechanism that does not reproduce
the full concentration of observed employment hubs. Because the
behavioral mechanism and simulated support are shared by both spatial
backends, these limitations affect **absolute fit**, not the
differential H3-vs-Continuous non-degradation claim.

### Madrid purpose sensitivity

The Madrid INE M1 source represents daytime/usual mobility rather than a
pure home-work commuting matrix. The primary comparison uses the
simulated work OD export for consistency with the LODES-based cities,
while an additional `all`-purpose sensitivity run increases simulated
movements from approximately 11K to 50K. The Continuous-H3 error gap
remains near zero under both choices, supporting robustness of the
architectural conclusion to this purpose-selection choice.

### External-validation figure

![Cross-city H7 external OD
validation](outputs/cross_city_external_validation/paper_figures/FIG_EV1_cross_city_H7_external_validation_routed.png)

Observed and simulated residual OD structures are realized over the same
physical OSM network. Continuous and H3 residuals use the same signed
scale within each city. The visual comparison is illustrative;
quantitative non-degradation is established by the ΔTVD and ΔJSD metrics
above rather than by visual subtraction.

## 7. Main paper figures

The paths below follow the current cross-city output convention.

### Cross-city computational speedup

![Cross-city computational
speedup](outputs/cross_city/paper_figures/FIG_CC1_speedup_scaling.png)

### Full-agent decision scaling

![Full-agent decision
latency](outputs/cross_city/paper_figures/FIG_CC2_full_decision_scaling.png)

### Resolution-dependent urban fidelity

![Cross-city urban
fidelity](outputs/cross_city/paper_figures/FIG_CC3_fidelity_resolution.png)

### Accessibility speedup versus fidelity

![Accessibility speedup versus
fidelity](outputs/cross_city/paper_figures/FIG_CC4_accessibility_speedup_vs_fidelity.png)

### Temporal end-to-end speedup

![Temporal end-to-end
speedup](outputs/cross_city/paper_figures/FIG_CC5_end_to_end_temporal_speedup.png)

### Cold/warm break-even

![Break-even
decisions](outputs/cross_city/paper_figures/FIG_CC6_break_even_decisions.png)

> If your local analysis script writes these images to a different
> subdirectory, change only the Markdown image paths; the numerical
> master tables remain the source of truth.

------------------------------------------------------------------------

## 8. Physical route visualization

Representative OD routes are projected over the common OSM walking
network. Current paper assets include:

### OD Madrid

![OD Madrid
Zoom](outputs/paper_figures/FIG_ROUTED_OD_H8_street_zoom.png)

### OD Cambridge

![OD Cambridge
Zoom](outputs/cambridge_ma/paper_figures/FIG_ROUTED_OD_H8_kendall.png)

### OD New York

![OD Mantattan
Zoom](outputs/new_york_ny/paper_figures/FIG_ROUTED_OD_H8_manhattan.png)

They represent: - **New York City:** Manhattan / Midtown zoom; -
**Cambridge:** Kendall / Volpe-area zoom; - **Madrid:** street-level
local zoom.

These maps are **not routing-performance comparisons**. They visualize
how destination differences produced by the spatial architectures
propagate to physically realizable paths over identical routing
infrastructure.

------------------------------------------------------------------------

## 9. Repository structure

``` text
madrid_h3_urban_benchmark/
├── analysis/
│   ├── 04_final_madrid_visual_analysis.py
│   ├── 05_cross_city_paper_analysis.py
│   ├── 06_master_cross_city_analysis.py
│   ├── 08_cross_city_external_validation_analysis.py
│   ├── 08a_nyc_external_validation_audit.py
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
│   ├── 02_end_to_end_benchmark.py
│   ├── 03_urban_fidelity.py
│   └── 07_external_od_validation.py
├── outputs/
│   ├── architecture/
│   ├── urban_fidelity/
│   ├── cambridge_ma/
│   ├── new_york_ny/
│   ├── external_od_validation/
│   ├── external_od_validation_sensitivity/
│   ├── cross_city_external_validation/
│   └── cross_city/
├── scripts/
├── src/h3uab/
├── tests/
├── README.md
└── requirements.txt
```

Large raw rasters, GraphML networks and binary caches should remain
outside Git tracking.

------------------------------------------------------------------------

## 10. Installation

``` bash
git clone <YOUR_REPOSITORY_URL>
cd madrid_h3_urban_benchmark

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 11. Build city data

### Cambridge, Massachusetts

The builder prepares the municipal boundary, OSM walking graph, OSM
POIs, WorldPop raster, clipped population raster and H3 r9 population
aggregation.

``` bash
python scripts/build_cambridge_data.py \
  --worldpop-url "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/USA/usa_ppp_2020_UNadj_constrained.tif"
```

Verify it with:

``` bash
python scripts/smoke_test_cambridge.py
```

### New York City

``` bash
python scripts/build_new_york_data.py
python scripts/smoke_test_new_york.py
```

A successful build produces the boundary, walking graph, OSM POIs,
clipped WorldPop raster and H3 population file.

------------------------------------------------------------------------

## 12. Semantic normalization

Raw OSM tags are normalized into semantic urban layers (`D3`-`D25`).

  Layer   Meaning
  ------- ----------------------
  D3      Retail Store
  D4      Commercial
  D5      Education
  D8      Healthcare
  D14     Headquarter Office
  D15     Grocery-Market
  D16     Recreation & Fitness
  D17     Pharmacy
  D19     Daycare Center
  D20     Coworking Office
  D21     Restaurant
  D22     Cafe
  D23     Bar
  D24     Pub
  D25     Park

Check normalization with:

``` bash
python experiments/00_semantic_city_check.py --city cambridge_ma
python experiments/00_semantic_city_check.py --city new_york_ny
```

------------------------------------------------------------------------

## 13. Core smoke test

Before a large benchmark:

``` bash
python experiments/00_city_smoke_test.py --city cambridge_ma --agents 1000
python experiments/00_city_smoke_test.py --city new_york_ny --agents 1000
```

The smoke test verifies world construction, both computational backends
and paired decisions.

------------------------------------------------------------------------

## 14. Architecture benchmark

One population:

``` bash
python experiments/01_architecture_benchmark.py \
  --city new_york_ny \
  --agents 10000 \
  --repeats 3
```

Complete 10K-to-1M sweep:

``` bash
python experiments/01_architecture_benchmark.py \
  --city new_york_ny \
  --agents 10000 100000 500000 1000000 \
  --repeats 3
```

The experiment reports Continuous and H3 latency, throughput and
speedup. Routing is excluded because it is common physical
infrastructure.

------------------------------------------------------------------------

## 15. Temporal end-to-end experiment

Run the temporal decision-loop benchmark separately from the 1M
architecture sweep.

``` bash
python experiments/02_end_to_end_benchmark.py --city madrid --agents 10000 --repeats 3 --static-city-mode both
python experiments/02_end_to_end_benchmark.py --city madrid --agents 100000 --repeats 5 --static-city-mode both
```

Repeat with `cambridge_ma` and `new_york_ny`. The temporal protocol
generates five spatial decisions per agent in all three cities. Semantic
diversity can differ with each city's active-layer coverage and schedule
fallbacks even though the decision count is held constant.

The benchmark distinguishes temporal runtime, cold preparation, warm
preparation, cold/warm total time, and break-even. Temporal speedup is
summarized using the **median of paired per-repeat speedup ratios**.
Break-even is computed from aggregated median per-decision costs and
median preparation times, with sensitivity bounds derived from observed
per-decision extrema.

------------------------------------------------------------------------

## 16. Architectural-agreement experiment

``` bash
python experiments/03_urban_fidelity.py \
  --city new_york_ny \
  --agents 10000
```

Supported city keys are:

``` text
madrid
cambridge_ma
new_york_ny
```

The fidelity run produces a summary and diagnostics including:

``` text
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

The paired endpoint diagnostic allows representative routes to be
realized over the common real OSM walking graph without repeating the
architecture sweep.

------------------------------------------------------------------------

## 17. External OD validation workflow

The external validation is run **after** the 10K paired
architectural-agreement experiment because it consumes the paired OD
exports produced by `03_urban_fidelity.py`.

### Cambridge, MA --- LODES JT00

``` bash
python experiments/07_external_od_validation.py \
  --city cambridge_ma \
  --observed-od /path/to/ma_od_main_JT00_2023.csv \
  --tiger-blocks data/external_od/cambridge_ma/tiger/tl_2020_25_tabblock20/tl_2020_25_tabblock20.shp \
  --domain-boundary data/cambridge_ma/processed/cambridge_boundary.geojson \
  --domain-rule both \
  --paired-sim-od-dir outputs/cambridge_ma/urban_fidelity/diagnostics_n10000
```

### New York City --- LODES JT00

``` bash
python experiments/07_external_od_validation.py \
  --city new_york_ny \
  --observed-od /path/to/ny_od_main_JT00_2023.csv \
  --tiger-blocks data/external_od/new_york_ny/tiger/tl_2020_36_tabblock20/tl_2020_36_tabblock20.shp \
  --domain-boundary data/new_york_ny/processed/new_york_boundary.geojson \
  --domain-rule both \
  --paired-sim-od-dir outputs/new_york_ny/urban_fidelity/diagnostics_n10000
```

### Madrid --- INE mobility areas

Madrid uses the exact rectangular simulation AOI defined by the
benchmark configuration (`lat 40.38–40.50`, `lng -3.75–-3.55`) and
stored as `data/processed/madrid_boundary.geojson`.

``` bash
python experiments/07_external_od_validation.py \
  --city madrid \
  --observed-od /path/to/datos_disponibles.zip \
  --madrid-areas /path/to/shapefiles_celdas_marzo2020.zip \
  --domain-boundary data/processed/madrid_boundary.geojson \
  --domain-rule both \
  --paired-sim-od-dir outputs/urban_fidelity/diagnostics_n10000 \
  --paired-sim-od-kind work
```

Madrid all-purpose sensitivity:

``` bash
python experiments/07_external_od_validation.py \
  --city madrid \
  --observed-od /path/to/datos_disponibles.zip \
  --madrid-areas /path/to/shapefiles_celdas_marzo2020.zip \
  --domain-boundary data/processed/madrid_boundary.geojson \
  --domain-rule both \
  --paired-sim-od-dir outputs/urban_fidelity/diagnostics_n10000 \
  --paired-sim-od-kind all \
  --output-dir outputs/external_od_validation_sensitivity
```

### Cross-city synthesis and NYC audit

``` bash
python analysis/08_cross_city_external_validation_analysis.py
python analysis/08a_nyc_external_validation_audit.py
```

The cross-city analysis consumes the already generated validation
tables; it does not alter the frozen validation protocol. The NYC audit
is diagnostic only and does not change the benchmark or behavioral
model.

## 18. Master cross-city tables

After completing the three cities:

``` bash
python analysis/06_master_cross_city_analysis.py
```

Generated files:

``` text
outputs/cross_city/
├── MASTER_cross_city_full.csv
├── city_characteristics_long.csv
├── computational_performance_1M_long.csv
├── fidelity_10K_long.csv
├── end_to_end_temporal_long.csv
├── TABLE_1_city_characteristics.csv
├── TABLE_2_computational_performance_1M.csv
├── TABLE_3_fidelity_10K.csv
├── TABLE_4_end_to_end_temporal_amortization.csv
├── TABLE_4_end_to_end_temporal_amortization.tex
└── TABLE_PAPER_cross_city_summary.csv
```

These are the numerical source for the paper's experimental tables.

------------------------------------------------------------------------

## 19. Generate figures

For the analysis pipeline:

``` bash
python analysis/run_all_analysis.py
```

The visualization convention is intentional: - Continuous and H3
heatmaps use the **same color scale**; - difference maps are
**zero-centered and symmetric**; - H7 represents city-scale structure; -
H8 represents neighborhood-scale structure; - H9 exposes finer
local-allocation divergence.

------------------------------------------------------------------------

## 20. Reproducibility workflow

``` text
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
Paired architectural-agreement validation at 10K
            |
            v
External OD validation against LODES / INE at H7-H9
            |
            v
Cross-city external-validation synthesis + NYC diagnostic audit
            |
            v
Temporal end-to-end: 10K (3 repeats) + 100K (5 repeats)
            |
            v
Cold/warm amortization analysis
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

Expensive raw-data preparation and architecture sweeps therefore do
**not** need to be repeated simply to regenerate paper figures.

------------------------------------------------------------------------

## 21. How to interpret the experiment

### Supported by the benchmark

The experiments test whether hierarchical H3 indexing can reduce the
computational cost of repeated urban spatial queries while preserving
aggregate urban patterns.

The current three-city results show: - substantial speedups in every
tested spatial operation; - strong end-to-end full-agent decision
acceleration; - very high H7 OD/occupancy agreement; - strong H8
fidelity; - increasing local divergence at H9; - identical city-level
amenity-usage correlation in the paired experiments; - external
observed-mobility validation in all three cities; - near-zero
incremental H3-vs-Continuous TVD/JSD error gaps across the primary H7
comparisons.

### Not claimed

The benchmark does **not** claim that: - H3 replaces the physical street
network; - H3 routing is faster than OSM routing; - H11/H12 provide
additional urban-analysis fidelity; - exact destination identity is
always preserved; - fine-scale H9 allocation is identical to the
Continuous backend; - the shared behavioral destination model reproduces
every empirical employment hub or the full observed OD support; -
external non-degradation implies absolute behavioral realism.

The central claim is a **speed-fidelity trade-off for spatial
computation with externally validated architectural non-degradation**,
evaluated independently from physical routing. The observed-data
experiments support the narrower statement that replacing the Continuous
spatial representation with H3 does not materially worsen agreement with
the selected mobility references.

------------------------------------------------------------------------

## 22. Data provenance and Git policy

The project uses real geospatial infrastructure including
OpenStreetMap-derived POIs and walking networks, WorldPop population
rasters, municipal boundaries and H3 hierarchical spatial indexing.
External validation additionally uses U.S. Census LEHD
Origin-Destination Employment Statistics (LODES) JT00 flows with
TIGER/Line Census Block geometries for Cambridge and New York City, and
INE mobility-study OD flows with INE mobility-area geometries for
Madrid.

Raw and generated geospatial assets are intentionally excluded from Git
because some files are hundreds of megabytes or larger. The repository
should contain the **builders, algorithms, experiments, compact result
tables and publication figures** needed to reproduce and audit the work.

------------------------------------------------------------------------

## 23. Current experimental status

  Component                                   Madrid   Cambridge   NYC
  ------------------------------------------ -------- ----------- -----
  City data                                    Yes        Yes      Yes
  Semantic normalization                       Yes        Yes      Yes
  OSM walking graph                            Yes        Yes      Yes
  Architecture benchmark to 1M                 Yes        Yes      Yes
  Architectural agreement at 10K               Yes        Yes      Yes
  Temporal end-to-end at 10K                   Yes        Yes      Yes
  Temporal end-to-end at 100K                  Yes        Yes      Yes
  Cold/warm preparation analysis               Yes        Yes      Yes
  Break-even analysis                          Yes        Yes      Yes
  Cross-city master tables (Tables 1-4)        Yes        Yes      Yes
  External observed OD validation              Yes        Yes      Yes
  H7/H8/H9 external validation                 Yes        Yes      Yes
  Cross-city external-validation synthesis     Yes        Yes      Yes
  Paper visual analysis (CC1-CC6)              Yes        Yes      Yes
  Routed external-validation figure            Yes        Yes      Yes
  Candidate-architecture ablation                No         Yes       No

The repository is now a **three-city cross-city validation framework
with independent observed-mobility validation**, not a Madrid-only
benchmark.

The candidate-architecture ablation is a **separate Cambridge-only
experiment**. Madrid and NYC were not run in that ablation because the
required POI and population inputs were unavailable in the measurement
environment; the three-city results above belong to the earlier protocols.

------------------------------------------------------------------------

## 24. Scientific-analysis package

The former scientific-analysis README described the frozen experiment as
three components: architectural efficiency, urban fidelity and spatial
visualization. It also established the crucial rules that H11/H12 are
localization/visualization only and that real OSM mobility is common
infrastructure.

Those principles are now consolidated here so the repository can use
**one authoritative README**.

## Candidate-architecture ablation (Cambridge only)

This experiment separates exhaustive-radius retrieval, post-retrieval
candidate-score reduction, bounded BallTree kNN retrieval, and the
H3-based cached/bucketed architecture. It replays identical independent
queries through five configurations, reports paired timings and realized
candidate counts, and compares destination identity with the original
continuous implementation (a reference, not behavioural truth).

For the Cambridge million-query workload, median complete-decision latencies
were 63.33 µs (exhaustive Continuous), 69.26 µs (radius-cap24), 62.65 µs
(BallTree kNN24), 23.36 µs (H3-24), and 23.21 µs (H3-48), across five paired
repetitions. H3-24 scored only 1.07 POIs on average versus 21.43 for either
bounded Continuous variant. The residual speed gap is therefore an
architecture contrast, not an isolated effect of H3 indexing.

The runner, exact reproduction commands, and fairness boundaries are in
[`experiments/CANDIDATE_ABLATION.md`](experiments/CANDIDATE_ABLATION.md).
Machine-readable timings, diagnostics, metadata and publication figures are
in [`results/candidate_ablation/`](results/candidate_ablation/). The
cross-city ablation remains outstanding.

------------------------------------------------------------------------

## External validation data sources

The repository does not redistribute the raw external mobility datasets.
Reproducibility requires obtaining them from the original providers.

-   **U.S. Census Bureau --- LEHD Origin-Destination Employment
    Statistics (LODES):** JT00 Origin-Destination files are used for
    Massachusetts and New York. JT00 represents all jobs and provides
    Census Block home/work identifiers and `S000` flow counts.
-   **U.S. Census Bureau --- LEHD/LODES technical documentation:** use
    the official LODES documentation and version/data-year metadata
    corresponding to the downloaded files.
-   **U.S. Census Bureau --- TIGER/Line 2020 Census Blocks:**
    `tl_2020_25_tabblock20` for Massachusetts and
    `tl_2020_36_tabblock20` for New York provide the block geometries
    used to spatially align LODES endpoints.
-   **Instituto Nacional de Estadística (INE), Spain --- Estudio de
    Movilidad a partir de la telefonía móvil:** Madrid validation uses
    the available OD mobility files together with the March 2020
    mobility-area shapefiles. The M1/daytime-usual-mobility source is
    not interpreted as a pure work-commuting matrix.

The manuscript bibliography contains the formal citations for the
Census/LEHD/LODES and INE sources. Raw-source versions and local paths
should be recorded when reproducing an experiment.

## Citation

If you use this repository in academic work, cite the associated paper
once its final bibliographic information is available.

## License

Add the repository license here when finalized.
