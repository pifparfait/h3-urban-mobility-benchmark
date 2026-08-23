# Routed OD visualization update

Replace exactly these three files:

- `src/h3uab/fidelity.py`
- `experiments/03_urban_fidelity.py`
- `analysis/04_final_madrid_visual_analysis.py`

Then run:

```bash
python experiments/03_urban_fidelity.py --agents 10000
python analysis/04_final_madrid_visual_analysis.py
```

The first command only repeats the 10K urban-fidelity run. It does NOT repeat the 10K→1M architecture sweep.

New diagnostic:
`outputs/urban_fidelity/diagnostics_n10000/paired_trip_endpoints.csv`

New figures:
- `FIG_ROUTED_OD_H7_city.png`
- `FIG_ROUTED_OD_H8_zoom.png`

The routed figures use exact representative origin/destination coordinates from the paired simulation and route them over the common real OSM walk graph. H3 is not used to discretize physical routing.
# Final scientific analysis package

This package is designed for the frozen experiment:

1. **Architectural efficiency**
   - urban-state lookup
   - urban-context retrieval
   - accessibility query
   - candidate retrieval
   - full agent decision

2. **Urban fidelity**
   - accessibility
   - destination spatial equivalence
   - OD H7/H8/H9
   - final occupancy H7/H8/H9
   - trip-distance fidelity
   - amenity usage

3. **Spatial visualization**
   - Continuous vs H3 final occupancy
   - Continuous vs H3 cumulative visit density
   - H3 - Continuous difference maps

## Architectural rules

- H11/H12 are localization / visualization only and never enter the fidelity claim.
- Physical routing is not discretized. Real OSM mobility is common infrastructure.
- The spatial heatmaps aggregate real agent/destination coordinates at H7/H8/H9 for analysis.

## One fast diagnostic rerun

To obtain the exact destination ECDF and clean heatmap source data, replace:

```text
src/h3uab/fidelity.py
experiments/03_urban_fidelity.py
```

with the supplied versions and run:

```bash
python experiments/03_urban_fidelity.py --agents 10000
```

This does NOT repeat the architecture sweep and does NOT rebuild routing or city data.

It creates:

```text
outputs/urban_fidelity/diagnostics_n10000/
    destination_separations.csv
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

## Generate all figures

```bash
python analysis/run_all_analysis.py
```

The Continuous and H3 heatmaps deliberately use the same color scale.
Difference maps are zero-centered and symmetric.

## Main paper figures

Recommended core set:

1. `architecture_speedup_vs_population.png`
2. `full_decision_latency_vs_population.png`
3. `full_decision_throughput_vs_population.png`
4. `urban_fidelity_pearson_H7_H9.png`
5. `destination_separation_ecdf.png`
6. `visit_density_H8_continuous_count.png`
7. `visit_density_H8_h3_count.png`
8. `visit_density_H8_difference.png`

H7 gives the city-scale result, H8 the neighborhood-scale result, and H9 exposes local allocation divergence.
