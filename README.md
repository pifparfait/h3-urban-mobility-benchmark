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
