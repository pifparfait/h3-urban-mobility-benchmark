from __future__ import annotations

from collections import Counter
import math

import h3
import numpy as np
import pandas as pd

from .agent_profiles import PURPOSE_WORK


def jsd_counter(a: Counter, b: Counter):
    keys = set(a) | set(b)
    if not keys:
        return np.nan
    x = np.asarray([a.get(k, 0) for k in keys], dtype=float)
    y = np.asarray([b.get(k, 0) for k in keys], dtype=float)
    if x.sum() == 0 or y.sum() == 0:
        return np.nan
    p, q = x / x.sum(), y / y.sum()
    m = 0.5 * (p + q)
    def kl(u, v):
        mask = u > 0
        return float(np.sum(u[mask] * np.log2(u[mask] / v[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def paired_counter_metrics(a: Counter, b: Counter):
    keys = sorted(set(a) | set(b))
    if not keys:
        return {"pearson": np.nan, "nmae": np.nan, "jsd": np.nan}
    x = np.asarray([a.get(k, 0) for k in keys], dtype=float)
    y = np.asarray([b.get(k, 0) for k in keys], dtype=float)
    pearson = (
        float(np.corrcoef(x, y)[0, 1])
        if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0
        else np.nan
    )
    nmae = float(np.mean(np.abs(y - x)) / max(np.mean(np.abs(x)), 1e-12))
    return {"pearson": pearson, "nmae": nmae, "jsd": jsd_counter(a, b)}


def haversine_scalar(lat1, lon1, lat2, lon2):
    r = 6_371_008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(min(1.0, math.sqrt(a)))


class UrbanFidelity:
    """
    Paired urban-output validation.

    IMPORTANT:
      - H7/H8/H9 are ANALYSIS scales.
      - H11/H12 never appear here.
      - OSM routing is not discretized and is not a fidelity dimension here.
    """
    def __init__(self, world, resolutions=(7, 8, 9)):
        self.world = world
        self.resolutions = tuple(resolutions)
        self.od_c = {r: Counter() for r in self.resolutions}
        self.od_h = {r: Counter() for r in self.resolutions}

        # Purpose-specific commuting OD used only for external validation
        # against observed home-to-work mobility sources such as LODES.
        # The canonical all-purpose OD counters above remain unchanged.
        self.work_od_c = {r: Counter() for r in self.resolutions}
        self.work_od_h = {r: Counter() for r in self.resolutions}

        self.occ_c = {r: Counter() for r in self.resolutions}
        self.occ_h = {r: Counter() for r in self.resolutions}

        # Cumulative destination/visit density at the genuine urban-analysis
        # scales. These are aggregations of real chosen coordinates; they do
        # NOT discretize physical routing.
        self.visit_c = {r: Counter() for r in self.resolutions}
        self.visit_h = {r: Counter() for r in self.resolutions}
        self.dest_exact = 0
        self.dest_valid = 0

        # Destination-substitution fidelity:
        # quantify whether H3 selects a different real POI that remains
        # spatially close to the Continuous-indexed destination.
        self.dest_100 = 0
        self.dest_250 = 0
        self.dest_500 = 0
        self.dest_1000 = 0
        self.destination_separations_m = []

        self.n = 0
        self.access_c = []
        self.access_h = []
        self.trip_c = []
        self.trip_h = []
        self.usage_c = Counter()
        self.usage_h = Counter()

        # Exact paired decision coordinates used ONLY for post-hoc physical
        # route visualization. These preserve the real continuous origin and
        # destination coordinates selected by each architecture.
        self.trip_records = []

    def update(
        self,
        old_c,
        old_h,
        new_c,
        new_h,
        dc,
        dh,
        ac,
        ah,
        tc,
        th,
        agent_id=None,
        step=None,
        purpose_id=None,
    ):
        self.n += 1
        self.dest_exact += int(dc == dh)
        if dc >= 0 and dh >= 0:
            self.dest_valid += 1
            sep = haversine_scalar(
                float(self.world.amenity_lat[dc]), float(self.world.amenity_lng[dc]),
                float(self.world.amenity_lat[dh]), float(self.world.amenity_lng[dh]),
            )
            self.destination_separations_m.append(float(sep))
            self.dest_100 += int(sep <= 100.0)
            self.dest_250 += int(sep <= 250.0)
            self.dest_500 += int(sep <= 500.0)
            self.dest_1000 += int(sep <= 1000.0)

            self.usage_c[str(self.world.amenity_layer_code[dc])] += 1
            self.usage_h[str(self.world.amenity_layer_code[dh])] += 1

        self.access_c.append(float(ac))
        self.access_h.append(float(ah))
        if np.isfinite(tc) and np.isfinite(th):
            self.trip_c.append(float(tc))
            self.trip_h.append(float(th))

        # Preserve exact paired endpoints for routed-OD visualization.
        # No H3 centroid is used as a physical origin or destination here.
        record = {
            "agent_id": int(agent_id) if agent_id is not None else -1,
            "step": int(step) if step is not None else -1,
            "purpose_id": int(purpose_id) if purpose_id is not None else -1,
            "continuous_origin_lat": float(old_c[0]),
            "continuous_origin_lng": float(old_c[1]),
            "continuous_destination_lat": float(new_c[0]),
            "continuous_destination_lng": float(new_c[1]),
            "h3_origin_lat": float(old_h[0]),
            "h3_origin_lng": float(old_h[1]),
            "h3_destination_lat": float(new_h[0]),
            "h3_destination_lng": float(new_h[1]),
            "continuous_destination_id": int(dc),
            "h3_destination_id": int(dh),
        }

        for r in self.resolutions:
            record[f"continuous_origin_h{r}"] = h3.latlng_to_cell(
                float(old_c[0]), float(old_c[1]), r
            )
            record[f"continuous_destination_h{r}"] = h3.latlng_to_cell(
                float(new_c[0]), float(new_c[1]), r
            )
            record[f"h3_origin_h{r}"] = h3.latlng_to_cell(
                float(old_h[0]), float(old_h[1]), r
            )
            record[f"h3_destination_h{r}"] = h3.latlng_to_cell(
                float(new_h[0]), float(new_h[1]), r
            )

        self.trip_records.append(record)

        for r in self.resolutions:
            oc = h3.latlng_to_cell(float(old_c[0]), float(old_c[1]), r)
            oh = h3.latlng_to_cell(float(old_h[0]), float(old_h[1]), r)
            nc = h3.latlng_to_cell(float(new_c[0]), float(new_c[1]), r)
            nh = h3.latlng_to_cell(float(new_h[0]), float(new_h[1]), r)
            self.od_c[r][(oc, nc)] += 1
            self.od_h[r][(oh, nh)] += 1

            if (
                purpose_id is not None
                and int(purpose_id) == PURPOSE_WORK
            ):
                self.work_od_c[r][(oc, nc)] += 1
                self.work_od_h[r][(oh, nh)] += 1

            self.visit_c[r][nc] += 1
            self.visit_h[r][nh] += 1

    def finalize_occupancy(self, lat_c, lng_c, lat_h, lng_h):
        for r in self.resolutions:
            self.occ_c[r] = Counter(
                h3.latlng_to_cell(float(a), float(b), r)
                for a, b in zip(lat_c, lng_c)
            )
            self.occ_h[r] = Counter(
                h3.latlng_to_cell(float(a), float(b), r)
                for a, b in zip(lat_h, lng_h)
            )

    def export_diagnostics(self, output_dir):
        """
        Export the minimal post-hoc diagnostics needed for exact scientific
        visualization without storing the full agent-decision table.

        Exports:
          - destination separation samples for an exact ECDF;
          - final occupancy at H7/H8/H9;
          - cumulative visit density at H7/H8/H9;
          - OD flows at H7/H8/H9;
          - work-purpose OD flows at H7/H8/H9 for external validation.

        H11/H12 are deliberately absent.
        Physical OSM routing is deliberately absent.
        """
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "destination_separation_m": np.asarray(
                    self.destination_separations_m,
                    dtype=float,
                )
            }
        ).to_csv(
            output_dir / "destination_separations.csv",
            index=False,
        )

        # Exact paired endpoints. This file is intentionally diagnostic:
        # it is used to reconstruct representative physical routes over the
        # common OSM graph without altering any benchmark or fidelity metric.
        pd.DataFrame(
            self.trip_records
        ).to_csv(
            output_dir / "paired_trip_endpoints.csv",
            index=False,
        )

        def cell_table(counter_c, counter_h):
            keys = sorted(set(counter_c) | set(counter_h))
            rows = []

            for cell in keys:
                lat, lng = h3.cell_to_latlng(cell)

                c = int(counter_c.get(cell, 0))
                hh = int(counter_h.get(cell, 0))

                rows.append(
                    {
                        "h3_id": cell,
                        "lat": float(lat),
                        "lng": float(lng),
                        "continuous_count": c,
                        "h3_count": hh,
                        "difference": hh - c,
                        "absolute_difference": abs(hh - c),
                    }
                )

            return pd.DataFrame(rows)

        def od_table(counter_c, counter_h):
            keys = sorted(set(counter_c) | set(counter_h))
            rows = []

            for origin, destination in keys:
                olat, olng = h3.cell_to_latlng(origin)
                dlat, dlng = h3.cell_to_latlng(destination)

                c = int(counter_c.get((origin, destination), 0))
                hh = int(counter_h.get((origin, destination), 0))

                rows.append(
                    {
                        "origin_h3": origin,
                        "destination_h3": destination,
                        "origin_lat": float(olat),
                        "origin_lng": float(olng),
                        "destination_lat": float(dlat),
                        "destination_lng": float(dlng),
                        "continuous_count": c,
                        "h3_count": hh,
                        "difference": hh - c,
                        "absolute_difference": abs(hh - c),
                    }
                )

            return pd.DataFrame(rows)

        for r in self.resolutions:
            cell_table(
                self.occ_c[r],
                self.occ_h[r],
            ).to_csv(
                output_dir / f"occupancy_H{r}.csv",
                index=False,
            )

            cell_table(
                self.visit_c[r],
                self.visit_h[r],
            ).to_csv(
                output_dir / f"visit_density_H{r}.csv",
                index=False,
            )

            od_table(
                self.od_c[r],
                self.od_h[r],
            ).to_csv(
                output_dir / f"od_flows_H{r}.csv",
                index=False,
            )

            od_table(
                self.work_od_c[r],
                self.work_od_h[r],
            ).to_csv(
                output_dir / f"work_od_flows_H{r}.csv",
                index=False,
            )

    def summary(self):
        ac = np.asarray(self.access_c)
        ah = np.asarray(self.access_h)
        access_pearson = (
            float(np.corrcoef(ac, ah)[0, 1])
            if len(ac) > 1 and np.std(ac) > 0 and np.std(ah) > 0 else np.nan
        )
        access_nmae = float(
            np.mean(np.abs(ah-ac)) / max(np.mean(np.abs(ac)), 1e-12)
        )
        tc = np.asarray(self.trip_c)
        th = np.asarray(self.trip_h)
        trip_mae = float(np.mean(np.abs(th-tc))) if len(tc) else np.nan

        destination_sep = np.asarray(
            self.destination_separations_m,
            dtype=float,
        )
        destination_median_m = (
            float(np.median(destination_sep))
            if destination_sep.size
            else np.nan
        )
        destination_p90_m = (
            float(np.percentile(destination_sep, 90))
            if destination_sep.size
            else np.nan
        )

        rows = [{
            "metric": "accessibility",
            "scale": "decision",
            "pearson": access_pearson,
            "nmae": access_nmae,
            "jsd": np.nan,
            "primary_value": access_pearson,
        }, {
            "metric": "destination",
            "scale": "decision",
            "pearson": np.nan,
            "nmae": np.nan,
            "jsd": np.nan,
            "primary_value": self.dest_exact / max(self.n, 1),
            "within_100m": self.dest_100 / max(self.dest_valid, 1),
            "within_250m": self.dest_250 / max(self.dest_valid, 1),
            "within_500m": self.dest_500 / max(self.dest_valid, 1),
            "within_1000m": self.dest_1000 / max(self.dest_valid, 1),
            "median_separation_m": destination_median_m,
            "p90_separation_m": destination_p90_m,
        }, {
            "metric": "trip_distance",
            "scale": "decision",
            "pearson": (
                float(np.corrcoef(tc, th)[0,1])
                if len(tc)>1 and np.std(tc)>0 and np.std(th)>0 else np.nan
            ),
            "nmae": (
                float(np.mean(np.abs(th-tc))/max(np.mean(np.abs(tc)),1e-12))
                if len(tc) else np.nan
            ),
            "jsd": np.nan,
            "primary_value": trip_mae,
        }]

        for r in self.resolutions:
            od = paired_counter_metrics(self.od_c[r], self.od_h[r])
            occ = paired_counter_metrics(self.occ_c[r], self.occ_h[r])
            rows.append({
                "metric": "od_flow", "scale": f"H{r}",
                **od, "primary_value": od["pearson"],
            })

            work_od = paired_counter_metrics(
                self.work_od_c[r],
                self.work_od_h[r],
            )
            rows.append({
                "metric": "work_od_flow",
                "scale": f"H{r}",
                **work_od,
                "primary_value": work_od["pearson"],
            })

            rows.append({
                "metric": "occupancy", "scale": f"H{r}",
                **occ, "primary_value": occ["pearson"],
            })

        usage = paired_counter_metrics(self.usage_c, self.usage_h)
        rows.append({
            "metric": "amenity_usage", "scale": "city",
            **usage, "primary_value": usage["pearson"],
        })
        return pd.DataFrame(rows)
