from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
import heapq
import math
import pickle
import time

import h3
import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import LineString

from h3uab.geometry import haversine_m
from h3uab.routing_matrix import RoutingMatrixOracle
from profiler import StepTiming


# ============================================================================
# Static H3 road index
# ============================================================================

def _edge_geometry(G, u, v, key):
    data = G.edges[u, v, key]
    geom = data.get("geometry")

    if geom is not None:
        return geom

    return LineString(
        [
            (
                float(G.nodes[u]["x"]),
                float(G.nodes[u]["y"]),
            ),
            (
                float(G.nodes[v]["x"]),
                float(G.nodes[v]["y"]),
            ),
        ]
    )


def _edge_length_m(G, u, v, key):
    try:
        return float(
            G.edges[u, v, key].get("length", 0.0)
        )
    except Exception:
        return 0.0


def _choose_parallel_edge_key(G, u, v, weight="length"):
    data = G.get_edge_data(u, v)

    if not data:
        raise KeyError(f"Missing edge {u}->{v}")

    best_key = None
    best_value = math.inf

    for key, attrs in data.items():
        try:
            value = float(
                attrs.get(
                    weight,
                    attrs.get("length", 1.0),
                )
            )
        except Exception:
            value = 1.0

        if value < best_value:
            best_value = value
            best_key = key

    return best_key


def _sample_linestring_latlng(geometry, spacing_m):
    coords = list(geometry.coords)

    if not coords:
        return []

    samples = [
        (
            float(coords[0][1]),
            float(coords[0][0]),
        )
    ]

    for (x1, y1), (x2, y2) in zip(
        coords[:-1],
        coords[1:],
    ):
        seg_m = float(
            haversine_m(
                y1,
                x1,
                y2,
                x2,
            )
        )

        n = max(
            1,
            int(math.ceil(seg_m / max(spacing_m, 1.0))),
        )

        for j in range(1, n + 1):
            t = j / n

            samples.append(
                (
                    float(y1 + t * (y2 - y1)),
                    float(x1 + t * (x2 - x1)),
                )
            )

    return samples


@dataclass
class RoadH3Index:
    interaction_res: int
    position_res: int

    nodes_by_h12: dict
    node_h12: dict

    edges_by_h11: dict
    edge_h11_cells: dict

    n_nodes: int
    n_edges: int
    build_time_s: float


def build_road_h3_index(G, cfg):
    t0 = time.perf_counter()

    nodes_by_h12_mut = defaultdict(list)
    node_h12 = {}

    for node, data in G.nodes(data=True):
        lat = float(data["y"])
        lng = float(data["x"])

        cell = h3.latlng_to_cell(
            lat,
            lng,
            cfg.h3.position_res,
        )

        node_h12[node] = cell
        nodes_by_h12_mut[cell].append(node)

    nodes_by_h12 = {
        cell: tuple(nodes)
        for cell, nodes in nodes_by_h12_mut.items()
    }

    edges_by_h11_mut = defaultdict(list)
    edge_h11_cells = {}

    for u, v, key, data in G.edges(
        keys=True,
        data=True,
    ):
        edge = (u, v, key)

        geom = _edge_geometry(
            G,
            u,
            v,
            key,
        )

        cells = []
        last = None

        for lat, lng in _sample_linestring_latlng(
            geom,
            cfg.routing.edge_h11_sample_spacing_m,
        ):
            cell = h3.latlng_to_cell(
                lat,
                lng,
                cfg.h3.interaction_res,
            )

            if cell != last:
                cells.append(cell)
                last = cell

        unique_cells = tuple(
            dict.fromkeys(cells)
        )

        edge_h11_cells[edge] = unique_cells

        for cell in unique_cells:
            edges_by_h11_mut[cell].append(edge)

    edges_by_h11 = {
        cell: tuple(edges)
        for cell, edges in edges_by_h11_mut.items()
    }

    return RoadH3Index(
        interaction_res=cfg.h3.interaction_res,
        position_res=cfg.h3.position_res,
        nodes_by_h12=nodes_by_h12,
        node_h12=node_h12,
        edges_by_h11=edges_by_h11,
        edge_h11_cells=edge_h11_cells,
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        build_time_s=time.perf_counter() - t0,
    )


def save_road_h3_index(index, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        pickle.dump(
            index,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_road_h3_index(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


# ============================================================================
# Graph preparation
# ============================================================================

def _largest_component(G):
    if G.is_directed():
        components = nx.weakly_connected_components(G)
    else:
        components = nx.connected_components(G)

    largest = max(
        components,
        key=len,
    )

    return G.subgraph(largest).copy()


def download_madrid_graph(cfg):
    west = cfg.lng_min
    south = cfg.lat_min
    east = cfg.lng_max
    north = cfg.lat_max

    bbox = (
        west,
        south,
        east,
        north,
    )

    try:
        G = ox.graph_from_bbox(
            bbox,
            network_type=cfg.routing.network_type,
            simplify=True,
            retain_all=True,
        )
    except TypeError:
        G = ox.graph_from_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            network_type=cfg.routing.network_type,
            simplify=True,
            retain_all=True,
        )

    return _largest_component(G)


def prepare_graph_files(cfg, force=False):
    graph_path = Path(cfg.routing.graphml_path)
    index_path = Path(cfg.routing.h3_index_path)

    graph_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        graph_path.exists()
        and index_path.exists()
        and not force
    ):
        print("[ROAD] Existing graph and H3 index found.")
        return

    print("[ROAD] Downloading real OSM graph...")

    G = download_madrid_graph(cfg)

    print(
        f"[ROAD] Nodes={G.number_of_nodes():,}, "
        f"Edges={G.number_of_edges():,}"
    )

    ox.save_graphml(
        G,
        graph_path,
    )

    index = build_road_h3_index(
        G,
        cfg,
    )

    save_road_h3_index(
        index,
        index_path,
    )

    print(
        f"[ROAD] H3@11 road cells={len(index.edges_by_h11):,}"
    )
    print(
        f"[ROAD] H3@12 road-node cells={len(index.nodes_by_h12):,}"
    )


_GRAPH_CACHE = {}


def load_graph_index_oracle(cfg):
    key = (
        str(Path(cfg.routing.graphml_path).resolve()),
        str(Path(cfg.routing.h3_index_path).resolve()),
        str(Path(cfg.routing.routing_matrix_dir).resolve()),
    )

    cached = _GRAPH_CACHE.get(key)

    if cached is not None:
        return cached

    t0 = time.perf_counter()

    G = ox.load_graphml(
        cfg.routing.graphml_path
    )

    index = load_road_h3_index(
        cfg.routing.h3_index_path
    )

    oracle = RoutingMatrixOracle.load(
        cfg
    )

    elapsed = time.perf_counter() - t0

    result = (
        G,
        index,
        oracle,
        elapsed,
    )

    _GRAPH_CACHE[key] = result

    return result


# ============================================================================
# Online routing — V9.3 ZERO ONLINE SNAPPING
# ============================================================================

@dataclass(frozen=True)
class RouteTemplate:
    routable: bool
    distance_m: float
    travel_time_s: float
    route_node_count: int
    route_edge_count: int
    edges: tuple
    h11_cells: tuple


def _actual_node_map(G):
    return {
        str(node): node
        for node in G.nodes
    }


def build_poi_osm_node_array(G, oracle):
    """
    POI -> already-precomputed OSM destination node.
    No nearest-node search.
    """
    actual = _actual_node_map(G)

    poi_node_strings = (
        oracle.destination_node_strings_for_pois()
    )

    out = np.empty(
        len(poi_node_strings),
        dtype=object,
    )

    for i, node_string in enumerate(poi_node_strings):
        key = str(node_string)
        node = actual.get(key)

        if node is None:
            raise KeyError(
                f"Routing-matrix node {key} is not in the loaded OSM graph."
            )

        out[i] = node

    return out


def build_initial_agent_osm_nodes(G, world, cfg):
    """
    Setup-only initial snapping.

    Deduplicate initial H3@12 cells and perform ONE vectorized nearest_nodes
    query. Online simulation then uses only precomputed node arrays.
    """
    t0 = time.perf_counter()

    n = len(world.agent_lat)

    agent_h12 = np.empty(
        n,
        dtype=object,
    )

    for i in range(n):
        agent_h12[i] = h3.latlng_to_cell(
            float(world.agent_lat[i]),
            float(world.agent_lng[i]),
            cfg.h3.position_res,
        )

    unique_h12, inverse = np.unique(
        agent_h12,
        return_inverse=True,
    )

    centroids = [
        h3.cell_to_latlng(cell)
        for cell in unique_h12
    ]

    unique_lat = np.asarray(
        [x[0] for x in centroids],
        dtype=float,
    )

    unique_lng = np.asarray(
        [x[1] for x in centroids],
        dtype=float,
    )

    # ONE batch call in SETUP, never in the online loop.
    unique_nodes = np.asarray(
        ox.nearest_nodes(
            G,
            X=unique_lng,
            Y=unique_lat,
        ),
        dtype=object,
    )

    agent_nodes = unique_nodes[
        inverse
    ].astype(
        object,
        copy=False,
    )

    return (
        agent_nodes,
        int(len(unique_h12)),
        time.perf_counter() - t0,
    )


class RoadRouter:
    """
    V9.4 router.

    - Zero online nearest-node searches.
    - O(1) distance/time via square routing matrix.
    - POI->POI route geometry uses next-hop when available.
    - If next-hop cannot express an exact physical intermediate node,
      fall back to the exact multi-target graph search and cache the result.
    """

    def __init__(
        self,
        G,
        index,
        oracle,
        cfg,
    ):
        self.G = G
        self.index = index
        self.oracle = oracle
        self.cfg = cfg

        self.actual_node_map = {
            str(node):
                node
            for node
            in G.nodes
        }

        self.route_cache = OrderedDict()

        self.edge_flows = Counter()
        self.h11_route_flows = Counter()

        self.cache_hits = 0
        self.cache_misses = 0

        self.next_hop_routes = 0
        self.next_hop_fallbacks = 0

        self.routable_trips = 0
        self.unroutable_trips = 0

        self.route_time_s = 0.0
        self.new_route_compute_time_s = 0.0
        self.cached_route_apply_time_s = 0.0

        self.online_nearest_node_calls = 0

    def _cache_get(
        self,
        pair,
    ):
        item = self.route_cache.get(
            pair
        )

        if item is not None:
            self.route_cache.move_to_end(
                pair
            )

            self.cache_hits += 1

            return item

        self.cache_misses += 1

        return None

    def _cache_put(
        self,
        pair,
        template,
    ):
        self.route_cache[
            pair
        ] = template

        self.route_cache.move_to_end(
            pair
        )

        limit = int(
            self.cfg.routing.route_cache_max_entries
        )

        if limit > 0:
            while (
                len(
                    self.route_cache
                )
                > limit
            ):
                self.route_cache.popitem(
                    last=False
                )

    def _multi_target_paths(
        self,
        source,
        targets,
    ):
        target_set = set(
            targets
        )

        if not target_set:
            return {}

        dist = {
            source:
                0.0
        }

        parent = {
            source:
                None
        }

        heap = [
            (
                0.0,
                source,
            )
        ]

        settled = set()
        found = {}

        while (
            heap
            and target_set
        ):
            d, u = (
                heapq.heappop(
                    heap
                )
            )

            if u in settled:
                continue

            settled.add(
                u
            )

            if u in target_set:
                path = []
                cur = u

                while cur is not None:
                    path.append(
                        cur
                    )

                    cur = parent[
                        cur
                    ]

                path.reverse()

                found[
                    u
                ] = path

                target_set.remove(
                    u
                )

                if not target_set:
                    break

            for v, keyed in (
                self.G.adj[
                    u
                ].items()
            ):
                best_weight = math.inf

                for key, attrs in (
                    keyed.items()
                ):
                    try:
                        w = float(
                            attrs.get(
                                self.cfg.routing.route_weight,
                                attrs.get(
                                    "length",
                                    1.0,
                                ),
                            )
                        )
                    except Exception:
                        w = 1.0

                    if w < best_weight:
                        best_weight = w

                nd = (
                    d
                    + best_weight
                )

                if nd < dist.get(
                    v,
                    math.inf,
                ):
                    dist[
                        v
                    ] = nd

                    parent[
                        v
                    ] = u

                    heapq.heappush(
                        heap,
                        (
                            nd,
                            v,
                        ),
                    )

        return found

    def _template_from_node_path(
        self,
        origin_node,
        destination_node,
        nodes,
    ):
        if not nodes:
            return RouteTemplate(
                routable=False,
                distance_m=np.nan,
                travel_time_s=np.nan,
                route_node_count=0,
                route_edge_count=0,
                edges=(),
                h11_cells=(),
            )

        edges = []
        h11_cells = []

        last_cell = None
        exact_length = 0.0

        for u, v in zip(
            nodes[:-1],
            nodes[1:],
        ):
            key = (
                _choose_parallel_edge_key(
                    self.G,
                    u,
                    v,
                    self.cfg.routing.route_weight,
                )
            )

            edge = (
                u,
                v,
                key,
            )

            edges.append(
                edge
            )

            exact_length += (
                _edge_length_m(
                    self.G,
                    u,
                    v,
                    key,
                )
            )

            for cell in (
                self.index.edge_h11_cells.get(
                    edge,
                    (),
                )
            ):
                if cell != last_cell:
                    h11_cells.append(
                        cell
                    )

                    last_cell = cell

        matrix_distance = (
            self.oracle.distance_by_nodes(
                origin_node,
                destination_node,
            )
        )

        distance_m = (
            matrix_distance
            if np.isfinite(
                matrix_distance
            )
            else exact_length
        )

        travel_time_s = (
            distance_m
            / max(
                self.cfg.routing.walking_speed_m_s,
                1e-9,
            )
        )

        return RouteTemplate(
            routable=True,
            distance_m=float(
                distance_m
            ),
            travel_time_s=float(
                travel_time_s
            ),
            route_node_count=len(
                nodes
            ),
            route_edge_count=len(
                edges
            ),
            edges=tuple(
                edges
            ),
            h11_cells=tuple(
                h11_cells
            ),
        )

    def _route_template_from_next_hop(
        self,
        origin_node,
        destination_node,
    ):
        """
        Attempt POI-universe next-hop reconstruction.

        Because the physical immediate road node may not be one of the K POI
        destination nodes, the next-hop matrix can be unavailable for some OD
        pairs. Those pairs are handled by the exact fallback.
        """
        info = (
            self.oracle.next_hop_indices(
                origin_node,
                destination_node,
            )
        )

        if info is None:
            return None

        (
            origin_idx,
            destination_idx,
            next_idx,
        ) = info

        if (
            origin_idx
            == destination_idx
        ):
            return RouteTemplate(
                routable=True,
                distance_m=0.0,
                travel_time_s=0.0,
                route_node_count=1,
                route_edge_count=0,
                edges=(),
                h11_cells=(),
            )

        # This compressed next-hop representation only works if every hop
        # remains inside the K-node universe. Follow with a strict guard.
        idx_path = [
            origin_idx
        ]

        current = origin_idx

        guard = 0

        while (
            current
            != destination_idx
        ):
            nxt = int(
                self.oracle.next_hop_matrix[
                    current,
                    destination_idx
                ]
            )

            if nxt < 0:
                return None

            if nxt == current:
                return None

            idx_path.append(
                nxt
            )

            current = nxt

            guard += 1

            if guard > len(
                self.oracle.node_strings
            ):
                return None

        node_path = []

        for idx in idx_path:
            node_string = str(
                self.oracle.node_strings[
                    idx
                ]
            )

            node = self.actual_node_map.get(
                node_string
            )

            if node is None:
                return None

            node_path.append(
                node
            )

        # Consecutive K-universe nodes must be directly connected physically;
        # otherwise this compressed next-hop cannot represent the exact road
        # sequence and we fall back.
        for u, v in zip(
            node_path[:-1],
            node_path[1:],
        ):
            if not self.G.has_edge(
                u,
                v,
            ):
                return None

        return (
            self._template_from_node_path(
                origin_node,
                destination_node,
                node_path,
            )
        )

    def route_batch_nodes(
        self,
        origin_nodes,
        destination_nodes,
        progress_label="routing",
        step_timing=None,
    ):
        if step_timing is None:
            from profiler import StepTiming

            step_timing = StepTiming(
                step=-1
            )

        t_batch = (
            time.perf_counter()
        )

        origin_nodes = np.asarray(
            origin_nodes,
            dtype=object,
        )

        destination_nodes = np.asarray(
            destination_nodes,
            dtype=object,
        )

        if len(
            origin_nodes
        ) != len(
            destination_nodes
        ):
            raise ValueError(
                "origin_nodes and destination_nodes "
                "must have equal length."
            )

        n = len(
            origin_nodes
        )

        step_timing.od_requests = int(
            n
        )

        t_nodes = (
            time.perf_counter()
        )

        pairs = list(
            zip(
                origin_nodes.tolist(),
                destination_nodes.tolist(),
            )
        )

        multiplicity = Counter(
            pairs
        )

        node_elapsed = (
            time.perf_counter()
            - t_nodes
        )

        step_timing.node_lookup_s += (
            node_elapsed
        )

        step_timing.online_nearest_node_calls = (
            0
        )

        print(
            f"    [{progress_label}] "
            f"OSM node addressing: "
            f"requests={2*n:,} | "
            f"online nearest_nodes=0 | "
            f"time={node_elapsed:.4f}s"
        )

        t_matrix = (
            time.perf_counter()
        )

        step_timing.unique_od_pairs = (
            len(
                multiplicity
            )
        )

        matrix_hits = 0
        matrix_misses = 0

        for (
            origin,
            destination,
        ) in multiplicity.keys():
            d = (
                self.oracle.distance_by_nodes(
                    origin,
                    destination,
                )
            )

            if np.isfinite(
                d
            ):
                matrix_hits += 1
            else:
                matrix_misses += 1

        matrix_elapsed = (
            time.perf_counter()
            - t_matrix
        )

        step_timing.matrix_lookup_s += (
            matrix_elapsed
        )

        step_timing.matrix_hits = (
            matrix_hits
        )

        step_timing.matrix_misses = (
            matrix_misses
        )

        print(
            f"    [{progress_label}] "
            f"routing matrix: "
            f"unique OD={len(multiplicity):,} | "
            f"hits={matrix_hits:,} | "
            f"misses={matrix_misses:,} | "
            f"time={matrix_elapsed:.4f}s"
        )

        templates = {}
        missing_by_origin = defaultdict(
            set
        )

        route_hit0 = (
            self.cache_hits
        )

        route_miss0 = (
            self.cache_misses
        )

        next_hop_before = (
            self.next_hop_routes
        )

        next_fallback_before = (
            self.next_hop_fallbacks
        )

        for pair in multiplicity:
            template = (
                self._cache_get(
                    pair
                )
            )

            if template is not None:
                templates[
                    pair
                ] = template

                continue

            (
                origin,
                destination,
            ) = pair

            template = None

            if getattr(
                self.cfg.routing,
                "use_compressed_next_hop",
                False,
            ):
                template = (
                    self._route_template_from_next_hop(
                        origin,
                        destination,
                    )
                )

            if template is not None:
                self.next_hop_routes += 1

                templates[
                    pair
                ] = template

                self._cache_put(
                    pair,
                    template,
                )

                continue

            self.next_hop_fallbacks += 1

            missing_by_origin[
                origin
            ].add(
                destination
            )

        step_timing.route_cache_hits = (
            self.cache_hits
            - route_hit0
        )

        step_timing.route_cache_misses = (
            self.cache_misses
            - route_miss0
        )

        step_timing.new_routes = sum(
            len(
                targets
            )
            for targets
            in missing_by_origin.values()
        )

        step_timing.unique_origins_searched = (
            len(
                missing_by_origin
            )
        )

        next_hop_used = (
            self.next_hop_routes
            - next_hop_before
        )

        next_hop_fallbacks = (
            self.next_hop_fallbacks
            - next_fallback_before
        )

        print(
            f"    [{progress_label}] "
            f"route cache: "
            f"hits={step_timing.route_cache_hits:,} | "
            f"misses={step_timing.route_cache_misses:,}"
        )

        print(
            f"    [{progress_label}] "
            f"next-hop routes={next_hop_used:,} | "
            f"fallbacks={next_hop_fallbacks:,} | "
            f"graph-search origins="
            f"{step_timing.unique_origins_searched:,}"
        )

        t_search = (
            time.perf_counter()
        )

        total_origins = len(
            missing_by_origin
        )

        progress_every = max(
            25,
            min(
                250,
                total_origins // 20
                if total_origins
                else 25,
            ),
        )

        for (
            origin_i,
            (
                origin,
                targets,
            ),
        ) in enumerate(
            missing_by_origin.items(),
            start=1,
        ):
            found = (
                self._multi_target_paths(
                    origin,
                    targets,
                )
            )

            step_timing.dijkstra_calls += (
                1
            )

            rec_start = (
                time.perf_counter()
            )

            for destination in targets:
                nodes = (
                    found.get(
                        destination
                    )
                )

                template = (
                    self._template_from_node_path(
                        origin,
                        destination,
                        nodes,
                    )
                )

                pair = (
                    origin,
                    destination,
                )

                templates[
                    pair
                ] = template

                self._cache_put(
                    pair,
                    template,
                )

            step_timing.route_reconstruction_s += (
                time.perf_counter()
                - rec_start
            )

            if (
                origin_i
                % progress_every
                == 0
                or origin_i
                == total_origins
            ):
                elapsed = (
                    time.perf_counter()
                    - t_search
                )

                rate = (
                    origin_i
                    / max(
                        elapsed,
                        1e-9,
                    )
                )

                remaining = (
                    total_origins
                    - origin_i
                ) / max(
                    rate,
                    1e-9,
                )

                print(
                    f"    [{progress_label}] "
                    f"graph search "
                    f"{origin_i:,}/"
                    f"{total_origins:,} origins | "
                    f"elapsed={elapsed:.1f}s | "
                    f"ETA={remaining:.1f}s"
                )

        search_elapsed = (
            time.perf_counter()
            - t_search
        )

        self.new_route_compute_time_s += (
            search_elapsed
        )

        step_timing.route_search_s += (
            search_elapsed
        )

        t_apply = (
            time.perf_counter()
        )

        weighted_edges = 0
        weighted_h11 = 0

        for (
            pair,
            count,
        ) in multiplicity.items():
            template = (
                templates[
                    pair
                ]
            )

            if template.routable:
                self.routable_trips += (
                    count
                )

                weighted_edges += (
                    len(
                        template.edges
                    )
                    * count
                )

                weighted_h11 += (
                    len(
                        template.h11_cells
                    )
                    * count
                )

                for edge in (
                    template.edges
                ):
                    self.edge_flows[
                        edge
                    ] += count

                for cell in (
                    template.h11_cells
                ):
                    self.h11_route_flows[
                        cell
                    ] += count
            else:
                self.unroutable_trips += (
                    count
                )

        apply_elapsed = (
            time.perf_counter()
            - t_apply
        )

        self.cached_route_apply_time_s += (
            apply_elapsed
        )

        step_timing.route_apply_s += (
            apply_elapsed
        )

        step_timing.traversed_osm_edges_weighted = int(
            weighted_edges
        )

        step_timing.traversed_h11_cells_weighted = int(
            weighted_h11
        )

        network_distance = np.full(
            n,
            np.nan,
            dtype=float,
        )

        travel_time = np.full(
            n,
            np.nan,
            dtype=float,
        )

        route_edge_count = np.zeros(
            n,
            dtype=np.int32,
        )

        route_node_count = np.zeros(
            n,
            dtype=np.int32,
        )

        routable = np.zeros(
            n,
            dtype=bool,
        )

        for i, pair in enumerate(
            pairs
        ):
            template = (
                templates[
                    pair
                ]
            )

            if template.routable:
                network_distance[
                    i
                ] = template.distance_m

                travel_time[
                    i
                ] = template.travel_time_s

                route_edge_count[
                    i
                ] = template.route_edge_count

                route_node_count[
                    i
                ] = template.route_node_count

                routable[
                    i
                ] = True

        batch_elapsed = (
            time.perf_counter()
            - t_batch
        )

        self.route_time_s += (
            batch_elapsed
        )

        print(
            f"    [{progress_label}] "
            f"routing total="
            f"{batch_elapsed:.3f}s"
        )

        return {
            "network_distance_m":
                network_distance,

            "travel_time_s":
                travel_time,

            "route_edge_count":
                route_edge_count,

            "route_node_count":
                route_node_count,

            "routable":
                routable,

            "step_timing":
                step_timing,

            "next_hop_routes":
                next_hop_used,

            "next_hop_fallbacks":
                next_hop_fallbacks,
        }
