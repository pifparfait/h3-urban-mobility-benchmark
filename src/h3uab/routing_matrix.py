from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import time

import numpy as np
import osmnx as ox
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

from h3uab.osm_data import load_osm_pois


def _paths(cfg):
    root = Path(
        cfg.routing.routing_matrix_dir
    )

    return {
        "root":
            root,

        "distance":
            root
            / cfg.routing.routing_matrix_distance_file,

        "nodes":
            root
            / cfg.routing.routing_matrix_nodes_file,

        "poi_index":
            root
            / cfg.routing.routing_matrix_poi_index_file,

        "metadata":
            root
            / cfg.routing.routing_matrix_metadata_file,

        "next_hop":
            root
            / cfg.routing.routing_matrix_next_hop_file,
    }


def _node_string_array(nodes):
    values = [
        str(
            x
        )
        for x in nodes
    ]

    width = max(
        1,
        max(
            len(
                x
            )
            for x
            in values
        ),
    )

    return np.asarray(
        values,
        dtype=f"<U{width}",
    )


def _actual_node_map(G):
    return {
        str(
            node
        ):
            node
        for node
        in G.nodes
    }


def _vectorized_snap_pois(
    G,
    pois,
):
    nodes = ox.nearest_nodes(
        G,
        X=pois[
            "lng"
        ].to_numpy(
            float
        ),
        Y=pois[
            "lat"
        ].to_numpy(
            float
        ),
    )

    return np.asarray(
        nodes,
        dtype=object,
    )


def _build_min_weight_csr(
    G,
    node_list,
):
    node_to_idx = {
        node:
            i
        for i, node
        in enumerate(
            node_list
        )
    }

    best = {}

    for (
        u,
        v,
        key,
        data,
    ) in G.edges(
        keys=True,
        data=True,
    ):
        try:
            w = float(
                data.get(
                    "length",
                    1.0,
                )
            )
        except Exception:
            w = 1.0

        i = node_to_idx[
            u
        ]

        j = node_to_idx[
            v
        ]

        pair = (
            i,
            j,
        )

        old = best.get(
            pair
        )

        if (
            old is None
            or w < old
        ):
            best[
                pair
            ] = w

    rows = np.fromiter(
        (
            k[
                0
            ]
            for k
            in best.keys()
        ),
        dtype=np.int64,
        count=len(
            best
        ),
    )

    cols = np.fromiter(
        (
            k[
                1
            ]
            for k
            in best.keys()
        ),
        dtype=np.int64,
        count=len(
            best
        ),
    )

    data = np.fromiter(
        best.values(),
        dtype=np.float64,
        count=len(
            best
        ),
    )

    return csr_matrix(
        (
            data,
            (
                rows,
                cols,
            ),
        ),
        shape=(
            len(
                node_list
            ),
            len(
                node_list
            ),
        ),
        dtype=np.float64,
    )


def estimate_matrix_size_bytes(
    n,
):
    return (
        int(
            n
        )
        * int(
            n
        )
        * np.dtype(
            np.float32
        ).itemsize
    )


def estimate_next_hop_size_bytes(
    n,
):
    return (
        int(
            n
        )
        * int(
            n
        )
        * np.dtype(
            np.int32
        ).itemsize
    )


def human_bytes(
    n,
):
    n = float(
        n
    )

    for unit in (
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ):
        if n < 1024.0:
            return (
                f"{n:.2f} "
                f"{unit}"
            )

        n /= 1024.0

    return (
        f"{n:.2f} PB"
    )


def build_square_destination_routing_matrix(
    cfg,
    force=False,
):
    """
    Build/refresh the destination-node distance matrix.

    V9.4 note:
      the next-hop matrix is built separately by
      build_square_destination_next_hop_matrix().
    """
    paths = _paths(
        cfg
    )

    required = (
        paths[
            "distance"
        ],
        paths[
            "nodes"
        ],
        paths[
            "poi_index"
        ],
        paths[
            "metadata"
        ],
    )

    if (
        all(
            p.exists()
            for p
            in required
        )
        and not force
    ):
        print(
            "[MATRIX] Distance matrix already exists."
        )
        print(
            f"[MATRIX] "
            f"{paths['root']}"
        )
        return

    graph_path = Path(
        cfg.routing.graphml_path
    )

    if not graph_path.exists():
        raise FileNotFoundError(
            f"{graph_path} not found.\n"
            "Run first:\n"
            "    python prepare_osm_graph.py"
        )

    if (
        force
        and paths[
            "root"
        ].exists()
    ):
        # Keep directory but replace matrix artifacts.
        for key in (
            "distance",
            "nodes",
            "poi_index",
            "metadata",
            "next_hop",
        ):
            try:
                paths[
                    key
                ].unlink()
            except FileNotFoundError:
                pass

    paths[
        "root"
    ].mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[MATRIX] Loading real OSM graph..."
    )

    G = ox.load_graphml(
        graph_path
    )

    print(
        f"[MATRIX] Graph: "
        f"{G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges"
    )

    print(
        "[MATRIX] Loading all possible POI destinations..."
    )

    pois = load_osm_pois(
        cfg
    )

    print(
        f"[MATRIX] POIs: "
        f"{len(pois):,}"
    )

    print(
        "[MATRIX] Snapping all POIs to OSM nodes once..."
    )

    poi_nodes = (
        _vectorized_snap_pois(
            G,
            pois,
        )
    )

    unique_node_strings = sorted(
        set(
            str(
                x
            )
            for x
            in poi_nodes
        )
    )

    actual_map = _actual_node_map(
        G
    )

    destination_nodes = [
        actual_map[
            s
        ]
        for s
        in unique_node_strings
    ]

    n_dest = len(
        destination_nodes
    )

    print(
        f"[MATRIX] Unique destination OSM nodes: "
        f"{n_dest:,}"
    )

    matrix_bytes = (
        estimate_matrix_size_bytes(
            n_dest
        )
    )

    free_bytes = shutil.disk_usage(
        paths[
            "root"
        ]
    ).free

    print(
        f"[MATRIX] Square matrix shape: "
        f"{n_dest:,} x {n_dest:,}"
    )

    print(
        f"[MATRIX] float32 disk size: "
        f"{human_bytes(matrix_bytes)}"
    )

    print(
        f"[MATRIX] Free disk: "
        f"{human_bytes(free_bytes)}"
    )

    if matrix_bytes > free_bytes * 0.85:
        raise RuntimeError(
            "Routing matrix would consume more "
            "than 85% of free disk."
        )

    node_strings = (
        _node_string_array(
            destination_nodes
        )
    )

    np.save(
        paths[
            "nodes"
        ],
        node_strings,
        allow_pickle=False,
    )

    matrix_index_by_node_string = {
        str(
            node
        ):
            i
        for i, node
        in enumerate(
            destination_nodes
        )
    }

    poi_to_matrix_index = np.asarray(
        [
            matrix_index_by_node_string[
                str(
                    node
                )
            ]
            for node
            in poi_nodes
        ],
        dtype=np.int32,
    )

    np.save(
        paths[
            "poi_index"
        ],
        poi_to_matrix_index,
        allow_pickle=False,
    )

    print(
        "[MATRIX] Building sparse graph adjacency..."
    )

    graph_nodes = list(
        G.nodes
    )

    graph_index = {
        node:
            i
        for i, node
        in enumerate(
            graph_nodes
        )
    }

    adjacency = (
        _build_min_weight_csr(
            G,
            graph_nodes,
        )
    )

    destination_graph_indices = np.asarray(
        [
            graph_index[
                node
            ]
            for node
            in destination_nodes
        ],
        dtype=np.int64,
    )

    D = np.memmap(
        paths[
            "distance"
        ],
        dtype=np.float32,
        mode="w+",
        shape=(
            n_dest,
            n_dest,
        ),
    )

    batch_size = max(
        1,
        int(
            cfg.routing.routing_matrix_batch_size
        ),
    )

    t0 = time.perf_counter()

    print(
        f"[MATRIX] Running shortest paths "
        f"in batches of {batch_size} sources..."
    )

    for start in range(
        0,
        n_dest,
        batch_size,
    ):
        stop = min(
            start
            + batch_size,
            n_dest,
        )

        source_graph_indices = (
            destination_graph_indices[
                start:stop
            ]
        )

        dist_to_all_graph_nodes = (
            scipy_dijkstra(
                adjacency,
                directed=True,
                indices=(
                    source_graph_indices
                ),
                return_predecessors=False,
            )
        )

        block = (
            dist_to_all_graph_nodes[
                :,
                destination_graph_indices,
            ]
        )

        D[
            start:stop,
            :
        ] = block.astype(
            np.float32,
            copy=False,
        )

        D.flush()

        elapsed = (
            time.perf_counter()
            - t0
        )

        done = stop

        rate = (
            done
            / max(
                elapsed,
                1e-9,
            )
        )

        remaining = (
            n_dest
            - done
        ) / max(
            rate,
            1e-9,
        )

        print(
            f"[MATRIX] "
            f"{done:,}/{n_dest:,} sources "
            f"({100.0 * done / n_dest:5.1f}%) | "
            f"elapsed {elapsed/60:.1f} min | "
            f"ETA {remaining/60:.1f} min"
        )

    total_time = (
        time.perf_counter()
        - t0
    )

    for i in range(
        n_dest
    ):
        D[
            i,
            i
        ] = 0.0

    D.flush()
    del D

    metadata = {
        "matrix_type":
            "square_all_possible_poi_destination_nodes",

        "network_type":
            cfg.routing.network_type,

        "weight":
            cfg.routing.route_weight,

        "dtype":
            "float32",

        "n_pois":
            int(
                len(
                    pois
                )
            ),

        "n_unique_destination_nodes":
            int(
                n_dest
            ),

        "shape":
            [
                int(
                    n_dest
                ),
                int(
                    n_dest
                ),
            ],

        "distance_units":
            "meters",

        "travel_time_rule":
            "distance / walking_speed_m_s",

        "walking_speed_m_s":
            float(
                cfg.routing.walking_speed_m_s
            ),

        "matrix_size_bytes":
            int(
                matrix_bytes
            ),

        "build_time_s":
            float(
                total_time
            ),

        "unreachable_value":
            "inf",

        "next_hop_available":
            False,
    }

    paths[
        "metadata"
    ].write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "[MATRIX] Distance matrix done."
    )
    print(
        f"[MATRIX] Build time: "
        f"{total_time/60:.2f} min"
    )


def build_square_destination_next_hop_matrix(
    cfg,
    force=False,
):
    """
    Build a square next-hop matrix over the same K destination nodes.

    next_hop[i, j] is the NEXT DESTINATION-UNIVERSE INDEX to follow from i
    toward j when both endpoints are destination nodes.

    Important:
      - int32 memmap
      - -1 means unreachable / unavailable
      - diagonal points to itself
      - no online Dijkstra is needed for POI->POI route reconstruction
      - initial WorldPop->POI origins may still need one-time cached graph
        searches because those origins are not necessarily in the K-node
        destination universe
    """
    paths = _paths(
        cfg
    )

    if (
        paths[
            "next_hop"
        ].exists()
        and not force
    ):
        print(
            "[NEXT-HOP] Matrix already exists."
        )
        return

    if not all(
        p.exists()
        for p
        in (
            paths[
                "distance"
            ],
            paths[
                "nodes"
            ],
            paths[
                "metadata"
            ],
        )
    ):
        raise FileNotFoundError(
            "Distance matrix infrastructure is incomplete. "
            "Run prepare_routing_matrix.py first."
        )

    graph_path = Path(
        cfg.routing.graphml_path
    )

    G = ox.load_graphml(
        graph_path
    )

    node_strings = np.load(
        paths[
            "nodes"
        ],
        allow_pickle=False,
    )

    actual_map = _actual_node_map(
        G
    )

    destination_nodes = [
        actual_map[
            str(
                s
            )
        ]
        for s
        in node_strings
    ]

    n_dest = len(
        destination_nodes
    )

    next_bytes = (
        estimate_next_hop_size_bytes(
            n_dest
        )
    )

    free_bytes = shutil.disk_usage(
        paths[
            "root"
        ]
    ).free

    print(
        f"[NEXT-HOP] Shape: "
        f"{n_dest:,} x {n_dest:,}"
    )

    print(
        f"[NEXT-HOP] int32 disk size: "
        f"{human_bytes(next_bytes)}"
    )

    if next_bytes > free_bytes * 0.85:
        raise RuntimeError(
            "Next-hop matrix would consume more "
            "than 85% of free disk."
        )

    graph_nodes = list(
        G.nodes
    )

    graph_index = {
        node:
            i
        for i, node
        in enumerate(
            graph_nodes
        )
    }

    graph_idx_to_node = np.asarray(
        graph_nodes,
        dtype=object,
    )

    adjacency = (
        _build_min_weight_csr(
            G,
            graph_nodes,
        )
    )

    destination_graph_indices = np.asarray(
        [
            graph_index[
                node
            ]
            for node
            in destination_nodes
        ],
        dtype=np.int64,
    )

    dest_graph_to_matrix = {
        int(
            graph_idx
        ):
            int(
                matrix_idx
            )
        for matrix_idx, graph_idx
        in enumerate(
            destination_graph_indices
        )
    }

    NH = np.memmap(
        paths[
            "next_hop"
        ],
        dtype=np.int32,
        mode="w+",
        shape=(
            n_dest,
            n_dest,
        ),
    )

    NH[
        :,
        :
    ] = -1

    batch_size = max(
        1,
        int(
            cfg.routing.routing_matrix_batch_size
        ),
    )

    t0 = time.perf_counter()

    print(
        "[NEXT-HOP] Building predecessor-based next-hop matrix..."
    )

    for start in range(
        0,
        n_dest,
        batch_size,
    ):
        stop = min(
            start
            + batch_size,
            n_dest,
        )

        source_graph_indices = (
            destination_graph_indices[
                start:stop
            ]
        )

        (
            dist,
            predecessors,
        ) = scipy_dijkstra(
            adjacency,
            directed=True,
            indices=source_graph_indices,
            return_predecessors=True,
        )

        # For each source and destination graph-node target, walk predecessor
        # pointers backward only until the source is reached, then store the
        # first graph node after source. The result is mapped back to the
        # destination-node universe whenever possible.
        for local_row, source_graph_idx in enumerate(
            source_graph_indices
        ):
            row_idx = (
                start
                + local_row
            )

            NH[
                row_idx,
                row_idx
            ] = row_idx

            pred_row = predecessors[
                local_row
            ]

            for target_matrix_idx, target_graph_idx in enumerate(
                destination_graph_indices
            ):
                if (
                    target_matrix_idx
                    == row_idx
                ):
                    continue

                if not np.isfinite(
                    dist[
                        local_row,
                        target_graph_idx
                    ]
                ):
                    continue

                cur = int(
                    target_graph_idx
                )

                prev = int(
                    pred_row[
                        cur
                    ]
                )

                if prev < 0:
                    continue

                # Walk backward until predecessor is the source.
                guard = 0

                while (
                    prev
                    != int(
                        source_graph_idx
                    )
                ):
                    cur = prev

                    prev = int(
                        pred_row[
                            cur
                        ]
                    )

                    guard += 1

                    if (
                        prev < 0
                        or guard
                        > len(
                            graph_nodes
                        )
                    ):
                        cur = -1
                        break

                if cur < 0:
                    continue

                next_matrix_idx = (
                    dest_graph_to_matrix.get(
                        int(
                            cur
                        )
                    )
                )

                # The immediate physical next node may not itself be one of
                # the K POI destination nodes. In that case, leave -1 and
                # runtime will fall back to the exact route-template cache.
                if next_matrix_idx is not None:
                    NH[
                        row_idx,
                        target_matrix_idx
                    ] = int(
                        next_matrix_idx
                    )

        NH.flush()

        elapsed = (
            time.perf_counter()
            - t0
        )

        done = stop

        rate = (
            done
            / max(
                elapsed,
                1e-9,
            )
        )

        remaining = (
            n_dest
            - done
        ) / max(
            rate,
            1e-9,
        )

        print(
            f"[NEXT-HOP] "
            f"{done:,}/{n_dest:,} sources "
            f"({100.0 * done/n_dest:5.1f}%) | "
            f"elapsed={elapsed/60:.1f} min | "
            f"ETA={remaining/60:.1f} min"
        )

    total = (
        time.perf_counter()
        - t0
    )

    NH.flush()
    del NH

    metadata = json.loads(
        paths[
            "metadata"
        ].read_text(
            encoding="utf-8"
        )
    )

    metadata[
        "next_hop_available"
    ] = True

    metadata[
        "next_hop_dtype"
    ] = "int32"

    metadata[
        "next_hop_size_bytes"
    ] = int(
        next_bytes
    )

    metadata[
        "next_hop_build_time_s"
    ] = float(
        total
    )

    paths[
        "metadata"
    ].write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "[NEXT-HOP] Done."
    )
    print(
        f"[NEXT-HOP] Build time: "
        f"{total/60:.2f} min"
    )


@dataclass
class RoutingMatrixOracle:
    node_strings: np.ndarray
    poi_to_matrix_index: np.ndarray
    distance_matrix: np.memmap
    node_to_matrix_index: dict
    walking_speed_m_s: float
    metadata: dict
    next_hop_matrix: np.memmap | None

    @classmethod
    def load(
        cls,
        cfg,
    ):
        paths = _paths(
            cfg
        )

        missing = [
            str(
                p
            )
            for key, p
            in paths.items()
            if key
            in (
                "distance",
                "nodes",
                "poi_index",
                "metadata",
            )
            and not p.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Routing matrix is incomplete.\nMissing:\n  "
                + "\n  ".join(
                    missing
                )
                + "\nRun:\n"
                "    python prepare_routing_matrix.py"
            )

        metadata = json.loads(
            paths[
                "metadata"
            ].read_text(
                encoding="utf-8"
            )
        )

        nodes = np.load(
            paths[
                "nodes"
            ],
            allow_pickle=False,
        )

        poi_index = np.load(
            paths[
                "poi_index"
            ],
            allow_pickle=False,
        )

        n = int(
            metadata[
                "n_unique_destination_nodes"
            ]
        )

        D = np.memmap(
            paths[
                "distance"
            ],
            dtype=np.float32,
            mode="r",
            shape=(
                n,
                n,
            ),
        )

        mapping = {
            str(
                node
            ):
                i
            for i, node
            in enumerate(
                nodes
            )
        }

        next_hop = None

        if paths[
            "next_hop"
        ].exists():
            next_hop = np.memmap(
                paths[
                    "next_hop"
                ],
                dtype=np.int32,
                mode="r",
                shape=(
                    n,
                    n,
                ),
            )

        return cls(
            node_strings=nodes,
            poi_to_matrix_index=poi_index,
            distance_matrix=D,
            node_to_matrix_index=mapping,
            walking_speed_m_s=float(
                cfg.routing.walking_speed_m_s
            ),
            metadata=metadata,
            next_hop_matrix=next_hop,
        )

    def destination_node_strings_for_pois(
        self,
    ):
        return self.node_strings[
            self.poi_to_matrix_index
        ]

    def matrix_index_for_graph_node(
        self,
        node,
    ):
        return self.node_to_matrix_index.get(
            str(
                node
            )
        )

    def distance_by_nodes(
        self,
        origin_node,
        destination_node,
    ):
        i = (
            self.matrix_index_for_graph_node(
                origin_node
            )
        )

        j = (
            self.matrix_index_for_graph_node(
                destination_node
            )
        )

        if (
            i is None
            or j is None
        ):
            return np.nan

        return float(
            self.distance_matrix[
                i,
                j
            ]
        )

    def travel_time_by_nodes(
        self,
        origin_node,
        destination_node,
    ):
        d = self.distance_by_nodes(
            origin_node,
            destination_node,
        )

        if not np.isfinite(
            d
        ):
            return np.nan

        return (
            d
            / max(
                self.walking_speed_m_s,
                1e-9,
            )
        )

    def next_hop_indices(
        self,
        origin_node,
        destination_node,
    ):
        """
        Return the K-universe next hop if available.
        """
        if self.next_hop_matrix is None:
            return None

        i = (
            self.matrix_index_for_graph_node(
                origin_node
            )
        )

        j = (
            self.matrix_index_for_graph_node(
                destination_node
            )
        )

        if (
            i is None
            or j is None
        ):
            return None

        nxt = int(
            self.next_hop_matrix[
                i,
                j
            ]
        )

        if nxt < 0:
            return None

        return (
            i,
            j,
            nxt,
        )
