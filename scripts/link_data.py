#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]

MAPPING = {
    "data/madrid_osm_pois.csv": "data/processed/madrid_osm_pois.csv",
    "data/madrid_population_h3_r9.csv": "data/processed/madrid_population_h3_r9.csv",
    "data/madrid_static_city_h3.pkl": "data/cache/madrid_static_city_h3.pkl",
    "data/madrid_walk_graph.graphml": "data/cache/madrid_walk_graph.graphml",
    "data/madrid_walk_graph_h3_index.pkl": "data/cache/madrid_walk_graph_h3_index.pkl",
    "data/routing_matrix": "data/cache/routing_matrix",
    "data/worldpop/madrid_worldpop_2020.tif":
        "data/processed/worldpop/madrid_worldpop_2020.tif",
    "data/worldpop/esp_ppp_2020_UNadj_constrained.tif":
        "data/raw/worldpop/esp_ppp_2020_UNadj_constrained.tif",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old_repo", type=Path)
    ap.add_argument("--copy", action="store_true",
                    help="Copy instead of symlink (uses much more disk).")
    args = ap.parse_args()
    old = args.old_repo.expanduser().resolve()

    for src_rel, dst_rel in MAPPING.items():
        src = old / src_rel
        dst = ROOT / dst_rel
        if not src.exists():
            print(f"[missing, skipped] {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if args.copy:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            print(f"[copied] {dst_rel}")
        else:
            os.symlink(src, dst, target_is_directory=src.is_dir())
            print(f"[linked] {dst_rel} -> {src}")

if __name__ == "__main__":
    main()
