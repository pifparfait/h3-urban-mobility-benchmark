#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_FILES = [
    "README.md",
    "requirements.txt",
    "pyproject.toml",

    "src/h3uab/__init__.py",
    "src/h3uab/agent_profiles.py",
    "src/h3uab/agent_sampling.py",
    "src/h3uab/behavior.py",
    "src/h3uab/benchmarking.py",
    "src/h3uab/config.py",
    "src/h3uab/continuous_backend.py",
    "src/h3uab/continuous_indexed_backend.py",
    "src/h3uab/dynamic_state.py",
    "src/h3uab/fidelity.py",
    "src/h3uab/geometry.py",
    "src/h3uab/h3_backend.py",
    "src/h3uab/h3_utils.py",
    "src/h3uab/models.py",
    "src/h3uab/osm_data.py",
    "src/h3uab/road_network.py",
    "src/h3uab/robust_metrics.py",
    "src/h3uab/routing_matrix.py",
    "src/h3uab/semantic_layers.py",
    "src/h3uab/spatial_base.py",
    "src/h3uab/static_city.py",
    "src/h3uab/world.py",

    "experiments/01_architecture_benchmark.py",
    "experiments/02_end_to_end_benchmark.py",
    "experiments/03_urban_fidelity.py",

    "analysis/01_architecture_scalability.py",
    "analysis/02_urban_fidelity_figures.py",
    "analysis/03_spatial_heatmaps.py",
    "analysis/04_final_madrid_visual_analysis.py",
    "analysis/run_all_analysis.py",

    "scripts/link_data.py",

    "tests/test_architecture_contract.py",
]

EXPECTED_DATA = [
    "data/processed/madrid_osm_pois.csv",
    "data/processed/madrid_population_h3_r9.csv",
    "data/processed/worldpop/madrid_worldpop_2020.tif",
    "data/raw/worldpop/esp_ppp_2020_UNadj_constrained.tif",
    "data/cache/madrid_static_city_h3.pkl",
    "data/cache/madrid_walk_graph.graphml",
    "data/cache/madrid_walk_graph_h3_index.pkl",
    "data/cache/routing_matrix",
]

CRITICAL_CHECKS = [
    {
        "file": "src/h3uab/spatial_base.py",
        "name": "routing lazy import",
        "must_contain": [
            "if not cfg.routing.enabled",
            "from h3uab.road_network import",
        ],
        "must_not_match_at_top": r"^from h3uab\.road_network import",
    },
    {
        "file": "src/h3uab/static_city.py",
        "name": "legacy pickle compatibility",
        "must_contain": [
            '"static_city": "h3uab.static_city"',
            '"config": "h3uab.config"',
            "sys.modules[legacy_name] = module",
        ],
    },
    {
        "file": "src/h3uab/fidelity.py",
        "name": "urban fidelity diagnostics",
        "must_contain": [
            "class UrbanFidelity",
            "def summary(self)",
            "def export_diagnostics(self",
            "self.destination_separations_m",
            "self.visit_c",
            "self.visit_h",
        ],
    },
    {
        "file": "experiments/03_urban_fidelity.py",
        "name": "H7/H8/H9 fidelity only",
        "must_contain": [
            "UrbanFidelity(world, resolutions=(7, 8, 9))",
            "fidelity.export_diagnostics",
        ],
        "must_not_contain": [
            "resolutions=(11",
            "resolutions=(12",
        ],
    },
    {
        "file": "experiments/01_architecture_benchmark.py",
        "name": "checkpointed architecture sweep",
        "must_contain": [
            "architecture_summary_n",
            "architecture_raw_n",
            "architecture_sweep_summary.csv",
            "--skip-existing",
        ],
    },
    {
        "file": "analysis/04_final_madrid_visual_analysis.py",
        "name": "Madrid map + OD final analysis",
        "must_contain": [
            "madrid_walk_graph.graphml",
            "FIG_MAP_visit_density",
            "FIG_OD_H",
            "destination_separations.csv",
        ],
    },
]

def status(ok: bool) -> str:
    return "OK" if ok else "FAIL"

def print_section(title: str):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

def check_files():
    print_section("1. REPOSITORY STRUCTURE")
    missing = []
    for rel in EXPECTED_FILES:
        p = ROOT / rel
        ok = p.exists()
        print(f"[{status(ok):4}] {rel}")
        if not ok:
            missing.append(rel)
    return missing

def check_data():
    print_section("2. DATA / CACHE LINKS")
    missing = []
    for rel in EXPECTED_DATA:
        p = ROOT / rel
        exists = p.exists()
        link = p.is_symlink()
        target = None
        if link:
            try:
                target = os.readlink(p)
            except Exception:
                pass

        state = "OK" if exists else "FAIL"
        suffix = ""
        if link:
            suffix = f" | symlink -> {target}"
        elif exists:
            suffix = " | local file/dir"

        print(f"[{state:4}] {rel}{suffix}")

        if not exists:
            missing.append(rel)
    return missing

def check_critical_code():
    print_section("3. CRITICAL ARCHITECTURAL CONTRACTS")
    failures = []

    for check in CRITICAL_CHECKS:
        rel = check["file"]
        p = ROOT / rel
        if not p.exists():
            print(f"[FAIL] {check['name']}: missing {rel}")
            failures.append(check["name"])
            continue

        text = p.read_text(encoding="utf-8", errors="replace")
        ok = True
        notes = []

        for needle in check.get("must_contain", []):
            if needle not in text:
                ok = False
                notes.append(f"missing: {needle}")

        for needle in check.get("must_not_contain", []):
            if needle in text:
                ok = False
                notes.append(f"forbidden: {needle}")

        top_pattern = check.get("must_not_match_at_top")
        if top_pattern:
            first_40 = "\n".join(text.splitlines()[:40])
            if re.search(top_pattern, first_40, flags=re.MULTILINE):
                ok = False
                notes.append("road_network imported at module level")

        print(f"[{status(ok):4}] {check['name']} ({rel})")
        for note in notes:
            print(f"       - {note}")

        if not ok:
            failures.append(check["name"])

    return failures

def check_python_syntax():
    print_section("4. PYTHON SYNTAX")
    failures = []

    dirs = [
        ROOT / "src" / "h3uab",
        ROOT / "experiments",
        ROOT / "analysis",
        ROOT / "scripts",
        ROOT / "tests",
    ]

    files = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.py")))

    for p in files:
        try:
            compile(
                p.read_text(encoding="utf-8", errors="replace"),
                str(p),
                "exec",
            )
            print(f"[ OK ] {p.relative_to(ROOT)}")
        except Exception as exc:
            print(f"[FAIL] {p.relative_to(ROOT)} -> {exc}")
            failures.append(str(p.relative_to(ROOT)))

    return failures

def check_imports():
    print_section("5. CORE IMPORTS")
    sys.path.insert(0, str(ROOT / "src"))

    modules = [
        "h3uab.config",
        "h3uab.models",
        "h3uab.static_city",
        "h3uab.spatial_base",
        "h3uab.continuous_indexed_backend",
        "h3uab.h3_backend",
        "h3uab.fidelity",
        "h3uab.benchmarking",
    ]

    failures = []
    for module in modules:
        try:
            __import__(module)
            print(f"[ OK ] import {module}")
        except Exception as exc:
            print(f"[FAIL] import {module} -> {type(exc).__name__}: {exc}")
            failures.append(module)

    return failures

def check_static_cache():
    print_section("6. STATIC CITY CACHE")
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from h3uab.config import Config
        from h3uab.static_city import load_static_city

        cfg = Config()
        city = load_static_city(cfg.static_city_cache)

        print(f"[ OK ] loaded {cfg.static_city_cache}")
        print(f"       type: {type(city)}")
        return []
    except Exception as exc:
        print(f"[FAIL] static city cache -> {type(exc).__name__}: {exc}")
        return ["static city cache"]

def check_outputs():
    print_section("7. EXPERIMENT OUTPUTS")

    expected_arch = [
        "outputs/architecture/architecture_summary_n10000.csv",
        "outputs/architecture/architecture_summary_n100000.csv",
        "outputs/architecture/architecture_summary_n500000.csv",
        "outputs/architecture/architecture_summary_n1000000.csv",
        "outputs/architecture/architecture_sweep_summary.csv",
    ]

    expected_fidelity = [
        "outputs/urban_fidelity/urban_fidelity_n10000.csv",
        "outputs/urban_fidelity/diagnostics_n10000/destination_separations.csv",
        "outputs/urban_fidelity/diagnostics_n10000/visit_density_H7.csv",
        "outputs/urban_fidelity/diagnostics_n10000/visit_density_H8.csv",
        "outputs/urban_fidelity/diagnostics_n10000/visit_density_H9.csv",
        "outputs/urban_fidelity/diagnostics_n10000/occupancy_H7.csv",
        "outputs/urban_fidelity/diagnostics_n10000/occupancy_H8.csv",
        "outputs/urban_fidelity/diagnostics_n10000/occupancy_H9.csv",
        "outputs/urban_fidelity/diagnostics_n10000/od_flows_H7.csv",
        "outputs/urban_fidelity/diagnostics_n10000/od_flows_H8.csv",
        "outputs/urban_fidelity/diagnostics_n10000/od_flows_H9.csv",
    ]

    missing = []

    print("Architecture:")
    for rel in expected_arch:
        ok = (ROOT / rel).exists()
        print(f"[{status(ok):4}] {rel}")
        if not ok:
            missing.append(rel)

    print()
    print("Urban fidelity:")
    for rel in expected_fidelity:
        ok = (ROOT / rel).exists()
        print(f"[{status(ok):4}] {rel}")
        if not ok:
            missing.append(rel)

    return missing

def check_gitignore():
    print_section("8. GIT SAFETY")
    p = ROOT / ".gitignore"
    if not p.exists():
        print("[WARN] .gitignore is missing.")
        print("       Do NOT git add the repository before creating one.")
        return ["missing .gitignore"]

    text = p.read_text(encoding="utf-8", errors="replace")
    required = [
        ".venv/",
        "data/cache/",
        "*.pkl",
        "*.graphml",
        "*.tif",
    ]

    failures = []
    for pattern in required:
        ok = pattern in text
        print(f"[{status(ok):4}] ignore pattern: {pattern}")
        if not ok:
            failures.append(pattern)

    return failures

def main():
    print("=" * 100)
    print("MADRID H3 URBAN BENCHMARK — REPOSITORY AUDIT")
    print("=" * 100)
    print(f"Root: {ROOT}")
    print(f"Python: {sys.executable}")
    print()

    results = {
        "missing_repository_files": check_files(),
        "missing_data_assets": check_data(),
        "critical_contract_failures": check_critical_code(),
        "syntax_failures": check_python_syntax(),
        "import_failures": check_imports(),
        "cache_failures": check_static_cache(),
        "missing_outputs": check_outputs(),
        "git_safety_failures": check_gitignore(),
    }

    print_section("9. FINAL AUDIT SUMMARY")

    total_failures = sum(len(v) for v in results.values())

    for category, items in results.items():
        print(f"{category:30}: {len(items)}")
        for item in items:
            print(f"    - {item}")

    print()
    if total_failures == 0:
        print("AUDIT STATUS: PASS")
        print("Repository is structurally ready for final runs and GitHub.")
    else:
        print("AUDIT STATUS: ACTION REQUIRED")
        print(f"Total findings: {total_failures}")
        print(
            "Fix structure/core-code issues before running expensive experiments. "
            "Missing experiment outputs are expected if they have not yet been regenerated."
        )

if __name__ == "__main__":
    main()
