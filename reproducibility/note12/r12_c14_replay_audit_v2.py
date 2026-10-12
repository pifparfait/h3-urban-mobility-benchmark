#!/usr/bin/env python3
"""NOTE12 R12-C14 v2. Review-only deliverable; DO NOT run before approval.

Runs the committed NOTE05 experiment in a clean worktree, captures paired
accessibility losslessly, and checks all pre-registered gates. No scientific
source files are edited. Inputs must already be staged in the clean worktree; never chmod inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import time
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import traceback
import uuid

import numpy as np

CITIES = {
    "madrid": ("urban_fidelity", "madrid_traced"),
    "cambridge_ma": ("cambridge_ma/urban_fidelity", "cambridge_traced"),
    "new_york_ny": ("new_york_ny/urban_fidelity", "nyc_traced"),
}
EXPECTED_COMMIT = "a63ee7f766e1b7dc1e71c8fb71024c6d62161df7"
TOL = 1e-12
N = 50000
EXPECTED_INVALID_DESTINATIONS = {"madrid": 24, "cambridge_ma": 8, "new_york_ny": 363}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def destination_digest(rows):
    """SHA-256 over canonical UTF-8 agent,step,continuous_id,h3_id rows."""
    h = hashlib.sha256()
    for row in rows:
        canonical = ",".join(str(int(row[k])) for k in (
            "agent_id", "step", "continuous_destination_id", "h3_destination_id"
        )) + "\n"
        h.update(canonical.encode("ascii"))
    return h.hexdigest()


def load_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def metric(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or len(a) == 0:
        raise AssertionError("Empty or unpaired metric vectors")
    finite = np.isfinite(a) & np.isfinite(b)
    # Historical accessibility code appends every decision, including nonfinite;
    # it does not filter. Distance filters only paired finite values.
    return {
        "pearson": float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0 else float("nan"),
        "nmae": float(np.mean(np.abs(b - a)) / max(np.mean(np.abs(a)), 1e-12)),
        "n": len(a), "finite_pairs": int(finite.sum()),
    }


def check_equal(actual, expected, label):
    if not (math.isfinite(actual) and math.isfinite(expected)
            and abs(actual - expected) <= TOL):
        raise AssertionError(f"{label}: actual={actual!r} historical={expected!r}")


def atomic_json(path, value):
    """Durably replace status, never expose a partially written JSON file."""
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as f:
            json.dump(value, f, indent=2, allow_nan=False, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def input_snapshot(paths):
    """Hash resolved targets, not merely the symlink directory entries."""
    result = {}
    for name, path in paths.items():
        resolved = Path(os.path.realpath(path))
        result[name] = {
            "declared_path": str(path.absolute()),
            "realpath": str(resolved),
            "is_symlink": path.is_symlink(),
            "sha256": sha256(resolved),
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worktree", type=Path, required=True)
    p.add_argument("--historical-root", type=Path, required=True)
    p.add_argument("--result-root", type=Path, required=True)
    p.add_argument("--city", choices=list(CITIES), required=True)
    args = p.parse_args()

    # The audit status lives outside the experiment output. Create FAIL first,
    # before checking commit, inputs, dependencies, or importing scientific code.
    outroot = args.result_root.absolute()
    audit_dir = outroot / f"r12_c14_{args.city}"
    status_path = audit_dir / "status.json"
    original_cwd = Path.cwd()
    original_argv = sys.argv[:]
    original_path = sys.path[:]
    original_dont_write_bytecode = sys.dont_write_bytecode
    original_update = None
    fidelity_cls = None
    inputs = None
    before = None
    root = None
    state = {
        "status": "FAIL", "reason": "incomplete", "city": args.city,
        "commit_expected": EXPECTED_COMMIT, "started_unix": time.time(),
        "worktree_requested": str(args.worktree),
        "historical_root_requested": str(args.historical_root),
    }
    # Do not overwrite an earlier audit (PASS or FAIL).
    # If mkdir itself is impossible, no program can persist a status there.
    # Reject unsafe output placement before creating any directory.
    resolved_out = outroot.resolve()
    resolved_work = args.worktree.resolve()
    resolved_history = args.historical_root.resolve()
    if (outroot.is_symlink() or resolved_out == resolved_work or
            resolved_work in resolved_out.parents or resolved_out in resolved_work.parents or
            resolved_out == resolved_history or resolved_history in resolved_out.parents or
            resolved_out in resolved_history.parents):
        raise SystemExit("Unsafe audit result directory; no files created")
    outroot.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(exist_ok=False)
    atomic_json(status_path, state)
    try:
        root = args.worktree.resolve(strict=True)
        history = args.historical_root.resolve(strict=True)
        actual_outroot = outroot.resolve(strict=True)
        if (root == history or root == actual_outroot or
                root in actual_outroot.parents or actual_outroot in root.parents or
                history == actual_outroot or history in actual_outroot.parents or
                actual_outroot in history.parents):
            raise AssertionError("Unsafe overlap among worktree, history and result root")
        if git(root, "rev-parse", "HEAD") != EXPECTED_COMMIT:
            raise AssertionError("Wrong worktree commit")
        initial_status = git(root, "status", "--porcelain")
        state["worktree_status_before"] = initial_status
        if initial_status:
            raise AssertionError("Worktree not clean at start")
        os.chdir(root)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root / "src"))
        from h3uab.city_config import config_for_city
        from h3uab.fidelity import UrbanFidelity
        fidelity_cls = UrbanFidelity
        cfg = config_for_city(args.city, n_agents=10000)
        if cfg.seed != 42 or cfg.rebuild_static_city is not False or cfg.routing.enabled is not False:
            raise AssertionError("Unexpected seed, static rebuild or routing")
        inputs = {
            "poi": Path(cfg.osm_cache_csv),
            "population": Path(cfg.population_h3_csv),
            "static_cache": Path(cfg.static_city_cache),
        }
        for name, path in inputs.items():
            if not path.is_file():
                raise AssertionError(f"Missing input {name}: {path}")
        before = input_snapshot(inputs)
        state["inputs_before"] = before
        atomic_json(status_path, state)

        original_update = UrbanFidelity.update
        signature = inspect.signature(original_update)
        captured = []

        def capture(self, *a, **kw):
            bound = signature.bind(self, *a, **kw)
            values = bound.arguments
            captured.append({
                "agent_id": str(int(values["agent_id"])),
                "step": str(int(values["step"])),
                "purpose_id": "" if values.get("purpose_id") is None else str(int(values["purpose_id"])),
                "continuous_destination_id": str(int(values["dc"])),
                "h3_destination_id": str(int(values["dh"])),
                "ac_hex": float(values["ac"]).hex(),
                "ah_hex": float(values["ah"]).hex(),
                "tc_hex": float(values["tc"]).hex(),
                "th_hex": float(values["th"]).hex(),
            })
            return original_update(self, *a, **kw)

        UrbanFidelity.update = capture
        experiment = root / "experiments/03_urban_fidelity.py"
        experiment_output = audit_dir / "experiment"  # must not exist for --note05-output
        if experiment_output.exists() or experiment_output.is_symlink():
            raise AssertionError("Experiment output path already exists")
        sys.argv = [str(experiment), "--city", args.city, "--agents", "10000",
                    "--note05-output", str(experiment_output)]
        runpy.run_path(str(experiment), run_name="__main__")
        if len(captured) != N:
            raise AssertionError(f"Decision count {len(captured)} != {N}")
        trace_path = audit_dir / "accessibility_pairs.csv"
        with trace_path.open("x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(captured[0]))
            w.writeheader()
            w.writerows(captured)

        historical_dir = history / "outputs" / CITIES[args.city][0]
        historical_endpoints = historical_dir / "diagnostics_n10000/paired_trip_endpoints.csv"
        historical_summary = historical_dir / "urban_fidelity_n10000.csv"
        old = load_csv(historical_endpoints)
        if len(old) != N:
            raise AssertionError("Historical endpoint count differs from 50000")
        old_hash = destination_digest(old)
        new_hash = destination_digest(captured)
        state["decisions"] = len(captured)
        state["historical_destination_sequence_sha256"] = old_hash
        state["destination_sequence_sha256"] = new_hash
        if new_hash != old_hash:
            raise AssertionError(f"Destination sequence mismatch {new_hash} != {old_hash}")

        invalid = sum(int(int(r["continuous_destination_id"]) < 0 or
                          int(r["h3_destination_id"]) < 0) for r in captured)
        state["invalid_destinations_either_backend"] = invalid
        if invalid != EXPECTED_INVALID_DESTINATIONS[args.city]:
            raise AssertionError(f"Invalid-destination count {invalid} differs from expected "
                                 f"{EXPECTED_INVALID_DESTINATIONS[args.city]}")

        historical_metrics = {r["metric"]: r for r in load_csv(historical_summary)}
        ac = [float.fromhex(r["ac_hex"]) for r in captured]
        ah = [float.fromhex(r["ah_hex"]) for r in captured]
        accessibility = metric(ac, ah)
        tc = [float.fromhex(r["tc_hex"]) for r in captured]
        th = [float.fromhex(r["th_hex"]) for r in captured]
        valid = [math.isfinite(x) and math.isfinite(y) for x, y in zip(tc, th)]
        trip = metric([x for x, v in zip(tc, valid) if v],
                      [y for y, v in zip(th, valid) if v])
        # EXACT HISTORICAL INCLUSION RULE (committed src/h3uab/fidelity.py,
        # UrbanFidelity.update): ac/ah are appended for EVERY paired decision,
        # including dc<0 or dh<0. tc/th are appended ONLY if both finite.
        # No destination-validity filter is applied to either metric. Summary
        # Pearson/NMAE formulas are reproduced by metric() above.
        state["metric_inclusion_rule"] = (
            "accessibility: all decisions including invalid destinations; "
            "trip distance: finite tc/th pairs only, regardless of destination IDs"
        )
        state["accessibility"] = accessibility
        state["trip_distance"] = trip
        state["nonfinite_trip_pairs_excluded"] = N - sum(valid)
        state["historical_endpoints_sha256"] = sha256(historical_endpoints)
        state["historical_summary_sha256"] = sha256(historical_summary)
        state["capture_sha256"] = sha256(trace_path)
        atomic_json(status_path, state)  # crash still leaves FAIL

        for key, result in (("accessibility", accessibility), ("trip_distance", trip)):
            for field in ("pearson", "nmae"):
                check_equal(result[field], float(historical_metrics[key][field]),
                            f"{args.city}/{key}/{field}")
        state["python"] = platform.python_version()
        state["packages"] = {k: importlib.metadata.version(k) for k in
                             ("numpy", "scipy", "scikit-learn", "h3", "pandas")}
        state["scientific_checks_passed"] = True
    except BaseException as exc:
        state["reason"] = f"{type(exc).__name__}: {exc}"
        state["traceback"] = traceback.format_exc()
    finally:
        # This block runs on exceptions and Ctrl-C. SIGKILL/power loss retains
        # the initial FAIL; no Python cleanup is guaranteed in those cases.
        try:
            if fidelity_cls is not None and original_update is not None:
                fidelity_cls.update = original_update
            sys.argv = original_argv
            sys.path[:] = original_path
            sys.dont_write_bytecode = original_dont_write_bytecode
            os.chdir(original_cwd)
        except BaseException as exc:
            state["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        try:
            if root is not None:
                status_after = git(root, "status", "--porcelain")
                state["worktree_status_after"] = status_after
                state["worktree_clean_after"] = status_after == ""
            else:
                state["worktree_clean_after"] = None
        except BaseException as exc:
            state["worktree_status_error"] = f"{type(exc).__name__}: {exc}"
        try:
            if inputs is not None and before is not None:
                after = input_snapshot({k: root / v for k, v in inputs.items()})
                state["inputs_after"] = after
                state["inputs_unchanged"] = all(
                    before[k]["realpath"] == after[k]["realpath"] and
                    before[k]["sha256"] == after[k]["sha256"]
                    for k in before
                )
            else:
                state["inputs_unchanged"] = None
        except BaseException as exc:
            state["inputs_after_error"] = f"{type(exc).__name__}: {exc}"
        success = (state.get("scientific_checks_passed") is True and
                   state.get("worktree_clean_after") is True and
                   state.get("inputs_unchanged") is True and
                   "cleanup_error" not in state)
        if success:
            state["status"] = "PASS"
            state["reason"] = "all pre-registered checks passed"
        else:
            state["status"] = "FAIL"
            if state.get("reason") == "incomplete":
                state["reason"] = "post-run integrity or cleanliness check failed"
        state["finished_unix"] = time.time()
        atomic_json(status_path, state)

    print(json.dumps({"city": args.city, "status": state["status"],
                      "audit_directory": str(audit_dir), "status_file": str(status_path)}, indent=2))
    if state["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
