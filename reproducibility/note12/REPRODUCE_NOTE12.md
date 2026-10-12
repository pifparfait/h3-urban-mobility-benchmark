# NOTE12 — Reproduction Guide

## Certified scientific version

Git commit:

`a63ee7f766e1b7dc1e71c8fb71024c6d62161df7`

Repository:

https://github.com/pifparfait/h3-urban-mobility-benchmark

## Scope

The R12-C14 audit compares 50,000 paired destination decisions
per city across Madrid, Cambridge (MA), and New York City.

The historical audit and its fresh-Python-environment repetition
passed on the same machine.

Independent cross-machine reproduction has not yet been established.

## Required artifacts

1. Scientific repository at the certified commit.
2. Nine certified input files, preserving their relative paths.
3. Six historical comparison CSV files, preserving their paths.
4. `r12_c14_replay_audit_v2.py`.
5. A Python environment with the recorded dependencies.

The input and historical archives include SHA-256 manifests.
Verify these manifests before using the artifacts.

## 1. Prepare a clean scientific checkout

```bash
git clone https://github.com/pifparfait/h3-urban-mobility-benchmark.git note12_science
cd note12_science
git checkout --detach a63ee7f766e1b7dc1e71c8fb71024c6d62161df7
```

The scientific checkout must pass `git status --porcelain` with empty output. Certified inputs excluded by `.gitignore` may be restored without making the checkout dirty. Do not modify tracked scientific files.

## 2. Restore certified inputs

Copy the nine certified files into their original relative paths
under the scientific checkout.

These paths include the Madrid, Cambridge, and New York City
processed POI and population CSV files and static-city pickle caches.

Verify the nine SHA-256 values against the data archive manifest. If an input is already tracked, verify its identity rather than silently overwriting it. After restoring the inputs, confirm that `git status --porcelain` remains empty.

Do not rebuild these inputs if the objective is exact R12-C14 replay.

## 3. Prepare historical comparisons

Create a separate directory for the six historical CSV files,
preserving the `outputs/...` directory hierarchy.

This directory will be passed through `--historical-root`.

Verify its `SHA256SUMS` manifest.

## 4. Install Python dependencies

From the scientific checkout, create an isolated Python environment and install the recorded dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r /path/to/note12/fresh_env_requirements.txt
python -m pip install --no-deps -e .
```

The exact historical dependency snapshot is also supplied. Dependency installation must be checked against the target Python version and operating system.

The historical and rebuilt environments were certified on
the same machine. Package installation and portability on other
platforms are not guaranteed.

## 5. Execute the auditor

The auditor accepts:

- `--worktree`: clean scientific checkout at the certified commit
- `--historical-root`: historical comparison directory
- `--result-root`: writable output directory for the new audit
- `--city`: target city

Example for Madrid:

```bash
python /path/to/note12/r12_c14_replay_audit_v2.py \
  --worktree /path/to/note12_science \
  --historical-root /path/to/historical_archive \
  --result-root /path/to/audit_results_madrid \
  --city madrid
```

Run the same procedure for `cambridge_ma` and `new_york_ny`,
using separate result directories.

**Warning:** This step executes the urban-fidelity experiment.
It is not a read-only verification command.

## 6. Examine the certification

Inspect each generated `status.json` for the audit result,
comparison checks, metric values, input identities and
destination-sequence checks.

Compare these results with the sanitized historical and
fresh-environment certificates in the evidence package.

Do not assume a PASS merely because the command completes.

## Limitations

- Exact input identity is established by SHA-256.
- The existing certification covers the original and rebuilt
  Python environments on the same machine.
- Cross-machine reproduction remains unverified.
- Python pickle files must only be loaded from trusted sources.
- Third-party data redistribution is subject to upstream terms.
- Historical and fresh-environment evidence is supplied
  separately from the nine scientific inputs.
