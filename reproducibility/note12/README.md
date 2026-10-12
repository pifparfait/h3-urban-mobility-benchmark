# NOTE12 — Reproducibility evidence

Certified scientific commit:
a63ee7f766e1b7dc1e71c8fb71024c6d62161df7

Repository:
https://github.com/pifparfait/h3-urban-mobility-benchmark

## Scope

Three cities: Madrid, Cambridge (MA), and New York (NY).

Each round contains 50,000 paired decisions per city.
Two rounds were completed on the same machine:
the historical environment and a rebuilt Python environment.

The comparison passed 59/59 cross-environment checks.

## Included files

- r12_c14_replay_audit_v2.py: audit implementation.
- certifications/: sanitized certification records.
- reproduction_summary.json: cross-environment metrics.
- historical_pip_freeze.txt: historical environment snapshot.
- fresh_env_requirements.txt: reconstructed environment dependencies.
- SHA256SUMS: hashes of the public evidence files.

## Important limitations

The certification records are evidence of completed local runs.
They do not themselves include the complete experimental inputs
or constitute an independent cross-machine reproduction.

The auditor requires the repository, historical comparison
artifacts, and the relevant city datasets. Public reproduction
instructions and data provenance must be completed before release.

Original unmodified certification records are retained outside
this public package.
