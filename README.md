# terra-incognita

MLflow training + serving side of the camera-trap failure-mode CV project. The deeper
goal is **learning ML-Engineering best practices** (MLflow tracking, model registry +
aliases, reproducible runs, a clean serving contract, local/prod parity) and making the
whole system **observable** via OpenTelemetry wide events.

> Cross-repo contracts and design live in [`.plans/`](./.plans) — a read-only,
> `git subtree`-vendored mirror of [`terra-carta`](https://github.com/kevinher7/terra-carta).
> The single source of truth for shared interfaces. Do not edit it here.

## Slice 1 — scaffold (this slice)

A thin testable vertical: a Typer CLI command emits a registry-validated `training.run`
wide event captured by an in-memory OTel exporter and asserted in a unit test, all
behind a green CI gate.

| Piece | Where |
|---|---|
| Typed config (env-driven, nothing hardcoded) | `src/terra_incognita/config.py` |
| Typed OTel wide-event helper (Pydantic-backed) | `src/terra_incognita/obs/` |
| CLI (one stub per pipeline step + `demo-event`) | `src/terra_incognita/cli.py` |
| Reproducible step wrapper | `MLproject` + `python_env.yaml` |
| Command surface | `justfile` |
| CI gate (uv sync → ruff → ty → pytest) | `.github/workflows/ci.yml` |

The wide-event field names come from `.plans/contracts/observability.attributes.yaml`;
a CI guard test fails if the typed models drift from that registry.

## Quickstart

```sh
just sync          # core + dev deps (lean; no torch). Mirrors CI.
just check         # ruff lint + format-check + ty type-check + pytest
just demo-event    # emit a training.run wide event (set OTEL_EXPORTER_OTLP_ENDPOINT to ship it)
just sync-ml       # add the heavy ML stack (torch/ultralytics/mlflow) for later slices
```

Copy [`.env.example`](./.env.example) to `.env` for local config. Torch wheels are
platform-routed (macOS → MPS-capable CPU wheels, Linux → CUDA cu124) in
`pyproject.toml` so the same project installs on a laptop and the GPU box unchanged.
